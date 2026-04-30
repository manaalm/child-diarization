"""
CLAP Zero-Shot Baseline (Tier 3) for child vocalization detection.

Uses laion/clap-htsat-fused (contrastive audio-text model) to score each clip
against positive and negative text prompts. No training required.

Score = sigmoid(cos_sim(audio, mean_positive_text) - cos_sim(audio, mean_negative_text))

This complements the Qwen2-Audio zero-shot baseline (autoregressive) with a
contrastive architecture family. If AUROC ≈ Qwen2's 0.725, the zero-shot audio
discrimination is architecture-agnostic.

Usage:
    python baselines/clap_baseline.py --split val
    python baselines/clap_baseline.py --split test

    # Cross-child split:
    python baselines/clap_baseline.py --split val \\
        --splits-dir baselines/splits \\
        --output-dir baselines/clap_baseline_runs/clap_htsat_fused_cross_child

    # Dry run:
    python baselines/clap_baseline.py --split val --max-clips 5 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

MODEL_ID = "laion/clap-htsat-fused"
SPLITS_DIR_SEEN = _REPO / "whisper-modeling/seen_child_splits"
SPLITS_DIR_CROSS = _REPO / "baselines/splits"
RESULTS_BASE = _REPO / "baselines/clap_baseline_runs"
SR = 48000  # CLAP expects 48kHz

POSITIVE_PROMPTS = [
    "a young child vocalizing",
    "a baby or toddler making sounds",
    "child speech and vocalizations",
]
NEGATIVE_PROMPTS = [
    "an adult speaking",
    "adult voice talking",
    "silence and background noise",
]


def load_audio_48k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0).numpy()


def build_text_embeddings(processor, model, prompts, device):
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embs = model.get_text_features(**inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    return text_embs.mean(0)  # (D,) mean of prompt embeddings, normalized below


def score_clip(
    audio_array: np.ndarray,
    processor,
    model,
    pos_text_emb: torch.Tensor,
    neg_text_emb: torch.Tensor,
    device: str,
) -> float:
    inputs = processor(
        audios=audio_array.astype(np.float32),
        sampling_rate=SR,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_emb = model.get_audio_features(**inputs)
    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    audio_emb = audio_emb.squeeze()

    pos_score = float(torch.dot(audio_emb, pos_text_emb).item())
    neg_score = float(torch.dot(audio_emb, neg_text_emb).item())
    raw = pos_score - neg_score
    return float(torch.sigmoid(torch.tensor(raw)).item())


def run_split(
    df: pd.DataFrame,
    processor,
    model,
    pos_text_emb: torch.Tensor,
    neg_text_emb: torch.Tensor,
    device: str,
    max_clips: int = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    records = []
    total = min(len(df), max_clips) if max_clips else len(df)
    for i, row in enumerate(df.itertuples()):
        if max_clips and i >= max_clips:
            break
        label = int(row.label)
        if dry_run:
            prob = 0.5
        else:
            try:
                audio = load_audio_48k(row.audio_path)
                prob = score_clip(audio, processor, model, pos_text_emb, neg_text_emb, device)
            except Exception as e:
                print(f"  [{i+1}/{total}] ERROR {Path(row.audio_path).name}: {e}")
                prob = 0.5
        records.append({"audio_path": row.audio_path, "label": label, "prob": prob})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]")
    return pd.DataFrame(records)


def add_timepoint_metrics(preds: pd.DataFrame, meta: pd.DataFrame,
                          threshold: float, out_path: Path) -> None:
    if "timepoint_norm" not in meta.columns:
        return
    tp_col = meta.set_index("audio_path")["timepoint_norm"]
    p = preds.copy()
    p["timepoint_norm"] = p["audio_path"].map(tp_col)
    rows = []
    for tp, grp in p.groupby("timepoint_norm"):
        m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
        rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
    save_csv(pd.DataFrame(rows), str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS_DIR_SEEN
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_BASE / "clap_htsat_fused"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model={MODEL_ID}  split={args.split}  device={args.device}")
    print(f"Splits: {splits_dir}")
    print(f"Output: {out_dir}")

    if not args.dry_run:
        from transformers import ClapModel, ClapProcessor
        print("Loading CLAP model...")
        processor = ClapProcessor.from_pretrained(MODEL_ID)
        model = ClapModel.from_pretrained(MODEL_ID).to(args.device)
        model.eval()

        print("Computing text embeddings...")
        pos_emb = build_text_embeddings(processor, model, POSITIVE_PROMPTS, args.device)
        neg_emb = build_text_embeddings(processor, model, NEGATIVE_PROMPTS, args.device)
        # Normalize after averaging
        pos_emb = pos_emb / pos_emb.norm()
        neg_emb = neg_emb / neg_emb.norm()
        print(f"Positive prompts: {POSITIVE_PROMPTS}")
        print(f"Negative prompts: {NEGATIVE_PROMPTS}")
    else:
        processor = model = pos_emb = neg_emb = None

    if args.split == "test":
        val_path = out_dir / "val_metrics_tuned.json"
        if not val_path.exists():
            print(f"ERROR: {val_path} not found. Run --split val first.", file=sys.stderr)
            sys.exit(2)
        with open(val_path) as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold={threshold:.4f} from val")

        meta = pd.read_csv(splits_dir / "test.csv")
        if "audio_exists" in meta.columns:
            meta = meta[meta["audio_exists"].astype(bool)]
        print(f"Test clips: {len(meta)}")

        preds = run_split(meta, processor, model, pos_emb, neg_emb, args.device,
                          args.max_clips, args.dry_run)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "model": MODEL_ID, "n": len(preds)})

        save_json(metrics, str(out_dir / "test_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "test_predictions.csv"))
        add_timepoint_metrics(preds, meta, threshold, out_dir / "test_metrics_by_timepoint.csv")

    else:
        meta = pd.read_csv(splits_dir / "val.csv")
        if "audio_exists" in meta.columns:
            meta = meta[meta["audio_exists"].astype(bool)]
        print(f"Val clips: {len(meta)}")

        preds = run_split(meta, processor, model, pos_emb, neg_emb, args.device,
                          args.max_clips, args.dry_run)
        threshold = tune_threshold(preds["label"].values, preds["prob"].values)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "model": MODEL_ID, "n": len(preds)})

        save_json(metrics, str(out_dir / "val_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "val_predictions.csv"))
        add_timepoint_metrics(preds, meta, threshold, out_dir / "val_metrics_by_timepoint.csv")

    save_json(
        {"model": MODEL_ID, "split": args.split, "splits_dir": str(splits_dir),
         "positive_prompts": POSITIVE_PROMPTS, "negative_prompts": NEGATIVE_PROMPTS,
         "seed": args.seed},
        str(out_dir / "config.json"),
    )

    print(f"\n{args.split.capitalize()} metrics (threshold={threshold:.4f}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
