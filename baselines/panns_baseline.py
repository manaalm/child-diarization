"""
PANNS AudioSet Feature Baseline (Tier 3) for child vocalization detection.

Uses frozen CNN14 (PANNS, AudioSet-pretrained) to extract 2048-dim clip embeddings,
then trains a logistic regression head on the seen-child train split.

AudioSet classes include: "Speech", "Child speech, kid speaking", "Baby cry,
infant cry", "Babbling" — directly relevant to the task. PANNS embeddings may
carry complementary signal to speech-pretrained encoders (Whisper, WavLM).

Pipeline:
  1. Extract 2048-dim CNN14 embedding per clip (full clip, no windowing)
  2. Train LR head on seen-child train split (labels 0/1)
  3. Score val/test clips with LR probability output
  4. Tune threshold on val; report test metrics

Usage:
    python baselines/panns_baseline.py --split val
    python baselines/panns_baseline.py --split test

    # Cross-child split (no training — uses seen-child LR head):
    python baselines/panns_baseline.py --split val \\
        --splits-dir baselines/splits \\
        --output-dir baselines/panns_baseline_runs/cnn14_cross_child \\
        --lr-weights baselines/panns_baseline_runs/cnn14/lr_weights.npz

    # Dry run:
    python baselines/panns_baseline.py --split val --max-clips 5 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

SPLITS_DIR_SEEN = _REPO / "whisper-modeling/seen_child_splits"
SPLITS_DIR_CROSS = _REPO / "baselines/splits"
RESULTS_BASE = _REPO / "baselines/panns_baseline_runs"
SR = 32000  # CNN14 expects 32kHz
EMB_DIM = 2048


def load_audio_32k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0).numpy()


def extract_embedding(audio: np.ndarray, model) -> np.ndarray:
    """Extract 2048-dim CNN14 embedding for a single clip."""
    t = torch.from_numpy(audio[np.newaxis, :]).float()
    with torch.no_grad():
        output = model(t)
    # panns_inference CNN14 returns dict with 'embedding' (2048-dim) and 'clipwise_output'
    emb = output["embedding"].squeeze(0).cpu().numpy()
    return emb


def extract_embeddings_split(
    df: pd.DataFrame,
    model,
    max_clips: Optional[int] = None,
    dry_run: bool = False,
) -> np.ndarray:
    N = min(len(df), max_clips) if max_clips else len(df)
    embs = np.zeros((N, EMB_DIM), dtype=np.float32)
    for i, row in enumerate(df.itertuples()):
        if i >= N:
            break
        if dry_run:
            embs[i] = np.random.randn(EMB_DIM).astype(np.float32)
        else:
            try:
                audio = load_audio_32k(row.audio_path)
                embs[i] = extract_embedding(audio, model)
            except Exception as e:
                print(f"  [{i+1}/{N}] ERROR {Path(row.audio_path).name}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}] embeddings extracted")
    return embs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--splits-dir", default=None,
                   help="Path to splits dir (default: seen-child splits)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--lr-weights", default=None,
                   help="Path to .npz LR weights for cross-child eval (skip training)")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cpu",
                   help="Device for CNN14 (cpu works fine for 2183 clips)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS_DIR_SEEN
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_BASE / "cnn14"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split={args.split}  device={args.device}")
    print(f"Splits: {splits_dir}")
    print(f"Output: {out_dir}")

    # Load PANNS CNN14 model
    if not args.dry_run:
        try:
            from panns_inference import AudioTagging
        except ImportError:
            print("ERROR: panns_inference not installed. Run: pip install panns_inference",
                  file=sys.stderr)
            sys.exit(1)
        print("Loading PANNS CNN14 (downloads ~120MB on first run)...")
        at = AudioTagging(checkpoint_path=None, device=args.device)
        panns_model = at.model
        panns_model.eval()
    else:
        panns_model = None

    # Load LR weights (for cross-child mode) or train from seen-child train split
    if args.lr_weights:
        d = np.load(args.lr_weights)
        lr_w = d["weights"]   # (EMB_DIM,)
        lr_b = d["bias"]      # scalar
        print(f"Loaded LR weights from {args.lr_weights}")
    else:
        # Train LR on seen-child train split
        train_df = pd.read_csv(SPLITS_DIR_SEEN / "train.csv")
        if "audio_exists" in train_df.columns:
            train_df = train_df[train_df["audio_exists"].astype(bool)]
        print(f"Training clips: {len(train_df)}")
        print("Extracting train embeddings...")
        train_embs = extract_embeddings_split(train_df, panns_model, dry_run=args.dry_run)
        train_labels = train_df["label"].values[:len(train_embs)]

        print("Training logistic regression head...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        train_embs_scaled = scaler.fit_transform(train_embs)

        lr = LogisticRegression(max_iter=1000, random_state=args.seed, C=1.0)
        lr.fit(train_embs_scaled, train_labels)
        lr_w = lr.coef_[0]
        lr_b = lr.intercept_[0]
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_

        # Save weights
        np.savez(str(out_dir / "lr_weights.npz"),
                 weights=lr_w, bias=np.array([lr_b]),
                 scaler_mean=scaler_mean, scaler_scale=scaler_scale)
        print(f"LR weights saved to {out_dir}/lr_weights.npz")

        # Save scaler for scoring
        def score_embs(embs):
            embs_scaled = (embs - scaler_mean) / scaler_scale
            return sigmoid(embs_scaled @ lr_w + lr_b)

    if args.lr_weights:
        # If LR loaded from file, check for scaler
        scaler_mean = d.get("scaler_mean", None)
        scaler_scale = d.get("scaler_scale", None)
        if scaler_mean is not None:
            def score_embs(embs):
                embs_scaled = (embs - scaler_mean) / scaler_scale
                return sigmoid(embs_scaled @ lr_w + lr_b)
        else:
            def score_embs(embs):
                return sigmoid(embs @ lr_w + float(lr_b))

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
        print(f"Extracting test embeddings ({len(meta)} clips)...")
        test_embs = extract_embeddings_split(meta, panns_model, args.max_clips, args.dry_run)
        meta_slice = meta.iloc[:len(test_embs)]
        probs = score_embs(test_embs)
        labels = meta_slice["label"].values

        preds = meta_slice[["audio_path", "label"]].copy()
        preds["prob"] = probs
        metrics = compute_metrics(labels, probs, threshold)
        metrics.update({"threshold": threshold, "model": "cnn14_panns", "n": len(preds)})

        save_json(metrics, str(out_dir / "test_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "test_predictions.csv"))

        if "timepoint_norm" in meta.columns:
            rows = []
            p_with_tp = preds.copy()
            tp_col = meta.set_index("audio_path")["timepoint_norm"]
            p_with_tp["timepoint_norm"] = p_with_tp["audio_path"].map(tp_col)
            for tp, grp in p_with_tp.groupby("timepoint_norm"):
                m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
                rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
            save_csv(pd.DataFrame(rows), str(out_dir / "test_metrics_by_timepoint.csv"))

    else:
        meta = pd.read_csv(splits_dir / "val.csv")
        if "audio_exists" in meta.columns:
            meta = meta[meta["audio_exists"].astype(bool)]
        print(f"Extracting val embeddings ({len(meta)} clips)...")
        val_embs = extract_embeddings_split(meta, panns_model, args.max_clips, args.dry_run)
        meta_slice = meta.iloc[:len(val_embs)]
        probs = score_embs(val_embs)
        labels = meta_slice["label"].values

        threshold = tune_threshold(labels, probs)
        metrics = compute_metrics(labels, probs, threshold)
        metrics.update({"threshold": threshold, "model": "cnn14_panns", "n": len(meta_slice)})

        preds = meta_slice[["audio_path", "label"]].copy()
        preds["prob"] = probs
        save_json(metrics, str(out_dir / "val_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "val_predictions.csv"))

        if "timepoint_norm" in meta.columns:
            rows = []
            p_with_tp = preds.copy()
            tp_col = meta.set_index("audio_path")["timepoint_norm"]
            p_with_tp["timepoint_norm"] = p_with_tp["audio_path"].map(tp_col)
            for tp, grp in p_with_tp.groupby("timepoint_norm"):
                m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
                rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
            save_csv(pd.DataFrame(rows), str(out_dir / "val_metrics_by_timepoint.csv"))

    save_json(
        {"model": "cnn14_panns", "split": args.split, "splits_dir": str(splits_dir),
         "emb_dim": EMB_DIM, "seed": args.seed},
        str(out_dir / "config.json"),
    )

    print(f"\n{args.split.capitalize()} metrics (threshold={threshold:.4f}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
