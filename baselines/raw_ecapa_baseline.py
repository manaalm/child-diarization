"""
Raw-Clip ECAPA Baseline (Tier 2) — diarizer-free test-side ablation.

Builds per-child ECAPA prototypes using BabAR KCHI segments from POSITIVE training
clips (identical to BabAR enrollment). Scores each TEST clip by sliding 1.5s windows
over the ENTIRE raw clip (no diarizer on test side) and computing cosine similarity
to the child prototype.

This ablates the diarizer contribution at TEST TIME:
  - If AUROC ≈ BabAR (0.820): the diarizer frontend adds little value at test time.
  - If AUROC is much lower (e.g., 0.65): KCHI selection at test time is critical.

The prototype remains pure-child (from KCHI segments), so the enrollment signal is
not contaminated. Only the test-side scoring is raw.

Aggregation modes:
  mean  — duration-weighted mean cosine similarity across all windows
  max   — maximum cosine similarity across all windows
  top3  — mean cosine similarity of the top-3 windows by similarity

Window params: 1.5s, 50% overlap (matches BabAR enrollment segment length).
Seen-child split only (requires per-child prototypes from BabAR KCHI training data).

Usage:
    python baselines/raw_ecapa_baseline.py --mode mean --split val
    python baselines/raw_ecapa_baseline.py --mode mean --split test
    python baselines/raw_ecapa_baseline.py --mode max --split val
    python baselines/raw_ecapa_baseline.py --mode max --split test
    python baselines/raw_ecapa_baseline.py --mode top3 --split val
    python baselines/raw_ecapa_baseline.py --mode top3 --split test

    # Dry run (5 clips):
    python baselines/raw_ecapa_baseline.py --mode mean --split val --max-clips 5 --dry-run
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

SPLITS_DIR = _REPO / "whisper-modeling/seen_child_splits"
RESULTS_BASE = _REPO / "baselines/raw_ecapa_baseline_runs"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
BABAR_RTTM_DIR = _REPO / "babar/babar_output/rttm"
SR = 16000

# Window parameters matching BabAR enrollment segment length
WIN_SEC = 1.5
HOP_SEC = 0.75  # 50% overlap


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def load_audio_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0)  # (T,)


def parse_kchi_segments(audio_path: str) -> List[Dict]:
    """Return KCHI segments from BabAR RTTM for this audio file."""
    stem = Path(audio_path).stem
    rttm_path = BABAR_RTTM_DIR / f"{stem}.rttm"
    if not rttm_path.exists():
        return []
    segs = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            if label == "KCHI" and dur >= 0.5:
                segs.append({"start": start, "end": start + dur, "dur": dur})
    return segs


def embed_segments(
    audio_path: str,
    segments: List[Dict],
    ecapa,
    device: str,
) -> List[Tuple[np.ndarray, float]]:
    """Embed KCHI segments and return (embedding, duration) pairs."""
    if not segments:
        return []
    wav = load_audio_mono(audio_path)
    pairs = []
    for seg in segments:
        start_s = int(seg["start"] * SR)
        end_s = int(seg["end"] * SR)
        chunk = wav[start_s:end_s]
        if len(chunk) < int(0.5 * SR):
            continue
        try:
            chunk_dev = chunk.unsqueeze(0).to(device)
            emb = ecapa.encode_batch(chunk_dev).squeeze().detach().cpu().numpy()
            pairs.append((l2_normalize(emb), seg["dur"]))
        except Exception:
            pass
    return pairs


def build_prototypes(
    train_df: pd.DataFrame,
    ecapa,
    device: str,
) -> Dict[str, np.ndarray]:
    """Build per-(child_id, timepoint_norm) prototypes from BabAR KCHI train segments."""
    prototypes: Dict[str, np.ndarray] = {}
    pos_train = train_df[train_df["label"] == 1].copy()
    groups = list(pos_train.groupby(["child_id", "timepoint_norm"]))
    print(f"Building prototypes for {len(groups)} (child, timepoint) pairs "
          f"from {len(pos_train)} positive training clips...")
    n_missing_rttm = 0

    for idx, ((child_id, timepoint), sub) in enumerate(groups):
        proto_key = f"{child_id}__{timepoint}"
        all_embs: List[np.ndarray] = []
        all_durs: List[float] = []

        for row in sub.itertuples():
            segs = parse_kchi_segments(row.audio_path)
            if not segs:
                n_missing_rttm += 1
                continue
            pairs = embed_segments(row.audio_path, segs, ecapa, device)
            for emb, dur in pairs:
                all_embs.append(emb)
                all_durs.append(dur)

        if not all_embs:
            continue

        embs = np.stack(all_embs)
        weights = np.array(all_durs)
        proto = np.average(embs, axis=0, weights=weights)
        prototypes[proto_key] = l2_normalize(proto)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(groups)}] {len(prototypes)} prototypes built")

    print(f"Prototypes built: {len(prototypes)}/{len(groups)} "
          f"({n_missing_rttm} clips had no BabAR RTTM)")
    return prototypes


def window_and_score(
    audio_path: str,
    proto: np.ndarray,
    ecapa,
    device: str,
    mode: str,
    win_samples: int,
    hop_samples: int,
    min_samples: int,
) -> float:
    """Slide windows over raw clip and return similarity score."""
    wav = load_audio_mono(audio_path)
    sims: List[Tuple[float, float]] = []
    start = 0
    while start + min_samples <= len(wav):
        end = min(start + win_samples, len(wav))
        chunk = wav[start:end]
        if len(chunk) < min_samples:
            break
        try:
            chunk_dev = chunk.unsqueeze(0).to(device)
            emb = ecapa.encode_batch(chunk_dev).squeeze().detach().cpu().numpy()
            emb = l2_normalize(emb)
            sim = cosine_similarity(emb, proto)
            dur = len(chunk) / SR
            sims.append((sim, dur))
        except Exception:
            pass
        start += hop_samples

    if not sims:
        return 0.0

    if mode == "mean":
        total_dur = sum(d for _, d in sims)
        return float(sum(s * d for s, d in sims) / max(total_dur, 1e-8))
    elif mode == "max":
        return float(max(s for s, _ in sims))
    elif mode == "top3":
        top = sorted(sims, key=lambda x: x[0], reverse=True)[:3]
        return float(np.mean([s for s, _ in top]))
    raise ValueError(f"Unknown mode: {mode}")


def run_split(
    df: pd.DataFrame,
    prototypes: Dict[str, np.ndarray],
    ecapa,
    device: str,
    mode: str,
    win_samples: int,
    hop_samples: int,
    min_samples: int,
    max_clips: Optional[int] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    records = []
    total = min(len(df), max_clips) if max_clips else len(df)
    n_missing_proto = 0
    for i, row in enumerate(df.itertuples()):
        if max_clips and i >= max_clips:
            break
        proto_key = f"{row.child_id}__{row.timepoint_norm}"
        label = int(row.label)
        if dry_run:
            prob = 0.5
        elif proto_key not in prototypes:
            prob = 0.0
            n_missing_proto += 1
        else:
            try:
                prob = window_and_score(
                    row.audio_path, prototypes[proto_key],
                    ecapa, device, mode,
                    win_samples, hop_samples, min_samples,
                )
            except Exception as e:
                print(f"  [{i+1}/{total}] ERROR {Path(row.audio_path).name}: {e}")
                prob = 0.0
        records.append({"audio_path": row.audio_path, "label": label,
                        "child_id": row.child_id, "timepoint_norm": row.timepoint_norm,
                        "prob": prob})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]")
    if n_missing_proto > 0:
        print(f"  WARNING: {n_missing_proto} clips had no prototype (scored as 0.0)")
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mean", "max", "top3"], required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_BASE / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode={args.mode}  split={args.split}  device={args.device}")
    print(f"BabAR RTTM dir: {BABAR_RTTM_DIR}")
    print(f"Output: {out_dir}")

    win_samples = int(WIN_SEC * SR)
    hop_samples = int(HOP_SEC * SR)
    min_samples = int(0.5 * SR)

    # Load ECAPA embedder
    from speechbrain.inference.speaker import EncoderClassifier
    print(f"Loading ECAPA-TDNN from {ECAPA_SOURCE}...")
    ecapa = EncoderClassifier.from_hparams(
        source=ECAPA_SOURCE,
        run_opts={"device": args.device},
    )
    ecapa.eval()
    print("ECAPA loaded.")

    # Build prototypes from BabAR KCHI training segments
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    if "audio_exists" in train_df.columns:
        train_df = train_df[train_df["audio_exists"].astype(bool)]

    prototypes = build_prototypes(train_df, ecapa, args.device)

    if args.split == "test":
        val_metrics_path = out_dir / "val_metrics_tuned.json"
        if not val_metrics_path.exists():
            print(f"ERROR: {val_metrics_path} not found. Run --split val first.",
                  file=sys.stderr)
            sys.exit(2)
        with open(val_metrics_path) as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold={threshold:.4f} from val")

        meta_df = pd.read_csv(SPLITS_DIR / "test.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Test clips: {len(meta_df)}")

        preds = run_split(meta_df, prototypes, ecapa, args.device, args.mode,
                          win_samples, hop_samples, min_samples,
                          args.max_clips, args.dry_run)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "mode": args.mode, "n": len(preds)})

        save_json(metrics, str(out_dir / "test_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "test_predictions.csv"))

        rows = []
        for tp, grp in preds.groupby("timepoint_norm"):
            m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
            rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
        save_csv(pd.DataFrame(rows), str(out_dir / "test_metrics_by_timepoint.csv"))

    else:
        meta_df = pd.read_csv(SPLITS_DIR / "val.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Val clips: {len(meta_df)}")

        preds = run_split(meta_df, prototypes, ecapa, args.device, args.mode,
                          win_samples, hop_samples, min_samples,
                          args.max_clips, args.dry_run)
        threshold = tune_threshold(preds["label"].values, preds["prob"].values)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "mode": args.mode, "n": len(preds)})

        save_json(metrics, str(out_dir / "val_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "val_predictions.csv"))

        rows = []
        for tp, grp in preds.groupby("timepoint_norm"):
            m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
            rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
        save_csv(pd.DataFrame(rows), str(out_dir / "val_metrics_by_timepoint.csv"))

    save_json(
        {"mode": args.mode, "split": args.split, "win_sec": WIN_SEC, "hop_sec": HOP_SEC,
         "ecapa_source": ECAPA_SOURCE, "babar_rttm_dir": str(BABAR_RTTM_DIR),
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
