"""
VAD Coverage Baseline for child vocalization detection.

Scores each clip by the fraction of audio frames detected as speech (Silero VAD)
or voiced by RMS energy threshold. High speech_fraction → more likely child present.

This is an agnostic Tier-1 baseline: it doesn't model speaker identity, only speech
presence. Expected AUROC ~0.50–0.60; if near 0.5, confirms the task requires
speaker-specific discrimination rather than simple speech detection.

Modes:
  silero  — Silero VAD (neural, per 30ms frame)
  energy  — RMS energy > threshold, per 20ms frame

Usage:
    python baselines/vad_baseline.py --mode silero --split val
    python baselines/vad_baseline.py --mode silero --split test

    # Cross-child split:
    python baselines/vad_baseline.py --mode silero --split val \\
        --splits-dir baselines/splits \\
        --output-dir baselines/vad_baseline_runs/silero_cross_child

    # Dry run (5 clips):
    python baselines/vad_baseline.py --mode silero --split val --max-clips 5 --dry-run
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

SPLITS_DIR_SEEN = _REPO / "whisper-modeling/seen_child_splits"
SPLITS_DIR_CROSS = _REPO / "baselines/splits"
RESULTS_BASE = _REPO / "baselines/vad_baseline_runs"
SR = 16000


def load_audio(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0).numpy()


def score_silero(audio: np.ndarray, model, utils) -> float:
    get_speech_timestamps = utils[0]
    t = torch.from_numpy(audio.copy())
    try:
        timestamps = get_speech_timestamps(t, model, sampling_rate=SR)
        speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
        return float(speech_samples) / max(len(audio), 1)
    except Exception:
        return 0.5


def score_energy(audio: np.ndarray, frame_ms: int = 20, threshold_db: float = 40.0) -> float:
    frame_samples = int(SR * frame_ms / 1000)
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return 0.5
    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    # Linear threshold: -40 dBFS ≈ 0.01 linear (typical for normalized 16-bit audio)
    threshold_linear = 10 ** (-threshold_db / 20.0)
    voiced = (rms > threshold_linear).sum()
    return float(voiced) / n_frames


def run_split(
    df: pd.DataFrame,
    mode: str,
    silero_model=None,
    silero_utils=None,
    max_clips: int = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    records = []
    total = min(len(df), max_clips) if max_clips else len(df)
    for i, row in enumerate(df.itertuples()):
        if max_clips and i >= max_clips:
            break
        audio_path = row.audio_path
        label = int(row.label)
        if dry_run:
            prob = 0.5
        else:
            try:
                audio = load_audio(audio_path)
                if mode == "silero":
                    prob = score_silero(audio, silero_model, silero_utils)
                else:
                    prob = score_energy(audio)
            except Exception as e:
                print(f"  [{i+1}/{total}] ERROR {Path(audio_path).name}: {e}")
                prob = 0.5
        records.append({"audio_path": audio_path, "label": label, "prob": prob})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}]")
    return pd.DataFrame(records)


def add_timepoint_metrics(preds_df: pd.DataFrame, meta_df: pd.DataFrame,
                          threshold: float, out_path: Path) -> None:
    if "timepoint_norm" not in meta_df.columns:
        return
    tp_col = meta_df.set_index("audio_path")["timepoint_norm"]
    preds = preds_df.copy()
    preds["timepoint_norm"] = preds["audio_path"].map(tp_col)
    rows = []
    for tp, grp in preds.groupby("timepoint_norm"):
        m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
        rows.append({"timepoint_norm": tp, **m, "n": len(grp)})
    save_csv(pd.DataFrame(rows), str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["silero", "energy"], required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--splits-dir", default=None,
                   help="Path to splits dir (default: seen-child splits)")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: baselines/vad_baseline_runs/{mode})")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS_DIR_SEEN
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_BASE / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode={args.mode}  split={args.split}  splits_dir={splits_dir}")
    print(f"Output: {out_dir}")

    # Load Silero model once (only needed for silero mode)
    silero_model, silero_utils = None, None
    if args.mode == "silero" and not args.dry_run:
        print("Loading Silero VAD...")
        torch.hub.set_dir(str(_REPO / ".cache/torch_hub"))
        silero_model, silero_utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        silero_model.eval()
        print("Silero VAD loaded.")

    if args.split == "test":
        val_metrics_path = out_dir / "val_metrics_tuned.json"
        if not val_metrics_path.exists():
            print(f"ERROR: {val_metrics_path} not found. Run --split val first.",
                  file=sys.stderr)
            sys.exit(2)
        with open(val_metrics_path) as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold={threshold:.4f} from val")

        meta_df = pd.read_csv(splits_dir / "test.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Test clips: {len(meta_df)}")

        preds = run_split(meta_df, args.mode, silero_model, silero_utils,
                          args.max_clips, args.dry_run)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "mode": args.mode, "n": len(preds)})

        save_json(metrics, str(out_dir / "test_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "test_predictions.csv"))
        add_timepoint_metrics(preds, meta_df, threshold,
                              out_dir / "test_metrics_by_timepoint.csv")

    else:
        meta_df = pd.read_csv(splits_dir / "val.csv")
        if "audio_exists" in meta_df.columns:
            meta_df = meta_df[meta_df["audio_exists"].astype(bool)]
        print(f"Val clips: {len(meta_df)}")

        preds = run_split(meta_df, args.mode, silero_model, silero_utils,
                          args.max_clips, args.dry_run)
        threshold = tune_threshold(preds["label"].values, preds["prob"].values)
        metrics = compute_metrics(preds["label"].values, preds["prob"].values, threshold)
        metrics.update({"threshold": threshold, "mode": args.mode, "n": len(preds)})

        save_json(metrics, str(out_dir / "val_metrics_tuned.json"))
        save_csv(preds, str(out_dir / "val_predictions.csv"))
        add_timepoint_metrics(preds, meta_df, threshold,
                              out_dir / "val_metrics_by_timepoint.csv")

    save_json(
        {"mode": args.mode, "split": args.split,
         "splits_dir": str(splits_dir), "seed": args.seed},
        str(out_dir / "config.json"),
    )

    print(f"\n{args.split.capitalize()} metrics (threshold={threshold:.4f}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
