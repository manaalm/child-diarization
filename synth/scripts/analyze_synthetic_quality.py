#!/usr/bin/env python3
"""
Analyze distribution similarity between synthetic and real training data.

Produces:
    {output_dir}/duration_distribution.png
    {output_dir}/snr_distribution.png
    {output_dir}/loudness_distribution.png
    {output_dir}/child_adult_ratio.png
    {output_dir}/real_vs_synthetic_embedding_umap.png  (if umap-learn installed)

Usage:
    python synth/scripts/analyze_synthetic_quality.py \\
      --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \\
      --real-train-csv     whisper-modeling/seen_child_splits/train.csv \\
      --output-dir         synth_results/augmentation_experiments/default_14_18mo/figures/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_synth(synth_csv: str) -> pd.DataFrame:
    df = pd.read_csv(synth_csv, low_memory=False)
    required = {"target_child_duration_sec", "adult_duration_sec",
                "age_band", "snr_db", "synthetic_scene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Synthetic manifest missing columns: {sorted(missing)}")
    return df


def _load_real(real_csv: str) -> pd.DataFrame:
    df = pd.read_csv(real_csv, low_memory=False)
    return df


def _plot_duration_hist(synth_df, real_df, out_dir):
    """Duration distribution: real vs. synthetic total child duration per clip."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping duration plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Synthetic: child + adult duration per scene
    axes[0].hist(
        synth_df["target_child_duration_sec"].dropna(),
        bins=40, alpha=0.7, color="steelblue", label="Synthetic child",
    )
    axes[0].hist(
        synth_df["adult_duration_sec"].dropna(),
        bins=40, alpha=0.7, color="darkorange", label="Synthetic adult",
    )
    axes[0].set_xlabel("Duration per scene (sec)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Synthetic: child vs. adult duration")
    axes[0].legend()

    # By age band
    for band, grp in synth_df.groupby("age_band"):
        axes[1].hist(
            grp["target_child_duration_sec"].dropna(),
            bins=30, alpha=0.6, label=band,
        )
    axes[1].set_xlabel("Child duration per scene (sec)")
    axes[1].set_title("Synthetic: child duration by age band")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(str(out_dir / "duration_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote duration_distribution.png")


def _plot_snr_hist(synth_df, out_dir):
    """SNR distribution of synthetic clips."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    snr_vals = synth_df["snr_db"].dropna()
    if len(snr_vals) == 0:
        print("  No SNR data in synthetic manifest; skipping SNR plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(snr_vals, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Synthetic: SNR Distribution")
    fig.tight_layout()
    fig.savefig(str(out_dir / "snr_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote snr_distribution.png")


def _plot_loudness(synth_df, out_dir):
    """Approximate loudness from child_duration as a proxy for activity level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    synth_df = synth_df.copy()
    # Proxy: fraction of scene with child speech
    duration_sec = 30.0  # default scene duration
    synth_df["child_activity"] = synth_df["target_child_duration_sec"] / duration_sec

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(synth_df["child_activity"].dropna(), bins=30, color="mediumseagreen",
            edgecolor="white")
    ax.set_xlabel("Child speech fraction (per 30s scene)")
    ax.set_ylabel("Count")
    ax.set_title("Synthetic: Child Activity Distribution")
    fig.tight_layout()
    fig.savefig(str(out_dir / "loudness_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote loudness_distribution.png")


def _plot_child_adult_ratio(synth_df, out_dir):
    """Child-to-adult duration ratio per scene."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    child = synth_df["target_child_duration_sec"].fillna(0)
    adult = synth_df["adult_duration_sec"].fillna(0)
    total = child + adult
    ratio = np.where(total > 0, child / total, 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratio, bins=30, color="orchid", edgecolor="white")
    ax.set_xlabel("Child / (Child + Adult) duration fraction")
    ax.set_ylabel("Count")
    ax.set_title("Synthetic: Child-to-Adult Duration Ratio")
    fig.tight_layout()
    fig.savefig(str(out_dir / "child_adult_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote child_adult_ratio.png")


def _plot_umap_embeddings(synth_df, real_df, out_dir, encoder_model, n_clips=200):
    """Optional UMAP of WavLM embeddings (real vs. synthetic)."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed; skipping UMAP plot.")
        return

    try:
        import soundfile as sf
        import torch
        from transformers import AutoModel, AutoFeatureExtractor
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Required libraries for UMAP not available.")
        return

    print(f"  Computing {encoder_model} embeddings for UMAP …")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(encoder_model)
    model = AutoModel.from_pretrained(encoder_model).to(device)
    model.eval()

    def _embed(audio_path: str) -> Optional[np.ndarray]:
        try:
            wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import torchaudio.functional as F_ta
                import torch
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                wav = F_ta.resample(wav_t, sr, 16000).squeeze(0).numpy()
            # Take first 5 seconds
            wav = wav[:80000]
            inputs = extractor(wav, sampling_rate=16000, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state.mean(dim=1)
            return out.squeeze(0).cpu().numpy()
        except Exception:
            return None

    rng = np.random.default_rng(42)
    synth_sample = synth_df.sample(min(n_clips, len(synth_df)),
                                   random_state=42)["audio_path"].tolist()
    real_audio_col = "audio_path" if "audio_path" in real_df.columns else None
    if real_audio_col is None:
        print("  Real CSV has no audio_path column; skipping UMAP.")
        return
    real_sample = real_df.sample(min(n_clips, len(real_df)),
                                 random_state=42)[real_audio_col].tolist()

    synth_embs = [e for p in synth_sample if (e := _embed(p)) is not None]
    real_embs = [e for p in real_sample if (e := _embed(p)) is not None]

    if not synth_embs or not real_embs:
        print("  No embeddings computed; skipping UMAP.")
        return

    X = np.stack(synth_embs + real_embs)
    labels = ["synthetic"] * len(synth_embs) + ["real"] * len(real_embs)

    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, color in [("synthetic", "steelblue"), ("real", "darkorange")]:
        mask = np.array(labels) == lbl
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.5, color=color, label=lbl)
    ax.legend()
    ax.set_title("UMAP: Real vs. Synthetic WavLM Embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(out_dir / "real_vs_synthetic_embedding_umap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote real_vs_synthetic_embedding_umap.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze distribution similarity: synthetic vs. real."
    )
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--real-train-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--encoder-model",
        default="microsoft/wavlm-base-plus",
        help="HuggingFace model for UMAP embeddings (default: wavlm-base-plus).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading synthetic manifest: {args.synthetic_manifest}")
    synth_df = _load_synth(args.synthetic_manifest)
    print(f"  {len(synth_df)} synthetic scenes.")

    print(f"Loading real train CSV: {args.real_train_csv}")
    real_df = _load_real(args.real_train_csv)
    print(f"  {len(real_df)} real clips.")

    print(f"\nGenerating figures → {out_dir}")
    _plot_duration_hist(synth_df, real_df, out_dir)
    _plot_snr_hist(synth_df, out_dir)
    _plot_loudness(synth_df, out_dir)
    _plot_child_adult_ratio(synth_df, out_dir)
    _plot_umap_embeddings(synth_df, real_df, out_dir,
                          args.encoder_model, n_clips=200)

    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
