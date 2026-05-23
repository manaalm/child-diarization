"""Qualitative figures: spectrogram + seg-MIL attention overlay.

Picks:
  1 confident-correct positive   (label=1, pred=1, high prob)
  1 confident-correct negative   (label=0, pred=0, low prob)
  1 false positive               (label=0, pred=1, high prob)
  1 false negative               (label=1, pred=0, low prob)

For each, loads the audio, computes a mel-spectrogram via torchaudio, and
overlays the seg-MIL attention weights as colored bands on the time axis.

Outputs PNGs to evaluation/figures/qual_*.png.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
PRED = os.path.join(REPO, "mil/mil_results/seg_mil/babar_vtc_gated_attention/test_predictions.csv")
WEIGHTS = os.path.join(REPO, "mil/mil_results/seg_mil/babar_vtc_gated_attention/test_segment_weights.csv")
FIG_DIR = os.path.join(REPO, "evaluation", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def pick_examples(pred_df: pd.DataFrame) -> dict:
    out = {}
    # Confident correct positive
    pos = pred_df[(pred_df["label"] == 1) & (pred_df["pred"] == 1)].sort_values("prob", ascending=False)
    if len(pos) > 0:
        out["correct_positive"] = pos.iloc[0]
    # Confident correct negative
    neg = pred_df[(pred_df["label"] == 0) & (pred_df["pred"] == 0)].sort_values("prob")
    if len(neg) > 0:
        out["correct_negative"] = neg.iloc[0]
    # False positive (high prob, label=0)
    fp = pred_df[(pred_df["label"] == 0) & (pred_df["pred"] == 1)].sort_values("prob", ascending=False)
    if len(fp) > 0:
        out["false_positive"] = fp.iloc[0]
    # False negative (low prob, label=1)
    fn = pred_df[(pred_df["label"] == 1) & (pred_df["pred"] == 0)].sort_values("prob")
    if len(fn) > 0:
        out["false_negative"] = fn.iloc[0]
    return out


def plot_one(audio_path: str, segs: pd.DataFrame, prob: float, label: int, pred: int, title: str, out_png: str):
    if not os.path.isfile(audio_path):
        print(f"SKIP {title}: missing {audio_path}")
        return
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    duration = wav.shape[1] / sr

    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=160, n_mels=80, f_min=20, f_max=8000
    )
    mel = mel_xform(wav).squeeze(0)
    mel_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel).numpy()

    fig, (ax_spec, ax_att) = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1]})
    extent = [0, duration, 0, sr / 2 / 1000]
    ax_spec.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=extent)
    ax_spec.set_ylabel("kHz")
    color_target = "tab:green" if label == 1 else "tab:red"
    ax_spec.set_title(f"{title}    label={label} pred={pred} prob={prob:.3f}", color=color_target)

    # Attention overlay below spectrogram
    if len(segs) > 0:
        max_w = max(segs["attention_weight"].max(), 1e-6)
        for _, r in segs.iterrows():
            w = float(r["attention_weight"]) / max_w
            ax_att.axvspan(r["seg_start"], r["seg_end"], color="tab:blue", alpha=0.15 + 0.7 * w)
        # Mark top-attended segment
        top_pos = int(segs["attention_weight"].to_numpy().argmax())
        top = segs.iloc[top_pos]
        ax_att.axvline(top["seg_start"], color="orange", linestyle="--", alpha=0.7)
        ax_att.axvline(top["seg_end"], color="orange", linestyle="--", alpha=0.7)
    ax_att.set_xlim(0, duration)
    ax_att.set_ylim(0, 1)
    ax_att.set_xlabel("Time (s)")
    ax_att.set_yticks([])
    ax_att.set_ylabel("Attn", rotation=0, labelpad=20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"OK   {title} → {out_png}")


def main():
    pred_df = pd.read_csv(PRED)
    weights_df = pd.read_csv(WEIGHTS)
    examples = pick_examples(pred_df)
    print(f"Picked {len(examples)} examples")

    for kind, row in examples.items():
        ap = row["audio_path"]
        segs = weights_df[weights_df["audio_path"] == ap].copy()
        out = os.path.join(FIG_DIR, f"qual_{kind}.png")
        plot_one(ap, segs, float(row["prob"]), int(row["label"]), int(row["pred"]),
                 f"Seg-MIL gated_attn — {kind.replace('_', ' ')}", out)


if __name__ == "__main__":
    main()
