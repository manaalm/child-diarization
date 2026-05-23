"""Generate the pipeline-family summary figure for the thesis.

Horizontal bar chart of seen-child test AUROC, grouped by pipeline family
(Diarization+Enrollment, Encoder+MIL, AV/Manual Fusion, Ensemble), with the
zero-shot Audio LLM as an extra reference row. Numbers are sourced from
canonical result JSONs — update the SYSTEMS list if metrics change.

Usage:
    python synthesis/scripts/plot_pipeline_family_summary.py
    # → figures/pipeline_family_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "pipeline_family_summary.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (family, label, F1, AUROC, AUPRC, marker_kind)
# marker_kind ∈ {"normal", "best", "null"} — "null" = degenerate / random;
# "best" = bolded as family-best.
SYSTEMS = [
    # Diarization + enrollment (10 frontends + BabAR combined)
    ("Diarization + Enrollment", "USC-SAIL",                 0.874, 0.625, 0.793, "normal"),
    ("Diarization + Enrollment", "Pyannote",                 0.858, 0.661, 0.826, "normal"),
    ("Diarization + Enrollment", "VBx",                      0.858, 0.686, 0.851, "normal"),
    ("Diarization + Enrollment", "Sortformer",               0.844, 0.664, 0.841, "normal"),
    ("Diarization + Enrollment", "EEND-EDA",                 0.844, 0.528, 0.781, "null"),
    ("Diarization + Enrollment", "VTC (KCHI+OCH)",           0.888, 0.787, 0.895, "normal"),
    ("Diarization + Enrollment", "BabAR / VTC-KCHI",         0.874, 0.820, 0.918, "normal"),
    ("Diarization + Enrollment", "BabAR combined (LR)",      0.881, 0.858, 0.944, "best"),
    ("Diarization + Enrollment", "TalkNet-ASD",              0.336, 0.569, 0.791, "null"),
    ("Diarization + Enrollment", "LocoNet-ECAPA",            0.000, 0.500, 0.760, "null"),

    # Encoder + MIL
    ("Encoder + MIL", "WavLM-MIL (frame-window)",            0.882, 0.771, 0.893, "normal"),
    ("Encoder + MIL", "HuBERT-large MIL (layersum)",         0.878, 0.813, 0.920, "normal"),
    ("Encoder + MIL", "Seg-MIL (babar+vtc exp-softmax)",     0.877, 0.816, 0.924, "normal"),
    ("Encoder + MIL", "Pseudo-frame WavLM",                  0.869, 0.831, 0.937, "normal"),
    ("Encoder + MIL", "Whisper-MIL ACMIL-max",               0.891, 0.842, 0.936, "normal"),
    ("Encoder + MIL", "Whisper-MIL (frame-window)",          0.886, 0.853, 0.946, "normal"),
    ("Encoder + MIL", "Whisper-MIL TS-MIL concat",           0.896, 0.869, 0.944, "best"),

    # AV / Manual fusion
    ("AV / Manual Fusion", "AV always-fuse (manual+age-band)", 0.882, 0.853, 0.941, "normal"),
    ("AV / Manual Fusion", "Speaker-informed AV (US3)",       0.904, 0.871, 0.953, "best"),

    # Foundation / zero-shot reference (TODO: update F1/AUROC/AUPRC after
    # Qwen2.5-Omni-7B SLURM job 13152385 lands; current numbers are from the
    # v1 Qwen2-Audio-7B run preserved at qwen2_audio_7b/)
    ("Foundation (zero-shot)", "Audio LLM (Qwen2.5-Omni-7B)",  0.871, 0.725, 0.853, "normal"),

    # Ensemble
    ("Ensemble", "best_audio_mil (mean)",                    0.893, 0.878, 0.956, "normal"),
    ("Ensemble", "12-sys metadata stacker",                  0.905, 0.904, 0.966, "normal"),
    ("Ensemble", "12-sys + visual stacker",                  0.898, 0.905, 0.968, "best"),
]

FAMILY_ORDER = [
    "Diarization + Enrollment",
    "Encoder + MIL",
    "AV / Manual Fusion",
    "Foundation (zero-shot)",
    "Ensemble",
]
# Tab10-derived colors, one per family
FAMILY_COLORS = {
    "Diarization + Enrollment": "#1f77b4",   # blue
    "Encoder + MIL":            "#2ca02c",   # green
    "AV / Manual Fusion":       "#d62728",   # red
    "Foundation (zero-shot)":   "#9467bd",   # purple
    "Ensemble":                 "#ff7f0e",   # orange
}


def main() -> None:
    # Sort within each family by AUROC ascending (so best-of-family ends up
    # at the top of its band when we plot reversed)
    grouped: dict[str, list] = {f: [] for f in FAMILY_ORDER}
    for fam, label, f1, auroc, auprc, kind in SYSTEMS:
        grouped[fam].append((label, f1, auroc, auprc, kind))
    for f in grouped:
        grouped[f].sort(key=lambda r: r[2])  # by AUROC

    # Build flat lists with separators between families
    labels, aurocs, colors, kinds, family_for_row = [], [], [], [], []
    family_band_starts: dict[str, int] = {}
    for fam in FAMILY_ORDER:
        family_band_starts[fam] = len(labels)
        for label, f1, auroc, auprc, kind in grouped[fam]:
            labels.append(label)
            aurocs.append(auroc)
            colors.append(FAMILY_COLORS[fam])
            kinds.append(kind)
            family_for_row.append(fam)

    n = len(labels)
    fig_h = max(5.5, 0.28 * n + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ypos = np.arange(n)

    # alpha must be scalar; apply per-bar via face color RGBA after
    bars = ax.barh(
        ypos, aurocs, color=colors,
        edgecolor=["black" if k == "best" else "none" for k in kinds],
        linewidth=[1.6 if k == "best" else 0 for k in kinds],
    )
    for bar, kind in zip(bars, kinds):
        bar.set_alpha(0.45 if kind == "null" else 1.0)

    # Annotate each bar with its AUROC
    for y, val, k in zip(ypos, aurocs, kinds):
        weight = "bold" if k == "best" else "normal"
        ax.text(val + 0.005, y, f"{val:.3f}", va="center", ha="left",
                fontsize=8.5, fontweight=weight)

    # Reference lines
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6,
               label="chance (0.5)")
    ax.axvline(0.878, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="best_audio_mil mean (0.878)")

    # Y axis
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # families top-to-bottom in declared order

    # Family band shading + label on the right
    for fam in FAMILY_ORDER:
        rows = [i for i, f in enumerate(family_for_row) if f == fam]
        if not rows:
            continue
        y0, y1 = min(rows) - 0.5, max(rows) + 0.5
        ax.axhspan(y0, y1, color=FAMILY_COLORS[fam], alpha=0.06, zorder=0)
        # Family label in the right margin, vertically centered
        ax.text(
            1.005, (y0 + y1) / 2, fam,
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=10, fontweight="bold",
            color=FAMILY_COLORS[fam],
        )

    # X axis
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Test AUROC (seen-child, n=441)", fontsize=10)
    ax.set_title(
        "Pipeline family summary — clip-level child-presence detection\n"
        "(seen-child test split; n=441; bold = family-best; faded = null/degenerate)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave room for right-margin family labels
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB after save)")


if __name__ == "__main__":
    main()
