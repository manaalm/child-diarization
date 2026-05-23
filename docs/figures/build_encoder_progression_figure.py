"""Encoder progression figure — what each addition contributes.

Renders `docs/figures/encoder_progression.{png,pdf}`: a stacked ladder of the
six configurations from `encoders/baseline_encoders.py` main(), showing what
component each step adds over the previous one, plus headline test metrics
(BA-tuned threshold, canonical seen-child BIDS test, from
`evaluation/balanced_metrics_ba_tuned_summary.csv`).

Steps (matching Phase 1 → Phase 5 in baseline_encoders.py main()):
    1. Base: frozen Whisper-small encoder + mean pool + linear head
    2. + Attention pooling (replaces mean)
    3. + Layer-weighted sum (mix all 12 transformer layers before pool)
    4. + Stats pooling (mean+std replaces attn)
    5. + Fused encoder (Whisper + WavLM in parallel, concat)
    6. + Unfreeze last 2 layers of each encoder (no longer fully frozen)

Each row shades the NEW component(s) in orange; carried-over blocks are blue.
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# Visual constants
# -------------------------------------------------------------------
CARRY_FACE = "#dceaf6"   # blue: unchanged block carried from previous step
CARRY_EDGE = "#1f77b4"
NEW_FACE = "#ffe1c4"     # orange: NEW or CHANGED in this step
NEW_EDGE = "#c75c2d"
AUDIO_FACE = "#fff3d6"
AUDIO_EDGE = "#a07000"
POOL_FACE = "#e9f7da"
POOL_EDGE = "#5b9c44"
HEAD_FACE = "#f3e6f7"
HEAD_EDGE = "#7b3a8c"
PROB_FACE = "#f0f0f0"
PROB_EDGE = "#555555"

# Vertical bands inside each row (all rows span y=[0, 10]):
#   y in [7.4, 10] : title + description text
#   y in [4.2, 7.0] : architecture diagram (single-encoder rows)
#   y in [1.0, 7.0] : architecture diagram (fused rows — needs more height)
#   y in [0.0, 0.8] : footer notes (e.g., "last 2 layers trainable")
# Metrics box is placed at top-right of the title band.


def _block(ax, x, y, w, h, label, sublabel=None, face=CARRY_FACE, edge=CARRY_EDGE,
           is_new=False, lw=1.2):
    if is_new:
        face = NEW_FACE
        edge = NEW_EDGE
        lw = 2.0
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=8.8, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h * 0.25, sublabel, ha="center", va="center",
                fontsize=7.2, color="#444444")


def _arrow(ax, x1, y1, x2, y2, lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color="#555555"))


def _header(ax, title, description, metrics):
    """Title + wrapped description on the left; metrics card on the right."""
    ax.text(0.05, 9.55, title, ha="left", va="top",
            fontsize=11.5, fontweight="bold", color="#222222")
    wrapped = textwrap.fill(description, width=110)
    ax.text(0.05, 8.55, wrapped, ha="left", va="top",
            fontsize=8.8, color="#555555", style="italic")
    ax.text(10.95, 9.55, metrics, ha="right", va="top",
            fontsize=8.8, color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa",
                      edgecolor="#bbbbbb", linewidth=1.0))


# -------------------------------------------------------------------
# x-grid for single-encoder rows
#   audio   2.0       enc 2.0        [layermix 1.6]    pool 1.5      head 1.5    prob 0.9
# -------------------------------------------------------------------

def _single_encoder_row(ax, *, title, description, metrics,
                        pool_label="mean",
                        new_encoder=False, new_pool=False,
                        new_lw=False, show_lw=False,
                        new_unfreeze=False, unfreeze_note=None):
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    _header(ax, title, description, metrics)

    yb, hb = 4.4, 1.8
    cy = yb + hb / 2

    # x-coords
    audio_x = 0.30
    enc_x = 2.20
    lw_x = 4.30
    pool_x = 6.10 if show_lw else 4.30
    head_x = 7.95 if show_lw else 6.15
    prob_x = head_x + 1.85

    _block(ax, audio_x, yb, 1.5, hb, "audio", "16 kHz",
           face=AUDIO_FACE, edge=AUDIO_EDGE)
    _block(ax, enc_x, yb, 1.8, hb, "frozen", "Whisper-small\nencoder",
           is_new=new_encoder)
    if show_lw:
        _block(ax, lw_x, yb, 1.6, hb,
               "layer-w sum", "softmax(α)·\nstack(h₁..h₁₂)",
               is_new=new_lw, face=HEAD_FACE, edge=HEAD_EDGE)
    _block(ax, pool_x, yb, 1.6, hb, "pooling", pool_label,
           is_new=new_pool, face=POOL_FACE, edge=POOL_EDGE)
    _block(ax, head_x, yb, 1.6, hb, "linear", "FC head",
           face=HEAD_FACE, edge=HEAD_EDGE)
    _block(ax, prob_x, yb + 0.35, 0.85, hb - 0.7, "P(child)",
           face=PROB_FACE, edge=PROB_EDGE)

    # Arrows
    _arrow(ax, audio_x + 1.5, cy, enc_x, cy)
    _arrow(ax, enc_x + 1.8, cy, lw_x if show_lw else pool_x, cy)
    if show_lw:
        _arrow(ax, lw_x + 1.6, cy, pool_x, cy)
    _arrow(ax, pool_x + 1.6, cy, head_x, cy)
    _arrow(ax, head_x + 1.6, cy, prob_x, cy)

    if new_unfreeze and unfreeze_note:
        ax.text(enc_x + 0.9, yb - 0.45, unfreeze_note,
                ha="center", va="top", fontsize=8.2, color=NEW_EDGE,
                fontweight="bold")


def _fused_row(ax, *, title, description, metrics, new_unfreeze=False):
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    _header(ax, title, description, metrics)

    # Diagram band: y in roughly [1.2, 7.4]
    audio_x = 0.30
    enc_x = 2.20
    concat_x = 4.30
    pool_x = 6.10
    head_x = 7.95
    prob_x = 9.80

    # Audio (centred vertically)
    _block(ax, audio_x, 3.7, 1.5, 1.6, "audio", "16 kHz",
           face=AUDIO_FACE, edge=AUDIO_EDGE)

    # Two encoders in parallel — WavLM is the addition
    _block(ax, enc_x, 5.4, 1.8, 1.6, "frozen", "Whisper-small")
    _block(ax, enc_x, 1.8, 1.8, 1.6, "frozen", "WavLM-base+",
           is_new=True)

    _block(ax, concat_x, 3.7, 1.6, 1.6, "concat", "T × 1536",
           is_new=True, face=HEAD_FACE, edge=HEAD_EDGE)
    _block(ax, pool_x, 3.7, 1.6, 1.6, "attn pool", "1536-d",
           face=POOL_FACE, edge=POOL_EDGE)
    _block(ax, head_x, 3.7, 1.6, 1.6, "linear", "FC head",
           face=HEAD_FACE, edge=HEAD_EDGE)
    _block(ax, prob_x, 4.05, 0.95, 0.9, "P(child)",
           face=PROB_FACE, edge=PROB_EDGE)

    # Arrows
    cy = 4.5
    _arrow(ax, audio_x + 1.5, cy + 0.3, enc_x, 6.2)
    _arrow(ax, audio_x + 1.5, cy - 0.3, enc_x, 2.6)
    _arrow(ax, enc_x + 1.8, 6.2, concat_x, cy + 0.5)
    _arrow(ax, enc_x + 1.8, 2.6, concat_x, cy - 0.5)
    _arrow(ax, concat_x + 1.6, cy, pool_x, cy)
    _arrow(ax, pool_x + 1.6, cy, head_x, cy)
    _arrow(ax, head_x + 1.6, cy, prob_x, cy)

    if new_unfreeze:
        ax.text(enc_x + 0.9, 1.3,
                "last 2 transformer layers trainable in BOTH encoders",
                ha="center", va="top", fontsize=8.2, color=NEW_EDGE,
                fontweight="bold")


# -------------------------------------------------------------------
# Headline metrics — pulled from evaluation/balanced_metrics_ba_tuned_summary.csv
# (canonical seen-child BIDS test; n=635 for splits retrained post-BIDS
# correction, n=441 for legacy retrains.)
# -------------------------------------------------------------------
METRICS = {
    "mean":             ("F1 0.862   BA 0.795", "AUROC 0.870", "n=635"),
    "attn":             ("F1 0.880   BA 0.779", "AUROC 0.848", "n=635"),
    "attn_lw":          ("F1 0.876   BA 0.776", "AUROC 0.886", "n=441"),
    "stats_lw":         ("F1 0.850   BA 0.722", "AUROC 0.871", "n=441"),
    "fused_attn":       ("F1 0.870   BA 0.749", "AUROC 0.886", "n=441"),
    "fused_unfreeze2":  ("F1 0.892   BA 0.797", "AUROC 0.903", "n=441"),
}


def _fmt(key):
    a, b, n = METRICS[key]
    return f"{a}\n{b}\n{n}, BA-tuned"


def main():
    nrows = 6
    fig = plt.figure(figsize=(14, 22))
    gs = fig.add_gridspec(
        nrows=nrows, ncols=1,
        height_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.20,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]

    _single_encoder_row(
        axes[0],
        title="Step 1 — Base baseline",
        description=("Frozen Whisper-small encoder. Take the last hidden state, "
                     "mean-pool over time, run a 1-layer FC head."),
        metrics=_fmt("mean"),
        pool_label="mean",
        new_encoder=True, new_pool=True,
    )

    _single_encoder_row(
        axes[1],
        title="Step 2 — + Attention pooling",
        description=("Replace mean with a learned attention head that scores each frame and "
                     "computes a softmax-weighted sum — lets the model focus on child-vocal frames."),
        metrics=_fmt("attn"),
        pool_label="attention",
        new_pool=True,
    )

    _single_encoder_row(
        axes[2],
        title="Step 3 — + Layer-weighted sum",
        description=("Mix all 12 transformer layers with a learned softmax(α) instead of using only the "
                     "last hidden state — earlier layers carry phonetic info, later layers carry semantics."),
        metrics=_fmt("attn_lw"),
        pool_label="attention",
        show_lw=True, new_lw=True,
    )

    _single_encoder_row(
        axes[3],
        title="Step 4 — + Statistical pooling (mean + std)",
        description=("Swap attention for mean+std pooling — the head gets both central tendency "
                     "and variability across frames (output dim doubles to 1536)."),
        metrics=_fmt("stats_lw"),
        pool_label="mean + std\n(2× dim)",
        show_lw=True, new_pool=True,
    )

    _fused_row(
        axes[4],
        title="Step 5 — + Fused encoder (Whisper + WavLM)",
        description=("Add a parallel WavLM-base+ branch; concatenate per-frame embeddings (T × 1536) "
                     "before pooling — combines Whisper's ASR-style features with WavLM's speaker-style features."),
        metrics=_fmt("fused_attn"),
    )

    _fused_row(
        axes[5],
        title="Step 6 — + Unfreeze last 2 layers of each encoder",
        description=("Stop freezing the top 2 transformer layers in both branches; train them at lr=1e-5 "
                     "(head still at lr=1e-3). Small targeted adaptation, biggest single jump in BA + AUROC."),
        metrics=_fmt("fused_unfreeze2"),
        new_unfreeze=True,
    )

    legend_handles = [
        mpatches.Patch(facecolor=NEW_FACE, edgecolor=NEW_EDGE,
                       label="new / changed in this step"),
        mpatches.Patch(facecolor=CARRY_FACE, edgecolor=CARRY_EDGE,
                       label="carried over from previous step"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.985), fontsize=9.5, frameon=True)

    fig.suptitle(
        "Encoder baseline progression — what each addition contributes",
        fontsize=15, fontweight="bold", y=0.997,
    )

    fig.text(
        0.5, 0.005,
        "Metrics: balanced-accuracy-tuned threshold on canonical seen-child BIDS test. "
        "n=635 for splits retrained post-BIDS correction; n=441 for legacy retrains. "
        "Source: evaluation/balanced_metrics_ba_tuned_summary.csv (spec-022).",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    png = os.path.join(OUT_DIR, "encoder_progression.png")
    pdf = os.path.join(OUT_DIR, "encoder_progression.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
