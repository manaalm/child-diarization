"""Render the encoder-pipeline figure for the thesis chapter (spec 022 US4 / FR-018).

Produces `docs/figures/encoder_pipeline.{png,pdf}` showing the canonical
4-step pipeline used by the encoder baselines, plus a fused-encoder panel.

Pipeline:
    waveform → frozen encoder → pooling → linear classifier → score

Variants:
    1. Whisper × {mean, attention} pooling
    2. WavLM × {mean, attention} pooling
    3. Fused (Whisper + WavLM, concat-then-pool) × attention pooling
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _block(ax, x, y, w, h, label, sublabel=None, color="#dceaf6", edge="#1f77b4"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                fontsize=8.5, color="#444444")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))


def _panel_single(ax, encoder_label, pool_label, title, hidden_dim):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11.5, pad=8)

    _block(ax, 0.2, 1.3, 1.6, 1.4, "audio", "16 kHz mono", "#fff3d6", "#a07000")
    _block(ax, 2.2, 1.3, 1.8, 1.4, "frozen", encoder_label, "#dceaf6", "#1f77b4")
    _block(ax, 4.4, 1.3, 1.8, 1.4, "pooling", pool_label, "#e9f7da", "#5b9c44")
    _block(ax, 6.6, 1.3, 1.8, 1.4, "linear", "1-layer FC", "#fbe1d8", "#c75c2d")
    _block(ax, 8.8, 1.5, 1.0, 1.0, "P(child)", None, "#f0f0f0", "#555555")

    _arrow(ax, 1.8, 2.0, 2.2, 2.0)
    _arrow(ax, 4.0, 2.0, 4.4, 2.0)
    _arrow(ax, 6.2, 2.0, 6.6, 2.0)
    _arrow(ax, 8.4, 2.0, 8.8, 2.0)

    # Shape annotations beneath arrows
    ax.text(2.0, 1.05, f"(T, {hidden_dim})", ha="center", fontsize=8, color="#555555")
    ax.text(4.2, 1.05, f"(T, {hidden_dim})", ha="center", fontsize=8, color="#555555")
    ax.text(6.4, 1.05, f"({hidden_dim},)", ha="center", fontsize=8, color="#555555")
    ax.text(8.6, 1.05, "(1,)", ha="center", fontsize=8, color="#555555")


def _panel_fused(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_axis_off()
    ax.set_title("Fused encoder (Whisper + WavLM)", fontsize=11.5, pad=8)

    _block(ax, 0.2, 1.4, 1.4, 1.2, "audio", "16 kHz mono", "#fff3d6", "#a07000")

    # Two encoders in parallel
    _block(ax, 2.0, 2.4, 1.7, 1.0, "frozen", "Whisper-small (768)", "#dceaf6", "#1f77b4")
    _block(ax, 2.0, 0.6, 1.7, 1.0, "frozen", "WavLM-base+ (768)", "#dceaf6", "#1f77b4")

    _block(ax, 4.1, 1.4, 1.4, 1.2, "concat", "T × 1536", "#f3e6f7", "#7b3a8c")
    _block(ax, 5.9, 1.4, 1.4, 1.2, "attn pool", "1536-d", "#e9f7da", "#5b9c44")
    _block(ax, 7.7, 1.4, 1.4, 1.2, "linear", "1-layer FC", "#fbe1d8", "#c75c2d")
    _block(ax, 9.3, 1.6, 0.55, 0.8, "P", None, "#f0f0f0", "#555555")

    _arrow(ax, 1.6, 2.4, 2.0, 2.9)
    _arrow(ax, 1.6, 1.6, 2.0, 1.1)
    _arrow(ax, 3.7, 2.9, 4.1, 2.3)
    _arrow(ax, 3.7, 1.1, 4.1, 1.7)
    _arrow(ax, 5.5, 2.0, 5.9, 2.0)
    _arrow(ax, 7.3, 2.0, 7.7, 2.0)
    _arrow(ax, 9.1, 2.0, 9.3, 2.0)


def main():
    fig, axes = plt.subplots(5, 1, figsize=(10, 14.5))
    _panel_single(axes[0], "Whisper-small (768)", "mean", "Whisper × mean pool", 768)
    _panel_single(axes[1], "Whisper-small (768)", "attention", "Whisper × attention pool", 768)
    _panel_single(axes[2], "WavLM-base+ (768)", "mean", "WavLM × mean pool", 768)
    _panel_single(axes[3], "WavLM-base+ (768)", "attention", "WavLM × attention pool", 768)
    _panel_fused(axes[4])
    fig.suptitle("Encoder baselines — frozen backbone → pooling → linear head",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    png = os.path.join(OUT_DIR, "encoder_pipeline.png")
    pdf = os.path.join(OUT_DIR, "encoder_pipeline.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
