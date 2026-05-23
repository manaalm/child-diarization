"""Calibration analysis for the headline systems.

For each system in TOP_SYSTEMS, reads val + test predictions, computes:
  - Brier score (BS)
  - Expected Calibration Error, 10 equal-width bins (ECE)
  - Maximum Calibration Error (MCE)
  - Per-bin reliability data (mean predicted prob vs actual accuracy)
  - Optional: temperature-scaled (single scalar T fit on val) variants

Outputs:
  evaluation/calibration_metrics.csv      one row per (system, variant)
  evaluation/calibration_per_bin.csv      per-bin reliability points (long)
  evaluation/figures/reliability_<sys>.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
OUT_CSV = os.path.join(REPO, "evaluation", "calibration_metrics.csv")
OUT_BINS = os.path.join(REPO, "evaluation", "calibration_per_bin.csv")
FIG_DIR = os.path.join(REPO, "evaluation", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class System:
    name: str
    test_csv: str
    val_csv: str
    score_col: str


TOP_SYSTEMS = [
    System(
        "whisper_mil",
        "mil/mil_results/whisper_mil/test_predictions.csv",
        "mil/mil_results/whisper_mil/val_predictions.csv",
        "score",
    ),
    System(
        "whisper_mil_tsmil_concat",
        "mil/mil_results/whisper_mil_tsmil_concat/test_predictions.csv",
        "mil/mil_results/whisper_mil_tsmil_concat/val_predictions.csv",
        "score",
    ),
    System(
        "wavlm_mil",
        "mil/mil_results/wavlm_mil/test_predictions.csv",
        "mil/mil_results/wavlm_mil/val_predictions.csv",
        "score",
    ),
    System(
        "babar_enrollment",
        "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",
        "babar_ecapa_enrollment_runs/enroll_val_predictions.csv",
        "prob",
    ),
    System(
        "vtc_enrollment",
        "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv",
        "vtc_ecapa_enrollment_runs/enroll_val_predictions.csv",
        "prob",
    ),
    System(
        "vtc_kchi_enrollment",
        "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv",
        "vtc_kchi_ecapa_enrollment_runs/enroll_val_predictions.csv",
        "prob",
    ),
    System(
        "best_audio_mil_ensemble",
        "ensemble_runs/test_predictions.csv",
        "",  # no val file at this path; temp-scaling skipped
        "best_audio_mil_mean",
    ),
    System(
        "metadata_stacker",
        "ensemble_runs/metadata_stack/test_predictions.csv",
        "ensemble_runs/metadata_stack/val_predictions.csv",
        "score",
    ),
    System(
        "wavlm_pseudo_frame",
        "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv",
        "pseudo_frame/results/wavlm_pseudo_frame/val_predictions.csv",
        "score",
    ),
    System(
        "audio_llm_qwen25_omni",
        "baselines/audio_llm_baseline_runs/qwen25_omni_7b/test_predictions.csv",
        "baselines/audio_llm_baseline_runs/qwen25_omni_7b/val_predictions.csv",
        "prob",
    ),
]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Equal-width binning. Returns DataFrame with one row per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                dict(bin=i, lo=lo, hi=hi, n=0, mean_pred=np.nan, mean_label=np.nan, gap=np.nan)
            )
            continue
        mean_pred = float(p[mask].mean())
        mean_label = float(y[mask].mean())
        rows.append(
            dict(bin=i, lo=lo, hi=hi, n=n, mean_pred=mean_pred, mean_label=mean_label, gap=abs(mean_pred - mean_label))
        )
    return pd.DataFrame(rows)


def ece_mce(bins_df: pd.DataFrame) -> tuple[float, float]:
    bins_used = bins_df.dropna(subset=["mean_pred"])
    total = bins_used["n"].sum()
    if total == 0:
        return float("nan"), float("nan")
    weighted = (bins_used["n"] / total) * bins_used["gap"]
    return float(weighted.sum()), float(bins_used["gap"].max())


def temperature_scale(p_val: np.ndarray, y_val: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Single-scalar temperature scaling. Logit-clip 1e-6 to avoid inf.

    Fits T to minimize NLL on val, then applies to test logits.
    """
    eps = 1e-6
    p_val = np.clip(p_val, eps, 1 - eps)
    p_test = np.clip(p_test, eps, 1 - eps)
    logits_val = np.log(p_val / (1 - p_val))
    logits_test = np.log(p_test / (1 - p_test))

    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        z = logits_val / T
        log_p = -np.log1p(np.exp(-z))
        log_1mp = -z - np.log1p(np.exp(-z))
        return float(-np.mean(y_val * log_p + (1 - y_val) * log_1mp))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T = float(res.x)
    z = logits_test / T
    return 1.0 / (1.0 + np.exp(-z)), T


def load_pred(path: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    p = df[col].astype(float).clip(0.0, 1.0).to_numpy()
    y = df["label"].astype(int).to_numpy()
    return p, y


def main():
    metric_rows = []
    bin_rows = []
    for sys in TOP_SYSTEMS:
        test_p_path = os.path.join(REPO, sys.test_csv)
        val_p_path = os.path.join(REPO, sys.val_csv)
        if not os.path.exists(test_p_path):
            print(f"SKIP {sys.name}: missing test {test_p_path}")
            continue
        try:
            p_test, y_test = load_pred(test_p_path, sys.score_col)
        except Exception as e:
            print(f"SKIP {sys.name}: load error {e}")
            continue

        # Raw calibration
        bs = brier(p_test, y_test)
        bins = reliability_bins(p_test, y_test, n_bins=10)
        ece, mce = ece_mce(bins)
        n_pos = int(y_test.sum())
        n_neg = int((1 - y_test).sum())
        metric_rows.append(
            dict(system=sys.name, variant="raw", n=len(y_test), n_pos=n_pos, n_neg=n_neg,
                 brier=bs, ece10=ece, mce10=mce, temperature=np.nan)
        )
        for _, r in bins.iterrows():
            bin_rows.append(dict(system=sys.name, variant="raw", **r.to_dict()))

        # Temperature-scaled (if val exists)
        if sys.val_csv and os.path.isfile(val_p_path):
            try:
                p_val, y_val = load_pred(val_p_path, sys.score_col)
                p_test_T, T = temperature_scale(p_val, y_val, p_test)
                bs_T = brier(p_test_T, y_test)
                bins_T = reliability_bins(p_test_T, y_test, n_bins=10)
                ece_T, mce_T = ece_mce(bins_T)
                metric_rows.append(
                    dict(system=sys.name, variant="temp_scaled", n=len(y_test), n_pos=n_pos, n_neg=n_neg,
                         brier=bs_T, ece10=ece_T, mce10=mce_T, temperature=T)
                )
                for _, r in bins_T.iterrows():
                    bin_rows.append(dict(system=sys.name, variant="temp_scaled", **r.to_dict()))
            except Exception as e:
                print(f"  {sys.name}: temp scaling skipped ({e})")

        # Reliability diagram
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        ax[0].plot(bins["mean_pred"], bins["mean_label"], "o-", label=f"Raw  (BS={bs:.3f}, ECE={ece:.3f})")
        if sys.val_csv and os.path.isfile(val_p_path):
            try:
                ax[0].plot(bins_T["mean_pred"], bins_T["mean_label"], "s-", color="C2",
                           label=f"Temp T={T:.2f} (BS={bs_T:.3f}, ECE={ece_T:.3f})")
            except Exception:
                pass
        ax[0].set_xlabel("Mean predicted probability (bin)")
        ax[0].set_ylabel("Empirical positive rate (bin)")
        ax[0].set_title(f"Reliability — {sys.name}")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].grid(alpha=0.3)

        ax[1].bar(bins["bin"], bins["n"], color="C0", alpha=0.7)
        ax[1].set_xlabel("Bin")
        ax[1].set_ylabel("Count")
        ax[1].set_title("Bin populations (raw)")
        ax[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f"reliability_{sys.name}.png")
        plt.savefig(out_png, dpi=120)
        plt.close()
        print(f"OK   {sys.name}: BS={bs:.4f} ECE={ece:.4f} MCE={mce:.4f} → {out_png}")

    pd.DataFrame(metric_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(bin_rows).to_csv(OUT_BINS, index=False)
    print(f"\nWrote {OUT_CSV} ({len(metric_rows)} rows)")
    print(f"Wrote {OUT_BINS} ({len(bin_rows)} rows)")


if __name__ == "__main__":
    main()
