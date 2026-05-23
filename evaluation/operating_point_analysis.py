"""Operating-point and decision-curve analysis.

For each top system, given test_predictions.csv:
  - Recall at FPR ∈ {0.01, 0.05, 0.10}
  - Precision at recall ∈ {0.50, 0.75, 0.90, 0.95}
  - Specificity at recall ∈ {0.95, 0.99} (high-sensitivity regime)
  - "Review burden": fraction of clips you must review (top-k by score) to capture
      X% of true positives, for X ∈ {0.50, 0.75, 0.90, 0.95}
  - Decision-curve net benefit (Vickers 2006) over threshold grid [0.01, 0.99]

Outputs:
  evaluation/operating_points.csv          one row per (system, metric)
  evaluation/decision_curves.csv           per-(system, threshold) net benefit
  evaluation/figures/decision_curve.png    single overlay plot
  evaluation/figures/review_burden.png     single overlay plot
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
FIG_DIR = os.path.join(REPO, "evaluation", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class System:
    name: str
    test_csv: str
    score_col: str


TOP_SYSTEMS = [
    System("whisper_mil",              "mil/mil_results/whisper_mil/test_predictions.csv",                      "score"),
    System("whisper_mil_tsmil_concat", "mil/mil_results/whisper_mil_tsmil_concat/test_predictions.csv",         "score"),
    System("wavlm_mil",                "mil/mil_results/wavlm_mil/test_predictions.csv",                        "score"),
    System("babar_enrollment",         "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",               "prob"),
    System("vtc_enrollment",           "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv",                 "prob"),
    System("vtc_kchi_enrollment",      "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv",            "prob"),
    System("best_audio_mil_ensemble",  "ensemble_runs/test_predictions.csv",                                    "best_audio_mil_mean"),
    System("metadata_stacker",         "ensemble_runs/metadata_stack/test_predictions.csv",                     "score"),
    System("wavlm_pseudo_frame",       "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv",          "score"),
    System("audio_llm_qwen25_omni",    "baselines/audio_llm_baseline_runs/qwen25_omni_7b/test_predictions.csv", "prob"),
]


def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    # interp tpr at requested fpr
    return float(np.interp(fpr_target, fpr, tpr))


def precision_at_recall(y, p, recall_target):
    precisions, recalls, _ = precision_recall_curve(y, p)
    # PR curve is reversed (recall descending in sklearn output for some versions)
    order = np.argsort(recalls)
    return float(np.interp(recall_target, recalls[order], precisions[order]))


def review_burden_at_recall(y, p, recall_target):
    """Sort clips by score desc, take top-k. Find min k/N such that recall ≥ target."""
    order = np.argsort(-p)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    if total_pos == 0:
        return float("nan")
    needed = int(np.ceil(recall_target * total_pos))
    idx = np.searchsorted(cum_pos, needed, side="left")
    return float((idx + 1) / len(y))


def net_benefit(y, p, threshold, prevalence):
    """Vickers 2006 net benefit at a single threshold.

    NB = TP/N - (FP/N) * (pt / (1 - pt))
    where pt is the threshold being treated as a 'cost ratio'.
    """
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n = len(y)
    if threshold >= 1.0:
        return 0.0
    pt = threshold
    return tp / n - (fp / n) * (pt / max(1e-9, 1 - pt))


def main():
    op_rows = []
    dc_rows = []

    fig_dc, ax_dc = plt.subplots(figsize=(8, 5))
    fig_rb, ax_rb = plt.subplots(figsize=(8, 5))

    fpr_targets = [0.01, 0.05, 0.10]
    recall_targets = [0.50, 0.75, 0.90, 0.95]

    for i, sys in enumerate(TOP_SYSTEMS):
        path = os.path.join(REPO, sys.test_csv)
        if not os.path.isfile(path):
            print(f"SKIP {sys.name}: missing {path}")
            continue
        df = pd.read_csv(path)
        if sys.score_col not in df.columns:
            print(f"SKIP {sys.name}: column '{sys.score_col}' not in {list(df.columns)[:6]}...")
            continue
        y = df["label"].astype(int).to_numpy()
        p = df[sys.score_col].astype(float).clip(0.0, 1.0).to_numpy()
        prev = float(y.mean())

        # Op points
        for ft in fpr_targets:
            op_rows.append(dict(system=sys.name, kind="recall_at_fpr", target=ft, value=recall_at_fpr(y, p, ft)))
        for rt in recall_targets:
            op_rows.append(dict(system=sys.name, kind="precision_at_recall", target=rt, value=precision_at_recall(y, p, rt)))
            op_rows.append(dict(system=sys.name, kind="review_burden_at_recall", target=rt, value=review_burden_at_recall(y, p, rt)))

        # Decision curve
        thr_grid = np.linspace(0.01, 0.99, 99)
        nbs = [net_benefit(y, p, t, prev) for t in thr_grid]
        for t, nb in zip(thr_grid, nbs):
            dc_rows.append(dict(system=sys.name, threshold=float(t), net_benefit=float(nb)))

        ax_dc.plot(thr_grid, nbs, label=sys.name)

        # Review-burden curve
        recall_grid = np.linspace(0.05, 1.0, 50)
        burdens = [review_burden_at_recall(y, p, r) for r in recall_grid]
        ax_rb.plot(recall_grid, burdens, label=sys.name)

        print(f"OK   {sys.name}: prev={prev:.3f}, R@FPR0.05={recall_at_fpr(y, p, 0.05):.3f}, "
              f"P@R0.95={precision_at_recall(y, p, 0.95):.3f}, "
              f"burden@R0.95={review_burden_at_recall(y, p, 0.95):.3f}")

    # Treat-all reference for DC
    if dc_rows:
        prev = float(pd.read_csv(os.path.join(REPO, TOP_SYSTEMS[0].test_csv))["label"].mean())
        thr_grid = np.linspace(0.01, 0.99, 99)
        treat_all = [prev - (1 - prev) * (t / max(1e-9, 1 - t)) for t in thr_grid]
        ax_dc.plot(thr_grid, treat_all, "--", color="black", alpha=0.6, label="treat-all")
        ax_dc.axhline(0, color="gray", linestyle=":", alpha=0.7, label="treat-none")

    ax_dc.set_xlabel("Threshold (= cost ratio)")
    ax_dc.set_ylabel("Net benefit")
    ax_dc.set_title("Decision curves (Vickers 2006)")
    ax_dc.legend(loc="lower left", fontsize=7)
    ax_dc.grid(alpha=0.3)
    ax_dc.set_ylim(-0.1, 0.8)
    fig_dc.tight_layout()
    fig_dc.savefig(os.path.join(FIG_DIR, "decision_curve.png"), dpi=120)
    plt.close(fig_dc)

    ax_rb.set_xlabel("Recall target")
    ax_rb.set_ylabel("Fraction of clips that must be reviewed")
    ax_rb.set_title("Review burden vs recall")
    ax_rb.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="random review")
    ax_rb.legend(loc="upper left", fontsize=7)
    ax_rb.grid(alpha=0.3)
    ax_rb.set_xlim(0, 1)
    ax_rb.set_ylim(0, 1)
    fig_rb.tight_layout()
    fig_rb.savefig(os.path.join(FIG_DIR, "review_burden.png"), dpi=120)
    plt.close(fig_rb)

    out_op = os.path.join(REPO, "evaluation", "operating_points.csv")
    out_dc = os.path.join(REPO, "evaluation", "decision_curves.csv")
    pd.DataFrame(op_rows).to_csv(out_op, index=False)
    pd.DataFrame(dc_rows).to_csv(out_dc, index=False)
    print(f"\nWrote {out_op}  ({len(op_rows)} rows)")
    print(f"Wrote {out_dc}  ({len(dc_rows)} rows)")
    print(f"Wrote {os.path.join(FIG_DIR, 'decision_curve.png')}")
    print(f"Wrote {os.path.join(FIG_DIR, 'review_burden.png')}")


if __name__ == "__main__":
    main()
