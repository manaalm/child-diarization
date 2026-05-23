"""DeLong's analytic test for paired AUROC differences + effect-size summary.

For each ordered pair (A, B) of headline systems:
  - DeLong z-statistic and two-sided p-value for AUROC_A vs AUROC_B (paired)
  - AUROC delta with 95% paired-bootstrap CI
  - Cohen's d on the per-clip score difference (A's score - B's score, restricted
    to clips where label = 1 vs label = 0)

Inputs: test_predictions.csv files of the headline systems.
Outputs:
  evaluation/delong_pairwise.csv
  evaluation/effect_sizes.csv

DeLong implementation follows Sun & Xu 2014 (linear-time variant).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
OUT_DELONG = os.path.join(REPO, "evaluation", "delong_pairwise.csv")
OUT_EFFECT = os.path.join(REPO, "evaluation", "effect_sizes.csv")


@dataclass
class System:
    name: str
    csv: str
    score_col: str


SYSTEMS = [
    System("whisper_mil",              "mil/mil_results/whisper_mil/test_predictions.csv",                      "score"),
    System("whisper_mil_tsmil_concat", "mil/mil_results/whisper_mil_tsmil_concat/test_predictions.csv",         "score"),
    System("wavlm_mil",                "mil/mil_results/wavlm_mil/test_predictions.csv",                        "score"),
    System("babar_enrollment",         "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",               "prob"),
    System("vtc_enrollment",           "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv",                 "prob"),
    System("vtc_kchi_enrollment",      "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv",            "prob"),
    System("vbx_enrollment",           "vbx_ecapa_enrollment_runs/enroll_test_predictions.csv",                 "prob"),
    System("pyannote_enrollment",      "pyannote/pyannote_enrollment_runs/test_predictions.csv",                "prob"),
    System("usc_sail_enrollment",      "whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv", "prob"),
    System("sortformer_enrollment",    "sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv",          "prob"),
    System("eend_eda_enrollment",      "eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv",            "prob"),
    System("best_audio_mil_ensemble",  "ensemble_runs/test_predictions.csv",                                    "best_audio_mil_mean"),
    System("metadata_stacker",         "ensemble_runs/metadata_stack/test_predictions.csv",                     "score"),
    System("wavlm_pseudo_frame",       "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv",          "score"),
    System("audio_llm_qwen25_omni",    "baselines/audio_llm_baseline_runs/qwen25_omni_7b/test_predictions.csv", "prob"),
]


def fast_delong(scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray):
    """Paired DeLong covariance (Sun & Xu 2014).

    Returns auroc_a, auroc_b, var_a, var_b, cov_ab, z, two_sided_p
    """
    pos = labels == 1
    neg = labels == 0
    m = int(pos.sum())
    n = int(neg.sum())

    # Midrank computation per Sun & Xu
    def compute_midrank(x):
        order = np.argsort(x, kind="stable")
        x_sorted = x[order]
        T = np.zeros_like(x, dtype=float)
        i = 0
        N = len(x)
        while i < N:
            j = i
            while j < N and x_sorted[j] == x_sorted[i]:
                j += 1
                # midrank of tied group
            T[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return T

    aucs, vs = [], []
    XY = []
    for s in (scores_a, scores_b):
        x = s[pos]   # positive scores
        y = s[neg]   # negative scores
        # full midrank on combined, plus midrank within pos and neg
        TX = compute_midrank(x)
        TY = compute_midrank(y)
        TZ = compute_midrank(np.concatenate([x, y]))
        TZx = TZ[:m]
        TZy = TZ[m:]
        auc = (TZx.sum() / m - (m + 1) / 2.0) / n
        aucs.append(auc)
        v01 = (TZx - TX) / n
        v10 = 1 - (TZy - TY) / m
        XY.append((v01, v10))

    auc_a, auc_b = aucs
    v01_a, v10_a = XY[0]
    v01_b, v10_b = XY[1]
    sx = np.cov(np.vstack([v01_a, v01_b]), bias=False)  # 2x2 over m positives
    sy = np.cov(np.vstack([v10_a, v10_b]), bias=False)  # 2x2 over n negatives
    var_a = sx[0, 0] / m + sy[0, 0] / n
    var_b = sx[1, 1] / m + sy[1, 1] / n
    cov_ab = sx[0, 1] / m + sy[0, 1] / n
    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return auc_a, auc_b, var_a, var_b, cov_ab, np.nan, np.nan
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2.0 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, var_a, var_b, cov_ab, float(z), float(p)


def paired_bootstrap_auroc(scores_a, scores_b, labels, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            a_auc = roc_auc_score(labels[idx], scores_a[idx])
            b_auc = roc_auc_score(labels[idx], scores_b[idx])
            deltas[b] = a_auc - b_auc
        except Exception:
            deltas[b] = np.nan
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5)), float(np.nanmean(deltas))


def cohens_d_score_diff(scores_a, scores_b, labels):
    """Effect size on per-clip score *advantage* of A over B, separated by class.

    For positives: how much higher does A score them than B does?
    For negatives: how much lower does A score them than B?
    Reported as Cohen's d of (A - B) within positives and within negatives.
    Combined effect uses pooled std.
    """
    diff = scores_a - scores_b
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return float("nan"), float("nan"), float("nan")
    d_pos = float(diff[pos].mean()) / float(diff[pos].std() + 1e-9)
    d_neg = -float(diff[neg].mean()) / float(diff[neg].std() + 1e-9)
    pooled_std = np.sqrt((diff[pos].std()**2 + diff[neg].std()**2) / 2)
    d_combined = (float(diff[pos].mean()) - float(diff[neg].mean())) / (pooled_std + 1e-9)
    return d_pos, d_neg, d_combined


def main():
    # Load all predictions, align by audio_path
    loaded = []
    for s in SYSTEMS:
        path = os.path.join(REPO, s.csv)
        if not os.path.isfile(path):
            print(f"SKIP {s.name}: missing {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"SKIP {s.name}: {e}")
            continue
        if s.score_col not in df.columns:
            print(f"SKIP {s.name}: column {s.score_col} not found")
            continue
        if "audio_path" not in df.columns:
            print(f"SKIP {s.name}: no audio_path column")
            continue
        keep = df[["audio_path", "label", s.score_col]].rename(columns={s.score_col: "score"})
        keep["score"] = keep["score"].astype(float).clip(0.0, 1.0)
        keep["label"] = keep["label"].astype(int)
        loaded.append((s.name, keep))
        print(f"OK   load {s.name} n={len(keep)}")

    # Align all on shared audio_path
    if not loaded:
        return
    common = set(loaded[0][1]["audio_path"])
    for _, df in loaded[1:]:
        common &= set(df["audio_path"])
    print(f"Aligned on {len(common)} clips")

    aligned = {}
    for name, df in loaded:
        sub = df[df["audio_path"].isin(common)].drop_duplicates("audio_path").set_index("audio_path").sort_index()
        aligned[name] = sub
    # Sanity: shared label vector
    label_ref = next(iter(aligned.values()))["label"].to_numpy()
    for name, df in aligned.items():
        if not (df["label"].to_numpy() == label_ref).all():
            print(f"WARN  label mismatch for {name}; using its own labels per-pair")

    delong_rows = []
    effect_rows = []
    names = list(aligned.keys())
    for a, b in combinations(names, 2):
        sa = aligned[a]["score"].to_numpy()
        sb = aligned[b]["score"].to_numpy()
        y = label_ref
        try:
            auc_a, auc_b, va, vb, cov, z, p = fast_delong(sa, sb, y)
        except Exception as e:
            print(f"DeLong fail {a} vs {b}: {e}")
            continue
        ci_lo, ci_hi, mean_delta = paired_bootstrap_auroc(sa, sb, y, n_boot=500)
        d_pos, d_neg, d_comb = cohens_d_score_diff(sa, sb, y)

        delong_rows.append(dict(
            sys_a=a, sys_b=b, auroc_a=round(auc_a, 4), auroc_b=round(auc_b, 4),
            auroc_diff=round(auc_a - auc_b, 4),
            delong_z=round(z, 3) if z == z else None,
            delong_p=round(p, 5) if p == p else None,
            paired_boot_lo=round(ci_lo, 4), paired_boot_hi=round(ci_hi, 4),
            paired_boot_mean=round(mean_delta, 4),
            significant_p05=bool(p < 0.05) if p == p else None,
        ))
        effect_rows.append(dict(
            sys_a=a, sys_b=b,
            auroc_diff=round(auc_a - auc_b, 4),
            cohens_d_pos=round(d_pos, 3),
            cohens_d_neg=round(d_neg, 3),
            cohens_d_combined=round(d_comb, 3),
        ))

    pd.DataFrame(delong_rows).to_csv(OUT_DELONG, index=False)
    pd.DataFrame(effect_rows).to_csv(OUT_EFFECT, index=False)
    print(f"\nWrote {OUT_DELONG}  ({len(delong_rows)} pairs)")
    print(f"Wrote {OUT_EFFECT}  ({len(effect_rows)} pairs)")
    print("\nTop significant pairs by |delta|:")
    df_d = pd.DataFrame(delong_rows)
    if not df_d.empty:
        df_d["abs_delta"] = df_d["auroc_diff"].abs()
        sig = df_d[df_d["significant_p05"] == True].sort_values("abs_delta", ascending=False)
        print(sig.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
