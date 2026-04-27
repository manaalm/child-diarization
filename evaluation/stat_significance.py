#!/usr/bin/env python3
"""Bootstrap pairwise statistical significance tests across all diarizers."""
import json, sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIARIZER_PREDS = {
    "BabAR":       REPO / "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "Pyannote":    REPO / "pyannote/pyannote_enrollment_runs/test_predictions.csv",
    "USC-SAIL":    REPO / "whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv",
    "VTC":         REPO / "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "VTC-KCHI":    REPO / "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "VBx":         REPO / "vbx_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "TalkNet-ASD": REPO / "video_asd_ecapa_enrollment_runs/talknet_asd/enroll_test_predictions.csv",
    "EEND-EDA":    REPO / "eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "Sortformer":  REPO / "sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv",
    "WavLM-MIL":   REPO / "mil/mil_results/wavlm_mil/test_predictions.csv",
    "Whisper-MIL": REPO / "mil/mil_results/whisper_mil/test_predictions.csv",
}
SEG_MIL_PREDS = REPO / "mil/mil_results/seg_mil/babar_vtc_gated_attention/test_predictions.csv"
if SEG_MIL_PREDS.exists():
    DIARIZER_PREDS["SegMIL-BabAR-Gated"] = SEG_MIL_PREDS

def load_preds(path):
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if c.lower() in ("label", "y_true", "gt")), None)
    prob_col  = next((c for c in df.columns if c.lower() in ("prob", "score", "probability")), None)
    if not label_col or not prob_col:
        return None, None
    return df[label_col].astype(int).values, df[prob_col].astype(float).values

# Load all predictions — use clip_id alignment for paired bootstrap
# First pass: load all, get clip_ids if available
diar_data = {}
for name, path in DIARIZER_PREDS.items():
    if not path.exists():
        print(f"SKIP {name}")
        continue
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if c.lower() in ("label", "y_true", "gt")), None)
    prob_col  = next((c for c in df.columns if c.lower() in ("prob", "score", "probability")), None)
    id_col    = next((c for c in df.columns if "clip_id" in c.lower()), None)
    if not label_col or not prob_col:
        print(f"SKIP {name}: missing columns")
        continue
    diar_data[name] = {"y_true": df[label_col].astype(int).values,
                       "y_score": df[prob_col].astype(float).values,
                       "clip_id": df[id_col].values if id_col else np.arange(len(df))}
    print(f"Loaded {name}: n={len(df)}, id_col={id_col}")

def bootstrap_paired_auroc(y_true_a, y_score_a, y_true_b, y_score_b, n_boot=2000, seed=42):
    """Bootstrap test for AUROC_A > AUROC_B: return p-value (one-sided)."""
    rng = np.random.RandomState(seed)
    n = len(y_true_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_a, ys_a = y_true_a[idx], y_score_a[idx]
        yt_b, ys_b = y_true_b[idx], y_score_b[idx]
        if yt_a.sum() == 0 or yt_a.sum() == len(yt_a):
            continue
        try:
            diff = roc_auc_score(yt_a, ys_a) - roc_auc_score(yt_b, ys_b)
            diffs.append(diff)
        except Exception:
            pass
    if not diffs:
        return float("nan"), float("nan")
    diffs = np.array(diffs)
    observed = np.mean(diffs)
    # two-sided p: proportion of bootstrap samples on the other side of 0
    p_two = 2 * min((diffs >= 0).mean(), (diffs < 0).mean())
    return float(observed), float(p_two)

names = list(diar_data.keys())
rows = []
for a, b in combinations(names, 2):
    da, db = diar_data[a], diar_data[b]
    # Align on clip_id if possible
    if len(da["y_true"]) == len(db["y_true"]):
        y_true_a, y_score_a = da["y_true"], da["y_score"]
        y_true_b, y_score_b = db["y_true"], db["y_score"]
    else:
        # Use shorter; may not be paired perfectly
        n = min(len(da["y_true"]), len(db["y_true"]))
        y_true_a, y_score_a = da["y_true"][:n], da["y_score"][:n]
        y_true_b, y_score_b = db["y_true"][:n], db["y_score"][:n]
    
    auroc_a = roc_auc_score(y_true_a, y_score_a) if y_true_a.sum() > 0 and y_true_a.sum() < len(y_true_a) else float("nan")
    auroc_b = roc_auc_score(y_true_b, y_score_b) if y_true_b.sum() > 0 and y_true_b.sum() < len(y_true_b) else float("nan")
    
    diff, pval = bootstrap_paired_auroc(y_true_a, y_score_a, y_true_b, y_score_b)
    rows.append({"diarizer_a": a, "diarizer_b": b,
                 "auroc_a": round(auroc_a, 4), "auroc_b": round(auroc_b, 4),
                 "auroc_diff": round(diff, 4), "p_value_two_sided": round(pval, 4),
                 "significant_p05": pval < 0.05 if not np.isnan(pval) else False})
    print(f"  {a} vs {b}: AUROC diff={diff:.3f}, p={pval:.4f}")

df_out = pd.DataFrame(rows)
df_out = df_out.sort_values("p_value_two_sided")
out_path = OUT_DIR / "pairwise_significance_auroc.csv"
df_out.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")
print(f"Significant pairs (p<0.05): {df_out['significant_p05'].sum()} / {len(df_out)}")
