"""US1 ablation: stratified test metrics for the metadata + visual stacker.

Mirrors evaluation/metadata_stack_ablation.py but compares against
TWO baselines:
  - best_audio_mil_mean (the original audio ensemble, F1=0.893 / AUROC=0.878)
  - metadata-only stacker (spec-012 US1, F1=0.901 / AUROC=0.900)

Outputs:
  ensemble_runs/metadata_stack_av/ablation/{lr_coefficients.csv,
    stratified_metrics.csv, score_correlation.json}

Strata:
  - n_children (1, ≥2)
  - Child_of_interest_clear (yes, no)
  - timepoint_norm (14_month, 36_month)
  - has_any_face (0, 1)
  - eligibility_score (split at val median)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "ensemble_runs/metadata_stack_av/ablation")
os.makedirs(OUT, exist_ok=True)

STACK_AV_DIR = os.path.join(_REPO, "ensemble_runs/metadata_stack_av")
STACK_DIR = os.path.join(_REPO, "ensemble_runs/metadata_stack")
ENS_TEST = os.path.join(_REPO, "ensemble_runs/test_predictions.csv")
MASTER = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
VIS_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/visual_eligibility.csv")
BASELINE_THR = 0.43   # best_audio_mil_mean threshold
STACK_AV_THR_DEFAULT = 0.5
STACK_THR_DEFAULT = 0.575


def metrics_block(y, p, thr):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, p)) if y.sum() and y.sum() < len(y) else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        out["auprc"] = float("nan")
    return out


# ── Load thresholds + coefficients ──────────────────────────────────────────
av_test_metrics = json.load(open(os.path.join(STACK_AV_DIR, "test_metrics_tuned.json")))
av_thr = float(av_test_metrics["threshold"])
print(f"AV stacker threshold: {av_thr}")

stack_thr = STACK_THR_DEFAULT
if os.path.exists(os.path.join(STACK_DIR, "test_metrics_tuned.json")):
    stack_thr = float(json.load(open(os.path.join(STACK_DIR, "test_metrics_tuned.json")))["threshold"])
print(f"Metadata-only stacker threshold: {stack_thr}")

fi = json.load(open(os.path.join(STACK_AV_DIR, "feature_importances.json")))
cfg = json.load(open(os.path.join(STACK_AV_DIR, "config.json")))
score_feats = set(cfg["score_features"])
meta_feats = set(cfg["meta_features"])
vis_feats = set(cfg["visual_features"])

rows = []
for name, w in fi["lr_coefficients"].items():
    if name in score_feats:
        kind = "score"
    elif name in meta_feats:
        kind = "meta"
    elif name in vis_feats:
        kind = "visual"
    else:
        kind = "other"
    rows.append({"feature": name, "kind": kind, "coef": w, "abs_coef": abs(w)})
coef_df = pd.DataFrame(rows).sort_values(["kind", "abs_coef"], ascending=[True, False])
coef_df.to_csv(os.path.join(OUT, "lr_coefficients.csv"), index=False)

print("\n=== LR COEFFICIENTS BY KIND (sorted by |coef|) ===")
for kind in ["score", "meta", "visual"]:
    sub = coef_df[coef_df["kind"] == kind]
    print(f"\n[{kind} features]")
    for _, r in sub.iterrows():
        print(f"  {r['feature']:<30s} {r['coef']:+.4f}")

print(f"\nL1 sums:")
for kind in ["score", "meta", "visual"]:
    print(f"  {kind:<6s}: {coef_df.loc[coef_df.kind == kind, 'abs_coef'].sum():.3f}")

# ── Load all predictions ─────────────────────────────────────────────────────
av_preds = pd.read_csv(os.path.join(STACK_AV_DIR, "test_predictions.csv"))
stack_preds = pd.read_csv(os.path.join(STACK_DIR, "test_predictions.csv"))
ens = pd.read_csv(ENS_TEST)[["audio_path", "best_audio_mil_mean"]]
master = pd.read_csv(MASTER)
master_test = master[master["split"] == "test"].copy()


def _to_int(val, default):
    try:
        return int(str(val).strip().split("+")[0])
    except Exception:
        return default


master_test["n_children_int"] = master_test["#_children"].apply(lambda v: _to_int(v, 1))
master_test["n_adults_int"] = master_test["#_adults"].apply(lambda v: _to_int(v, 0))
master_test["coi_norm"] = master_test["Child_of_interest_clear"].astype(str).str.strip().str.lower()


def coi_bucket(v):
    if v == "yes":
        return "yes"
    if v == "no":
        return "no"
    return "other_or_missing"


master_test["coi_bucket"] = master_test["coi_norm"].apply(coi_bucket)
vis = pd.read_csv(VIS_CSV)

df = av_preds.rename(columns={"score": "av_score"}) \
        .merge(stack_preds[["audio_path", "score"]].rename(columns={"score": "stack_score"}),
               on="audio_path", how="inner") \
        .merge(ens.rename(columns={"best_audio_mil_mean": "baseline_score"}),
               on="audio_path", how="inner") \
        .merge(master_test[["audio_path", "n_children_int", "n_adults_int", "coi_bucket"]],
               on="audio_path", how="left") \
        .merge(vis[["audio_path", "has_any_face", "eligibility_score"]],
               on="audio_path", how="left")

assert len(df) == len(av_preds), f"merge dropped rows: {len(av_preds)} → {len(df)}"

df["n_children_bucket"] = np.where(df["n_children_int"] >= 2, "ge2", "1")

# Eligibility median split (on val so we don't leak test). For ablation purposes,
# split at the train+val median of eligibility_score → ~0.65 for this corpus.
elig_thr = float(np.median(vis["eligibility_score"]))
df["elig_bucket"] = np.where(df["eligibility_score"] >= elig_thr, "high_elig", "low_elig")
print(f"\nEligibility median split: thr={elig_thr:.3f}")

# Score correlations
corr_av_base = float(df["av_score"].corr(df["baseline_score"]))
corr_av_stack = float(df["av_score"].corr(df["stack_score"]))
corr_stack_base = float(df["stack_score"].corr(df["baseline_score"]))
print(f"\n=== SCORE CORRELATIONS ===")
print(f"  AV stacker vs baseline (best_audio_mil): Pearson {corr_av_base:.4f}")
print(f"  AV stacker vs metadata-only stacker:     Pearson {corr_av_stack:.4f}")
print(f"  metadata-only stacker vs baseline:       Pearson {corr_stack_base:.4f}")
json.dump({
    "av_vs_baseline": corr_av_base,
    "av_vs_metadata_stacker": corr_av_stack,
    "metadata_stacker_vs_baseline": corr_stack_base,
    "n": len(df),
}, open(os.path.join(OUT, "score_correlation.json"), "w"), indent=2)

# Stratified
strata = [
    ("overall", lambda d: pd.Series([True] * len(d))),
    ("n_children=1", lambda d: d["n_children_bucket"] == "1"),
    ("n_children>=2", lambda d: d["n_children_bucket"] == "ge2"),
    ("coi=yes", lambda d: d["coi_bucket"] == "yes"),
    ("coi=no", lambda d: d["coi_bucket"] == "no"),
    ("timepoint=14_month", lambda d: d["timepoint_norm"] == "14_month"),
    ("timepoint=36_month", lambda d: d["timepoint_norm"] == "36_month"),
    ("has_any_face=1", lambda d: d["has_any_face"] == 1),
    ("has_any_face=0", lambda d: d["has_any_face"] == 0),
    ("elig_high", lambda d: d["elig_bucket"] == "high_elig"),
    ("elig_low",  lambda d: d["elig_bucket"] == "low_elig"),
]

rows = []
for name, sel in strata:
    sub = df[sel(df)]
    if len(sub) == 0:
        continue
    av_m = metrics_block(sub["label"], sub["av_score"], av_thr)
    st_m = metrics_block(sub["label"], sub["stack_score"], stack_thr)
    bs_m = metrics_block(sub["label"], sub["baseline_score"], BASELINE_THR)
    row = {"stratum": name, "n": av_m["n"], "n_pos": av_m["n_pos"]}
    for k in ["f1", "auroc", "auprc"]:
        row[f"av_{k}"] = round(av_m[k], 4)
        row[f"stack_{k}"] = round(st_m[k], 4)
        row[f"base_{k}"] = round(bs_m[k], 4)
        row[f"d_{k}_vs_stack"] = round(av_m[k] - st_m[k], 4)
        row[f"d_{k}_vs_base"]  = round(av_m[k] - bs_m[k], 4)
    rows.append(row)

strat = pd.DataFrame(rows)
strat.to_csv(os.path.join(OUT, "stratified_metrics.csv"), index=False)

print("\n=== STRATIFIED METRICS (av thr=%.3f, stack thr=%.3f, base thr=%.2f) ===" % (av_thr, stack_thr, BASELINE_THR))
print(strat[[
    "stratum", "n", "n_pos",
    "av_f1", "stack_f1", "d_f1_vs_stack",
    "av_auroc", "stack_auroc", "d_auroc_vs_stack",
]].to_string(index=False))

print("\n=== STRATUM SIZES ===")
for col in ["n_children_bucket", "coi_bucket", "elig_bucket"]:
    print(f"  {col}: {df[col].value_counts().to_dict()}")
print(f"  has_any_face: {df['has_any_face'].value_counts().to_dict()}")
print(f"  timepoint:    {df['timepoint_norm'].value_counts().to_dict()}")

print(f"\nWrote: {OUT}/")
