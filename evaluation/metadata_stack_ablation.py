"""Metadata stacker ablation: read LR coefficients + per-stratum metrics on test.

Diagnostic comparison: does the metadata stacker (F1=0.901, AUROC=0.900) gain
uniformly over the best_audio_mil baseline (F1=0.893, AUROC=0.878), or is the
gain concentrated in clips where metadata flags multi-child / unclear-target /
specific-age conditions?

Strata: n_children (1 vs >=2), Child_of_interest_clear (yes/no/other),
timepoint_norm (14_month, 36_month).

Outputs:
  - ensemble_runs/metadata_stack/ablation/lr_coefficients.csv
  - ensemble_runs/metadata_stack/ablation/stratified_metrics.csv
  - ensemble_runs/metadata_stack/ablation/score_correlation.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "ensemble_runs/metadata_stack/ablation")
os.makedirs(OUT, exist_ok=True)

STACK_DIR = os.path.join(_REPO, "ensemble_runs/metadata_stack")
ENS_TEST = os.path.join(_REPO, "ensemble_runs/test_predictions.csv")
MASTER = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
BASELINE_THR = 0.43   # best_audio_mil_mean threshold from ensemble_results.json
STACK_THR = 0.575     # from metadata_stack/config.json


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


# ── 1) LR coefficients ──────────────────────────────────────────────────────
fi = json.load(open(os.path.join(STACK_DIR, "feature_importances.json")))
cfg = json.load(open(os.path.join(STACK_DIR, "config.json")))
score_feats = set(cfg["score_features"])
meta_feats = set(cfg["meta_features"])
rows = []
for name, w in fi["lr_coefficients"].items():
    kind = "score" if name in score_feats else ("meta" if name in meta_feats else "other")
    rows.append({"feature": name, "kind": kind, "coef": w, "abs_coef": abs(w)})
coef_df = pd.DataFrame(rows).sort_values(["kind", "abs_coef"], ascending=[True, False])
coef_df.to_csv(os.path.join(OUT, "lr_coefficients.csv"), index=False)

print("\n=== LR COEFFICIENTS (sorted by |coef| within kind) ===")
for kind in ["score", "meta"]:
    sub = coef_df[coef_df["kind"] == kind]
    print(f"\n[{kind} features]")
    for _, r in sub.iterrows():
        print(f"  {r['feature']:<22s} {r['coef']:+.4f}")

print("\nMeta L1-norm sum:", coef_df.loc[coef_df.kind == "meta", "abs_coef"].sum())
print("Score L1-norm sum:", coef_df.loc[coef_df.kind == "score", "abs_coef"].sum())

# ── 2) Load stacker preds, baseline score, metadata strata ──────────────────
stack = pd.read_csv(os.path.join(STACK_DIR, "test_predictions.csv"))
ens = pd.read_csv(ENS_TEST)[["audio_path", "best_audio_mil_mean", "label", "timepoint_norm"]]
master = pd.read_csv(MASTER)
master_test = master[master["split"] == "test"].copy()


def _to_int(val, default):
    try:
        return int(str(val).strip().split("+")[0])
    except Exception:
        return default


master_test["n_children_int"] = master_test["#_children"].apply(lambda v: _to_int(v, 1))
master_test["n_adults_int"] = master_test["#_adults"].apply(lambda v: _to_int(v, 0))
master_test["coi_clear_norm"] = master_test["Child_of_interest_clear"].astype(str).str.strip().str.lower()


def coi_bucket(v):
    if v in {"yes"}:
        return "yes"
    if v in {"no"}:
        return "no"
    if v in {"unclear", "partial", "partially", "sometimes"}:
        return "unclear"
    return "other_or_missing"


master_test["coi_bucket"] = master_test["coi_clear_norm"].apply(coi_bucket)

meta_keep = ["audio_path", "n_children_int", "n_adults_int", "coi_bucket"]
df = stack.merge(ens.rename(columns={"best_audio_mil_mean": "baseline_score"}),
                 on=["audio_path", "label", "timepoint_norm"], how="inner") \
          .merge(master_test[meta_keep], on="audio_path", how="left")

assert len(df) == len(stack), f"Merge dropped rows: stack={len(stack)} merged={len(df)}"

df["n_children_bucket"] = np.where(df["n_children_int"] >= 2, "ge2", "1")
df["n_adults_bucket"] = np.where(df["n_adults_int"] >= 2, "ge2", df["n_adults_int"].astype(str))

# Score columns
df = df.rename(columns={"score": "stack_score"})

# ── 3) Score correlation (collinearity check) ──────────────────────────────
corr_pearson = float(df["stack_score"].corr(df["baseline_score"]))
corr_spearman = float(df["stack_score"].corr(df["baseline_score"], method="spearman"))
print(f"\n=== SCORE CORRELATION (stack vs baseline) ===")
print(f"  Pearson:  {corr_pearson:.4f}")
print(f"  Spearman: {corr_spearman:.4f}")
json.dump({"pearson": corr_pearson, "spearman": corr_spearman, "n": len(df)},
          open(os.path.join(OUT, "score_correlation.json"), "w"), indent=2)

# ── 4) Stratified metrics ──────────────────────────────────────────────────
strata_specs = [
    ("overall", lambda d: pd.Series([True] * len(d))),
    ("n_children=1", lambda d: d["n_children_bucket"] == "1"),
    ("n_children>=2", lambda d: d["n_children_bucket"] == "ge2"),
    ("coi=yes", lambda d: d["coi_bucket"] == "yes"),
    ("coi=no", lambda d: d["coi_bucket"] == "no"),
    ("coi=unclear", lambda d: d["coi_bucket"] == "unclear"),
    ("coi=other_or_missing", lambda d: d["coi_bucket"] == "other_or_missing"),
    ("timepoint=14_month", lambda d: d["timepoint_norm"] == "14_month"),
    ("timepoint=36_month", lambda d: d["timepoint_norm"] == "36_month"),
    ("n_adults>=2", lambda d: d["n_adults_bucket"] == "ge2"),
    ("n_adults=1", lambda d: d["n_adults_bucket"] == "1"),
    ("n_adults=0", lambda d: d["n_adults_bucket"] == "0"),
]

rows = []
for name, sel in strata_specs:
    mask = sel(df)
    sub = df[mask]
    if len(sub) == 0:
        continue
    stack_m = metrics_block(sub["label"], sub["stack_score"], STACK_THR)
    base_m = metrics_block(sub["label"], sub["baseline_score"], BASELINE_THR)
    row = {"stratum": name, "n": stack_m["n"], "n_pos": stack_m["n_pos"]}
    for k in ["f1", "precision", "recall", "auroc", "auprc"]:
        row[f"stack_{k}"] = round(stack_m[k], 4)
        row[f"base_{k}"] = round(base_m[k], 4)
        row[f"delta_{k}"] = round(stack_m[k] - base_m[k], 4)
    rows.append(row)

strat = pd.DataFrame(rows)
strat.to_csv(os.path.join(OUT, "stratified_metrics.csv"), index=False)

print("\n=== STRATIFIED METRICS (stack threshold=0.575, baseline threshold=0.43) ===")
print(strat[["stratum", "n", "n_pos", "stack_f1", "base_f1", "delta_f1",
             "stack_auroc", "base_auroc", "delta_auroc"]].to_string(index=False))

# ── 5) Distribution of strata ─────────────────────────────────────────────
print("\n=== STRATUM SIZES ===")
print("n_children:", df["n_children_bucket"].value_counts().to_dict())
print("coi_bucket:", df["coi_bucket"].value_counts().to_dict())
print("n_adults:  ", df["n_adults_bucket"].value_counts().to_dict())
print("timepoint: ", df["timepoint_norm"].value_counts().to_dict())

print(f"\nWrote: {OUT}/")
