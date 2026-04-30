"""Per-branch attention diagnostics for ACMIL runs (spec-014 US3).

Reads branch_attention_{split}.csv produced by mil_evaluate.py (one row per
(clip, instance), one column per branch) and computes per-branch summary
statistics + branch-pair attention overlap. The key question this answers:
"did MBA actually produce diverse branches, or did they collapse to the same
attention pattern?"

Outputs (per ACMIL run dir):
  branch_diagnostics_{split}.json:
    mean_attn:           per-branch mean attention across all (clip, instance) rows
    std_attn:            per-branch std
    pairwise_cosine:     pairwise cosine similarity between per-branch attention
                          vectors (concatenated across clips). Healthy MBA → low.
    pairwise_overlap:    pairwise top-K overlap (Jaccard at K=10% per clip).

Usage:
    python mil/eval_acmil_branches.py \\
        --results-dir mil/mil_results/wavlm_mil_acmil \\
        --split test
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_branch_attention(results_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(results_dir, f"branch_attention_{split}.csv")
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found. Did mil_evaluate.py run on an ACMIL config?",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def _branch_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("branch_") and c.endswith("_weight")])


def _pairwise_cosine(branch_arrays: List[np.ndarray]) -> np.ndarray:
    """Pairwise cosine similarity matrix over branches."""
    n = len(branch_arrays)
    out = np.zeros((n, n), dtype=np.float64)
    norms = [np.linalg.norm(a) + 1e-12 for a in branch_arrays]
    for i in range(n):
        for j in range(n):
            out[i, j] = float(np.dot(branch_arrays[i], branch_arrays[j]) / (norms[i] * norms[j]))
    return out


def _pairwise_topk_overlap(df: pd.DataFrame, branch_cols: List[str], k_frac: float = 0.1) -> np.ndarray:
    """For each clip, compute pairwise Jaccard overlap of top-K instances per branch."""
    n_branches = len(branch_cols)
    pair_overlaps = np.zeros((n_branches, n_branches), dtype=np.float64)
    pair_counts = np.zeros((n_branches, n_branches), dtype=np.float64)
    for _path, sub in df.groupby("audio_path"):
        n_inst = len(sub)
        k = max(1, int(k_frac * n_inst))
        topk_per_branch = []
        for col in branch_cols:
            order = np.argsort(-sub[col].values)
            topk_per_branch.append(set(order[:k].tolist()))
        for i in range(n_branches):
            for j in range(n_branches):
                inter = len(topk_per_branch[i] & topk_per_branch[j])
                union = len(topk_per_branch[i] | topk_per_branch[j])
                pair_overlaps[i, j] += inter / max(union, 1)
                pair_counts[i, j] += 1
    return pair_overlaps / np.maximum(pair_counts, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACMIL per-branch diagnostics")
    parser.add_argument("--results-dir", required=True, help="ACMIL run directory")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--k-frac", type=float, default=0.1,
                        help="Top-K fraction for branch-pair overlap (default 0.1)")
    args = parser.parse_args()

    results_dir = args.results_dir if os.path.isabs(args.results_dir) else os.path.join(
        _REPO, args.results_dir
    )
    df = _load_branch_attention(results_dir, args.split)
    branch_cols = _branch_columns(df)
    if len(branch_cols) < 2:
        print("Only one branch found — nothing to compare.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} (clip, instance) rows  |  {len(branch_cols)} branches", flush=True)

    # Per-branch global statistics
    summary = {}
    for col in branch_cols:
        summary[col] = {
            "mean_attn": float(df[col].mean()),
            "std_attn": float(df[col].std()),
            "min_attn": float(df[col].min()),
            "max_attn": float(df[col].max()),
        }

    # Pairwise cosine over the full attention vectors (rows = (clip, instance) pairs)
    branch_arrays = [df[col].values.astype(np.float64) for col in branch_cols]
    cos_mat = _pairwise_cosine(branch_arrays)
    overlap_mat = _pairwise_topk_overlap(df, branch_cols, k_frac=args.k_frac)

    output = {
        "n_rows": int(len(df)),
        "n_branches": len(branch_cols),
        "k_frac": args.k_frac,
        "per_branch": summary,
        "pairwise_cosine": {
            f"{i}_{j}": float(cos_mat[i, j])
            for i in range(len(branch_cols)) for j in range(len(branch_cols))
        },
        "pairwise_topk_overlap": {
            f"{i}_{j}": float(overlap_mat[i, j])
            for i in range(len(branch_cols)) for j in range(len(branch_cols))
        },
    }
    # Off-diagonal mean (branch-collapse indicator: ≥ ~0.95 = collapsed)
    off = ~np.eye(len(branch_cols), dtype=bool)
    output["mean_off_diag_cosine"] = float(cos_mat[off].mean())
    output["mean_off_diag_overlap"] = float(overlap_mat[off].mean())

    out_path = os.path.join(results_dir, f"branch_diagnostics_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved diagnostics → {out_path}", flush=True)
    print(f"  mean off-diag cosine = {output['mean_off_diag_cosine']:.4f} "
          f"(< 0.95 indicates non-collapsed branches)", flush=True)
    print(f"  mean off-diag top-{int(args.k_frac*100)}%-overlap = "
          f"{output['mean_off_diag_overlap']:.4f}", flush=True)


if __name__ == "__main__":
    main()
