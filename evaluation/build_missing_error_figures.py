"""Build missing error-chapter artifacts for thesis_v2:

1. Multi-child prevalence-by-age-band table (LaTeX fragment + CSV).
2. Cross-system hard-clip histogram figure (PNG).

Runs offline from existing CSVs:
- whisper-modeling/seen_child_splits/master_with_split.csv (test split)
- cross_experiment_error_analysis_seen_child/fn_ranked_by_frequency.csv
- cross_experiment_error_analysis_seen_child/fp_ranked_by_frequency.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
META_CSV = ROOT / "whisper-modeling/seen_child_splits/master_with_split.csv"
FN_CSV = ROOT / "cross_experiment_error_analysis_seen_child/fn_ranked_by_frequency.csv"
FP_CSV = ROOT / "cross_experiment_error_analysis_seen_child/fp_ranked_by_frequency.csv"
SUMMARY_CSV = ROOT / "cross_experiment_error_analysis_seen_child/summary.csv"

CH8_FIG_DIR = ROOT / "thesis_v2/figures/ch8"
CH8_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "evaluation/error_chapter_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_multichild_prevalence_table():
    meta = pd.read_csv(META_CSV)
    test = meta[meta["split"] == "test"].copy()
    test["n_children"] = pd.to_numeric(test["#_children"], errors="coerce").fillna(0).astype(int)
    test["multi_child"] = test["n_children"] >= 2

    rows = []
    for tp in ["14_month", "36_month"]:
        sub = test[test["timepoint_norm"] == tp]
        n_total = len(sub)
        n_multi = int(sub["multi_child"].sum())
        n_single = n_total - n_multi
        mean_n = sub["n_children"].mean()
        max_n = int(sub["n_children"].max())
        rows.append(
            {
                "timepoint": tp.replace("_month", "m"),
                "n_clips": n_total,
                "n_single_child": n_single,
                "n_multi_child": n_multi,
                "pct_multi_child": 100.0 * n_multi / n_total if n_total else 0.0,
                "mean_n_children": round(mean_n, 2),
                "max_n_children": max_n,
            }
        )

    sub_all = test
    rows.append(
        {
            "timepoint": "all",
            "n_clips": len(sub_all),
            "n_single_child": len(sub_all) - int(sub_all["multi_child"].sum()),
            "n_multi_child": int(sub_all["multi_child"].sum()),
            "pct_multi_child": 100.0 * sub_all["multi_child"].sum() / len(sub_all),
            "mean_n_children": round(sub_all["n_children"].mean(), 2),
            "max_n_children": int(sub_all["n_children"].max()),
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "multichild_prevalence_by_age.csv", index=False)
    print("multi-child prevalence by age band:")
    print(df.to_string(index=False))
    return df


def build_cross_system_hard_clip_histogram():
    """Histogram of how many of the 32 catalog systems get each clip wrong.

    For positives: use FN_count from fn_ranked_by_frequency.csv.
    For negatives: use FP_count from fp_ranked_by_frequency.csv.
    Combine into a single error_count per clip across all 441 test clips.
    """
    fn = pd.read_csv(FN_CSV)
    fp = pd.read_csv(FP_CSV)
    fn = fn.rename(columns={"FN_count": "error_count"})[
        ["audio_path", "timepoint_norm", "#_children", "error_count", "n_systems_with_pred"]
    ]
    fn["clip_label"] = "positive"
    fp = fp.rename(columns={"FP_count": "error_count"})[
        ["audio_path", "timepoint_norm", "#_children", "error_count", "n_systems_with_pred"]
    ]
    fp["clip_label"] = "negative"
    clips = pd.concat([fn, fp], ignore_index=True)
    n_systems_max = int(clips["n_systems_with_pred"].max())

    bin_edges = np.arange(-0.5, n_systems_max + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos_counts, _ = np.histogram(fn["error_count"], bins=bin_edges)
    neg_counts, _ = np.histogram(fp["error_count"], bins=bin_edges)
    centers = np.arange(0, n_systems_max + 1)

    ax.bar(centers, pos_counts, width=0.8, color="#4477AA", alpha=0.85, label=f"FN among positives (n={len(fn)})")
    ax.bar(
        centers,
        neg_counts,
        width=0.8,
        bottom=pos_counts,
        color="#EE6677",
        alpha=0.85,
        label=f"FP among negatives (n={len(fp)})",
    )

    n_easy = int(((clips["error_count"] <= 1)).sum())
    n_mod = int(((clips["error_count"] >= 2) & (clips["error_count"] <= 4)).sum())
    n_hard = int((clips["error_count"] >= 5).sum())
    pct_easy = 100.0 * n_easy / len(clips)
    pct_mod = 100.0 * n_mod / len(clips)
    pct_hard = 100.0 * n_hard / len(clips)

    ax.axvline(1.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(4.5, color="gray", linestyle=":", linewidth=1)
    ymax = (pos_counts + neg_counts).max()
    ax.text(0.75, ymax * 0.92, f"easy\n{n_easy} ({pct_easy:.0f}%)", ha="center", va="top", fontsize=9, color="#222222")
    ax.text(3.0, ymax * 0.92, f"moderate\n{n_mod} ({pct_mod:.0f}%)", ha="center", va="top", fontsize=9, color="#222222")
    ax.text(
        (5 + n_systems_max) / 2.0,
        ymax * 0.92,
        f"structurally hard\n{n_hard} ({pct_hard:.0f}%)",
        ha="center",
        va="top",
        fontsize=9,
        color="#222222",
    )

    ax.set_xlabel("Number of catalog systems wrong on this clip")
    ax.set_ylabel("Number of test clips")
    ax.set_xticks(centers[::2])
    ax.set_xlim(-0.7, n_systems_max + 0.7)
    ax.set_title(
        "Cross-system error consistency across the seen-child test set "
        f"(n={len(clips)} clips, {n_systems_max} systems)"
    )
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    out = CH8_FIG_DIR / "cross_system_hard_clip_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"easy (0-1): {n_easy} ({pct_easy:.1f}%)  moderate (2-4): {n_mod} ({pct_mod:.1f}%)  hard (5+): {n_hard} ({pct_hard:.1f}%)")
    return {
        "n_clips": len(clips),
        "n_systems": n_systems_max,
        "n_easy": n_easy,
        "n_mod": n_mod,
        "n_hard": n_hard,
        "pct_easy": pct_easy,
        "pct_mod": pct_mod,
        "pct_hard": pct_hard,
    }


if __name__ == "__main__":
    df = build_multichild_prevalence_table()
    stats = build_cross_system_hard_clip_histogram()
    summary = {
        "multichild_prevalence": df.to_dict(orient="records"),
        "hard_clip_stats": stats,
    }
    import json

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'summary.json'}")
