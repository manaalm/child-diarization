#!/usr/bin/env python3
"""Generate the headline figures for thesis_v2/.

Produces 9 PNGs in thesis_v2/figures/:
    ch3/label_distribution.png
    ch5/auroc_headline_bar.png
    ch5/pr_curves_key_systems.png
    ch6/onset_f1_by_dataset.png
    ch8/error_rate_by_age_band.png
    ch8/fp_rate_by_n_children.png
    ch8/fn_rate_by_n_adults.png
    ch8/av_failure_by_visibility.png
    ch8/cross_system_hard_clip_histogram.png

Run from repository root:
    python evaluation/generate_thesis_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os, json, csv
import sklearn.metrics as skm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(ROOT, "thesis_v2", "figures")
for sub in ("ch3", "ch5", "ch6", "ch8"):
    os.makedirs(os.path.join(FIG, sub), exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def thr_for(predictions_path, default=0.5):
    """Look up tuned threshold from the companion metrics JSON.

    MIL / pseudo-frame / baseline systems write `test_metrics_tuned.json`;
    ECAPA-enrollment systems write `enroll_test_metrics.json` (with the
    val-tuned threshold already in the `threshold` field). Falls back to
    `default` only if neither exists.
    """
    candidates = [
        predictions_path.replace("test_predictions.csv", "test_metrics_tuned.json"),
        predictions_path.replace("_predictions.csv", "_metrics.json"),
    ]
    for mp in candidates:
        if os.path.exists(mp):
            try:
                return float(json.load(open(mp)).get("threshold", default))
            except Exception:
                continue
    return default


def load_master():
    return pd.read_csv(os.path.join(ROOT, "whisper-modeling", "seen_child_splits", "master_with_split.csv"))


# Top-K systems consumed by the per-clip-prediction figures (8.1, 8.2, 8.3).
# Refreshed 2026-05-15 against evaluation/balanced_metrics_ba_tuned_summary.csv
# restricted to BIDS-corrected canonical seen-child test (n=635). Drops the
# n=441 legacy fused-small PU2, the WavLM pseudo-frame variant (BA=0.692, n=438
# subset), the n=541 legacy TS-MIL concat whose BIDS retrain is incomplete
# (n=628), and USC-SAIL enrollment (BA=0.714, AUROC=0.715 — low catalog
# rank). Swaps Qwen2.5-Omni for the stronger Qwen3-Omni-30B-A3B-Thinking
# (BA=0.705 vs 0.695; AUROC=0.786 vs 0.773). Retains Whisper pseudo-frame
# even though it is defined only on the n=438 subset (VTC-KCHI ∩ USC-SAIL
# pseudo-label coverage) because its catalog-leading AUROC=0.885 makes
# excluding it misleading; the n=438 caveat is noted in figure captions.
TOP_K = [
    ("Whisper-medium-MIL",         f"{ROOT}/mil/mil_results/whisper_medium_mil/test_predictions.csv", "score"),
    ("Whisper-MIL",                f"{ROOT}/mil/mil_results/whisper_mil/test_predictions.csv", "score"),
    ("BabAR enroll",               f"{ROOT}/babar_ecapa_enrollment_runs/enroll_test_predictions.csv", "prob"),
    ("Whisper-MIL ACMIL max",      f"{ROOT}/mil/mil_results/whisper_mil_acmil_max/test_predictions.csv", "score"),
    ("Whisper pseudo-frame",       f"{ROOT}/pseudo_frame/results/whisper_pseudo_frame/test_predictions.csv", "score"),
    ("Fused $\\times$ W-medium",   f"{ROOT}/baseline_results_seen_child/fused_attn_unfreeze2_whisper_medium/test_predictions.csv", "prob"),
    ("VTC enroll",                 f"{ROOT}/vtc_ecapa_enrollment_runs/enroll_test_predictions.csv", "prob"),
    ("Qwen3-Omni-Thinking 0-shot", f"{ROOT}/baselines/audio_llm_baseline_runs/qwen3_omni_30b_thinking/test_predictions.csv", "prob"),
]


def long_form_predictions():
    """Combine top-K system predictions into a long-form (system, clip) frame."""
    rows = []
    for name, path, score_col in TOP_K:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "audio_path" not in df.columns or score_col not in df.columns:
            continue
        thr = thr_for(path)
        df["pred_bin"] = (df[score_col].astype(float) >= thr).astype(int)
        for _, r in df.iterrows():
            rows.append({"system": name, "audio_path": r["audio_path"],
                         "pred": int(r["pred_bin"])})
    return pd.DataFrame(rows)


# =============================================================================
# Fig 3.1 — label distribution by (split, age band)
# =============================================================================
def fig_label_distribution():
    master = load_master()
    g = master.groupby(["split", "timepoint_norm"])["label"].agg(["mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    splits, tps = ["train", "val", "test"], ["14_month", "36_month"]
    width, x = 0.35, np.arange(len(splits))
    for i, tp in enumerate(tps):
        rates = []
        for s in splits:
            sub = g[(g.split == s) & (g.timepoint_norm == tp)]
            rates.append(float(sub["mean"].iloc[0]) if len(sub) else 0)
        bars = ax.bar(x + (i - 0.5) * width, rates, width, label=tp.replace("_", " "))
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{r:.2f}", ha="center", fontsize=8)
    ax.axhline(0.76, color="grey", linestyle="--", linewidth=0.8, label="overall (0.76)")
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Positive prevalence")
    ax.set_title("SAILS BIDS positive prevalence by (split, age band)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIG}/ch3/label_distribution.png"); plt.close()


# =============================================================================
# Fig 5.1 — AUROC headline bar with overlap band
# =============================================================================
def fig_auroc_bar():
    # Single-split AUROCs from each system's canonical metrics JSON.
    # Includes the new systems landed in May 2026 (Whisper pseudo-frame,
    # Whisper-medium-MIL, fused × medium / large-v3).
    systems = [
        ("Fused × Whisper-large-v3 (PU)", 0.907, "encoder"),
        ("Fused × Whisper-medium (PU)",   0.892, "encoder"),
        ("Fused × Whisper-small (PU)",    0.885, "encoder"),
        ("Whisper pseudo-frame",          0.881, "frame"),
        ("Whisper-medium-MIL",            0.873, "MIL"),
        ("Whisper-MIL TS-MIL concat",     0.869, "MIL"),
        ("Whisper-MIL",                   0.853, "MIL"),
        ("Whisper-MIL ACMIL max",         0.842, "MIL"),
        ("WavLM pseudo-frame",            0.831, "frame"),
        ("BabAR enrollment",              0.826, "enroll"),
        ("VTC-KCHI enrollment",           0.826, "enroll"),
        ("VTC enrollment",                0.813, "enroll"),
        ("WavLM-MIL",                    0.771, "MIL"),
        ("Qwen2.5-Omni zero-shot",       0.770, "LLM"),
        ("Sortformer enrollment",        0.691, "enroll"),
        ("Pyannote enrollment",          0.678, "enroll"),
        ("VBx enrollment",               0.675, "enroll"),
        ("USC-SAIL enrollment",          0.658, "enroll"),
        ("EEND-EDA enrollment",          0.521, "enroll"),
        ("Mean ensemble (best_audio_mil)", 0.878, "ensemble"),
    ]
    systems.sort(key=lambda s: s[1])
    family_color = {"encoder": "#1f77b4", "MIL": "#ff7f0e", "enroll": "#2ca02c",
                    "LLM": "#d62728", "ensemble": "#9467bd", "frame": "#8c564b"}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [s[0] for s in systems]
    aucs = [s[1] for s in systems]
    colors = [family_color[s[2]] for s in systems]
    bars = ax.barh(labels, aucs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
    for b, a in zip(bars, aucs):
        ax.text(a + 0.005, b.get_y() + b.get_height() / 2, f"{a:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvspan(0.82, 0.93, alpha=0.10, color="grey")
    ax.set_xlabel("AUROC (seen-child test)")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Headline AUROC by system family (single-split test)")
    fam_patches = [Patch(color=c, label=f) for f, c in family_color.items()]
    ax.legend(handles=fam_patches + [
        plt.Line2D([], [], color="grey", linestyle="--", label="trivial (AUROC=0.5)"),
        Patch(facecolor="grey", alpha=0.10, label="overlap band [0.82, 0.93]"),
    ], loc="lower right", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch5/auroc_headline_bar.png"); plt.close()


# =============================================================================
# Fig 5.2 — PR curves for one rep per family
# =============================================================================
def fig_pr_curves():
    KEY = [
        ("Fused encoder (small)", f"{ROOT}/baseline_results_seen_child/fused_attn_unfreeze2/test_predictions.csv", "label", "prob"),
        ("Whisper pseudo-frame",  f"{ROOT}/pseudo_frame/results/whisper_pseudo_frame/test_predictions.csv", "label", "prob"),
        ("Whisper-medium-MIL",    f"{ROOT}/mil/mil_results/whisper_medium_mil/test_predictions.csv", "label", "score"),
        ("Whisper-MIL",           f"{ROOT}/mil/mil_results/whisper_mil/test_predictions.csv", "label", "score"),
        ("BabAR enrollment",      f"{ROOT}/babar_ecapa_enrollment_runs/enroll_test_predictions.csv", "label", "prob"),
        ("Qwen2.5-Omni 0-shot",   f"{ROOT}/baselines/audio_llm_baseline_runs/qwen25_omni_7b/test_predictions.csv", "label", "prob"),
        ("AV gated late fusion",  f"{ROOT}/av_fusion/av_results/manual_only/predictions_test.csv", "label", "proba_gated_av"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, path, yt_col, ys_col in KEY:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if yt_col not in df.columns or ys_col not in df.columns:
            continue
        yt = df[yt_col].astype(int).values
        ys = df[ys_col].astype(float).values
        p, r, _ = skm.precision_recall_curve(yt, ys)
        ax.plot(r, p, label=f"{name} (AUPRC={skm.auc(r, p):.3f})", linewidth=1.5)
    ax.axhline(0.76, color="grey", linestyle="--", linewidth=0.8,
               label="trivial-positive (precision=0.76)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0.55, 1.02)
    ax.set_title("Precision–recall curves: one representative per family")
    ax.legend(loc="lower left", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch5/pr_curves_key_systems.png"); plt.close()


# =============================================================================
# Fig 6.1 — onset-F1 @ 250ms, system × dataset
# =============================================================================
def fig_onset_f1():
    ot = pd.read_csv(f"{ROOT}/evaluation/onset_tolerance_f1.csv")
    sub = ot[ot.tolerance_ms == 250]
    SYS = ["usc_sail", "babar", "vtc", "vtc_kchi", "pyannote", "sortformer", "vbx", "eend_eda", "joint_asr_diar"]
    PRETTY = {"usc_sail": "USC-SAIL", "babar": "BabAR", "vtc": "VTC", "vtc_kchi": "VTC-KCHI",
              "pyannote": "Pyannote", "sortformer": "Sortformer", "vbx": "VBx", "eend_eda": "EEND-EDA",
              "joint_asr_diar": "Joint ASR+Diar"}
    DS = ["providence", "playlogue", "synth_holdout"]
    DS_PRETTY = {"providence": "Providence", "playlogue": "Playlogue", "synth_holdout": "Synth holdout"}
    COLOR = {"providence": "#2ca02c", "playlogue": "#1f77b4", "synth_holdout": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SYS)); width = 0.27
    for i, ds in enumerate(DS):
        vals = []
        for s in SYS:
            r = sub[(sub.system == s) & (sub.dataset == ds)]
            vals.append(float(r.f1.iloc[0]) if len(r) else 0)
        ax.bar(x + (i - 1) * width, vals, width, label=DS_PRETTY[ds],
               color=COLOR[ds], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[s] for s in SYS], rotation=20, ha="right")
    ax.set_ylabel("Onset-F$_1$ at $\\pm 250$ ms")
    ax.set_title("Onset-tolerance F$_1$ by system and dataset")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 0.6)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch6/onset_f1_by_dataset.png"); plt.close()


# =============================================================================
# Ch.8 figures — error analysis on top-K predictions × metadata strata
# =============================================================================
def _build_merged():
    master = load_master()
    test_meta = master[master.split == "test"][[
        "audio_path", "label", "timepoint_norm",
        "#_children", "#_adults",
        "Video_Quality_Child_Face_Visibility",
        "Child_of_interest_clear", "Vocalizations"
    ]].rename(columns={"label": "label_gt"})
    preds = long_form_predictions()
    merged = preds.merge(test_meta, on="audio_path", how="left").dropna(subset=["label_gt"])
    merged["err"] = (merged["pred"] != merged["label_gt"]).astype(int)
    merged["fp"]  = ((merged["pred"] == 1) & (merged["label_gt"] == 0)).astype(int)
    merged["fn"]  = ((merged["pred"] == 0) & (merged["label_gt"] == 1)).astype(int)
    merged["n_ch"] = pd.to_numeric(merged["#_children"], errors="coerce")
    merged["n_ch_stratum"] = pd.cut(merged["n_ch"], [-1, 1, 2, np.inf], labels=["1", "2", "≥3"])
    merged["n_ad"] = pd.to_numeric(merged["#_adults"], errors="coerce")
    merged["n_ad_stratum"] = pd.cut(merged["n_ad"], [-1, 0, 1, np.inf], labels=["0", "1", "≥2"])
    return merged


def fig_err_age(merged):
    err_age = merged.groupby(["system", "timepoint_norm"], observed=True)["err"].mean().unstack()
    sys_order = err_age.mean(axis=1).sort_values().index.tolist()
    err_age = err_age.loc[sys_order]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(err_age)); width = 0.4
    for i, tp in enumerate(["14_month", "36_month"]):
        if tp in err_age.columns:
            ax.bar(x + (i - 0.5) * width, err_age[tp].values, width, label=tp.replace("_", " "))
    ax.axhline(1 - 0.760, color="grey", linestyle="--", linewidth=0.8,
               label="trivial-positive error ($1 - 0.76 = 0.24$)")
    ax.set_xticks(x); ax.set_xticklabels(err_age.index, rotation=22, ha="right")
    ax.set_ylabel("Error rate ((FN + FP) / N)")
    ax.set_title("Per-system error rate by age band (seen-child test)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch8/error_rate_by_age_band.png"); plt.close()
    return sys_order


def fig_fp_nchildren(merged, sys_order):
    fp_only = merged[merged.label_gt == 0]
    fp_rate = fp_only.groupby(["system", "n_ch_stratum"], observed=True)["fp"].mean().unstack()
    fp_rate = fp_rate.loc[sys_order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    strata = ["1", "2", "≥3"]
    x = np.arange(len(fp_rate)); width = 0.27
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for i, st in enumerate(strata):
        if st in fp_rate.columns:
            ax.bar(x + (i - 1) * width, fp_rate[st].fillna(0).values, width,
                   label=f"#children = {st}", color=colors[i], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(fp_rate.index, rotation=22, ha="right")
    ax.set_ylabel("False-positive rate (negative clips only)")
    ax.set_title("FP rate by number of children present in clip")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch8/fp_rate_by_n_children.png"); plt.close()


def fig_fn_nadults(merged, sys_order):
    """FN rate per system, stratified by #_adults in the clip.

    Restricts to ground-truth positives (label_gt == 1) so the rate is
    FN / N_pos within each stratum (the symmetric definition of the FP
    figure's FP / N_neg). Strata are 0 / 1 / >=2 adults — the >=2 bin
    pools the small 2- and 3-adult cells so the rate stays stable.

    Implementation gotchas (load-bearing):
      * Predictions must be re-binarized with each system's val-tuned
        threshold; the fused-encoder CSV already carries `pred_label`,
        but other CSVs do not. `_build_merged()` already did this via
        `long_form_predictions()`; we only consume `merged.fn` here.
      * Stratum is BIDS metadata (#_adults), independent of any system
        in TOP_K, so there is no circular dependency between the
        x-axis and the systems being plotted.
    """
    fn_only = merged[merged.label_gt == 1]
    fn_rate = fn_only.groupby(["system", "n_ad_stratum"], observed=True)["fn"].mean().unstack()
    fn_rate = fn_rate.loc[sys_order]
    n_per_stratum = fn_only.groupby("n_ad_stratum", observed=True)["audio_path"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    strata = ["0", "1", "≥2"]
    x = np.arange(len(fn_rate)); width = 0.27
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for i, st in enumerate(strata):
        if st in fn_rate.columns:
            n_pos = int(n_per_stratum.get(st, 0))
            ax.bar(x + (i - 1) * width, fn_rate[st].fillna(0).values, width,
                   label=f"#adults = {st} (n={n_pos} pos)", color=colors[i], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(fn_rate.index, rotation=22, ha="right")
    ax.set_ylabel("False-negative rate (positive clips only)")
    ax.set_title("FN rate by number of adults present in clip")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch8/fn_rate_by_n_adults.png"); plt.close()


def fig_cross_system_hardclips():
    """Cross-system error consistency histogram on the BIDS-corrected
    seen-child test set.

    Reads the BIDS aggregator output at
    cross_experiment_error_analysis_seen_child_bids/error_count_per_clip.csv
    (27 catalog systems with full BIDS coverage, n=635 clips). Easy / moderate
    / hard regimes annotated at 0-1 / 2-4 / 5+ thresholds. Pseudo-frame
    variants (n=438 subset) are handled in the chapter text rather than the
    figure to avoid mixing definedness regimes on one axis.
    """
    base = f"{ROOT}/cross_experiment_error_analysis_seen_child_bids"
    clips = pd.read_csv(f"{base}/error_count_per_clip.csv")
    n_systems_max = int(clips["error_count"].max())
    bin_edges = np.arange(-0.5, n_systems_max + 1.5, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = clips[clips.label == 1]
    neg = clips[clips.label == 0]
    pos_counts, _ = np.histogram(pos["error_count"], bins=bin_edges)
    neg_counts, _ = np.histogram(neg["error_count"], bins=bin_edges)
    centers = np.arange(0, n_systems_max + 1)
    ax.bar(centers, pos_counts, width=0.8, color="#4477AA", alpha=0.85,
           label=f"FN among positives (n={len(pos)})")
    ax.bar(centers, neg_counts, width=0.8, bottom=pos_counts, color="#EE6677", alpha=0.85,
           label=f"FP among negatives (n={len(neg)})")
    n_easy = int((clips["error_count"] <= 1).sum())
    n_mod = int(((clips["error_count"] >= 2) & (clips["error_count"] <= 4)).sum())
    n_hard = int((clips["error_count"] >= 5).sum())
    pct_easy = 100.0 * n_easy / len(clips)
    pct_mod = 100.0 * n_mod / len(clips)
    pct_hard = 100.0 * n_hard / len(clips)
    ax.axvline(1.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(4.5, color="gray", linestyle=":", linewidth=1)
    ymax = (pos_counts + neg_counts).max()
    ax.text(0.75, ymax * 0.92, f"easy\n{n_easy} ({pct_easy:.0f}%)",
            ha="center", va="top", fontsize=9, color="#222222")
    ax.text(3.0, ymax * 0.92, f"moderate\n{n_mod} ({pct_mod:.0f}%)",
            ha="center", va="top", fontsize=9, color="#222222")
    ax.text((5 + n_systems_max) / 2.0, ymax * 0.92,
            f"structurally hard\n{n_hard} ({pct_hard:.0f}%)",
            ha="center", va="top", fontsize=9, color="#222222")
    ax.set_xlabel("Number of catalog systems wrong on this clip (27 systems)")
    ax.set_ylabel("Number of test clips")
    ax.set_xticks(centers[::2])
    ax.set_xlim(-0.7, n_systems_max + 0.7)
    ax.set_title(f"Cross-system error consistency (n={len(clips)} clips)")
    ax.legend(frameon=False, loc="upper center")
    plt.tight_layout(); plt.savefig(f"{FIG}/ch8/cross_system_hard_clip_histogram.png"); plt.close()


def fig_av_visibility():
    test_meta = load_master()
    test_meta = test_meta[test_meta.split == "test"][[
        "audio_path", "label", "Video_Quality_Child_Face_Visibility"]].rename(columns={"label": "label_gt"})
    av_preds = []
    for name, path, score_col in [
        ("TalkNet-ASD enroll", f"{ROOT}/video_asd_ecapa_enrollment_runs/talknet_asd/enroll_test_predictions.csv", "prob"),
        ("BabAR enroll (audio ref)", f"{ROOT}/babar_ecapa_enrollment_runs/enroll_test_predictions.csv", "prob"),
    ]:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        if "audio_path" not in df.columns or score_col not in df.columns: continue
        thr = thr_for(path)
        for _, r in df.iterrows():
            av_preds.append({"system": name, "audio_path": r["audio_path"],
                             "pred": int(float(r[score_col]) >= thr)})
    # Manual-feature gated AV via av_master_features bridge
    gp = f"{ROOT}/av_fusion/av_results/manual_only/predictions_test.csv"
    am = f"{ROOT}/av_fusion/av_results/manual_only/av_master_features.csv"
    if os.path.exists(gp) and os.path.exists(am):
        df = pd.read_csv(gp)
        bridge = pd.read_csv(am)[["clip_id", "audio_path"]].drop_duplicates()
        df2 = df.merge(bridge, on="clip_id", how="left").dropna(subset=["audio_path"])
        for _, r in df2.iterrows():
            av_preds.append({"system": "Manual-feature gated AV",
                             "audio_path": r["audio_path"],
                             "pred": int(r["pred_gated_av"])})
    av = pd.DataFrame(av_preds).merge(test_meta, on="audio_path", how="left")
    av = av.dropna(subset=["Video_Quality_Child_Face_Visibility"])
    av["err"] = (av["pred"] != av["label_gt"]).astype(int)
    err_vis = av.groupby(["system", "Video_Quality_Child_Face_Visibility"], observed=True)["err"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = err_vis.columns.tolist()
    sys_av = err_vis.index.tolist()
    x = np.arange(len(sys_av)); width = 0.8 / max(1, len(cats))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    for i, cat in enumerate(cats):
        ax.bar(x + (i - len(cats) / 2 + 0.5) * width, err_vis[cat].fillna(0).values,
               width, label=str(cat), color=cmap[i])
    ax.set_xticks(x); ax.set_xticklabels(sys_av, rotation=15, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_title("AV system error rate by face-visibility category")
    ax.legend(title="Face visibility", loc="upper right", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/ch8/av_failure_by_visibility.png"); plt.close()


def main():
    fig_label_distribution(); print("✓ ch3/label_distribution.png")
    fig_auroc_bar();          print("✓ ch5/auroc_headline_bar.png")
    fig_pr_curves();          print("✓ ch5/pr_curves_key_systems.png")
    fig_onset_f1();           print("✓ ch6/onset_f1_by_dataset.png")
    merged = _build_merged()
    sys_order = fig_err_age(merged); print("✓ ch8/error_rate_by_age_band.png")
    fig_fp_nchildren(merged, sys_order); print("✓ ch8/fp_rate_by_n_children.png")
    fig_fn_nadults(merged, sys_order);   print("✓ ch8/fn_rate_by_n_adults.png")
    fig_av_visibility();                 print("✓ ch8/av_failure_by_visibility.png")
    fig_cross_system_hardclips();        print("✓ ch8/cross_system_hard_clip_histogram.png")


if __name__ == "__main__":
    main()
