"""Cross-experiment consistency-of-failure ranking.

Supports both the seen-child and cross-child splits. For each clip in the
chosen test split, counts how many independent systems mispredicted it
(FN if label=1 & pred=0, FP if label=0 & pred=1) and ranks clips by
failure frequency. Joins per-clip metadata for diagnostic context.

Usage:
    python cross_experiment_error_comparison_seen_child.py --split seen_child
    python cross_experiment_error_comparison_seen_child.py --split cross_child

Outputs to cross_experiment_error_analysis_{split}/:
  fn_ranked_by_frequency.csv  — top FN clips ranked by miss count
  fp_ranked_by_frequency.csv  — top FP clips ranked
  fn_top100.md / fp_top100.md — human-readable tables (top 100 each)
  per_system_error_counts.csv — per-system FN/FP totals
  systems_used.txt            — exact prediction files included
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent


def _seen_child_rules() -> list[tuple[str, str]]:
    return [
        # Enrollment-based diarizers
        ("usc_sail",       "whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv"),
        ("pyannote",       "pyannote_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("babar",          "babar_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("vtc",            "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("vtc_kchi",       "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("vbx",            "vbx_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("eend_eda",       "eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("sortformer",     "sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        ("talknet_asd",    "video_asd_ecapa_enrollment_runs/talknet_asd/enroll_test_predictions.csv"),
        ("loconet_ecapa",  "video_asd_ecapa_enrollment_runs/loconet_ecapa/enroll_test_predictions.csv"),
        ("pyannote_alt",   "pyannote/pyannote_enrollment_runs/test_predictions.csv"),
        # Frame-window MIL
        ("wavlm_mil",                 "mil/mil_results/wavlm_mil/test_predictions.csv"),
        ("whisper_mil",               "mil/mil_results/whisper_mil/test_predictions.csv"),
        ("hubert_large_mil_layersum", "mil/mil_results/hubert_large_mil_layersum/test_predictions.csv"),
        ("wav2vec2_large_mil",        "mil/mil_results/wav2vec2_large_mil/test_predictions.csv"),
        ("seg_mil_usc_sail_transformer",  "mil/mil_results/seg_mil/usc_sail_transformer/test_predictions.csv"),
        ("seg_mil_babar_vtc_transformer", "mil/mil_results/seg_mil/babar_vtc_transformer/test_predictions.csv"),
        ("seg_mil_pyannote_transformer",  "mil/mil_results/seg_mil/pyannote_transformer/test_predictions.csv"),
        ("pseudo_frame_wavlm",     "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv"),
        # Foundation-model baselines
        ("audio_llm_qwen2",        "baselines/audio_llm_baseline_runs/qwen2_audio_7b/test_predictions.csv"),
        ("parakeet_tdt",           "baselines/parakeet_baseline_runs/parakeet_tdt_0.6b_v2/test_predictions.csv"),
        ("granite_speech",         "baselines/audio_model_baseline_runs/granite_speech_1b/test_predictions.csv"),
        ("cohere_transcribe",      "baselines/audio_model_baseline_runs/cohere_transcribe/test_predictions.csv"),
        ("panns_cnn14",            "baselines/panns_baseline_runs/cnn14/test_predictions.csv"),
        ("clap_htsat_fused",       "baselines/clap_baseline_runs/clap_htsat_fused/test_predictions.csv"),
        ("raw_ecapa_top3",         "baselines/raw_ecapa_baseline_runs/top3/test_predictions.csv"),
        ("vad_silero",             "baselines/vad_baseline_runs/silero/test_predictions.csv"),
        ("vad_energy",             "baselines/vad_baseline_runs/energy/test_predictions.csv"),
        # Encoder baselines (representative)
        ("baseline_fused_attn_unfreeze2", "baseline_results_seen_child/fused_attn_unfreeze2/test_predictions.csv"),
        ("baseline_whisper_attn",         "baseline_results_seen_child/whisper_attn/test_predictions.csv"),
        # Ensembles / stackers
        ("ensemble_metadata_stack",       "ensemble_runs/metadata_stack/test_predictions.csv"),
        ("ensemble_metadata_stack_av",    "ensemble_runs/metadata_stack_av/test_predictions.csv"),
    ]


def _cross_child_rules() -> list[tuple[str, str]]:
    return [
        # Cross-child role-only diarizers (no enrollment for cross-child since speakers are unseen)
        ("babar_role_only",     "evaluation/cross_child_babar_role_only/test_predictions.csv"),
        ("vtc_role_only",       "evaluation/cross_child_vtc_role_only/test_predictions.csv"),
        ("vtc_kchi_role_only",  "evaluation/cross_child_vtc_kchi_role_only/test_predictions.csv"),
        # Frame-window MIL (cross-child)
        ("wavlm_mil_cross_child",        "mil/mil_results/wavlm_mil_cross_child/test_predictions.csv"),
        ("whisper_mil_cross_child",      "mil/mil_results/whisper_mil_cross_child/test_predictions.csv"),
        ("wavlm_mil_cross_child_synth",  "mil/mil_results/wavlm_mil_cross_child_synth/test_predictions.csv"),
        ("whisper_mil_cross_child_synth","mil/mil_results/whisper_mil_cross_child_synth/test_predictions.csv"),
        # Foundation-model baselines (cross-child variants)
        ("audio_llm_qwen2_cross_child",  "baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child/test_predictions.csv"),
        ("parakeet_tdt_cross_child",     "baselines/parakeet_baseline_runs/parakeet_tdt_0.6b_v2_cross_child/test_predictions.csv"),
        ("granite_speech_cross_child",   "baselines/audio_model_baseline_runs/granite_speech_1b_cross_child/test_predictions.csv"),
        ("cohere_transcribe_cross_child","baselines/audio_model_baseline_runs/cohere_transcribe_cross_child/test_predictions.csv"),
        ("panns_cnn14_cross_child",      "baselines/panns_baseline_runs/cnn14_cross_child/test_predictions.csv"),
        ("clap_htsat_fused_cross_child", "baselines/clap_baseline_runs/clap_htsat_fused_cross_child/test_predictions.csv"),
        ("vad_silero_cross_child",       "baselines/vad_baseline_runs/silero_cross_child/test_predictions.csv"),
        ("vad_energy_cross_child",       "baselines/vad_baseline_runs/energy_cross_child/test_predictions.csv"),
        # Encoder baselines (full set — these are the primary cross-child baselines)
        ("baseline_fused_attn",            "baselines/baseline_results/fused_attn/test_predictions.csv"),
        ("baseline_fused_attn_lw",         "baselines/baseline_results/fused_attn_lw/test_predictions.csv"),
        ("baseline_fused_attn_unfreeze2",  "baselines/baseline_results/fused_attn_unfreeze2/test_predictions.csv"),
        ("baseline_wavlm_attn",            "baselines/baseline_results/wavlm_attn/test_predictions.csv"),
        ("baseline_wavlm_attn_lw",         "baselines/baseline_results/wavlm_attn_lw/test_predictions.csv"),
        ("baseline_wavlm_mean",            "baselines/baseline_results/wavlm_mean/test_predictions.csv"),
        ("baseline_wavlm_stats_lw",        "baselines/baseline_results/wavlm_stats_lw/test_predictions.csv"),
        ("baseline_whisper_attn",          "baselines/baseline_results/whisper_attn/test_predictions.csv"),
        ("baseline_whisper_attn_aug",      "baselines/baseline_results/whisper_attn_aug/test_predictions.csv"),
        ("baseline_whisper_attn_aug_ptt",  "baselines/baseline_results/whisper_attn_aug_ptt/test_predictions.csv"),
        ("baseline_whisper_attn_lw",       "baselines/baseline_results/whisper_attn_lw/test_predictions.csv"),
        ("baseline_whisper_attn_ptt",      "baselines/baseline_results/whisper_attn_ptt/test_predictions.csv"),
        ("baseline_whisper_attn_unfreeze2","baselines/baseline_results/whisper_attn_unfreeze2/test_predictions.csv"),
        ("baseline_whisper_mean",          "baselines/baseline_results/whisper_mean/test_predictions.csv"),
        ("baseline_whisper_stats_lw",      "baselines/baseline_results/whisper_stats_lw/test_predictions.csv"),
        # Cross-child ensembles
        ("ensemble_cross_child_audio_mil",         "ensemble_runs/cross_child_best_audio_mil/test_predictions.csv"),
        ("ensemble_cross_child_audio_mil_clap",    "ensemble_runs/cross_child_best_audio_mil_with_clap/test_predictions.csv"),
    ]


def discover_systems(rules: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    out = []
    for name, rel in rules:
        p = REPO / rel
        if p.exists():
            out.append((name, p))
        else:
            print(f"  [skip] {name}: {p} not found")
    return out


def load_pred_binary(path: Path) -> pd.DataFrame:
    """Load a prediction CSV and normalize to (audio_path, label, pred_bin)."""
    df = pd.read_csv(path)
    if "audio_path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path}: missing audio_path or label")

    # Find the binary prediction column
    pred_col = next(
        (c for c in ["prediction", "pred_label", "pred", "predicted"] if c in df.columns),
        None,
    )
    if pred_col is None:
        # Fall back to thresholding prob/score at 0.5
        prob_col = next((c for c in ["prob", "score", "probability"] if c in df.columns), None)
        if prob_col is None:
            raise ValueError(f"{path}: no binary prediction or prob column")
        df["pred_bin"] = (df[prob_col].astype(float) >= 0.5).astype(int)
    else:
        df["pred_bin"] = df[pred_col].astype(float).round().astype(int)

    df["label"] = df["label"].astype(int)
    return df[["audio_path", "label", "pred_bin"]].copy()


SPLIT_CONFIG = {
    "seen_child": {
        "test_csv": "whisper-modeling/seen_child_splits/test.csv",
        "out_dir": "cross_experiment_error_analysis_seen_child",
        "rules_fn": _seen_child_rules,
    },
    "cross_child": {
        "test_csv": "baselines/splits/test.csv",
        "out_dir": "cross_experiment_error_analysis_cross_child",
        "rules_fn": _cross_child_rules,
    },
}


def load_metadata(split: str) -> pd.DataFrame:
    test_csv = REPO / SPLIT_CONFIG[split]["test_csv"]
    df = pd.read_csv(test_csv)
    keep = [c for c in [
        "audio_path", "child_id", "timepoint_norm", "task", "session",
        "#_children", "#_adults", "Child_of_interest_clear",
        "Video_Quality_Child_Face_Visibility", "Video_Quality_Lighting",
        "Video_Quality_Resolution",
    ] if c in df.columns]
    return df[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["seen_child", "cross_child"], default="seen_child")
    args = ap.parse_args()
    split = args.split

    out_dir = REPO / SPLIT_CONFIG[split]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = SPLIT_CONFIG[split]["rules_fn"]()
    systems = discover_systems(rules)
    print(f"\n=== Split: {split} ===")
    print(f"Using {len(systems)} systems:")
    for n, p in systems:
        print(f"  {n:35s}  {p.relative_to(REPO)}")

    with open(out_dir / "systems_used.txt", "w") as f:
        for n, p in systems:
            f.write(f"{n}\t{p.relative_to(REPO)}\n")

    # Load all systems → DataFrame indexed by audio_path with one column per system
    base = None
    sys_cols = []
    for name, path in systems:
        d = load_pred_binary(path).rename(columns={"pred_bin": name})
        if base is None:
            base = d
        else:
            # Sanity: labels must match across systems for shared audio_paths
            merged = base.merge(d, on="audio_path", how="outer", suffixes=("", "_dup"))
            if "label_dup" in merged.columns:
                mismatches = merged[merged["label"] != merged["label_dup"]]
                if len(mismatches) > 0:
                    print(f"  [warn] {name}: {len(mismatches)} label mismatches vs base — overwriting with base label")
                merged = merged.drop(columns=["label_dup"])
            base = merged
        sys_cols.append(name)

    # Some systems may be missing rows; fill missing predictions as NaN-aware
    n_total = len(base)
    print(f"\nMerged matrix: {n_total} unique audio_paths × {len(sys_cols)} systems")

    # FN / FP counting per clip
    label = base["label"].astype(int)
    fn_counts = pd.Series(0, index=base.index, name="FN_count")
    fp_counts = pd.Series(0, index=base.index, name="FP_count")
    n_systems = pd.Series(0, index=base.index, name="n_systems_with_pred")
    for col in sys_cols:
        v = base[col]
        present = v.notna()
        n_systems += present.astype(int)
        # FN: label==1 & pred==0
        fn_counts += ((label == 1) & present & (v.fillna(-1).astype(int) == 0)).astype(int)
        # FP: label==0 & pred==1
        fp_counts += ((label == 0) & present & (v.fillna(-1).astype(int) == 1)).astype(int)

    base["FN_count"] = fn_counts
    base["FP_count"] = fp_counts
    base["n_systems_with_pred"] = n_systems

    # Which specific systems missed each clip
    def systems_failing(row, mode: str) -> str:
        names = []
        for col in sys_cols:
            v = row.get(col)
            if pd.isna(v):
                continue
            v = int(v)
            if mode == "fn" and row["label"] == 1 and v == 0:
                names.append(col)
            elif mode == "fp" and row["label"] == 0 and v == 1:
                names.append(col)
        return ";".join(names)

    base["systems_FN"] = base.apply(lambda r: systems_failing(r, "fn"), axis=1)
    base["systems_FP"] = base.apply(lambda r: systems_failing(r, "fp"), axis=1)

    # Join metadata
    meta = load_metadata(split)
    base = base.merge(meta, on="audio_path", how="left")
    print(f"After metadata join: {len(base)} rows ({base['child_id'].notna().sum()} matched seen-child split)")

    # Per-system totals
    per_system_rows = []
    for col in sys_cols:
        v = base[col]
        present = v.notna()
        fn = int(((label == 1) & present & (v.fillna(-1).astype(int) == 0)).sum())
        fp = int(((label == 0) & present & (v.fillna(-1).astype(int) == 1)).sum())
        per_system_rows.append({
            "system": col,
            "n_predictions": int(present.sum()),
            "FN": fn, "FP": fp,
            "errors": fn + fp,
        })
    pd.DataFrame(per_system_rows).sort_values("errors").to_csv(
        out_dir / "per_system_error_counts.csv", index=False
    )

    # Ranked outputs
    base_meta_cols = [c for c in [
        "audio_path", "child_id", "timepoint_norm", "task", "#_children", "#_adults",
        "Child_of_interest_clear",
    ] if c in base.columns]

    fn_rank = base[base["label"] == 1].sort_values(
        ["FN_count", "child_id"], ascending=[False, True]
    )[base_meta_cols + ["FN_count", "n_systems_with_pred", "systems_FN"]].reset_index(drop=True)
    fp_rank = base[base["label"] == 0].sort_values(
        ["FP_count", "child_id"], ascending=[False, True]
    )[base_meta_cols + ["FP_count", "n_systems_with_pred", "systems_FP"]].reset_index(drop=True)

    fn_rank.to_csv(out_dir / "fn_ranked_by_frequency.csv", index=False)
    fp_rank.to_csv(out_dir / "fp_ranked_by_frequency.csv", index=False)

    # Markdown top-100 summaries (manual format to avoid tabulate dep)
    def to_markdown(df: pd.DataFrame, title: str, count_col: str, n: int = 100) -> str:
        sub = df.head(n).copy()
        sub["clip_short"] = sub["audio_path"].apply(
            lambda p: "/".join(Path(p).parts[-3:]).replace("_audio.wav", "")
        )
        keep = ["clip_short", count_col, "n_systems_with_pred",
                "child_id", "timepoint_norm", "task", "#_children"]
        keep = [c for c in keep if c in sub.columns]
        sub2 = sub[keep].copy()
        header = "| " + " | ".join(keep) + " |"
        sep = "|" + "|".join(["---"] * len(keep)) + "|"
        rows = []
        for _, r in sub2.iterrows():
            cells = []
            for c in keep:
                v = r[c]
                if pd.isna(v):
                    cells.append("")
                else:
                    cells.append(str(v))
            rows.append("| " + " | ".join(cells) + " |")
        return f"# {title} (top {len(sub)} of {len(df)})\n\n" + "\n".join([header, sep] + rows)

    (out_dir / "fn_top100.md").write_text(
        to_markdown(fn_rank, "Top-100 consistently-missed clips (False Negatives)", "FN_count")
    )
    (out_dir / "fp_top100.md").write_text(
        to_markdown(fp_rank, "Top-100 consistently-mispredicted clips (False Positives)", "FP_count")
    )

    # Headline summary
    n_fn_pos = (fn_rank["FN_count"] > 0).sum()
    n_fp_pos = (fp_rank["FP_count"] > 0).sum()
    n_systems = len(sys_cols)
    summary = {
        "n_systems": n_systems,
        "n_clips": int(len(base)),
        "n_positive_clips": int((label == 1).sum()),
        "n_negative_clips": int((label == 0).sum()),
        "n_clips_with_any_FN": int(n_fn_pos),
        "n_clips_with_any_FP": int(n_fp_pos),
        "max_FN_count": int(fn_rank["FN_count"].max()),
        "max_FP_count": int(fp_rank["FP_count"].max()),
        "median_FN_count_among_positives": float(base.loc[label == 1, "FN_count"].median()),
        "median_FP_count_among_negatives": float(base.loc[label == 0, "FP_count"].median()),
    }
    pd.Series(summary).to_csv(out_dir / "summary.csv")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
