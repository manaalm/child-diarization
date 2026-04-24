"""Assemble the master AV feature table for fusion model training and evaluation.

Merges clip metadata, manual BIDS annotation fields, audio baseline scores,
and (optionally) visual features and ASD features into av_master_features.csv.

Audio scores are joined for val and test clips only; train clips have
existing_audio_score = NaN (train-set enrollment scores are not available
without data leakage). Fusion models use late fusion at inference time.

Usage:
    python av_fusion/scripts/build_av_feature_table.py \\
        --metadata-csv        whisper-modeling/seen_child_splits/master_with_split.csv \\
        --audio-scores-val    babar_ecapa_enrollment_runs/enroll_val_predictions.csv \\
        --audio-scores-test   babar_ecapa_enrollment_runs/enroll_test_predictions.csv \\
        --audio-score-col     prob \\
        --output-dir          av_fusion/av_results/manual_only/ \\
        [--visual-features-csv  av_fusion/av_results/manual_only/visual_features.csv] \\
        [--asd-features-csv     av_fusion/av_results/manual_only/asd_features.csv] \\
        [--run-name             manual_only]

Exit codes:
    0 = success (split integrity verified)
    1 = split integrity violation (stops execution)
    2 = required input CSV missing
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import assert_split_integrity, get_repo_root, save_json

_REPO = get_repo_root()

_MANUAL_ANNOTATION_COLS = [
    "Video_Quality_Child_Face_Visibility",
    "Video_Quality_Child_Body_Visibility",
    "Video_Quality_Child_Hand_Visibility",
    "Video_Quality_Lighting",
    "Video_Quality_Resolution",
    "Video_Quality_Motion",
    "Child_of_interest_clear",
    "#_adults",
    "#_children",
    "Body_Parts_Visible",
    "Angle_of_Body",
]

_AUTO_VISUAL_COLS = [
    "n_faces_detected_mean",
    "n_faces_detected_max",
    "n_face_tracks",
    "max_face_track_duration_sec",
    "max_face_track_fraction_clip",
    "mean_face_detection_confidence",
    "max_face_detection_confidence",
    "mean_face_box_area_fraction",
    "max_face_box_area_fraction",
    "min_face_box_area_fraction",
    "face_center_motion_std",
    "visual_quality_score",
    "child_visible_score",
    "off_camera_likely_score",
]

_ASD_COLS = [
    "max_asd_score_any_face",
    "mean_asd_score_any_face",
    "max_asd_score_smallest_face",
    "mean_asd_score_smallest_face",
    "fraction_frames_any_active",
    "fraction_frames_child_active",
    "n_active_speaker_tracks",
]


def _clip_id(row: pd.Series) -> str:
    if "clip_id" in row.index:
        return str(row["clip_id"])
    if "Unnamed: 0" in row.index:
        return str(int(row["Unnamed: 0"]))
    return str(row.name)


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def build_table(
    metadata_csv: str,
    audio_scores_val: Optional[str],
    audio_scores_test: Optional[str],
    audio_score_col: str,
    visual_features_csv: Optional[str] = None,
    asd_features_csv: Optional[str] = None,
    gpt4o_features_csv: Optional[str] = None,
    extra_asd_csvs: Optional[List[Tuple[str, str]]] = None,
    run_name: str = "default",
) -> pd.DataFrame:
    # Load metadata
    df = pd.read_csv(metadata_csv, low_memory=False)

    # Derive clip_id
    df["clip_id"] = df.apply(_clip_id, axis=1)

    # Resolve video_path from BidsProcessed → BidsRaw → video_path
    def resolve_vpath(row):
        for col in ("BidsProcessed", "BidsRaw", "video_path"):
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    return str(val)
        return None

    df["video_path"] = df.apply(resolve_vpath, axis=1)

    # Normalize split column
    if "split" not in df.columns:
        raise ValueError("metadata CSV must have a 'split' column")

    # Label column
    if "label" not in df.columns:
        raise ValueError("metadata CSV must have a 'label' column")

    # Normalize age band
    if "timepoint_norm" in df.columns:
        df["age_band"] = df["timepoint_norm"].map({
            "14_month": "14_18_months",
            "36_month": "34_38_months",
        }).fillna("unknown")
        df["age_band_binary"] = (df["age_band"] == "34_38_months").astype(int)
    else:
        df["age_band"] = "unknown"
        df["age_band_binary"] = 0

    # Manual BIDS annotation derived features
    if "Child_of_interest_clear" in df.columns:
        df["child_of_interest_clear_binary"] = (
            df["Child_of_interest_clear"].str.lower().str.strip() == "yes"
        ).astype(float)
    else:
        df["child_of_interest_clear_binary"] = float("nan")

    if "Video_Quality_Child_Face_Visibility" in df.columns:
        df["manual_face_visibility_norm"] = pd.to_numeric(
            df["Video_Quality_Child_Face_Visibility"], errors="coerce"
        ) / 10.0
    else:
        df["manual_face_visibility_norm"] = float("nan")

    lighting = pd.to_numeric(df.get("Video_Quality_Lighting", float("nan")), errors="coerce")
    resolution = pd.to_numeric(df.get("Video_Quality_Resolution", float("nan")), errors="coerce")
    df["manual_quality_norm"] = (lighting + resolution) / 20.0

    adults = pd.to_numeric(df.get("#_adults", 0), errors="coerce").fillna(0)
    children = pd.to_numeric(df.get("#_children", 0), errors="coerce").fillna(0)
    df["n_people_total"] = (adults + children).astype(int)
    df["multi_person_clip"] = (df["n_people_total"] > 1).astype(int)

    # Audio scores — val and test only (train scores not available without leakage)
    df["existing_audio_score"] = float("nan")
    for split_name, csv_path in [("val", audio_scores_val), ("test", audio_scores_test)]:
        if csv_path and os.path.exists(csv_path):
            audio_df = pd.read_csv(csv_path)
            if audio_score_col not in audio_df.columns:
                print(f"WARNING: column '{audio_score_col}' not in {csv_path}; skipping", file=sys.stderr)
                continue
            audio_map = dict(zip(audio_df["audio_path"], audio_df[audio_score_col]))
            mask = df["split"] == split_name
            if "audio_path" in df.columns:
                df.loc[mask, "existing_audio_score"] = df.loc[mask, "audio_path"].map(audio_map)
            else:
                print(f"WARNING: no 'audio_path' column in metadata; cannot join audio scores", file=sys.stderr)

    # Visual features (optional)
    for col in _AUTO_VISUAL_COLS:
        df[col] = float("nan")

    # visual_eligibility_score: use automatic if available, else manual proxy
    df["visual_eligibility_score"] = float("nan")

    if visual_features_csv and os.path.exists(visual_features_csv):
        vis_df = pd.read_csv(visual_features_csv)
        if "clip_id" in vis_df.columns:
            vis_df["clip_id"] = vis_df["clip_id"].astype(str)
            df = df.merge(
                vis_df[["clip_id"] + [c for c in _AUTO_VISUAL_COLS + ["visual_eligibility_score"] if c in vis_df.columns]],
                on="clip_id",
                how="left",
                suffixes=("", "_vis"),
            )
            # Update columns from merge
            for col in _AUTO_VISUAL_COLS + ["visual_eligibility_score"]:
                merged = col + "_vis"
                if merged in df.columns:
                    df[col] = df[merged].combine_first(df[col])
                    df.drop(columns=[merged], inplace=True)
        else:
            print("WARNING: visual_features.csv has no 'clip_id' column; skipping join", file=sys.stderr)

    # Fall back to manual proxy for eligibility when automatic not available
    no_auto_eligibility = df["visual_eligibility_score"].isna()
    df.loc[no_auto_eligibility, "visual_eligibility_score"] = (
        0.6 * df.loc[no_auto_eligibility, "manual_face_visibility_norm"].fillna(0)
        + 0.4 * df.loc[no_auto_eligibility, "manual_quality_norm"].fillna(0)
    )

    # ASD features (optional)
    for col in _ASD_COLS:
        df[col] = float("nan")

    if asd_features_csv and os.path.exists(asd_features_csv):
        asd_df = pd.read_csv(asd_features_csv)
        if "clip_id" in asd_df.columns:
            asd_df["clip_id"] = asd_df["clip_id"].astype(str)
            df = df.merge(
                asd_df[["clip_id"] + [c for c in _ASD_COLS if c in asd_df.columns]],
                on="clip_id",
                how="left",
                suffixes=("", "_asd"),
            )
            for col in _ASD_COLS:
                merged = col + "_asd"
                if merged in df.columns:
                    df[col] = df[merged].combine_first(df[col])
                    df.drop(columns=[merged], inplace=True)

    # GPT-4o features (optional, 007 extension)
    _GPT4O_COLS = [
        "child_visible_gpt4o", "child_vocalizing_gpt4o",
        "n_children_visible_mean", "visual_quality_gpt4o",
    ]
    if gpt4o_features_csv and os.path.exists(gpt4o_features_csv):
        gpt_df = pd.read_csv(gpt4o_features_csv)
        if "clip_id" in gpt_df.columns:
            gpt_df["clip_id"] = gpt_df["clip_id"].astype(str)
            merge_cols = ["clip_id"] + [c for c in _GPT4O_COLS if c in gpt_df.columns]
            df = df.merge(gpt_df[merge_cols], on="clip_id", how="left", suffixes=("", "_gpt"))
            for col in _GPT4O_COLS:
                if col + "_gpt" in df.columns:
                    df[col] = df[col + "_gpt"].combine_first(df.get(col, pd.Series(dtype=float)))
                    df.drop(columns=[col + "_gpt"], inplace=True)
            print(f"  Merged GPT-4o features from {gpt4o_features_csv}")
        else:
            print("WARNING: gpt4o_features.csv has no 'clip_id' column; skipping", file=sys.stderr)

    # Extra multi-model ASD features (optional, 007 extension)
    # extra_asd_csvs: list of (model_name, csv_path) tuples
    # Adds column asd_{model_name}_max_score from max_asd_score_target_candidate
    if extra_asd_csvs:
        for model_name, csv_path in extra_asd_csvs:
            if not csv_path or not os.path.exists(csv_path):
                continue
            asd_ext_df = pd.read_csv(csv_path)
            if "clip_id" not in asd_ext_df.columns:
                print(f"WARNING: {csv_path} has no 'clip_id'; skipping", file=sys.stderr)
                continue
            asd_ext_df["clip_id"] = asd_ext_df["clip_id"].astype(str)
            score_col = "max_asd_score_target_candidate"
            if score_col not in asd_ext_df.columns:
                score_col = "max_asd_score_smallest_face"  # legacy alias
            if score_col not in asd_ext_df.columns:
                continue
            new_col = f"asd_{model_name}_max_score"
            df = df.merge(
                asd_ext_df[["clip_id", score_col]].rename(columns={score_col: new_col}),
                on="clip_id", how="left",
            )
            print(f"  Merged ASD features for model '{model_name}' from {csv_path}")

    # Assert split integrity
    assert_split_integrity(df)

    df["run_name"] = run_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble AV master feature table from metadata, audio, and visual sources."
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--audio-scores-val", default=None,
                        help="BabAR enrollment val predictions CSV")
    parser.add_argument("--audio-scores-test", default=None,
                        help="BabAR enrollment test predictions CSV")
    parser.add_argument("--audio-score-col", default="prob")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-features-csv", default=None)
    parser.add_argument("--asd-features-csv", default=None)
    parser.add_argument("--gpt4o-features-csv", default=None,
                        help="Path to gpt4o_features.csv (007 extension); merged on clip_id")
    parser.add_argument("--asd-features-csv-extra", action="append", default=None,
                        metavar="MODEL:PATH",
                        help="Extra ASD model features: 'loconet:path/to/asd_features_loconet.csv'. "
                             "May be specified multiple times. Adds asd_{model}_max_score column.")
    parser.add_argument("--run-name", default="default")
    args = parser.parse_args()

    metadata_csv = _resolve_path(args.metadata_csv)
    output_dir = _resolve_path(args.output_dir)
    visual_csv = _resolve_path(args.visual_features_csv)
    asd_csv = _resolve_path(args.asd_features_csv)
    gpt4o_csv = _resolve_path(args.gpt4o_features_csv) if args.gpt4o_features_csv else None
    audio_val = _resolve_path(args.audio_scores_val)
    audio_test = _resolve_path(args.audio_scores_test)

    # Parse extra ASD CSVs: "model_name:csv_path"
    extra_asd: List[Tuple[str, str]] = []
    for entry in (args.asd_features_csv_extra or []):
        if ":" in entry:
            model_name, csv_path = entry.split(":", 1)
            extra_asd.append((model_name.strip(), _resolve_path(csv_path.strip())))
        else:
            print(f"WARNING: --asd-features-csv-extra '{entry}' must be in 'model:path' format; skipping",
                  file=sys.stderr)

    for path, name in [(metadata_csv, "metadata-csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}", file=sys.stderr)
            sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = build_table(
            metadata_csv=metadata_csv,
            audio_scores_val=audio_val,
            audio_scores_test=audio_test,
            audio_score_col=args.audio_score_col,
            visual_features_csv=visual_csv,
            asd_features_csv=asd_csv,
            gpt4o_features_csv=gpt4o_csv,
            extra_asd_csvs=extra_asd if extra_asd else None,
            run_name=args.run_name,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Write per-split CSVs
    df.to_csv(os.path.join(output_dir, "av_master_features.csv"), index=False)
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        split_df.to_csv(os.path.join(output_dir, f"av_{split}.csv"), index=False)
        print(f"  {split}: {len(split_df)} clips")

    # Feature manifest
    manual_cols = [c for c in _MANUAL_ANNOTATION_COLS if c in df.columns]
    derived_manual = ["child_of_interest_clear_binary", "manual_face_visibility_norm", "manual_quality_norm",
                      "n_people_total", "multi_person_clip", "age_band_binary"]
    auto_present = [c for c in _AUTO_VISUAL_COLS if not df[c].isna().all()]
    asd_present = [c for c in _ASD_COLS if not df[c].isna().all()]

    manifest = {
        "run_name": args.run_name,
        "n_clips": len(df),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "audio_score_available_val": int(df[df["split"] == "val"]["existing_audio_score"].notna().sum()),
        "audio_score_available_test": int(df[df["split"] == "test"]["existing_audio_score"].notna().sum()),
        "feature_sources": {
            "manual_annotations": manual_cols,
            "derived_manual": derived_manual,
            "automatic_visual": auto_present,
            "asd": asd_present,
            "audio": ["existing_audio_score"],
        },
    }
    save_json(manifest, os.path.join(output_dir, "feature_manifest.json"))

    # Split integrity report
    integrity = {
        "leakage_detected": False,
        "children_per_split": {
            s: int((df[df["split"] == s]["child_id"].nunique()))
            for s in ("train", "val", "test")
        },
    }
    save_json(integrity, os.path.join(output_dir, "split_integrity_report.json"))

    print(f"\nMaster feature table written to: {output_dir}")
    print(f"Total clips: {len(df)}")
    print(f"Auto visual features: {'present' if auto_present else 'NaN (not extracted yet)'}")
    print(f"ASD features: {'present' if asd_present else 'NaN (optional, not extracted)'}")
    print(f"Audio scores: val={manifest['audio_score_available_val']}, test={manifest['audio_score_available_test']}")


if __name__ == "__main__":
    main()
