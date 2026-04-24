"""Zero-shot ASD evaluation on Ego4D Active Video Dataset (AVD).

Documents the Ego4D integration pathway and, if data is accessible,
evaluates ASD models zero-shot on the Ego4D AVD subset before applying
them to child home video. Results are written as Ego4DExperimentRecord rows.

If --ego4d-metadata-csv does not exist, the script exits cleanly with a
written documentation record (not an error). This is the expected outcome
when Ego4D access has not been granted.

Access pathway (for future reference):
  1. Register at https://ego4d-data.org (48h approval)
  2. pip install ego4d
  3. ego4d --output_directory /path/to/ego4d/ --datasets full_scale --benchmarks AV
  4. ASD annotations: ego4d/v2/annotations/av_val.json (or av_train.json)
  5. Provide --ego4d-metadata-csv pointing to a CSV derived from av_val.json
     with columns: clip_id, audio_path, video_path, label (1=active, 0=not)

Usage:
    python av_fusion/scripts/ego4d_experiment.py \\
        --ego4d-metadata-csv /path/to/ego4d/av_val_metadata.csv \\
        --output             av_fusion/av_results/run1/ego4d_experiment_results.csv \\
        --asd-model          talknet \\
        --n-clips            50
"""

import argparse
import os
import subprocess
import sys
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root, save_json

_REPO = get_repo_root()

_ACCESS_NOTES = (
    "Ego4D data access requires registration at https://ego4d-data.org (typically 48h approval). "
    "After approval: (1) pip install ego4d; "
    "(2) ego4d --output_directory /path/to/ego4d/ --datasets full_scale --benchmarks AV; "
    "(3) ASD annotations at ego4d/v2/annotations/av_val.json. "
    "The AVD benchmark contains ~50h of annotated egocentric video with active speaker labels. "
    "For this project, Ego4D is used as a zero-shot evaluation domain to measure the gap "
    "between broadcast-trained ASD models and naturalistic home video conditions."
)


def _write_not_found_record(output: str, asd_model: str, notes: str) -> None:
    row = {
        "experiment_id": f"ego4d_{asd_model}_zeroshot",
        "asd_model": asd_model,
        "adaptation_type": "not_run",
        "ego4d_subset": "avd_val",
        "val_auroc_home_video": float("nan"),
        "baseline_auroc": float("nan"),
        "delta_auroc": float("nan"),
        "notes": notes,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(f"Ego4D experiment record written to: {output}")
    print(f"  Status: not_run — {notes[:80]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot ASD evaluation on Ego4D AVD subset (or document access pathway)."
    )
    parser.add_argument("--ego4d-metadata-csv", default=None,
                        help="CSV with Ego4D clip metadata (clip_id, audio_path, video_path, label)")
    parser.add_argument("--output", required=True,
                        help="Output path for ego4d_experiment_results.csv")
    parser.add_argument("--asd-model", default="talknet",
                        choices=["talknet", "loconet", "light_asd"],
                        help="ASD model to evaluate (default: talknet)")
    parser.add_argument("--n-clips", type=int, default=50,
                        help="Number of Ego4D clips to evaluate (default: 50)")
    parser.add_argument("--run-name", default="ego4d_eval",
                        help="Run name for output organization")
    args = parser.parse_args()

    output = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)

    # Case 1: No metadata CSV provided or path doesn't exist → document the pathway
    if args.ego4d_metadata_csv is None:
        _write_not_found_record(
            output, args.asd_model,
            "No --ego4d-metadata-csv provided. " + _ACCESS_NOTES
        )
        return

    meta_path = (args.ego4d_metadata_csv if os.path.isabs(args.ego4d_metadata_csv)
                 else os.path.join(_REPO, args.ego4d_metadata_csv))

    if not os.path.exists(meta_path):
        _write_not_found_record(
            output, args.asd_model,
            f"Ego4D metadata CSV not found at {meta_path}. " + _ACCESS_NOTES
        )
        return

    # Case 2: Data available — run zero-shot evaluation
    print(f"Loading Ego4D metadata from {meta_path}...")
    ego4d_df = pd.read_csv(meta_path, low_memory=False)

    required_cols = {"clip_id", "audio_path", "label"}
    missing = required_cols - set(ego4d_df.columns)
    if missing:
        _write_not_found_record(
            output, args.asd_model,
            f"Ego4D metadata CSV missing required columns: {missing}. "
            f"Available columns: {list(ego4d_df.columns)}"
        )
        return

    # Sample subset
    eval_df = ego4d_df.sample(min(args.n_clips, len(ego4d_df)), random_state=42)
    print(f"Evaluating {len(eval_df)} Ego4D clips with {args.asd_model}...")

    # Extract ASD features via existing script
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_meta = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_out = f.name

    try:
        eval_df.to_csv(tmp_meta, index=False)
        cmd = [
            sys.executable,
            os.path.join(_REPO, "av_fusion", "scripts", "extract_asd_features.py"),
            "--metadata-csv", tmp_meta,
            "--output", tmp_out,
            "--model", args.asd_model,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            _write_not_found_record(
                output, args.asd_model,
                f"ASD extraction failed (exit {result.returncode}): {result.stderr[:200]}"
            )
            return

        asd_df = pd.read_csv(tmp_out)
    finally:
        for p in [tmp_meta, tmp_out]:
            if os.path.exists(p):
                os.unlink(p)

    # Compute AUROC
    merged = eval_df[["clip_id", "label"]].merge(asd_df[["clip_id", "max_asd_score_target_candidate"]],
                                                   on="clip_id", how="left")
    y_true = merged["label"].values
    y_score = merged["max_asd_score_target_candidate"].fillna(0.0).values

    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        auroc = float("nan")
        print(f"  WARNING: could not compute AUROC: {e}", file=sys.stderr)

    print(f"  Ego4D zero-shot AUROC ({args.asd_model}): {auroc:.4f}")

    row = {
        "experiment_id": f"ego4d_{args.asd_model}_zeroshot",
        "asd_model": args.asd_model,
        "adaptation_type": "zero_shot",
        "ego4d_subset": f"avd_val_n{len(eval_df)}",
        "val_auroc_home_video": float("nan"),  # not tested on child video in this run
        "baseline_auroc": auroc,
        "delta_auroc": float("nan"),
        "notes": f"Zero-shot evaluation on {len(eval_df)} Ego4D AVD clips. "
                 f"baseline_auroc = ASD model AUROC against Ego4D active speaker labels.",
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(f"Ego4D experiment results written to: {output}")


if __name__ == "__main__":
    main()
