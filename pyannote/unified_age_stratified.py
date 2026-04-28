"""
Age-stratified enrollment evaluation for child vocalization detection.

Wraps unified.py's enrollment pipeline with per-age-group filtering.
Age groups are derived from the seen_child_splits timepoint_norm column:
  14_month → 12_16m
  36_month → 34_38m

Usage:
    python pyannote/unified_age_stratified.py --diarizer babar --age-group 12_16m
    python pyannote/unified_age_stratified.py --diarizer usc_sail --age-group all
    python pyannote/unified_age_stratified.py --diarizer vbx --age-group all

Outputs per age_group under {output_dir}/{age_group}/:
    config.json, test_metrics_tuned.json, val_metrics_tuned.json,
    test_predictions.csv, val_predictions.csv, test_metrics_by_timepoint.csv,
    val_metrics_by_timepoint.csv, child_prototype_stats.csv,
    role_only_*.{json,csv}
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from unified import (
    BaseConfig,
    BabARFrontend,
    ECAPAEmbedder,
    VBxFrontend,
    VTCFrontend,
    add_pred_labels,
    build_child_prototypes,
    build_frontend,
    compute_metrics,
    per_timepoint_metrics,
    role_df_to_pred_df,
    run_enrollment,
    run_role_only,
    save_json,
    tune_role_only_threshold,
    tune_similarity_threshold,
)

TIMEPOINT_TO_AGE = {
    "14_month": "12_16m",
    "36_month": "34_38m",
}
AGE_GROUPS = ["12_16m", "34_38m"]


def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = out["timepoint_norm"].map(TIMEPOINT_TO_AGE).fillna("other")
    return out


def _load_splits(split_dir: str):
    tr = _add_age_group(pd.read_csv(os.path.join(split_dir, "train.csv")))
    vl = _add_age_group(pd.read_csv(os.path.join(split_dir, "val.csv")))
    te = _add_age_group(pd.read_csv(os.path.join(split_dir, "test.csv")))
    return tr, vl, te


def _run_one_age_group(
    train_df, val_df, test_df,
    age_group, frontend, embedder, cfg, output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    cfg_copy = BaseConfig(**asdict(cfg))
    save_json(asdict(cfg_copy), os.path.join(output_dir, "config.json"))

    tr = train_df[train_df["age_group"] == age_group].copy()
    vl = val_df[val_df["age_group"] == age_group].copy()
    te = test_df[test_df["age_group"] == age_group].copy()

    print(f"\n[{age_group}] Train={len(tr)}  Val={len(vl)}  Test={len(te)}")
    if len(tr) == 0 or len(vl) == 0 or len(te) == 0:
        print(f"[{age_group}] WARNING: empty split — skipping")
        return

    # Role-only baseline
    val_role = run_role_only(vl, frontend, cfg)
    test_role = run_role_only(te, frontend, cfg)
    role_t, role_val_m = tune_role_only_threshold(val_role, cfg)
    val_role_pred = role_df_to_pred_df(val_role, role_t)
    test_role_pred = role_df_to_pred_df(test_role, role_t)
    val_role_pred.to_csv(os.path.join(output_dir, "role_only_val_predictions.csv"), index=False)
    test_role_pred.to_csv(os.path.join(output_dir, "role_only_test_predictions.csv"), index=False)
    save_json({"threshold_sec": role_t, **role_val_m},
              os.path.join(output_dir, "role_only_val_metrics_tuned.json"))
    role_test_m = compute_metrics(
        test_role_pred["label"].to_numpy(),
        test_role_pred["prob"].to_numpy(),
        threshold=role_t,
    )
    save_json({"threshold_sec": role_t, **role_test_m},
              os.path.join(output_dir, "role_only_test_metrics_tuned.json"))
    per_timepoint_metrics(val_role_pred, role_t).to_csv(
        os.path.join(output_dir, "role_only_val_metrics_by_timepoint.csv"), index=False)
    per_timepoint_metrics(test_role_pred, role_t).to_csv(
        os.path.join(output_dir, "role_only_test_metrics_by_timepoint.csv"), index=False)

    # Enrollment
    prototypes, child_stats = build_child_prototypes(tr, frontend, embedder, cfg)
    child_stats.to_csv(os.path.join(output_dir, "child_prototype_stats.csv"), index=False)
    print(f"[{age_group}] Built prototypes for {len(prototypes)} children.")

    val_enroll = run_enrollment(vl, prototypes, frontend, embedder, cfg)
    test_enroll = run_enrollment(te, prototypes, frontend, embedder, cfg)

    sim_t, val_sim_m = tune_similarity_threshold(val_enroll, cfg)
    val_enroll = add_pred_labels(val_enroll, sim_t)
    test_enroll = add_pred_labels(test_enroll, sim_t)

    val_enroll.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
    test_enroll.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    save_json({"threshold": sim_t, **val_sim_m},
              os.path.join(output_dir, "val_metrics_tuned.json"))

    test_sim_m = compute_metrics(
        test_enroll["label"].to_numpy(),
        test_enroll["prob"].to_numpy(),
        threshold=sim_t,
    )
    save_json({"threshold": sim_t, **test_sim_m},
              os.path.join(output_dir, "test_metrics_tuned.json"))
    per_timepoint_metrics(val_enroll, sim_t).to_csv(
        os.path.join(output_dir, "val_metrics_by_timepoint.csv"), index=False)
    per_timepoint_metrics(test_enroll, sim_t).to_csv(
        os.path.join(output_dir, "test_metrics_by_timepoint.csv"), index=False)

    print(f"[{age_group}] threshold={sim_t:.3f}  test={test_sim_m}")


def main():
    parser = argparse.ArgumentParser(description="Age-stratified enrollment evaluation.")
    parser.add_argument("--diarizer", required=True,
                        choices=["usc_sail", "pyannote", "babar", "vtc", "vtc_kchi", "vbx"])
    parser.add_argument("--age-group", default="all",
                        choices=["all", "12_16m", "34_38m"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--splits-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--babar-dir", default="")
    parser.add_argument("--babar-batch-size", type=int, default=32)
    parser.add_argument("--vtc-dir", default="")
    parser.add_argument("--vtc-batch-size", type=int, default=64)
    parser.add_argument("--vbx-dir", default="")
    parser.add_argument("--vbx-max-speakers", type=int, default=8)
    parser.add_argument("--vbx-niters", type=int, default=10)
    parser.add_argument("--vbx-Fa", type=float, default=0.1)
    parser.add_argument("--vbx-Fb", type=float, default=17.0)
    parser.add_argument("--vbx-loopP", type=float, default=0.99)
    parser.add_argument("--vbx-win-duration", type=float, default=1.5)
    parser.add_argument("--vbx-win-step", type=float, default=0.25)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg = BaseConfig()
    if args.splits_dir:
        cfg.split_dir = args.splits_dir
    if args.babar_dir:
        cfg.babar_dir = args.babar_dir
    cfg.babar_batch_size = args.babar_batch_size
    if args.vtc_dir:
        cfg.vtc_dir = args.vtc_dir
    cfg.vtc_batch_size = args.vtc_batch_size
    if args.vbx_dir:
        cfg.vbx_dir = args.vbx_dir
    cfg.vbx_max_speakers = args.vbx_max_speakers
    cfg.vbx_niters = args.vbx_niters
    cfg.vbx_Fa = args.vbx_Fa
    cfg.vbx_Fb = args.vbx_Fb
    cfg.vbx_loopP = args.vbx_loopP
    cfg.vbx_win_duration = args.vbx_win_duration
    cfg.vbx_win_step = args.vbx_win_step

    here = Path(__file__).parent
    output_base = args.output_dir or str(here / f"{args.diarizer}_age_stratified")

    train_df, val_df, test_df = _load_splits(cfg.split_dir)
    all_audio = list({
        p for df in (train_df, val_df, test_df) for p in df["audio_path"]
    })
    print(f"Diarizer: {args.diarizer}  Age filter: {args.age_group}")
    print(f"Total audio files: {len(all_audio)}")

    frontend = build_frontend(args.diarizer, cfg)
    if isinstance(frontend, (BabARFrontend, VTCFrontend, VBxFrontend)):
        frontend.prepare(all_audio)

    embedder = ECAPAEmbedder(cfg.ecapa_source, cfg.device)

    groups = AGE_GROUPS if args.age_group == "all" else [args.age_group]
    for ag in groups:
        _run_one_age_group(
            train_df, val_df, test_df, ag,
            frontend, embedder, cfg,
            os.path.join(output_base, ag),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
