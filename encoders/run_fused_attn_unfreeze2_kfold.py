"""Run fused_attn_unfreeze2 (Whisper+WavLM, attn pool, last-2 layers unfrozen)
on the within-child 3-fold splits used by the MIL k-fold.

Splits live under whisper-modeling/seen_child_splits_kfold_3fold/fold_{0,1,2}/
(same protocol as mil/configs/kfold_3fold/whisper_mil_fold*.yaml — 109 children
present in every train/val/test, clip-level k-fold).

Output: baseline_results_seen_child/fused_attn_unfreeze2{_<size>}_kfold3_f{FOLD}/
        — same schema as the headline single-split run.

Usage:
    FOLD=0 python baselines/run_fused_attn_unfreeze2_kfold.py [--backbone {small,medium,large}]
or via SLURM array (see baselines/slurm/run_fused_attn_unfreeze2_kfold.sh).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

# baseline_encoders.py exports CFG / Config / load_seen_child_split / run_experiment
from baselines.baseline_encoders import CFG, load_seen_child_split, run_experiment


WHISPER_HF_BY_SIZE = {
    "small":  "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large":  "openai/whisper-large-v3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=int(os.environ.get("FOLD", 0)),
                        help="k-fold index (0/1/2)")
    parser.add_argument("--backbone", choices=list(WHISPER_HF_BY_SIZE.keys()),
                        default="small",
                        help="Whisper backbone size (small / medium / large-v3).")
    parser.add_argument("--results-root", default="./baseline_results_seen_child")
    args = parser.parse_args()

    fold = args.fold
    size = args.backbone
    whisper_hf = WHISPER_HF_BY_SIZE[size]
    split_subdir = f"whisper-modeling/seen_child_splits_kfold_3fold/fold_{fold}"
    suffix = "" if size == "small" else f"_whisper_{size}"
    variant = f"fused_attn_unfreeze2{suffix}_kfold3_f{fold}"
    exp_dir = os.path.join(args.results_root, variant)

    base = replace(
        CFG,
        seen_child_splits=True,
        seen_child_split_dir=split_subdir,
        results_root=args.results_root,
        whisper_name=whisper_hf,
    )

    cfg = replace(
        base,
        experiment_name=variant,
        model_type="fused",
        pooling="attn",
        use_layer_weights=False,
        unfreeze_last_n_layers=2,
        batch_size=1,
        num_workers=2,
        per_timepoint_threshold=True,
        save_path=os.path.join(exp_dir, "best_model.pt"),
    )

    train_df, val_df, test_df = load_seen_child_split(cfg)
    print(f"=== {variant} | whisper_backbone={whisper_hf} | split={split_subdir} ===")
    print(f"Train rows: {len(train_df)} | children: {train_df['child_id'].nunique()}")
    print(f"Val rows:   {len(val_df)} | children: {val_df['child_id'].nunique()}")
    print(f"Test rows:  {len(test_df)} | children: {test_df['child_id'].nunique()}")

    os.makedirs(exp_dir, exist_ok=True)
    run_experiment(cfg, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
