"""Run fused_attn_unfreeze2 with a swapped Whisper backbone on the
BIDS-corrected cross-child split.

Parallel to ``run_fused_attn_unfreeze2_backbone_swap.py`` (which is
seen-child only) — this one points the loader at ``baselines/splits/``
so the medium and large backbone runs can fill the missing
``Whisper+WavLM fused (medium/large, PU last 2)'' rows in
Tab.~\\ref{tab:headline-cross} BIDS-corrected single-split column.

Usage:
    python encoders/run_fused_unfreeze2_backbone_swap_cross_child.py --backbone medium
    python encoders/run_fused_unfreeze2_backbone_swap_cross_child.py --backbone large
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from encoders.baseline_encoders import CFG, load_or_create_split, run_experiment


WHISPER_HF_BY_SIZE = {
    "medium": "openai/whisper-medium",
    "large":  "openai/whisper-large-v3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=list(WHISPER_HF_BY_SIZE.keys()), required=True)
    parser.add_argument("--results-root", default="./baselines/baseline_results_cross_child_bids")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default 1; fused stack is memory-heavy)")
    args = parser.parse_args()

    size = args.backbone
    whisper_hf = WHISPER_HF_BY_SIZE[size]
    variant = f"fused_attn_unfreeze2_whisper_{size}"
    exp_dir = os.path.join(args.results_root, variant)

    base = replace(
        CFG,
        seen_child_splits=False,       # cross-child path: loads baselines/splits/
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
        batch_size=args.batch_size if args.batch_size else 1,
        num_workers=2,
        per_timepoint_threshold=False,  # cross-child has very small per-tp sets; global only
        save_path=os.path.join(exp_dir, "best_model.pt"),
    )

    train_df, val_df, test_df = load_or_create_split(cfg)
    print(f"=== {variant} | whisper_backbone={whisper_hf} | BIDS cross-child split ===")
    print(f"Train rows: {len(train_df)} | children: {train_df['child_id'].nunique()}")
    print(f"Val rows:   {len(val_df)} | children: {val_df['child_id'].nunique()}")
    print(f"Test rows:  {len(test_df)} | children: {test_df['child_id'].nunique()}")

    os.makedirs(exp_dir, exist_ok=True)
    run_experiment(cfg, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
