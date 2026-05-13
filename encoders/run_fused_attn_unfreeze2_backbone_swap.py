"""Run fused_attn_unfreeze2 with a swapped Whisper backbone (medium / large).

Fused = Whisper + WavLM-Base+ feature concat -> attention pool -> linear head,
with last-2 encoder layers of Whisper unfrozen. Default Whisper backbone is
small (244M); this script lets us swap in medium (764M) or large-v3 (1550M)
to substantiate the takeaway-#6 claim that scaling Whisper buys additional
AUROC.

Splits: standard seen-child split (whisper-modeling/seen_child_splits/).
Output: baseline_results_seen_child/fused_attn_unfreeze2_<size>/

Usage:
    python baselines/run_fused_attn_unfreeze2_backbone_swap.py --backbone medium
    python baselines/run_fused_attn_unfreeze2_backbone_swap.py --backbone large
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from baselines.baseline_encoders import CFG, load_seen_child_split, run_experiment


WHISPER_HF_BY_SIZE = {
    "small":  "openai/whisper-small",   # baseline (already in baseline_results_seen_child)
    "medium": "openai/whisper-medium",
    "large":  "openai/whisper-large-v3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=list(WHISPER_HF_BY_SIZE.keys()), required=True)
    parser.add_argument("--results-root", default="./baseline_results_seen_child")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: 1 for small/medium, 1 for large; bs=1 minimum because the fused stack is memory-heavy already)")
    args = parser.parse_args()

    size = args.backbone
    whisper_hf = WHISPER_HF_BY_SIZE[size]
    variant = f"fused_attn_unfreeze2_whisper_{size}"
    exp_dir = os.path.join(args.results_root, variant)

    base = replace(
        CFG,
        seen_child_splits=True,
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
        per_timepoint_threshold=True,
        save_path=os.path.join(exp_dir, "best_model.pt"),
    )

    train_df, val_df, test_df = load_seen_child_split(cfg)
    print(f"=== {variant} | whisper_backbone={whisper_hf} | seen_child split ===")
    print(f"Train rows: {len(train_df)} | children: {train_df['child_id'].nunique()}")
    print(f"Val rows:   {len(val_df)} | children: {val_df['child_id'].nunique()}")
    print(f"Test rows:  {len(test_df)} | children: {test_df['child_id'].nunique()}")

    os.makedirs(exp_dir, exist_ok=True)
    run_experiment(cfg, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
