"""
Augmentation evaluation: retrain enrollment prototypes with synthetic child speech
added to the training split, then evaluate on the same val/test split as baseline.

Reads registry.jsonl from --synthetic-dir to find generated WAV files.
Merges synthetic WAVs into training split at specified aug-ratio (synthetic:real).
Retrains ECAPA enrollment prototypes on augmented training data.
Evaluates on unchanged val/test splits.

Usage:
    python pyannote/augmentation_eval.py \\
        --diarizer babar \\
        --synthetic-dir synthesis/generated/vae_12m_v1 \\
        --age-group 12_16m --aug-ratio 1.0 --seed 42

Outputs canonical result structure under {output_dir}/{age_group}_ratio{aug_ratio}/:
    config.json, test_metrics_tuned.json, val_metrics_tuned.json,
    test_predictions.csv, val_predictions.csv, test_metrics_by_timepoint.csv
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    extract_segment_embeddings,
    l2_normalize,
    load_audio_mono,
    per_timepoint_metrics,
    run_enrollment,
    run_role_only,
    save_json,
    tune_role_only_threshold,
    tune_similarity_threshold,
)

TIMEPOINT_TO_AGE = {"14_month": "12_16m", "36_month": "34_38m"}


def _load_splits_with_age(split_dir: str):
    def _add_age(df):
        df = df.copy()
        df["age_group"] = df["timepoint_norm"].map(TIMEPOINT_TO_AGE).fillna("other")
        return df

    tr = _add_age(pd.read_csv(os.path.join(split_dir, "train.csv")))
    vl = _add_age(pd.read_csv(os.path.join(split_dir, "val.csv")))
    te = _add_age(pd.read_csv(os.path.join(split_dir, "test.csv")))
    return tr, vl, te


def _load_registry(synthetic_dir: str, age_group: str) -> list:
    registry_path = os.path.join(synthetic_dir, "registry.jsonl")
    if not os.path.exists(registry_path):
        # Try age_group subdir
        registry_path = os.path.join(synthetic_dir, age_group, "registry.jsonl")
    if not os.path.exists(registry_path):
        print(f"WARNING: No registry.jsonl found in {synthetic_dir}", file=sys.stderr)
        return []

    records = []
    with open(registry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return [r for r in records if r.get("age_group") == age_group]


def _build_augmented_prototypes(
    train_df: pd.DataFrame,
    synthetic_records: list,
    aug_ratio: float,
    frontend,
    embedder: ECAPAEmbedder,
    cfg: BaseConfig,
    age_group: str,
) -> dict:
    from collections import defaultdict
    import math

    # Real prototypes from training data
    real_protos, _ = build_child_prototypes(train_df, frontend, embedder, cfg)

    if not synthetic_records:
        print("No synthetic records — using real prototypes only.")
        return real_protos

    # For each child that has a real prototype, add synthetic embeddings
    # proportional to aug_ratio * number of real segments used
    pos_train = train_df[train_df["label"] == 1]
    n_real_per_child = pos_train.groupby("child_id").size().to_dict()

    # Embed all synthetic samples (shared across children by age group)
    syn_wavs_paths = [r["path"] for r in synthetic_records if os.path.exists(r["path"])]
    if not syn_wavs_paths:
        print("WARNING: No synthetic WAVs found on disk — using real prototypes only.")
        return real_protos

    print(f"Embedding {len(syn_wavs_paths)} synthetic samples for augmentation...")
    syn_embs = []
    for path in syn_wavs_paths[:500]:  # cap to avoid OOM
        try:
            wav = load_audio_mono(path, cfg.sample_rate)
            emb = embedder.embed_waveform(wav)
            syn_embs.append(emb)
        except Exception:
            continue

    if not syn_embs:
        return real_protos

    syn_embs_arr = np.stack(syn_embs)
    syn_mean = np.mean(syn_embs_arr, axis=0)

    augmented = {}
    for child_id, proto in real_protos.items():
        n_real = n_real_per_child.get(child_id, 1)
        n_syn = max(1, int(math.ceil(n_real * aug_ratio)))
        n_syn = min(n_syn, len(syn_embs))

        # Weighted average: real prototype + synthetic mean
        w_real = n_real
        w_syn = n_syn
        combined = (proto * w_real + syn_mean * w_syn) / (w_real + w_syn)
        augmented[child_id] = l2_normalize(combined)

    return augmented


def main():
    parser = argparse.ArgumentParser(description="Augmentation evaluation.")
    parser.add_argument("--diarizer", required=True,
                        choices=["usc_sail", "pyannote", "babar", "vtc", "vtc_kchi", "vbx"])
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--age-group", default="all",
                        choices=["all", "12_16m", "34_38m"])
    parser.add_argument("--aug-ratio", type=float, default=1.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", default="")
    parser.add_argument("--babar-dir", default="")
    parser.add_argument("--vtc-dir", default="")
    parser.add_argument("--vbx-dir", default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.synthetic_dir):
        print(f"ERROR: --synthetic-dir not found: {args.synthetic_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = BaseConfig()
    if args.splits_dir:
        cfg.split_dir = args.splits_dir
    if args.babar_dir:
        cfg.babar_dir = args.babar_dir
    if args.vtc_dir:
        cfg.vtc_dir = args.vtc_dir
    if args.vbx_dir:
        cfg.vbx_dir = args.vbx_dir

    here = Path(__file__).parent
    output_base = args.output_dir or str(here / f"{args.diarizer}_augmented")

    train_df, val_df, test_df = _load_splits_with_age(cfg.split_dir)
    all_audio = list({p for df in (train_df, val_df, test_df) for p in df["audio_path"]})

    frontend = build_frontend(args.diarizer, cfg)
    if isinstance(frontend, (BabARFrontend, VTCFrontend, VBxFrontend)):
        frontend.prepare(all_audio)

    embedder = ECAPAEmbedder(cfg.ecapa_source, cfg.device)

    age_groups = ["12_16m", "34_38m"] if args.age_group == "all" else [args.age_group]
    for ag in age_groups:
        tag = f"{ag}_ratio{args.aug_ratio}"
        out_dir = os.path.join(output_base, tag)
        os.makedirs(out_dir, exist_ok=True)

        meta = {**asdict(cfg), "aug_ratio": args.aug_ratio, "age_group": ag,
                "synthetic_dir": args.synthetic_dir, "seed": args.seed}
        save_json(meta, os.path.join(out_dir, "config.json"))

        tr = train_df[train_df["age_group"] == ag].copy()
        vl = val_df[val_df["age_group"] == ag].copy()
        te = test_df[test_df["age_group"] == ag].copy()
        print(f"\n[{ag}] Train={len(tr)} Val={len(vl)} Test={len(te)}")

        if len(tr) == 0 or len(vl) == 0 or len(te) == 0:
            print(f"[{ag}] WARNING: empty split — skipping")
            continue

        synthetic_records = _load_registry(args.synthetic_dir, ag)
        print(f"[{ag}] Synthetic records in registry: {len(synthetic_records)}")

        prototypes = _build_augmented_prototypes(
            tr, synthetic_records, args.aug_ratio, frontend, embedder, cfg, ag,
        )
        print(f"[{ag}] Augmented prototypes for {len(prototypes)} children.")

        val_enroll = run_enrollment(vl, prototypes, frontend, embedder, cfg)
        test_enroll = run_enrollment(te, prototypes, frontend, embedder, cfg)

        sim_t, val_m = tune_similarity_threshold(val_enroll, cfg)
        val_enroll = add_pred_labels(val_enroll, sim_t)
        test_enroll = add_pred_labels(test_enroll, sim_t)

        val_enroll.to_csv(os.path.join(out_dir, "val_predictions.csv"), index=False)
        test_enroll.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
        save_json({"threshold": sim_t, **val_m}, os.path.join(out_dir, "val_metrics_tuned.json"))

        test_m = compute_metrics(test_enroll["label"].to_numpy(),
                                  test_enroll["prob"].to_numpy(), threshold=sim_t)
        save_json({"threshold": sim_t, **test_m}, os.path.join(out_dir, "test_metrics_tuned.json"))
        per_timepoint_metrics(val_enroll, sim_t).to_csv(
            os.path.join(out_dir, "val_metrics_by_timepoint.csv"), index=False)
        per_timepoint_metrics(test_enroll, sim_t).to_csv(
            os.path.join(out_dir, "test_metrics_by_timepoint.csv"), index=False)

        print(f"[{ag}] threshold={sim_t:.3f}  test={test_m}")

    print("\nAugmentation eval done.")


if __name__ == "__main__":
    main()
