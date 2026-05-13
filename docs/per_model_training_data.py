"""Introspect every system's saved config.json and emit a per-model training-data
CSV (spec 022 US4 / FR-020).

Output schema (matches contracts/per_model_training_data.schema.md):
    system_name, system_family, train_split, train_children, train_clip_count,
    includes_synthetic, synth_corpus_version, synth_clip_count,
    pretrained_backbone, backbone_frozen, eval_splits,
    config_path, result_dir

Best-effort extraction — many configs are sparse; missing fields default to
empty/NaN. The script never throws on a malformed config; it logs and skips
the row.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"

CANONICAL_ROOTS = [
    ("mil/mil_results", "mil_frame_window"),
    ("pseudo_frame/results", "pseudo_frame"),
    ("baselines/audio_llm_baseline_runs", "audio_llm"),
    ("baselines/scene_analysis_runs", "audio_scene_analysis"),
    ("baselines/baseline_results", "encoder_baseline"),
    ("whisper-modeling/usc_sail_enrollment_runs", "usc_sail_whisper"),
    ("babar_ecapa_enrollment_runs", "pyannote_family"),
    ("babar_combined_runs", "pyannote_family"),
    ("vtc_ecapa_enrollment_runs", "pyannote_family"),
    ("vtc_kchi_ecapa_enrollment_runs", "pyannote_family"),
    ("vbx_ecapa_enrollment_runs", "pyannote_family"),
    ("pyannote/pyannote_enrollment_runs", "pyannote_family"),
    ("eend_eda_ecapa_enrollment_runs", "pyannote_family"),
    ("sortformer_ecapa_enrollment_runs", "pyannote_family"),
    ("ensemble_runs", "ensemble"),
    ("joint_asr_diar_ecapa_enrollment_runs", "audio_llm"),
]

SYNTH_VERSION_PATTERNS = {
    "v4_hardneg": [r"hardneg_synth_v4"],
    "v4_cross_child": [r"cross_child_synth_v4"],
    "v4": [r"_synth_v4$"],
    "v3_perturb": [r"perturb", r"_synth_v3$"],
    "v2": [r"_synth$", r"_synth_v2$", r"_synth_aug"],
    "v1": [r"_synth_v1$"],
}


def _detect_synth_version(system_name: str, config: dict) -> str | None:
    """Match known synth-corpus tags against the system slug / config strings."""
    haystack = system_name + " " + json.dumps(config)
    for version, patterns in SYNTH_VERSION_PATTERNS.items():
        if any(re.search(p, haystack) for p in patterns):
            return version
    return None


def _extract_train_clip_count(config: dict, result_dir: str) -> int:
    """Best-effort: look for train CSV path in config and count rows, or use
    a `n_train_clips` field if present, else 0."""
    for k in ("n_train_clips", "train_n", "train_size", "train_clip_count"):
        if k in config and isinstance(config[k], (int, float)):
            return int(config[k])
    # Look for a train CSV path
    for k in ("train_csv", "train_split_csv", "data.train_csv"):
        path = config.get(k)
        if isinstance(path, str) and os.path.exists(path):
            try:
                with open(path) as f:
                    return sum(1 for _ in f) - 1
            except Exception:
                pass
    return 0


def _extract_train_children(config: dict) -> int:
    for k in ("n_train_children", "train_children", "n_children_train"):
        if k in config and isinstance(config[k], (int, float)):
            return int(config[k])
    return 0


def _extract_backbone(config: dict, family: str, system_name: str) -> tuple[str, bool]:
    """Return (backbone_id, backbone_frozen)."""
    for k in ("backbone_name", "encoder_name", "backbone", "pretrained_backbone", "model", "model_name"):
        if k in config and isinstance(config[k], str):
            backbone = config[k]
            frozen = config.get("backbone_frozen", config.get("freeze_backbone", True))
            return backbone, bool(frozen)
    # Family-based defaults
    family_defaults = {
        "mil_frame_window": ("openai/whisper-small or microsoft/wavlm-base-plus", True),
        "pseudo_frame": ("openai/whisper-small or microsoft/wavlm-base-plus", True),
        "audio_llm": ("Qwen/Qwen2.5-Omni-7B", True),
        "audio_scene_analysis": ("AudioSet pretrained (YAMNet or AST)", True),
        "encoder_baseline": ("openai/whisper-small or microsoft/wavlm-base-plus", True),
        "usc_sail_whisper": ("openai/whisper-base + LoRA", False),
        "pyannote_family": ("frozen-ecapa", True),
        "ensemble": ("composite of audio-MIL", True),
    }
    return family_defaults.get(family, ("unknown", True))


def _detect_train_split(system_name: str, config: dict) -> str:
    sn = system_name.lower()
    if "audio_llm" in sn or "scene_analysis" in sn or "qwen" in sn or "yamnet" in sn or sn.endswith("/ast"):
        return "zero_shot"
    if "cross_child" in sn:
        return "cross_child_train"
    if "_synth" in sn:
        sv = _detect_synth_version(system_name, config)
        return f"synth_train_{sv}" if sv else "synth_train_unknown"
    if "kfold3" in sn:
        return "seen_child_kfold3_legacy"
    if "groupstrat" in sn:
        return "seen_child_groupstrat"
    return "seen_child_train"


def _detect_eval_splits(result_dir: str) -> list[str]:
    splits = []
    if os.path.exists(os.path.join(result_dir, "test_metrics_tuned.json")) or \
       os.path.exists(os.path.join(result_dir, "test_predictions.csv")) or \
       os.path.exists(os.path.join(result_dir, "enroll_test_predictions.csv")):
        splits.append("seen_child_test")
    if os.path.exists(os.path.join(result_dir, "test_all_predictions.csv")):
        splits.append("all_children_coverage")
    if not splits:
        splits.append("unknown")
    return splits


def _process_one(config_path: str, family_hint: str) -> dict | None:
    result_dir = os.path.dirname(config_path)
    system_name = os.path.relpath(result_dir, REPO_ROOT)
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"  [skip] {config_path}: {e}", file=sys.stderr)
        return None
    if not isinstance(config, dict):
        return None

    synth_version = _detect_synth_version(system_name, config)
    train_clip_count = _extract_train_clip_count(config, result_dir)
    backbone, frozen = _extract_backbone(config, family_hint, system_name)
    train_split = _detect_train_split(system_name, config)
    eval_splits = _detect_eval_splits(result_dir)

    return {
        "system_name": system_name,
        "system_family": family_hint,
        "train_split": train_split,
        "train_children": _extract_train_children(config),
        "train_clip_count": train_clip_count,
        "includes_synthetic": bool(synth_version),
        "synth_corpus_version": synth_version or "",
        "synth_clip_count": int(config.get("n_synth_clips", config.get("synth_clip_count", 0))),
        "pretrained_backbone": backbone,
        "backbone_frozen": frozen,
        "eval_splits": json.dumps(eval_splits),
        "config_path": config_path,
        "result_dir": result_dir,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "docs", "per_model_training_data.csv"))
    args = ap.parse_args()

    rows = []
    for stub, family in CANONICAL_ROOTS:
        root = os.path.join(REPO_ROOT, stub)
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            cfg = os.path.join(d, "config.json")
            if not os.path.exists(cfg):
                continue
            r = _process_one(cfg, family)
            if r is not None:
                rows.append(r)

    if not rows:
        print("no rows produced; check CANONICAL_ROOTS exist on disk", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    family_counts = {}
    for r in rows:
        family_counts[r["system_family"]] = family_counts.get(r["system_family"], 0) + 1
    print("by family:", family_counts)


if __name__ == "__main__":
    main()
