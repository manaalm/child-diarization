#!/usr/bin/env python3
"""
Generate synthetic child-adult audio scenes.

Usage example (smoke test):
    python synth/scripts/generate_scenes.py \\
        --config  synth/configs/default_14_18mo.yaml \\
        --manifest synth_results/manifests/segment_manifest.csv \\
        --n-scenes 50 \\
        --output-dir synth_results/synthetic_scenes/

Full run (5000 scenes):
    python synth/scripts/generate_scenes.py \\
        --config  synth/configs/default_14_18mo.yaml \\
        --manifest synth_results/manifests/segment_manifest.csv \\
        --output-dir synth_results/synthetic_scenes/
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synth.manifest import load_manifest
from synth.labels import write_clip_labels_row
from synth.scene_generator import SceneComposer


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate scene-type probability sum
    sampling = cfg.get("sampling", {})
    keys = [
        "positive_scene_probability",
        "adult_only_negative_probability",
        "background_speech_negative_probability",
        "noise_only_negative_probability",
    ]
    total = sum(float(sampling.get(k, 0.0)) for k in keys)
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Scene-type probabilities must sum to 1.0, got {total:.4f} "
            f"in {config_path}"
        )

    sr = cfg.get("project", {}).get("sample_rate", 16000)
    if int(sr) != 16000:
        raise ValueError(f"sample_rate must be 16000, got {sr}")

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic child-adult audio scenes."
    )
    parser.add_argument(
        "--config", required=True, help="Path to scene config YAML."
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to segment manifest CSV."
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=None,
        help="Number of scenes to generate (overrides config value).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for wav/, rttm/, json/ sub-dirs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed (overrides config value).",
    )
    parser.add_argument(
        "--rir-dir",
        default=None,
        help="Override mixing.rir_dir in config: path to directory of RIR WAV/FLAC files.",
    )
    parser.add_argument(
        "--noise-dir",
        default=None,
        help="Override mixing.noise_dir in config: path to directory of noise WAV files.",
    )
    args = parser.parse_args()

    # --- Load config and manifest ---
    cfg = _load_config(args.config)

    # Apply CLI overrides for acoustic augmentation paths
    if args.rir_dir is not None:
        cfg.setdefault("mixing", {})["rir_dir"] = args.rir_dir
    if args.noise_dir is not None:
        cfg.setdefault("mixing", {})["noise_dir"] = args.noise_dir

    n_scenes: int = int(
        args.n_scenes
        if args.n_scenes is not None
        else cfg["scene"]["n_scenes"]
    )
    global_seed: int = int(
        args.seed
        if args.seed is not None
        else cfg["project"].get("random_seed", 42)
    )
    config_name: str = str(cfg["project"]["name"])

    print(f"Loading manifest: {args.manifest}")
    manifest_df = load_manifest(args.manifest)
    print(f"  {len(manifest_df)} total rows; "
          f"{manifest_df['usable_for_training'].sum()} usable for training.")

    # --- Set up SceneComposer ---
    composer = SceneComposer(cfg, manifest_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest output: one level up from wav/rttm/json, in manifests/
    manifest_out_dir = output_dir.parent / "manifests"
    manifest_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = manifest_out_dir / "synthetic_manifest.csv"

    # Collect clip-label rows
    clip_rows: list = []

    print(
        f"Generating {n_scenes} scenes "
        f"(config={config_name}, seed={global_seed}) → {output_dir}"
    )

    for i in range(n_scenes):
        scene_id = f"{config_name}_{global_seed}_{i:06d}"

        # Skip if WAV already exists (idempotent)
        wav_path = output_dir / "wav" / f"{scene_id}.wav"
        if wav_path.exists():
            # Still need to collect the clip-label row for the manifest
            rttm_path = output_dir / "rttm" / f"{scene_id}.rttm"
            # Re-read from existing JSON if available
            json_path = output_dir / "json" / f"{scene_id}.json"
            if json_path.exists():
                import json as _json
                with open(json_path) as f:
                    existing_meta = _json.load(f)
                existing_meta["audio_path"] = str(wav_path.resolve())
                existing_meta["rttm_path"] = str(rttm_path.resolve())
                # Reconstruct tracks for clip-labels row
                tracks = []
                for ss in existing_meta.get("source_segments", []):
                    tracks.append({
                        "speaker_label": ss["speaker_label"],
                        "start_sec": ss["start_sec"],
                        "end_sec": ss["end_sec"],
                    })
                existing_meta["tracks"] = tracks
                existing_meta.setdefault("snr_db", existing_meta.get("mean_snr_db"))
                existing_meta.setdefault("noise_type", "")
                existing_meta.setdefault("rir_type", "")
                existing_meta.setdefault("age_band", existing_meta.get("target_age_band", ""))
                clip_rows.append(write_clip_labels_row(existing_meta))
            if (i + 1) % 100 == 0 or i == n_scenes - 1:
                print(f"  [{i + 1}/{n_scenes}] (skipped existing)")
            continue

        per_scene_rng = np.random.default_rng(global_seed + i)
        scene_meta = composer.compose(scene_id, per_scene_rng)
        scene_meta["random_seed"] = global_seed + i

        composer.write(scene_meta, str(output_dir))
        clip_rows.append(write_clip_labels_row(scene_meta))

        if (i + 1) % 100 == 0 or i == n_scenes - 1:
            print(f"  [{i + 1}/{n_scenes}] scene_type={scene_meta['scene_type']}")

    # --- Write synthetic_manifest.csv ---
    if clip_rows:
        fieldnames = list(clip_rows[0].keys())
        write_header = not manifest_csv_path.exists() or len(clip_rows) == n_scenes
        mode = "w" if write_header else "a"
        with open(manifest_csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerows(clip_rows)
        print(f"Synthetic manifest written: {manifest_csv_path} ({len(clip_rows)} rows)")
    else:
        print("No new scenes generated.")


if __name__ == "__main__":
    main()
