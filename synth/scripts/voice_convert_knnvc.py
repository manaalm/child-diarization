"""Bulk KNN-VC voice conversion: per training-child, convert N positive synth scenes.

T130 (spec-017 US2). For each training child with at least one positive clip,
collects ~30 s of that child's positive clips as the WavLM matching set, then
converts N randomly-sampled positive synth scenes to the child's voice. Writes
per-child sub-directories under synth_results/voice_converted/ and a manifest
CSV that the WavLM-MIL pipeline can ingest as additional positives.

Children come from the seen-child training split. Synth scenes come from the v2
corpus (synth_results/synthetic_scenes_v2/) since that's the source mix the
spec-016 v2 rerun used.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/train.csv")
SYNTH_V2_JSON_DIR = os.path.join(_REPO, "synth_results/synthetic_scenes_v2/json")
OUT_AUDIO_DIR = os.path.join(_REPO, "synth_results/voice_converted")
OUT_MANIFEST = os.path.join(_REPO, "synth_results/manifests/synthetic_voice_converted.csv")


def positive_synth_scenes() -> List[Dict]:
    """Return list of v2 positive scenes as dicts: {scene_id, audio_path}."""
    out = []
    for jp in sorted(glob.glob(os.path.join(SYNTH_V2_JSON_DIR, "*.json"))):
        d = json.load(open(jp))
        if d.get("target_child_vocalized"):
            out.append({"scene_id": d["synthetic_scene_id"], "audio_path": d["audio_path"]})
    return out


def child_reference_clips(df: pd.DataFrame, child_id: str, max_dur_sec: float = 30.0) -> List[str]:
    rows = df[(df["child_id"] == child_id) & (df["label"] == 1)]
    paths: List[str] = []
    cum = 0.0
    for _, row in rows.iterrows():
        ap = row.get("audio_path")
        if not isinstance(ap, str) or not os.path.exists(ap):
            continue
        try:
            paths.append(ap)
            cum += sf.info(ap).duration
            if cum >= max_dur_sec:
                break
        except Exception:
            continue
    return paths


def md5short(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-child", type=int, default=10)
    ap.add_argument("--children-csv", type=str, default=TRAIN_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-child-idx", type=int, default=0)
    ap.add_argument("--end-child-idx", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bulk] device={device} n_per_child={args.n_per_child} seed={args.seed}")

    df = pd.read_csv(args.children_csv)
    pos_counts = df[df["label"] == 1].groupby("child_id").size().sort_values(ascending=False)
    children = list(pos_counts.index)
    print(f"[bulk] {len(children)} candidate training children with >=1 positive clip")
    children = children[args.start_child_idx:args.end_child_idx]

    scenes = positive_synth_scenes()
    print(f"[bulk] {len(scenes)} v2 positive synth scenes available")

    # Load model
    print("[bulk] loading knn-vc...")
    t0 = time.time()
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device=device)
    print(f"[bulk] knn-vc loaded in {time.time()-t0:.1f}s")

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)

    manifest_rows: List[Dict] = []
    n_done = n_skipped = n_failed = 0
    t_last = time.time()

    for ci, cid in enumerate(children):
        ref_paths = child_reference_clips(df, cid, max_dur_sec=30.0)
        if not ref_paths:
            print(f"[bulk] child={cid} SKIP (no reference audio)")
            continue
        try:
            matching_set = knn_vc.get_matching_set(ref_paths)
        except Exception as e:
            print(f"[bulk] child={cid} REF-FAILED ({e})")
            n_failed += 1
            continue

        child_dir = os.path.join(OUT_AUDIO_DIR, cid)
        os.makedirs(child_dir, exist_ok=True)

        # Deterministic per-child sample of scenes
        local_rng = random.Random(args.seed + hash(cid) % (10**6))
        scene_subset = local_rng.sample(scenes, k=min(args.n_per_child, len(scenes)))

        for s in scene_subset:
            scene_md5 = md5short(s["audio_path"])
            out_wav = os.path.join(child_dir, f"{s['scene_id']}__{scene_md5}.wav")
            if os.path.exists(out_wav):
                n_skipped += 1
                manifest_rows.append({
                    "audio_path": out_wav,
                    "child_id": cid,
                    "scene_id": s["scene_id"],
                    "label": 1,
                    "source": "knnvc_converted",
                })
                continue
            try:
                query_seq = knn_vc.get_features(s["audio_path"])
                out_wav_t = knn_vc.match(query_seq, matching_set, topk=4)
                sf.write(out_wav, out_wav_t.cpu().numpy(), 16000, subtype="PCM_16")
                manifest_rows.append({
                    "audio_path": out_wav,
                    "child_id": cid,
                    "scene_id": s["scene_id"],
                    "label": 1,
                    "source": "knnvc_converted",
                })
                n_done += 1
            except Exception as e:
                print(f"[bulk] child={cid} scene={s['scene_id']} FAIL ({e})")
                n_failed += 1

        if time.time() - t_last > 30:
            print(f"[bulk] child {ci+1}/{len(children)} ({cid})  done={n_done} skip={n_skipped} fail={n_failed}")
            t_last = time.time()

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(OUT_MANIFEST, index=False)
        print(f"[bulk] wrote manifest: {OUT_MANIFEST} ({len(manifest_rows)} rows)")
    print(f"[bulk] DONE  done={n_done} skip={n_skipped} fail={n_failed} children={len(children)}")


if __name__ == "__main__":
    main()
