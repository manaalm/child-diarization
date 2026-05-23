"""KNN-VC smoke test: convert one synth scene to one training-child voice.

T111 (spec-017 US2). Picks a random training child with the most positive clips,
concatenates ~30 s of that child's positive train audio as the reference, then
runs KNN-VC matching+vocoding on a single positive synth scene. Saves WAV +
prints simple diagnostics so a human can verify the converted audio audibly
resembles the reference child.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/train.csv")
SYNTH_V2_DIR = os.path.join(_REPO, "synth_results/synthetic_scenes_v2")


def find_first_positive_synth_scene() -> str:
    import glob
    for j in sorted(glob.glob(os.path.join(SYNTH_V2_DIR, "json", "*.json"))):
        d = json.load(open(j))
        if d.get("target_child_vocalized"):
            return d["audio_path"]
    raise RuntimeError("no positive v2 synth scene found")


def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    t = torch.from_numpy(wav).unsqueeze(0)
    if sr != target_sr:
        t = torchaudio.functional.resample(t, sr, target_sr)
    return t  # (1, n_samples)


def collect_child_reference(child_id: str, max_dur_sec: float = 30.0) -> List[str]:
    df = pd.read_csv(TRAIN_CSV)
    rows = df[(df["child_id"] == child_id) & (df["label"] == 1)]
    paths: List[str] = []
    cum = 0.0
    for _, row in rows.iterrows():
        ap = row.get("audio_path")
        if not isinstance(ap, str) or not os.path.exists(ap):
            continue
        try:
            info = sf.info(ap)
            paths.append(ap)
            cum += info.duration
            if cum >= max_dur_sec:
                break
        except Exception:
            continue
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None,
                    help="Path to a positive synth scene .wav. If omitted, picks first v2 positive.")
    ap.add_argument("--child", type=str, default=None,
                    help="Training child ID. If omitted, picks the child with the most positive clips.")
    ap.add_argument("--out", type=str, default="/tmp/knnvc_smoke.wav")
    ap.add_argument("--ref-out", type=str, default="/tmp/knnvc_ref_clip.wav",
                    help="Where to save the reference child clip used (for human listening).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    # Pick child
    if args.child is None:
        df = pd.read_csv(TRAIN_CSV)
        pos = df[df["label"] == 1]
        counts = pos.groupby("child_id").size().sort_values(ascending=False)
        args.child = counts.index[0]
        print(f"[smoke] picked child={args.child} ({counts.iloc[0]} positive clips)")

    # Pick scene
    if args.scene is None:
        args.scene = find_first_positive_synth_scene()
    print(f"[smoke] source scene: {os.path.basename(args.scene)}")

    # Collect reference
    ref_paths = collect_child_reference(args.child, max_dur_sec=30.0)
    if not ref_paths:
        print(f"[smoke] FAILED: no reference audio found for child={args.child}"); sys.exit(2)
    print(f"[smoke] ref clips: {len(ref_paths)}, total ~{sum(sf.info(p).duration for p in ref_paths):.1f}s")

    # Load model
    print("[smoke] loading knn-vc...")
    t0 = time.time()
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device=device)
    print(f"[smoke] knn-vc loaded in {time.time()-t0:.1f}s")

    # Build reference matching set: extract WavLM features over all ref clips
    print("[smoke] computing reference matching set...")
    matching_set = knn_vc.get_matching_set(ref_paths)
    print(f"[smoke] matching_set shape: {tuple(matching_set.shape)}")

    # Source query
    query_seq = knn_vc.get_features(args.scene)
    print(f"[smoke] query feats shape: {tuple(query_seq.shape)}")

    # Convert
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)  # (n_samples,)
    out_np = out_wav.cpu().numpy()
    sf.write(args.out, out_np, 16000, subtype="PCM_16")
    print(f"[smoke] wrote converted audio: {args.out} ({len(out_np)/16000:.1f}s)")

    # Save reference clip too (just the first one) for A/B listening
    ref_wav, ref_sr = sf.read(ref_paths[0])
    sf.write(args.ref_out, ref_wav, ref_sr, subtype="PCM_16")
    print(f"[smoke] wrote reference: {args.ref_out}")
    print("[smoke] OK — converted spectrogram should resemble reference child's formant envelope")


if __name__ == "__main__":
    main()
