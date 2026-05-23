"""Extract AV-HuBERT-Large per-clip visual-speech embeddings.

T120 (spec-017 US1). Loads the LRS3 large checkpoint, processes each BIDS clip:
  - Reads cached face track JSON from av_fusion/face_track_cache/<md5>.json
  - Crops the longest track at 25 fps -> 88x88 grayscale mouth ROI (lower 60%)
  - Loads the matching .wav, computes 26-dim log-mel at 100 fps, stacks 4-of-1 to 104-dim
  - Forwards audio+video through the encoder, returns the pre-projection features (T, 1024)

Output:
  pseudo_frame/visual_features/avhubert_embeddings/<md5>.npy     -- per-clip (T, 1024)
  pseudo_frame/visual_features/avhubert_pooled.csv               -- mean/std/max/p95 per clip

Run with `--smoke <md5_prefix>` to process exactly one clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from python_speech_features import logfbank


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

AVH_REPO = os.path.join(_REPO, "pseudo_frame", "av_hubert")
AVH_CKPT = os.path.join(AVH_REPO, "checkpoints", "large_lrs3.pt")
# Do NOT prepend AVH_REPO itself: it contains a non-package fairseq/ subdir that
# Python would mistake for a namespace package and shadow the editable install.
# Avhubert submodules are loaded explicitly via importlib in load_avhubert(),
# so there is no need to put avhubert/ on sys.path either (which would cause
# duplicate model registration via the bare-import lines).

CACHE_DIR = os.path.join(_REPO, "av_fusion/face_track_cache")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
OUT_DIR = os.path.join(_REPO, "pseudo_frame/visual_features/avhubert_embeddings")
POOLED_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/avhubert_pooled.csv")

CROP_SIZE = 88
VIDEO_FPS = 25
IMAGE_MEAN = 0.421
IMAGE_STD = 0.165
STACK_ORDER = 4
AUDIO_FPS = 100  # 10 ms hop; AV-HuBERT stacks 4 -> 25 fps


def cache_key(bp: str) -> str:
    return hashlib.md5(str(bp).encode()).hexdigest()


def face_cache_path(bp: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_key(bp)}.json")


def select_longest_track(tracks: List[dict]) -> Optional[dict]:
    if not tracks:
        return None
    return max(tracks, key=lambda t: len(t.get("frames", [])))


def crop_mouth_seq(video_path: str, track: dict, n_target_frames: int) -> Optional[np.ndarray]:
    """Return (T, 88, 88) float32 normalized mouth crops at 25 fps.

    Reads the video once sequentially (random seek over NFS is ~100x slower),
    extracting mouth crops only for the frame_idx values present in the track.
    Pads with the last available crop so the output length is n_target_frames.
    """
    frames = track.get("frames", [])
    if not frames:
        return None
    by_idx: Dict[int, dict] = {int(f["frame_idx"]): f for f in frames}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames <= 0:
        cap.release()
        return None
    n_target_video_frames = int(round(n_target_frames * video_fps / VIDEO_FPS))
    n_target_video_frames = min(n_target_video_frames, total_video_frames)

    crops_by_idx: Dict[int, np.ndarray] = {}
    needed = set(by_idx.keys()) & set(range(n_target_video_frames))
    if not needed:
        cap.release()
        return None
    max_needed = max(needed) + 1
    fi = 0
    try:
        while fi < max_needed:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if fi in needed:
                f = by_idx[fi]
                H, W = img.shape[:2]
                x1, y1, x2, y2 = [int(round(c)) for c in f["bbox"]]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
                if x2 - x1 >= 16 and y2 - y1 >= 16:
                    face = img[y1:y2, x1:x2]
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (CROP_SIZE, CROP_SIZE * 5 // 3))
                    mouth_h_start = int(face_resized.shape[0] * 0.6)
                    mouth = face_resized[mouth_h_start:mouth_h_start + CROP_SIZE, :]
                    if mouth.shape != (CROP_SIZE, CROP_SIZE):
                        mouth = cv2.resize(mouth, (CROP_SIZE, CROP_SIZE))
                    crops_by_idx[fi] = mouth
            fi += 1
    finally:
        cap.release()

    if not crops_by_idx:
        return None

    sorted_idx = sorted(crops_by_idx.keys())
    out: List[np.ndarray] = []
    last_crop = crops_by_idx[sorted_idx[0]]
    for t in range(n_target_frames):
        target_video_idx = min(int(round(t * video_fps / VIDEO_FPS)), total_video_frames - 1)
        if target_video_idx in crops_by_idx:
            last_crop = crops_by_idx[target_video_idx]
        out.append(last_crop)
    arr = np.stack(out, axis=0).astype(np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    return arr  # (T, 88, 88)


def load_log_mel(audio_path: str, n_audio_frames: int) -> np.ndarray:
    """Return (n_video_frames, 104) stacked log-mel matching n_audio_frames at 25 fps."""
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        # fallback resample via torchaudio
        import torchaudio
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav = wav_t.squeeze(0).numpy()
        sr = 16000
    feats = logfbank(wav, samplerate=sr, nfilt=26).astype(np.float32)  # (T_audio, 26)
    # Pad/truncate to a multiple of STACK_ORDER then reshape
    T_target_audio = n_audio_frames * STACK_ORDER
    if feats.shape[0] < T_target_audio:
        feats = np.pad(feats, ((0, T_target_audio - feats.shape[0]), (0, 0)))
    else:
        feats = feats[:T_target_audio]
    stacked = feats.reshape(n_audio_frames, STACK_ORDER * 26)  # (T_video, 104)
    return stacked


def normalize_audio(feats: np.ndarray) -> np.ndarray:
    """Per-clip mean/std normalization (per AV-HuBERT spec)."""
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mu) / sd


def get_video_n_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return int(round(n * VIDEO_FPS / fps))


def load_avhubert(device: str = "cuda"):
    """Load AV-HuBERT-Large with proper avhubert package bootstrapping.

    Three competing constraints in the upstream code make plain `import` brittle:
      - avhubert/hubert_pretraining.py uses `from .hubert_dataset import ...`,
        requiring it to be loaded as part of a package.
      - avhubert/hubert.py and hubert_asr.py use `from hubert_pretraining import ...`
        and `from hubert import ...` (bare absolute), so those names must also
        resolve at the top level.
      - Loading both ways re-registers AVHubertModel → fairseq raises
        "Cannot register duplicate model".
    Solution: load avhubert as a package via importlib (relative imports work),
    then *alias* its internal modules into top-level sys.modules so the bare
    imports inside upstream files find the same module objects (no duplicate
    registration).
    """
    import fairseq  # noqa: F401  (also aliases fairseq.metrics into sys.modules)
    import fairseq.tasks  # noqa: F401  (forces task registry init)
    import importlib, importlib.util
    pkg_path = os.path.join(AVH_REPO, "avhubert")

    # Pre-load each submodule under its dotted name (avhubert.hubert etc.) AND
    # alias the same module object as a bare top-level name. We do this via a
    # manual import-then-alias loop to avoid re-execution.
    submodules = ["hubert_dataset", "hubert_pretraining", "resnet", "decoder", "utils",
                  "sequence_generator", "hubert", "hubert_asr", "hubert_criterion"]
    # Step 1: register the avhubert package shell so .submodule loading resolves.
    spec = importlib.util.spec_from_file_location(
        "avhubert", os.path.join(pkg_path, "__init__.py"),
        submodule_search_locations=[pkg_path],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["avhubert"] = pkg
    # Step 2: load each submodule individually as avhubert.<name> AND alias to top-level.
    for sub in submodules:
        sub_path = os.path.join(pkg_path, f"{sub}.py")
        if not os.path.exists(sub_path):
            continue
        if sub in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(f"avhubert.{sub}", sub_path)
        sub_mod = importlib.util.module_from_spec(sub_spec)
        # Make available BOTH as the package submodule and as bare top-level.
        sys.modules[f"avhubert.{sub}"] = sub_mod
        sys.modules[sub] = sub_mod
        sub_spec.loader.exec_module(sub_mod)
    # Step 3: now run avhubert/__init__.py's `from .hubert import *` etc. — every
    # referenced submodule is already in sys.modules, no duplicate registration.
    spec.loader.exec_module(pkg)
    from fairseq import checkpoint_utils
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([AVH_CKPT])
    model = models[0]
    model.eval()
    model = model.to(device)
    return model, cfg, task


def extract_one(model, device: str, video_path: str, audio_path: str, ftc_path: str) -> Optional[np.ndarray]:
    """Return (T, 1024) encoder features at 25 fps. None if no track / video missing."""
    if not (video_path and os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(ftc_path)):
        return None
    try:
        tracks = json.load(open(ftc_path))
    except Exception:
        return None
    track = select_longest_track(tracks)
    if track is None:
        return None
    n_frames = get_video_n_frames(video_path)
    if n_frames < 5:
        return None
    n_frames = min(n_frames, 750)  # cap at 30 s @ 25 fps
    mouth = crop_mouth_seq(video_path, track, n_frames)
    if mouth is None:
        return None
    audio = load_log_mel(audio_path, n_frames)
    audio = normalize_audio(audio)
    # Build batch (B=1, ...)
    video_tensor = torch.from_numpy(mouth).unsqueeze(0).unsqueeze(1).to(device)  # (1, 1, T, 88, 88)
    audio_tensor = torch.from_numpy(audio).t().unsqueeze(0).to(device)  # (1, 104, T)
    with torch.no_grad():
        # AVHubertModel.extract_finetune: returns (features, padding_mask)
        source = {"audio": audio_tensor, "video": video_tensor}
        try:
            features, _ = model.extract_finetune(source=source, padding_mask=None, output_layer=None)
        except AttributeError:
            # fall back to forward + grab encoder out
            ret = model(source=source, target_list=None, padding_mask=None, mask=False, features_only=True)
            features = ret.get("x") if isinstance(ret, dict) else ret[0]
    feats = features.squeeze(0).cpu().numpy()  # (T, D)
    return feats.astype(np.float32)


def pooled_stats(arr: np.ndarray, prefix: str = "avh") -> Dict[str, float]:
    out = {}
    for stat, fn in (("mean", lambda a: np.mean(a, axis=0)),
                     ("std",  lambda a: np.std(a, axis=0)),
                     ("max",  lambda a: np.max(a, axis=0)),
                     ("p95",  lambda a: np.quantile(a, 0.95, axis=0))):
        v = fn(arr)
        for d, val in enumerate(v):
            out[f"{prefix}_{stat}_d{d:04d}"] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=str, default=None,
                    help="md5 prefix; runs on first matching clip and writes /tmp/avhubert_smoke.npy")
    ap.add_argument("--all", action="store_true", help="bulk extraction over all eligible clips")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[avh] device={device}")

    print(f"[avh] loading model from {AVH_CKPT}")
    t0 = time.time()
    model, cfg, task = load_avhubert(device=device)
    print(f"[avh] model loaded in {time.time()-t0:.1f}s; model class={type(model).__name__}")

    df = pd.read_csv(SPLIT_CSV)

    if args.smoke:
        df["md5"] = df["BidsProcessed"].fillna("").apply(cache_key)
        match = df[df["md5"].str.startswith(args.smoke)]
        if match.empty:
            # try first eligible clip with face cache
            for _, row in df.iterrows():
                bp = row.get("BidsProcessed")
                if pd.notna(bp) and os.path.exists(face_cache_path(bp)):
                    match = pd.DataFrame([row])
                    break
        row = match.iloc[0]
        bp = row["BidsProcessed"]
        ap_ = row.get("audio_path")
        ftc = face_cache_path(bp)
        print(f"[smoke] clip md5={cache_key(bp)} bp={os.path.basename(str(bp))} ftc_exists={os.path.exists(ftc)}")
        feats = extract_one(model, device, bp, ap_, ftc)
        if feats is None:
            print("[smoke] FAILED (no features extracted)"); sys.exit(2)
        out_path = "/tmp/avhubert_smoke.npy"
        np.save(out_path, feats)
        print(f"[smoke] OK shape={feats.shape} mean={feats.mean():.4f} std={feats.std():.4f} -> {out_path}")
        return

    if not args.all:
        ap.error("Specify --smoke <md5_prefix> or --all")

    os.makedirs(OUT_DIR, exist_ok=True)
    df["md5"] = df["BidsProcessed"].fillna("").apply(cache_key)
    rows = df.iloc[args.start:args.end] if args.end is not None else df.iloc[args.start:]
    print(f"[bulk] processing {len(rows)} clips ({args.start}..{args.start + len(rows)})")

    pooled_records: List[Dict] = []
    cursor = 0
    n_done = n_skipped = n_failed = 0
    last_print = time.time()
    for _, row in rows.iterrows():
        cursor += 1
        bp = row.get("BidsProcessed")
        ap_ = row.get("audio_path")
        md5 = cache_key(bp) if pd.notna(bp) else ""
        out_npy = os.path.join(OUT_DIR, f"{md5}.npy")
        if os.path.exists(out_npy):
            n_skipped += 1
            arr = np.load(out_npy, mmap_mode="r")
            rec = {"audio_path": ap_, "md5": md5, "n_frames": int(arr.shape[0]), "embed_dim": int(arr.shape[1])}
            rec.update(pooled_stats(np.asarray(arr)))
            pooled_records.append(rec)
            continue
        ftc = face_cache_path(bp) if pd.notna(bp) else ""
        feats = extract_one(model, device, bp, ap_, ftc)
        if feats is None:
            n_failed += 1
            continue
        np.save(out_npy, feats)
        rec = {"audio_path": ap_, "md5": md5, "n_frames": int(feats.shape[0]), "embed_dim": int(feats.shape[1])}
        rec.update(pooled_stats(feats))
        pooled_records.append(rec)
        n_done += 1
        if time.time() - last_print > 30:
            print(f"[bulk] {cursor}/{len(rows)}  done={n_done} skipped={n_skipped} failed={n_failed}", flush=True)
            last_print = time.time()

    if pooled_records:
        pd.DataFrame(pooled_records).to_csv(POOLED_CSV, index=False)
        print(f"[bulk] wrote {len(pooled_records)} rows -> {POOLED_CSV}")
    print(f"[bulk] done={n_done} skipped={n_skipped} failed={n_failed} total={cursor}")


if __name__ == "__main__":
    main()
