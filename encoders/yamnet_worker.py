"""Subprocess worker for YAMNet (spec 022 US3).

Runs inside the yamnet-eval sibling Python env (TensorFlow + tensorflow-hub).
Reads audio paths from stdin (one per line), emits CSV to stdout with
per-clip child-vocalisation probabilities.

Output columns:
  audio_path, p_child_speech, p_babbling, p_baby_cry, p_children_shouting, p_child_voc

p_child_voc = max of the four child-vocalisation class probabilities.

YAMNet operates at 16 kHz mono. Class index names (528-class AudioSet ontology
+ silence) match Google's official ontology. Indices are pulled from the
class map CSV bundled with the TFHub model.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_MODEL_URL = os.environ.get("YAMNET_MODEL_URL", "https://tfhub.dev/google/yamnet/1")

# AudioSet ontology display names for the four child-vocalisation classes
TARGET_LABELS = [
    "Child speech, kid speaking",
    "Babbling",
    "Baby cry, infant cry",
    "Children shouting",
]

OUT_COLS = [
    "audio_path",
    "p_child_speech",
    "p_babbling",
    "p_baby_cry",
    "p_children_shouting",
    "p_child_voc",
]


def _load_yamnet():
    print("loading YAMNet…", file=sys.stderr, flush=True)
    model = hub.load(YAMNET_MODEL_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with open(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return model, class_names


def _resolve_target_indices(class_names):
    out = {}
    for label in TARGET_LABELS:
        if label in class_names:
            out[label] = class_names.index(label)
        else:
            print(f"  [warn] target label not found in YAMNet ontology: {label!r}", file=sys.stderr)
            out[label] = None
    return out


def _load_audio_16k_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        # Naive resample to 16k; for higher fidelity use scipy. YAMNet itself
        # accepts arbitrary length 16k mono float32 in [-1, 1].
        from scipy import signal
        n_target = int(round(len(wav) * 16000 / sr))
        wav = signal.resample(wav, n_target)
    return wav.astype(np.float32)


def _score_clip(model, wav: np.ndarray, target_idx: dict) -> dict:
    scores, _, _ = model(wav)
    # scores shape: (n_frames, n_classes). Clip-level prob via mean over frames.
    clip_scores = tf.reduce_mean(scores, axis=0).numpy()
    out = {}
    for label, idx in target_idx.items():
        col_key = {
            "Child speech, kid speaking": "p_child_speech",
            "Babbling":                   "p_babbling",
            "Baby cry, infant cry":       "p_baby_cry",
            "Children shouting":          "p_children_shouting",
        }[label]
        out[col_key] = float(clip_scores[idx]) if idx is not None else float("nan")
    out["p_child_voc"] = float(max(v for v in out.values() if v == v))
    return out


def main():
    audio_paths = [line.strip() for line in sys.stdin if line.strip()]
    if not audio_paths:
        print("no audio paths on stdin", file=sys.stderr)
        sys.exit(1)

    model, class_names = _load_yamnet()
    target_idx = _resolve_target_indices(class_names)

    writer = csv.DictWriter(sys.stdout, fieldnames=OUT_COLS)
    writer.writeheader()

    n = len(audio_paths)
    for i, p in enumerate(audio_paths):
        if i % 50 == 0:
            print(f"  YAMNet {i+1}/{n}: {Path(p).name}", file=sys.stderr, flush=True)
        try:
            wav = _load_audio_16k_mono(p)
            scores = _score_clip(model, wav, target_idx)
            writer.writerow({"audio_path": p, **scores})
        except Exception as e:
            print(f"  [error] {p}: {e}", file=sys.stderr)
            writer.writerow({"audio_path": p,
                             "p_child_speech": "nan", "p_babbling": "nan",
                             "p_baby_cry": "nan", "p_children_shouting": "nan",
                             "p_child_voc": "nan"})


if __name__ == "__main__":
    main()
