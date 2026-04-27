"""
Evaluate synthesis quality for a set of generated samples.

Metrics:
  - MCD (Mel Cepstral Distortion): spectral quality vs. held-out real speech
  - ECAPA speaker similarity: cosine similarity to age-group prototype
  - Age-group classifier accuracy: LR classifier trained on real ECAPA embeddings
  - F0 distribution statistics: pitch comparison

Usage:
    python synthesis/evaluate.py \\
        --generated-dir synthesis/generated/vae_12m_v1/12_16m/ \\
        --reference-dir synthesis/data/12_16m/ \\
        --age-group 12_16m \\
        --output-path synthesis/eval_results/12_16m/eval_results.json

Exit codes:
    0 = success
    1 = input error
    2 = metric computation failure
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000
N_MFCC = 13
N_MEL = 80
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def _load_wavs(directory: str, max_files: int = 500) -> list:
    exts = {".wav", ".flac"}
    files = [f for f in Path(directory).glob("*") if f.suffix.lower() in exts]
    files = sorted(files)[:max_files]
    wavs = []
    for f in files:
        try:
            wav, sr = torchaudio.load(str(f))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            wavs.append(wav.squeeze(0))
        except Exception:
            continue
    return wavs


def compute_mcd(gen_wavs: list, ref_wavs: list) -> tuple:
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
        melkwargs={"n_mels": N_MEL, "n_fft": 1024, "hop_length": 256},
    )
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        use_dtw = True
    except ImportError:
        use_dtw = False

    log10_2 = np.log(10) / np.log(2)
    mcds = []
    rng = np.random.default_rng(42)
    pairs = min(len(gen_wavs), len(ref_wavs), 200)
    ref_indices = rng.choice(len(ref_wavs), size=pairs, replace=len(ref_wavs) < pairs)
    gen_indices = rng.choice(len(gen_wavs), size=pairs, replace=len(gen_wavs) < pairs)

    for gi, ri in zip(gen_indices, ref_indices):
        g_mfcc = mfcc_transform(gen_wavs[gi]).numpy().T
        r_mfcc = mfcc_transform(ref_wavs[ri]).numpy().T
        if use_dtw:
            dist, _ = fastdtw(g_mfcc, r_mfcc, dist=euclidean)
            n = max(len(g_mfcc), len(r_mfcc))
            mcd = (10.0 / log10_2) * (dist / n)
        else:
            min_len = min(len(g_mfcc), len(r_mfcc))
            diff = g_mfcc[:min_len] - r_mfcc[:min_len]
            mcd = (10.0 / log10_2) * np.sqrt(2 * np.mean(np.sum(diff ** 2, axis=1)))
        mcds.append(float(mcd))

    return float(np.mean(mcds)), float(np.std(mcds))


def compute_speaker_similarity(gen_wavs: list, prototype_path: str,
                                device: str = "cpu") -> float:
    if not prototype_path or not os.path.exists(prototype_path):
        return float("nan")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        return float("nan")

    proto = torch.load(prototype_path, map_location=device)
    if isinstance(proto, dict):
        proto = proto.get("embedding", next(iter(proto.values())))
    proto = proto.squeeze().to(device).float()
    proto = proto / proto.norm().clamp(min=1e-8)

    model = EncoderClassifier.from_hparams(source=ECAPA_SOURCE,
                                            run_opts={"device": device})
    sims = []
    for wav in gen_wavs[:200]:
        wav_t = wav.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_batch(wav_t).squeeze().float()
        emb = emb / emb.norm().clamp(min=1e-8)
        sims.append(float(torch.dot(emb, proto).item()))

    return float(np.mean(sims)) if sims else float("nan")


def compute_age_classifier_accuracy(gen_wavs: list, ref_wavs_12: list,
                                     ref_wavs_34: list, age_group: str,
                                     device: str = "cpu") -> float:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return float("nan")

    if not ref_wavs_12 or not ref_wavs_34:
        return float("nan")

    model = EncoderClassifier.from_hparams(source=ECAPA_SOURCE,
                                            run_opts={"device": device})

    def embed_batch(wavs, max_n=100):
        embs = []
        for wav in wavs[:max_n]:
            with torch.no_grad():
                e = model.encode_batch(wav.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            embs.append(e)
        return np.stack(embs)

    X_12 = embed_batch(ref_wavs_12)
    X_34 = embed_batch(ref_wavs_34)
    y = np.array([0] * len(X_12) + [1] * len(X_34))
    X_train = np.concatenate([X_12, X_34], axis=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y)

    X_gen = embed_batch(gen_wavs)
    X_gen = scaler.transform(X_gen)
    preds = clf.predict(X_gen)
    target = 0 if age_group == "12_16m" else 1
    return float((preds == target).mean())


def compute_f0_stats(wavs: list) -> dict:
    try:
        import librosa
    except ImportError:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}

    f0s = []
    for wav in wavs[:200]:
        audio = wav.numpy()
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=SAMPLE_RATE,
        )
        if f0 is not None:
            voiced = f0[voiced_flag] if voiced_flag is not None else f0
            voiced = voiced[~np.isnan(voiced)]
            f0s.extend(voiced.tolist())

    if not f0s:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    arr = np.array(f0s)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "median": float(np.median(arr))}


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis quality.")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--age-group", required=True, choices=["12_16m", "34_38m"])
    parser.add_argument("--prototype-path", default="")
    parser.add_argument("--age-classifier", default="",
                        help="Not used (classifier trained on-the-fly from reference dirs).")
    parser.add_argument("--reference-12-dir", default="",
                        help="Reference dir for 12_16m (for age classifier). "
                             "Defaults to reference-dir/../12_16m if age-group=34_38m.")
    parser.add_argument("--reference-34-dir", default="",
                        help="Reference dir for 34_38m (for age classifier).")
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    if not os.path.isdir(args.generated_dir):
        print(f"ERROR: --generated-dir not found: {args.generated_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.reference_dir):
        print(f"ERROR: --reference-dir not found: {args.reference_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output_path
    if not out_path:
        out_path = str(Path(args.generated_dir) / "eval_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading generated wavs from {args.generated_dir}...")
    gen_wavs = _load_wavs(args.generated_dir)
    print(f"Loading reference wavs from {args.reference_dir}...")
    ref_wavs = _load_wavs(args.reference_dir)

    if not gen_wavs:
        print("ERROR: No generated WAVs found.", file=sys.stderr)
        sys.exit(1)
    if not ref_wavs:
        print("ERROR: No reference WAVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Generated: {len(gen_wavs)}  Reference: {len(ref_wavs)}")

    try:
        print("Computing MCD...")
        mcd_mean, mcd_std = compute_mcd(gen_wavs, ref_wavs)
        print(f"  MCD: {mcd_mean:.2f} ± {mcd_std:.2f} dB")

        print("Computing speaker similarity...")
        speaker_sim = compute_speaker_similarity(gen_wavs, args.prototype_path, device)
        print(f"  Speaker similarity: {speaker_sim:.4f}")

        print("Computing age-group accuracy...")
        ref12_dir = args.reference_12_dir
        ref34_dir = args.reference_34_dir
        if not ref12_dir or not ref34_dir:
            ref_base = Path(args.reference_dir).parent
            ref12_dir = ref12_dir or str(ref_base / "12_16m")
            ref34_dir = ref34_dir or str(ref_base / "34_38m")
        ref_wavs_12 = _load_wavs(ref12_dir, max_files=200) if os.path.isdir(ref12_dir) else []
        ref_wavs_34 = _load_wavs(ref34_dir, max_files=200) if os.path.isdir(ref34_dir) else []
        age_acc = compute_age_classifier_accuracy(gen_wavs, ref_wavs_12, ref_wavs_34,
                                                   args.age_group, device)
        print(f"  Age classifier accuracy: {age_acc:.4f}")

        print("Computing F0 statistics...")
        f0_stats = compute_f0_stats(gen_wavs)
        print(f"  F0 mean={f0_stats['mean']:.1f}Hz  std={f0_stats['std']:.1f}Hz")

    except Exception as e:
        print(f"ERROR: Metric computation failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(2)

    results = {
        "mcd_mean": mcd_mean,
        "mcd_std": mcd_std,
        "speaker_similarity_mean": speaker_sim,
        "age_classifier_accuracy": age_acc,
        "f0_stats": f0_stats,
        "n_generated": len(gen_wavs),
        "n_reference": len(ref_wavs),
        "age_group": args.age_group,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    if mcd_mean <= 8.0:
        print(f"✓ SC-003: MCD {mcd_mean:.2f} ≤ 8.0 dB")
    else:
        print(f"✗ SC-003: MCD {mcd_mean:.2f} > 8.0 dB")
    if not np.isnan(age_acc) and age_acc >= 0.70:
        print(f"✓ SC-003: Age accuracy {age_acc:.2%} ≥ 70%")
    else:
        print(f"✗ SC-003: Age accuracy {age_acc:.2%} < 70%")


if __name__ == "__main__":
    main()
