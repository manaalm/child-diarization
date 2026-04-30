"""C7 — Voice-transferred synth augmentation in WavLM feature space.

Hypothesis (spec-016 follow-up #1): per-child voice transfer of synth scenes
into training children's voices improves seen-child child-vocalization detection
beyond plain WavLM features.

Mechanism:
  1. Extract mean-pooled WavLM-Base+ frame embeddings for every clip in the
     seen-child split + every synth scene (5000 scenes from spec-009).
  2. For each train child with ≥3 positive clips, compute their voice prototype
     `p_child` = mean of their training positive embeddings (768-d).
  3. Compute `p_generic_synth` = mean of all synth positive embeddings.
  4. Voice transfer: feat_aug = feat_synth - p_generic_synth + p_child  (linear
     mean-shift in WavLM feature space; the literature calls this "prototype
     subtraction" or "first-order voice transfer").
  5. Train two logistic regression classifiers:
        (a) baseline: real seen-child train only
        (b) voice-transfer: real seen-child train + voice-transferred synth positives
     Both tuned on val, evaluated on test.

Note: this is a feature-space proxy for full voice cloning. It avoids the
need for a vocoder or audio synthesis library, and keeps the experiment
compatible with the existing `child-vocalizations` env (no SPARC/coqui-tts
install required, which broke senselab earlier in this work).

Usage:
  PYTHONPATH=. python synth/scripts/voice_transfer_experiment.py
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoFeatureExtractor, WavLMModel

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 30.0    # cap clip length for feature extraction (memory)
BATCH_SIZE = 8
SEED = 42

# Output dir
OUT_DIR = os.path.join(_REPO, "synth_results/voice_transfer_experiment")
FEATURE_CACHE = os.path.join(OUT_DIR, "wavlm_mean_features.npz")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_audio(path: str, max_sec: float = MAX_AUDIO_SEC) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    n_max = int(max_sec * SAMPLE_RATE)
    if wav.shape[1] > n_max:
        wav = wav[:, :n_max]
    return wav.squeeze(0).float()  # (T,)


@torch.no_grad()
def extract_features_batch(model, feat_ext, paths, device):
    """Return mean-pooled WavLM frame embeddings: (B, 768)."""
    audios = [load_audio(p) for p in paths]
    inputs = feat_ext(
        [a.numpy() for a in audios],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    iv = inputs.input_values.to(device)
    am = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    out = model(input_values=iv, attention_mask=am, output_hidden_states=False)
    hs = out.last_hidden_state  # (B, T_frames, 768)
    if am is not None:
        # Mean-pool over valid frames only. WavLM frame stride = 320 samples.
        frame_lens = (am.sum(dim=1) // 320).clamp(min=1)
        masks = torch.zeros_like(hs[..., 0])
        for i, fl in enumerate(frame_lens):
            masks[i, :fl] = 1.0
        feats = (hs * masks.unsqueeze(-1)).sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        feats = hs.mean(dim=1)
    return feats.detach().cpu().numpy().astype(np.float32)


def precompute_features(real_df, synth_df, model, feat_ext, device):
    """Return dicts {audio_path: 768-d feature}."""
    all_paths = list(real_df["audio_path"].unique()) + list(synth_df["audio_path"].unique())
    seen = set()
    paths = [p for p in all_paths if not (p in seen or seen.add(p))]
    print(f"  Extracting WavLM features for {len(paths)} unique clips ({len(real_df)} real + {len(synth_df)} synth)", flush=True)

    feats = {}
    t0 = time.time()
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i+BATCH_SIZE]
        try:
            f = extract_features_batch(model, feat_ext, batch, device)
            for p, vec in zip(batch, f):
                feats[p] = vec
        except Exception as e:
            print(f"    batch err at {i}: {e}", flush=True)
            for p in batch:
                try:
                    f = extract_features_batch(model, feat_ext, [p], device)
                    feats[p] = f[0]
                except Exception as ee:
                    print(f"    skip {p}: {ee}", flush=True)
        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + BATCH_SIZE) / max(elapsed, 1e-3)
            eta = (len(paths) - i) / max(rate, 1e-6)
            print(f"    {i+len(batch)}/{len(paths)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

    print(f"  Total features extracted: {len(feats)} in {(time.time()-t0)/60:.1f}min", flush=True)
    return feats


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc":     float(roc_auc_score(y_true, y_score)),
        "auprc":     float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "n":         int(len(y_true)),
    }


def tune_threshold(y_true, y_score):
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-augment-per-child", type=int, default=10,
                    help="Number of voice-transferred synth scenes per train child (default 10)")
    ap.add_argument("--min-pos-clips-per-child", type=int, default=3,
                    help="Min positive train clips required to compute a child prototype")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    device = args.device if torch.cuda.is_available() else "cpu"

    # === Data ===
    print("=== Loading splits ===", flush=True)
    real_df = pd.read_csv(os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv"))
    real_df = real_df[real_df["audio_exists"] == True].reset_index(drop=True)
    synth_df = pd.read_csv(os.path.join(_REPO, "synth_results/manifests/synthetic_manifest.csv"))
    synth_df["label"] = synth_df["target_child_vocalized"].astype(int)
    print(f"  Real: {len(real_df)} clips ({real_df['split'].value_counts().to_dict()})", flush=True)
    print(f"  Synth: {len(synth_df)} scenes ({synth_df['label'].value_counts().to_dict()})", flush=True)

    # === WavLM features ===
    if os.path.exists(FEATURE_CACHE):
        print(f"=== Loading cached features from {FEATURE_CACHE} ===", flush=True)
        npz = np.load(FEATURE_CACHE, allow_pickle=True)
        feats = {str(k): v for k, v in zip(npz["paths"], npz["features"])}
        print(f"  Loaded {len(feats)} features", flush=True)
    else:
        print(f"=== Extracting WavLM features (one-time) on {device} ===", flush=True)
        feat_ext = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        feats = precompute_features(real_df, synth_df, model, feat_ext, device)
        paths = list(feats.keys())
        features = np.stack([feats[p] for p in paths], axis=0)
        np.savez_compressed(FEATURE_CACHE, paths=np.array(paths), features=features)
        print(f"  Saved cache → {FEATURE_CACHE}", flush=True)
        del model, feat_ext
        if device == "cuda":
            torch.cuda.empty_cache()

    # === Per-child prototypes ===
    print("=== Computing per-child WavLM prototypes ===", flush=True)
    train = real_df[real_df["split"] == "train"]
    train_pos = train[train["label"] == 1]
    child_protos = {}
    for child, grp in train_pos.groupby("child_id"):
        if len(grp) < args.min_pos_clips_per_child:
            continue
        vecs = [feats[p] for p in grp["audio_path"] if p in feats]
        if len(vecs) >= args.min_pos_clips_per_child:
            child_protos[str(child)] = np.mean(np.stack(vecs, axis=0), axis=0)
    print(f"  {len(child_protos)} children with prototypes (min {args.min_pos_clips_per_child} pos clips)", flush=True)

    # === Generic synth positive prototype ===
    synth_pos = synth_df[synth_df["label"] == 1]
    synth_pos_paths = [p for p in synth_pos["audio_path"] if p in feats]
    p_generic_synth = np.mean(np.stack([feats[p] for p in synth_pos_paths], axis=0), axis=0)
    print(f"  Generic synth prototype from {len(synth_pos_paths)} positive scenes", flush=True)

    # === Voice transfer: per-child synth augmentation ===
    print(f"=== Voice transfer: {args.n_augment_per_child} synth scenes × {len(child_protos)} children ===", flush=True)
    aug_X = []
    aug_y = []
    aug_meta = []
    for child, p_child in child_protos.items():
        sample_idx = rng.choice(len(synth_pos_paths), size=min(args.n_augment_per_child, len(synth_pos_paths)), replace=False)
        for i in sample_idx:
            p_synth_path = synth_pos_paths[i]
            feat_aug = feats[p_synth_path] - p_generic_synth + p_child
            aug_X.append(feat_aug)
            aug_y.append(1)
            aug_meta.append({"child_id": child, "synth_path": p_synth_path})
    aug_X = np.stack(aug_X, axis=0) if aug_X else np.zeros((0, 768), dtype=np.float32)
    aug_y = np.array(aug_y)
    print(f"  Generated {len(aug_X)} voice-transferred positive features", flush=True)

    # === Build train / val / test feature matrices ===
    def build_xy(split_name):
        sdf = real_df[real_df["split"] == split_name]
        X, y, meta = [], [], []
        for _, r in sdf.iterrows():
            p = r["audio_path"]
            if p not in feats:
                continue
            X.append(feats[p])
            y.append(int(r["label"]))
            meta.append({"audio_path": p, "child_id": r["child_id"], "timepoint": r.get("timepoint_norm", "")})
        return np.stack(X, axis=0), np.array(y), meta

    X_train, y_train, _ = build_xy("train")
    X_val, y_val, _ = build_xy("val")
    X_test, y_test, meta_test = build_xy("test")
    print(f"  Real: train {X_train.shape} val {X_val.shape} test {X_test.shape}", flush=True)

    # === Baseline: LR on real train only ===
    print("=== Baseline LR on real train only ===", flush=True)
    lr_base = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X_train, y_train)
    val_score_base = lr_base.predict_proba(X_val)[:, 1]
    test_score_base = lr_base.predict_proba(X_test)[:, 1]
    thr_base = tune_threshold(y_val, val_score_base)
    val_metrics_base = compute_metrics(y_val, val_score_base, thr_base)
    test_metrics_base = compute_metrics(y_test, test_score_base, thr_base)
    print(f"  baseline val: {val_metrics_base}", flush=True)
    print(f"  baseline test: {test_metrics_base}", flush=True)

    # === Voice-transfer LR: real train + voice-transferred synth positives ===
    print("=== Voice-transfer LR (real + voice-transferred synth) ===", flush=True)
    X_train_aug = np.concatenate([X_train, aug_X], axis=0)
    y_train_aug = np.concatenate([y_train, aug_y], axis=0)
    print(f"  augmented train shape: {X_train_aug.shape}  pos:neg = {(y_train_aug==1).sum()}:{(y_train_aug==0).sum()}", flush=True)
    lr_aug = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X_train_aug, y_train_aug)
    val_score_aug = lr_aug.predict_proba(X_val)[:, 1]
    test_score_aug = lr_aug.predict_proba(X_test)[:, 1]
    thr_aug = tune_threshold(y_val, val_score_aug)
    val_metrics_aug = compute_metrics(y_val, val_score_aug, thr_aug)
    test_metrics_aug = compute_metrics(y_test, test_score_aug, thr_aug)
    print(f"  vt-aug val: {val_metrics_aug}", flush=True)
    print(f"  vt-aug test: {test_metrics_aug}", flush=True)

    # === Control: real train + UN-transferred synth positives (just append, no voice transfer) ===
    print("=== Control LR (real + raw synth, no voice transfer) ===", flush=True)
    n_ctl = len(aug_X)  # match volume
    ctl_idx = rng.choice(len(synth_pos_paths), size=min(n_ctl, len(synth_pos_paths)), replace=True)
    ctl_X = np.stack([feats[synth_pos_paths[i]] for i in ctl_idx], axis=0)
    ctl_y = np.ones(len(ctl_X), dtype=int)
    X_train_ctl = np.concatenate([X_train, ctl_X], axis=0)
    y_train_ctl = np.concatenate([y_train, ctl_y], axis=0)
    lr_ctl = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X_train_ctl, y_train_ctl)
    val_score_ctl = lr_ctl.predict_proba(X_val)[:, 1]
    test_score_ctl = lr_ctl.predict_proba(X_test)[:, 1]
    thr_ctl = tune_threshold(y_val, val_score_ctl)
    val_metrics_ctl = compute_metrics(y_val, val_score_ctl, thr_ctl)
    test_metrics_ctl = compute_metrics(y_test, test_score_ctl, thr_ctl)
    print(f"  control val: {val_metrics_ctl}", flush=True)
    print(f"  control test: {test_metrics_ctl}", flush=True)

    # === Save results ===
    results = {
        "experiment": "voice_transfer_wavlm_feature_space",
        "spec": "016-synth-augmentation-extensions follow-up #1 (C7)",
        "config": {
            "n_augment_per_child": args.n_augment_per_child,
            "min_pos_clips_per_child": args.min_pos_clips_per_child,
            "n_train_children_with_proto": len(child_protos),
            "n_voice_transferred_positives": int(len(aug_X)),
            "n_synth_positive_pool": len(synth_pos_paths),
            "seed": SEED,
        },
        "baseline_lr": {"val": val_metrics_base, "test": test_metrics_base},
        "voice_transfer_lr": {"val": val_metrics_aug, "test": test_metrics_aug},
        "control_raw_synth_lr": {"val": val_metrics_ctl, "test": test_metrics_ctl},
        "delta_voice_transfer_vs_baseline": {
            "f1":    test_metrics_aug["f1"]    - test_metrics_base["f1"],
            "auroc": test_metrics_aug["auroc"] - test_metrics_base["auroc"],
            "auprc": test_metrics_aug["auprc"] - test_metrics_base["auprc"],
        },
        "delta_voice_transfer_vs_control": {
            "f1":    test_metrics_aug["f1"]    - test_metrics_ctl["f1"],
            "auroc": test_metrics_aug["auroc"] - test_metrics_ctl["auroc"],
            "auprc": test_metrics_aug["auprc"] - test_metrics_ctl["auprc"],
        },
    }
    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results written to {out_json} ===", flush=True)

    # Quick comparison table
    print("\n=== Δ vs baseline (test) ===", flush=True)
    print(f"  voice transfer: F1 {results['delta_voice_transfer_vs_baseline']['f1']:+.4f}  "
          f"AUROC {results['delta_voice_transfer_vs_baseline']['auroc']:+.4f}  "
          f"AUPRC {results['delta_voice_transfer_vs_baseline']['auprc']:+.4f}", flush=True)
    print("=== Δ vs raw-synth-augmented control (test) ===", flush=True)
    print(f"  voice transfer: F1 {results['delta_voice_transfer_vs_control']['f1']:+.4f}  "
          f"AUROC {results['delta_voice_transfer_vs_control']['auroc']:+.4f}  "
          f"AUPRC {results['delta_voice_transfer_vs_control']['auprc']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
