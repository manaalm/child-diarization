#!/usr/bin/env python3
"""Adultification evaluation battery for synthetic child segments.

Implements the protocol from LITERATURE_REVIEW.md §7.7:

1. F0 mean / std / range (librosa.pyin).
2. F1-F4 estimates via LPC root extraction (scipy).
3. Vocal-tract length (VTL) estimate via Reby & McComb (2003) /
   Anikin et al. (2024) inverse formant dispersion:
       L_k = (2k - 1) * c / (4 * F_k)
   averaged over k = 1..4 (with c = 350 m/s for warm humid mouth air).
4. Spectral centroid + zero-crossing rate as additional acoustic markers.
5. Logistic-regression child-vs-adult classifier trained on real
   reference sets, scored on synth segments. Probability close to 1
   = "sounds like a real child"; close to 0 = "adultified".

Inputs are CSV manifests with the columns ``path,role,age_band``;
``role`` should be ``child`` or ``adult``. The script outputs a
per-segment feature CSV plus aggregate JSON comparing real-child,
real-adult, and synth-evaluated feature distributions.

Usage
-----
::

    python synth/scripts/adultification_eval.py \
        --real-child-csv synth_results/manifests/adultification_refs/real_child_14_18.csv \
        --real-adult-csv synth_results/manifests/adultification_refs/real_adult.csv \
        --eval-csv      synth_results/manifests/adultification_refs/synth_v3_14_18.csv \
        --output-dir    synth_results/adultification_eval/v3_14_18mo \
        --max-clips-per-set 500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

SPEED_OF_SOUND = 350.0  # m/s (warm humid mouth-air; Anikin et al. 2024)


def _load_audio(
    path: str,
    sr: int = 16000,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    import librosa  # local import keeps top-of-module cheap

    offset = 0.0 if start_sec is None else max(0.0, float(start_sec))
    duration = None
    if end_sec is not None and start_sec is not None:
        duration = max(0.0, float(end_sec) - float(start_sec))
        if duration <= 0:
            duration = None
    y, _ = librosa.load(path, sr=sr, mono=True, offset=offset,
                        duration=duration)
    return y, sr


def estimate_f0(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Return F0 mean/std/min/max/voiced_fraction over voiced frames."""
    import librosa

    if y.size < int(0.1 * sr):
        return {"f0_mean": float("nan"), "f0_std": float("nan"),
                "f0_min": float("nan"), "f0_max": float("nan"),
                "voiced_fraction": 0.0}
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=60.0, fmax=1200.0, sr=sr,
            frame_length=2048, hop_length=512, fill_na=np.nan,
        )
    except Exception:
        return {"f0_mean": float("nan"), "f0_std": float("nan"),
                "f0_min": float("nan"), "f0_max": float("nan"),
                "voiced_fraction": 0.0}
    voiced = f0[~np.isnan(f0)]
    voiced_frac = float(voiced.size / max(1, f0.size))
    if voiced.size == 0:
        return {"f0_mean": float("nan"), "f0_std": float("nan"),
                "f0_min": float("nan"), "f0_max": float("nan"),
                "voiced_fraction": voiced_frac}
    return {
        "f0_mean": float(np.mean(voiced)),
        "f0_std": float(np.std(voiced)),
        "f0_min": float(np.percentile(voiced, 5)),
        "f0_max": float(np.percentile(voiced, 95)),
        "voiced_fraction": voiced_frac,
    }


def estimate_formants_lpc(
    y: np.ndarray, sr: int, n_formants: int = 4
) -> Dict[str, float]:
    """Estimate F1..Fn via LPC over a pre-emphasized signal.

    Approach: split audio into 25 ms frames with 10 ms hop, pre-emphasize,
    fit an LPC of order ``2 + sr/1000`` per frame, find roots, convert
    angles to Hz, keep complex roots inside the unit circle, sort by Hz,
    and aggregate per-formant medians across frames. This is the standard
    LPC formant-tracking recipe used in many open implementations.
    """
    from scipy.signal import lfilter
    from numpy.linalg import lstsq

    if y.size < int(0.1 * sr):
        return {f"f{k+1}_hz": float("nan") for k in range(n_formants)}

    pre = lfilter([1.0, -0.97], [1.0], y).astype(np.float64)
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    if win < 32:
        return {f"f{k+1}_hz": float("nan") for k in range(n_formants)}

    order = int(2 + sr / 1000)  # standard heuristic
    formant_frames: List[List[float]] = [[] for _ in range(n_formants)]
    hamming = np.hamming(win)

    for s in range(0, max(1, pre.size - win), hop):
        frame = pre[s:s + win] * hamming
        if np.max(np.abs(frame)) < 1e-4:
            continue
        # Solve Yule-Walker via Levinson would be faster, but lstsq is fine
        # for short clips.
        try:
            R = np.correlate(frame, frame, mode="full")[win - 1: win - 1 + order + 1]
            if R[0] <= 0:
                continue
            # Build Toeplitz LHS
            T = np.zeros((order, order))
            for i in range(order):
                for j in range(order):
                    T[i, j] = R[abs(i - j)]
            r = R[1: order + 1]
            a = lstsq(T, r, rcond=None)[0]
            coeffs = np.concatenate(([1.0], -a))
            roots = np.roots(coeffs)
        except Exception:
            continue
        roots = roots[np.imag(roots) > 0]
        if roots.size == 0:
            continue
        # Angle of complex root -> Hz
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        # Bandwidth filter: keep formant-like roots
        bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + 1e-12)
        keep = (freqs > 90) & (freqs < sr / 2 - 50) & (bandwidths < 600)
        freqs = np.sort(freqs[keep])
        for k in range(n_formants):
            if k < freqs.size:
                formant_frames[k].append(float(freqs[k]))

    out: Dict[str, float] = {}
    for k in range(n_formants):
        vals = formant_frames[k]
        out[f"f{k+1}_hz"] = float(np.median(vals)) if vals else float("nan")
    return out


def estimate_vtl(
    formants_hz: Dict[str, float], n_formants_used: int = 4
) -> float:
    """Anikin / Reby inverse-dispersion VTL estimate (cm)."""
    pieces = []
    for k in range(1, n_formants_used + 1):
        Fk = formants_hz.get(f"f{k}_hz", float("nan"))
        if not (math.isfinite(Fk) and Fk > 0):
            continue
        # L_k in metres; convert to cm
        L_m = (2 * k - 1) * SPEED_OF_SOUND / (4 * Fk)
        pieces.append(L_m * 100.0)
    if not pieces:
        return float("nan")
    return float(np.mean(pieces))


def acoustic_extras(y: np.ndarray, sr: int) -> Dict[str, float]:
    import librosa

    if y.size < int(0.1 * sr):
        return {"sc_mean": float("nan"), "zcr_mean": float("nan"),
                "rms_mean": float("nan")}
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        rms = librosa.feature.rms(y=y)[0]
    except Exception:
        return {"sc_mean": float("nan"), "zcr_mean": float("nan"),
                "rms_mean": float("nan")}
    return {
        "sc_mean": float(np.mean(sc)),
        "zcr_mean": float(np.mean(zcr)),
        "rms_mean": float(np.mean(rms)),
    }


# ---------------------------------------------------------------------------
# Per-segment pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeatureRow:
    path: str
    role: str
    age_band: str
    set_name: str
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    voiced_fraction: float
    f1_hz: float
    f2_hz: float
    f3_hz: float
    f4_hz: float
    vtl_cm: float
    sc_mean: float
    zcr_mean: float
    rms_mean: float


def featurize_one(path: str, role: str, age_band: str, set_name: str,
                  sr: int = 16000,
                  start_sec: Optional[float] = None,
                  end_sec: Optional[float] = None) -> Optional[FeatureRow]:
    try:
        y, sr = _load_audio(path, sr=sr, start_sec=start_sec, end_sec=end_sec)
    except Exception as e:
        print(f"  [skip] {path}: load failed ({e})")
        return None
    if y.size < int(0.2 * sr):
        return None
    f0 = estimate_f0(y, sr)
    formants = estimate_formants_lpc(y, sr, n_formants=4)
    vtl = estimate_vtl(formants, n_formants_used=4)
    extras = acoustic_extras(y, sr)
    return FeatureRow(
        path=path, role=role, age_band=age_band, set_name=set_name,
        f0_mean=f0["f0_mean"], f0_std=f0["f0_std"],
        f0_min=f0["f0_min"], f0_max=f0["f0_max"],
        voiced_fraction=f0["voiced_fraction"],
        f1_hz=formants["f1_hz"], f2_hz=formants["f2_hz"],
        f3_hz=formants["f3_hz"], f4_hz=formants["f4_hz"],
        vtl_cm=vtl,
        sc_mean=extras["sc_mean"], zcr_mean=extras["zcr_mean"],
        rms_mean=extras["rms_mean"],
    )


def featurize_csv(csv_path: Path, set_name: str, max_clips: Optional[int],
                  sr: int = 16000) -> List[FeatureRow]:
    rows: List[FeatureRow] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        manifest = list(reader)
    if max_clips is not None:
        rng = np.random.default_rng(seed=42)
        if len(manifest) > max_clips:
            manifest = [manifest[i] for i in
                        rng.choice(len(manifest), max_clips, replace=False)]
    print(f"Featurizing {set_name}: {len(manifest)} segments")
    for i, r in enumerate(manifest):
        path = r.get("path") or r.get("wav") or r.get("file_path")
        role = (r.get("role") or "").lower()
        age = r.get("age_band") or ""
        if not path or not role:
            continue
        # Optional segment offsets (used when path points at a longer file).
        s_str = (r.get("start_sec") or "").strip()
        e_str = (r.get("end_sec") or "").strip()
        try:
            s_val = float(s_str) if s_str else None
        except ValueError:
            s_val = None
        try:
            e_val = float(e_str) if e_str else None
        except ValueError:
            e_val = None
        fr = featurize_one(path, role, age, set_name, sr=sr,
                           start_sec=s_val, end_sec=e_val)
        if fr is not None:
            rows.append(fr)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(manifest)}")
    return rows


# ---------------------------------------------------------------------------
# Aggregate / classifier / report
# ---------------------------------------------------------------------------

NUMERIC_KEYS = (
    "f0_mean", "f0_std", "f0_min", "f0_max", "voiced_fraction",
    "f1_hz", "f2_hz", "f3_hz", "f4_hz", "vtl_cm",
    "sc_mean", "zcr_mean", "rms_mean",
)


def aggregate_stats(rows: List[FeatureRow]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not rows:
        return out
    for k in NUMERIC_KEYS:
        vals = [getattr(r, k) for r in rows]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        if not vals:
            out[k] = {"n": 0}
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    return out


def js_divergence(a: List[float], b: List[float], n_bins: int = 30) -> float:
    """Jensen-Shannon divergence between two empirical distributions."""
    a = [v for v in a if v is not None and math.isfinite(v)]
    b = [v for v in b if v is not None and math.isfinite(v)]
    if not a or not b:
        return float("nan")
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)
    pa = pa / max(1, pa.sum())
    pb = pb / max(1, pb.sum())
    m = 0.5 * (pa + pb)
    eps = 1e-12
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def train_child_adult_classifier(
    real_child: List[FeatureRow], real_adult: List[FeatureRow]
) -> Tuple[Optional[object], Optional[List[str]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    feats = list(NUMERIC_KEYS)
    X = []
    y = []
    for r in real_child:
        v = [getattr(r, k) for k in feats]
        if all(math.isfinite(x) for x in v):
            X.append(v)
            y.append(1)  # child
    for r in real_adult:
        v = [getattr(r, k) for k in feats]
        if all(math.isfinite(x) for x in v):
            X.append(v)
            y.append(0)  # adult
    if len(set(y)) < 2 or len(X) < 8:
        print("WARNING: Not enough finite-feature data to train classifier.")
        return None, None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(X, y)
    return pipe, feats


def score_with_classifier(
    rows: List[FeatureRow], pipe, feats: List[str]
) -> Tuple[List[float], List[float]]:
    """Return (probabilities, mask of usable rows)."""
    probs: List[float] = []
    mask: List[float] = []
    if pipe is None:
        return probs, mask
    for r in rows:
        v = [getattr(r, k) for k in feats]
        if all(math.isfinite(x) for x in v):
            probs.append(float(pipe.predict_proba(np.asarray(v).reshape(1, -1))[0, 1]))
            mask.append(1.0)
        else:
            probs.append(float("nan"))
            mask.append(0.0)
    return probs, mask


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_feature_csv(rows: List[FeatureRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["set_name", "role", "age_band", "path", *NUMERIC_KEYS]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([
                r.set_name, r.role, r.age_band, r.path,
                *[getattr(r, k) for k in NUMERIC_KEYS],
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-child-csv", type=Path, required=True)
    p.add_argument("--real-adult-csv", type=Path, required=True)
    p.add_argument("--eval-csv", type=Path, required=True,
                   help="CSV of synth segments to evaluate.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-clips-per-set", type=int, default=500)
    p.add_argument("--sample-rate", type=int, default=16000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_child = featurize_csv(args.real_child_csv, "real_child",
                               args.max_clips_per_set, args.sample_rate)
    real_adult = featurize_csv(args.real_adult_csv, "real_adult",
                               args.max_clips_per_set, args.sample_rate)
    eval_rows = featurize_csv(args.eval_csv, "eval_synth",
                              args.max_clips_per_set, args.sample_rate)

    write_feature_csv(real_child + real_adult + eval_rows,
                      args.output_dir / "features.csv")

    # Aggregate stats
    agg = {
        "real_child": aggregate_stats(real_child),
        "real_adult": aggregate_stats(real_adult),
        "eval_synth": aggregate_stats(eval_rows),
    }

    # JS divergences (synth vs real-child, synth vs real-adult)
    js: Dict[str, Dict[str, float]] = {"vs_real_child": {}, "vs_real_adult": {}}
    for k in NUMERIC_KEYS:
        s_vals = [getattr(r, k) for r in eval_rows]
        c_vals = [getattr(r, k) for r in real_child]
        a_vals = [getattr(r, k) for r in real_adult]
        js["vs_real_child"][k] = js_divergence(s_vals, c_vals)
        js["vs_real_adult"][k] = js_divergence(s_vals, a_vals)

    # Train child-vs-adult classifier on real refs, score eval
    pipe, feats = train_child_adult_classifier(real_child, real_adult)
    score_summary: Dict[str, float] = {}
    if pipe is not None and feats is not None:
        probs_child, mask_c = score_with_classifier(real_child, pipe, feats)
        probs_adult, mask_a = score_with_classifier(real_adult, pipe, feats)
        probs_eval, mask_e = score_with_classifier(eval_rows, pipe, feats)
        # Save per-segment probabilities
        prob_path = args.output_dir / "child_probabilities.csv"
        with prob_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set_name", "path", "p_child", "usable"])
            for r, p, m in zip(real_child, probs_child, mask_c):
                w.writerow([r.set_name, r.path, p, m])
            for r, p, m in zip(real_adult, probs_adult, mask_a):
                w.writerow([r.set_name, r.path, p, m])
            for r, p, m in zip(eval_rows, probs_eval, mask_e):
                w.writerow([r.set_name, r.path, p, m])

        def _agg(probs, mask):
            arr = np.asarray([p for p, m in zip(probs, mask) if m > 0], dtype=float)
            if arr.size == 0:
                return {"n": 0}
            return {"n": int(arr.size), "mean_p_child": float(arr.mean()),
                    "frac_p_gt_0_5": float((arr > 0.5).mean())}
        score_summary = {
            "real_child": _agg(probs_child, mask_c),
            "real_adult": _agg(probs_adult, mask_a),
            "eval_synth": _agg(probs_eval, mask_e),
        }

    summary = {
        "n_real_child": len(real_child),
        "n_real_adult": len(real_adult),
        "n_eval_synth": len(eval_rows),
        "aggregate_features": agg,
        "js_divergence": js,
        "child_classifier": score_summary,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.output_dir / 'summary.json'}")
    if score_summary:
        e = score_summary.get("eval_synth", {})
        c = score_summary.get("real_child", {})
        a = score_summary.get("real_adult", {})
        print(
            "\nClassifier 'sounds-like-real-child' (P_child) summary:\n"
            f"  real_child mean = {c.get('mean_p_child', float('nan')):.3f} "
            f"(n={c.get('n', 0)})\n"
            f"  real_adult mean = {a.get('mean_p_child', float('nan')):.3f} "
            f"(n={a.get('n', 0)})\n"
            f"  eval_synth mean = {e.get('mean_p_child', float('nan')):.3f} "
            f"(n={e.get('n', 0)})\n"
            "  → eval_synth closer to real_child = less adultified."
        )


if __name__ == "__main__":
    main()
