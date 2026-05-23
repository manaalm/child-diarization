#!/usr/bin/env python3
"""Fit empirical turn-taking distributions from real RTTM files.

Reads Providence + Playlogue RTTMs, classifies each segment as ``CHI``
(target child), other-child, or adult, then computes per-age-band
empirical distributions for:

* child / adult turn duration
* inter-turn pause duration
* overlap duration and overlap probability
* number of turns per file (proxied via per-30s window count)

Output is a JSON file consumable by :class:`synth.turn_taking.TurnTakingSimulator`
(see ``--write-config-stub`` to produce a YAML config patch).

Usage
-----
::

    python synth/scripts/fit_empirical_turn_taking.py \
        --providence-rttm-dir providence/rttm \
        --playlogue-rttm-dir playlogue/rttm \
        --playlogue-manifest playlogue/manifest.csv \
        --output synth_results/manifests/empirical_turn_taking.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Speaker-label -> role mapping
# ---------------------------------------------------------------------------

CHILD_LABELS = {"CHI"}
OTHER_CHILD_LABELS = {"SIS", "SI1", "SI2", "BRO", "BR1", "BR2"}
ADULT_LABELS = {
    "MOT", "FAT", "GRA", "GRN", "ADU", "FRI", "AD1", "AD2",
    "OP1", "OP2", "OPE", "GR1", "GR2", "TO1", "TO2", "ADULT",
}
EXCLUDE_LABELS = {"NON", "ENV", "OVL"}


def classify_role(label: str) -> Optional[str]:
    """Map an RTTM speaker token to a coarse role.

    Returns one of ``"child"`` (target child), ``"other_child"``, ``"adult"``,
    or ``None`` (excluded / unknown).
    """
    label = label.strip().upper()
    if label in CHILD_LABELS:
        return "child"
    if label in OTHER_CHILD_LABELS:
        return "other_child"
    if label in ADULT_LABELS:
        return "adult"
    if label in EXCLUDE_LABELS:
        return None
    return None


# ---------------------------------------------------------------------------
# Age-band parsing (Providence filename convention: name_YYMMDD.rttm where
# YYMMDD is age in years/months/days, e.g. alex_010427 = 1;04.27 = 16 mo)
# ---------------------------------------------------------------------------

_PROV_NAME = re.compile(r"^[A-Za-z]+_(\d{6})\.rttm$")


def providence_age_months(filename: str) -> Optional[int]:
    m = _PROV_NAME.match(filename)
    if not m:
        return None
    raw = m.group(1)
    try:
        years = int(raw[0:2])
        months = int(raw[2:4])
        days = int(raw[4:6])
    except ValueError:
        return None
    if years < 0 or years > 9 or months < 0 or months > 11:
        return None
    total = years * 12 + months + (days / 30.0)
    return int(round(total))


def in_age_band(age_months: Optional[int], band: str) -> bool:
    """Check whether age in months falls in band (e.g. ``"14_18"``)."""
    if age_months is None:
        return False
    lo, hi = (int(x) for x in band.split("_"))
    return lo <= age_months <= hi


# ---------------------------------------------------------------------------
# RTTM parsing
# ---------------------------------------------------------------------------

def parse_rttm(path: Path) -> List[Tuple[str, float, float]]:
    """Return list of (role, start_sec, dur_sec) sorted by start time."""
    rows: List[Tuple[str, float, float]] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError:
                continue
            role = classify_role(parts[7])
            if role is None or dur <= 0:
                continue
            rows.append((role, start, dur))
    rows.sort(key=lambda r: r[1])
    return rows


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def collect_durations(rows: List[Tuple[str, float, float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"child": [], "other_child": [], "adult": []}
    for role, _, dur in rows:
        out[role].append(float(dur))
    return out


def collect_gaps(rows: List[Tuple[str, float, float]]) -> Tuple[List[float], List[float], int]:
    """Return (pauses, overlap_durations, n_consecutive_pairs).

    A "pause" is a positive gap between consecutive turns.  An "overlap" is
    a negative gap (i.e. next turn starts before previous ends), reported
    as the absolute overlap duration.
    """
    pauses: List[float] = []
    overlaps: List[float] = []
    n_pairs = 0
    for i in range(len(rows) - 1):
        _, s0, d0 = rows[i]
        _, s1, _ = rows[i + 1]
        end0 = s0 + d0
        gap = s1 - end0
        n_pairs += 1
        if gap >= 0:
            pauses.append(gap)
        else:
            overlaps.append(-gap)
    return pauses, overlaps, n_pairs


def collect_n_turns_per_window(
    rows: List[Tuple[str, float, float]], window_sec: float = 30.0
) -> List[int]:
    """Bin turns into non-overlapping windows of width ``window_sec``."""
    if not rows:
        return []
    total_dur = max(s + d for _, s, d in rows)
    n_windows = int(np.ceil(total_dur / window_sec))
    counts = [0] * n_windows
    for _, s, _ in rows:
        idx = min(int(s // window_sec), n_windows - 1)
        counts[idx] += 1
    return counts


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _trim(samples: List[float], lo_pct: float = 1.0, hi_pct: float = 99.0) -> List[float]:
    """Clip outliers by percentile."""
    if not samples:
        return samples
    arr = np.asarray(samples, dtype=float)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    return [float(x) for x in arr[(arr >= lo) & (arr <= hi)]]


def summarize(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"n": 0}
    arr = np.asarray(samples, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Source ingestion
# ---------------------------------------------------------------------------

def ingest_providence(rttm_dir: Path, age_bands: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate Providence stats per age band.

    Returns ``{band: {child_dur: [...], adult_dur: [...], pauses: [...],
    overlaps: [...], n_pairs: int, n_turns_30s: [...]}}``.
    """
    bucket: Dict[str, Dict[str, List]] = {
        b: {"child_dur": [], "adult_dur": [], "pauses": [], "overlaps": [],
            "n_pairs": 0, "n_turns_30s": []}
        for b in age_bands
    }
    for rttm in sorted(rttm_dir.glob("*.rttm")):
        age = providence_age_months(rttm.name)
        if age is None:
            continue
        rows = parse_rttm(rttm)
        if not rows:
            continue
        durs = collect_durations(rows)
        pauses, overlaps, n_pairs = collect_gaps(rows)
        n_turns_30s = collect_n_turns_per_window(rows, 30.0)
        for band in age_bands:
            if not in_age_band(age, band):
                continue
            bucket[band]["child_dur"].extend(durs["child"])
            bucket[band]["adult_dur"].extend(durs["adult"])
            bucket[band]["pauses"].extend(pauses)
            bucket[band]["overlaps"].extend(overlaps)
            bucket[band]["n_pairs"] += n_pairs
            bucket[band]["n_turns_30s"].extend(n_turns_30s)
    return bucket


def ingest_playlogue(
    rttm_dir: Path,
    manifest_csv: Optional[Path],
    age_bands: List[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate Playlogue stats per age band using manifest's ``age_group``.

    Playlogue ``age_group`` strings look like ``12_16m``, ``17_22m`` etc.;
    we emit a ``play_<group>`` band per recording so caller can map to
    target bands.
    """
    bucket: Dict[str, Dict[str, List]] = {
        b: {"child_dur": [], "adult_dur": [], "pauses": [], "overlaps": [],
            "n_pairs": 0, "n_turns_30s": []}
        for b in age_bands
    }
    rec_to_age: Dict[str, str] = {}
    if manifest_csv is not None and manifest_csv.exists():
        with manifest_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec_id = row.get("recording_id")
                age_group = (row.get("age_group") or "").strip()
                if rec_id and age_group:
                    rec_to_age[rec_id] = age_group

    for rttm in sorted(rttm_dir.glob("*.rttm")):
        rec_id = rttm.stem
        age_group = rec_to_age.get(rec_id, "")
        # Map "12_16m" -> central age in months and check membership
        m = re.match(r"(\d+)_(\d+)m", age_group)
        if not m:
            continue
        a_lo, a_hi = int(m.group(1)), int(m.group(2))
        rows = parse_rttm(rttm)
        if not rows:
            continue
        durs = collect_durations(rows)
        pauses, overlaps, n_pairs = collect_gaps(rows)
        n_turns_30s = collect_n_turns_per_window(rows, 30.0)
        for band in age_bands:
            b_lo, b_hi = (int(x) for x in band.split("_"))
            # Include if Playlogue group overlaps this band
            if a_hi < b_lo or a_lo > b_hi:
                continue
            bucket[band]["child_dur"].extend(durs["child"])
            bucket[band]["adult_dur"].extend(durs["adult"])
            bucket[band]["pauses"].extend(pauses)
            bucket[band]["overlaps"].extend(overlaps)
            bucket[band]["n_pairs"] += n_pairs
            bucket[band]["n_turns_30s"].extend(n_turns_30s)
    return bucket


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def merge_buckets(*sources: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"child_dur": [], "adult_dur": [], "pauses": [], "overlaps": [],
                 "n_pairs": 0, "n_turns_30s": []}
    )
    for src in sources:
        for band, vals in src.items():
            for k in ("child_dur", "adult_dur", "pauses", "overlaps", "n_turns_30s"):
                out[band][k].extend(vals[k])
            out[band]["n_pairs"] += vals["n_pairs"]
    return dict(out)


def to_simulator_payload(bucket: Dict[str, Dict[str, List]]) -> Dict:
    payload: Dict[str, Dict] = {}
    for band, vals in bucket.items():
        n_pairs = vals["n_pairs"]
        n_overlaps = len(vals["overlaps"])
        overlap_prob = (n_overlaps / n_pairs) if n_pairs > 0 else 0.0
        # Trim outliers before computing Gaussian moments so the simulator
        # doesn't draw 60-second turns from a parametric tail.
        ch = _trim(vals["child_dur"])
        ad = _trim(vals["adult_dur"])
        pa = _trim(vals["pauses"])
        ov = _trim(vals["overlaps"])
        payload[band] = {
            "summary": {
                "child_turn": summarize(vals["child_dur"]),
                "adult_turn": summarize(vals["adult_dur"]),
                "pause": summarize(vals["pauses"]),
                "overlap": summarize(vals["overlaps"]),
                "n_turns_per_30s": summarize(
                    [float(x) for x in vals["n_turns_30s"]]
                ),
                "overlap_probability": float(overlap_prob),
                "n_pairs_observed": int(n_pairs),
            },
            "gaussian_fit": {
                "child_turn_duration_mean_sec": float(np.mean(ch)) if ch else 1.0,
                "child_turn_duration_std_sec": float(np.std(ch)) if ch else 0.5,
                "adult_turn_duration_mean_sec": float(np.mean(ad)) if ad else 3.0,
                "adult_turn_duration_std_sec": float(np.std(ad)) if ad else 1.5,
                "pause_mean_sec": float(np.mean(pa)) if pa else 0.6,
                "pause_std_sec": float(np.std(pa)) if pa else 0.3,
                "overlap_dur_mean_sec": float(np.mean(ov)) if ov else 0.4,
                "overlap_dur_std_sec": float(np.std(ov)) if ov else 0.2,
                "overlap_probability": float(overlap_prob),
            },
            # Bootstrap pool: simulator can sample from these directly when
            # ``turn_taking.sampling_mode == "bootstrap"``.
            "bootstrap_samples": {
                "child_turn": ch[:5000],
                "adult_turn": ad[:5000],
                "pause": pa[:5000],
                "overlap": ov[:5000],
            },
        }
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--providence-rttm-dir", type=Path, required=True)
    p.add_argument("--playlogue-rttm-dir", type=Path, default=None)
    p.add_argument("--playlogue-manifest", type=Path, default=None)
    p.add_argument(
        "--age-bands",
        nargs="+",
        default=["14_18", "34_38"],
        help="Age bands in months as 'lo_hi'",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--write-config-stub",
        type=Path,
        default=None,
        help="If given, also write a YAML stub mapping each age band to a "
        "turn_taking: block usable inside a synth config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bands = list(args.age_bands)

    print(f"Ingesting Providence from {args.providence_rttm_dir} ...")
    prov_bucket = ingest_providence(args.providence_rttm_dir, bands)

    play_bucket: Dict[str, Dict[str, List]] = {b: {} for b in bands}
    if args.playlogue_rttm_dir is not None:
        print(f"Ingesting Playlogue from {args.playlogue_rttm_dir} ...")
        play_bucket = ingest_playlogue(
            args.playlogue_rttm_dir, args.playlogue_manifest, bands
        )

    merged = merge_buckets(prov_bucket, play_bucket)
    payload = to_simulator_payload(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote empirical turn-taking JSON to {args.output}")

    # Quick CLI summary
    for band, vals in payload.items():
        s = vals["summary"]
        print(
            f"\n=== Band {band}_months ===\n"
            f"  child turn:  n={s['child_turn']['n']:>6d} "
            f"mean={s['child_turn'].get('mean', 0):.3f}s "
            f"std={s['child_turn'].get('std', 0):.3f}s "
            f"p50={s['child_turn'].get('p50', 0):.3f}s\n"
            f"  adult turn:  n={s['adult_turn']['n']:>6d} "
            f"mean={s['adult_turn'].get('mean', 0):.3f}s "
            f"std={s['adult_turn'].get('std', 0):.3f}s "
            f"p50={s['adult_turn'].get('p50', 0):.3f}s\n"
            f"  pause:       n={s['pause']['n']:>6d} "
            f"mean={s['pause'].get('mean', 0):.3f}s "
            f"std={s['pause'].get('std', 0):.3f}s\n"
            f"  overlap:     n={s['overlap']['n']:>6d} "
            f"prob={s['overlap_probability']:.3f}\n"
            f"  n_pairs:     {s['n_pairs_observed']}"
        )

    if args.write_config_stub is not None:
        try:
            import yaml
        except ImportError:
            print(
                "WARNING: PyYAML not installed; --write-config-stub will produce "
                "JSON instead of YAML. Activate the child-vocalizations env to "
                "get YAML output."
            )
            yaml = None

        stub: Dict[str, Dict] = {}
        for band, vals in payload.items():
            g = vals["gaussian_fit"]
            stub[f"turn_taking_{band}"] = {
                "child_turn_duration_mean_sec": round(g["child_turn_duration_mean_sec"], 3),
                "child_turn_duration_std_sec": round(g["child_turn_duration_std_sec"], 3),
                "adult_turn_duration_mean_sec": round(g["adult_turn_duration_mean_sec"], 3),
                "adult_turn_duration_std_sec": round(g["adult_turn_duration_std_sec"], 3),
                "pause_mean_sec": round(g["pause_mean_sec"], 3),
                "pause_std_sec": round(g["pause_std_sec"], 3),
                "overlap_probability": round(g["overlap_probability"], 3),
                "overlap_dur_mean_sec": round(g["overlap_dur_mean_sec"], 3),
                "overlap_dur_std_sec": round(g["overlap_dur_std_sec"], 3),
            }
        args.write_config_stub.parent.mkdir(parents=True, exist_ok=True)
        with args.write_config_stub.open("w") as f:
            if yaml is not None:
                yaml.safe_dump(stub, f, sort_keys=False)
            else:
                json.dump(stub, f, indent=2)
        print(f"Wrote config stub to {args.write_config_stub}")


if __name__ == "__main__":
    main()
