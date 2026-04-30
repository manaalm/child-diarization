"""Status tracker for spec-014 fire-and-forget runs.

Reads mil/spec014_jobs.json (produced by mil/slurm/run_spec014.sh) and reports:
  - SLURM state per job (PENDING / RUNNING / COMPLETED / FAILED / TIMEOUT / etc.)
  - Whether the expected output file exists
  - Last 8 lines of the .err log for any FAILED job (so the failure mode is visible)
  - A summary table at the end

Optional: --resubmit N  — for any FAILED job, re-submit the same SLURM script
                          (up to N attempts total per job, tracked via the manifest).

Usage:
    python mil/scripts/track_spec014.py                    # one-shot status
    python mil/scripts/track_spec014.py --watch            # poll every 5 min
    python mil/scripts/track_spec014.py --resubmit 1       # auto-resubmit failures
    python mil/scripts/track_spec014.py --diagnose-failed  # print error logs
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANIFEST = os.path.join(_REPO, "mil", "spec014_jobs.json")
_LOG_DIR = os.path.join(_REPO, "logs", "mil")

# How each variant should be re-submitted on failure. Variants with a YAML config
# go through train_eval_spec014.sh; the prototype cache and seg_mil sweep have
# their own scripts.
_RESUBMIT_RECIPE = {
    # variant prefix: (script_path, [extra_args])
    "proto_cache_seen_child":     ("mil/slurm/build_prototype_cache.sh",
                                    ["babar_vtc",
                                     "whisper-modeling/seen_child_splits/train.csv",
                                     "mil/prototypes/babar_vtc.npz"]),
    "proto_cache_cross_child":    ("mil/slurm/build_prototype_cache.sh",
                                    ["babar_vtc",
                                     "baselines/splits/train.csv",
                                     "mil/prototypes/babar_vtc_cross_child.npz"]),
    "seg_mil_sweep":              ("mil/slurm/seg_mil_sweep.sh", []),
}


def _sacct(job_ids: List[str]) -> Dict[str, str]:
    """Query SLURM accounting for a list of job IDs; return {jobid: state}.

    Uses sacct --format JobID,State; takes only the first row per job (the
    .batch and .extern steps are ignored).
    """
    if not job_ids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(job_ids),
        "--format=JobID,State", "--noheader", "--parsable2",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  sacct query failed: {e}", file=sys.stderr)
        return {jid: "UNKNOWN" for jid in job_ids}
    states: Dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jobid = parts[0]
        # Only take base job rows (no .batch / .extern / .0 etc.)
        if "." in jobid:
            continue
        state = parts[1].strip()
        # State may be "CANCELLED by 12345" — normalize
        states[jobid] = state.split()[0] if state else "UNKNOWN"
    for jid in job_ids:
        states.setdefault(jid, "UNKNOWN")
    return states


def _output_exists(rel_path: str) -> bool:
    full = rel_path if os.path.isabs(rel_path) else os.path.join(_REPO, rel_path)
    return os.path.isfile(full) or os.path.isdir(full)


def _err_log_tail(job_id: str, n_lines: int = 8) -> str:
    """Return the last n_lines of the .err log for a job, or empty string."""
    candidates = [
        os.path.join(_LOG_DIR, f"spec014_{job_id}.err"),
        os.path.join(_LOG_DIR, f"proto_cache_{job_id}.err"),
        os.path.join(_LOG_DIR, f"seg_mil_{job_id}.err"),
        os.path.join(_LOG_DIR, f"train_{job_id}.err"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                lines = f.readlines()
            tail = "".join(lines[-n_lines:]) if lines else ""
            return f"  [.err: {path}]\n{tail.rstrip()}"
    return "  (no .err log found)"


def _read_manifest() -> dict:
    if not os.path.isfile(_MANIFEST):
        print(f"ERROR: manifest not found at {_MANIFEST}. "
              f"Did you run mil/slurm/run_spec014.sh first?", file=sys.stderr)
        sys.exit(2)
    with open(_MANIFEST) as f:
        return json.load(f)


def _write_manifest(manifest: dict) -> None:
    with open(_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def _resubmit_recipe(variant: str) -> Optional[Tuple[str, List[str]]]:
    if variant in _RESUBMIT_RECIPE:
        return _RESUBMIT_RECIPE[variant]
    cfg_path = os.path.join("mil", "configs", f"{variant}.yaml")
    if os.path.isfile(os.path.join(_REPO, cfg_path)):
        return ("mil/slurm/train_eval_spec014.sh", [cfg_path])
    return None


def _resubmit(variant: str) -> Optional[str]:
    recipe = _resubmit_recipe(variant)
    if recipe is None:
        return None
    script, args = recipe
    cmd = ["sbatch", script] + args
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        print(f"  resubmit FAILED for {variant}: {e}", file=sys.stderr)
        return None
    # "Submitted batch job NNNNNN"
    m = re.search(r"(\d+)$", out.strip())
    return m.group(1) if m else None


def _print_status(manifest: dict, diagnose_failed: bool = False) -> Dict[str, int]:
    job_ids = [j["job_id"] for j in manifest["jobs"]]
    states = _sacct(job_ids)
    counts: Dict[str, int] = {}
    print(
        f"\n{'Variant':45s}  {'Story':10s}  {'JobID':10s}  {'State':12s}  Output"
    )
    print("-" * 110)
    for j in manifest["jobs"]:
        jid = j["job_id"]
        state = states.get(jid, "UNKNOWN")
        counts[state] = counts.get(state, 0) + 1
        out_ok = "✓" if _output_exists(j["expected_output"]) else "—"
        print(
            f"  {j['variant']:43s}  {j['story']:10s}  {jid:10s}  {state:12s}  "
            f"{out_ok}  {j['expected_output']}"
        )
        attempt = j.get("attempt", 1)
        if attempt > 1:
            print(f"      (attempt {attempt})")
    print("-" * 110)
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  {summary}")

    if diagnose_failed:
        for j in manifest["jobs"]:
            state = states.get(j["job_id"], "UNKNOWN")
            if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
                print(f"\n=== Failure diagnostic for {j['variant']} (job {j['job_id']}, state={state}) ===")
                print(_err_log_tail(j["job_id"]))
    return counts


def _try_resubmit_failures(manifest: dict, max_attempts: int) -> bool:
    """Resubmit any FAILED/TIMEOUT/NODE_FAIL/OOM job whose attempt count < max_attempts.

    Returns True if at least one job was resubmitted (manifest updated).
    """
    job_ids = [j["job_id"] for j in manifest["jobs"]]
    states = _sacct(job_ids)
    changed = False
    failure_states = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}
    for j in manifest["jobs"]:
        state = states.get(j["job_id"], "UNKNOWN")
        if state not in failure_states:
            continue
        attempt = j.get("attempt", 1)
        if attempt >= max_attempts:
            print(f"  {j['variant']}  state={state}  attempts={attempt}/{max_attempts}  giving up")
            continue
        if _output_exists(j["expected_output"]):
            print(f"  {j['variant']}  state={state} but output exists — counting as success")
            continue
        new_jid = _resubmit(j["variant"])
        if new_jid:
            print(f"  {j['variant']}  state={state}  attempt {attempt} → resubmitted as {new_jid}")
            j["job_id"] = new_jid
            j["attempt"] = attempt + 1
            j["last_state"] = state
            changed = True
    if changed:
        _write_manifest(manifest)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Track spec-014 SLURM jobs")
    parser.add_argument("--watch", action="store_true",
                        help="Poll every 5 minutes until no PENDING/RUNNING jobs remain")
    parser.add_argument("--diagnose-failed", action="store_true",
                        help="Print last lines of .err log for any FAILED job")
    parser.add_argument("--resubmit", type=int, default=0,
                        help="Resubmit FAILED jobs up to this many total attempts per job (default 0 = no resubmit)")
    parser.add_argument("--once", action="store_true", help="Single status pass; exit 0")
    args = parser.parse_args()

    manifest = _read_manifest()

    while True:
        manifest = _read_manifest()
        counts = _print_status(manifest, diagnose_failed=args.diagnose_failed)

        if args.resubmit:
            print(f"\n[resubmit] checking for failed jobs (max attempts = {args.resubmit})")
            _try_resubmit_failures(manifest, max_attempts=args.resubmit)

        active = counts.get("PENDING", 0) + counts.get("RUNNING", 0) + counts.get("REQUEUED", 0)
        if not args.watch:
            sys.exit(0)
        if active == 0:
            print("\n[watch] all jobs terminal — exiting watch loop.")
            sys.exit(0)
        print(f"\n[watch] {active} jobs still pending/running. Sleeping 300s...")
        time.sleep(300)


if __name__ == "__main__":
    main()
