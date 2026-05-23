"""Auto-harvest wavlm_attn groupstrat3 folds 1 and 2 once they pass epoch 12.

When a fold's most-recent epoch number in its SLURM log is >= EPOCH_CUTOFF
and best_model.pt exists, this script will:
  - scancel the corresponding fill_xc_gaps task
  - submit an eval-only wrapper that harvests the best ckpt
The auto-updater (running in parallel) will pick up the resulting test_metrics.

Exits when both folds 1 and 2 have either landed naturally or been harvested.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
EPOCH_CUTOFF = 13  # harvest once epoch >= 13 (i.e. past 12)

FOLDS = {
    "f1": {
        "task_idx": 6,
        "slurm_array_job_id": "14230498",
        "log_dir": REPO / "logs/baselines",
        "ckpt_dir": REPO / "baseline_results_seen_child/wavlm_attn_groupstrat3_f1",
        "fold_num": 1,
        "harvest_done": False,
    },
    "f2": {
        "task_idx": 7,
        "slurm_array_job_id": "14230498",
        "log_dir": REPO / "logs/baselines",
        "ckpt_dir": REPO / "baseline_results_seen_child/wavlm_attn_groupstrat3_f2",
        "fold_num": 2,
        "harvest_done": False,
    },
}

EPOCH_RE = re.compile(r"\[wavlm_attn_groupstrat3_f\d\] Epoch (\d+) \|")


def latest_epoch(fold_info: dict) -> int | None:
    """Read the SLURM log and return the highest 'Epoch NN' seen."""
    pattern = fold_info["log_dir"] / f"fill_xc_gaps_*_{fold_info['task_idx']}.out"
    logs = sorted(fold_info["log_dir"].glob(pattern.name))
    if not logs:
        return None
    log = logs[-1]
    try:
        text = log.read_text()
    except FileNotFoundError:
        return None
    eps = [int(m) for m in EPOCH_RE.findall(text)]
    return max(eps) if eps else None


def write_eval_only_script(fold_num: int) -> Path:
    """Write a per-fold eval-only Python wrapper (parallel to
    eval_only_wavlm_attn_gs_f0.py)."""
    out_py = REPO / f"baselines/eval_only_wavlm_attn_gs_f{fold_num}.py"
    base = (REPO / "baselines/eval_only_wavlm_attn_gs_f0.py").read_text()
    # Swap fold 0 → fold N
    new = base.replace("wavlm_attn_groupstrat3_f0", f"wavlm_attn_groupstrat3_f{fold_num}")
    new = new.replace("fold_0", f"fold_{fold_num}")
    new = new.replace("BIDS groupstrat3 f0 eval-only on epoch-8 best ckpt",
                      f"BIDS groupstrat3 f{fold_num} eval-only on best ckpt (training cancelled past epoch 12)")
    out_py.write_text(new)
    return out_py


def write_eval_only_slurm(fold_num: int) -> Path:
    """Write the SLURM wrapper for the per-fold eval-only run."""
    out_sh = REPO / f"baselines/slurm/eval_only_wavlm_attn_gs_f{fold_num}.sh"
    base = (REPO / "baselines/slurm/eval_only_wavlm_attn_gs_f0.sh").read_text()
    new = base.replace("wavlm_attn_f0_evalonly", f"wavlm_attn_f{fold_num}_evalonly")
    new = new.replace("eval_only_wavlm_attn_gs_f0", f"eval_only_wavlm_attn_gs_f{fold_num}")
    out_sh.write_text(new)
    out_sh.chmod(0o755)
    return out_sh


def harvest(fold_key: str, fold_info: dict):
    """Cancel the long-running task and submit eval-only."""
    fold_num = fold_info["fold_num"]
    task_id = f"{fold_info['slurm_array_job_id']}_{fold_info['task_idx']}"
    ckpt = fold_info["ckpt_dir"] / "best_model.pt"
    if not ckpt.exists():
        print(f"[SKIP] {fold_key}: best_model.pt missing, cannot harvest", flush=True)
        return
    print(f"[HARVEST] {fold_key} (task {task_id}): cancelling and submitting eval-only", flush=True)
    subprocess.run(["scancel", task_id], check=False)
    py = write_eval_only_script(fold_num)
    sh = write_eval_only_slurm(fold_num)
    print(f"  wrote {py.name}, {sh.name}", flush=True)
    res = subprocess.run(["sbatch", str(sh)], cwd=str(REPO), capture_output=True, text=True)
    print(f"  sbatch -> {res.stdout.strip()} {res.stderr.strip()}", flush=True)
    fold_info["harvest_done"] = True


def fold_landed_naturally(fold_info: dict) -> bool:
    """Has test_metrics_tuned.json appeared (job finished on its own)?"""
    return (fold_info["ckpt_dir"] / "test_metrics_tuned.json").exists()


def main():
    while True:
        any_pending = False
        for k, info in FOLDS.items():
            if info["harvest_done"] or fold_landed_naturally(info):
                continue
            any_pending = True
            ep = latest_epoch(info)
            if ep is not None and ep >= EPOCH_CUTOFF:
                harvest(k, info)
        if not any_pending:
            print(f"[DONE] both wavlm_attn fold 1 and fold 2 either landed or harvested", flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
