#!/bin/bash
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH -o logs/seedlings/download_%A_%a.out
#SBATCH -e logs/seedlings/download_%A_%a.err
#SBATCH --array=0-174%5

mkdir -p logs/seedlings

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python - << 'EOF'
import json, os

with open("/orcd/scratch/orcd/008/manaal/child-adult-diarization/seedlings_tasks.json") as f:
    tasks = json.load(f)

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
vol_id, vol_label, sess = tasks[idx]

import importlib.util
spec = importlib.util.spec_from_file_location(
    "worker",
    "/orcd/scratch/orcd/008/manaal/child-adult-diarization/seedlings_import.py"
)
worker = importlib.util.module_from_spec(spec)  # fix: was load_from_spec
spec.loader.exec_module(worker)
worker.download_session(vol_id, vol_label, sess)
EOF