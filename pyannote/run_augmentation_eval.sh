#!/bin/bash
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/augmentation_eval_%j.out
#SBATCH -e logs/augmentation_eval_%j.err
set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

echo "=== T047: Augmentation eval 12_16m ==="
python pyannote/augmentation_eval.py \
    --diarizer babar \
    --synthetic-dir synthesis/generated/12_16m_vae_20260427_173133/12_16m \
    --age-group 12_16m \
    --aug-ratio 1.0 \
    --seed 42 \
    --output-dir pyannote/babar_augmented

echo "=== T048: Augmentation eval 34_38m ==="
python pyannote/augmentation_eval.py \
    --diarizer babar \
    --synthetic-dir synthesis/generated/34_38m_vae_20260427_173126/34_38m \
    --age-group 34_38m \
    --aug-ratio 1.0 \
    --seed 42 \
    --output-dir pyannote/babar_augmented

echo "=== T049: Compute delta vs baseline ==="
python - <<'EOF'
import json, os, csv

output_rows = []

for age_group, timepoint in [("12_16m", "14_month"), ("34_38m", "36_month")]:
    aug_path = f"pyannote/babar_augmented/{age_group}_ratio1.0/test_metrics_tuned.json"
    base_path = f"pyannote/babar_age_stratified/{age_group}/{age_group}/test_metrics_tuned.json"

    if not os.path.exists(aug_path):
        print(f"WARNING: {aug_path} not found")
        continue
    if not os.path.exists(base_path):
        # Try alternate path
        base_path = f"pyannote/babar_age_stratified/{timepoint}/{age_group}/test_metrics_tuned.json"
    if not os.path.exists(base_path):
        print(f"WARNING: baseline not found for {age_group}, trying overall baseline")
        base_path = "babar_ecapa_enrollment_runs/enroll_test_metrics.json"

    with open(aug_path) as f:
        aug = json.load(f)
    with open(base_path) as f:
        base = json.load(f)

    row = {
        "age_group": age_group,
        "aug_ratio": 1.0,
        "base_f1": round(base.get("f1", float("nan")), 4),
        "aug_f1": round(aug.get("f1", float("nan")), 4),
        "delta_f1": round(aug.get("f1", 0) - base.get("f1", 0), 4),
        "base_auroc": round(base.get("auroc", float("nan")), 4),
        "aug_auroc": round(aug.get("auroc", float("nan")), 4),
        "delta_auroc": round(aug.get("auroc", 0) - base.get("auroc", 0), 4),
        "base_auprc": round(base.get("auprc", float("nan")), 4),
        "aug_auprc": round(aug.get("auprc", float("nan")), 4),
        "delta_auprc": round(aug.get("auprc", 0) - base.get("auprc", 0), 4),
    }
    output_rows.append(row)
    print(f"{age_group}: delta_f1={row['delta_f1']:+.4f} delta_auroc={row['delta_auroc']:+.4f} delta_auprc={row['delta_auprc']:+.4f}")

if output_rows:
    out_path = "evaluation/augmentation_delta.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved augmentation delta to {out_path}")
EOF

echo "=== Done ==="
