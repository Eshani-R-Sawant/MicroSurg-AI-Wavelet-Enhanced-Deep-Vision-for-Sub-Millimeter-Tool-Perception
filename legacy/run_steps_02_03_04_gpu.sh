#!/bin/bash
#SBATCH --job-name=run_steps_02_03_04
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=run_steps_02_03_04_%j.log

set -euo pipefail
module load python/3.10.pytorch
source /scratch/m25cse012/myenv/bin/activate

cd /scratch/m25cse012/CV_major

echo "=== GPU CHECK ==="
nvidia-smi
python3 - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name(0))
PY



echo "=== STEP 03: Teacher Oracle Distillation ==="
python3 03_teacher_oracle.py

if [ $? -ne 0 ]; then
  echo "ERROR: Step 03 failed. Exiting."
  exit 1
fi

echo "=== STEP 04: Build Metadata ==="
python3 04_build_metadata.py

if [ $? -ne 0 ]; then
  echo "ERROR: Step 04 failed. Exiting."
  exit 1
fi

echo "✅ Pipeline completed: steps 02, 03, and 04."