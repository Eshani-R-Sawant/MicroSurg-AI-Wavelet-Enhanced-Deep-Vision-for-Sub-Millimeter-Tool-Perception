#!/bin/bash
#SBATCH --job-name=teacher_oracle
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --output=teacher_oracle_%j.log

module load python/3.10.pytorch
source /scratch/m25cse012/myenv/bin/activate

# Install dependencies
pip install transformers torch torchvision google-generativeai

cd /scratch/m25cse012/CV_major

python3 03_teacher_oracle.py