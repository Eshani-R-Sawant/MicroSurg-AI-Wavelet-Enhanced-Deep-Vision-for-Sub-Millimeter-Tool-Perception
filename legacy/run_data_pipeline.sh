#!/bin/bash
#SBATCH --job-name=surg_data_prep
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --output=data_prep_%j.log

# Environment Setup
module load python/3.10.pytorch
source /scratch/m25cse012/myenv/bin/activate

# Project Root
cd /scratch/m25cse012/CV_major

echo "🚀 Starting Data Pipeline..."

# STEP 1: Metadata Subset (Already run, but ensuring sync)
echo "--- Step 1: Subsetting (8,000 samples) ---"
python3 01_setup_and_subset.py

# STEP 2: Micro-Region Extraction (SAM-based)
echo "--- Step 2: SAM Micro-Region Extraction ---"
# We run this on the subset images currently available
python3 02_micro_region_extractor.py

# STEP 3: Teacher Oracle Distillation
if [ -z "$GENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: Set GENAI_API_KEY or GOOGLE_API_KEY before running step 3."
  exit 1
fi

echo "--- Step 3: Teacher Oracle VQA tracking ---"
python3 03_teacher_oracle.py

echo "✅ Data Pipeline Step Completed."
