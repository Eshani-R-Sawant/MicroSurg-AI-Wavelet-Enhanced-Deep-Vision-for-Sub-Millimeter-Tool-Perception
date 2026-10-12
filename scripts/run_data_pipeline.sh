#!/bin/bash
#SBATCH --job-name=surg_data_prep
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --output=data_prep_%j.log

set -euo pipefail
module load python/3.10.pytorch || true

echo "🚀 Starting ZWZ Surgical Data Pipeline..."

# STEP 1: Metadata Subsetting (8,000 balanced samples)
echo "--- Step 1: Subsetting (8,000 samples) ---"
python3 data_pipeline/setup_and_subset.py

# STEP 2: Micro-Region Extraction (SAM ViT-H + Saliency)
echo "--- Step 2: SAM Micro-Region Extraction ---"
python3 data_pipeline/micro_region_extractor.py

# STEP 3: Teacher Oracle Distillation (Gemini 3.1 Flash-Lite)
if [ -z "${GENAI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "⚠️ Warning: GENAI_API_KEY or GOOGLE_API_KEY not set. Check configs/api_key.env.example."
fi

echo "--- Step 3: Teacher Oracle Distillation ---"
python3 data_pipeline/teacher_oracle.py

echo "✅ Data Pipeline Stage Completed."
