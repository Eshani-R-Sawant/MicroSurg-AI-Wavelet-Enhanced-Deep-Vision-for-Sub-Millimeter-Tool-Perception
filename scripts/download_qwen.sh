#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --output=download_qwen_%j.log

module load python/3.10.pytorch
source /scratch/m25cse012/myenv/bin/activate

cd /scratch/m25cse012/CV_major

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_PATH = 'Qwen/Qwen-VL-Chat'
LOCAL_PATH = './Qwen-VL-Chat'

if not os.path.exists(LOCAL_PATH):
    print('Downloading model...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(LOCAL_PATH)
    model.save_pretrained(LOCAL_PATH)
    print('Model downloaded and saved locally.')
else:
    print('Model already exists locally.')
"