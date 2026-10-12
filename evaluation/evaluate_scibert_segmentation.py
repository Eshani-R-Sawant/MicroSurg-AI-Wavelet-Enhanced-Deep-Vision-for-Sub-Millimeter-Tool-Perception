"""
Medical Evaluation with SciBERT BERTScore, Watershed & Otsu Segmentation Overlays.

Evaluates surgical VQA with domain-specific medical BERTScore (allenai/scibert_scidict),
generates binary Otsu tissue masks, Watershed separation maps, and Grad-CAM overlays.
"""

import glob
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import PeftModel

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(BASE_DIR / "output" / "inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_medical_metrics(preds, gts):
    if not preds or not gts:
        return {"medical_bert_f1": 0.0, "exact_match": 0.0}

    em = sum(1 for p, g in zip(preds, gts) if p.lower().strip() == g.lower().strip()) / len(preds)
    f1_score = 0.0
    if HAS_BERTSCORE:
        try:
            _, _, F1 = bert_score(
                preds, gts,
                model_type="allenai/scibert_scidict",
                lang="en",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            f1_score = round(F1.mean().item(), 4)
        except Exception as e:
            print(f"⚠️ BERTScore error: {e}")

    return {"medical_bert_f1": f1_score, "exact_match": round(em, 4)}


def save_segmentation_artifacts(img_path, base_name, idx):
    img = cv2.imread(img_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_MASK.jpg", mask)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_trans, 0.7 * dist_trans.max(), 255, 0)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[unknown == 255] = 0
    img_ws = img.copy()
    cv2.watershed(img_ws, markers)
    img_ws[markers == -1] = [0, 0, 255]
    cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_WATERSHED.jpg", img_ws)
