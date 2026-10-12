#!/usr/bin/env python3
"""
Root Entrypoint for SurgMLLM / ZWZ Model Evaluation & Grounding.
Runs inference, calculates Word-F1, IoU, and mAP, and produces Grad-CAM heatmaps.
"""

from evaluation.evaluate_surgical import run_evaluation

if __name__ == "__main__":
    run_evaluation()
