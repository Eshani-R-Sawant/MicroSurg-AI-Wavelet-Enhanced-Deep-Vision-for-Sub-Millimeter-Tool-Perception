#!/usr/bin/env python3
"""
Root Entrypoint for SurgMLLM / ZWZ Model Training.
Runs QLoRA 4-bit fine-tuning of Qwen2-VL with Wavelet Branch and Mask Decoder.
"""

from training.train_zooming import train

if __name__ == "__main__":
    train()
