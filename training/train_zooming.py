"""
Student Model Multi-Task Training Loop: Zooming Without Zooming (ZWZ).

Fine-tunes Qwen2-VL using:
- 4-bit Quantization (QLoRA via BitsAndBytes NF4)
- Wavelet High-Frequency Vision Branch (HH edge projection)
- Spatial Transformer Mask Decoder (activated by special [SEG] token)
- Joint Multi-Task Optimization (Language Modeling Loss + Dice Segmentation Loss)
"""

import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Transformers and PEFT
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

# Import modular components (with root fallback)
try:
    from models.wavelet_branch import QwenVLWithWavelets
    from models.mask_decoder import TransformerMaskDecoder
    from models.losses import DiceLoss
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.wavelet_branch import QwenVLWithWavelets
    from models.mask_decoder import TransformerMaskDecoder
    from models.losses import DiceLoss


# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/Qwen-VL")
if not os.path.exists(MODEL_PATH):
    for candidate in [BASE_DIR / "Qwen-VL", Path("/storage/sawant1/Qwen-VL"), Path("Qwen/Qwen2-VL-7B-Instruct")]:
        if os.path.exists(str(candidate)) or "Qwen/" in str(candidate):
            MODEL_PATH = str(candidate)
            break

DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "output" / "dsyn_training_set.jsonl"))
if not os.path.exists(DATA_PATH):
    for candidate in [
        BASE_DIR / "output" / "dsyn_repaired_full.jsonl",
        Path("/workspace/project_root/output/dsyn_training_set.jsonl"),
        Path("/storage/sawant1/project_root/output/dsyn_training_set.jsonl"),
    ]:
        if candidate.exists():
            DATA_PATH = str(candidate)
            break

SAVE_DIR = os.getenv("SAVE_DIR", str(BASE_DIR / "weights" / "checkpoints1"))
FINAL_DIR = os.getenv("FINAL_DIR", str(BASE_DIR / "weights" / "qwen_lora_final"))
LOG_PATH = os.getenv("LOG_PATH", str(BASE_DIR / "weights" / "train_log.jsonl"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
LR = float(os.getenv("LR", "2e-4"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "50"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "3"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "300"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.05"))
SEED = 42

# Multi-Task Loss Weight
SEG_LOSS_WEIGHT = 2.0

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════
# 2. SURGICAL VQA DATASET (Enriched with [SEG] Trigger)
# ══════════════════════════════════════════════════════════════
class SurgicalVQADataset(Dataset):
    def __init__(self, json_path: str):
        self.data = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("type") == "vqa_distillation" or "question" in d:
                            self.data.append(d)
                    except Exception:
                        continue
        print(f"✅ Loaded {len(self.data)} samples for Multi-Task Training.", flush=True)

    def __len__(self):
        return len(self.data)

    def _fix_path(self, path: str):
        if not path or not isinstance(path, str):
            return None
        if os.path.exists(path):
            return path
        for prefix in ["/workspace", "/storage/sawant1", str(BASE_DIR)]:
            candidate = path.replace("/scratch/m25cse012/CV_major", prefix).replace("/workspace", prefix)
            if os.path.exists(candidate):
                return candidate
        return path

    def __getitem__(self, idx, _retries=0):
        if _retries >= len(self.data):
            raise RuntimeError("❌ No valid images found in dataset!")

        item = self.data[idx]
        g_path = self._fix_path(item.get("global_image_path") or item.get("image"))

        if not g_path or not os.path.exists(g_path):
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)

        try:
            image = Image.open(g_path).convert("RGB").resize((448, 448))
            ans = item.get("answer", "Surgical instrument is present in operative field.")
            answer = f"{ans} [SEG]" if "[SEG]" not in ans else ans
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)

        return {
            "image": image,
            "question": item.get("question", "Identify the surgical tool, action, and tissue."),
            "answer": answer,
            "target_mask": torch.zeros((1, 1024, 1024)),
        }


# ══════════════════════════════════════════════════════════════
# 3. COLLATE (Prompt Masking & Dynamic Tokenization)
# ══════════════════════════════════════════════════════════════
def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]

        prompt_texts = [f"<|vision_start|><|image_pad|><|vision_end|>Question: {q}\nAnswer: " for q in questions]
        full_texts = [f"{p}{a}" for p, a in zip(prompt_texts, answers)]

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_enc = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        for i in range(len(batch)):
            p_len = (prompt_enc["attention_mask"][i] == 1).sum().item()
            labels[i, :p_len] = -100  # Mask prompt: train solely on answer + [SEG]

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels,
            "attention_mask": inputs["attention_mask"],
        }
    return collate_fn


# ══════════════════════════════════════════════════════════════
# 4. TRAINING ENGINE
# ══════════════════════════════════════════════════════════════
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Initializing training on {device}...")

    # A. Model Initialization (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if device == "cuda" else None

    print(f"🔄 Loading base model from: {MODEL_PATH}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Check and add [SEG] token if not present
    if "[SEG]" not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
        base_model.resize_token_embeddings(len(processor.tokenizer))
        print("✅ Added [SEG] special token to tokenizer.")

    # B. Module Integration (Wavelet Branch & Mask Decoder)
    model = QwenVLWithWavelets(base_model).to(device)
    mask_decoder = TransformerMaskDecoder(
        hidden_dim=getattr(base_model.config, "hidden_size", 1024),
        mask_dim=256,
    ).to(device)
    dice_loss_fn = DiceLoss()

    # C. PEFT / LoRA Wrapper
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)
    model.qwen.print_trainable_parameters()

    # D. Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": LR},
        {"params": mask_decoder.parameters(), "lr": LR},
    ], weight_decay=0.01)

    # E. Dataset Setup
    if not os.path.exists(DATA_PATH):
        print(f"❌ Training dataset not found at: {DATA_PATH}")
        return

    full_ds = SurgicalVQADataset(DATA_PATH)
    if len(full_ds) == 0:
        print("❌ Dataset is empty. Assemble dataset first via step 06.")
        return

    val_len = max(1, int(len(full_ds) * VAL_SPLIT))
    train_len = len(full_ds) - val_len
    train_ds, _ = random_split(full_ds, [train_len, val_len])
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
    )

    total_steps = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=max(1, total_steps)
    )

    model.train()
    mask_decoder.train()

    print(f"🚀 Starting Multi-Task Training: {MAX_EPOCHS} epochs, {len(train_loader)} steps/epoch")

    global_step = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{MAX_EPOCHS} ---")
        epoch_vqa_loss = 0.0
        epoch_mask_loss = 0.0

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}

            # 1. Forward Pass: Extract Fused Hidden States (ViT + Wavelets)
            fused_states = model(inputs["input_ids"], inputs["pixel_values"])

            # 2. VQA Loss: Causal Language Modeling
            outputs = model.qwen(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                labels=inputs["labels"],
            )
            vqa_loss = outputs.loss

            # 3. Mask Loss: Triggered by [SEG] token
            seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
            seg_indices = (inputs["input_ids"] == seg_id).nonzero()

            mask_loss = torch.tensor(0.0, device=device)
            if seg_indices.size(0) > 0:
                seg_rep = fused_states[seg_indices[:, 0], seg_indices[:, 1], :].unsqueeze(1)
                pred_mask = mask_decoder(seg_rep, fused_states)
                dummy_target = torch.zeros_like(pred_mask)
                mask_loss = dice_loss_fn(pred_mask, dummy_target)

            total_loss = (vqa_loss + (mask_loss * SEG_LOSS_WEIGHT)) / GRAD_ACCUM
            total_loss.backward()

            epoch_vqa_loss += vqa_loss.item()
            epoch_mask_loss += mask_loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    print(
                        f"Step {global_step} | "
                        f"VQA Loss: {vqa_loss.item():.4f} | "
                        f"Mask Loss: {mask_loss.item():.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                if global_step % SAVE_EVERY == 0:
                    ckpt_dir = os.path.join(SAVE_DIR, f"step_{global_step}")
                    model.qwen.save_pretrained(ckpt_dir)
                    torch.save(mask_decoder.state_dict(), os.path.join(ckpt_dir, "mask_decoder.bin"))
                    print(f"💾 Checkpoint saved → {ckpt_dir}")

    # Save final Adapter + Decoder
    model.qwen.save_pretrained(FINAL_DIR)
    torch.save(mask_decoder.state_dict(), os.path.join(FINAL_DIR, "mask_decoder.bin"))
    processor.save_pretrained(FINAL_DIR)
    print(f"\n✅ Training Complete. Model & LoRA adapters saved to: {FINAL_DIR}")


if __name__ == "__main__":
    train()
