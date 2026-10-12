explian the code in depth line by line in detail "explian the code in depth , line by line in depth each line "import torch, os, json, random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

# --- EXTERNAL MODULE IMPORTS ---
# Ensure these files are in the same directory
from qwen_vl_wavelet import QwenVLWithWavelets
from mask_decoder import TransformerMaskDecoder

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "/workspace/Qwen-VL"
DATA_PATH   = "/workspace/project_root/output/dsyn_training_set.jsonl"
SAVE_DIR    = "/workspace/project_root/weights/checkpoints1"
FINAL_DIR   = "/workspace/project_root/weights/qwen_lora_final"
LOG_PATH    = "/workspace/project_root/weights/train_log.jsonl"

BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
WARMUP_STEPS = 50
MAX_EPOCHS   = 3
SAVE_EVERY   = 300
VAL_SPLIT    = 0.05
SEED         = 42

# Multi-Task Loss Weighting
SEG_LOSS_WEIGHT = 2.0 

# LoRA Params
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════
# 2. DATASET (Enriched with [SEG] Trigger)
# ══════════════════════════════════════════════════════════════
class SurgicalVQADataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            all_data = [json.loads(line) for line in f]
        self.data = [d for d in all_data if d.get("type") == "vqa_distillation" or "question" in d]
        self.old_prefix = "/scratch/m25cse012/CV_major"
        self.new_prefix = "/workspace"
        print(f"✅ Loaded {len(self.data)} samples for Multi-Task Training.", flush=True)

    def __len__(self):
        return len(self.data)

    def _fix_path(self, path):
        if path and isinstance(path, str):
            return path.replace(self.old_prefix, self.new_prefix).replace("//", "/")
        return path

    def __getitem__(self, idx, _retries=0):
        if _retries >= len(self.data):
            raise RuntimeError("❌ No valid images found!")

        item   = self.data[idx]
        g_path = self._fix_path(item.get("global_image_path"))

        if not g_path or not os.path.exists(g_path):
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)

        try:
            # We maintain 448 for the ViT branch, but Wavelet will use the tensor directly
            image = Image.open(g_path).convert("RGB").resize((448, 448))
            
            # Injection of [SEG] token into training answer
            ans = item.get("answer", "Surgical procedure.")
            answer = f"{ans} [SEG]" if "[SEG]" not in ans else ans

        except Exception:
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)

        return {
            "image":    image,
            "question": item.get("question", "Identify the tool."),
            "answer":   answer,
            # Placeholder for mask distillation (Dice Loss)
            "target_mask": torch.zeros((1, 1024, 1024)) 
        }

# ══════════════════════════════════════════════════════════════
# 3. COLLATE (Answer Masking & Tokenization)
# ══════════════════════════════════════════════════════════════
def make_collate_fn(processor):
    def collate_fn(batch):
        images    = [b["image"]    for b in batch]
        questions = [b["question"] for b in batch]
        answers   = [b["answer"]   for b in batch]

        prompt_texts = [f"<|vision_start|><|image_pad|><|vision_end|>Question: {q}\nAnswer: " for q in questions]
        full_texts   = [f"{p}{a}" for p, a in zip(prompt_texts, answers)]

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_enc = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        for i in range(len(batch)):
            p_len = (prompt_enc["attention_mask"][i] == 1).sum().item()
            labels[i, :p_len] = -100 # Mask prompt so we only train on answer + [SEG]

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels,
            "attention_mask": inputs["attention_mask"]
        }
    return collate_fn

# ══════════════════════════════════════════════════════════════
# 4. TRAINING ENGINE (Multi-Task Distillation)
# ══════════════════════════════════════════════════════════════
def train():
    device = "cuda"

    # A. Model Initialization (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # B. Module Integration (Wavelets & Decoder)
    # Wrap Qwen with the Wavelet branch (HH-band extraction)
    model = QwenVLWithWavelets(base_model).to(device)
    
    # Initialize the spatial mask generator
    mask_decoder = TransformerMaskDecoder(
        hidden_dim=base_model.config.hidden_size, mask_dim=256
    ).to(device)

    # C. PEFT/LoRA Wrapper
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, task_type=TaskType.CAUSAL_LM,
    )
    model.qwen = get_peft_model(model.qwen, lora_config)

    # D. Optimizer (Joint Optimization)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': LR},
        {'params': mask_decoder.parameters(), 'lr': LR}
    ], weight_decay=0.01)

    # ... Dataset Loader Setup ...
    full_ds = SurgicalVQADataset(DATA_PATH)
    train_ds, val_ds = random_split(full_ds, [int(len(full_ds)*0.95), len(full_ds)-int(len(full_ds)*0.95)])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=make_collate_fn(processor))

    model.train(); mask_decoder.train()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"--- Epoch {epoch} ---")
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # 1. Forward Pass: Extract Fused Hidden States (ViT + Wavelets)
            fused_states = model(inputs["input_ids"], inputs["pixel_values"])
            
            # 2. VQA Loss: Standard Language Modeling
            outputs = model.qwen(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                labels=inputs["labels"]
            )
            vqa_loss = outputs.loss

            # 3. Mask Loss: Triggered by [SEG] token
            seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
            seg_indices = (inputs["input_ids"] == seg_id).nonzero()
            
            mask_loss = torch.tensor(0.0).to(device)
            if seg_indices.size(0) > 0:
                # Grab the [SEG] hidden state from the fused sequence
                # fused_states shape: [B, Seq_Len, Hidden_Dim]
                seg_rep = fused_states[seg_indices[:,0], seg_indices[:,1], :].unsqueeze(1)
                
                # Pass Instruction (SEG) and Answer (Vision Feats) to Decoder
                pred_mask = mask_decoder(seg_rep, fused_states)
                
                # In full R2I, you compare pred_mask to Teacher's SAM mask here
                # mask_loss = F.binary_cross_entropy_with_logits(pred_mask, target_mask)

            # 4. Backward
            total_loss = (vqa_loss + (mask_loss * SEG_LOSS_WEIGHT)) / GRAD_ACCUM
            total_loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step(); optimizer.zero_grad()

    # Save final Adapter + Decoder
    model.qwen.save_pretrained(FINAL_DIR)
    torch.save(mask_decoder.state_dict(), f"{FINAL_DIR}/mask_decoder.bin")
    print(f"✅ Mission Complete. Model saved to {FINAL_DIR}")

if __name__ == "__main__":
    train
