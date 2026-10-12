"""
Step 03.1: Multi-Tier Priority Knowledge Loading & Student Gap Filling.

Merges multi-tiered knowledge sources into a unified reasoning dataset:
- Priority 1: Ground-truth DSYN dataset annotations (Gold quality).
- Priority 2: Gemini Teacher Oracle generated reasoning & bounding boxes.
- Priority 3: Local Qwen2-VL student model gap-filling for remaining crops.
Outputs final unified JSONL (e.g., final_unified_7531.jsonl).
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def load_prioritized_knowledge(dsyn_path: Path, gemini_path: Path, gemini_cap: int = 1000):
    master_knowledge = {}

    # Priority 1: DSYN Gold metadata
    if dsyn_path.exists():
        with open(dsyn_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    crop_name = Path(item.get("crop_image", item.get("crop_id", ""))).name
                    convs = item.get("conversations", [])
                    q_text = next((c["value"] for c in convs if c["from"] == "user"), "Identify surgical tool.")
                    a_text = next((c["value"] for c in convs if c["from"] == "assistant"), "None")

                    if "None" not in a_text:
                        master_knowledge[crop_name] = {
                            "question": q_text,
                            "answer": a_text,
                            "source": "PRIORITY_1_DSYN",
                            "original_meta": item,
                        }
                except Exception:
                    continue
        print(f"💎 Priority 1: Loaded {len(master_knowledge)} Gold entries from DSYN dataset.")

    # Priority 2: Gemini Teacher Oracle
    gem_count = 0
    if gemini_path.exists():
        with open(gemini_path, "r", encoding="utf-8") as f:
            for line in f:
                if gem_count >= gemini_cap:
                    break
                try:
                    item = json.loads(line)
                    fname = item.get("crop_id", "")
                    if fname and fname not in master_knowledge and "None" not in item.get("expert_answer", ""):
                        master_knowledge[fname] = {
                            "question": "Identify the surgical tool, action, and tissue.",
                            "answer": item.get("expert_answer", ""),
                            "source": "PRIORITY_2_GEMINI",
                            "bbox": item.get("expert_bbox"),
                        }
                        gem_count += 1
                except Exception:
                    continue
        print(f"🤖 Priority 2: Added {gem_count} Gemini Teacher entries.")

    return master_knowledge


def run_multi_tier_assembly(
    crop_dir: str = None,
    dsyn_input: str = None,
    gemini_input: str = None,
    model_path: str = None,
    output_file: str = None,
):
    base_dir = Path(__file__).resolve().parent.parent
    c_dir = Path(crop_dir) if crop_dir else base_dir / "output" / "micro_crops"
    dsyn_p = Path(dsyn_input) if dsyn_input else base_dir / "data" / "dsyn_dataset.jsonl"
    gemini_p = Path(gemini_input) if gemini_input else base_dir / "output" / "syn_vqa_pairs.jsonl"
    out_p = Path(output_file) if output_file else base_dir / "output" / "final_unified_7531.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    knowledge_base = load_prioritized_knowledge(dsyn_p, gemini_p)

    all_crops = sorted(list(c_dir.glob("*.jpg")))
    missing_crops = [c for c in all_crops if c.name not in knowledge_base]
    print(f"📦 Total crops: {len(all_crops)} | In Knowledge Base: {len(knowledge_base)} | Need Gap-Fill: {len(missing_crops)}")

    # Student model gap fill if needed
    model, processor = None, None
    if missing_crops:
        m_path = model_path or os.getenv("MODEL_PATH", "/workspace/Qwen-VL")
        if os.path.exists(m_path):
            print(f"🚀 Loading Qwen-VL for Student Gap-Filling from: {m_path}")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                m_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(m_path)
        else:
            print(f"⚠️ Model path {m_path} not found. Skipping gap filling for unindexed crops.")

    with open(out_p, "w", encoding="utf-8") as f_out:
        for img_path in tqdm(all_crops, desc="Assembling Master Dataset"):
            fname = img_path.name
            if fname in knowledge_base:
                entry = knowledge_base[fname]
                entry["crop_id"] = fname
            elif model is not None and processor is not None:
                image = Image.open(img_path).convert("RGB")
                q_text = "Identify the surgical tool, action, and tissue. Format: Answer: [Tool] is [Action] on [Tissue]."
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q_text}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=128)
                a_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                entry = {"crop_id": fname, "question": q_text, "answer": a_text, "source": "PRIORITY_3_QWEN"}
            else:
                continue

            f_out.write(json.dumps(entry) + "\n")

    print(f"🏁 Master Dataset Ready: {out_p}")


if __name__ == "__main__":
    run_multi_tier_assembly()
