"""
Semantic Similarity Benchmark Evaluation on EndoVis2018.

Evaluates predicted surgical responses against EndoVis2018 ground truth answers
using SentenceTransformer semantic embedding cosine similarity.
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent.parent

LORA_ADAPTER = os.getenv("LORA_ADAPTER", str(BASE_DIR / "weights" / "checkpoints1" / "step_6900"))
BASE_MODEL = os.getenv("BASE_MODEL", "/workspace/Qwen-VL")
ENDOVIS_DIR = os.getenv("ENDOVIS_DIR", str(BASE_DIR / "SurgMLLMBench" / "SurgMLLMBench" / "EndoVis2018"))
ANNOTATIONS = os.getenv("ANNOTATIONS", f"{ENDOVIS_DIR}/test_vqa.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "output" / "semantic_eval"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_semantic_evaluation():
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔄 Loading Fine-Tuned ZWZ Model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if os.path.exists(LORA_ADAPTER):
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER).eval()
    else:
        model = base_model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    if not os.path.exists(ANNOTATIONS):
        print(f"❌ Annotations file not found: {ANNOTATIONS}")
        return

    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    results = []
    total_similarity = 0.0

    print(f"🚀 Semantically evaluating {len(gt_data)} samples...")
    for i, item in enumerate(gt_data):
        img_path = os.path.join(ENDOVIS_DIR, item["image"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>Question: {item['question']}\nAnswer:"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)

        pred_text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        gt_text = item.get("answer", "")

        emb1 = similarity_model.encode(pred_text, convert_to_tensor=True)
        emb2 = similarity_model.encode(gt_text, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        total_similarity += score
        is_correct = score > 0.8

        results.append({
            "gt": gt_text,
            "pred": pred_text,
            "similarity_score": round(score, 4),
            "match": is_correct,
        })

    if results:
        mean_sim = total_similarity / len(results)
        accuracy = (sum(1 for r in results if r["match"]) / len(results)) * 100
        print(f"\n✅ EndoVis2018 Semantic Report: Mean Sim={mean_sim:.4f}, Accuracy (>0.8)={accuracy:.2f}%")
        with open(f"{OUTPUT_DIR}/semantic_report.json", "w", encoding="utf-8") as f:
            json.dump({"mean_similarity": mean_sim, "accuracy": accuracy, "data": results}, f, indent=4)


if __name__ == "__main__":
    run_semantic_evaluation()
