"""
Streaming Benchmark Evaluation on ColonBench (ajhamdi/colon-bench).

Evaluates the fine-tuned ZWZ model on endoscopic video/image VQA unprompted MCQ questions
using SentenceTransformer semantic cosine choice selection.
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent.parent

# Read HF token securely from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", str(BASE_DIR / "weights" / "checkpoints1" / "step_6900"))
BASE_MODEL = os.getenv("BASE_MODEL", "/workspace/Qwen-VL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "output" / "colon_bench_results"))
DATASET_NAME = "ajhamdi/colon-bench"
SUB_TASK = "vqa-unprompted"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_streaming_eval():
    print("🔄 Loading SentenceTransformer for Semantic Choice Scorer...")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_semantic_choice(model_sentence, choices_dict):
        sentence_embedding = similarity_model.encode(model_sentence, convert_to_tensor=True)
        choice_letters = [k for k in choices_dict.keys() if choices_dict[k]]
        choice_texts = [choices_dict[k] for k in choice_letters]
        choice_embeddings = similarity_model.encode(choice_texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sentence_embedding, choice_embeddings)[0]
        return choice_letters[torch.argmax(cosine_scores).item()]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if torch.cuda.is_available() else None

    print(f"🔄 Loading Base Model from: {BASE_MODEL}...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    if os.path.exists(LORA_ADAPTER):
        print(f"✅ Loading Specialist Adapter from: {LORA_ADAPTER}...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER).eval()
    else:
        model = base_model.eval()

    print(f"📂 Streaming {DATASET_NAME} [{SUB_TASK}]...")
    dataset = load_dataset(DATASET_NAME, SUB_TASK, split="test", token=HF_TOKEN, streaming=True)

    results, correct_count, total = [], 0, 0
    for item in dataset:
        total += 1
        visual_key = "image" if "image" in item else "video"
        raw_visual = item[visual_key]
        if isinstance(raw_visual, list):
            image = raw_visual[len(raw_visual) // 2].convert("RGB")
        else:
            image = raw_visual.convert("RGB")

        choices = {
            "A": item.get("choice_A"),
            "B": item.get("choice_B"),
            "C": item.get("choice_C"),
            "D": item.get("choice_D"),
            "E": item.get("choice_E"),
        }
        options_text = ", ".join([f"{k}: {v}" for k, v in choices.items() if v])
        prompt = (
            f"<|vision_start|><|image_pad|><|vision_end|>"
            f"Question: {item['question']}\nOptions: {options_text}\nAnswer concisely in one sentence:"
        )

        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.pad_token_id)

        gen_text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pred_letter = get_semantic_choice(gen_text, choices)

        is_correct = (pred_letter == item.get("answer"))
        if is_correct:
            correct_count += 1

        if total % 10 == 0:
            print(f"Sample {total} | Acc: {(correct_count / total) * 100:.2f}% | Pred: {pred_letter}")

        results.append({
            "id": total,
            "pred": pred_letter,
            "gt": item.get("answer"),
            "text": gen_text,
            "correct": is_correct,
        })

        del inputs, output
        if total % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_acc = (correct_count / total * 100) if total > 0 else 0.0
    print(f"✅ Final ColonBench Accuracy: {final_acc:.2f}% ({correct_count}/{total})")

    out_file = f"{OUTPUT_DIR}/results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": final_acc, "total": total, "details": results}, f, indent=4)


if __name__ == "__main__":
    run_streaming_eval()
