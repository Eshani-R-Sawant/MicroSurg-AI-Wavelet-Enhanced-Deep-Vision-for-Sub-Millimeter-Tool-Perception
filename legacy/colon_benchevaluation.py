import torch, json, os, time, re
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# --- CONFIG ---
HF_TOKEN     = os.environ.get("HF_TOKEN", "")  # Set via environment variable; never hardcode
LORA_ADAPTER = "/workspace/project_root/weights/checkpoints1/step_6900"
BASE_MODEL   = "/workspace/Qwen-VL"
OUTPUT_DIR   = "/workspace/project_root/output/colon_bench_results"
DATASET_NAME = "ajhamdi/colon-bench"
SUB_TASK     = "vqa-unprompted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load semantic model for scoring
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantic_choice(model_sentence, choices_dict):
    sentence_embedding = similarity_model.encode(model_sentence, convert_to_tensor=True)
    choice_letters = [k for k in choices_dict.keys() if choices_dict[k]]
    choice_texts = [choices_dict[k] for k in choice_letters]
    choice_embeddings = similarity_model.encode(choice_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(sentence_embedding, choice_embeddings)[0]
    return choice_letters[torch.argmax(cosine_scores).item()]

def run_streaming_eval():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    print("🔄 Loading Model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER).eval()

    print(f"📂 Streaming {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, SUB_TASK, split="test", token=HF_TOKEN, streaming=True)

    results, correct_count, total = [], 0, 0

    for item in dataset:
        total += 1
        
        # FIX: Handle the video/image key. 
        # In ColonBench, 'image' might be nested or named 'video' or 'frame'.
        # We try to get the visual content dynamically.
        visual_key = 'image' if 'image' in item else 'video'
        
        # If it's a list (video frames), take the middle/first frame for VQA
        raw_visual = item[visual_key]
        if isinstance(raw_visual, list):
            image = raw_visual[len(raw_visual)//2].convert("RGB")
        else:
            image = raw_visual.convert("RGB")

        choices = {"A": item['choice_A'], "B": item['choice_B'], "C": item['choice_C'], "D": item['choice_D'], "E": item.get('choice_E')}
        options_text = ", ".join([f"{k}: {v}" for k, v in choices.items() if v])
        
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>Question: {item['question']}\nOptions: {options_text}\nAnswer concisely in one sentence:"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.pad_token_id)
        
        gen_text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pred_letter = get_semantic_choice(gen_text, choices)
        
        is_correct = (pred_letter == item['answer'])
        if is_correct: correct_count += 1

        if total % 5 == 0:
            print(f"Sample {total} | Acc: {(correct_count/total)*100:.2f}% | Pred: {pred_letter}")

        results.append({"id": total, "pred": pred_letter, "gt": item['answer'], "text": gen_text})
        
        del inputs, output
        if total % 10 == 0: torch.cuda.empty_cache()

    print(f"✅ Final Accuracy: {(correct_count/total)*100:.2f}%")
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump({"accuracy": (correct_count/total)*100, "details": results}, f, indent=4)

if __name__ == "__main__":
    run_streaming_eval()
