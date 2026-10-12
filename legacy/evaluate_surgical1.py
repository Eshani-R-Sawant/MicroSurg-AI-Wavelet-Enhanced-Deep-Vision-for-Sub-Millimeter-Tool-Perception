import torch, json, os, re, cv2, random, time, glob
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# --- UPDATED PATHS ---
LORA_ADAPTER = "/workspace/project_root/weights/checkpoints1/step_6900"
BASE_MODEL = "/workspace/Qwen-VL"
TEST_JSONL = "/workspace/project_root/output/clean_vqa_merged.jsonl"
OUTPUT_DIR = "/workspace/project_root/output/inference_results1"
N_SAMPLES = 10   

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_endo_frame(img_path: str) -> bool:
    name = os.path.basename(img_path).lower()
    skip_patterns = ["mask", "watershed", "label", "annotation", "depth", "flow", "binary"]
    if any(p in name for p in skip_patterns): 
        return False
    try:
        img = cv2.imread(img_path)
        if img is None: 
            return False
        if img.mean() < 20: 
            return False
        return True
    except Exception: 
        return False

def get_eval_images(n: int = 10) -> list[str]:
    all_imgs = []
    roots = ["/workspace/project_root/data/images", "/workspace/SurgMLLMBench"]
    for root in roots:
        if os.path.exists(root):
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                all_imgs.extend(glob.glob(f"{root}/**/{ext}", recursive=True))
    valid = [p for p in all_imgs if is_valid_endo_frame(p)]
    random.seed(int(time.time()))
    random.shuffle(valid)
    return valid[:n]

def build_gt_lookup() -> dict:
    lookup = {}
    try:
        with open(TEST_JSONL) as f:
            for line in f:
                d = json.loads(line)
                p = d.get("global_image_path", "")
                if p:
                    lookup[os.path.basename(p)] = {
                        "question": d.get("question", "Identify the surgical tool, action, and tissue."),
                        "answer": d.get("answer", ""),
                    }
    except Exception: 
        pass
    return lookup

def save_viz(img_path: str, question: str, pred: str, gt: str, idx: int):
    img = cv2.imread(img_path)
    if img is None: 
        return
    img = cv2.resize(img, (900, 560))
    h, w = img.shape[:2]
    
    # Text Panel (No bounding boxes drawn here)
    panel = img.copy()
    cv2.rectangle(panel, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.75, img, 0.25, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Q: {question[:85]}", (10, h-90), font, 0.45, (220, 220, 220), 1)
    cv2.putText(img, f"GT: {gt[:85]}", (10, h-60), font, 0.45, (0, 255, 0), 1)
    cv2.putText(img, f"PRED: {pred[:85]}", (10, h-30), font, 0.45, (0, 255, 255), 1)
    
    cv2.imwrite(f"{OUTPUT_DIR}/result_{idx+1:02d}.jpg", img)

def run():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"🔄 Loading model from: {LORA_ADAPTER}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        quantization_config=bnb, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    
    # Load LoRA from the 'best' checkpoint
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    model.eval()

    img_paths = get_eval_images(N_SAMPLES)
    gt_lookup = build_gt_lookup()
    f1s, lats, log = [], [], []

    for i, img_path in enumerate(img_paths):
        bname = os.path.basename(img_path)
        gt_info = gt_lookup.get(bname, {})
        q = gt_info.get("question", "Identify the surgical tool, action, and tissue.")
        gt = gt_info.get("answer", "")

        image = Image.open(img_path).convert("RGB")
        prompt = f"<|vision_start|><|image_pad|><|vision_end|>Question: {q}\nAnswer:"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=100)
        lat = time.time() - t0
        
        pred = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        
        
        
        
        lats.append(lat)
        save_viz(img_path, q, pred, gt, i)
        
        log.append({"id": i+1, "img": bname, "pred": pred})

    # Final summary with only valid metrics
    print(f"\n📊 RESULTS: Avg Latency: {np.mean(lats):.2f}s")
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(log, f, indent=4)

if __name__ == "__main__":
    run()
