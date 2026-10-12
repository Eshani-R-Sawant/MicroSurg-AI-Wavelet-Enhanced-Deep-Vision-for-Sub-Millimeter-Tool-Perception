import torch, json, os, cv2, random, time, glob, math
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score

# --- CONFIGURATION ---
# Point this to the base folder where your checkpoints are stored
LORA_BASE_PATH = "/workspace/project_root/weights/checkpoints1/step_6900"
BASE_MODEL_PATH = "/workspace/Qwen-VL"
TEST_JSONL      = "/workspace/project_root/output/dsyn_repaired_full.jsonl"
OUTPUT_DIR      = "/workspace/project_root/output/inference_results"
N_SAMPLES       = 40

IMAGE_ROOTS = [
    "/workspace/project_root/data/images",
    "/workspace/SurgMLLMBench/SurgMLLMBench",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER: FIND VALID LORA PATH ---
def get_valid_lora_path(base_path):
    """Recursively finds the folder containing adapter_config.json"""
    if os.path.exists(os.path.join(base_path, "adapter_config.json")):
        return base_path
    
    # Check for subdirectories like 'checkpoint-6900'
    subdirs = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if subdirs:
        # Sort by step number to get the latest one
        subdirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
        for s in subdirs:
            if os.path.exists(os.path.join(s, "adapter_config.json")):
                print(f"🔎 Found config in subdirectory: {s}")
                return s
    return None

# --- MODULE 1: MEDICAL EVALUATION ---
def compute_medical_metrics(preds, gts):
    if not preds or not gts:
        return {"medical_bert_f1": 0.0, "exact_match": 0.0}
    
    # Exact Match
    em = sum(1 for p, g in zip(preds, gts) if p.lower().strip() == g.lower().strip()) / len(preds)
    
    # SciBERT Medical Similarity
    try:
        P, R, F1 = bert_score(preds, gts, model_type="allenai/scibert_scidict", 
                              lang="en", device="cuda" if torch.cuda.is_available() else "cpu")
        f1_score = round(F1.mean().item(), 4)
    except Exception as e:
        print(f"⚠️ BERTScore error: {e}")
        f1_score = 0.0

    return {"medical_bert_f1": f1_score, "exact_match": round(em, 4)}

# --- MODULE 2: VISUALIZATION & SEGMENTATION ---
def save_grouped_outputs(img_path, question, pred, gt, base_name, idx):
    img = cv2.imread(img_path)
    if img is None: return
    img_res = cv2.resize(img, (900, 560))
    h, w = img_res.shape[:2]

    # 1. PRED Overlay
    ovl = img_res.copy()
    cv2.rectangle(ovl, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ovl, 0.75, img_res, 0.25, 0, img_res)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_res, f"Q: {question[:85]}", (10, h-92), font, 0.44, (200,200,200), 1)
    cv2.putText(img_res, f"GT: {gt[:85]}", (10, h-62), font, 0.44, (80, 255, 80), 1)
    cv2.putText(img_res, f"PR: {pred[:85]}", (10, h-32), font, 0.44, (0, 220, 255), 1)
    cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_PRED.jpg", img_res)

    # 2. MASK & WATERSHED
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_MASK.jpg", mask)
    
    # Watershed logic
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_trans, 0.7*dist_trans.max(), 255, 0)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[unknown == 255] = 0
    img_ws = img.copy()
    cv2.watershed(img_ws, markers)
    img_ws[markers == -1] = [0, 0, 255]
    cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_WATERSHED.jpg", img_ws)

def run_gradcam(model, inputs, img_path, base_name, idx):
    # Dynamically locate visual layer
    target = None
    for fn in [lambda m: m.model.visual.blocks[-1].norm1, lambda m: m.visual.blocks[-1]]:
        try: target = fn(model); break
        except: continue
    if target is None: return

    act, grad = [None], [None]
    h1 = target.register_forward_hook(lambda m, i, o: act.__setitem__(0, o.detach().clone()))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grad.__setitem__(0, go[0].detach().clone()))

    try:
        ci = {k: v.clone().float().requires_grad_(True) if (isinstance(v, torch.Tensor) and v.is_floating_point()) 
              else (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        out = model(**ci)
        out.logits[0, -1, :].max().backward()

        if act[0] is not None and grad[0] is not None:
            w = grad[0].mean(dim=list(range(grad[0].dim() - 1)), keepdim=True)
            cam = torch.relu((w * act[0]).sum(-1).squeeze())
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam.cpu().numpy(), (900, 560))), cv2.COLORMAP_JET)
            fused = cv2.addWeighted(cv2.resize(cv2.imread(img_path), (900, 560)), 0.5, heat, 0.5, 0)
            cv2.imwrite(f"{OUTPUT_DIR}/S{idx+1:02d}_{base_name}_GRADCAM.jpg", fused)
    except: pass
    finally: h1.remove(); h2.remove()

# --- MAIN RUN ---
def run():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    
    print("🚀 Loading Base Model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, quantization_config=bnb_config, 
                                                                 device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # Smart LoRA Loading
    valid_lora = get_valid_lora_path(LORA_BASE_PATH)
    if valid_lora:
        print(f"✅ Loading Specialist Model from: {valid_lora}")
        model = PeftModel.from_pretrained(base_model, valid_lora)
    else:
        print("⚠️ No config found! Running ZERO-SHOT Baseline.")
        model = base_model
    
    model.eval()

    # Image Selection
    all_imgs = []
    for r in IMAGE_ROOTS:
        for ext in ["*.jpg", "*.png"]: all_imgs.extend(glob.glob(f"{r}/**/{ext}", recursive=True))
    random.shuffle(all_imgs)
    test_set = all_imgs[:N_SAMPLES]

    all_preds, all_gts = [], []

    for i, p in enumerate(test_set):
        bname = os.path.splitext(os.path.basename(p))[0]
        image = Image.open(p).convert("RGB")
        prompt = "<|vision_start|><|image_pad|><|vision_end|>Question: Identify tool, action, tissue.\nAnswer:"
        
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64)
            pred = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        
        all_preds.append(pred)
        all_gts.append("Sample GT") # Replace with your JSONL lookup logic if needed

        save_grouped_outputs(p, "Identify tool, action, tissue.", pred, "Sample GT", bname, i)
        run_gradcam(model, inputs, p, bname, i)
        print(f"Done {i+1}/{N_SAMPLES}: {bname}")

    print("\n--- FINAL EVAL ---")
    metrics = compute_medical_metrics(all_preds, all_gts)
    print(f"Medical BERTScore: {metrics['medical_bert_f1']}")
    print(f"Exact Match: {metrics['exact_match']}")

if __name__ == "__main__":
    run()
