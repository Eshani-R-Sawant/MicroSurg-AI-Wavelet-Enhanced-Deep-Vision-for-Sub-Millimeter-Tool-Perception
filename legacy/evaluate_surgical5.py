import torch, json, os, cv2, random, time, glob, math
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURATION ---
LORA_ADAPTER = "/workspace/project_root/weights/qwen_lora_final"
BASE_MODEL   = "/workspace/Qwen-VL"
TEST_JSONL   = "/workspace/project_root/output/dsyn_repaired_full.jsonl"
OUTPUT_DIR   = "/workspace/project_root/output/inference_results"
N_SAMPLES    = 10

IMAGE_ROOTS = [
    "/workspace/project_root/data/images",
    "/workspace/SurgMLLMBench/SurgMLLMBench",
    "/workspace/SurgMLLMBench",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UTILITIES ---

def is_valid_endo_frame(img_path: str) -> bool:
    try:
        img = cv2.imread(img_path)
        if img is None: return False
        if img.shape[0] < 100 or img.shape[1] < 100: return False
        if img.mean() < 20: return False
        return True
    except Exception:
        return False

def get_eval_images(n: int = 10) -> list:
    all_imgs = []
    for root in IMAGE_ROOTS:
        if os.path.exists(root):
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                all_imgs.extend(glob.glob(f"{root}/**/{ext}", recursive=True))
    valid = [p for p in all_imgs if is_valid_endo_frame(p)]
    print(f"✅ Valid endo frames found: {len(valid)}", flush=True)
    random.seed(int(time.time()))
    random.shuffle(valid)
    return valid

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
                        "answer":   d.get("answer", ""),
                    }
    except Exception as e:
        print(f"  ⚠️  GT lookup error: {e}", flush=True)
    print(f"✅ GT lookup: {len(lookup)} entries.", flush=True)
    return lookup

# --- IMAGE PROCESSING (WATERSHED & MASKING) ---

def run_watershed_analysis(img_path: str, idx: int):
    """Generates segmentation mask and masked image using Watershed."""
    img = cv2.imread(img_path)
    if img is None: return
    img_res = cv2.resize(img, (900, 560))
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_res, markers)
    
    # Create the binary mask
    mask = np.zeros(img_res.shape[:2], dtype=np.uint8)
    mask[markers > 1] = 255
    
    # Create the masked image (foreground only)
    masked_img = cv2.bitwise_and(img_res, img_res, mask=mask)

    # Naming convention
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    prefix = f"{idx + 1:02d}_{base_name}"
    
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_watershed_mask.jpg", mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_masked_image.jpg", masked_img)

# --- VISUALISATION (PREDICTION OVERLAY) ---

def save_viz(img_path: str, question: str, pred: str, gt: str, idx: int):
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.resize(img, (900, 560))
    h, w = img.shape[:2]

    panel = img.copy()
    cv2.rectangle(panel, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.75, img, 0.25, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Q:    {question[:90]}",  (10, h - 92), font, 0.44, (200, 200, 200), 1)
    cv2.putText(img, f"GT:   {gt[:90]}",         (10, h - 62), font, 0.44, (80, 255, 80),  1)
    cv2.putText(img, f"PRED: {pred[:90]}",        (10, h - 32), font, 0.44, (0, 220, 255),  1)
    cv2.rectangle(img, (0, 0), (140, 38), (0, 0, 0), -1)
    cv2.putText(img, f"Sample #{idx + 1}", (6, 26), font, 0.7, (255, 255, 0), 2)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out = f"{OUTPUT_DIR}/{idx + 1:02d}_{base_name}_result.jpg"
    cv2.imwrite(out, img)
    print(f"    💾 {out}", flush=True)

# --- GRAD-CAM ---

def get_target_layer(model):
    candidates = [
        lambda m: m.model.visual.blocks[-1].norm1,
        lambda m: m.model.visual.blocks[-1].norm2,
        lambda m: m.model.visual.blocks[-1],
        lambda m: m.visual.blocks[-1].norm1,
        lambda m: m.visual.blocks[-1],
        lambda m: [mod for name, mod in m.named_modules() if "visual" in name and isinstance(mod, torch.nn.LayerNorm)][-1],
    ]
    for fn in candidates:
        try:
            layer = fn(model)
            if layer is not None: return layer
        except Exception: continue
    return None

def run_gradcam(model, inputs: dict, img_path: str, idx: int):
    target = get_target_layer(model)
    if target is None: return
    act, grad = [None], [None]
    h1 = target.register_forward_hook(lambda m, i, o: act.__setitem__(0, o.detach().clone()))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grad.__setitem__(0, go[0].detach().clone()))

    try:
        ci = {k: (v.clone().float().requires_grad_(True) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else (v.clone() if isinstance(v, torch.Tensor) else v)) for k, v in inputs.items()}
        out = model(**ci)
        score = out.logits[0, -1, :].max()
        model.zero_grad()
        score.backward(retain_graph=True)

        if act[0] is not None and grad[0] is not None:
            w = grad[0].mean(dim=list(range(grad[0].dim() - 1)), keepdim=True)
            cam = torch.relu((w * act[0]).sum(-1).squeeze())
            if cam.numel() > 0:
                s = int(math.sqrt(cam.numel())) if cam.dim() == 1 else cam.shape[0]
                cam = cam.float().cpu().numpy()
                if cam.ndim == 1: cam = cam[:s * s].reshape(s, s)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam = cv2.resize(cam, (900, 560))
                img = cv2.resize(cv2.imread(img_path), (900, 560))
                heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                fused = cv2.addWeighted(img, 0.5, heat, 0.5, 0)
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_p = f"{OUTPUT_DIR}/{idx + 1:02d}_{base_name}_gradcam.jpg"
                cv2.imwrite(out_p, fused)
    except Exception as e:
        print(f"    ⚠️  GradCAM error: {e}", flush=True)
    finally:
        h1.remove()
        h2.remove()

# --- MAIN RUN ---

def run():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if os.path.exists(LORA_ADAPTER) and os.path.exists(os.path.join(LORA_ADAPTER, "adapter_config.json")):
        print(f"✅ Loading LoRA adapter: {LORA_ADAPTER}", flush=True)
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    else:
        print("⚠️ Running base model.", flush=True)
        model = base_model

    model.eval()
    img_paths = get_eval_images(N_SAMPLES)
    gt_lookup = build_gt_lookup()
    if not img_paths: return

    PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>Question: {q}\nAnswer with the tool name, action, and tissue.\nAnswer:"

    for i, img_path in enumerate(img_paths):
        bname = os.path.basename(img_path)
        gt_info = gt_lookup.get(bname, {})
        question = gt_info.get("question", "Identify the surgical tool, action, and tissue.")
        gt_text = gt_info.get("answer", "")

        image = Image.open(img_path).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(q=question)
        raw = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=120, do_sample=False, repetition_penalty=1.15, pad_token_id=processor.tokenizer.eos_token_id)
        lat = time.time() - t0
        pred = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

        # Step 1: Text Viz
        save_viz(img_path, question, pred, gt_text or "N/A", i)
        # Step 2: Model Attention Viz
        run_gradcam(model, inputs, img_path, i)
        # Step 3: Segmentation Viz
        run_watershed_analysis(img_path, i)

        print(f"✅ Sample {i+1} completed: {bname}", flush=True)

if __name__ == "__main__":
    run()
