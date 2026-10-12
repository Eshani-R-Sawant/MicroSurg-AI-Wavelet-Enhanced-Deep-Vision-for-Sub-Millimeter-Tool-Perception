import torch, json, os, re, cv2, random, time, glob
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sklearn.metrics import average_precision_score

# ==========================================
# CONFIG
# ==========================================
CHECKPOINT  = "/workspace/project_root/weights/qwen_final"
FALLBACK    = "/workspace/Qwen-VL"
TEST_JSONL  = "/workspace/project_root/output/dsyn_training_set.jsonl"
OUTPUT_DIR  = "/workspace/project_root/output/inference_results"
N_SAMPLES   = 5
SEED        = 42

# WHERE YOUR REAL IMAGES LIVE ON THE SERVER
IMAGE_ROOTS = [
    "/workspace/project_root/data/images",
    "/workspace/SurgMLLMBench/SurgMLLMBench",
    "/workspace/SurgMLLMBench",
]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# BUILD REAL IMAGE POOL FROM FILESYSTEM
# ==========================================
def get_random_images(n=5):
    all_imgs = []
    for root in IMAGE_ROOTS:
        if os.path.exists(root):
            found = glob.glob(f"{root}/**/*.jpg", recursive=True) + \
                    glob.glob(f"{root}/**/*.png", recursive=True)
            all_imgs.extend(found)
    print(f"✅ Found {len(all_imgs)} real images on server.", flush=True)
    random.seed(SEED)
    random.shuffle(all_imgs)
    # Filter: readable + not too small
    valid = []
    for p in all_imgs:
        if len(valid) >= n: break
        try:
            img = cv2.imread(p)
            if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                valid.append(p)
        except: continue
    return valid

# ==========================================
# LOAD GT ANSWERS FROM JSONL (if available)
# ==========================================
def build_gt_lookup():
    """Maps image basename → GT answer from JSONL."""
    lookup = {}
    try:
        with open(TEST_JSONL, "r") as f:
            for line in f:
                d = json.loads(line)
                p = d.get("global_image_path","")
                if p and "answer" in d:
                    lookup[os.path.basename(p)] = {
                        "question": d.get("question","Identify the surgical tool, action, tissue."),
                        "answer":   d.get("answer",""),
                        "bbox":     d.get("global_bbox")
                    }
    except: pass
    print(f"✅ GT lookup: {len(lookup)} entries.", flush=True)
    return lookup

# ==========================================
# HELPERS
# ==========================================
def parse_bbox(text):
    match = re.search(r'[\[\(]([\d\.\s,]+)[\]\)]', text)
    if match:
        try:
            c = [float(x.strip()) for x in match.group(1).split(',')]
            if len(c) == 4:
                return [v/1000.0 if v > 1.1 else v for v in c]
        except: pass
    return None

def word_f1(pred, gt):
    stops = {"the","a","an","is","in","on","at","of","and","to","with",
             "this","image","shows","i","see","answer","there","no","present"}
    pw = set(pred.lower().split()) - stops
    gw = set(gt.lower().split()) - stops
    if not gw or not pw: return 0.0
    ov = pw & gw
    pr = len(ov)/len(pw)
    rc = len(ov)/len(gw)
    return 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0

def compute_iou(p, g):
    py1,px1,py2,px2 = [max(0,min(1,v)) for v in p]
    gy1,gx1,gy2,gx2 = [max(0,min(1,v)) for v in g]
    i = max(0,min(py2,gy2)-max(py1,gy1))*max(0,min(px2,gx2)-max(px1,gx1))
    u = (py2-py1)*(px2-px1)+(gy2-gy1)*(gx2-gx1)-i
    return i/u if u > 0 else 0.0

def save_viz(img_path, question, pred, gt, bbox, idx):
    img = cv2.imread(img_path)
    if img is None: return
    # FILL THE FULL CANVAS - no more tiny thumbnails
    img = cv2.resize(img, (900, 560))
    h, w = img.shape[:2]
    # Draw bbox on the ACTUAL image (not black canvas)
    if bbox:
        y1=int(bbox[0]*h); x1=int(bbox[1]*w)
        y2=int(bbox[2]*h); x2=int(bbox[3]*w)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)
        cv2.putText(img,"PRED TOOL",(x1,max(y1-8,15)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
    # Text panel at bottom
    panel = img.copy()
    cv2.rectangle(panel,(0,h-120),(w,h),(0,0,0),-1)
    cv2.addWeighted(panel,0.75,img,0.25,0,img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,f"Q:    {question[:90]}",  (10,h-92),font,0.44,(200,200,200),1)
    cv2.putText(img,f"GT:   {gt[:90]}",        (10,h-62),font,0.44,(80,255,80),1)
    cv2.putText(img,f"PRED: {pred[:90]}",      (10,h-32),font,0.44,(0,220,255),1)
    # Sample badge
    cv2.rectangle(img,(0,0),(140,38),(0,0,0),-1)
    cv2.putText(img,f"Sample #{idx+1}",(6,26),font,0.7,(255,255,0),2)
    out = f"{OUTPUT_DIR}/result_{idx+1:02d}.jpg"
    cv2.imwrite(out, img)
    print(f"   💾 {out}", flush=True)

def run_gradcam(model, inputs, img_path, idx):
    # Try known Qwen2-VL layer paths
    target = None
    for fn in [lambda m: m.visual.blocks[-1].norm1,
               lambda m: m.visual.merger,
               lambda m: m.model.visual.blocks[-1].norm1]:
        try: target = fn(model); break
        except: continue
    if target is None:
        print("   ⚠️ GradCAM: layer not found", flush=True); return

    act, grad = [None], [None]
    h1 = target.register_forward_hook(lambda m,i,o: act.__setitem__(0,o.detach().clone()))
    h2 = target.register_full_backward_hook(lambda m,gi,go: grad.__setitem__(0,go[0].detach().clone()))
    try:
        ci = {k: v.clone().requires_grad_(True)
              if isinstance(v,torch.Tensor) and v.is_floating_point()
              else (v.clone() if isinstance(v,torch.Tensor) else v)
              for k,v in inputs.items()}
        out   = model(**ci)
        score = out.logits[0,-1,:].max()
        model.zero_grad(); score.backward(retain_graph=True)

        if act[0] is None or grad[0] is None: return
        w   = grad[0].mean(dim=(0,1),keepdim=True)
        cam = torch.relu((w*act[0]).sum(-1).squeeze())
        s   = int(cam.numel()**0.5)
        cam = cam[:s*s].reshape(s,s).float().cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        cam = cv2.resize(cam,(900,560))

        img  = cv2.resize(cv2.imread(img_path),(900,560))
        heat = cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
        fused= cv2.addWeighted(img,0.5,heat,0.5,0)
        cv2.putText(fused,"GRAD-CAM: Model Attention",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        cv2.putText(fused,"Red=High Attention  Blue=Low Attention",(10,58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        out_p = f"{OUTPUT_DIR}/gradcam_{idx+1:02d}.jpg"
        cv2.imwrite(out_p, fused)
        print(f"   🔥 GradCAM: {out_p}", flush=True)
    except Exception as e:
        print(f"   ⚠️ GradCAM error: {e}", flush=True)
    finally:
        h1.remove(); h2.remove()

# ==========================================
# MAIN
# ==========================================
def run():
    model_dir = CHECKPOINT if os.path.exists(CHECKPOINT) else FALLBACK
    print(f"🔄 Loading: {model_dir}", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(FALLBACK)
    model.eval()
    print("✅ Model ready.\n", flush=True)

    # Get REAL diverse images from filesystem
    img_paths = get_random_images(N_SAMPLES)
    gt_lookup = build_gt_lookup()

    if not img_paths:
        print("❌ No images found. Check IMAGE_ROOTS paths.", flush=True)
        return

    f1s, ious, lats = [], [], []
    det_s, det_l = [], []
    log = []
    Q = "Identify the surgical tool, action, and tissue."

    for i, img_path in enumerate(img_paths):
        bname = os.path.basename(img_path)
        gt_info = gt_lookup.get(bname, {})
        question = gt_info.get("question", Q)
        gt_text  = gt_info.get("answer", "")
        gt_bbox  = gt_info.get("bbox")

        print(f"\n{'='*55}", flush=True)
        print(f"[{i+1}/{N_SAMPLES}] {bname}", flush=True)
        if gt_text: print(f"GT:   {gt_text[:90]}", flush=True)

        image  = Image.open(img_path).convert("RGB")
        prompt = (f"<|vision_start|><|image_pad|><|vision_end|>"
                  f"Question: {question}\n"
                  f"Answer with tool name, action, tissue, and bounding box "
                  f"[y1, x1, y2, x2] normalized 0.0-1.0:\nAnswer:")

        raw    = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in raw.items()}

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=120,
                                 do_sample=False, repetition_penalty=1.15)
        lat = time.time()-t0
        lats.append(lat)

        pred = processor.batch_decode(gen[:,inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
        print(f"PRED: {pred[:90]}", flush=True)
        print(f"⏱️  {lat:.1f}s", flush=True)

        # Metrics
        f1 = word_f1(pred, gt_text) if gt_text else 0.0
        f1s.append(f1)
        print(f"Word-F1: {f1:.3f}", flush=True)

        pred_bbox = parse_bbox(pred)
        iou_val   = 0.0
        has_gt    = gt_bbox and any(v>0 for v in gt_bbox)
        det_l.append(1 if has_gt else 0)
        det_s.append(1.0 if pred_bbox else 0.0)
        if pred_bbox and has_gt:
            norm_gt = [v/1024.0 for v in gt_bbox]
            iou_val = compute_iou(pred_bbox, norm_gt)
            ious.append(iou_val)
            print(f"IoU:     {iou_val:.3f}", flush=True)

        save_viz(img_path, question, pred, gt_text or "N/A", pred_bbox, i)
        run_gradcam(model, inputs, img_path, i)

        log.append({"sample":i+1,"image":bname,"question":question,
                    "ground_truth":gt_text,"prediction":pred,
                    "word_f1":round(f1,4),"iou":round(iou_val,4),
                    "latency_s":round(lat,2)})

    # Summary
    avg_f1  = np.mean(f1s)*100 if f1s else 0
    avg_iou = np.mean(ious) if ious else 0
    avg_lat = np.mean(lats)
    try:    mAP = average_precision_score(det_l, det_s)
    except: mAP = 0.0

    print(f"\n{'='*55}", flush=True)
    print(f"  📊 FINAL RESULTS", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Samples       : {len(log)}", flush=True)
    print(f"  Word-F1 Acc   : {avg_f1:.1f}%", flush=True)
    print(f"  Mean IoU      : {avg_iou:.4f}", flush=True)
    print(f"  mAP           : {mAP:.4f}", flush=True)
    print(f"  Avg Latency   : {avg_lat:.2f}s/image", flush=True)
    print(f"  Images        : {OUTPUT_DIR}/result_*.jpg", flush=True)
    print(f"  GradCAM       : {OUTPUT_DIR}/gradcam_*.jpg", flush=True)
    print(f"{'='*55}", flush=True)

    with open(f"{OUTPUT_DIR}/summary.json","w") as f:
        json.dump({"word_f1_pct":round(avg_f1,2),"mIoU":round(avg_iou,4),
                   "mAP":round(mAP,4),"avg_latency_s":round(avg_lat,2),
                   "samples":log}, f, indent=2)
    print(f"✅ Saved: {OUTPUT_DIR}/summary.json", flush=True)

if __name__ == "__main__":
    run()























