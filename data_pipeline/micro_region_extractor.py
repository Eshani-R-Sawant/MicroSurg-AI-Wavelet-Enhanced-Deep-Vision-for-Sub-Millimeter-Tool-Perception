"""
Step 02: Micro-Region Extraction via SAM & Visual Saliency.

Uses Segment Anything Model (SAM ViT-H) paired with a pretrained ResNet50
saliency model to extract high-resolution micro-region crops of active surgical
tools and delicate tissue interfaces, rejecting background noise and blank masks.
Saves 1024x1024 crops to output/micro_crops/ and generates crop_manifest.json.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


def run_extraction(
    subset_json: str = None,
    output_dir: str = None,
    sam_checkpoint: str = None,
    max_crops_per_image: int = 5,
):
    base_dir = Path(__file__).resolve().parent.parent

    # Configure paths
    if subset_json is None:
        candidates = [
            base_dir / "data" / "annotations" / "active_subset_8k.json",
            Path("/workspace/project_root/data/annotations/active_subset_8k.json"),
            Path("/storage/sawant1/project_root/data/annotations/active_subset_8k.json"),
        ]
        subset_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        subset_path = Path(subset_json)

    if output_dir is None:
        out_dir = base_dir / "output" / "micro_crops"
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "crop_manifest.json"

    possible_roots = [
        base_dir / "data" / "images",
        base_dir / "SurgMLLMBench" / "SurgMLLMBench",
        base_dir / "SurgMLLMBench",
        Path("/workspace/project_root/data/images"),
        Path("/workspace/SurgMLLMBench/SurgMLLMBench"),
        Path("/storage/sawant1/project_root/data/images"),
    ]

    print("🔍 Building fast image search index...")
    image_index = {}
    for root in possible_roots:
        if not root.exists():
            continue
        for folder, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_index[f] = os.path.join(folder, f)
    print(f"✅ Indexed {len(image_index)} image files.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find SAM checkpoint
    if sam_checkpoint is None:
        sam_candidates = [
            base_dir / "weights" / "sam_vit_h_4b8939.pth",
            Path("/workspace/weights/sam_vit_h_4b8939.pth"),
            Path("/storage/sawant1/weights/sam_vit_h_4b8939.pth"),
        ]
        sam_checkpoint = next((str(p) for p in sam_candidates if p.exists()), str(sam_candidates[0]))

    if not HAS_SAM or not os.path.exists(sam_checkpoint):
        print(f"⚠️ SAM not available or checkpoint not found at {sam_checkpoint}.")
        print("   Running in fallback saliency contour extraction mode.")
        mask_generator = None
    else:
        print(f"⚡ Loading SAM ViT-H on {device}...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
        mask_generator = SamAutomaticMaskGenerator(sam)

    print("⚡ Loading ResNet50 Saliency Model...")
    saliency_model = resnet50(pretrained=True).eval().to(device)

    def get_saliency_map(img_bgr):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_t = transform(img_bgr).unsqueeze(0).to(device)
        img_t.requires_grad = True
        out = saliency_model(img_t)
        tc = out.argmax(dim=1)
        grad = torch.autograd.grad(out[:, tc], img_t)[0]
        grad = grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
        return cv2.resize(grad, (img_bgr.shape[1], img_bgr.shape[0]))

    if not subset_path.exists():
        print(f"❌ Subset JSON not found at: {subset_path}")
        return

    with open(subset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    manifest = []
    saved_total = 0

    print(f"--- 🚀 Starting High-Precision Extraction on {len(samples)} samples ---")
    for idx, item in enumerate(samples):
        image_field = item.get("image", "")
        img_name = os.path.basename(image_field)
        img_path = image_index.get(img_name)

        if img_path is None or not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        s_map = get_saliency_map(img)
        img_h, img_w = img.shape[:2]
        crop_count = 0

        candidate_bboxes = []
        if mask_generator is not None:
            masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for m in masks:
                candidate_bboxes.append(list(map(int, m["bbox"])))
        else:
            # Fallback contour detection on thresholded saliency
            thresh = (s_map > (s_map.mean() + s_map.std())).astype(np.uint8) * 255
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                candidate_bboxes.append([x, y, w, h])

        for (x, y, w, h) in candidate_bboxes:
            if crop_count >= max_crops_per_image:
                break
            if w * h <= 0:
                continue

            # Saliency filter
            s_score = np.mean(s_map[y:y+h, x:x+w])
            if s_score < 0.3:
                continue
            # Area filter: skip bounding boxes occupying >10% full image
            if (w * h) / (img_h * img_w) > 0.10:
                continue

            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (1024, 1024))

            sample_id = item.get("id", f"sample_{idx}")
            crop_filename = f"{sample_id}_{crop_count}.jpg"
            out_file = out_dir / crop_filename
            cv2.imwrite(str(out_file), crop_resized)

            manifest.append({
                "crop_id": crop_filename,
                "global_image": str(os.path.abspath(img_path)),
                "offset": [x, y, w, h],
            })
            crop_count += 1
            saved_total += 1

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(samples)}] Total crops saved: {saved_total}")
            with open(manifest_path, "w", encoding="utf-8") as fm:
                json.dump(manifest, fm, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, indent=2)
    print(f"🏁 Finished. Total Crops Saved: {saved_total} → Manifest: {manifest_path}")


if __name__ == "__main__":
    run_extraction()
