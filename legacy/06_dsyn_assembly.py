"""

Changes vs original:
  - Uses clean_vqa_merged.jsonl (pre-cleaned) not raw final_unified
  - Correct bbox remapping (pixel → normalised 0–1000 scale)
  - Handles missing manifest entries gracefully
  - Deduplicates on crop_id
"""

import json, os
from pathlib import Path

WORKSPACE    = Path("/workspace/project_root")
VQA_INPUT    = WORKSPACE / "output/dsyn_repaired_full.jsonl"        # cleaned
MANIFEST     = WORKSPACE / "output/micro_crops/crop_manifest.json"
FINAL_DSYN   = WORKSPACE / "output/dsyn_training_set.jsonl"

def assemble():
    print("🚀 Running Assembly (clean VQA → dsyn)...", flush=True)

    if not MANIFEST.exists():
        print("❌ crop_manifest.json not found — copy from 12k run output")
        return

    if not VQA_INPUT.exists():
        print(f"❌ {VQA_INPUT} not found — run 00_clean_vqa.py first")
        return

    # Load manifest keyed by basename
    with open(MANIFEST) as f:
        raw_manifest = json.load(f)
    meta_map = {os.path.basename(m["crop_id"]): m for m in raw_manifest}
    print(f"   Manifest entries : {len(meta_map)}", flush=True)

    assembled = 0
    skipped   = 0
    seen_crop = set()

    with open(VQA_INPUT) as f_in, open(FINAL_DSYN, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                vqa = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            cid = os.path.basename(vqa.get("crop_id", ""))
            if not cid or cid in seen_crop:
                skipped += 1
                continue

            m = meta_map.get(cid)
            if m is None:
                # No manifest entry — use existing global_image_path if present
                if vqa.get("global_image_path"):
                    vqa["type"] = "vqa_distillation"
                    f_out.write(json.dumps(vqa) + "\n")
                    assembled += 1
                    seen_crop.add(cid)
                else:
                    skipped += 1
                continue

            # Remap local-crop bbox → global pixel bbox
            ox, oy, cw, ch = m["offset"]        # x, y, w, h of crop in global image
            raw_bbox = vqa.get("bbox") or [0, 0, 0, 0]

            if raw_bbox and any(v > 0 for v in raw_bbox):
                # Stored as [y1, x1, y2, x2] on 0–1000 scale (crop space)
                y1_c, x1_c, y2_c, x2_c = raw_bbox
                # Map to global pixel coordinates
                global_bbox = [
                    int(oy + (y1_c * ch / 1000)),
                    int(ox + (x1_c * cw / 1000)),
                    int(oy + (y2_c * ch / 1000)),
                    int(ox + (x2_c * cw / 1000)),
                ]
            else:
                global_bbox = None

            vqa.update({
                "type":             "vqa_distillation",
                "global_image_path": m["global_image"],
                "crop_image_path":  str(WORKSPACE / "output/micro_crops" / cid),
                "global_bbox":      global_bbox,
            })
            # Remove old local bbox key to avoid confusion
            vqa.pop("bbox", None)

            f_out.write(json.dumps(vqa) + "\n")
            assembled += 1
            seen_crop.add(cid)

    print(f"\n✅ Assembly complete.", flush=True)
    print(f"   Assembled : {assembled}", flush=True)
    print(f"   Skipped   : {skipped}", flush=True)
    print(f"   Output    : {FINAL_DSYN}", flush=True)


if __name__ == "__main__":
    assemble()
