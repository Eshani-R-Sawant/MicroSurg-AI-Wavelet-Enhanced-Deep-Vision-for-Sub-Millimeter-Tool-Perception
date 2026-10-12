"""
Step 04: Build Metadata & Map Local Crop Coordinates to Global Pixels.

Remaps teacher bounding boxes from 1024 crop space back into the original
high-resolution global image pixel space using crop offsets recorded in crop_manifest.json.
Formats output as multi-turn conversational VQA dataset (dsyn_surgical_dataset.jsonl).
"""

import json
from pathlib import Path


def assemble_metadata(
    manifest_path: str = None,
    vqa_input: str = None,
    output_jsonl: str = None,
):
    base_dir = Path(__file__).resolve().parent.parent
    man_p = Path(manifest_path) if manifest_path else base_dir / "output" / "micro_crops" / "crop_manifest.json"
    vqa_p = Path(vqa_input) if vqa_input else base_dir / "output" / "syn_vqa_pairs.jsonl"
    out_p = Path(output_jsonl) if output_jsonl else base_dir / "data" / "dsyn_surgical_dataset.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not man_p.exists():
        print(f"❌ Manifest not found: {man_p}")
        return

    if not vqa_p.exists():
        print(f"❌ VQA input not found: {vqa_p}")
        return

    with open(man_p, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    manifest = {m.get("crop_id", m.get("crop_path", "")): m for m in manifest_raw}

    count = 0
    with open(vqa_p, "r", encoding="utf-8") as fin, open(out_p, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                vqa = json.loads(line)
            except Exception:
                continue

            crop_id = vqa.get("crop_id", "")
            meta = manifest.get(crop_id)
            if not meta:
                continue

            # Remap local 1024 crop coordinates to global pixel space
            ox, oy, cw, ch = meta["offset"]
            t_bbox = vqa.get("expert_bbox") or vqa.get("teacher_bbox_1024") or [0, 0, 1000, 1000]
            t_ymin, t_xmin = t_bbox[0], t_bbox[1]

            gx = int(ox + (t_xmin * cw / 1000))
            gy = int(oy + (t_ymin * ch / 1000))

            answer = vqa.get("expert_answer") or vqa.get("answer", "Surgical instrument is active.")

            entry = {
                "image": meta["global_image"],
                "crop_id": crop_id,
                "point": [gx, gy],
                "conversations": [
                    {"from": "user", "value": f"<image>\nWhat is the tool state at ({gx}, {gy})?"},
                    {"from": "assistant", "value": answer},
                ],
            }
            fout.write(json.dumps(entry) + "\n")
            count += 1

    print(f"✅ Training Dataset Ready: {count} samples → {out_p}")


if __name__ == "__main__":
    assemble_metadata()
