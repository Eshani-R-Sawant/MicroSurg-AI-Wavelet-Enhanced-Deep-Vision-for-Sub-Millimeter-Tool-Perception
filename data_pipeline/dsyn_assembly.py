"""
Step 06: Multimodal Dataset Assembly (R2I / Region-to-Instruction).

Combines:
1. Repaired VQA distillation pairs (tool, action, tissue).
2. SAM crop offsets (remapping 1024 crop bboxes into global pixel space).
3. Temporal geometric homography pairs.
Outputs master student training dataset: output/dsyn_training_set.jsonl.
"""

import glob
import json
import os
from pathlib import Path


def assemble_training_dataset(
    vqa_file: str = None,
    manifest_file: str = None,
    cross_view_file: str = None,
    output_file: str = None,
):
    base_dir = Path(__file__).resolve().parent.parent

    # Find VQA input (repaired preferred)
    if vqa_file is None:
        vqa_candidates = [
            base_dir / "output" / "dsyn_repaired_full.jsonl",
            base_dir / "output" / "final_unified_7531.jsonl",
            base_dir / "output" / "syn_vqa_pairs.jsonl",
            Path("/workspace/project_root/output/dsyn_repaired_full.jsonl"),
            Path("/storage/sawant1/project_root/output/dsyn_repaired_full.jsonl"),
        ]
        vqa_p = next((p for p in vqa_candidates if p.exists()), vqa_candidates[0])
    else:
        vqa_p = Path(vqa_file)

    if manifest_file is None:
        man_candidates = [
            base_dir / "output" / "micro_crops" / "crop_manifest.json",
            Path("/workspace/project_root/output/micro_crops/crop_manifest.json"),
            Path("/storage/sawant1/project_root/output/micro_crops/crop_manifest.json"),
        ]
        man_p = next((p for p in man_candidates if p.exists()), man_candidates[0])
    else:
        man_p = Path(manifest_file)

    if cross_view_file is None:
        cv_candidates = [
            base_dir / "output" / "cross_view_pairs.json",
            base_dir / "cross_view_pairs.json",
            Path("/workspace/project_root/output/cross_view_pairs.json"),
            Path("/storage/sawant1/project_root/output/cross_view_pairs.json"),
        ]
        cv_p = next((p for p in cv_candidates if p.exists()), None)
    else:
        cv_p = Path(cross_view_file)

    out_p = Path(output_file) if output_file else base_dir / "output" / "dsyn_training_set.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 Assembling Master Multimodal Training Dataset...", flush=True)

    # 1. Load crop manifest
    meta_map = {}
    if man_p.exists():
        with open(man_p, "r", encoding="utf-8") as f:
            manifest_raw = json.load(f)
            for m in manifest_raw:
                cid = os.path.basename(m.get("crop_id", m.get("crop_path", "")))
                meta_map[cid] = m
        print(f"   Manifest entries indexed: {len(meta_map)}")
    else:
        print(f"⚠️ crop_manifest.json not found at {man_p}. Will use inline global paths if present.")

    assembled = 0
    skipped = 0
    seen_crops = set()

    with open(out_p, "w", encoding="utf-8") as f_out:
        # Process VQA pairs
        if vqa_p.exists():
            print(f"📦 Processing VQA pairs: {vqa_p.name}...")
            with open(vqa_p, "r", encoding="utf-8") as f_in:
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
                    if not cid or cid in seen_crops:
                        skipped += 1
                        continue

                    m = meta_map.get(cid)
                    if m is None:
                        if vqa.get("global_image_path"):
                            vqa["type"] = "vqa_distillation"
                            f_out.write(json.dumps(vqa) + "\n")
                            assembled += 1
                            seen_crops.add(cid)
                        else:
                            skipped += 1
                        continue

                    # Remap crop bbox to global pixel space
                    ox, oy, cw, ch = m["offset"]
                    raw_bbox = vqa.get("bbox") or vqa.get("expert_bbox")
                    if raw_bbox and any(v > 0 for v in raw_bbox):
                        y1_c, x1_c, y2_c, x2_c = raw_bbox
                        # If coordinates are 0-1000 scale:
                        scale = 1000.0 if max(raw_bbox) > 1.0 else 1.0
                        global_bbox = [
                            int(oy + (y1_c * ch / scale)),
                            int(ox + (x1_c * cw / scale)),
                            int(oy + (y2_c * ch / scale)),
                            int(ox + (x2_c * cw / scale)),
                        ]
                    else:
                        global_bbox = None

                    crop_img_path = str(base_dir / "output" / "micro_crops" / cid)
                    entry = {
                        "type": "vqa_distillation",
                        "crop_id": cid,
                        "global_image_path": m["global_image"],
                        "crop_image_path": crop_img_path,
                        "question": vqa.get("question", "Identify the surgical tool, action, and tissue."),
                        "answer": vqa.get("answer", vqa.get("expert_answer", "")),
                        "global_bbox": global_bbox,
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    assembled += 1
                    seen_crops.add(cid)

        # Process Homography Geometric Consistency pairs
        if cv_p and cv_p.exists():
            print(f"🔄 Appending geometric consistency pairs from {cv_p.name}...")
            with open(cv_p, "r", encoding="utf-8") as f_cv:
                cv_pairs = json.load(f_cv)
                for p in cv_pairs:
                    f_out.write(json.dumps({
                        "type": "geometric_consistency",
                        "frame_1": p["frame_1"],
                        "frame_2": p["frame_2"],
                        "homography": p["homography_matrix"],
                    }) + "\n")
                    assembled += 1

    print("\n✅ Multimodal Assembly Complete.", flush=True)
    print(f"   Samples Assembled : {assembled}")
    print(f"   Skipped / Dups    : {skipped}")
    print(f"   Output Dataset    : {out_p}")


if __name__ == "__main__":
    assemble_training_dataset()
