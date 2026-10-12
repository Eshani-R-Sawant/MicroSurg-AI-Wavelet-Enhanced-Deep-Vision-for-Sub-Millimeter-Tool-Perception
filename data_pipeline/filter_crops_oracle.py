"""
Step 03.2: Filter Highest-Quality Crops & Generate Compact Dataset.

Filters micro-crops by bounding box area to select high-salience instruments
over background tissue, capping at 4,000 top samples and producing final_unified_4000.jsonl.
"""

import json
from pathlib import Path


def filter_top_crops(
    manifest_path: str = None,
    input_vqa: str = None,
    output_file: str = None,
    top_k: int = 4000,
):
    base_dir = Path(__file__).resolve().parent.parent
    man_p = Path(manifest_path) if manifest_path else base_dir / "output" / "micro_crops" / "crop_manifest.json"
    in_vqa = Path(input_vqa) if input_vqa else base_dir / "output" / "final_unified_7531.jsonl"
    out_p = Path(output_file) if output_file else base_dir / "output" / "final_unified_4000.jsonl"

    if not man_p.exists():
        print(f"❌ Manifest not found at: {man_p}")
        return

    with open(man_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Sort crops by bounding box area (w * h)
    manifest.sort(key=lambda m: m.get("offset", [0, 0, 0, 0])[2] * m.get("offset", [0, 0, 0, 0])[3], reverse=True)
    top_manifest = manifest[:top_k]
    valid_ids = {m["crop_id"] for m in top_manifest}
    print(f"🔍 Filtered top {len(valid_ids)} crops by instrument surface area.")

    count = 0
    if in_vqa.exists():
        with open(in_vqa, "r", encoding="utf-8") as f_in, open(out_p, "w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    item = json.loads(line)
                    if item.get("crop_id") in valid_ids:
                        f_out.write(json.dumps(item) + "\n")
                        count += 1
                except Exception:
                    continue
        print(f"✅ Filtered dataset ready: {count} samples → {out_p}")
    else:
        print(f"⚠️ Input VQA file {in_vqa} not found.")


if __name__ == "__main__":
    filter_top_crops()
