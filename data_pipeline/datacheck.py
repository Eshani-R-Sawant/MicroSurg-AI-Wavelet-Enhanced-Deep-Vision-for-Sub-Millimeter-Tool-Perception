import json
import os
from pathlib import Path
from collections import defaultdict

SUBSET_JSON = "/scratch/m25cse012/CV_major/project_root/data/annotations/active_subset_8k.json"
POSSIBLE_DATASET_ROOTS = [
    Path("/scratch/m25cse012/CV_major/project_root/data/images"),
    Path("/scratch/m25cse012/CV_major/SurgMLLMBench/SurgMLLMBench"),
]

def main():
    if not os.path.exists(SUBSET_JSON):
        print(f"❌ Subset file not found: {SUBSET_JSON}")
        return

    print(f"🔍 Loading subset from {SUBSET_JSON}...")
    with open(SUBSET_JSON, "r") as f:
        samples = json.load(f)

    print(f"📊 Total samples in subset: {len(samples)}")

    # Pre-index images for speed if needed, but for 8k we can just check
    found = 0
    missing = 0
    per_dataset = defaultdict(lambda: {"found": 0, "missing": 0})

    for item in samples:
        dataset = item.get("source_dataset", "Unknown")
        image_field = item.get("image")
        
        if not image_field:
            missing += 1
            per_dataset[dataset]["missing"] += 1
            continue

        # Try relative to POSSIBLE_DATASET_ROOTS
        is_found = False
        for root in POSSIBLE_DATASET_ROOTS:
            # The image field might be relative to the dataset root
            # e.g. CASE01/image_00001.jpg
            # We check root/dataset/image_field and root/image_field
            paths_to_check = [
                root / image_field,
                root / dataset / image_field,
                root / dataset / "frames" / image_field # Specifically for MAVIS frames/ if download_repo used it
            ]
            
            for p in paths_to_check:
                if p.exists():
                    is_found = True
                    break
            if is_found:
                break
        
        if is_found:
            found += 1
            per_dataset[dataset]["found"] += 1
        else:
            missing += 1
            per_dataset[dataset]["missing"] += 1

    print("\n✅ Verification Results:")
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    
    print("\n📈 Per Dataset Breakdown:")
    for ds, stats in per_dataset.items():
        total = stats["found"] + stats["missing"]
        perc = (stats["found"] / total * 100) if total > 0 else 0
        print(f" - {ds}: {stats['found']}/{total} ({perc:.1f}%)")

    if missing > 0:
        print("\n⚠️  Some images are still missing. Ensure the download script has completed and extraction is correct.")
    else:
        print("\n✨ ALL IMAGES FOUND! Pipeline ready for next steps.")

if __name__ == "__main__":
    main()