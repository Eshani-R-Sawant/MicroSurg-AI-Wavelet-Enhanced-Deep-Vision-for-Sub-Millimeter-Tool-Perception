"""
Step 01: Setup and Subset Curation Pipeline.

Recursively collects annotation files (.json, .jsonl) from benchmark sources
(e.g., SurgMLLMBench), tags each sample with its dataset origin, caps each dataset
at 2,000 samples to avoid class imbalance, shuffles, and exports a balanced subset
of 8,000 samples to data/annotations/active_subset_8k.json.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path


def create_subset(
    source_root: str = None,
    output_dir: str = None,
    subset_size: int = 8000,
    per_dataset_limit: int = 2000,
    seed: int = 42,
):
    random.seed(seed)

    # Resolve paths
    base_dir = Path(__file__).resolve().parent.parent
    if source_root is None:
        source_candidates = [
            Path(os.getenv("SOURCE_ROOT", "")),
            base_dir / "SurgMLLMBench" / "SurgMLLMBench",
            base_dir / "SurgMLLMBench",
            Path("/workspace/SurgMLLMBench/SurgMLLMBench"),
            Path("/workspace/SurgMLLMBench"),
            Path("/storage/sawant1/SurgMLLMBench/SurgMLLMBench"),
        ]
        source_root = next((p for p in source_candidates if p.exists()), base_dir / "SurgMLLMBench")
    else:
        source_root = Path(source_root)

    if output_dir is None:
        ann_dir = base_dir / "data" / "annotations"
    else:
        ann_dir = Path(output_dir)

    ann_dir.mkdir(parents=True, exist_ok=True)
    out_path = ann_dir / "active_subset_8k.json"

    print(f"🔍 Searching annotations in: {source_root}")
    meta_files = list(source_root.rglob("*.json*"))
    meta_files = [f for f in meta_files if f.suffix in [".json", ".jsonl"]]
    print(f"📂 Found {len(meta_files)} potential annotation files.")

    all_samples = []
    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                if meta_path.suffix == ".jsonl":
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading {meta_path.name}: {e}")
            continue

        if not isinstance(data, list):
            continue

        try:
            dataset_name = meta_path.relative_to(source_root).parts[0]
        except Exception:
            dataset_name = meta_path.parent.name

        for item in data:
            if not isinstance(item, dict):
                continue
            item["source_dataset"] = dataset_name
            all_samples.append(item)

    print(f"📊 Total raw samples collected: {len(all_samples)}")
    if not all_samples:
        print("⚠️ No samples found in SOURCE_ROOT. Check dataset directory.")
        return []

    # Group by dataset
    grouped = defaultdict(list)
    for s in all_samples:
        grouped[s["source_dataset"]].append(s)

    final_subset = []
    for ds_name, ds_samples in grouped.items():
        print(f"   • Dataset {ds_name}: {len(ds_samples)} samples")
        if len(ds_samples) > per_dataset_limit:
            sampled = random.sample(ds_samples, per_dataset_limit)
        else:
            sampled = ds_samples
        final_subset.extend(sampled)

    # Shuffle and cap to SUBSET_SIZE
    if len(final_subset) > subset_size:
        random.shuffle(final_subset)
        final_subset = final_subset[:subset_size]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_subset, f, indent=2)

    print(f"✅ Saved {len(final_subset)} balanced samples → {out_path}")
    return final_subset


if __name__ == "__main__":
    create_subset()
