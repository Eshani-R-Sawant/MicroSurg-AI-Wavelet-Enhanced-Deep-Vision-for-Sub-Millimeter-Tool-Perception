import json
import pickle
import random
from pathlib import Path

# ================= CONFIG =================
DATA_ROOT = Path("./SurgMLLMBench")   # after git clone + lfs pull
PROJECT_ROOT = Path("./project_root")
SUBSET_SIZE = 8000

# ================= HELPER =================
def load_pickle_structure(p_path):
    with open(p_path, "rb") as f:
        data = pickle.load(f)

    info = {
        "type": type(data).__name__,
        "length": len(data) if hasattr(data, "__len__") else None
    }

    if isinstance(data, dict):
        info["keys"] = list(data.keys())

    return info


# ================= MAIN =================
def map_dataset_instance():
    all_samples = []

    meta_files = list(DATA_ROOT.rglob("*metadata*.json")) + \
                 list(DATA_ROOT.rglob("*metadata*.jsonl"))

    print(f"Found {len(meta_files)} metadata files")

    for meta_path in meta_files:
        print(f"\nProcessing: {meta_path}")

        # ---------- Load metadata ----------
        with open(meta_path, 'r') as f:
            if meta_path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        split_dir = meta_path.parent

        # ---------- Find corresponding .p ----------
        p_files = list(split_dir.glob("*.p"))

        if not p_files:
            print("⚠️ No .p file found in this split")
            continue

        p_path = p_files[0]

        # Debug structure once
        print("🔍 Inspecting pickle:", p_path.name)
        try:
            info = load_pickle_structure(p_path)
            print("   → Type:", info["type"], "| Length:", info["length"])
            if "keys" in info:
                print("   → Keys:", info["keys"][:5])
        except Exception as e:
            print("❌ Failed to inspect pickle:", e)
            continue

        # ---------- Build samples ----------
        for idx, item in enumerate(data):

            sample = {
                "frame_index": idx,  # mapping assumption
                "pickle_path": str(p_path),
                "source_meta": str(meta_path),
                "original_data": item
            }

            all_samples.append(sample)

    print(f"\n✅ Total collected samples: {len(all_samples)}")

    # ---------- Subset ----------
    subset = random.sample(all_samples, min(len(all_samples), SUBSET_SIZE))

    save_path = PROJECT_ROOT / "data/annotations/active_subset_8k.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"✅ Saved {len(subset)} samples → {save_path}")


# ================= RUN =================
map_dataset_instance()