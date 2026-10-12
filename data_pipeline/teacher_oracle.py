"""
Step 03: Teacher Oracle Distillation Pipeline.

Leverages Gemini 3.1 Flash-Lite as a multimodal Teacher Oracle to extract
expert surgical reasoning annotations and tool-tip bounding boxes for each micro-crop.
Produces output in syn_vqa_pairs.jsonl with automatic rate limiting and resume support.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from PIL import Image

# Import Google GenAI SDK (supports both 2026 google.genai and legacy google.generativeai)
HAS_GENAI = False
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        HAS_GENAI = "legacy"
    except ImportError:
        HAS_GENAI = False


def initialize_client(api_key: str = None):
    key = api_key or os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("❌ Error: GENAI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        print("   Set it via: export GENAI_API_KEY='your_api_key'")
        return None, None

    if HAS_GENAI is True:
        client = genai.Client(api_key=key)
        return client, "v2"
    elif HAS_GENAI == "legacy":
        legacy_genai.configure(api_key=key)
        model = legacy_genai.GenerativeModel("gemini-1.5-flash")
        return model, "legacy"
    else:
        print("❌ Error: google-genai or google-generativeai package not installed.")
        return None, None


def get_surgical_analysis(client, client_type, image_path: Path):
    prompt = (
        "Analyze this surgical image crop:\n"
        "1. Identify the tool and action (e.g., Hook Dissecting, Grasper Retracting).\n"
        "2. Provide tool tip coordinates in [ymin, xmin, ymax, xmax] normalized 0-1000 scale.\n"
        "Return strictly:\n"
        "Answer: [Tool] is [Action] on [Tissue].\n"
        "BBox: [ymin, xmin, ymax, xmax]"
    )

    try:
        img = Image.open(image_path).convert("RGB")
        if client_type == "v2":
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, img],
            )
            text = response.text
        else:
            response = client.generate_content([prompt, img])
            text = response.text

        bbox_match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
        ans_match = re.search(r"Answer:\s*(.*)", text)

        return {
            "crop_id": image_path.name,
            "expert_answer": ans_match.group(1).strip() if ans_match else text[:100],
            "expert_bbox": [int(x) for x in bbox_match.groups()] if bbox_match else None,
        }
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower():
            return "RATE_LIMIT"
        print(f"⚠️ API Error on {image_path.name}: {e}")
        return None


def run_distillation(
    target_samples: int = 1000,
    rpm_delay: float = 4.6,
    crop_dir: str = None,
    output_file: str = None,
    api_key: str = None,
):
    base_dir = Path(__file__).resolve().parent.parent
    c_dir = Path(crop_dir) if crop_dir else base_dir / "output" / "micro_crops"
    out_file = Path(output_file) if output_file else base_dir / "output" / "syn_vqa_pairs.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not c_dir.exists():
        print(f"❌ Error: Crop directory {c_dir} does not exist. Run step 02 first.")
        return

    client, client_type = initialize_client(api_key)
    if client is None:
        return

    # Check already processed samples (resume support)
    processed_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "crop_id" in data:
                        processed_ids.add(data["crop_id"])
                except Exception:
                    continue

    already_done = len(processed_ids)
    print(f"📊 Resume Check: {already_done} samples already completed.")

    if already_done >= target_samples:
        print(f"✅ Target reached ({already_done}/{target_samples}). Moving to next phase.")
        return

    all_crops = sorted(list(c_dir.glob("*.jpg")))
    queue = [c for c in all_crops if c.name not in processed_ids][: target_samples - already_done]
    print(f"🚀 Processing remaining {len(queue)} teacher labels...")

    with open(out_file, "a", encoding="utf-8") as f:
        for idx, crop_path in enumerate(queue):
            res = get_surgical_analysis(client, client_type, crop_path)
            if res == "RATE_LIMIT":
                print("🛑 Rate limit encountered. Sleeping 60s for IP cooldown...")
                time.sleep(60)
                res = get_surgical_analysis(client, client_type, crop_path)

            if res and res != "RATE_LIMIT":
                f.write(json.dumps(res) + "\n")
                f.flush()
                print(f"✅ [{already_done + idx + 1}/{target_samples}] {crop_path.name}")

            time.sleep(rpm_delay)

    print(f"\n🏁 Teacher distillation complete. Saved to: {out_file}")


if __name__ == "__main__":
    run_distillation()
