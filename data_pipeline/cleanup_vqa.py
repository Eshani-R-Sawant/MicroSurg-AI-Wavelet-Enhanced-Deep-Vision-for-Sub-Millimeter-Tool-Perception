"""
VQA Distillation Data Cleaning & Medical Nomenclature Normalization.

Cleans chat template leakage, strips prompt pollution (assistant\\n, system\\n),
rescues unfulfilled template placeholders ([Tool], [Action], [Tissue]),
and maps colloquial instrument terms to standardized surgical nomenclature.
Outputs: output/dsyn_repaired_full.jsonl.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

# Standard surgical instrument canonical mapping
TOOL_MAP = {
    "hook dissecting": "Hook Dissector",
    "monopolar hook": "Hook Dissector",
    "monopolar curved scissors": "Curved Scissors",
    "maryland dissector": "Maryland Dissector",
    "maryland grasper": "Maryland Dissector",
    "fenestrated grasper": "Grasper",
    "bipolar forceps": "Bipolar Forceps",
    "suction cannula": "Suction Device",
    "needle driver": "Needle Driver",
    "cadiere forceps": "Cadiere Forceps",
    "clip applier": "Clip Applier",
}

RESCUE_ACTIONS = [
    "retracting", "dissecting", "grasping",
    "manipulating", "holding", "clamping", "cauterizing",
]

TEMPLATE_PLACEHOLDERS = [
    "[Surgical tool]", "[Tool]", "[Action]", "[Tissue]",
    "[Surgical action]", "Surgical procedure in progress",
]


def repair_answer(answer: str, question: str = "") -> str:
    """
    Cleans prompt pollution, strips template prefixes, and normalizes surgical instrument names.
    """
    if not answer or not isinstance(answer, str):
        return "Surgical instrument is visible in the operative field."

    # Step A: Strip Chat Template & Role Leakage
    for sep in ["assistant\n", "ASSISTANT:", "Answer: ", "Answer:"]:
        if sep in answer:
            answer = answer.split(sep)[-1].strip()

    answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.replace("<|endoftext|>", "").replace("<|im_end|>", "").rstrip(".")

    # Step B: Placeholder Rescue
    is_poisoned = any(p.lower() in answer.lower() for p in TEMPLATE_PLACEHOLDERS)
    if is_poisoned:
        q_lower = question.lower() if question else ""
        extracted_tool = None
        for alias, canonical in TOOL_MAP.items():
            if alias in q_lower:
                extracted_tool = canonical
                break

        if not extracted_tool:
            keywords = ["grasper", "scissors", "hook", "bipolar", "needle driver", "suction", "forceps"]
            for kw in keywords:
                if kw in q_lower:
                    extracted_tool = kw.title()
                    break

        if extracted_tool:
            action = random.choice(RESCUE_ACTIONS)
            return f"{extracted_tool} is {action} on surgical tissue."
        else:
            return "Surgical instrument is active in the operative field."

    # Step C: Normalization for Clean Answers
    answer_lower = answer.lower()
    for alias, canonical in TOOL_MAP.items():
        if alias in answer_lower:
            answer = re.sub(re.escape(alias), canonical, answer, flags=re.IGNORECASE)

    if len(answer.strip()) < 5:
        return "Surgical instrument is visible in frame."

    return answer.strip()


def run_repair(input_file: str = None, output_file: str = None):
    base_dir = Path(__file__).resolve().parent.parent
    in_p = Path(input_file) if input_file else base_dir / "output" / "final_unified_7531.jsonl"
    out_p = Path(output_file) if output_file else base_dir / "output" / "dsyn_repaired_full.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        print(f"❌ Input file not found: {in_p}")
        return

    stats = Counter()
    repaired_entries = []
    print(f"🔄 Cleaning and normalizing: {in_p}...")

    with open(in_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                orig_ans = d.get("answer", "")
                q = d.get("question", "")

                poisoned = any(p.lower() in orig_ans.lower() for p in TEMPLATE_PLACEHOLDERS)
                pollution = "assistant\n" in orig_ans or "system\n" in orig_ans

                cleaned_ans = repair_answer(orig_ans, q)
                if not cleaned_ans.endswith("."):
                    cleaned_ans += "."

                d["answer"] = cleaned_ans
                d["repaired"] = poisoned or pollution
                if d["repaired"]:
                    stats["repaired"] += 1
                else:
                    stats["kept_clean"] += 1

                repaired_entries.append(d)
            except json.JSONDecodeError:
                stats["corrupt_json"] += 1
                continue

    with open(out_p, "w", encoding="utf-8") as f_out:
        for entry in repaired_entries:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Cleaned {len(repaired_entries)} entries (Repaired: {stats['repaired']}, Kept: {stats['kept_clean']})")
    print(f"💾 Saved to: {out_p}")


if __name__ == "__main__":
    run_repair()
