"""
PyTorch Dataset and DataLoaders for Surgical VQA & Grounding.
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class SurgicalVQADataset(Dataset):
    """
    Surgical Visual Question Answering Dataset.
    Loads paired surgical frames and answers, with dynamic path resolving.
    Appends [SEG] token to activate spatial mask decoder.
    """
    def __init__(self, json_path: str, transform=None, append_seg_token: bool = True):
        self.transform = transform
        self.append_seg_token = append_seg_token
        self.data = []

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("type") == "vqa_distillation" or "question" in d:
                            self.data.append(d)
                    except Exception:
                        continue

        print(f"✅ Loaded {len(self.data)} surgical VQA samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, path: str):
        if not path or not isinstance(path, str):
            return None
        if os.path.exists(path):
            return path
        # Check common cluster path translations
        for prefix in ["/scratch/m25cse012/CV_major", "/workspace", "/storage/sawant1"]:
            candidate = path.replace("/scratch/m25cse012/CV_major", prefix).replace("/workspace", prefix)
            if os.path.exists(candidate):
                return candidate
        return None

    def __getitem__(self, idx, _retries=0):
        if _retries >= len(self.data):
            raise RuntimeError("❌ No valid readable images found in dataset.")

        item = self.data[idx]
        raw_path = item.get("global_image_path") or item.get("image")
        resolved_path = self._resolve_image_path(raw_path)

        if not resolved_path:
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)

        try:
            image = Image.open(resolved_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = image.resize((448, 448))

            ans = item.get("answer", "Surgical instrument is present in operative field.")
            if self.append_seg_token and "[SEG]" not in ans:
                answer = f"{ans} [SEG]"
            else:
                answer = ans

            return {
                "image": image,
                "question": item.get("question", "Identify the surgical tool, action, and tissue."),
                "answer": answer,
                "target_mask": torch.zeros((1, 1024, 1024)),
                "global_bbox": item.get("global_bbox"),
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data), _retries + 1)
