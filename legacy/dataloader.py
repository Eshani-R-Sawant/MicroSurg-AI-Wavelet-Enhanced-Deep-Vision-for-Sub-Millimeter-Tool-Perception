import json, os
from torch.utils.data import Dataset
from PIL import Image

class R2IDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.old_prefix = "/scratch/m25cse012/CV_major"
        self.new_prefix = "/workspace"

    def __len__(self): return len(self.data)

    def _fix_path(self, path):
        if path and isinstance(path, str):
            p = path.replace(self.old_prefix, self.new_prefix)
            # Handle double slash issues
            return p.replace("//", "/")
        return path

    def __getitem__(self, idx):
        item = self.data[idx]
        g_path = self._fix_path(item.get("global_image_path"))
        
        if not g_path or not os.path.exists(g_path):
            return self.__getitem__((idx + 1) % len(self.data))
            
        try:
            image = Image.open(g_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))

        # IMPORTANT: Using 'answer' from your VQA reasoning file
        label_text = item.get("answer", item.get("caption", "Surgical detail detected."))
        return {"image": image, "text": label_text}
