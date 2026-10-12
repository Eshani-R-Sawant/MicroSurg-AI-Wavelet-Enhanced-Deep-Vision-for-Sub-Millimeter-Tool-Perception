import pandas as pd
import os
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

def extract_parquet_images(parquet_dir, output_root):
    parquet_dir = Path(parquet_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")

    for pf in parquet_files:
        print(f"Processing {pf.name}...")
        df = pd.read_parquet(pf)
        
        # Subdirectory for this sequence
        seq_name = pf.stem
        seq_dir = output_root / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Assuming 'left_image' contains the binary data
            # Based on our check, column exists
            img_data = row['left_image']
            frame_id = row['frame_id']
            
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            else:
                img_bytes = img_data

            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.save(seq_dir / f"{frame_id:05d}.jpg", quality=95)
            except Exception as e:
                print(f"Error saving frame {frame_id}: {e}")

if __name__ == "__main__":
    extract_parquet_images(
        "/scratch/m25cse012/CV_major/project_root/data/images/EndoVis2018/data/train",
        "/scratch/m25cse012/CV_major/project_root/data/images/EndoVis2018/frames"
    )
