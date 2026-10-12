import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_repo(repo_id, folder_name, token=None):
    print(f"Downloading {repo_id} to {folder_name} (Authenticated: {token is not None})...")
    target_dir = Path(f"/scratch/m25cse012/CV_major/project_root/data/images/{folder_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        local_dir = snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            local_dir=str(target_dir),
            ignore_patterns=["*.mp4"],
            token=token
        )
        print(f"✅ {repo_id} successfully downloaded to {local_dir}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")

def main():
    token = os.getenv("HF_TOKEN")
    
    # 1. MAVIS (Micro-surgical)
    download_repo("KIST-HARILAB/MAVIS", "MAVIS", token=token)
    
    # 2. MISAW-Seg (Micro-surgical)
    download_repo("KIST-HARILAB/MISAW-Seg", "MISAW", token=token)

    # 3. CholecSeg8k (Laparoscopic)
    download_repo("minwoosun/CholecSeg8k", "CholecSeg8k", token=token)

    # 4. EndoVis2018 (Robot-Assisted)
    download_repo("Angelou0516/Endovis2018", "EndoVis2018", token=token)

if __name__ == "__main__":
    main()
