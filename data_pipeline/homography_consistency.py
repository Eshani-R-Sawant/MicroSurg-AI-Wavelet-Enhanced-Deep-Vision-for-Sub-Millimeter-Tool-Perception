"""
Step 05: Geometric Homography Consistency Estimation.

Computes projective homography transformations (H matrices) between adjacent
endoscopic video frames across temporal surgical sessions using SIFT feature
detection, FLANN matching, and RANSAC geometric estimation.
Outputs: output/cross_view_pairs.json.
"""

import glob
import json
import os
from pathlib import Path
import cv2
import numpy as np


def compute_homography(img_path1: str, img_path2: str):
    """
    Computes 3x3 Homography matrix between two frames using SIFT and FLANN.
    """
    img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return None

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or len(des1) < 4 or des2 is None or len(des2) < 4:
        return None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
    except Exception:
        pass
    return None


def run_homography_pipeline(
    images_root: str = None,
    output_pairs_file: str = None,
    max_pairs: int = 500,
    stride: int = 5,
):
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        Path(images_root) if images_root else None,
        base_dir / "data" / "images",
        base_dir / "SurgMLLMBench" / "SurgMLLMBench",
        Path("/workspace/project_root/data/images"),
        Path("/storage/sawant1/project_root/data/images"),
    ]
    img_root = next((p for p in candidates if p and p.exists()), base_dir / "data" / "images")
    out_file = Path(output_pairs_file) if output_pairs_file else base_dir / "output" / "cross_view_pairs.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- 🔄 Starting Inter-Frame Homography Calculation in {img_root} ---")
    all_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        all_images.extend(glob.glob(str(img_root / "**" / ext), recursive=True))

    sequences = {}
    for p in all_images:
        session = os.path.dirname(p)
        sequences.setdefault(session, []).append(p)

    print(f"📌 Found {len(sequences)} temporal sessions ({len(all_images)} frames total).")

    paired_data = []
    total_checked = 0

    for session, paths in sequences.items():
        paths.sort()
        if len(paths) <= 1:
            continue

        for i in range(0, len(paths) - 1, stride):
            p1, p2 = paths[i], paths[i + 1]
            H = compute_homography(p1, p2)
            total_checked += 1

            if H is not None:
                paired_data.append({
                    "session": session,
                    "frame_1": p1,
                    "frame_2": p2,
                    "homography_matrix": H.tolist(),
                })
                if len(paired_data) % 10 == 0:
                    print(f"   ✅ Pairs discovered: {len(paired_data)}")

            if len(paired_data) >= max_pairs:
                break
        if len(paired_data) >= max_pairs:
            break

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(paired_data, f, indent=2)

    print(f"🎉 Complete. Analyzed {total_checked} frame transitions. Saved {len(paired_data)} pairs → {out_file}")


if __name__ == "__main__":
    run_homography_pipeline()
