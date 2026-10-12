"""
Evaluation Metrics for Surgical Visual Question Answering, Grounding, and Segmentation.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def compute_iou(box_a, box_b) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes formatted as [y1, x1, y2, x2] normalized between 0.0 and 1.0.
    """
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ay1, ax1, ay2, ax2 = [max(0.0, min(1.0, float(v))) for v in box_a]
    by1, bx1, by2, bx2 = [max(0.0, min(1.0, float(v))) for v in box_b]

    # Intersection area
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_area = inter_h * inter_w

    # Union area
    area_a = max(0.0, (ay2 - ay1) * (ax2 - ax1))
    area_b = max(0.0, (by2 - by1) * (bx2 - bx1))
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0.0 else 0.0


def word_f1(pred: str, gt: str) -> float:
    """
    Token-level Word-F1 accuracy for surgical VQA answer evaluation.
    Filters common stopwords to focus on surgical instrument names, actions, and tissues.
    """
    if not pred or not gt:
        return 0.0

    stopwords = {
        "the", "a", "an", "is", "in", "on", "at", "of", "and", "to", "with",
        "this", "image", "shows", "i", "see", "answer", "there", "no", "present"
    }

    pred_tokens = set(pred.lower().strip().split()) - stopwords
    gt_tokens = set(gt.lower().strip().split()) - stopwords

    if not gt_tokens or not pred_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def compute_miou(pred_masks: torch.Tensor, gt_masks: torch.Tensor, num_classes: int = 2) -> float:
    """
    Mean Intersection over Union (mIoU) for binary/multi-class segmentation.
    pred_masks: [B, H, W] or [B, 1, H, W] probabilities or logits
    gt_masks: [B, H, W] or [B, 1, H, W] binary integer targets
    """
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)

    pred_binary = (pred_masks > 0.5).float()
    gt_binary = (gt_masks > 0.5).float()

    ious = []
    for cls_idx in range(num_classes):
        pred_c = (pred_binary == cls_idx)
        gt_c = (gt_binary == cls_idx)

        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()

        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())

    return float(np.mean(ious))


def compute_map(pred_scores, gt_labels) -> float:
    """
    Mean Average Precision (mAP) for surgical instrument detection.
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.detach().cpu().numpy()

    pred_scores = np.asarray(pred_scores).flatten()
    gt_labels = np.asarray(gt_labels).flatten()

    if len(gt_labels) == 0 or np.sum(gt_labels > 0) == 0:
        return 0.0

    try:
        return float(average_precision_score(gt_labels > 0, pred_scores))
    except Exception:
        return 0.0
