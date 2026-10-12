import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import numpy as np

def compute_miou(pred_masks, gt_masks, num_classes=2):
    """
    Compute Mean Intersection over Union (mIoU) for segmentation.
    pred_masks: [B, H, W] or [B, 1, H, W]
    gt_masks: [B, H, W] or [B, 1, H, W]
    """
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)
    
    pred_masks = (pred_masks > 0.5).float()
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_masks == cls)
        gt_cls = (gt_masks == cls)
        
        intersection = (pred_cls & gt_cls).sum().float()
        union = (pred_cls | gt_cls).sum().float()
        
        if union == 0:
            ious.append(1.0)  # Perfect score if no pixels
        else:
            ious.append((intersection / union).item())
    
    return np.mean(ious)

def compute_map(pred_logits, gt_boxes, num_classes=2):
    """
    Compute Mean Average Precision (mAP) for detection.
    pred_logits: [B, num_classes, H, W]
    gt_boxes: List of [x1, y1, x2, y2, class] per image
    Simplified version.
    """
    # This is a placeholder; full mAP requires detection boxes, not masks.
    # For now, treat as binary segmentation mAP.
    probs = torch.sigmoid(pred_logits).flatten()
    targets = (gt_boxes > 0).flatten().float()
    
    if targets.sum() == 0:
        return 0.0
    
    return average_precision_score(targets.cpu().numpy(), probs.cpu().numpy())

# Example usage in training loop
if __name__ == "__main__":
    # Mock data
    pred = torch.randn(2, 1, 64, 64)
    gt = torch.randint(0, 2, (2, 64, 64)).float()
    print(f"mIoU: {compute_miou(pred, gt)}")
    
    pred_logits = torch.randn(2, 2, 64, 64)
    gt_boxes = torch.randint(0, 2, (2, 64, 64)).float()
    print(f"mAP: {compute_map(pred_logits, gt_boxes)}")
