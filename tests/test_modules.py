"""
Sanity and Unit Tests for SurgMLLM / ZWZ Core Modules.
"""

import sys
from pathlib import Path
import torch

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.wavelet_branch import WaveletVisionBranch
from models.mask_decoder import TransformerMaskDecoder
from models.losses import ZoomingMultiLoss, DiceLoss
from models.metrics import compute_iou, word_f1, compute_miou


def test_wavelet_vision_branch():
    print("Testing WaveletVisionBranch...")
    branch = WaveletVisionBranch(embed_dim=256)
    dummy_input = torch.randn(2, 3, 1024, 1024)
    tokens = branch(dummy_input)
    assert tokens.shape == (2, 256, 256), f"Unexpected shape: {tokens.shape}"
    print("✅ WaveletVisionBranch passed. Output shape:", tokens.shape)


def test_mask_decoder():
    print("Testing TransformerMaskDecoder...")
    decoder = TransformerMaskDecoder(hidden_dim=256, mask_dim=64)
    dummy_seg = torch.randn(2, 1, 256)
    dummy_vision = torch.randn(2, 256, 256)
    mask = decoder(dummy_seg, dummy_vision, image_size=(512, 512))
    assert mask.shape == (2, 1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    print("✅ TransformerMaskDecoder passed. Output shape:", mask.shape)


def test_losses():
    print("Testing ZoomingMultiLoss...")
    criterion = ZoomingMultiLoss()
    preds = {
        "text_logits": torch.randn(2, 10, 100),
        "coord_pred": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]),
        "mask_logits": torch.randn(2, 1, 64, 64),
    }
    targets = {
        "text_targets": torch.randint(0, 100, (2, 10)),
        "coord_targets": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]),
        "mask_targets": torch.zeros(2, 1, 64, 64),
    }
    total_loss, loss_dict = criterion(preds, targets)
    assert total_loss.item() > 0, "Loss should be positive"
    print("✅ ZoomingMultiLoss passed. Total loss:", total_loss.item())


def test_metrics():
    print("Testing Metrics (Word-F1, IoU, mIoU)...")
    pred_ans = "Hook Dissector is dissecting gallbladder tissue."
    gt_ans = "Hook Dissector is dissecting on tissue."
    f1 = word_f1(pred_ans, gt_ans)
    assert f1 > 0.5, f"Expected high F1, got {f1}"

    box_a = [0.1, 0.1, 0.5, 0.5]
    box_b = [0.1, 0.1, 0.5, 0.5]
    iou = compute_iou(box_a, box_b)
    assert abs(iou - 1.0) < 1e-4, f"Expected IoU 1.0, got {iou}"

    m_pred = torch.ones(2, 1, 32, 32)
    m_gt = torch.ones(2, 1, 32, 32)
    miou = compute_miou(m_pred, m_gt)
    print("✅ Metrics passed. F1:", round(f1, 3), "IoU:", round(iou, 3), "mIoU:", round(miou, 3))


if __name__ == "__main__":
    print("🚀 Running ZWZ Core Modules Sanity Checks...")
    test_wavelet_vision_branch()
    test_mask_decoder()
    test_losses()
    test_metrics()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
