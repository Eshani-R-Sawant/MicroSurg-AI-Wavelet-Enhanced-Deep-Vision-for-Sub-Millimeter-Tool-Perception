"""
SurgMLLM / ZWZ (Zooming Without Zooming) - Models Package
Includes:
- WaveletVisionBranch & QwenVLWithWavelets: High-frequency edge token extraction.
- TransformerMaskDecoder: Spatial segmentation mask generation triggered by [SEG] token.
- ZoomingMultiLoss, DiceLoss, GeometricConsistencyLoss: Multi-task objectives.
- compute_miou, compute_map, word_f1, compute_iou: Evaluation metrics.
"""

from .wavelet_branch import WaveletVisionBranch, QwenVLWithWavelets
from .mask_decoder import TransformerMaskDecoder
from .losses import ZoomingMultiLoss, DiceLoss, GeometricConsistencyLoss
from .metrics import compute_miou, compute_map, compute_iou, word_f1

__all__ = [
    "WaveletVisionBranch",
    "QwenVLWithWavelets",
    "TransformerMaskDecoder",
    "ZoomingMultiLoss",
    "DiceLoss",
    "GeometricConsistencyLoss",
    "compute_miou",
    "compute_map",
    "compute_iou",
    "word_f1",
]
