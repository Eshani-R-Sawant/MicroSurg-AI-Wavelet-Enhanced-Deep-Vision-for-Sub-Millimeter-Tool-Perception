"""
Multi-Task Loss Formulations for Zooming Without Zooming (ZWZ) Architecture.

Combines:
1. L_text: Causal Language Modeling Cross-Entropy loss for surgical reasoning & VQA.
2. L_coord: Smooth L1 regression loss for normalized tool tip bounding boxes.
3. L_dice: Soft Dice loss for spatial instrument segmentation masks.
4. L_wavelet: High-frequency edge distillation loss.
5. L_geo: Inter-frame geometric homography consistency loss across temporal frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for surgical tool segmentation masks.
    Handles class imbalance between background tissue and thin instruments.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 1, H, W] or [B, H, W] raw unnormalized mask predictions
        targets: [B, 1, H, W] or [B, H, W] ground truth binary masks
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class GeometricConsistencyLoss(nn.Module):
    """
    Geometric Consistency Loss across consecutive temporal surgical frames.
    Compares predicted segmentation masks across homography-warped viewpoint pairs.
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, mask_pred_clean: torch.Tensor, mask_pred_warped: torch.Tensor, H_matrix=None) -> torch.Tensor:
        """
        Computes L1 difference between the warped clean mask and the adjacent frame mask.
        """
        if mask_pred_clean.shape != mask_pred_warped.shape:
            mask_pred_warped = F.interpolate(mask_pred_warped, size=mask_pred_clean.shape[-2:], mode="bilinear", align_corners=False)
        return self.l1_loss(mask_pred_clean, mask_pred_warped)


class ZoomingMultiLoss(nn.Module):
    """
    Unified Multi-Objective Loss:
    Total Loss = w_text * L_text + w_coord * L_coord + w_dice * L_dice + w_wavelet * L_wavelet + w_geo * L_geo
    """
    def __init__(
        self,
        w_text: float = 1.0,
        w_coord: float = 2.0,
        w_dice: float = 2.0,
        w_wavelet: float = 0.5,
        w_geo: float = 0.5,
    ):
        super().__init__()
        self.w_text = w_text
        self.w_coord = w_coord
        self.w_dice = w_dice
        self.w_wavelet = w_wavelet
        self.w_geo = w_geo

        self.text_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.coord_criterion = nn.SmoothL1Loss()
        self.dice_criterion = DiceLoss()
        self.wavelet_criterion = nn.L1Loss()
        self.geo_criterion = GeometricConsistencyLoss()

    def forward(self, preds: dict, targets: dict):
        """
        Args:
            preds: Dictionary containing model predictions:
                - 'text_logits': [B, Seq_Len, Vocab_Size]
                - 'coord_pred': [B, 4]
                - 'mask_logits': [B, 1, H, W]
                - 'wavelet_features': [B, N, Dim]
                - 'mask_clean', 'mask_warped': for geometric consistency
            targets: Dictionary containing ground truth targets:
                - 'text_targets': [B, Seq_Len]
                - 'coord_targets': [B, 4]
                - 'mask_targets': [B, 1, H, W]
                - 'wavelet_targets': [B, N, Dim]

        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters(), torch.tensor(0.0)).device if list(self.parameters()) else torch.device("cpu"))

        # 1. Text Loss (LM next-token cross-entropy)
        if "text_logits" in preds and "text_targets" in targets:
            text_loss = self.text_criterion(
                preds["text_logits"].view(-1, preds["text_logits"].size(-1)),
                targets["text_targets"].view(-1),
            )
            loss_dict["L_text"] = text_loss
            total_loss = total_loss + self.w_text * text_loss

        # 2. Coordinate Loss (Smooth L1 for Bounding Box regression)
        if "coord_pred" in preds and "coord_targets" in targets:
            coord_loss = self.coord_criterion(preds["coord_pred"], targets["coord_targets"])
            loss_dict["L_coord"] = coord_loss
            total_loss = total_loss + self.w_coord * coord_loss

        # 3. Dice Mask Loss (Instrument Segmentation)
        if "mask_logits" in preds and "mask_targets" in targets:
            dice_loss = self.dice_criterion(preds["mask_logits"], targets["mask_targets"])
            loss_dict["L_dice"] = dice_loss
            total_loss = total_loss + self.w_dice * dice_loss

        # 4. Wavelet Edge Distillation Loss
        if "wavelet_features" in preds and "wavelet_targets" in targets:
            wave_loss = self.wavelet_criterion(preds["wavelet_features"], targets["wavelet_targets"])
            loss_dict["L_wavelet"] = wave_loss
            total_loss = total_loss + self.w_wavelet * wave_loss

        # 5. Geometric Homography Consistency Loss
        if "mask_clean" in preds and "mask_warped" in preds:
            geo_loss = self.geo_criterion(preds["mask_clean"], preds["mask_warped"], preds.get("H_matrix"))
            loss_dict["L_geo"] = geo_loss
            total_loss = total_loss + self.w_geo * geo_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
