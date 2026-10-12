import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, 1, H, W]
        # targets: [B, 1, H, W]
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice_score

class GeometricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, mask_pred_clean, mask_pred_warped, H_matrix):
        """
        Placeholder for Homography/Geometric consistency loss.
        In practice, we warp `mask_pred_clean` using `H_matrix` and compare with `mask_pred_warped`.
        We'll use Kornia internally in the training loop, but for now we define the loss metric.
        """
        return self.l1_loss(mask_pred_warped, mask_pred_clean) 

class ZoomingMultiLoss(nn.Module):
    def __init__(self, w_text=1.0, w_coord=2.0, w_dice=2.0, w_wavelet=0.5, w_geo=0.5):
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

    def forward(self, preds, targets):
        """
        preds: dictionary containing predictions
        targets: dictionary containing ground truth
        
        Expected keys:
        - text_logits, text_targets -> for L_text
        - coord_pred, coord_targets -> for L_coord (boxes)
        - mask_logits, mask_targets -> for L_DiceMask
        - wavelet_features, wavelet_targets -> for L_wavelet
        - mask_clean, mask_warped, H_matrix -> for L_geo
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Text Loss (LM next-token prediction)
        if 'text_logits' in preds and 'text_targets' in targets:
            text_loss = self.text_criterion(
                preds['text_logits'].view(-1, preds['text_logits'].size(-1)), 
                targets['text_targets'].view(-1)
            )
            loss_dict['L_text'] = text_loss
            total_loss += self.w_text * text_loss
            
        # 2. Coordinate Loss (Bounding Box regression)
        if 'coord_pred' in preds and 'coord_targets' in targets:
            coord_loss = self.coord_criterion(preds['coord_pred'], targets['coord_targets'])
            loss_dict['L_coord'] = coord_loss
            total_loss += self.w_coord * coord_loss
            
        # 3. Dice Mask Loss (Segmentation)
        if 'mask_logits' in preds and 'mask_targets' in targets:
            dice_loss = self.dice_criterion(preds['mask_logits'], targets['mask_targets'])
            loss_dict['L_dice'] = dice_loss
            total_loss += self.w_dice * dice_loss
            
        # 4. Wavelet Edge Loss
        if 'wavelet_features' in preds and 'wavelet_targets' in targets:
            wave_loss = self.wavelet_criterion(preds['wavelet_features'], targets['wavelet_targets'])
            loss_dict['L_wavelet'] = wave_loss
            total_loss += self.w_wavelet * wave_loss
            
        # 5. Geometric Consistency Loss
        if 'mask_clean' in preds and 'mask_warped' in preds:
            geo_loss = self.geo_criterion(preds['mask_clean'], preds['mask_warped'], preds.get('H_matrix'))
            loss_dict['L_geo'] = geo_loss
            total_loss += self.w_geo * geo_loss
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

if __name__ == "__main__":
    # Test the multi-loss implementation
    criterion = ZoomingMultiLoss()
    print("Multi-Loss Objective module created with appropriate weights.")
    print(f"Weights: Text({criterion.w_text}), Coord({criterion.w_coord}), Dice({criterion.w_dice}), Wavelet({criterion.w_wavelet}), Geo({criterion.w_geo})")
