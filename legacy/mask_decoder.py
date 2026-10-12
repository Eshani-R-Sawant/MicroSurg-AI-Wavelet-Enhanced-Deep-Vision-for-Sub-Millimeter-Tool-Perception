import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerMaskDecoder(nn.Module):
    def __init__(self, hidden_dim=1024, mask_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        
        # Map LLM hidden state from [SEG] token to mask projection vector
        self.seg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mask_dim)
        )
        
        # Minimal ConvNet to upscale vision features into a spatial mask
        self.spatial_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 512, kernel_size=2, stride=2), # e.g. 16->32
            nn.GELU(),
            nn.ConvTranspose2d(512, mask_dim, kernel_size=2, stride=2),   # e.g. 32->64
            nn.GELU()
        )

    def forward(self, seg_hidden_state, vision_hidden_states, image_size=(1024, 1024)):
        """
        seg_hidden_state: The final hidden state of the [SEG] token [B, 1, hidden_dim]
        vision_hidden_states: Spatial visual features from ViT [B, seq_len, hidden_dim]
        """
        B, seq_len, dim = vision_hidden_states.shape
        
        # Reshape vision tokens back into spatial grid (assuming 16x16 grid = 256 tokens)
        grid_size = int(seq_len ** 0.5)
        spatial_feats = vision_hidden_states.transpose(1, 2).view(B, dim, grid_size, grid_size)
        
        # 1. Upsample vision features
        mask_embeddings = self.spatial_upsample(spatial_feats) # [B, mask_dim, H', W']
        
        # 2. Project [SEG] hidden state
        seg_vector = self.seg_proj(seg_hidden_state.squeeze(1)) # [B, mask_dim]
        
        # 3. Dot product between [SEG] vector and Spatial Mask Embeddings
        # [B, 1, mask_dim] x [B, mask_dim, H'*W'] -> [B, 1, H'*W']
        H_prime, W_prime = mask_embeddings.shape[2:]
        mask_embeddings = mask_embeddings.view(B, self.mask_dim, -1)
        
        logits = torch.bmm(seg_vector.unsqueeze(1), mask_embeddings)
        logits = logits.view(B, 1, H_prime, W_prime)
        
        # 4. Interpolate to original image size
        pred_mask = F.interpolate(logits, size=image_size, mode="bilinear", align_corners=False)
        return pred_mask

