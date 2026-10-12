"""
Wavelet Vision Branch for Zooming Without Zooming (ZWZ) Architecture.

Extracts high-frequency spatial edge features (HH band via 2D Discrete Wavelet Transform)
from full-resolution surgical video frames (e.g., 1024x1024) to preserve fine instrument
contours, micro-scissor tips, needles, and tissue boundaries without mechanical optical zoom.
Projects edge tokens directly into the LLM token embedding space.
"""

import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWTForward
    HAS_PYTORCH_WAVELETS = True
except ImportError:
    HAS_PYTORCH_WAVELETS = False


class NativeHaarDWT2d(nn.Module):
    """
    Pure PyTorch fallback implementation of 2D Haar Discrete Wavelet Transform.
    Computes LL, LH, HL, and HH subbands using fixed-weight 2D convolution filters.
    """
    def __init__(self):
        super().__init__()
        # 2x2 Haar filter weights
        # LL: low frequency approximation
        # LH: horizontal high frequency
        # HL: vertical high frequency
        # HH: diagonal high frequency
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Filters shape: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns:
            yl: [B, C, H/2, W/2] (LL band)
            yh: list with [B, C, 3, H/2, W/2] (LH, HL, HH bands)
        """
        b, c, h, w = x.shape
        # Apply convolution per channel
        x_reshaped = x.reshape(b * c, 1, h, w)
        out = nn.functional.conv2d(x_reshaped, self.filters, stride=2)
        out = out.reshape(b, c, 4, out.shape[-2], out.shape[-1])
        yl = out[:, :, 0, :, :]
        yh = [torch.stack([out[:, :, 1, :, :], out[:, :, 2, :, :], out[:, :, 3, :, :]], dim=2)]
        return yl, yh


class WaveletVisionBranch(nn.Module):
    """
    High-Frequency Wavelet Vision Branch.
    Extracts high-frequency HH-subband edge features from 1024x1024 inputs
    and projects them to sequence tokens matching the LLM embedding dimension.
    """
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        if HAS_PYTORCH_WAVELETS:
            self.dwt = DWTForward(J=1, mode="zero", wave="db1")
        else:
            self.dwt = NativeHaarDWT2d()

        # Project high-frequency edges into LLM hidden dimension
        # Output sequence length: 16x16 = 256 tokens
        self.projection = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(2),
        )

    def forward(self, high_res_image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_res_image_tensor: [B, C, 1024, 1024] high-resolution RGB frames
        Returns:
            edge_tokens: [B, 256, embed_dim] high-frequency edge tokens
        """
        yl, yh = self.dwt(high_res_image_tensor)
        # yh[0] shape: [B, C, 3, H/2, W/2] where index 2 is the HH band
        high_freq_bands = yh[0]
        hh_band = high_freq_bands[:, :, 2, :, :]

        # Project to LLM hidden space: [B, embed_dim, 256] -> [B, 256, embed_dim]
        edge_tokens = self.projection(hh_band)
        return edge_tokens.transpose(1, 2)


class QwenVLWithWavelets(nn.Module):
    """
    Multimodal wrapper combining base Qwen2-VL with the Wavelet High-Frequency Branch.
    Fuses visual token representations from standard vision-language backbone
    with high-frequency edge tokens extracted from uncompressed native frames.
    """
    def __init__(self, qwen_base_model):
        super().__init__()
        self.qwen = qwen_base_model
        hidden_size = getattr(qwen_base_model.config, "hidden_size", 1024)
        self.wavelet_branch = WaveletVisionBranch(embed_dim=hidden_size)

    def forward(self, input_ids, images, pixel_values=None, labels=None):
        """
        Forward pass extracting base hidden states and concatenating wavelet edge tokens.
        """
        # Determine image input representation
        img_input = images if images is not None else pixel_values

        base_outputs = self.qwen(
            input_ids=input_ids,
            pixel_values=img_input if hasattr(self.qwen, "visual") else None,
            output_hidden_states=True,
        )
        base_hidden_states = base_outputs.hidden_states[-1]

        # Extract edge tokens from high-res frames if tensor is 4D [B, C, H, W]
        if isinstance(img_input, torch.Tensor) and img_input.dim() == 4:
            if img_input.shape[-2:] != (1024, 1024):
                high_res = nn.functional.interpolate(img_input, size=(1024, 1024), mode="bilinear", align_corners=False)
            else:
                high_res = img_input
            edge_tokens = self.wavelet_branch(high_res)
            fused_states = torch.cat([base_hidden_states, edge_tokens], dim=1)
        else:
            fused_states = base_hidden_states

        return fused_states
