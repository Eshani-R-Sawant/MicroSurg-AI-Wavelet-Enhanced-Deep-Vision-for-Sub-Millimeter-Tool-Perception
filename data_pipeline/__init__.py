"""
SurgMLLM / ZWZ (Zooming Without Zooming) - Data Pipeline Package

End-to-end data curation, micro-crop extraction, oracle distillation,
geometric homography computation, and multimodal assembly pipeline.
"""

from .dataloader import SurgicalVQADataset
from .grounding_transform import ground_data

__all__ = ["SurgicalVQADataset", "ground_data"]
