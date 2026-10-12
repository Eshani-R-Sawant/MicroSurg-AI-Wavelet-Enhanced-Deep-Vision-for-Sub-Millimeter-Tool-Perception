"""
Visual Grounding Transformation Utility.

Superimposes visual bounding boxes and anchors question prompts
to specific spatial coordinate regions for grounded surgical instruction tuning.
"""

import cv2
import numpy as np


def ground_data(image: np.ndarray, question: str, bbox: list):
    """
    Args:
        image: BGR numpy image array
        question: natural language question string
        bbox: [x, y, w, h] pixel bounding box coordinates

    Returns:
        grounded_img: image with rendered bounding box overlay
        grounded_q: prompt augmented with spatial grounding guidance
    """
    x, y, w, h = map(int, bbox)
    grounded_img = image.copy()
    cv2.rectangle(grounded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    grounded_q = question + " Focus only on the object inside the red bounding box."
    return grounded_img, grounded_q
