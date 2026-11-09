#!/usr/bin/env python
"""
Image Conversion Helpers
"""

from typing import Union

import numpy as np
from PIL import Image


def convert_to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Convert a numpy array to a PIL Image if needed.

    Float arrays with values in [0, 1] are scaled to [0, 255]; anything else
    is cast to uint8 as-is.
    """
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        return Image.fromarray(image)
    return image


def convert_mask_to_grayscale(mask: Image.Image) -> Image.Image:
    """Ensure mask is in grayscale ("L") mode."""
    if mask.mode != "L":
        return mask.convert("L")
    return mask
