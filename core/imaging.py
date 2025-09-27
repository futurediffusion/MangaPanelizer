"""Tensor/PIL conversion helpers for MangaPanelizer."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI tensor into a PIL image."""
    array = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a ComfyUI tensor."""
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def crop_and_resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Center crop an image to match a target aspect ratio before resizing."""
    width, height = image.size
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        crop_width = int(height * target_aspect_ratio)
        crop_height = height
        left = (width - crop_width) // 2
        top = 0
    else:
        crop_height = int(width / target_aspect_ratio)
        crop_width = width
        left = 0
        top = (height - crop_height) // 2

    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
