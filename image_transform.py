"""Image transformation node for MangaPanelizer custom nodes."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def _tensor_to_pil_list(batch: torch.Tensor) -> List[Image.Image]:
    """Convert a batch tensor into a list of PIL images."""
    if batch.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (batch, height, width, channels), got shape {tuple(batch.shape)}"
        )

    images: List[Image.Image] = []
    for frame in batch:
        array = frame.detach().cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype("uint8")
        if array.ndim == 2:
            images.append(Image.fromarray(array, mode="L"))
        elif array.ndim == 3:
            channels = array.shape[2]
            if channels == 1:
                images.append(Image.fromarray(array.squeeze(-1), mode="L"))
            elif channels == 3:
                images.append(Image.fromarray(array, mode="RGB"))
            elif channels == 4:
                images.append(Image.fromarray(array, mode="RGBA"))
            else:
                raise ValueError(f"Unsupported channel count: {channels}")
        else:
            raise ValueError(f"Invalid array shape converted from tensor: {array.shape}")
    return images


def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images into a ComfyUI image tensor."""
    tensors: List[torch.Tensor] = []
    for image in images:
        array = np.array(image).astype(np.float32) / 255.0
        if array.ndim == 2:
            array = array[..., None]
        tensors.append(torch.from_numpy(array))
    if not tensors:
        return torch.zeros((0, 0, 0, 0), dtype=torch.float32)
    batch = torch.stack(tensors, dim=0)
    return batch


class MangaImageTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "y": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "shear": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "border_handling": (["edge", "constant", "reflect", "symmetric"], {"default": "reflect"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = "MangaPanelizer/Image"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling: str = "reflect",
    ):
        if image.size(0) == 0:
            return (image,)

        _, frame_height, frame_width, _ = image.size()

        x = int(x)
        y = int(y)
        angle = int(angle)

        new_height = int(frame_height * zoom)
        new_width = int(frame_width * zoom)

        diagonal = math.sqrt(frame_width**2 + frame_height**2)
        max_padding = math.ceil(diagonal * zoom - min(frame_width, frame_height))

        pw = int(frame_width - new_width) + abs(max_padding)
        ph = int(frame_height - new_height) + abs(max_padding)

        padding = [
            max(0, pw + x),
            max(0, ph + y),
            max(0, pw - x),
            max(0, ph - y),
        ]

        transformed_images: List[Image.Image] = []
        for img in _tensor_to_pil_list(image):
            padded = TF.pad(
                img,
                padding=padding,
                padding_mode=border_handling,
                fill=0,
            )
            transformed = TF.affine(
                padded,
                angle=angle,
                translate=[x, y],
                scale=zoom,
                shear=shear,
                interpolation=Image.BILINEAR,
            )

            left = abs(padding[0])
            upper = abs(padding[1])
            right = transformed.width - abs(padding[2])
            bottom = transformed.height - abs(padding[3])
            transformed_images.append(transformed.crop((left, upper, right, bottom)))

        return (_pil_list_to_tensor(transformed_images),)
