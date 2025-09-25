"""MangaPanelizer panel layout node."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from ..categories import icons
from ..config import COLORS, color_mapping


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI image tensor into a PIL image."""
    array = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image back into a ComfyUI-compatible tensor."""
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Translate a hex colour string into an RGB tuple."""
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def get_color_values(color: str, color_hex: str) -> tuple[int, int, int]:
    """Resolve a colour preset name or fallback to a manual hex value."""
    if color == "custom":
        return hex_to_rgb(color_hex)
    return color_mapping.get(color, (0, 0, 0))


def crop_and_resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Center crop the image and resize it to the panel size."""
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


def safe_digit(value: str, default: int = 1) -> int:
    """Convert a single character to an integer, falling back gracefully."""
    try:
        parsed = int(value)
        return max(parsed, 1)
    except (TypeError, ValueError):
        return default


def compute_panel_span(total: int, count: int, border: int, outline: int) -> int:
    """Determine how much space a panel can occupy along one axis."""
    available = max(total - (2 * count * (border + outline)), 1)
    return max(available // max(count, 1), 1)


def create_and_paste_panel(
    page: Image.Image,
    border_thickness: int,
    outline_thickness: int,
    panel_width: int,
    panel_height: int,
    page_width: int,
    panel_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    outline_color: tuple[int, int, int],
    images: List[Image.Image],
    row_index: int,
    column_index: int,
    image_index: int,
    total_images: int,
    reading_direction: str,
) -> None:
    """Build a single panel and paste it into place on the page."""
    panel = Image.new("RGB", (panel_width, panel_height), panel_color)

    if image_index < total_images:
        img = crop_and_resize_image(images[image_index], panel_width, panel_height)
        panel.paste(img, (0, 0))

    if outline_thickness > 0:
        panel = ImageOps.expand(panel, border=outline_thickness, fill=outline_color)
    if border_thickness > 0:
        panel = ImageOps.expand(panel, border=border_thickness, fill=background_color)

    new_width, new_height = panel.size
    if reading_direction == "right to left":
        paste_x = page_width - (column_index + 1) * new_width
    else:
        paste_x = column_index * new_width
    paste_y = row_index * new_height

    page.paste(panel, (paste_x, paste_y))


class CR_ComicPanelTemplates:
    """Generate quick comic panel grids ready for manga layouts."""

    @classmethod
    def INPUT_TYPES(cls):
        templates = [
            "custom",
            "G22",
            "G33",
            "H2",
            "H3",
            "H12",
            "H13",
            "H21",
            "H23",
            "H31",
            "H32",
            "V2",
            "V3",
            "V12",
            "V13",
            "V21",
            "V23",
            "V31",
            "V32",
        ]
        directions = ["left to right", "right to left"]

        return {
            "required": {
                "page_width": ("INT", {"default": 1024, "min": 8, "max": 4096}),
                "page_height": ("INT", {"default": 1536, "min": 8, "max": 4096}),
                "template": (templates,),
                "reading_direction": (directions,),
                "border_thickness": ("INT", {"default": 6, "min": 0, "max": 1024}),
                "outline_thickness": ("INT", {"default": 2, "min": 0, "max": 1024}),
                "outline_color": (COLORS,),
                "panel_color": (COLORS,),
                "background_color": (COLORS,),
            },
            "optional": {
                "images": ("IMAGE",),
                "custom_panel_layout": ("STRING", {"multiline": False, "default": "H123"}),
                "outline_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "panel_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "bg_color_hex": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "show_help")
    FUNCTION = "layout"
    CATEGORY = icons.get("MangaPanelizer/Templates")

    def layout(
        self,
        page_width: int,
        page_height: int,
        template: str,
        reading_direction: str,
        border_thickness: int,
        outline_thickness: int,
        outline_color: str,
        panel_color: str,
        background_color: str,
        images: Optional[Iterable[torch.Tensor]] = None,
        custom_panel_layout: str = "H123",
        outline_color_hex: str = "#000000",
        panel_color_hex: str = "#000000",
        bg_color_hex: str = "#FFFFFF",
    ):
        pil_images: List[Image.Image] = []
        if images is not None:
            pil_images = [tensor2pil(image) for image in images]

        outline_rgb = get_color_values(outline_color, outline_color_hex)
        panel_rgb = get_color_values(panel_color, panel_color_hex)
        background_rgb = get_color_values(background_color, bg_color_hex)

        content_width = max(page_width - (2 * border_thickness), 1)
        content_height = max(page_height - (2 * border_thickness), 1)
        page = Image.new("RGB", (content_width, content_height), background_rgb)

        if template == "custom":
            template = custom_panel_layout.strip() or "H123"

        first_char = template[0].upper()
        image_index = 0
        total_images = len(pil_images)
        draw = ImageDraw.Draw(page)

        if first_char == "G":
            rows = safe_digit(template[1], default=1)
            columns = safe_digit(template[2], default=1)
            panel_width = compute_panel_span(page.width, columns, border_thickness, outline_thickness)
            panel_height = compute_panel_span(page.height, rows, border_thickness, outline_thickness)
            for row in range(rows):
                for column in range(columns):
                    create_and_paste_panel(
                        page,
                        border_thickness,
                        outline_thickness,
                        panel_width,
                        panel_height,
                        page.width,
                        panel_rgb,
                        background_rgb,
                        outline_rgb,
                        pil_images,
                        row,
                        column,
                        image_index,
                        total_images,
                        reading_direction,
                    )
                    image_index += 1
        elif first_char == "H":
            rows = max(len(template) - 1, 1)
            panel_height = compute_panel_span(page.height, rows, border_thickness, outline_thickness)
            for row in range(rows):
                columns = safe_digit(template[row + 1], default=1)
                panel_width = compute_panel_span(page.width, columns, border_thickness, outline_thickness)
                for column in range(columns):
                    create_and_paste_panel(
                        page,
                        border_thickness,
                        outline_thickness,
                        panel_width,
                        panel_height,
                        page.width,
                        panel_rgb,
                        background_rgb,
                        outline_rgb,
                        pil_images,
                        row,
                        column,
                        image_index,
                        total_images,
                        reading_direction,
                    )
                    image_index += 1
        elif first_char == "V":
            columns = max(len(template) - 1, 1)
            panel_width = compute_panel_span(page.width, columns, border_thickness, outline_thickness)
            for column in range(columns):
                rows = safe_digit(template[column + 1], default=1)
                panel_height = compute_panel_span(page.height, rows, border_thickness, outline_thickness)
                for row in range(rows):
                    create_and_paste_panel(
                        page,
                        border_thickness,
                        outline_thickness,
                        panel_width,
                        panel_height,
                        page.width,
                        panel_rgb,
                        background_rgb,
                        outline_rgb,
                        pil_images,
                        row,
                        column,
                        image_index,
                        total_images,
                        reading_direction,
                    )
                    image_index += 1
        else:
            draw.text((10, 10), "Unknown template", fill=(255, 0, 0))

        if border_thickness > 0:
            page = ImageOps.expand(page, border_thickness, background_rgb)

        show_help = "MangaPanelizer: Configure manga-ready panel grids."
        return (pil2tensor(page), show_help)
