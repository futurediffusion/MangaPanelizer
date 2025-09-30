"""Speech bubble text overlay node tailored for manga layouts."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .categories import icons
from .config import COLORS, color_mapping
from .core.imaging import pil2tensor, tensor2pil

ALIGN_OPTIONS = ["center", "top", "bottom"]
ROTATE_OPTIONS = ["text center", "image center"]
JUSTIFY_OPTIONS = ["center", "left", "right"]


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("Expected a 6 character hex value.")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _resolve_color(selection: str, mapping: dict[str, Tuple[int, int, int]], hex_value: str | None = None) -> Tuple[int, int, int]:
    if selection == "custom" and hex_value:
        try:
            return _hex_to_rgb(hex_value)
        except ValueError:
            return mapping.get("black", (0, 0, 0))
    return mapping.get(selection, mapping.get("black", (0, 0, 0)))


def _load_font(font_dir: str, font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    font_path = os.path.join(font_dir, font_name)
    return ImageFont.truetype(font_path, font_size)


def _measure_lines(draw: ImageDraw.ImageDraw, lines: Sequence[str], font: ImageFont.FreeTypeFont) -> List[Tuple[str, int, int]]:
    measurements: List[Tuple[str, int, int]] = []
    ascent, descent = font.getmetrics()
    default_height = ascent + descent

    for line in lines:
        if line:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        else:
            width = 0
            height = default_height
        measurements.append((line, width, height))
    return measurements


def _text_block_size(lines: Sequence[Tuple[str, int, int]], line_spacing: int) -> Tuple[int, int]:
    max_width = 0
    total_height = 0

    for index, (_, width, height) in enumerate(lines):
        max_width = max(max_width, width)
        total_height += height
        if index < len(lines) - 1:
            total_height += line_spacing
    return max_width, total_height


def _compute_start_y(bubble_height: int, content_height: int, margins: int, align: str) -> float:
    if align == "top":
        return float(margins)
    if align == "bottom":
        return float(bubble_height - margins - content_height)
    return float((bubble_height - content_height) / 2)


def _compute_line_x(bubble_width: int, line_width: int, margins: int, justify: str) -> float:
    if justify == "left":
        return float(margins)
    if justify == "right":
        return float(bubble_width - margins - line_width)
    return float((bubble_width - line_width) / 2)


def _draw_text(draw: ImageDraw.ImageDraw,
               lines: Sequence[Tuple[str, int, int]],
               font: ImageFont.FreeTypeFont,
               bubble_width: int,
               bubble_height: int,
               margins: int,
               line_spacing: int,
               align: str,
               justify: str,
               fill: Tuple[int, int, int, int]) -> None:
    _, content_height = _text_block_size(lines, line_spacing)
    cursor_y = _compute_start_y(bubble_height, content_height, margins, align)

    for index, (line, width, height) in enumerate(lines):
        x = _compute_line_x(bubble_width, width, margins, justify)
        draw.text((x, cursor_y), line, font=font, fill=fill)
        cursor_y += height
        if index < len(lines) - 1:
            cursor_y += line_spacing


class MangaSpeechBubbleOverlay:
    """Overlay speech-bubble styled text onto an image."""

    @classmethod
    def INPUT_TYPES(cls):
        font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
        file_list = [
            f
            for f in os.listdir(font_dir)
            if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Say something!"}),
                "font_name": (file_list,),
                "font_size": ("INT", {"default": 42, "min": 1, "max": 1024}),
                "font_color": (COLORS,),
                "bubble_color": (COLORS,),
                "border_color": (COLORS,),
                "border_thickness": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "bubble_width": ("INT", {"default": 360, "min": 32, "max": 2048}),
                "bubble_height": ("INT", {"default": 320, "min": 32, "max": 2048}),
                "corner_radius": ("INT", {"default": 40, "min": 0, "max": 1024}),
                "align": (ALIGN_OPTIONS,),
                "justify": (JUSTIFY_OPTIONS,),
                "margins": ("INT", {"default": 24, "min": -512, "max": 512}),
                "line_spacing": ("INT", {"default": 4, "min": -256, "max": 256}),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotation_options": (ROTATE_OPTIONS,),
            },
            "optional": {
                "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "show_help")
    FUNCTION = "overlay"
    CATEGORY = icons.get("MangaPanelizer/SpeechBubbles", "🗨️ MangaPanelizer/SpeechBubbles")

    def overlay(
        self,
        image,
        text,
        font_name,
        font_size,
        font_color,
        bubble_color,
        border_color,
        border_thickness,
        bubble_width,
        bubble_height,
        corner_radius,
        align,
        justify,
        margins,
        line_spacing,
        position_x,
        position_y,
        rotation_angle,
        rotation_options,
        font_color_hex="#000000",
    ):
        if bubble_width <= 0 or bubble_height <= 0:
            raise ValueError("Speech bubble dimensions must be greater than zero.")

        text_color = _resolve_color(font_color, color_mapping, font_color_hex)
        fill_color = _resolve_color(bubble_color, color_mapping)
        outline_color = _resolve_color(border_color, color_mapping)

        image_3d = image[0, :, :, :]
        background = tensor2pil(image_3d).convert("RGBA")

        bubble_image = Image.new("RGBA", (bubble_width, bubble_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bubble_image, "RGBA")

        clamped_radius = max(0, min(int(corner_radius), min(bubble_width, bubble_height) // 2))

        border_px = max(0.0, float(border_thickness))
        outline_width = int(round(border_px)) if border_px >= 1.0 else 0

        draw.rounded_rectangle(
            (0, 0, bubble_width, bubble_height),
            radius=clamped_radius,
            fill=(*fill_color, 255),
            outline=(*outline_color, 255) if outline_width else None,
            width=outline_width or 1,
        )

        font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
        font = _load_font(font_dir, font_name, font_size)

        text_lines = text.splitlines() or [""]
        line_metrics = _measure_lines(draw, text_lines, font)

        text_fill = (*text_color, 255)
        _draw_text(
            draw,
            line_metrics,
            font,
            bubble_width,
            bubble_height,
            margins,
            line_spacing,
            align,
            justify,
            text_fill,
        )

        composite_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
        composite_layer.paste(bubble_image, (position_x, position_y), bubble_image)

        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
        if abs(rotation_angle) > 0.0:
            if rotation_options == "image center":
                center = (background.width / 2, background.height / 2)
            else:
                center = (
                    position_x + bubble_width / 2,
                    position_y + bubble_height / 2,
                )
            composite_layer = composite_layer.rotate(
                rotation_angle,
                resample=resampling,
                expand=False,
                center=center,
            )

        merged = Image.alpha_composite(background, composite_layer).convert("RGB")
        show_help = "Speech bubble overlay for MangaPanelizer"
        return pil2tensor(merged), show_help


NODE_CLASS_MAPPINGS = {
    "MangaSpeechBubbleOverlay": MangaSpeechBubbleOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaSpeechBubbleOverlay": "Manga Speech Bubble Overlay",
}
