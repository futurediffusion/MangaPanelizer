"""Speech bubble text overlay node tailored for manga layouts."""

from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .categories import icons
from .config import COLORS, color_mapping
from .core.imaging import pil2tensor, tensor2pil

ROTATE_OPTIONS = ["text center", "image center"]


def _resolve_color(
    selection: str | None,
    mapping: dict[str, Tuple[int, int, int]],
    fallback: str,
) -> Tuple[int, int, int]:
    """Resolve a named colour to an RGB triple with a safe fallback."""

    if not selection:
        selection = fallback

    normalised = selection.lower()
    if normalised == "custom":
        normalised = fallback

    return mapping.get(normalised, mapping.get(fallback, (0, 0, 0)))


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


def _draw_text(draw: ImageDraw.ImageDraw,
               lines: Sequence[Tuple[str, int, int]],
               font: ImageFont.FreeTypeFont,
               text_position_x: int,
               text_position_y: int,
               line_spacing: int,
               fill: Tuple[int, int, int, int]) -> None:
    cursor_y = float(text_position_y)
    for index, (line, width, height) in enumerate(lines):
        x = float(text_position_x)
        draw.text((x, cursor_y), line, font=font, fill=fill)
        cursor_y += height
        if index < len(lines) - 1:
            cursor_y += line_spacing


def _draw_pointer(
    draw: ImageDraw.ImageDraw,
    bubble_width: int,
    bubble_height: int,
    corner_radius: int,
    pointer_length: float,
    pointer_angle: float,
    fill_color: Tuple[int, int, int, int],
    outline_color: Tuple[int, int, int, int],
    outline_width: int,
) -> None:
    if pointer_length <= 0.0:
        return

    center_x = bubble_width / 2.0
    center_y = bubble_height / 2.0

    angle_rad = math.radians(pointer_angle % 360.0)
    direction_x = math.cos(angle_rad)
    direction_y = math.sin(angle_rad)

    if math.isclose(direction_x, 0.0, abs_tol=1e-6) and math.isclose(direction_y, 0.0, abs_tol=1e-6):
        return

    rx = max((bubble_width / 2.0) - max(corner_radius / 2.0, outline_width), 1.0)
    ry = max((bubble_height / 2.0) - max(corner_radius / 2.0, outline_width), 1.0)
    denom = math.sqrt((direction_x ** 2) / (rx ** 2) + (direction_y ** 2) / (ry ** 2))
    boundary_distance = 0.0 if denom == 0 else 1.0 / denom

    base_border_x = center_x + direction_x * boundary_distance
    base_border_y = center_y + direction_y * boundary_distance

    inner_offset = max(outline_width, 4.0)
    base_inner_x = base_border_x - direction_x * inner_offset
    base_inner_y = base_border_y - direction_y * inner_offset

    tangent_x = -direction_y
    tangent_y = direction_x
    tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
    tangent_x /= tangent_length
    tangent_y /= tangent_length

    base_half_width = max(pointer_length * 0.25, outline_width * 1.5, 12.0)

    base_inner_left = (
        base_inner_x + tangent_x * base_half_width,
        base_inner_y + tangent_y * base_half_width,
    )
    base_inner_right = (
        base_inner_x - tangent_x * base_half_width,
        base_inner_y - tangent_y * base_half_width,
    )

    tip = (
        base_border_x + direction_x * pointer_length,
        base_border_y + direction_y * pointer_length,
    )

    draw.polygon([base_inner_left, tip, base_inner_right], fill=fill_color)

    if outline_width > 0:
        border_line = [base_inner_left, tip, base_inner_right]
        draw.line(border_line, fill=outline_color, width=outline_width)


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
                "font_size": ("INT", {"default": 60, "min": 1, "max": 1024}),
                "font_color": (COLORS, {"default": "black"}),
                "bubble_color": (COLORS, {"default": "white"}),
                "border_color": (COLORS, {"default": "black"}),
                "border_thickness": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 64.0, "step": 0.5},
                ),
                "bubble_width": ("INT", {"default": 450, "min": 32, "max": 2048}),
                "bubble_height": ("INT", {"default": 200, "min": 32, "max": 2048}),
                "corner_radius": ("INT", {"default": 40, "min": 1, "max": 100}),
                "line_spacing": ("INT", {"default": 4, "min": -256, "max": 256}),
                "pointer_length": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "pointer_angle": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0},
                ),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": (
                    "FLOAT",
                    {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1},
                ),
                "rotation_options": (ROTATE_OPTIONS,),
            },
            "optional": {
                "text_position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "text_position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
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
        line_spacing,
        pointer_length,
        pointer_angle,
        position_x,
        position_y,
        rotation_angle,
        rotation_options,
        text_position_x: int | None = 0,
        text_position_y: int | None = 0,
    ):
        if bubble_width <= 0 or bubble_height <= 0:
            raise ValueError("Speech bubble dimensions must be greater than zero.")

        text_color = _resolve_color(font_color, color_mapping, "black")
        fill_color = _resolve_color(bubble_color, color_mapping, "white")
        outline_color = _resolve_color(border_color, color_mapping, "black")

        text_position_x = int(text_position_x or 0)
        text_position_y = int(text_position_y or 0)

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
        )

        pointer_fill = (*fill_color, 255)
        pointer_outline = (*outline_color, 255)
        _draw_pointer(
            draw,
            bubble_width,
            bubble_height,
            clamped_radius,
            float(pointer_length or 0.0),
            float(pointer_angle or 0.0),
            pointer_fill,
            pointer_outline,
            outline_width,
        )

        if outline_width:
            draw.rounded_rectangle(
                (0, 0, bubble_width, bubble_height),
                radius=clamped_radius,
                outline=(*outline_color, 255),
                width=outline_width,
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
            text_position_x,
            text_position_y,
            line_spacing,
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
