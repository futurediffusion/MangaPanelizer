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


def _angle_normalize(angle):
    """Normalize angle to [0, 360)."""
    return angle % 360.0


def _angle_in_range(angle, start, end):
    """Check if angle is in range [start, end], handling wraparound."""
    angle = _angle_normalize(angle)
    start = _angle_normalize(start)
    end = _angle_normalize(end)
    
    if start <= end:
        return start <= angle <= end
    else:
        return angle >= start or angle <= end


def _point_at_angle_on_rounded_rect(center_x, center_y, half_width, half_height, radius, angle_deg):
    """Find the exact boundary intersection on a rounded rectangle for a ray."""

    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center_x, center_y

    # Half-width/height of the straight segments (clamp to >= 0).
    straight_half_width = max(0.0, half_width - radius)
    straight_half_height = max(0.0, half_height - radius)

    # Try intersecting with the vertical straight edges first.
    if abs(dx) > 1e-9:
        sign_x = 1.0 if dx > 0 else -1.0
        edge_x = center_x + sign_x * half_width
        t_vertical = (edge_x - center_x) / dx
        if t_vertical > 0:
            y_at_edge = center_y + dy * t_vertical
            if abs(y_at_edge - center_y) <= straight_half_height + 1e-6:
                return edge_x, y_at_edge

    # Try intersecting with the horizontal straight edges.
    if abs(dy) > 1e-9:
        sign_y = 1.0 if dy > 0 else -1.0
        edge_y = center_y + sign_y * half_height
        t_horizontal = (edge_y - center_y) / dy
        if t_horizontal > 0:
            x_at_edge = center_x + dx * t_horizontal
            if abs(x_at_edge - center_x) <= straight_half_width + 1e-6:
                return x_at_edge, edge_y

    # Fall back to the rounded corner arcs.
    if radius <= 0:
        # Degenerate case, treat as rectangle
        # Project to whichever axis is feasible.
        if abs(dx) > abs(dy):
            sign_x = 1.0 if dx > 0 else -1.0
            edge_x = center_x + sign_x * half_width
            t = (edge_x - center_x) / dx if abs(dx) > 1e-9 else 0
            return edge_x, center_y + dy * t
        else:
            sign_y = 1.0 if dy > 0 else -1.0
            edge_y = center_y + sign_y * half_height
            t = (edge_y - center_y) / dy if abs(dy) > 1e-9 else 0
            return center_x + dx * t, edge_y

    # Determine which corner quadrant we're heading towards.
    sign_x = 1.0 if dx >= 0 else -1.0
    sign_y = 1.0 if dy >= 0 else -1.0

    corner_center_x = center_x + sign_x * (straight_half_width)
    corner_center_y = center_y + sign_y * (straight_half_height)

    # Solve intersection between ray and circle centred at the corner.
    # Ray: (center_x, center_y) + t * (dx, dy)
    # Circle: (x - corner_center_x)^2 + (y - corner_center_y)^2 = radius^2
    rel_cx = center_x - corner_center_x
    rel_cy = center_y - corner_center_y

    a = dx * dx + dy * dy  # should be 1
    b = 2 * (dx * rel_cx + dy * rel_cy)
    c = rel_cx * rel_cx + rel_cy * rel_cy - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # Numerical fallback – use the corner point directly.
        return corner_center_x + sign_x * radius, corner_center_y + sign_y * radius

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t_candidates = [t for t in (t1, t2) if t > 0]
    if not t_candidates:
        return corner_center_x + sign_x * radius, corner_center_y + sign_y * radius

    t_hit = min(t_candidates)
    return center_x + dx * t_hit, center_y + dy * t_hit


def _generate_rounded_rect_points(x0, y0, x1, y1, radius, exclude_start_angle=None, exclude_end_angle=None, segments_per_corner=12):
    """Generate points for rounded rectangle, optionally excluding an angular range."""
    points = []
    
    max_radius = min((x1 - x0) / 2, (y1 - y0) / 2)
    radius = min(radius, max_radius)
    
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    
    # Corner centers
    corners = [
        (x0 + radius, y0 + radius, 180, 270),  # Top-left
        (x1 - radius, y0 + radius, 270, 360),  # Top-right
        (x1 - radius, y1 - radius, 0, 90),     # Bottom-right
        (x0 + radius, y1 - radius, 90, 180),   # Bottom-left
    ]
    
    for cx, cy, angle_start, angle_end in corners:
        for i in range(segments_per_corner + 1):
            t = i / segments_per_corner
            arc_angle = angle_start + (angle_end - angle_start) * t
            arc_rad = math.radians(arc_angle)
            
            px = cx + radius * math.cos(arc_rad)
            py = cy + radius * math.sin(arc_rad)
            
            # Calculate angle from center
            point_angle = math.degrees(math.atan2(py - center_y, px - center_x)) % 360
            
            # Skip if in excluded range
            if exclude_start_angle is not None and exclude_end_angle is not None:
                if _angle_in_range(point_angle, exclude_start_angle, exclude_end_angle):
                    continue
            
            points.append((px, py))
    
    return points


def _create_bubble_with_pointer(
    width: int,
    height: int,
    corner_radius: int,
    pointer_length: float,
    pointer_angle: float,
    fill_color: Tuple[int, int, int, int],
    outline_color: Tuple[int, int, int, int],
    outline_width: int,
    pointer_buffer: int,
) -> Image.Image:
    """Create a speech bubble with pointer as a single unified polygon."""
    
    total_width = width + pointer_buffer * 2
    total_height = height + pointer_buffer * 2
    
    bubble_img = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bubble_img, "RGBA")
    
    bubble_x0 = float(pointer_buffer)
    bubble_y0 = float(pointer_buffer)
    bubble_x1 = float(pointer_buffer + width)
    bubble_y1 = float(pointer_buffer + height)
    
    center_x = (bubble_x0 + bubble_x1) / 2
    center_y = (bubble_y0 + bubble_y1) / 2
    half_width = width / 2
    half_height = height / 2
    
    clamped_radius = max(0, min(corner_radius, min(width, height) // 2))
    
    if pointer_length <= 0:
        # No pointer, simple rounded rectangle
        draw.rounded_rectangle(
            (bubble_x0, bubble_y0, bubble_x1, bubble_y1),
            radius=clamped_radius,
            fill=fill_color,
            outline=outline_color if outline_width > 0 else None,
            width=outline_width if outline_width > 0 else 1,
        )
        return bubble_img
    
    # Calculate pointer geometry
    pointer_angle_norm = _angle_normalize(pointer_angle)
    angle_rad = math.radians(pointer_angle_norm)
    direction_x = math.cos(angle_rad)
    direction_y = math.sin(angle_rad)
    
    # Find attachment point on bubble edge
    base_x, base_y = _point_at_angle_on_rounded_rect(
        center_x, center_y, half_width, half_height, clamped_radius, pointer_angle_norm
    )
    
    # Pointer base width
    pointer_base_width = min(max(pointer_length * 0.35, 18.0), 70.0)
    half_base = pointer_base_width / 2
    
    # Tangent (perpendicular to direction)
    tangent_x = -direction_y
    tangent_y = direction_x
    
    # Three pointer points
    base_left = (base_x + tangent_x * half_base, base_y + tangent_y * half_base)
    base_right = (base_x - tangent_x * half_base, base_y - tangent_y * half_base)
    tip = (base_x + direction_x * pointer_length, base_y + direction_y * pointer_length)
    
    # Calculate angles of base points
    angle_left = math.degrees(math.atan2(base_left[1] - center_y, base_left[0] - center_x)) % 360
    angle_right = math.degrees(math.atan2(base_right[1] - center_y, base_right[0] - center_x)) % 360
    
    # Determine exclusion range for bubble outline (expanded for safety)
    margin = 5  # degrees
    if abs(angle_right - angle_left) > 180:
        # Wraparound case
        exclude_start = _angle_normalize(angle_right - margin)
        exclude_end = _angle_normalize(angle_left + margin)
    else:
        exclude_start = _angle_normalize(min(angle_left, angle_right) - margin)
        exclude_end = _angle_normalize(max(angle_left, angle_right) + margin)
    
    # Generate bubble outline points, excluding pointer zone
    bubble_points = _generate_rounded_rect_points(
        bubble_x0, bubble_y0, bubble_x1, bubble_y1,
        clamped_radius,
        exclude_start,
        exclude_end,
        segments_per_corner=15
    )
    
    if not bubble_points:
        # Fallback
        draw.rounded_rectangle(
            (bubble_x0, bubble_y0, bubble_x1, bubble_y1),
            radius=clamped_radius,
            fill=fill_color,
        )
        draw.polygon([base_left, tip, base_right], fill=fill_color)
        return bubble_img
    
    # Determine the orientation of the pointer insertion relative to the
    # counter-clockwise bubble outline.  Instead of simply choosing the point
    # closest to one of the bases (which can lead to self-intersections when the
    # margin removes most of the neighbouring outline points), we compare the
    # travel distance along the outline between the two nearest locations.

    def _nearest_index(point):
        closest_index = 0
        closest_dist = float("inf")
        for idx, candidate in enumerate(bubble_points):
            dist = math.hypot(candidate[0] - point[0], candidate[1] - point[1])
            if dist < closest_dist:
                closest_dist = dist
                closest_index = idx
        return closest_index

    idx_left = _nearest_index(base_left)
    idx_right = _nearest_index(base_right)

    angle_diff = (angle_right - angle_left) % 360

    if idx_left == idx_right:
        # When both bases map to the same outline vertex (can happen with small
        # exclusion margins), fall back to angular ordering.
        if angle_diff < 180:
            insert_idx = idx_left
            insert_sequence = [base_left, tip, base_right]
        else:
            insert_idx = idx_right
            insert_sequence = [base_right, tip, base_left]
    else:
        total_points = len(bubble_points)
        forward_distance = (idx_right - idx_left) % total_points
        backward_distance = (idx_left - idx_right) % total_points

        if forward_distance != 0 and (forward_distance < backward_distance or backward_distance == 0):
            insert_idx = idx_left
            insert_sequence = [base_left, tip, base_right]
        else:
            insert_idx = idx_right
            insert_sequence = [base_right, tip, base_left]

    final_points = (
        bubble_points[:insert_idx + 1]
        + insert_sequence
        + bubble_points[insert_idx + 1:]
    )
    
    # Draw filled polygon
    draw.polygon(final_points, fill=fill_color)
    
    # Draw outline
    if outline_width > 0:
        closed_points = final_points + [final_points[0]]
        draw.line(closed_points, fill=outline_color, width=outline_width, joint="curve")
    
    return bubble_img


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
                    {"default": 0.0, "min": 0.0, "max": 2048.0, "step": 0.5},
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

        border_px = max(0.0, float(border_thickness))
        outline_width = int(round(border_px)) if border_px >= 1.0 else 0

        pointer_length_value = float(pointer_length or 0.0)
        pointer_buffer = 0
        if pointer_length_value > 0.0:
            pointer_buffer = int(math.ceil(pointer_length_value + outline_width * 2))

        # Create bubble with pointer
        bubble_image = _create_bubble_with_pointer(
            bubble_width,
            bubble_height,
            corner_radius,
            pointer_length_value,
            float(pointer_angle or 0.0),
            (*fill_color, 255),
            (*outline_color, 255),
            outline_width,
            pointer_buffer,
        )

        # Draw text
        draw = ImageDraw.Draw(bubble_image, "RGBA")
        font_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
        font = _load_font(font_dir, font_name, font_size)

        text_lines = text.splitlines() or [""]
        line_metrics = _measure_lines(draw, text_lines, font)

        text_fill = (*text_color, 255)
        text_offset_x = pointer_buffer + text_position_x
        text_offset_y = pointer_buffer + text_position_y

        _draw_text(
            draw,
            line_metrics,
            font,
            text_offset_x,
            text_offset_y,
            line_spacing,
            text_fill,
        )

        # Composite
        composite_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
        paste_position = (position_x - pointer_buffer, position_y - pointer_buffer)
        composite_layer.paste(bubble_image, paste_position, bubble_image)

        # Rotate
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
        show_help = "Speech bubble overlay - clean edge integration"
        return pil2tensor(merged), show_help


NODE_CLASS_MAPPINGS = {
    "MangaSpeechBubbleOverlay": MangaSpeechBubbleOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaSpeechBubbleOverlay": "Manga Speech Bubble Overlay",
}
