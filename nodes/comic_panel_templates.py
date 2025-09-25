"""MangaPanelizer panel layout node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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


def safe_int(value: str, default: int = 1) -> int:
    """Parse an integer from text, falling back gracefully."""
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


@dataclass
class PanelShape:
    polygon: List[Tuple[float, float]]


@dataclass
class DiagonalInfo:
    """Information about diagonal separators."""
    horizontal: bool = False  # Diagonal between rows
    vertical: bool = False    # Diagonal between columns within a row
    angle: float = 0.2        # Diagonal offset ratio (0.0 to 1.0)


def interpolate(value_left: float, value_right: float, position: float, span: float) -> float:
    """Linearly interpolate between two values across a span."""
    if span <= 0:
        return value_left
    ratio = min(max(position / span, 0.0), 1.0)
    return value_left + (value_right - value_left) * ratio


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to an inclusive range."""
    return max(minimum, min(value, maximum))


def parse_layout_sequence(sequence: str) -> tuple[List[int], List[bool]]:
    """Parse a sequence string (e.g. "1/23") into counts and diagonal flags."""
    counts: List[int] = []
    diagonals: List[bool] = []
    diagonal_pending = False

    for char in sequence:
        if char.isdigit():
            counts.append(safe_int(char))

            if len(counts) > 1 and len(diagonals) < len(counts) - 1:
                diagonals.extend([False] * (len(counts) - 1 - len(diagonals)))

            if diagonal_pending and len(counts) >= 2:
                diagonals[len(counts) - 2] = True
                diagonal_pending = False
        elif char == "/":
            if counts:
                diagonal_pending = True
        else:
            diagonal_pending = False

    if counts and len(diagonals) < len(counts) - 1:
        diagonals.extend([False] * (len(counts) - 1 - len(diagonals)))

    if not counts:
        counts = [1]
        diagonals = []

    return counts, diagonals


def parse_enhanced_layout_sequence(sequence: str) -> tuple[List[int], List[DiagonalInfo], List[DiagonalInfo]]:
    """
    Parse enhanced sequence string with support for vertical and horizontal diagonals.
    
    Examples:
    - "1|2"     -> [1,2], [DiagonalInfo(vertical=True)], []
    - "1/2"     -> [1,2], [DiagonalInfo(horizontal=True)], []  
    - "1/2|"    -> [1,2], [DiagonalInfo(horizontal=True)], [DiagonalInfo(vertical=True)]
    """
    counts: List[int] = []
    row_diagonals: List[DiagonalInfo] = []  # Between rows
    col_diagonals: List[DiagonalInfo] = []  # Within rows (between columns)
    
    # Split by angle specifier first
    if ':' in sequence:
        layout_part, angle_part = sequence.rsplit(':', 1)
        try:
            angle_degrees = float(angle_part)
            angle_ratio = min(max(angle_degrees / 90.0, 0.0), 1.0)
        except ValueError:
            angle_ratio = 0.2
    else:
        layout_part = sequence
        angle_ratio = 0.2
    
    # Parse each character
    i = 0
    pending_horizontal = False
    pending_vertical = False
    
    while i < len(layout_part):
        char = layout_part[i]
        
        if char.isdigit():
            counts.append(int(char))
            
            # Apply pending diagonal flags to row transitions
            if len(counts) > 1 and len(row_diagonals) < len(counts) - 1:
                diag_info = DiagonalInfo(angle=angle_ratio)
                if pending_horizontal:
                    diag_info.horizontal = True
                    pending_horizontal = False
                row_diagonals.append(diag_info)
            
            # Apply vertical diagonal to this specific row
            if pending_vertical:
                # Ensure we have col_diagonals for this row index
                while len(col_diagonals) < len(counts):
                    col_diagonals.append(DiagonalInfo(angle=angle_ratio))
                col_diagonals[len(counts) - 1].vertical = True
                pending_vertical = False
                
        elif char == '/':
            pending_horizontal = True
            
        elif char == '|':
            pending_vertical = True
            
        i += 1
    
    # Ensure proper counts
    while len(row_diagonals) < len(counts) - 1:
        row_diagonals.append(DiagonalInfo(angle=angle_ratio))
    
    while len(col_diagonals) < len(counts):
        col_diagonals.append(DiagonalInfo(angle=angle_ratio))
    
    if not counts:
        counts = [1]
    
    return counts, row_diagonals, col_diagonals


def mirror_polygon_horizontally(polygon: Sequence[Tuple[float, float]], width: int) -> List[Tuple[float, float]]:
    """Mirror a polygon across the vertical axis of the page."""
    mirrored: List[Tuple[float, float]] = []
    for x, y in polygon:
        mirrored.append((width - x, y))
    return mirrored


def build_panel_image(
    panel_width: int,
    panel_height: int,
    panel_color: tuple[int, int, int],
    outline_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    border_thickness: int,
    outline_thickness: int,
    images: List[Image.Image],
    image_index: int,
) -> tuple[Image.Image, int]:
    """Create a rectangular panel image with the configured styling."""
    panel = Image.new("RGB", (panel_width, panel_height), panel_color)
    if 0 <= image_index < len(images):
        img = crop_and_resize_image(images[image_index], panel_width, panel_height)
        panel.paste(img, (0, 0))

    if outline_thickness > 0:
        panel = ImageOps.expand(panel, border=outline_thickness, fill=outline_color)
    if border_thickness > 0:
        panel = ImageOps.expand(panel, border=border_thickness, fill=background_color)

    return panel, image_index + 1


def paste_panel_polygon(
    page: Image.Image,
    panel_image: Image.Image,
    polygon: Sequence[Tuple[float, float]],
    bounds: tuple[int, int, int, int],
) -> None:
    """Paste a panel image onto the page using a polygon mask."""
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)

    if panel_image.size != (width, height):
        panel_image = panel_image.resize((width, height), Image.Resampling.LANCZOS)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    relative = [(x - min_x, y - min_y) for x, y in polygon]
    draw.polygon(relative, fill=255)

    page.paste(panel_image, (min_x, min_y), mask)


def draw_polygon_borders(
    page: Image.Image,
    polygon: Sequence[Tuple[float, float]],
    border_thickness: int,
    outline_thickness: int,
    background_color: tuple[int, int, int],
    outline_color: tuple[int, int, int],
) -> None:
    """Render border and outline strokes around a polygon."""
    if border_thickness <= 0 and outline_thickness <= 0:
        return

    draw = ImageDraw.Draw(page)
    closed = list(polygon) + [polygon[0]]

    if border_thickness > 0:
        draw.line(closed, fill=background_color, width=max(border_thickness, 1))
    if outline_thickness > 0:
        draw.line(closed, fill=outline_color, width=max(outline_thickness, 1))


def polygon_bounds(polygon: Sequence[Tuple[float, float]]) -> tuple[int, int, int, int]:
    """Compute integer bounds for a polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x = int(np.floor(min(xs)))
    min_y = int(np.floor(min(ys)))
    max_x = int(np.ceil(max(xs)))
    max_y = int(np.ceil(max(ys)))
    return min_x, min_y, max_x, max_y


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
                "custom_panel_layout": ("STRING", {"multiline": False, "default": "H1|2:30"}),
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
        custom_panel_layout: str = "H1|2:30",
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
            template = custom_panel_layout.strip() or "H1|2:30"

        first_char = template[0].upper()
        image_index = 0
        total_images = len(pil_images)
        draw = ImageDraw.Draw(page)

        # Check for enhanced diagonal syntax
        use_enhanced = "|" in template or ":" in template
        use_diagonal = "/" in template or use_enhanced

        if not use_diagonal:
            # Original non-diagonal code remains the same
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
        else:
            # Enhanced diagonal handling
            panels: List[PanelShape] = []

            if first_char == "H":
                if use_enhanced:
                    row_counts, row_diagonals, row_verticals = parse_enhanced_layout_sequence(template[1:])
                else:
                    row_counts, old_diagonals = parse_layout_sequence(template[1:])
                    # Convert old format to new format
                    row_diagonals = [DiagonalInfo(horizontal=diag) for diag in old_diagonals]
                    row_verticals = [DiagonalInfo() for _ in row_counts]
                
                margin = border_thickness + outline_thickness
                panel_height = compute_panel_span(page.height, len(row_counts), border_thickness, outline_thickness)
                cell_height = panel_height + 2 * margin

                top_left = 0.0
                top_right = 0.0
                
                for index, count in enumerate(row_counts):
                    panel_width = compute_panel_span(page.width, count, border_thickness, outline_thickness)
                    cell_width = panel_width + 2 * margin

                    bottom_left = clamp(top_left + cell_height, 0.0, float(page.height))
                    bottom_right = clamp(top_right + cell_height, 0.0, float(page.height))
                    
                    diagonal_info = row_diagonals[index] if index < len(row_diagonals) else DiagonalInfo()
                    vertical_info = row_verticals[index] if index < len(row_verticals) else DiagonalInfo()
                    
                    # Apply horizontal diagonal
                    if diagonal_info.horizontal:
                        offset_y = page.height * diagonal_info.angle
                        bottom_right = clamp(bottom_right + offset_y, 0.0, float(page.height))

                    # Generate panels for this row
                    x_position = 0.0
                    for column in range(count):
                        left = clamp(x_position, 0.0, float(page.width))
                        right = clamp(left + cell_width, 0.0, float(page.width))
                        
                        top_left_y = clamp(interpolate(top_left, top_right, left, page.width), 0.0, float(page.height))
                        top_right_y = clamp(interpolate(top_left, top_right, right, page.width), 0.0, float(page.height))
                        bottom_left_y = clamp(interpolate(bottom_left, bottom_right, left, page.width), 0.0, float(page.height))
                        bottom_right_y = clamp(interpolate(bottom_left, bottom_right, right, page.width), 0.0, float(page.height))
                        
                        # Apply vertical diagonal within this row
                        if vertical_info.vertical and column < count - 1:
                            offset_x = page.width * vertical_info.angle
                            # Modify the right edge for vertical diagonal effect
                            top_right_y = clamp(top_right_y - offset_x, 0.0, float(page.height))
                            bottom_right_y = clamp(bottom_right_y + offset_x, 0.0, float(page.height))
                        
                        panels.append(
                            PanelShape(
                                polygon=[
                                    (left, top_left_y),
                                    (right, top_right_y),
                                    (right, bottom_right_y),
                                    (left, bottom_left_y),
                                ]
                            )
                        )
                        x_position = right

                    top_left = bottom_left
                    top_right = bottom_right
                    
            elif first_char == "V":
                # Similar logic for vertical layouts with enhanced diagonals
                if use_enhanced:
                    column_counts, column_diagonals, _column_verticals = parse_enhanced_layout_sequence(template[1:])
                else:
                    column_counts, old_diagonals = parse_layout_sequence(template[1:])
                    column_diagonals = [DiagonalInfo(horizontal=diag) for diag in old_diagonals]
                    _column_verticals = [DiagonalInfo() for _ in column_counts]
                
                margin = border_thickness + outline_thickness
                panel_width = compute_panel_span(page.width, len(column_counts), border_thickness, outline_thickness)
                cell_width = panel_width + 2 * margin

                left_top = 0.0
                left_bottom = 0.0
                
                for index, count in enumerate(column_counts):
                    panel_height = compute_panel_span(page.height, count, border_thickness, outline_thickness)
                    cell_height = panel_height + 2 * margin
                    
                    right_top = clamp(left_top + cell_width, 0.0, float(page.width))
                    right_bottom = clamp(left_bottom + cell_width, 0.0, float(page.width))

                    diagonal_info = column_diagonals[index] if index < len(column_diagonals) else DiagonalInfo()
                    
                    if diagonal_info.horizontal:
                        offset_x = page.width * diagonal_info.angle
                        right_bottom = clamp(right_bottom + offset_x, 0.0, float(page.width))

                    y_position = 0.0
                    for row in range(count):
                        top = clamp(y_position, 0.0, float(page.height))
                        bottom = clamp(top + cell_height, 0.0, float(page.height))
                        
                        left_top_x = clamp(interpolate(left_top, left_bottom, top, page.height), 0.0, float(page.width))
                        left_bottom_x = clamp(interpolate(left_top, left_bottom, bottom, page.height), 0.0, float(page.width))
                        right_top_x = clamp(interpolate(right_top, right_bottom, top, page.height), 0.0, float(page.width))
                        right_bottom_x = clamp(interpolate(right_top, right_bottom, bottom, page.height), 0.0, float(page.width))
                        
                        panels.append(
                            PanelShape(
                                polygon=[
                                    (left_top_x, top),
                                    (right_top_x, top),
                                    (right_bottom_x, bottom),
                                    (left_bottom_x, bottom),
                                ]
                            )
                        )
                        y_position = bottom

                    left_top = right_top
                    left_bottom = right_bottom
            else:
                draw.text((10, 10), "Unknown template", fill=(255, 0, 0))

            if reading_direction == "right to left":
                panels = [PanelShape(polygon=mirror_polygon_horizontally(panel.polygon, page.width)) for panel in panels]

            for panel in panels:
                min_x, min_y, max_x, max_y = polygon_bounds(panel.polygon)
                width = max(max_x - min_x, 1)
                height = max(max_y - min_y, 1)
                panel_image, image_index = build_panel_image(
                    width,
                    height,
                    panel_rgb,
                    outline_rgb,
                    background_rgb,
                    border_thickness,
                    outline_thickness,
                    pil_images,
                    image_index,
                )
                paste_panel_polygon(page, panel_image, panel.polygon, (min_x, min_y, max_x, max_y))
                draw_polygon_borders(page, panel.polygon, border_thickness, outline_thickness, background_rgb, outline_rgb)

        if border_thickness > 0:
            page = ImageOps.expand(page, border_thickness, background_rgb)

        show_help = "MangaPanelizer: Configure manga-ready panel grids. Use | for vertical diagonals, :angle for custom angles."
        return (pil2tensor(page), show_help)
