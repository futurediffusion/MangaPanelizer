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
    array = np.clip(255.0 * image.cpu().numpy().squeeze(),
                    0, 255).astype(np.uint8)
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


def compute_panel_span(total: float, count: int, padding: float) -> float:
    """Determine usable span for panels along one axis with optional internal padding."""
    if count <= 0:
        return float(max(total, 1))

    # Proper internal padding calculation
    effective_total = float(total) - max(count - 1, 0) * float(padding)
    if effective_total <= 0:
        return float(max(total, 1)) / float(max(count, 1))
    return effective_total / float(count)


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
    horizontal: bool = False  # Diagonal between rows (H) or between columns (V)
    # Diagonal within a row (H) or within a column (V)
    vertical: bool = False
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


def parse_enhanced_layout_sequence(sequence: str, orientation: str = "horizontal") -> tuple[List[int], List[DiagonalInfo], List[DiagonalInfo]]:
    """
    Parse enhanced sequence string with support for vertical and horizontal diagonals.

    For horizontal layouts (H):
    - "/" means diagonal between rows
    - "|" means vertical diagonal within a row (between columns in that row)

    For vertical layouts (V):
    - "/" means horizontal diagonal within a column (between panels in that column)  
    - "|" means diagonal between columns
    """
    counts: List[int] = []
    between_diagonals: List[DiagonalInfo] = []  # Between groups (rows/columns)
    within_diagonals: List[DiagonalInfo] = []   # Within a group

    if ":" in sequence:
        layout_part, angle_part = sequence.rsplit(":", 1)
        try:
            angle_degrees = float(angle_part)
            angle_ratio = min(max(angle_degrees / 90.0, 0.0), 1.0)
        except ValueError:
            angle_ratio = 0.2
    else:
        layout_part = sequence
        angle_ratio = 0.2

    # Determine which character means what based on orientation
    is_vertical = orientation.lower() == "vertical"
    # Between columns (V) or rows (H)
    between_char = "|" if is_vertical else "/"
    within_char = "/" if is_vertical else "|"   # Within column (V) or row (H)

    pending_between = False
    pending_within = False
    i = 0

    while i < len(layout_part):
        char = layout_part[i]

        if char.isdigit():
            counts.append(int(char))

            # Handle between-group diagonals
            if len(counts) > 1 and len(between_diagonals) < len(counts) - 1:
                diag_info = DiagonalInfo(angle=angle_ratio)
                if pending_between:
                    diag_info.horizontal = True
                between_diagonals.append(diag_info)
                pending_between = False

            # Handle within-group diagonals
            if pending_within:
                while len(within_diagonals) < len(counts):
                    within_diagonals.append(DiagonalInfo(angle=angle_ratio))
                within_diagonals[len(counts) - 1].vertical = True
                pending_within = False

        elif char == between_char:
            pending_between = True
        elif char == within_char:
            pending_within = True

        i += 1

    # Fill missing diagonal info
    while len(between_diagonals) < max(len(counts) - 1, 0):
        between_diagonals.append(DiagonalInfo(angle=angle_ratio))

    while len(within_diagonals) < len(counts):
        within_diagonals.append(DiagonalInfo(angle=angle_ratio))

    if not counts:
        counts = [1]

    return counts, between_diagonals, within_diagonals


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
    external_padding: int,
    outline_thickness: int,
    images: List[Image.Image],
    image_index: int,
) -> tuple[Image.Image, int]:
    """Create a rectangular panel image with the configured styling."""
    panel = Image.new("RGB", (panel_width, panel_height), panel_color)
    if 0 <= image_index < len(images):
        img = crop_and_resize_image(
            images[image_index], panel_width, panel_height)
        panel.paste(img, (0, 0))

    if outline_thickness > 0:
        panel = ImageOps.expand(
            panel, border=outline_thickness, fill=outline_color)
    if external_padding > 0:
        panel = ImageOps.expand(
            panel, border=external_padding, fill=background_color)

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
        panel_image = panel_image.resize(
            (width, height), Image.Resampling.LANCZOS)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    relative = [(x - min_x, y - min_y) for x, y in polygon]
    draw.polygon(relative, fill=255)

    page.paste(panel_image, (min_x, min_y), mask)


def draw_polygon_borders(
    page: Image.Image,
    polygons: Sequence[Sequence[Tuple[float, float]]],
    outline_thickness: int,
    outline_color: tuple[int, int, int],
) -> None:
    """Render outline strokes with uniform thickness and antialiasing."""
    if outline_thickness <= 0 or not polygons:
        return

    edge_lookup: dict[tuple[Tuple[int, int], Tuple[int, int]], tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for polygon in polygons:
        if not polygon:
            continue
        for index in range(len(polygon)):
            start_point = polygon[index]
            end_point = polygon[(index + 1) % len(polygon)]
            start_pixel = (int(round(start_point[0])), int(round(start_point[1])))
            end_pixel = (int(round(end_point[0])), int(round(end_point[1])))
            if start_pixel == end_pixel:
                continue
            if start_pixel <= end_pixel:
                edge_key = (start_pixel, end_pixel)
                edge_value = (start_point, end_point)
            else:
                edge_key = ((end_pixel), (start_pixel))
                edge_value = (end_point, start_point)
            edge_lookup.setdefault(edge_key, edge_value)

    if not edge_lookup:
        return

    scale_factor = 4
    temp_image = Image.new("RGBA", (page.width * scale_factor, page.height * scale_factor), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_image)
    scaled_width = max(outline_thickness, 1) * scale_factor
    rgba_color = tuple(outline_color) + (255,)

    for start_point, end_point in edge_lookup.values():
        draw.line(
            [
                (start_point[0] * scale_factor, start_point[1] * scale_factor),
                (end_point[0] * scale_factor, end_point[1] * scale_factor),
            ],
            fill=rgba_color,
            width=scaled_width,
        )

    antialiased = temp_image.resize(page.size, Image.Resampling.LANCZOS)
    page.paste(antialiased, (0, 0), antialiased)
def polygon_bounds(polygon: Sequence[Tuple[float, float]]) -> tuple[int, int, int, int]:
    """Compute integer bounds for a polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x = int(np.floor(min(xs)))
    min_y = int(np.floor(min(ys)))
    max_x = int(np.ceil(max(xs)))
    max_y = int(np.ceil(max(ys)))
    return min_x, min_y, max_x, max_y


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
            "H1|2",
            "H1/2",
            "H2|1",
            "H2/1",
            "V2",
            "V3",
            "V12",
            "V13",
            "V21",
            "V23",
            "V31",
            "V32",
            "V1|2",
            "V1/2",
            "V2|1",
            "V2/1",
            "V1/|2",
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
                "custom_panel_layout": ("STRING", {"multiline": True, "default": "H12", "forceInput": True}),
                "internal_padding": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "division_height_offset": ("INT", {"default": 0, "min": -200, "max": 200}),
                "division_horizontal_offset": ("INT", {"default": 0, "min": -200, "max": 200}),
            },
            "optional": {
                "images": ("IMAGE",),
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
        custom_panel_layout: str,
        images: Optional[Iterable[torch.Tensor]] = None,
        internal_padding: Optional[int] = None,
        division_height_offset: Optional[int] = None,
        division_horizontal_offset: Optional[int] = None,
        outline_color_hex: str = "#000000",
        panel_color_hex: str = "#000000",
        bg_color_hex: str = "#FFFFFF",
    ):
        # Convert images
        pil_images: List[Image.Image] = []
        if images is not None:
            pil_images = [tensor2pil(image) for image in images]

        # Get colors
        outline_rgb = get_color_values(outline_color, outline_color_hex)
        panel_rgb = get_color_values(panel_color, panel_color_hex)
        background_rgb = get_color_values(background_color, bg_color_hex)

        # Safe parameter conversion
        external_padding = max(border_thickness, 0)
        internal_padding_value = max(
            internal_padding if internal_padding is not None else 0, 0)

        # New parameters for division control
        height_offset = division_height_offset if division_height_offset is not None else 0
        horizontal_offset = division_horizontal_offset if division_horizontal_offset is not None else 0

        # Setup canvas
        content_width = max(page_width - (2 * external_padding), 1)
        content_height = max(page_height - (2 * external_padding), 1)
        page = Image.new(
            "RGB", (content_width, content_height), background_rgb)

        # Use custom template if selected
        if template == "custom":
            if custom_panel_layout and custom_panel_layout.strip():
                template = custom_panel_layout.strip()
            else:
                template = "H12"

        first_char = template[0].upper()
        image_index = 0
        panels: List[PanelShape] = []

        # Check for diagonal syntax
        use_enhanced = "|" in template or ":" in template
        use_diagonal = "/" in template or use_enhanced

        if not use_diagonal:
            # Regular grid layouts without diagonals
            if first_char == "G":
                # Grid layout
                rows = safe_digit(template[1], default=1)
                columns = safe_digit(template[2], default=1)

                panel_width = compute_panel_span(
                    content_width, columns, internal_padding_value)
                panel_height = compute_panel_span(
                    content_height, rows, internal_padding_value)

                # Calculate positions with proper internal padding
                x_positions: List[float] = []
                y_positions: List[float] = []

                current_x = 0.0
                for col in range(columns):
                    x_positions.append(current_x)
                    current_x += panel_width
                    if col < columns - 1:
                        current_x += internal_padding_value

                current_y = 0.0
                for row in range(rows):
                    y_positions.append(current_y)
                    current_y += panel_height
                    if row < rows - 1:
                        current_y += internal_padding_value

                for row in range(rows):
                    for col in range(columns):
                        left = x_positions[col]
                        top = y_positions[row]
                        right = left + panel_width
                        bottom = top + panel_height

                        panels.append(PanelShape(polygon=[
                            (left, top), (right, top),
                            (right, bottom), (left, bottom)
                        ]))

            elif first_char == "H":
                # Horizontal layout
                rows = max(len(template) - 1, 1)
                panel_height = compute_panel_span(
                    content_height, rows, internal_padding_value)

                current_y = 0.0
                for row in range(rows):
                    columns = safe_digit(template[row + 1], default=1)
                    panel_width = compute_panel_span(
                        content_width, columns, internal_padding_value)

                    current_x = 0.0
                    top = current_y
                    bottom = top + panel_height

                    for col in range(columns):
                        left = current_x
                        right = left + panel_width

                        panels.append(PanelShape(polygon=[
                            (left, top), (right, top),
                            (right, bottom), (left, bottom)
                        ]))

                        current_x += panel_width
                        if col < columns - 1:
                            current_x += internal_padding_value

                    current_y += panel_height
                    if row < rows - 1:
                        current_y += internal_padding_value

            elif first_char == "V":
                # Vertical layout
                columns = max(len(template) - 1, 1)
                panel_width = compute_panel_span(
                    content_width, columns, internal_padding_value)

                current_x = 0.0
                for col in range(columns):
                    rows = safe_digit(template[col + 1], default=1)
                    panel_height = compute_panel_span(
                        content_height, rows, internal_padding_value)

                    current_y = 0.0
                    left = current_x
                    right = left + panel_width

                    for row in range(rows):
                        top = current_y
                        bottom = top + panel_height

                        panels.append(PanelShape(polygon=[
                            (left, top), (right, top),
                            (right, bottom), (left, bottom)
                        ]))

                        current_y += panel_height
                        if row < rows - 1:
                            current_y += internal_padding_value

                    current_x += panel_width
                    if col < columns - 1:
                        current_x += internal_padding_value

        else:
            # Diagonal layouts
            if first_char == "H":
                if use_enhanced:
                    row_counts, row_diagonals, row_verticals = parse_enhanced_layout_sequence(
                        template[1:], orientation="horizontal")
                else:
                    row_counts, old_diagonals = parse_layout_sequence(
                        template[1:])
                    row_diagonals = [DiagonalInfo(
                        horizontal=diag) for diag in old_diagonals]
                    row_verticals = [DiagonalInfo() for _ in row_counts]

                panel_height = compute_panel_span(
                    content_height, len(row_counts), internal_padding_value)

                # Starting positions
                top_left = 0.0
                top_right = 0.0

                for index, count in enumerate(row_counts):
                    panel_width = compute_panel_span(
                        content_width, count, internal_padding_value)
                    diagonal_info = row_diagonals[index] if index < len(
                        row_diagonals) else DiagonalInfo()
                    vertical_info = row_verticals[index] if index < len(
                        row_verticals) else DiagonalInfo()

                    # Calculate bottom positions with offsets
                    bottom_left = top_left + panel_height
                    bottom_right = top_right + panel_height

                    if diagonal_info.horizontal:
                        # Apply diagonal offset with new parameters
                        base_offset = content_height * diagonal_info.angle
                        adjusted_offset = base_offset + height_offset
                        bottom_right = clamp(
                            bottom_right + adjusted_offset, 0.0, float(content_height))

                    # Calculate column boundaries
                    total_panel_width = panel_width * count
                    base_boundaries: List[float] = [
                        panel_width * boundary_index for boundary_index in range(count + 1)
                    ]

                    top_boundaries = base_boundaries.copy()
                    bottom_boundaries = base_boundaries.copy()
                    max_base_coordinate = max(total_panel_width, 0.0)

                    # Apply vertical diagonal offsets
                    if vertical_info.vertical and count > 1:
                        offset = panel_height * vertical_info.angle + horizontal_offset
                        for boundary_index in range(1, len(base_boundaries) - 1):
                            top_boundaries[boundary_index] = clamp(
                                base_boundaries[boundary_index] - offset/2,
                                0.0, max_base_coordinate
                            )
                            bottom_boundaries[boundary_index] = clamp(
                                base_boundaries[boundary_index] + offset/2,
                                0.0, max_base_coordinate
                            )

                    # Create panels for this row
                    for col in range(count):
                        padding_offset = internal_padding_value * col
                        left_top_x = clamp(
                            top_boundaries[col] + padding_offset,
                            0.0,
                            float(content_width),
                        )
                        right_top_x = clamp(
                            top_boundaries[col + 1] + padding_offset,
                            0.0,
                            float(content_width),
                        )
                        left_bottom_x = clamp(
                            bottom_boundaries[col] + padding_offset,
                            0.0,
                            float(content_width),
                        )
                        right_bottom_x = clamp(
                            bottom_boundaries[col + 1] + padding_offset,
                            0.0,
                            float(content_width),
                        )

                        top_left_y = interpolate(
                            top_left, top_right, left_top_x, content_width)
                        top_right_y = interpolate(
                            top_left, top_right, right_top_x, content_width)
                        bottom_left_y = interpolate(
                            bottom_left, bottom_right, left_bottom_x, content_width)
                        bottom_right_y = interpolate(
                            bottom_left, bottom_right, right_bottom_x, content_width)

                        panels.append(PanelShape(polygon=[
                            (left_top_x, top_left_y),
                            (right_top_x, top_right_y),
                            (right_bottom_x, bottom_right_y),
                            (left_bottom_x, bottom_left_y),
                        ]))
                    # Update for next row
                    if index < len(row_counts) - 1:
                        top_left = bottom_left + internal_padding_value
                        top_right = bottom_right + internal_padding_value
                    else:
                        top_left = bottom_left
                        top_right = bottom_right

            elif first_char == "V":
                if use_enhanced:
                    column_counts, column_diagonals, column_verticals = parse_enhanced_layout_sequence(
                        template[1:], orientation="vertical")
                else:
                    column_counts, old_diagonals = parse_layout_sequence(
                        template[1:])
                    column_diagonals = [DiagonalInfo(
                        horizontal=diag) for diag in old_diagonals]
                    column_verticals = [DiagonalInfo() for _ in column_counts]

                panel_width = compute_panel_span(
                    content_width, len(column_counts), internal_padding_value)

                # Starting positions - vertical equivalent to horizontal's top_left/top_right
                left_top = 0.0
                left_bottom = 0.0

                for index, count in enumerate(column_counts):
                    diagonal_info = column_diagonals[index] if index < len(
                        column_diagonals) else DiagonalInfo()
                    vertical_info = column_verticals[index] if index < len(
                        column_verticals) else DiagonalInfo()

                    # Calculate right positions - equivalent to bottom in horizontal
                    right_top = clamp(left_top + panel_width,
                                      0.0, float(content_width))
                    right_bottom = clamp(
                        left_bottom + panel_width, 0.0, float(content_width))

                    # Apply diagonal between columns (equivalent to diagonal between rows in H)
                    if diagonal_info.horizontal:
                        base_offset = content_width * diagonal_info.angle
                        adjusted_offset = base_offset + horizontal_offset
                        right_bottom = clamp(
                            right_bottom + adjusted_offset, 0.0, float(content_width))

                    panel_height = compute_panel_span(
                        content_height, count, internal_padding_value)

                    # Calculate row boundaries within this column
                    base_boundaries: List[float] = [
                        panel_height * boundary_index for boundary_index in range(count + 1)
                    ]

                    left_boundaries = base_boundaries.copy()
                    right_boundaries = base_boundaries.copy()
                    max_base_coordinate = panel_height * count

                    # Apply horizontal diagonal within column (equivalent to vertical diagonal within row in H)
                    if vertical_info.vertical and count > 1:
                        offset = panel_width * vertical_info.angle + height_offset
                        for boundary_index in range(1, len(base_boundaries) - 1):
                            left_boundaries[boundary_index] = clamp(
                                base_boundaries[boundary_index] - offset / 2,
                                0.0,
                                max_base_coordinate,
                            )
                            right_boundaries[boundary_index] = clamp(
                                base_boundaries[boundary_index] + offset / 2,
                                0.0,
                                max_base_coordinate,
                            )

                    # Create panels for this column
                    for row in range(count):
                        padding_offset = internal_padding_value * row
                        left_top_y = clamp(
                            left_boundaries[row] + padding_offset, 0.0, float(content_height))
                        left_bottom_y = clamp(
                            left_boundaries[row + 1] + padding_offset, 0.0, float(content_height))
                        right_top_y = clamp(
                            right_boundaries[row] + padding_offset, 0.0, float(content_height))
                        right_bottom_y = clamp(
                            right_boundaries[row + 1] + padding_offset, 0.0, float(content_height))

                        # Interpolate X coordinates based on diagonal between columns
                        left_top_x = clamp(
                            interpolate(left_top, left_bottom,
                                        left_top_y, content_height),
                            0.0,
                            float(content_width),
                        )
                        left_bottom_x = clamp(
                            interpolate(left_top, left_bottom,
                                        left_bottom_y, content_height),
                            0.0,
                            float(content_width),
                        )
                        right_top_x = clamp(
                            interpolate(right_top, right_bottom,
                                        right_top_y, content_height),
                            0.0,
                            float(content_width),
                        )
                        right_bottom_x = clamp(
                            interpolate(right_top, right_bottom,
                                        right_bottom_y, content_height),
                            0.0,
                            float(content_width),
                        )

                        panels.append(PanelShape(polygon=[
                            (left_top_x, left_top_y),
                            (right_top_x, right_top_y),
                            (right_bottom_x, right_bottom_y),
                            (left_bottom_x, left_bottom_y),
                        ]))

                    # Update for next column
                    if index < len(column_counts) - 1:
                        left_top = clamp(
                            right_top + internal_padding_value, 0.0, float(content_width))
                        left_bottom = clamp(
                            right_bottom + internal_padding_value, 0.0, float(content_width))
                    else:
                        left_top = right_top
                        left_bottom = right_bottom

        # Mirror if right to left
        if reading_direction == "right to left":
            panels = [PanelShape(polygon=mirror_polygon_horizontally(
                panel.polygon, content_width)) for panel in panels]

        # Render panels
        for panel in panels:
            bounds = polygon_bounds(panel.polygon)
            min_x, min_y, max_x, max_y = bounds
            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)

            panel_image, image_index = build_panel_image(
                width, height, panel_rgb, outline_rgb, background_rgb,
                0, 0, pil_images, image_index
            )

            paste_panel_polygon(page, panel_image, panel.polygon, bounds)

        draw_polygon_borders(
            page, [panel.polygon for panel in panels], outline_thickness, outline_rgb)
        # Add external padding
        if external_padding > 0:
            page = ImageOps.expand(page, external_padding, background_rgb)

        show_help = (
            "MangaPanelizer: Create manga panel layouts. Use 'internal_padding' for spacing between panels. "
            "Use 'division_height_offset' to control diagonal height and 'division_horizontal_offset' for horizontal adjustment. "
            "Custom layouts support: H123 (horizontal), V123 (vertical), / for diagonals, | for vertical splits, :angle for custom angles."
        )

        return (pil2tensor(page), show_help)
