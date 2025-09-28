"""Node definitions for MangaPanelizer."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from PIL import Image, ImageOps

from .categories import icons
from .config import COLORS, color_mapping
from .core.colors import get_color_values
from .core.geometry import (
    clamp,
    compute_panel_span,
    interpolate,
    mirror_polygon_horizontally,
    polygon_bounds,
)
from .core.imaging import pil2tensor, tensor2pil
from .core.parsing import (
    parse_enhanced_layout_sequence,
    parse_layout_sequence,
    safe_digit,
    safe_int,
)
from .core.rendering import (
    build_panel_image,
    draw_polygon_borders,
    paste_panel_polygon,
)
from .core.types import DiagonalInfo, PanelShape


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
                "custom_panel_layout": ("STRING", {"multiline": True, "default": "H12", "forceInput": True}),
                "internal_padding": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "first_division_angle": ("INT", {"default": 0, "min": -30, "max": 30, "label": "diagonal_angle_adjust"}),
                "second_division_angle": ("INT", {"default": 0, "min": -30, "max": 30, "label": "diagonal_slant_offset"}),
                "division_position": ("INT", {"default": 0, "min": -100, "max": 100, "label": "Division Position"}),
                "second_division_position": (
                    "INT",
                    {"default": 0, "min": -100, "max": 100, "label": "Secondary Division Position"},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
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
        first_division_angle: Optional[int] = None,
        second_division_angle: Optional[int] = None,
        division_position: Optional[int] = None,
        second_division_position: Optional[int] = None,
    ):
        pil_images: List[Image.Image] = []
        if images is not None:
            pil_images = [tensor2pil(image) for image in images]

        outline_rgb = get_color_values(outline_color, color_mapping)
        panel_rgb = get_color_values(panel_color, color_mapping)
        background_rgb = get_color_values(background_color, color_mapping)

        external_padding = max(border_thickness, 0)
        internal_padding_value = max(internal_padding if internal_padding is not None else 0, 0)

        horizontal_offset = second_division_angle if second_division_angle is not None else 0
        angle_adjust_value = float(first_division_angle if first_division_angle is not None else 0)
        angle_ratio = max(min(angle_adjust_value / 30.0, 1.0), -1.0)
        division_offset_value = float(division_position if division_position is not None else 0)
        division_offset_ratio = max(min(division_offset_value / 100.0, 1.0), -1.0)
        secondary_division_value = float(second_division_position if second_division_position is not None else 0)
        secondary_division_ratio = max(min(secondary_division_value / 100.0, 1.0), -1.0)

        def adjust_division_line(base_value: float, min_value: float, max_value: float) -> float:
            """Adjust a division line toward its limits based on the slider ratio."""

            bounded_min = min(min_value, max_value)
            bounded_max = max(bounded_min, max_value)
            if bounded_max <= bounded_min:
                return clamp(base_value, bounded_min, bounded_max)

            base = clamp(base_value, bounded_min, bounded_max)
            if division_offset_ratio >= 0.0:
                return base + (bounded_max - base) * division_offset_ratio
            return base + (base - bounded_min) * division_offset_ratio

        content_width = max(page_width - (2 * external_padding), 1)
        content_height = max(page_height - (2 * external_padding), 1)
        page = Image.new("RGB", (content_width, content_height), background_rgb)

        if template == "custom":
            if custom_panel_layout and custom_panel_layout.strip():
                template = custom_panel_layout.strip()
            else:
                template = "H12"

        template = template.replace("|", "").replace("*", "")

        first_char = template[0].upper()
        image_index = 0
        panels: List[PanelShape] = []

        use_enhanced = first_char in ("H", "V")
        use_diagonal = use_enhanced

        if not use_diagonal:
            if first_char == "G":
                rows = safe_digit(template[1], default=1)
                columns = safe_digit(template[2], default=1)

                panel_width = compute_panel_span(content_width, columns, internal_padding_value)
                panel_height = compute_panel_span(content_height, rows, internal_padding_value)

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
                            (left, top),
                            (right, top),
                            (right, bottom),
                            (left, bottom),
                        ]))

            elif first_char == "H":
                rows = max(len(template) - 1, 1)
                panel_height = compute_panel_span(content_height, rows, internal_padding_value)
                two_row_divider: Optional[float] = None
                if rows == 2:
                    min_divider = clamp(1.0, 0.0, float(content_height))
                    max_divider = clamp(
                        float(content_height) - internal_padding_value - 1.0,
                        min_divider,
                        float(content_height),
                    )
                    base_divider = clamp(panel_height, min_divider, max_divider)
                    two_row_divider = adjust_division_line(base_divider, min_divider, max_divider)

                current_y = 0.0
                for row in range(rows):
                    columns = safe_digit(template[row + 1], default=1)
                    panel_width = compute_panel_span(content_width, columns, internal_padding_value)

                    current_x = 0.0
                    top = current_y
                    if rows == 2:
                        if row == 0:
                            bottom = two_row_divider if two_row_divider is not None else top + panel_height
                        else:
                            bottom = float(content_height)
                    else:
                        bottom = top + panel_height
                    min_bottom = clamp(top + 1.0, 0.0, float(content_height))
                    bottom = clamp(bottom, min_bottom, float(content_height))
                    current_height = max(bottom - top, 1.0)

                    for col in range(columns):
                        left = current_x
                        right = left + panel_width

                        panels.append(PanelShape(polygon=[
                            (left, top),
                            (right, top),
                            (right, bottom),
                            (left, bottom),
                        ]))

                        current_x += panel_width
                        if col < columns - 1:
                            current_x += internal_padding_value

                    current_y = top + current_height
                    if row < rows - 1:
                        current_y += internal_padding_value

            elif first_char == "V":
                columns = max(len(template) - 1, 1)
                panel_width = compute_panel_span(content_width, columns, internal_padding_value)
                two_column_divider: Optional[float] = None
                if columns == 2:
                    min_divider = clamp(1.0, 0.0, float(content_width))
                    max_divider = clamp(
                        float(content_width) - internal_padding_value - 1.0,
                        min_divider,
                        float(content_width),
                    )
                    base_divider = clamp(panel_width, min_divider, max_divider)
                    two_column_divider = adjust_division_line(base_divider, min_divider, max_divider)

                current_x = 0.0
                for col in range(columns):
                    rows = safe_digit(template[col + 1], default=1)
                    panel_height = compute_panel_span(content_height, rows, internal_padding_value)

                    current_y = 0.0
                    left = current_x
                    if columns == 2:
                        if col == 0:
                            right = two_column_divider if two_column_divider is not None else left + panel_width
                        else:
                            right = float(content_width)
                    else:
                        right = left + panel_width
                    min_right = clamp(left + 1.0, 0.0, float(content_width))
                    right = clamp(right, min_right, float(content_width))
                    current_width = max(right - left, 1.0)

                    for row in range(rows):
                        top = current_y
                        bottom = top + panel_height

                        panels.append(PanelShape(polygon=[
                            (left, top),
                            (right, top),
                            (right, bottom),
                            (left, bottom),
                        ]))

                        current_y += panel_height
                        if row < rows - 1:
                            current_y += internal_padding_value

                    current_x = left + current_width
                    if col < columns - 1:
                        current_x += internal_padding_value

        else:
            if first_char == "H":
                if use_enhanced:
                    row_counts, row_diagonals, row_verticals = parse_enhanced_layout_sequence(
                        template[1:], orientation="horizontal")
                else:
                    row_counts, old_diagonals = parse_layout_sequence(template[1:])
                    row_diagonals = [DiagonalInfo(horizontal=diag) for diag in old_diagonals]
                    row_verticals = [DiagonalInfo() for _ in row_counts]

                panel_height = compute_panel_span(content_height, len(row_counts), internal_padding_value)
                top_left = 0.0
                top_right = 0.0

                for index, count in enumerate(row_counts):
                    panel_width = compute_panel_span(content_width, count, internal_padding_value)
                    diagonal_info = row_diagonals[index] if index < len(row_diagonals) else DiagonalInfo()
                    vertical_info = row_verticals[index] if index < len(row_verticals) else DiagonalInfo()

                    bottom_left = clamp(top_left + panel_height, 0.0, float(content_height))
                    max_bottom = float(content_height) if index >= len(row_counts) - 1 else max(float(content_height) - internal_padding_value, 0.0)
                    base_bottom_right = clamp(top_right + panel_height, 0.0, max_bottom)

                    if diagonal_info.horizontal and index < len(row_counts) - 1:
                        max_target = max(float(content_height) - internal_padding_value, 0.0)
                        min_target = 0.0
                        baseline = clamp(top_right + panel_height, min_target, max_target)
                        bottom_right = baseline + panel_height * (angle_ratio)
                        bottom_right = clamp(bottom_right, min_target, max_target)
                    else:
                        bottom_right = clamp(top_right + panel_height, 0.0, float(content_content_y := content_height))

                    if len(row_counts) == 2:
                        if index < len(row_counts) - 1:
                            min_bottom_left = clamp(top_left + 1.0, 0.0, float(content_height))
                            max_bottom_left = clamp(
                                float(content_height) - internal_padding_value - 1.0,
                                min_bottom_left,
                                float(content_height),
                            )
                            bottom_left = adjust_division_line(bottom_left, min_bottom_left, max_bottom_left)

                            min_bottom_right = clamp(top_right + 1.0, 0.0, float(content_height))
                            max_bottom_right = clamp(
                                float(content_height) - internal_padding_value - 1.0,
                                min_bottom_right,
                                float(content_height),
                            )
                            bottom_right = adjust_division_line(bottom_right, min_bottom_right, max_bottom_right)

                        else:
                            bottom_left = float(content_height)
                            bottom_right = float(content_height)

                    total_panel_width = panel_width * count
                    base_boundaries: List[float] = [
                        panel_width * boundary_index for boundary_index in range(count + 1)
                    ]

                    top_boundaries = base_boundaries.copy()
                    bottom_boundaries = base_boundaries.copy()
                    max_base_coordinate = max(total_panel_width, 0.0)

                    if count > 1:
                        base_span = max(panel_height - internal_padding_value, 0.0)
                        offset_ratio = max(min(horizontal_offset / 30.0, 1.0), -1.0)
                        offset = base_span * offset_ratio
                        if offset != 0.0:
                            interior_indices = range(1, len(base_boundaries) - 1)
                            if interior_indices:
                                max_offsets = []
                                for boundary_index in interior_indices:
                                    base_position = base_boundaries[boundary_index]
                                    margin = min(base_position, max_base_coordinate - base_position)
                                    max_offsets.append(max(0.0, margin * 2.0))
                                max_offset_allowed = min(max_offsets) if max_offsets else abs(offset)
                            else:
                                max_offset_allowed = abs(offset)
                            if max_offset_allowed > 0.0:
                                offset = clamp(offset, -max_offset_allowed, max_offset_allowed)
                            for boundary_index in interior_indices:
                                top_boundaries[boundary_index] = clamp(
                                    base_boundaries[boundary_index] - offset / 2,
                                    0.0,
                                    max_base_coordinate,
                                )
                                bottom_boundaries[boundary_index] = clamp(
                                    base_boundaries[boundary_index] + offset / 2,
                                    0.0,
                                    max_base_coordinate,
                                )

                    if count == 2 and len(top_boundaries) >= 3:
                        def shift_interior_boundary(boundaries: List[float]) -> None:
                            min_limit = clamp(boundaries[0] + 1.0, boundaries[0], boundaries[-1])
                            max_limit = clamp(boundaries[-1] - 1.0, min_limit, boundaries[-1])
                            base_value = clamp(boundaries[1], min_limit, max_limit)
                            if secondary_division_ratio >= 0.0:
                                adjusted = base_value + (max_limit - base_value) * secondary_division_ratio
                            else:
                                adjusted = base_value + (base_value - min_limit) * secondary_division_ratio
                            boundaries[1] = clamp(adjusted, min_limit, max_limit)

                        shift_interior_boundary(top_boundaries)
                        shift_interior_boundary(bottom_boundaries)

                    for col in range(count):
                        padding_offset = internal_padding_value * col
                        left_top_x = clamp(top_boundaries[col] + padding_offset, 0.0, float(content_width))
                        right_top_x = clamp(top_boundaries[col + 1] + padding_offset, 0.0, float(content_width))
                        left_bottom_x = clamp(bottom_boundaries[col] + padding_offset, 0.0, float(content_width))
                        right_bottom_x = clamp(bottom_boundaries[col + 1] + padding_offset, 0.0, float(content_width))

                        top_left_y = interpolate(top_left, top_right, left_top_x, content_width)
                        top_right_y = interpolate(top_left, top_right, right_top_x, content_width)
                        bottom_left_y = interpolate(bottom_left, bottom_right, left_bottom_x, content_width)
                        bottom_right_y = interpolate(bottom_left, bottom_right, right_bottom_x, content_width)

                        panels.append(PanelShape(polygon=[
                            (left_top_x, top_left_y),
                            (right_top_x, top_right_y),
                            (right_bottom_x, bottom_right_y),
                            (left_bottom_x, bottom_left_y),
                        ]))

                    if index < len(row_counts) - 1:
                        top_left = clamp(bottom_left + internal_padding_value, 0.0, float(content_height))
                        top_right = clamp(bottom_right + internal_padding_value, 0.0, float(content_height))
                    else:
                        top_left = bottom_left
                        top_right = bottom_right

            elif first_char == "V":
                if use_enhanced:
                    column_counts, column_diagonals, column_verticals = parse_enhanced_layout_sequence(
                        template[1:], orientation="vertical")
                else:
                    column_counts, old_diagonals = parse_layout_sequence(template[1:])
                    column_diagonals = [DiagonalInfo(horizontal=diag) for diag in old_diagonals]
                    column_verticals = [DiagonalInfo() for _ in column_counts]

                panel_width = compute_panel_span(content_width, len(column_counts), internal_padding_value)
                left_top = 0.0
                left_bottom = 0.0

                for index, count in enumerate(column_counts):
                    diagonal_info = column_diagonals[index] if index < len(column_diagonals) else DiagonalInfo()
                    vertical_info = column_verticals[index] if index < len(column_verticals) else DiagonalInfo()

                    right_top = clamp(left_top + panel_width, 0.0, float(content_width))
                    max_right = float(content_width) if index >= len(column_counts) - 1 else max(float(content_width) - internal_padding_value, 0.0)
                    base_right_bottom = clamp(left_bottom + panel_width, 0.0, max_right)

                    if diagonal_info.horizontal and index < len(column_counts) - 1:
                        max_target = max(float(content_width) - internal_padding_value, 0.0)
                        min_target = 0.0
                        baseline = clamp(base_right_bottom, min_target, max_target)
                        if angle_ratio > 0:
                            right_bottom = baseline + (max_target - baseline) * angle_ratio
                        elif angle_ratio < 0:
                            right_bottom = baseline + (baseline - min_target) * angle_ratio
                        else:
                            right_bottom = baseline
                        right_bottom = clamp(right_bottom, min_target, max_target)
                    else:
                        right_bottom = base_right_bottom

                    if index >= len(column_counts) - 1:
                        right_bottom = clamp(right_top, 0.0, float(content_width))

                    if len(column_counts) == 2:
                        if index < len(column_counts) - 1:
                            min_right_top = clamp(left_top + 1.0, 0.0, float(content_width))
                            max_right_top = clamp(
                                float(content_width) - internal_padding_value - 1.0,
                                min_right_top,
                                float(content_width),
                            )
                            right_top = adjust_division_line(right_top, min_right_top, max_right_top)

                            min_right_bottom = clamp(left_bottom + 1.0, 0.0, float(content_width))
                            max_right_bottom = clamp(
                                float(content_width) - internal_padding_value - 1.0,
                                min_right_bottom,
                                float(content_width),
                            )
                            right_bottom = adjust_division_line(right_bottom, min_right_bottom, max_right_bottom)
                        else:
                            right_top = float(content_width)
                            right_bottom = float(content_width)

                    panel_height = compute_panel_span(content_height, count, internal_padding_value)
                    base_boundaries: List[float] = [
                        panel_height * boundary_index for boundary_index in range(count + 1)
                    ]

                    left_boundaries = base_boundaries.copy()
                    right_boundaries = base_boundaries.copy()
                    max_base_coordinate = panel_height * count

                    if count > 1:
                        base_span = max(panel_width - internal_padding_value, 0.0)
                        offset_ratio = max(min(horizontal_offset / 30.0, 1.0), -1.0)
                        offset = base_span * offset_ratio
                        if offset != 0.0:
                            interior_indices = range(1, len(base_boundaries) - 1)
                            if interior_indices:
                                max_offsets = []
                                for boundary_index in interior_indices:
                                    base_position = base_boundaries[boundary_index]
                                    margin = min(base_position, max_base_coordinate - base_position)
                                    max_offsets.append(max(0.0, margin * 2.0))
                                max_offset_allowed = min(max_offsets) if max_offsets else abs(offset)
                            else:
                                max_offset_allowed = abs(offset)
                            if max_offset_allowed > 0.0:
                                offset = clamp(offset, -max_offset_allowed, max_offset_allowed)
                            for boundary_index in interior_indices:
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

                    if count == 2 and len(left_boundaries) >= 3:
                        def shift_interior_boundary(boundaries: List[float]) -> None:
                            min_limit = clamp(boundaries[0] + 1.0, boundaries[0], boundaries[-1])
                            max_limit = clamp(boundaries[-1] - 1.0, min_limit, boundaries[-1])
                            base_value = clamp(boundaries[1], min_limit, max_limit)
                            if secondary_division_ratio >= 0.0:
                                adjusted = base_value + (max_limit - base_value) * secondary_division_ratio
                            else:
                                adjusted = base_value + (base_value - min_limit) * secondary_division_ratio
                            boundaries[1] = clamp(adjusted, min_limit, max_limit)

                        shift_interior_boundary(left_boundaries)
                        shift_interior_boundary(right_boundaries)

                    for row in range(count):
                        padding_offset = internal_padding_value * row
                        left_top_y = clamp(left_boundaries[row] + padding_offset, 0.0, float(content_height))
                        left_bottom_y = clamp(left_boundaries[row + 1] + padding_offset, 0.0, float(content_height))
                        right_top_y = clamp(right_boundaries[row] + padding_offset, 0.0, float(content_height))
                        right_bottom_y = clamp(right_boundaries[row + 1] + padding_offset, 0.0, float(content_height))

                        left_top_x = clamp(interpolate(left_top, left_bottom, left_top_y, content_height), 0.0, float(content_width))
                        left_bottom_x = clamp(interpolate(left_top, left_bottom, left_bottom_y, content_height), 0.0, float(content_width))
                        right_top_x = clamp(interpolate(right_top, right_bottom, right_top_y, content_height), 0.0, float(content_width))
                        right_bottom_x = clamp(interpolate(right_top, right_bottom, right_bottom_y, content_height), 0.0, float(content_width))

                        panels.append(PanelShape(polygon=[
                            (left_top_x, left_top_y),
                            (right_top_x, right_top_y),
                            (right_bottom_x, right_bottom_y),
                            (left_bottom_x, left_bottom_y),
                        ]))

                    if index < len(column_counts) - 1:
                        left_top = clamp(right_top + internal_padding_value, 0.0, float(content_width))
                        left_bottom = clamp(right_bottom + internal_padding_value, 0.0, max(float(content_width) - internal_padding_value, 0.0))
                    else:
                        left_top = right_top
                        left_bottom = right_bottom

        if reading_direction == "right to left":
            panels = [PanelShape(polygon=mirror_polygon_horizontally(panel.polygon, content_width)) for panel in panels]

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
                0,
                0,
                pil_images,
                image_index,
            )
            paste_panel_polygon(page, panel_image, panel.polygon, (min_x, min_y, max_x, max_y))

        draw_polygon_borders(page, [panel.polygon for panel in panels], outline_thickness, outline_rgb)

        if external_padding > 0:
            page = ImageOps.expand(page, external_padding, background_rgb)

        show_help = (
            "MangaPanelizer: Create manga panel layouts. Use 'internal_padding' for spacing between panels. "
            "Use 'diagonal_angle_adjust' (first_division_angle) para definir el angulo de las divisiones y 'diagonal_slant_offset' (second_division_angle) para desplazarlas. "
            "Ajusta 'Division Position' para mover divisiones principales en layouts 2x y 'Secondary Division Position' para desplazar divisiones internas (H12/V12). "
            "Custom layouts support: H123 (horizontal) y V123 (vertical). Aplica ':angle' de forma opcional para establecer un angulo base si lo necesitas."
        )

        return (pil2tensor(page), show_help)




