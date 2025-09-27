"""Rendering helpers for MangaPanelizer."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

from .imaging import crop_and_resize_image


def build_panel_image(
    panel_width: int,
    panel_height: int,
    panel_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
    external_padding: int,
    outline_thickness: int,
    images: List[Image.Image],
    image_index: int,
) -> Tuple[Image.Image, int]:
    """Create a rectangular panel image with the configured styling."""
    panel = Image.new("RGB", (panel_width, panel_height), panel_color)
    if 0 <= image_index < len(images):
        img = crop_and_resize_image(images[image_index], panel_width, panel_height)
        panel.paste(img, (0, 0))

    if outline_thickness > 0:
        panel = ImageOps.expand(panel, border=outline_thickness, fill=outline_color)
    if external_padding > 0:
        panel = ImageOps.expand(panel, border=external_padding, fill=background_color)

    return panel, image_index + 1


def paste_panel_polygon(
    page: Image.Image,
    panel_image: Image.Image,
    polygon: Sequence[Tuple[float, float]],
    bounds: Tuple[int, int, int, int],
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
    polygons: Sequence[Sequence[Tuple[float, float]]],
    outline_thickness: int,
    outline_color: Tuple[int, int, int],
) -> None:
    """Render outline strokes with uniform thickness and antialiasing."""
    if outline_thickness <= 0 or not polygons:
        return

    edge_lookup: dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    vertex_points: set[Tuple[float, float]] = set()

    for polygon in polygons:
        if not polygon:
            continue
        for index in range(len(polygon)):
            start_point = polygon[index]
            end_point = polygon[(index + 1) % len(polygon)]
            vertex_points.add(start_point)
            start_pixel = (int(round(start_point[0])), int(round(start_point[1])))
            end_pixel = (int(round(end_point[0])), int(round(end_point[1])))
            if start_pixel == end_pixel:
                continue
            if start_pixel <= end_pixel:
                edge_key = (start_pixel, end_pixel)
                edge_value = (start_point, end_point)
            else:
                edge_key = (end_pixel, start_pixel)
                edge_value = (end_point, start_point)
            edge_lookup.setdefault(edge_key, edge_value)

    if not edge_lookup:
        return

    scale_factor = 4
    scaled_size = (page.width * scale_factor, page.height * scale_factor)
    temp_image = Image.new("RGBA", scaled_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_image)
    scaled_width = max(outline_thickness, 1) * scale_factor
    radius = max(1, scaled_width // 2)
    rgba_color = (*outline_color, 255)

    max_x = page.width * scale_factor - 1
    max_y = page.height * scale_factor - 1

    for start_point, end_point in edge_lookup.values():
        sx = max(0, min(int(round(start_point[0] * scale_factor)), max_x))
        sy = max(0, min(int(round(start_point[1] * scale_factor)), max_y))
        ex = max(0, min(int(round(end_point[0] * scale_factor)), max_x))
        ey = max(0, min(int(round(end_point[1] * scale_factor)), max_y))
        draw.line([(sx, sy), (ex, ey)], fill=rgba_color, width=scaled_width)

    for vx, vy in vertex_points:
        cx = max(0, min(int(round(vx * scale_factor)), max_x))
        cy = max(0, min(int(round(vy * scale_factor)), max_y))
        left = max(cx - radius, 0)
        top = max(cy - radius, 0)
        right = min(cx + radius, max_x)
        bottom = min(cy + radius, max_y)
        draw.ellipse((left, top, right, bottom), fill=rgba_color)

    antialiased = temp_image.resize(page.size, Image.Resampling.LANCZOS)
    page.paste(antialiased, (0, 0), antialiased)
