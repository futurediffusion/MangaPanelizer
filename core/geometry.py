"""Geometric utilities for MangaPanelizer layouts."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def compute_panel_span(total: float, count: int, padding: float) -> float:
    """Determine usable span for panels along one axis with optional padding."""
    if count <= 0:
        return float(max(total, 1))

    effective_total = float(total) - max(count - 1, 0) * float(padding)
    if effective_total <= 0:
        return float(max(total, 1)) / float(max(count, 1))
    return effective_total / float(count)


def interpolate(value_left: float, value_right: float, position: float, span: float) -> float:
    """Linearly interpolate between two values across a span."""
    if span <= 0:
        return value_left
    ratio = min(max(position / span, 0.0), 1.0)
    return value_left + (value_right - value_left) * ratio


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value to an inclusive range."""
    return max(minimum, min(value, maximum))


def polygon_bounds(polygon: Sequence[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """Compute integer bounds for a polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x = int(np.floor(min(xs)))
    min_y = int(np.floor(min(ys)))
    max_x = int(np.ceil(max(xs)))
    max_y = int(np.ceil(max(ys)))
    return min_x, min_y, max_x, max_y


def mirror_polygon_horizontally(polygon: Sequence[Tuple[float, float]], width: float) -> List[Tuple[float, float]]:
    """Mirror a polygon across the vertical axis of the page."""
    return [(width - x, y) for x, y in polygon]
