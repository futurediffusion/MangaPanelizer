"""Colour helpers for MangaPanelizer."""

from __future__ import annotations

from typing import Dict, Tuple


def get_color_values(color: str, mapping: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Resolve a colour preset name from the shared mapping."""
    return mapping.get(color, (0, 0, 0))
