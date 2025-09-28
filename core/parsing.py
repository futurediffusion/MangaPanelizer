"""Parsing helpers for MangaPanelizer layout strings."""

from __future__ import annotations

from typing import List, Tuple

from .types import DiagonalInfo


def safe_digit(value: str, default: int = 1) -> int:
    """Convert a single character into a positive integer."""
    try:
        parsed = int(value)
        return max(parsed, 1)
    except (TypeError, ValueError):
        return default


def safe_int(value: str, default: int = 1) -> int:
    """Parse an integer string, falling back to a default if needed."""
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def parse_layout_sequence(sequence: str) -> Tuple[List[int], List[bool]]:
    """Parse layout codes (e.g. "123") into counts and diagonal flags."""
    digits = [safe_int(char) for char in sequence if char.isdigit()]
    if not digits:
        digits = [1]
    diagonals = [True] * max(len(digits) - 1, 0)
    return digits, diagonals


def parse_enhanced_layout_sequence(sequence: str, orientation: str = "horizontal") -> Tuple[List[int], List[DiagonalInfo], List[DiagonalInfo]]:
    """Parse layout strings into counts and diagonal metadata."""
    orientation = orientation.lower()

    angle_ratio = 0.0
    layout_part = sequence
    if ":" in layout_part:
        layout_part, angle_part = layout_part.rsplit(":", 1)
        try:
            angle_degrees = float(angle_part)
            angle_ratio = min(max(angle_degrees / 90.0, 0.0), 1.0)
        except ValueError:
            angle_ratio = 0.0

    digits = [int(char) for char in layout_part if char.isdigit()]
    if not digits:
        digits = [1]

    counts = digits
    between_diagonals: List[DiagonalInfo] = []
    within_diagonals: List[DiagonalInfo] = []

    for index, count in enumerate(counts):
        if index < len(counts) - 1:
            info = DiagonalInfo(angle=angle_ratio)
            info.horizontal = True
            between_diagonals.append(info)

        vertical_info = DiagonalInfo(angle=angle_ratio)
        vertical_info.vertical = count > 1
        within_diagonals.append(vertical_info)

    return counts, between_diagonals, within_diagonals
