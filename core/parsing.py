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
    """Parse legacy layout codes (e.g. "1/23") into counts and diagonal flags."""
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


def parse_enhanced_layout_sequence(sequence: str, orientation: str = "horizontal") -> Tuple[List[int], List[DiagonalInfo], List[DiagonalInfo]]:
    """Parse enhanced sequence strings with diagonal flags.

    Parameters
    ----------
    sequence:
        Layout definition after the leading H/V character.
    orientation:
        "horizontal" when describing rows (H layouts) and "vertical" when describing columns (V layouts).
        The meaning of "/" and "|" swaps depending on the orientation.
    """
    orientation = orientation.lower()
    between_char = "/" if orientation != "vertical" else "|"
    within_char = "|" if orientation != "vertical" else "/"

    counts: List[int] = []
    between_diagonals: List[DiagonalInfo] = []  # Between row/column groups
    within_diagonals: List[DiagonalInfo] = []   # Within a row/column

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

    pending_between = False
    pending_within = False

    for char in layout_part:
        if char.isdigit():
            counts.append(int(char))

            if len(counts) > 1 and len(between_diagonals) < len(counts) - 1:
                info = DiagonalInfo(angle=angle_ratio)
                if pending_between:
                    info.horizontal = True
                between_diagonals.append(info)
                pending_between = False

            if pending_within:
                while len(within_diagonals) < len(counts):
                    within_diagonals.append(DiagonalInfo(angle=angle_ratio))
                within_diagonals[len(counts) - 1].vertical = True
                pending_within = False
        elif char == between_char:
            pending_between = True
        elif char == within_char:
            pending_within = True
        else:
            pending_between = False
            pending_within = False

    while len(between_diagonals) < max(len(counts) - 1, 0):
        between_diagonals.append(DiagonalInfo(angle=angle_ratio))

    while len(within_diagonals) < len(counts):
        within_diagonals.append(DiagonalInfo(angle=angle_ratio))

    if not counts:
        counts = [1]

    return counts, between_diagonals, within_diagonals
