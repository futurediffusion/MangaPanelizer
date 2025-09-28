"""Common data structures for MangaPanelizer."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PanelShape:
    """Polygon describing a panel region."""
    polygon: List[Tuple[float, float]]


@dataclass
class DiagonalInfo:
    """Information about diagonal separators for layouts."""
    horizontal: bool = False  # Between row/column groups
    vertical: bool = False    # Within a group (columns in H, rows in V)
    angle: float = 0.0        # Offset ratio relative to page span (0.0-1.0)
