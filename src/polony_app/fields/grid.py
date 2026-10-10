"""Counting grid geometry (canonical pixel coordinates; pixel centre = integer)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def polygon(self) -> list[list[float]]:
        return [[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]]


@dataclass(frozen=True)
class Grid:
    xs: tuple[float, ...]      # vertical line positions (canonical), sorted
    ys: tuple[float, ...]

    @property
    def cols(self) -> int:
        return len(self.xs) - 1

    @property
    def rows(self) -> int:
        return len(self.ys) - 1

    def shifted(self, dx: float, dy: float) -> "Grid":
        return Grid(tuple(x + dx for x in self.xs), tuple(y + dy for y in self.ys))

    def cells(self) -> list[Cell]:
        """Row-major (row 0 first, left to right)."""
        return [Cell(r, c, self.xs[c], self.ys[r], self.xs[c + 1], self.ys[r + 1])
                for r in range(self.rows) for c in range(self.cols)]

    def cell_index(self, x: float, y: float) -> Optional[int]:
        """Row-major index of the cell containing (x, y); None if outside the grid."""
        if not (self.xs[0] <= x < self.xs[-1] and self.ys[0] <= y < self.ys[-1]):
            return None
        c = int(np.searchsorted(self.xs, x, side="right")) - 1
        r = int(np.searchsorted(self.ys, y, side="right")) - 1
        return r * self.cols + c


def number_cells(n_cells: int, mapped: dict[int, int]) -> list[int]:
    """Field number of every cell (row-major). `mapped` = {cell index: field} from reference points;
    the remaining cells take the lowest unused positive numbers in row-major order."""
    used = set(mapped.values())
    out: list[int] = []
    nxt = 1
    for i in range(n_cells):
        if i in mapped:
            out.append(mapped[i])
            continue
        while nxt in used:
            nxt += 1
        out.append(nxt)
        used.add(nxt)
    return out


def clip_rect(cell: Cell, width: int, height: int) -> Optional[list[list[float]]]:
    """Cell polygon clipped to the image (pixel centres 0..W-1); None if nothing is left."""
    x0, y0 = max(cell.x0, -0.5), max(cell.y0, -0.5)
    x1, y1 = min(cell.x1, width - 0.5), min(cell.y1, height - 0.5)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2:
        return None
    x, y = np.asarray(a, float), np.asarray(b, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
