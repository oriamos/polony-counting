"""Counting grid geometry (canonical pixel coordinates; pixel centre = integer)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from polony_app.schemas import GridSpec

MAX_FIELD = 24   # the lab template has columns Field1..Field24


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


# ------------------------------------------------------------------ generated grid (GridSpec)
# Mirrors frontend/src/lib/labgrid.ts (generateGrid, randomShift, mulberry32): same numbering, same polygons, and
# the same seed gives the same shift in both. The server is authoritative.
def spec_cells(spec: "GridSpec") -> list[tuple[int, int, int, list[list[float]]]]:
    """(field_index, row, col, polygon) of every cell, in numbering order. Polygons are at the effective origin
    (origin + shift) and are not clipped to the image."""
    rows, cols = max(0, int(spec.rows)), max(0, int(spec.cols))
    rev_r = spec.corner in ("bl", "br")    # numbering starts at the bottom
    rev_c = spec.corner in ("tr", "br")    # numbering starts at the right
    r_order = [rows - 1 - i if rev_r else i for i in range(rows)]
    c_order = [cols - 1 - i if rev_c else i for i in range(cols)]
    if spec.order == "row-major":
        seq = [(r, c) for r in r_order for c in c_order]
    else:
        seq = [(r, c) for c in c_order for r in r_order]
    ox, oy = spec.origin_x + spec.shift_dx, spec.origin_y + spec.shift_dy
    out = []
    for n, (r, c) in enumerate(seq):
        x0, y0 = ox + c * spec.cell_w, oy + r * spec.cell_h
        x1, y1 = ox + (c + 1) * spec.cell_w, oy + (r + 1) * spec.cell_h
        out.append((spec.first_index + n, r, c, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return out


def spec_problem(spec: "GridSpec") -> Optional[str]:
    """Why a grid spec cannot be applied (None = fine)."""
    nums = (spec.cell_w, spec.cell_h, spec.origin_x, spec.origin_y, spec.shift_dx, spec.shift_dy)
    if not all(math.isfinite(v) for v in nums):
        return "grid sizes and positions must be finite numbers"
    if spec.rows < 1 or spec.cols < 1:
        return "rows and cols must be >= 1"
    if spec.cell_w <= 0 or spec.cell_h <= 0:
        return "cell_w and cell_h must be > 0"
    if spec.first_index < 1:
        return "first_index must be >= 1"
    last = spec.first_index + spec.rows * spec.cols - 1
    if last > MAX_FIELD:
        return f"field numbers {spec.first_index}..{last} exceed {MAX_FIELD} (the lab template ends at field {MAX_FIELD})"
    if spec.seed is not None and not (0 <= spec.seed < 2**32):
        return "seed must be in 0..2^32-1"
    return None


def mulberry32(seed: int):
    """Small seeded PRNG (uniform floats in [0, 1)), bit-identical to the TypeScript one in labgrid.ts."""
    m = 0xFFFFFFFF
    state = seed & m

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & m
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & m
        t = (((t + (((t ^ (t >> 7)) * (t | 61)) & m)) & m) ^ t) & m
        return ((t ^ (t >> 14)) & m) / 4294967296

    return nxt


def _axis_shift(origin: float, n: int, cell: float, size: int, u: float) -> float:
    """Shift along one axis keeping the whole grid on the image ([-0.5, size-0.5], pixel-centre convention).
    Whole pixels, uniform over the slack; the slack is 0 when the grid is larger than the image."""
    lo = -0.5 - origin
    hi = size - 0.5 - n * cell - origin
    if hi < lo:
        return 0.0
    a, b = math.ceil(lo), math.floor(hi)
    if a > b:                 # less than one pixel of slack: put it at the edge
        return float(lo)
    return float(a + min(b - a, int(u * (b - a + 1))))


def random_shift(spec: "GridSpec", width: int, height: int, seed: int) -> tuple[float, float]:
    """(dx, dy) relative to the spec's origin: cell size and count never change, the grid stays inside the image."""
    rng = mulberry32(seed)
    ux, uy = rng(), rng()
    return (_axis_shift(spec.origin_x, spec.cols, spec.cell_w, width, ux),
            _axis_shift(spec.origin_y, spec.rows, spec.cell_h, height, uy))
