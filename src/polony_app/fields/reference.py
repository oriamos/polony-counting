"""The researcher's marks embedded in a TIFF: ImageJ multi-point ROI (counter = field, position = plane)
and the Overlays counting grid. Pure reading; meaning of planes/colours is never assumed."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from polony_app.legacy_import.roi import read_embedded, shape_to_canonical, to_canonical
from polony_app.schemas import ImageKind, ImageMeta

from .grid import Grid, number_cells


@dataclass
class RefPoint:
    x: float
    y: float
    counter: Optional[int]
    plane: Optional[int]            # 0-based page; None = unspecified / beyond the planes: evidence for NO plane
    field: Optional[int]            # counter + 1 ("index" numbering)
    field_ordinal: Optional[int]    # rank of the counter among the counters present + 1


@dataclass
class Reference:
    available: bool
    source_file: Optional[str] = None
    reason: Optional[str] = None
    points: list[RefPoint] = field(default_factory=list)
    grid: Optional[Grid] = None
    grid_regular: bool = True
    warnings: list[str] = field(default_factory=list)

    # ---- derived
    def planes(self) -> list[dict]:
        c = Counter(p.plane for p in self.points if p.plane is not None)
        return [{"page": k, "point_count": c[k]} for k in sorted(c)]

    def unplaned_points(self) -> list[RefPoint]:
        """Marks whose plane is unspecified (ImageJ position missing/0) or beyond the planes of the file."""
        return [p for p in self.points if p.plane is None]

    def unplaned_fields(self, numbering: str = "index") -> list[int]:
        """Field numbers that contain at least one mark without a valid plane."""
        return sorted({f for p in self.unplaned_points() if (f := self.field_of(p, numbering)) is not None})

    def field_of(self, p: RefPoint, numbering: str = "index") -> Optional[int]:
        return p.field_ordinal if numbering == "ordinal" else p.field

    def counts(self, numbering: str = "index", plane: Optional[int] = None) -> dict[int, int]:
        """Marks per field on ``plane``. ``plane=None`` = any plane (layout only, never evidence). A mark without a
        valid plane (``p.plane is None``) is counted for no specific plane."""
        out: dict[int, int] = defaultdict(int)
        for p in self.points:
            f = self.field_of(p, numbering)
            if f is not None and (plane is None or p.plane == plane):
                out[f] += 1
        return dict(out)

    def field_cells(self, grid: Grid, numbering: str = "index", plane: Optional[int] = None
                    ) -> tuple[dict[int, int], list[str]]:
        """{field: row-major cell index} (majority cell of the field's points on `grid`) + warnings."""
        votes: dict[int, Counter] = defaultdict(Counter)
        outside = 0
        for p in self.points:
            f = self.field_of(p, numbering)
            if f is None or (plane is not None and p.plane != plane):
                continue
            ci = grid.cell_index(p.x, p.y)
            if ci is None:
                outside += 1
            else:
                votes[f][ci] += 1
        warns: list[str] = []
        if outside:
            warns.append(f"{outside} reference point(s) lie outside the grid and were not used to place fields.")
        out: dict[int, int] = {}
        for f in sorted(votes):
            ranked = votes[f].most_common()
            out[f] = ranked[0][0]
            if len(ranked) > 1:
                n = sum(votes[f].values())
                warns.append(f"Field {f}: {n - ranked[0][1]} of {n} marks lie in another grid cell "
                             "(the cell with most marks is used).")
        return out, warns

    def cell_field_numbers(self, numbering: str = "index") -> dict[tuple[int, int], int]:
        """{(row, col): field number} of the overlay grid (as drawn in ``to_dict``); {} without a grid."""
        if self.grid is None:
            return {}
        g = self.grid
        cells = g.cells()
        mapped, _ = self.field_cells(g, numbering)
        by_cell: dict[int, int] = {}
        for f in sorted(mapped):
            by_cell.setdefault(mapped[f], f)
        nums = number_cells(len(cells), by_cell)
        return {(c.row, c.col): nums[i] for i, c in enumerate(cells)}

    def to_dict(self) -> dict:
        d: dict = {"available": self.available, "source_file": self.source_file, "points": [],
                   "grid": {"source": "none", "origin": [0.0, 0.0], "cell": [0.0, 0.0], "cols": 0, "rows": 0, "cells": []},
                   "reference_counts": {}, "planes": [], "warnings": list(self.warnings)}
        if self.reason:
            d["reason"] = self.reason
        if not self.available:
            return d
        d["points"] = [{"x": round(p.x, 3), "y": round(p.y, 3), "counter": p.counter, "plane": p.plane,
                        "field": p.field, "field_ordinal": p.field_ordinal} for p in self.points]
        d["planes"] = self.planes()
        pages = sorted({p.plane for p in self.points if p.plane is not None} | {0, 1})
        up = self.unplaned_points()
        d["unplaned"] = {"count": len(up), "fields": self.unplaned_fields("index"),
                         "n_without_counter": sum(1 for p in up if p.field is None)}
        rc: dict[str, dict] = {}
        for f in sorted({p.field for p in self.points if p.field is not None}):
            rc[str(f)] = {f"plane{k}": 0 for k in pages}
            n_up = sum(1 for p in up if p.field == f)
            if n_up:
                rc[str(f)]["unplaned"] = n_up
        for p in self.points:
            if p.field is not None and p.plane is not None:
                rc[str(p.field)][f"plane{p.plane}"] += 1
        d["reference_counts"] = rc
        if self.grid is not None:
            g = self.grid
            cells = g.cells()
            mapped, _ = self.field_cells(g)
            # one number per cell: lowest field wins when two fields share a cell
            by_cell: dict[int, int] = {}
            for f in sorted(mapped):
                by_cell.setdefault(mapped[f], f)
            nums = number_cells(len(cells), by_cell)
            d["grid"] = {
                "source": "overlay", "origin": [g.xs[0], g.ys[0]],
                "cell": [(g.xs[-1] - g.xs[0]) / g.cols, (g.ys[-1] - g.ys[0]) / g.rows],
                "cols": g.cols, "rows": g.rows,
                "cells": [{"field": nums[i], "polygon": c.polygon, "row": c.row, "col": c.col,
                           "mapped": i in by_cell} for i, c in enumerate(cells)],
            }
        return d


def _unavailable(reason: str, source: Optional[str] = None) -> Reference:
    return Reference(False, source, reason, warnings=[reason])


def load_reference(originals_dir: Path, image: Optional[ImageMeta]) -> Reference:
    if image is None:
        return _unavailable("The analysis has no image.")
    if image.kind != ImageKind.raw_two_channel:
        return _unavailable("Embedded researcher marks are read only from a two-page ImageJ stack "
                            f"(this image is '{image.kind.value}').")
    src = image.channels[0].source_file
    path = Path(originals_dir) / src
    if not path.is_file():
        return _unavailable("The original file is not on this computer.", src)
    bundle = read_embedded(path)
    warns = [i.message for i in bundle.issues if i.severity != "info"]
    if not bundle.points and not bundle.grids:
        return _unavailable("The TIFF carries no embedded ImageJ point ROI or counting-grid overlay.", src)
    ref = Reference(True, src, warnings=warns)

    counters = sorted({p.counter for p in bundle.points if p.counter is not None})
    rank = {c: i + 1 for i, c in enumerate(counters)}
    n_planes = max(2, len(image.channels))
    unplaned = 0
    for p in bundle.points:
        # ImageJ position is 1-based; missing/0 ("all planes") or beyond the file's planes = evidence for NO plane
        plane = (p.position - 1) if (p.position and 1 <= p.position <= n_planes) else None
        if plane is None:
            unplaned += 1
        ref.points.append(RefPoint(
            x=to_canonical(p.x, p.subpixel), y=to_canonical(p.y, p.subpixel), counter=p.counter, plane=plane,
            field=(p.counter + 1) if p.counter is not None else None,
            field_ordinal=rank.get(p.counter) if p.counter is not None else None))
    if unplaned:
        ref.warnings.append(f"{unplaned} mark(s) carry no valid plane (slice) number (missing, 0 or beyond the {n_planes} "
                            "planes of the file). They are evidence for NO plane: every field that contains one is a "
                            "conflict on all planes in the comparison, and the training import drops them.")
    if bundle.points and any(p.counter is None for p in bundle.points):
        ref.warnings.append("Some marks carry no ImageJ counter, so their field number is unknown.")
    if counters and counters != list(range(counters[0], counters[-1] + 1)):
        missing = [c for c in range(counters[0], counters[-1] + 1) if c not in rank]
        ref.warnings.append(
            "Counter numbers are not contiguous (missing counter(s) "
            + ", ".join(map(str, missing)) + "). A missing counter is either a field with zero marks or a number "
            "that was skipped; this cannot be told from the file. 'index' numbering (field = counter + 1) keeps the "
            "gap, 'ordinal' numbering closes it - compare with your sheet.")
    if counters and counters[0] != 0:
        ref.warnings.append(f"The first counter is {counters[0]}, not 0.")
    if bundle.grids:
        g = bundle.grids[0]
        if len(bundle.grids) > 1:
            ref.warnings.append(f"{len(bundle.grids)} grid overlays found; the first one is used.")
        xs = [shape_to_canonical([(x, 0.0)])[0][0] for x in g.xs]
        ys = [shape_to_canonical([(0.0, y)])[0][1] for y in g.ys]
        ref.grid = Grid(tuple(xs), tuple(ys))
        ref.grid_regular = g.regular
        if not g.regular:
            ref.warnings.append("The grid spacing is not uniform; cells follow the stored lines.")
    else:
        ref.warnings.append("No counting-grid overlay in the file; fields cannot be placed automatically.")
    return ref
