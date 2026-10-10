"""Field regions from the grid, and the researcher-vs-app comparison."""
from __future__ import annotations

import re
from typing import Optional

from polony_app.review.counts import compute_counts
from polony_app.review.state import ReviewError
from polony_app.schemas import Analysis, Region, RegionKind

from .grid import clip_rect, number_cells, pearson
from .reference import Reference

MAX_LAB_FIELD = 24
FIELD_NAME = re.compile(r"^(Field|Cell)-\d+$")
COUNT_TYPES = ("red_positive", "green_positive", "red_only", "green_only", "yellow")
AMBIGUITY = ("Fields without marks are ambiguous: a field with no marks may be 'not counted' or 'counted, zero'. "
             "The file cannot tell which; decide per field.")


def plan_fields(ref: Reference, mode: str, shift: tuple[float, float], plane: Optional[int],
                numbering: str, width: int, height: int) -> tuple[list[dict], list[str]]:
    """[{field, name, field_index, polygon, row, col, n_marks}] + warnings. Pure."""
    if ref.grid is None:
        raise ReviewError("no_grid", "This image has no counting-grid overlay to build fields from. "
                          "Draw the field regions manually.", 422)
    grid = ref.grid.shifted(shift[0], shift[1])
    cells = grid.cells()
    warns: list[str] = []
    f2c, w = ref.field_cells(grid, numbering, plane)
    warns += w
    counts = ref.counts(numbering, plane)
    cell_field: dict[int, int] = {}
    for f in sorted(f2c):
        if f2c[f] in cell_field:
            warns.append(f"Fields {cell_field[f2c[f]]} and {f} fall in the same grid cell; only field "
                         f"{cell_field[f2c[f]]} is created.")
            continue
        cell_field[f2c[f]] = f
    if mode == "counted":
        if not cell_field:
            raise ReviewError("no_reference_fields", "No reference marks with a field number fall into the grid"
                              + (f" on plane {plane}" if plane is not None else "") + "; nothing to import.", 422)
        chosen = cell_field
        warns.append(AMBIGUITY)
    else:
        nums = number_cells(len(cells), cell_field)
        chosen = {i: nums[i] for i in range(len(cells))}
        if not cell_field:
            warns.append("No reference marks used: cells are numbered 1..N in reading order (row by row). "
                         "Check the numbering against your sheet.")
    out = []
    for i in sorted(chosen, key=lambda i: chosen[i]):
        f = chosen[i]
        poly = clip_rect(cells[i], width, height)
        if poly is None:
            warns.append(f"Field {f} lies outside the image after the shift and was skipped.")
            continue
        if poly != cells[i].polygon:
            warns.append(f"Field {f} was clipped to the image.")
        lab = f <= MAX_LAB_FIELD
        if not lab:
            warns.append(f"Cell {f} is beyond the lab template's {MAX_LAB_FIELD} fields: it gets no field number "
                         "(named Cell-N) so lab export is not blocked.")
        out.append({"field": f, "name": f"Field-{f}" if lab else f"Cell-{f}", "field_index": f if lab else None,
                    "polygon": poly, "row": cells[i].row, "col": cells[i].col, "n_marks": counts.get(f, 0)})
    return out, warns


def apply_fields(a: Analysis, planned: list[dict], replace: bool, new_id) -> list[Region]:
    """New region list: planned fields first (first listed region wins membership), then the other
    existing regions (e.g. a Chamber). Raises 409 when Field-* regions exist and replace is false."""
    existing = [r for r in a.regions if FIELD_NAME.match(r.name)]
    if existing and not replace:
        raise ReviewError("fields_exist", f"{len(existing)} Field-* region(s) already exist; send replace=true "
                          "to replace them (approvals of replaced regions are lost).", 409,
                          {"existing": [r.name for r in existing]})
    keep = [r for r in a.regions if not FIELD_NAME.match(r.name)]
    taken = {r.id for r in keep}
    new: list[Region] = []
    for p in planned:
        rid = new_id()
        while rid in taken:
            rid = new_id()
        taken.add(rid)
        new.append(Region(id=rid, kind=RegionKind.counting, name=p["name"], field_index=p["field_index"],
                          polygon=[tuple(q) for q in p["polygon"]]))
    return new + keep


def compare(a: Analysis, ref: Reference, plane: int, count_type: str, numbering: str = "index") -> dict:
    if count_type not in COUNT_TYPES:
        raise ReviewError("invalid_count_type", f"count_type must be one of {', '.join(COUNT_TYPES)}", 422)
    regs = [r for r in a.regions if r.kind == RegionKind.counting and r.field_index is not None]
    if not regs:
        raise ReviewError("no_field_regions", "There are no field regions (counting regions with a field number). "
                          "Import fields from the grid or draw them first.", 422)
    if not ref.available:
        raise ReviewError("no_reference", ref.reason or "No embedded reference marks.", 422)
    refc = ref.counts(numbering, plane)
    counts = {c.scope: c for c in compute_counts(a)["regions"]}
    rows = []
    for r in sorted(regs, key=lambda r: r.field_index):
        c = counts.get(r.id)
        app = int(getattr(c, count_type)) if c is not None else 0
        rc = refc.get(r.field_index, 0)
        rows.append({"field": r.field_index, "reference_count": rc, "app_count": app, "delta": app - rc,
                     "has_reference_marks": r.field_index in refc, "region": r.name})
    x = [row["reference_count"] for row in rows]
    y = [row["app_count"] for row in rows]
    sxx = sum(v * v for v in x)
    summary = {
        "n_fields": len(rows), "pearson_r": pearson(x, y),
        "slope": (sum(u * v for u, v in zip(x, y)) / sxx) if sxx else None,
        "mae": sum(abs(row["delta"]) for row in rows) / len(rows),
        "total_reference": sum(x), "total_app": sum(y),
    }
    warns = []
    nomark = [row["field"] for row in rows if not row["has_reference_marks"]]
    if nomark:
        warns.append("Fields without reference marks on this plane are compared as 0: " + ", ".join(map(str, nomark))
                     + ". " + AMBIGUITY)
    return {"rows": rows, "summary": summary, "plane": plane, "count_type": count_type, "numbering": numbering,
            "note": ("Side-by-side numbers: the marks stored in the TIFF (plane %d) and the app's current objects "
                     "per field region. Differences can come from either side; neither is treated as the truth here. "
                     "Slope is a least-squares fit through zero (app per reference)." % plane),
            "warnings": warns}
