"""Field regions from the grid, and the researcher-vs-app comparison."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from polony_app.review.counts import compute_counts
from polony_app.review.state import ReviewError
from polony_app.schemas import Analysis, Region, RegionKind

from . import examined as ex
from .grid import clip_rect, number_cells, pearson
from .reference import Reference

MAX_LAB_FIELD = 24
FIELD_NAME = re.compile(r"^(Field|Cell)-\d+$")
COUNT_TYPES = ("red_positive", "green_positive", "red_only", "green_only", "yellow")
UNPLANED_REASON = "mark with unspecified plane inside this field"
AMBIGUITY = ("A field with no marks on a page is ambiguous: it may be 'not counted' or 'counted, zero'. The file cannot "
             "tell which, so that field/page is 'unknown' and is excluded from every comparison metric unless an "
             "examined-fields manifest states that the researcher counted it (including a counted zero).")


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


def manifest_names(a: Analysis, originals_dir: Optional[Path] = None) -> tuple[list[str], Optional[str]]:
    """File names (stored name + the name the file was uploaded with) and the sha256 of the first channel file,
    used to find this analysis' entry in a manifest."""
    im = a.image
    if im is None:
        return [], None
    names = [c.source_file for c in im.channels if c.source_file]
    if originals_dir is not None:
        from polony_app.export.exporter import read_upload_manifest
        up = read_upload_manifest(originals_dir) or {}
        names += [up[n] for n in list(names) if up.get(n)]
    sha = im.sha256.get(names[0]) if names else None
    return names, sha


def summarize_manifest(a: Analysis, manifest: ex.ExaminedManifest, originals_dir: Optional[Path] = None) -> dict:
    """What a manifest says about THIS analysis' source file: {matched_file_entries, pages, n_entries, warnings}."""
    names, sha = manifest_names(a, originals_dir)
    warns: list[str] = []
    matched = sum(1 for f in manifest.files
                  if f.file in names or Path(f.file).name in {Path(n).name for n in names}
                  or (f.sha256 and sha and f.sha256.lower() == sha.lower()))
    pages = {int(k): len(v) for k, v in manifest.entries(names, sha).items()}
    if matched == 0:
        warns.append("No entry of the manifest names this analysis' source file (" + ", ".join(names or ["no image"])
                     + "); it will not change any comparison.")
    return {"matched_file_entries": matched, "pages": {str(k): v for k, v in sorted(pages.items())},
            "n_entries": sum(pages.values()), "warnings": warns}


def _resolve_manifest(a: Analysis, ref: Reference, manifest: Optional[ex.ExaminedManifest], numbering: str,
                      field_nums: set[int], originals_dir: Optional[Path] = None) -> tuple[dict[int, dict[int, list[ex.FieldEntry]]], list[str]]:
    """{page: {field number: [entries]}} for this file + warnings for entries that name no known field."""
    out: dict[int, dict[int, list[ex.FieldEntry]]] = {}
    warns: list[str] = []
    if manifest is None:
        return out, warns
    names, sha = manifest_names(a, originals_dir)
    try:
        per_page = manifest.entries(names, sha)
    except ex.ExaminedError as exc:
        raise ReviewError("examined_manifest_mismatch", str(exc), 422) from exc
    cells = ref.cell_field_numbers() if ref.available else {}
    for pg, ents in sorted(per_page.items()):
        for e in ents.values():
            f: Optional[int] = None
            if e.field is not None:
                # TRUSTED as the lab field number only because the manifest author stated it
                f = e.field
            elif e.counter is not None:
                # a counter entry resolves ONLY through the counter's ACTUAL points (their field / majority cell). A
                # counter without any point is never guessed as field counter+1: lab sheets number fields without gaps,
                # so after an absent counter (AM-B: counters 3 and 19) counter+1 would name the wrong field.
                pts = [p for p in ref.points if p.counter == e.counter] if ref.available else []
                if pts:
                    f = ref.field_of(pts[0], numbering)
            else:
                f = cells.get((e.row, e.col))
            label = f"page {pg}: " + ", ".join(f"{k}={v}" for k, v in
                                               (("counter", e.counter), ("field", e.field), ("row", e.row), ("col", e.col))
                                               if v is not None)
            if f is None and e.counter is not None:
                warns.append(f"Manifest entry ({label}) is ignored: counter {e.counter} has no point in the file, so it is "
                             "not resolved to a field (never guessed as counter+1: sheets number fields without gaps, "
                             "so after an absent counter that would name the wrong field). Name the field with 'field' "
                             "(lab number) or 'row'+'col' instead.")
            elif f is None:
                warns.append(f"Manifest entry ({label}) cannot be resolved to a field number and is ignored.")
            elif f not in field_nums:
                warns.append(f"Manifest entry ({label}) is field {f}, which has no Field-* region in this analysis; ignored.")
            else:
                out.setdefault(pg, {}).setdefault(f, []).append(e)
    return out, warns


def _metrics(elig: list[dict]) -> dict:
    x = [r["reference_count"] for r in elig]
    y = [r["app_count"] for r in elig]
    if not elig:
        return {"pearson_r": None, "slope": None, "mae": None, "total_reference": None, "total_app": None}
    sxx = sum(v * v for v in x)
    return {"pearson_r": pearson(x, y), "slope": (sum(u * v for u, v in zip(x, y)) / sxx) if sxx else None,
            "mae": sum(abs(u - v) for u, v in zip(x, y)) / len(elig), "total_reference": sum(x), "total_app": sum(y)}


def compare(a: Analysis, ref: Reference, plane: int, count_type: str, numbering: str = "index",
            manifest: Optional[ex.ExaminedManifest] = None, originals_dir: Optional[Path] = None) -> dict:
    """App vs researcher per Field-* region, ONLY over field/plane pairs the researcher provably examined
    (embedded marks on THIS plane in THIS field, or a manifest/sidecar entry; see fields/examined.py).
    Everything else is listed as 'unknown'/'conflict' with reference_count null and enters no metric."""
    if count_type not in COUNT_TYPES:
        raise ReviewError("invalid_count_type", f"count_type must be one of {', '.join(COUNT_TYPES)}", 422)
    regs = [r for r in a.regions if r.kind == RegionKind.counting and r.field_index is not None]
    if not regs:
        raise ReviewError("no_field_regions", "There are no field regions (counting regions with a field number). "
                          "Import fields from the grid or draw them first.", 422)
    if not ref.available and manifest is None:
        raise ReviewError("no_reference", ref.reason or "No embedded reference marks.", 422)
    refc = ref.counts(numbering, plane) if ref.available else {}
    counts = {c.scope: c for c in compute_counts(a)["regions"]}
    resolved, warns = _resolve_manifest(a, ref, manifest, numbering, {r.field_index for r in regs}, originals_dir)
    if not ref.available:
        warns.append((ref.reason or "No embedded reference marks.") + " Only manifest entries can prove a field was examined.")
    entries = resolved.get(plane, {})
    unplaned_pts = ref.unplaned_points() if ref.available else []
    bad_fields = set(ref.unplaned_fields(numbering)) if ref.available else set()
    rows = []
    for r in sorted(regs, key=lambda r: r.field_index):
        f = r.field_index
        c = counts.get(r.id)
        app = int(getattr(c, count_type)) if c is not None else 0
        n_marks = refc.get(f, 0)
        ents = entries.get(f, [])
        stated = {e.stated_count for e in ents if e.stated_count is not None}
        entry = ents[0] if ents else None
        if f in bad_fields:
            exam = ex.Examination(ex.CONFLICT, n_marks, (ex.SRC_EMBEDDED,), reason=UNPLANED_REASON)
        elif len(stated) > 1:
            exam = ex.Examination(ex.CONFLICT, n_marks, (ex.SRC_EMBEDDED, manifest.source_kind) if manifest else (),
                                  reason=f"the manifest states different counts for this field/plane: {sorted(stated)}")
        else:
            exam = ex.classify(n_marks, entry, mode=ex.MODE_MANIFEST if entry is not None else ex.MODE_MARKED_ONLY,
                               manifest=manifest)
        ok = exam.researcher_examined
        rc = n_marks if ok else None
        rows.append({"field": f, "reference_count": rc, "app_count": app, "delta": (app - rc) if ok else None,
                     "has_reference_marks": n_marks > 0, "region": r.name, "status": exam.status, "examined": ok,
                     "examination_source": exam.examination_source, "reference_marks": n_marks,
                     "reason": exam.reason or None})
    elig = [row for row in rows if row["examined"]]
    n = lambda s: sum(1 for row in rows if row["status"] == s)
    srcs: list[str] = []
    for row in elig:
        for t in row["examination_source"].split("+"):
            if t and t not in srcs:
                srcs.append(t)
    summary = {
        "n_fields": len(rows), "eligible_examined_fields": len(elig), "positive_marked_fields": n(ex.POSITIVE),
        "verified_zero_fields": n(ex.VERIFIED_ZERO), "unknown_fields_excluded": n(ex.UNKNOWN),
        "conflict_fields_excluded": n(ex.CONFLICT), "eligibility_source": "+".join(srcs) if srcs else "none",
        **_metrics(elig),
        "app_total_unknown_fields_not_counted": sum(row["app_count"] for row in rows if not row["examined"]),
        "unplaned_marks": len(unplaned_pts), "unplaned_mark_fields": sorted(bad_fields),
    }
    if unplaned_pts:
        n_nofield = sum(1 for p in unplaned_pts if ref.field_of(p, numbering) is None)
        bad_in = sorted(bad_fields & {row["field"] for row in rows})
        warns.insert(0, f"WARNING: {len(unplaned_pts)} reference mark(s) have no valid plane number (missing, 0 or beyond the "
                        f"planes of the file); they are evidence for no plane. Fields containing such marks"
                        + (f" ({', '.join(map(str, sorted(bad_fields)))})" if bad_fields else "")
                        + " are conflict on ALL planes and excluded from every metric"
                        + (f"; {n_nofield} of them carry no counter, so their field is unknown" if n_nofield else "")
                        + ". Fix the marks in ImageJ (one plane per mark) and re-upload.")
    excl = [row["field"] for row in rows if not row["examined"]]
    if excl:
        warns.append(f"Excluded from every metric (not provably examined on plane {plane}, never read as zero): "
                     + ", ".join(map(str, excl)) + ". " + AMBIGUITY)
    if not elig:
        warns.append(f"No field was provably examined on plane {plane}: no totals, MAE, slope or correlation are reported.")
    return {"rows": rows, "summary": summary, "plane": plane, "count_type": count_type, "numbering": numbering,
            "note": ("Side-by-side numbers: the marks stored in the TIFF (plane %d) and the app's current objects "
                     "per field region, over fields the researcher provably examined on this plane only. Differences can "
                     "come from either side; neither is treated as the truth here. "
                     "Slope is a least-squares fit through zero (app per reference)." % plane),
            "warnings": warns}
