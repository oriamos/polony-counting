"""'Save' of a reviewed slide: labelled scan + per-slide CSV + shared experiment CSV, next to the original.

Pure helpers (counting, labels, CSV text, experiment upsert) plus `render_labeled` and `save_slide`
(file writing). Field semantics follow the lab layout: a grid cell with a field region and no polonies is 0,
a cell without a region (switched off / never created) is blank.
"""
from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from polony_app.review.state import ReviewError
from polony_app.schemas import Analysis, ObjectClass, Region, RegionKind

N_FIELDS = 24
SLIDE_TOTAL_COL = "Total number of polonies"
EXPERIMENT_FILE = "experiment_counts.csv"
EXPERIMENT_HEADER = ["Slide name", "What counted", "NofPol", "NofGrids"] + [f"Field{i}" for i in range(1, N_FIELDS + 1)]
SLIDE_HEADER = ["SlideID", SLIDE_TOTAL_COL] + [f"Field{i}" for i in range(1, N_FIELDS + 1)]
COUNTABLE = (ObjectClass.red, ObjectClass.green, ObjectClass.yellow)   # `uncertain` is never counted
_RING = {ObjectClass.red: (255, 140, 0), ObjectClass.green: (0, 255, 255), ObjectClass.yellow: (255, 255, 0),
         ObjectClass.uncertain: (160, 160, 160)}


# ------------------------------------------------------------------ what is counted
def parse_classes(classes: Iterable[str]) -> tuple[ObjectClass, ...]:
    out: list[ObjectClass] = []
    for c in classes:
        try:
            oc = ObjectClass(c)
        except ValueError:
            raise ReviewError("invalid_selection", f"unknown class '{c}' (use red, green, yellow)", 422)
        if oc not in COUNTABLE:
            raise ReviewError("invalid_selection", "'uncertain' objects are never counted", 422)
        if oc not in out:
            out.append(oc)
    if not out:
        raise ReviewError("invalid_selection", "choose at least one thing to count (red, green or yellow)", 422)
    return tuple(c for c in COUNTABLE if c in out)


def default_label(classes: Iterable[ObjectClass]) -> str:
    """'red, not yellow' | 'yellow' | 'red and yellow' | 'red, green and yellow' ... (red/green are the exclusive classes)."""
    cs = [c.value for c in COUNTABLE if c in set(classes)]
    names = [c for c in cs if c != "yellow"]
    has_y = "yellow" in cs
    if not names:
        return "yellow"
    head = names[0] if len(names) == 1 else " and ".join(names)
    if not has_y:
        return f"{head}, not yellow"
    return f"{head} and yellow" if len(names) < 2 else f"{', '.join(names[:-1])}, {names[-1]} and yellow"


def slide_name(a: Analysis) -> str:
    """'Slide name' of the CSVs: the scan's file stem when opened from disk, else the analysis name."""
    src = a.image.source_path if a.image else None
    return Path(src).stem if src else a.name


def slug(label: str) -> str:
    return (re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-") or "count").lower()[:40]


def field_regions(a: Analysis) -> dict[int, Region]:
    """Counting regions with a lab field number 1..N_FIELDS."""
    out: dict[int, Region] = {}
    for r in a.regions:
        if r.kind == RegionKind.counting and r.field_index is not None:
            if not 1 <= r.field_index <= N_FIELDS:
                raise ReviewError("field_out_of_range", f"region '{r.name}' has field number {r.field_index}; "
                                  f"the CSV has fields 1..{N_FIELDS}", 422)
            if r.field_index in out:
                raise ReviewError("duplicate_field", f"field {r.field_index} is defined twice "
                                  f"('{out[r.field_index].name}', '{r.name}')", 422)
            out[r.field_index] = r
    return out


def count_fields(a: Analysis, classes: Iterable[ObjectClass]) -> dict[int, int]:
    """{field: count} for every counted field (zero is a real 0). Fields without a region are absent (blank)."""
    regs = field_regions(a)
    if not regs:
        raise ReviewError("no_field_regions", "No grid fields are selected for counting. Create the grid and keep "
                          "at least one field switched on.", 422)
    want = set(classes)
    by_region = {r.id: f for f, r in regs.items()}
    out = {f: 0 for f in regs}
    for o in a.objects:
        if o.deleted or o.cls not in want:
            continue
        f = by_region.get(o.region_id or "")
        if f is not None:
            out[f] += 1
    return out


# ------------------------------------------------------------------ CSV text
def _csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue()


def _field_cells(counts: dict[int, int]) -> list[Optional[int]]:
    return [counts.get(i) for i in range(1, N_FIELDS + 1)]


def slide_csv(slide_id: str, counts: dict[int, int]) -> str:
    return _csv(SLIDE_HEADER, [[slide_id, sum(counts.values())] + _field_cells(counts)])


def experiment_row(slide_name: str, what: str, counts: dict[int, int]) -> list:
    return [slide_name, what, sum(counts.values()), len(counts)] + _field_cells(counts)


def upsert_experiment(path: Path, rows: list[list]) -> None:
    """Add/replace rows keyed by (slide name, what counted); other rows are kept. Atomic write (UTF-8 with BOM for Excel)."""
    existing: list[list[str]] = []
    if path.exists():
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                rd = list(csv.reader(fh))
        except (OSError, UnicodeDecodeError) as e:
            raise ReviewError("experiment_csv_unreadable", f"cannot read {path.name}: {e}", 409)
        if rd and rd[0] != EXPERIMENT_HEADER:
            raise ReviewError("experiment_csv_mismatch", f"{path.name} exists with a different header; "
                              "move or rename it so a new one can be started", 409)
        existing = rd[1:]
    index = {(r[0], r[1]): i for i, r in enumerate(existing) if len(r) >= 2}
    for row in rows:
        cells = ["" if v is None else str(v) for v in row]
        k = (cells[0], cells[1])
        if k in index:
            existing[index[k]] = cells
        else:
            index[k] = len(existing)
            existing.append(cells)
    text = _csv(EXPERIMENT_HEADER, existing)
    _atomic_write(path, text.encode("utf-8-sig"))


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ------------------------------------------------------------------ labelled scan
def render_labeled(red: np.ndarray, green: np.ndarray, a: Analysis, selected: set[ObjectClass],
                   per_field: dict[int, int]) -> np.ndarray:
    """H x W x 3 uint8: red/green composite with the grid, counted objects ringed by class and field counts.
    Display scaling only (percentiles); the original file is never touched."""
    from PIL import Image, ImageDraw, ImageFont

    from polony_app.imaging.render import normalize, percentile_range

    h, w = red.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 0] = normalize(red, *percentile_range(red))
    rgb[..., 1] = normalize(green, *percentile_range(green))
    img = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    size = max(14, min(h, w) // 45)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    regs = field_regions(a)
    for r in a.regions:
        if r.kind == RegionKind.exclusion:
            d.polygon([(x, y) for x, y in r.polygon], outline=(255, 0, 255))
    for s in (s for s in a.slide.skipped_fields if s.field_index not in regs):
        pts = [(x, y) for x, y in s.polygon]
        d.polygon(pts, outline=(110, 110, 110))
        d.line([pts[0], pts[2]], fill=(110, 110, 110))
    for f, r in regs.items():
        pts = [(x, y) for x, y in r.polygon]
        d.polygon(pts, outline=(255, 255, 255))
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        d.text((x0 + 4, y0 + 2), f"{f}: {per_field.get(f, 0)}", fill=(255, 255, 255), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    by_region = {r.id for r in regs.values()}
    rad = max(5, size // 2)
    for o in a.objects:
        if o.deleted or o.region_id not in by_region or o.cls not in selected:
            continue
        d.ellipse([o.x - rad, o.y - rad, o.x + rad, o.y + rad], outline=_RING[o.cls], width=2)
    return np.asarray(img)


def _write_tiff(path: Path, arr: np.ndarray) -> None:
    import tifffile

    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tifffile.imwrite(tmp, arr, photometric="rgb", compression="deflate", software="Polony Review")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ------------------------------------------------------------------ save
def resolve_output_dir(a: Analysis, output_dir: Optional[str]) -> Path:
    src = a.image.source_path if a.image else None
    chosen = output_dir or (str(Path(src).parent) if src else None)
    if not chosen:
        raise ReviewError("no_output_dir", "This scan was uploaded, so its folder is unknown. Give an output folder, "
                          "or open the scan from disk to save next to it.", 422)
    p = Path(chosen).expanduser()
    if not p.is_dir():
        raise ReviewError("output_dir_missing", f"output folder does not exist: {p}", 422)
    if not os.access(p, os.W_OK):
        raise ReviewError("output_dir_readonly", f"no permission to write in: {p}", 422)
    return p


def save_slide(a: Analysis, red: np.ndarray, green: np.ndarray, selections: list[tuple[str, tuple[ObjectClass, ...]]],
               out_dir: Path) -> dict:
    """Write <stem>_labeled.tif, one per-slide CSV per selection and upsert experiment_counts.csv. Returns a report."""
    if not selections:
        raise ReviewError("invalid_selection", "choose at least one thing to count", 422)
    labels = [s[0] for s in selections]
    if len(set(labels)) != len(labels):
        raise ReviewError("invalid_selection", "'what counted' labels must be different from each other", 422)
    src = a.image.source_path if a.image else None
    stem = slide_name(a)
    fname = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", stem).strip(" .") or "slide"
    slide_id = stem
    per_sel = [(lab, cls, count_fields(a, cls)) for lab, cls in selections]
    union = {c for _, cls in selections for c in cls}
    # the labelled image shows the first selection's counts when there are several; rings cover the union
    img = render_labeled(red, green, a, union, per_sel[0][2])
    labeled = out_dir / f"{fname}_labeled.tif"
    if src and labeled.resolve() == Path(src).resolve():
        raise ReviewError("would_overwrite_original", "refusing to overwrite the original scan", 409)
    _write_tiff(labeled, img)
    files = [labeled.name]
    multi = len(per_sel) > 1
    for lab, _cls, counts in per_sel:
        p = out_dir / (f"{fname}__{slug(lab)}.csv" if multi else f"{fname}.csv")
        _atomic_write(p, slide_csv(slide_id, counts).encode("utf-8-sig"))
        files.append(p.name)
    upsert_experiment(out_dir / EXPERIMENT_FILE, [experiment_row(slide_id, lab, counts) for lab, _c, counts in per_sel])
    files.append(EXPERIMENT_FILE)
    unc = sum(1 for o in a.objects if not o.deleted and o.cls == ObjectClass.uncertain)
    warnings = []
    if unc:
        warnings.append(f"{unc} uncertain object(s) are not counted in any column.")
    unapproved = [r.name for r in field_regions(a).values() if not r.approved]
    if unapproved:
        warnings.append(f"{len(unapproved)} counted field(s) are not marked approved.")
    return {"directory": str(out_dir), "files": files, "slide_id": slide_id, "warnings": warnings,
            "rows": [{"what_counted": lab, "nofpol": sum(c.values()), "nofgrids": len(c),
                      "fields": {str(k): v for k, v in sorted(c.items())}} for lab, _cls, c in per_sel]}
