"""Lab-format export (see docs/app/LAB_EXPORT.md).

Two layouts that mirror the researcher's own workbooks:

* T7-style  - one sheet / CSV per count series, one row per slide, columns A-AG,
  column I = ``=AVERAGE(Jn:AGn)*(Fn/Gn)/En*1000`` (the lab formula; the concentration
  factor in column D is NOT part of it, exactly as in the lab's T7A/T7B files).
* Counts-style - ``Counts_<polony date>.xlsx``, one sheet per series, two rows per slide,
  replicates ``<sample>_A`` / ``<sample>_B`` paired, polonies/ml in AK which DOES multiply by
  the dilution (concentration factor), as in the lab's Counts workbook.

Only FINALIZED analyses are exported. Nothing is inferred: the count type of a series
(red-only / green-only / yellow / red-positive / green-positive) comes from
``SlideMeta.lab_series`` which the user sets; field numbers come from
``Region.field_index``; blank field cell = no such region, 0 = region with zero objects.
The XLSX files contain real Excel formulas AND cached values (so previewers that do not
recalculate still show numbers); Excel recalculates on load.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import polony_app
from polony_app.detection import masks as _masks
from polony_app.review.counts import compute_counts
from polony_app.schemas import Analysis, LabCountType, LabSeries, Region, RegionKind

N_FIELDS = 24

# EXACT header strings of the lab template (note the trailing space after "POLONY DATE" and
# the lab's own spellings "VOLUM" / "ABUNDENCE" / "FAGE").
T7_HEADER = (
    ["slide ID", "SAMPLING DATE", "POLONY DATE ", "CONCENTRATION FACTOR", "SAMPLE VOLUM[ul]",
     "field [Pixel^2]", "grid [Pixel^2]", "QC", "FAGE ABUNDENCE [fage mL-1]"]
    + [f"field {i}" for i in range(1, N_FIELDS + 1)]
)
T7_FORMULA = "=AVERAGE(J{r}:AG{r})*(F{r}/G{r})/E{r}*1000"  # column I, row r (template formula)
PARTIAL_EXCLUSION_THRESHOLD = 0.98  # net/own area below this => flagged in Notes

COUNT_TYPE_LABEL = {
    LabCountType.red_only: "red-only",
    LabCountType.green_only: "green-only",
    LabCountType.yellow: "yellow",
    LabCountType.red_positive: "red-positive (red-only + yellow)",
    LabCountType.green_positive: "green-positive (green-only + yellow)",
}
_CSV_ABUNDANCE_DIGITS = 10  # the lab CSVs carry 10 significant digits (Excel "General")


# ------------------------------------------------------------------ errors
class LabExportError(Exception):
    def __init__(self, status: int, code: str, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.detail = status, code, message, detail


# ------------------------------------------------------------------ small helpers
_DATE_RE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\s*$")


def parse_date(s: Optional[str]) -> Optional[date]:
    """dd/mm/yyyy (also dd-mm-yyyy / dd.mm.yyyy); None if absent or invalid."""
    if not s:
        return None
    m = _DATE_RE.match(s)
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def fmt_slash(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def fmt_dash(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "series"


def t7_filename(series_name: str, sampling: date, ext: str) -> str:
    return f"{_safe(series_name)}_{fmt_dash(sampling)}.{ext}"


def counts_filename(polony: date, ext: str = "xlsx") -> str:
    return f"Counts_{fmt_dash(polony)}.{ext}"


def _clean_num(v):
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    return v


def _num(v) -> str:
    """General-format number as the lab CSVs show it (no trailing .0)."""
    if v is None:
        return ""
    v = _clean_num(v)
    if isinstance(v, int):
        return str(v)
    return format(v, ".15g")


def _sheet_title(name: str, taken: Iterable[str] = ()) -> str:
    t = re.sub(r"[\[\]:*?/\\]", "_", name).strip("'")[:31] or "Sheet"
    base, i = t, 2
    low = {x.lower() for x in taken}
    while t.lower() in low:
        t = f"{base[:28]}_{i}"
        i += 1
    return t


# ------------------------------------------------------------------ prepared data
@dataclass
class FieldRow:
    field_index: int
    region_id: str
    region_name: str
    count: Optional[int]          # None = blank (region fully excluded)
    area_px2: Optional[float]
    area_fraction: Optional[float]   # net area / own polygon area (exclusion + overlap effect)
    note: str = ""


@dataclass
class LabSlide:
    analysis: Analysis
    series: LabSeries
    slide_id: str
    sampling_date: date
    polony_date: date
    concentration_factor: float
    sample_volume_ul: float
    field_area: float
    grid_area: float
    qc: int
    fields: list                   # len N_FIELDS, int or None
    field_table: list              # list[FieldRow]
    warnings: list = field(default_factory=list)

    @property
    def values(self) -> list:
        return [v for v in self.fields if v is not None]

    @property
    def mean_count(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def abundance(self) -> float:
        """Lab formula, same operation order as the Excel cell:
        AVERAGE(fields)*(field/grid)/volume*1000   [per ml]. No concentration factor."""
        return self.mean_count * (self.field_area / self.grid_area) / self.sample_volume_ul * 1000

    @property
    def sample_name(self) -> str:
        return split_slide_id(self.slide_id)[0]

    @property
    def replicate(self) -> Optional[str]:
        return split_slide_id(self.slide_id)[1]


_REP_RE = re.compile(r"^(.*)_([AB])$")


def split_slide_id(slide_id: str) -> tuple[str, Optional[str]]:
    """'AI_A' -> ('AI', 'A'); ids without a trailing _A/_B -> (id, None)."""
    m = _REP_RE.match(slide_id)
    return (m.group(1), m.group(2)) if m else (slide_id, None)


# ------------------------------------------------------------------ field extraction
def field_counts(a: Analysis, count_type: LabCountType):
    """Per-field counts of `count_type` from the counting regions that carry a field_index.

    Returns (fields[24], table, warnings, problems). `problems` block the export
    (field_index outside 1..24, duplicates); `warnings` go to the Notes sheet. A field whose
    region is entirely covered by exclusions is left BLANK (it was not counted at all), a
    partly covered one keeps its count (over the remaining area) and is flagged."""
    fields: list = [None] * N_FIELDS
    table: list[FieldRow] = []
    warnings: list[str] = []
    problems: list[str] = []
    counts = {c.scope: c for c in compute_counts(a)["regions"]}
    shape = (a.image.height, a.image.width) if a.image is not None else None
    areas = _masks.region_areas(list(a.regions), shape) if shape else {}
    seen: dict[int, Region] = {}
    ignored = []
    for r in a.regions:
        if r.kind != RegionKind.counting:
            continue
        if r.field_index is None:
            ignored.append(r.name)
            continue
        idx = r.field_index
        if not isinstance(idx, int) or idx < 1:
            problems.append(f"field_index {idx!r} of region '{r.name}' is not a positive integer")
            continue
        if idx > N_FIELDS:
            problems.append(
                f"field_index {idx} of region '{r.name}' is > {N_FIELDS} "
                f"(the lab template has columns for fields 1-{N_FIELDS} only)")
            continue
        if idx in seen:
            problems.append(f"duplicate field_index {idx}: regions '{seen[idx].name}' and '{r.name}'")
            continue
        seen[idx] = r
        c = counts.get(r.id)
        n = int(getattr(c, count_type.value)) if c is not None else 0
        net = areas.get(r.id)
        own = _masks.region_areas([r], shape).get(r.id) if shape else None
        frac = (net / own) if (net is not None and own) else None
        note = ""
        if frac is not None and net == 0:
            note = (f"field {idx} ('{r.name}') is entirely covered by exclusion region(s)/earlier fields: "
                    "left BLANK, not counted")
            warnings.append(note)
            n_out: Optional[int] = None
        else:
            n_out = n
            if frac is not None and frac < PARTIAL_EXCLUSION_THRESHOLD:
                note = (f"field {idx} ('{r.name}') is partly covered by exclusion region(s): "
                        f"only {frac:.1%} of its area was counted; the count is over the remaining area "
                        "(NOT scaled to the full field)")
                warnings.append(note)
        fields[idx - 1] = n_out
        table.append(FieldRow(idx, r.id, r.name, n_out, net, frac, note))
        if c is not None and c.uncertain:
            warnings.append(f"field {idx} ('{r.name}') holds {c.uncertain} unresolved 'uncertain' object(s), "
                            "which are not in any colour count")
    if ignored:
        warnings.append("counting region(s) without a field_index are NOT part of the lab export: "
                        + ", ".join(ignored))
    table.sort(key=lambda t: t.field_index)
    return fields, table, warnings, problems


# ------------------------------------------------------------------ validation / preparation
def _meta_problems(a: Analysis) -> list[str]:
    s = a.slide
    p: list[str] = []
    if not s.slide_id or not s.slide_id.strip():
        p.append("slide_id")
    for key in ("sampling_date", "polony_date"):
        v = getattr(s, key)
        if v is None or not str(v).strip():
            p.append(key)
        elif parse_date(v) is None:
            p.append(f"{key} (not dd/mm/yyyy: {v!r})")
    for key in ("sample_volume_ul", "field_area", "grid_area", "concentration_factor"):
        v = getattr(s, key)
        if v is None:
            p.append(key)
        elif not (v > 0 and v < float("inf")):
            p.append(f"{key} (must be > 0, got {v!r})")
    if s.qc is None:
        p.append("qc")
    elif s.qc not in (0, 1):
        p.append(f"qc (must be 0 or 1, got {s.qc!r})")
    if not s.concentration_formula_confirmed:
        p.append("concentration_formula_confirmed (field geometry and series mapping not confirmed)")
    if not s.lab_series:
        p.append("lab_series")
    else:
        names = [x.name for x in s.lab_series]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            p.append("lab_series (duplicate series name(s): " + ", ".join(dups) + ")")
    return p


def _label(a: Analysis) -> str:
    return a.slide.slide_id or a.name or a.id


def prepare(analyses: Sequence[Analysis], series_name: Optional[str] = None) -> list[LabSlide]:
    """Validate and turn finalized analyses into LabSlide rows.

    series_name given: one slide per analysis, each must define that series.
    series_name None : one slide per (analysis, series) it defines.
    Raises LabExportError 409 (not finalized) or 422 (missing / invalid inputs, with the exact
    list per analysis)."""
    if not analyses:
        raise LabExportError(422, "no_analyses", "no analyses selected")
    ids = [a.id for a in analyses]
    if len(set(ids)) != len(ids):
        raise LabExportError(422, "duplicate_analysis", "an analysis is listed more than once")
    drafts = [a for a in analyses if a.finalized_at is None]
    if drafts:
        raise LabExportError(
            409, "not_finalized",
            "lab export needs FINALIZED analyses; not finalized: "
            + ", ".join(f"{_label(a)} ({a.id})" for a in drafts),
            {"analyses": [{"analysis_id": a.id, "name": a.name, "slide_id": a.slide.slide_id} for a in drafts]})
    slides: list[LabSlide] = []
    blockers: list[dict] = []
    for a in analyses:
        items = _meta_problems(a)
        if series_name is not None:
            chosen = [x for x in a.slide.lab_series if x.name == series_name]
            if not chosen and a.slide.lab_series:
                items.append(f"lab_series (series {series_name!r} is not defined for this slide)")
        else:
            chosen = list(a.slide.lab_series)
        if not chosen and not a.slide.lab_series and "lab_series" not in items:
            items.append("lab_series")
        built: list[LabSlide] = []
        if not items:
            for ser in chosen[:1] if series_name is not None else chosen:
                fields, table, warns, probs = field_counts(a, ser.count_type)
                if not table and not probs:
                    items.append(f"field_regions (no counting region with a field_index for series {ser.name!r})")
                    continue
                items.extend(probs)
                if probs:
                    continue
                s = a.slide
                built.append(LabSlide(
                    analysis=a, series=ser, slide_id=s.slide_id.strip(),
                    sampling_date=parse_date(s.sampling_date), polony_date=parse_date(s.polony_date),
                    concentration_factor=s.concentration_factor, sample_volume_ul=s.sample_volume_ul,
                    field_area=s.field_area, grid_area=s.grid_area, qc=int(s.qc), fields=fields,
                    field_table=table, warnings=[f"{s.slide_id}: {w}" for w in warns]))
                if not any(v is not None for v in fields):
                    items.append(f"field_regions (every field of series {ser.name!r} is blank)")
                    built.pop()
        if items:
            blockers.append({"analysis_id": a.id, "name": a.name, "slide_id": a.slide.slide_id, "missing": items})
        else:
            slides.extend(built)
    if blockers:
        msg = "lab export blocked; " + "; ".join(
            f"{b['slide_id'] or b['name'] or b['analysis_id']}: " + ", ".join(b["missing"]) for b in blockers)
        raise LabExportError(422, "lab_export_incomplete", msg, {"analyses": blockers})
    # slide ids must be unique within a series
    seen: dict[tuple[str, str], str] = {}
    for sl in slides:
        k = (sl.series.name, sl.slide_id)
        if k in seen:
            raise LabExportError(422, "duplicate_slide_id",
                                 f"slide_id {sl.slide_id!r} is used by two analyses in series {sl.series.name!r}",
                                 {"slide_id": sl.slide_id, "analysis_ids": [seen[k], sl.analysis.id]})
        seen[k] = sl.analysis.id
    # one series name must mean one count type within an export
    types: dict[str, set] = {}
    for sl in slides:
        types.setdefault(sl.series.name, set()).add(sl.series.count_type)
    bad = {n: sorted(t.value for t in v) for n, v in types.items() if len(v) > 1}
    if bad:
        raise LabExportError(
            422, "inconsistent_series",
            "the same series name is mapped to different count types in this export: "
            + "; ".join(f"{n}: {', '.join(v)}" for n, v in bad.items()), {"series": bad})
    return slides


# ------------------------------------------------------------------ T7-style table
def t7_table(analyses: Sequence[Analysis], series_name: str) -> list[list]:
    """Rows of the T7-style table (without header): 33 values per slide; dates are
    datetime.date, empty field = None, abundance = the value the Excel formula yields."""
    return [_t7_row(s) for s in prepare(analyses, series_name)]


def _t7_row(s: LabSlide) -> list:
    return ([s.slide_id, s.sampling_date, s.polony_date, _clean_num(s.concentration_factor),
             _clean_num(s.sample_volume_ul), _clean_num(s.field_area), _clean_num(s.grid_area), s.qc,
             s.abundance] + list(s.fields))


def _group_dates(slides: Sequence[LabSlide]) -> list[date]:
    return sorted({s.sampling_date for s in slides})


def sampling_date_of(slides: Sequence[LabSlide]) -> date:
    return min(s.sampling_date for s in slides)


def write_t7_csv(analyses: Sequence[Analysis], series_name: str, digits: int = _CSV_ABUNDANCE_DIGITS) -> bytes:
    """CSV exactly like the lab's: UTF-8 BOM, CRLF row separators, no newline after the last
    row, dd/mm/yyyy dates, abundance with 10 significant digits, empty field cells."""
    return _t7_csv_bytes(prepare(analyses, series_name), digits)


def _t7_csv_bytes(slides: Sequence[LabSlide], digits: int = _CSV_ABUNDANCE_DIGITS) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(T7_HEADER)
    for s in slides:
        w.writerow([s.slide_id, fmt_slash(s.sampling_date), fmt_slash(s.polony_date),
                    _num(s.concentration_factor), _num(s.sample_volume_ul), _num(s.field_area),
                    _num(s.grid_area), s.qc, format(s.abundance, f".{digits}g")]
                   + ["" if v is None else v for v in s.fields])
    text = buf.getvalue()
    if text.endswith("\r\n"):
        text = text[:-2]
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def write_t7_xlsx(analyses: Sequence[Analysis], series_name: str, exported_at: Optional[str] = None) -> bytes:
    return _t7_xlsx_bytes(prepare(analyses, series_name), series_name, exported_at)


_T7_WIDTHS = {"A": 8.0, "B": 13.33203125, "C": 11.83203125, "D": 20.33203125, "E": 15.6640625,
              "F": 11.33203125, "G": 11.0, "H": 11.0, "I": 23.5}


def _t7_xlsx_bytes(slides: Sequence[LabSlide], series_name: str, exported_at: Optional[str] = None) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = _sheet_title(series_name)
    ws.sheet_format.defaultColWidth = 8.83203125
    ws.sheet_format.baseColWidth = 10
    for k, w in _T7_WIDTHS.items():
        ws.column_dimensions[k].width = w
    ws.append(T7_HEADER)
    cached: dict[str, float] = {}
    for i, s in enumerate(slides):
        r = i + 2
        row = _t7_row(s)
        for c, v in enumerate(row, start=1):
            if c == 9:
                ws.cell(r, c, T7_FORMULA.format(r=r))
                cached[f"I{r}"] = s.abundance
            elif v is not None:
                cell = ws.cell(r, c, v)
                if isinstance(v, date):
                    cell.value = datetime(v.year, v.month, v.day)
                    cell.number_format = "mm-dd-yy"   # template's (locale-dependent) short date
    _notes_sheet(wb, _notes_rows(slides, "T7-style", exported_at))
    return _finish(wb, {ws.title: cached})


# ------------------------------------------------------------------ cached values
def _finish(wb: Workbook, cached: dict[str, dict[str, float]]) -> bytes:
    """Save and write the python-computed result into every formula cell as its cached value
    (openpyxl writes none). Excel/LibreOffice still recalculate on load (fullCalcOnLoad)."""
    buf = io.BytesIO()
    wb.save(buf)
    zin = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    sheet_files = {ws.title: f"xl/worksheets/sheet{i}.xml" for i, ws in enumerate(wb.worksheets, start=1)}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            for title, f in sheet_files.items():
                if item.filename == f and cached.get(title):
                    xml = data.decode("utf-8")
                    for coord, val in cached[title].items():
                        if val is None:
                            continue
                        pat = re.compile(r'(<c r="%s"[^>]*>\s*<f>.*?</f>\s*)<v\s*/>' % coord, re.S)
                        xml, n = pat.subn(lambda m, v=val: m.group(1) + "<v>" + repr(float(v)) + "</v>", xml, count=1)
                        assert n == 1, f"formula cell {coord} not found in {title}"
                    data = xml.encode("utf-8")
            zout.writestr(item, data)
    return out.getvalue()


# ------------------------------------------------------------------ Notes sheet
def _notes_rows(slides: Sequence[LabSlide], style: str, exported_at: Optional[str],
                extra: Sequence[str] = ()) -> list[list]:
    rows: list[list] = [
        ["Lab export notes", None],
        ["style", style],
        ["status", "FINAL (only finalized analyses are exported)"],
        ["exported_at", exported_at or datetime.now().astimezone().isoformat(timespec="seconds")],
        ["app_version", f"polony_app {polony_app.__version__}"],
        ["abundance formula (T7 style)",
         "=AVERAGE(fields)*(field/grid)/SAMPLE VOLUME[ul]*1000. AVERAGE ignores blank cells and includes zeros. "
         "The CONCENTRATION FACTOR column is NOT used (as in the lab's T7 sheets)."],
        ["polonies/ml (Counts style)", "AK = AH*1000/AJ*AI : polonies per slide x 1000 / volume[ul] x dilution "
                                       "(CONCENTRATION FACTOR is applied here, as in the lab's Counts workbook)."],
        ["blank vs zero", "blank field cell = no counting region with that field_index; 0 = region with zero objects"],
    ]
    for e in extra:
        rows.append(["note", e])
    series = {}
    for s in slides:
        series.setdefault(s.series.name, s.series)
    rows.append([None, None])
    rows.append(["Series mapping used (chosen by the user, never inferred)", None])
    for name, ser in series.items():
        rows.append([name, f"{ser.count_type.value} = {COUNT_TYPE_LABEL[ser.count_type]}"
                           + (f"; sheet title {ser.sheet_title!r}" if ser.sheet_title else "")])
    rows.append([None, None])
    rows.append(["Slides", None])
    rows.append(["slide_id", "series | analysis id | analysis name | finalized_at | status | QC | "
                             "sha256 of original files"])
    for s in slides:
        a = s.analysis
        sha = "; ".join(f"{k}={v}" for k, v in (a.image.sha256.items() if a.image else []))
        rows.append([s.slide_id, f"{s.series.name} | {a.id} | {a.name} | {a.finalized_at} | FINAL | {s.qc} | {sha}"])
    rows.append([None, None])
    rows.append(["Field -> region table", None])
    rows.append(["slide_id / field", "region id | region name | count | counted area px2 | counted fraction of region"])
    for s in slides:
        for t in s.field_table:
            rows.append([f"{s.slide_id} / field {t.field_index}",
                         f"{t.region_id} | {t.region_name} | "
                         f"{'(blank)' if t.count is None else t.count} | "
                         f"{'' if t.area_px2 is None else int(t.area_px2)} | "
                         f"{'' if t.area_fraction is None else format(t.area_fraction, '.3f')}"])
    rows.append([None, None])
    rows.append(["Warnings", None])
    n = 0
    for s in slides:
        for w in s.warnings:
            rows.append(["warning", w])
            n += 1
    if not n:
        rows.append(["warning", "none"])
    return rows


def notes_text(slides: Sequence[LabSlide], style: str, exported_at: Optional[str] = None,
               extra: Sequence[str] = ()) -> str:
    out = []
    for k, v in _notes_rows(slides, style, exported_at, extra):
        if k is None and v is None:
            out.append("")
        elif v is None:
            out.append(str(k))
        else:
            out.append(f"{k}: {v}")
    return "\n".join(out) + "\n"


def all_warnings(slides: Sequence[LabSlide]) -> list[str]:
    return [w for s in slides for w in s.warnings]


def _notes_sheet(wb: Workbook, rows: list[list]) -> None:
    ws = wb.create_sheet("Notes")
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 140
    for r in rows:
        ws.append([x if x is not None else None for x in r])
    ws["A1"].font = Font(bold=True, size=14)
    for row in ws.iter_rows(min_row=2):
        a, b = row[0], row[1]
        if a.value is not None and b.value is None:
            a.font = Font(bold=True)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        a.alignment = Alignment(vertical="top")


# ------------------------------------------------------------------ Counts-style workbook
_THIN = Side(style="thin")
_BOX = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_GREEN = PatternFill("solid", fgColor="FFDAF2D0")
_PINK = PatternFill("solid", fgColor="FFFFCCCC")
_GREY = PatternFill("solid", fgColor="FFBFBFBF")
_RED = PatternFill("solid", fgColor="FFFF0000")
_DRED = PatternFill("solid", fgColor="FFC00000")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

_C = {"G": 7, "H": 8, "AE": 31, "AG": 33, "AH": 34, "AI": 35, "AJ": 36, "AK": 37, "AL": 38, "AM": 39,
      "AN": 40, "AO": 41, "AP": 42, "AQ": 43}


def _col(name: str) -> int:
    return _C[name]


def _is_ntc(sample: str) -> bool:
    return sample.upper().startswith("NTC")


def _counts_blocks(slides: Sequence[LabSlide]) -> list[list[LabSlide]]:
    """Group slides into sample blocks (first-appearance order): [A, B] pairs, singles."""
    order: list[str] = []
    groups: dict[str, list[LabSlide]] = {}
    for s in slides:
        key = s.sample_name
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(s)
    out = []
    for k in order:
        g = groups[k]
        reps = [x.replicate for x in g]
        if len(set(reps)) != len(reps) or (None in reps and len(g) > 1):
            raise LabExportError(
                422, "ambiguous_replicates",
                f"Counts-style export pairs replicates by slide_id '<sample>_<A|B>'; sample {k!r} is not "
                f"unique/consistent (slide ids: {', '.join(x.slide_id for x in g)})",
                {"sample": k, "slide_ids": [x.slide_id for x in g]})
        g.sort(key=lambda x: (x.replicate or ""))
        out.append(g)
    return out


def _counts_sheet(ws, series_title: str, count_type: LabCountType, slides: Sequence[LabSlide],
                  polony: date, efficiency_input: Optional[float]) -> dict[str, float]:
    cached: dict[str, float] = {}
    label = {LabCountType.red_only: "red-only", LabCountType.green_only: "green-only",
             LabCountType.yellow: "yellow", LabCountType.red_positive: "red-positive",
             LabCountType.green_positive: "green-positive"}[count_type]
    widths = {"A": 18.66, "B": 13.0, "C": 14.5, "D": 11.66, "E": 17.16, "F": 14.0}
    for k, w in widths.items():
        ws.column_dimensions[k].width = w
    for c in range(8, 32):
        ws.column_dimensions[get_column_letter(c)].width = 10.83
    ws["A1"] = "polony_app lab export"
    ws["A1"].font = Font(size=16)
    ws["B1"] = f"{series_title} counts"
    ws["B1"].font = Font(size=16)
    ws["A3"] = f"SLIDES FROM {fmt_dash(polony)}"
    ws["A3"].font = Font(size=16)
    heads = {
        "A4": f"Cy5 red & Cy3 green \n({label} counts)", "B4": "assay", "C4": "Phage & dilution",
        "D4": "Slide number", "E4": label, "F4": "summ / slide", "G4": "well area/grid area",
        "H4": f"total {label} polonies/ grid", "AG4": "average number of polonies/grid",
        "AH4": "polonies/ slide", "AI4": "dilution", "AJ4": "sample volume/ slide", "AK4": " polonies/ ml",
        "AL4": "%v diff btw slides", "AM4": "Average ", "AN4": "Average - NTC", "AO4": "Efficiency average",
        "AP4": "Efficiency average-NTC", "AQ4": "pol/slide -NTC",
    }
    for k, v in heads.items():
        ws[k] = v
        ws[k].alignment = _CENTER
    for k in ["B4", "C4", "D4", "E4", "F4", "G4", "H4", "AG4", "AH4", "AI4", "AJ4"]:
        ws[k].font = Font(bold=True)
        ws[k].border = _BOX
    ws["AH4"].fill = _PINK
    ws["AI4"].fill = ws["AJ4"].fill = _GREY
    ws["AK4"].fill = _DRED
    ws["AK4"].font = Font(color="FFFFFFFF")
    for k in ["B", "C", "D", "E", "F", "G", "AG", "AH", "AI", "AJ", "AK"]:
        ws.merge_cells(f"{k}4:{k}5")
    ws.merge_cells("H4:AE5")
    ws.row_dimensions[1].height = ws.row_dimensions[2].height = 22
    ws.row_dimensions[3].height = 23
    ws.row_dimensions[4].height = 51

    blocks = _counts_blocks(slides)
    ntc_ref: Optional[str] = None
    # first pass: row positions
    pos: dict[str, tuple[int, int]] = {}   # slide_id -> (header row, value row)
    r = 6
    for blk in blocks:
        for s in blk:
            pos[s.slide_id] = (r, r + 1)
            r += 2
        r += 1
    # NTC reference (first NTC sample of this sheet)
    for blk in blocks:
        if _is_ntc(blk[0].sample_name):
            if len(blk) == 2:
                ntc_ref = f"AM{pos[blk[0].slide_id][1]}"
            else:
                ntc_ref = f"AH{pos[blk[0].slide_id][1]}"
            break

    def vals(s):  # python mirror of the formulas
        mean = s.mean_count
        ah = mean * round(s.field_area / s.grid_area, 10)
        return mean, ah

    for blk in blocks:
        ntc_block = _is_ntc(blk[0].sample_name)
        ahs = {}
        for s in blk:
            ahs[s.slide_id] = vals(s)[1]
        for s in blk:
            h, v = pos[s.slide_id]
            rep = s.replicate
            first = s is blk[0]
            ws[f"B{h}"] = series_title if (not first or len(blk) == 1) else None
            ws[f"C{h}"] = fmt_dash(s.sampling_date) if first else s.sample_name
            if first and len(blk) == 1:
                ws[f"C{h}"] = s.sample_name
            ws[f"D{h}"] = rep
            ws[f"F{h}"] = f"=SUM(H{v}:AE{v})"
            cached[f"F{h}"] = float(sum(s.values))
            for col in "BCDEF":
                ws.merge_cells(f"{col}{h}:{col}{v}")
                c = ws[f"{col}{h}"]
                c.fill = _GREEN
                c.alignment = _CENTER
                c.border = _BOX
                c.font = Font(bold=col in "BC", size=11)
            for i in range(N_FIELDS):
                c = ws.cell(h, 8 + i, f"Field-{i + 1}")
                c.font = Font(bold=True, color="FFFF0000", name="Times New Roman")
                c.alignment = Alignment(horizontal="center")
                x = s.fields[i]
                if x is not None:
                    ws.cell(v, 8 + i, x)
            gval = _clean_num(round(s.field_area / s.grid_area, 10))
            ws[f"G{v}"] = gval
            ws[f"G{v}"].number_format = "0.00"
            ws[f"G{v}"].font = Font(bold=True)
            ws[f"G{v}"].alignment = Alignment(horizontal="center")
            mean, ah = vals(s)
            ws[f"AG{v}"] = f"=AVERAGE(H{v}:AE{v})"
            cached[f"AG{v}"] = mean
            ws[f"AH{v}"] = f"=AG{v}*G{v}"
            cached[f"AH{v}"] = ah
            ws[f"AI{v}"] = _clean_num(s.concentration_factor)
            ws[f"AJ{v}"] = _clean_num(s.sample_volume_ul)
            ak = ah * 1000 / s.sample_volume_ul * s.concentration_factor
            ws[f"AK{v}"] = f"=(AH{v})*1000/AJ{v}*AI{v}"
            cached[f"AK{v}"] = ak
            fm = {"AG": "0.00", "AH": "0.0", "AI": "General", "AJ": "General", "AK": "0.0E+00"}
            for col, nf in fm.items():
                c = ws[f"{col}{v}"]
                c.number_format = nf
                c.alignment = Alignment(horizontal="center")
                if col in ("AH", "AI", "AJ"):
                    c.font = Font(bold=True)
                    c.fill = _PINK
                if col == "AK":
                    c.font = Font(bold=True)
                    c.fill = _RED
            # replicate statistics, on the A row of a complete pair
            if rep == "A" and len(blk) == 2:
                b = blk[1]
                vb = pos[b.slide_id][1]
                ahb = ahs[b.slide_id]
                avg = (ah + ahb) / 2
                if avg != 0:
                    ws[f"AL{v}"] = f"=100*ABS(AH{v}-AH{vb})/AVERAGE(AH{v},AH{vb})"
                    cached[f"AL{v}"] = 100 * abs(ah - ahb) / avg
                ws[f"AM{v}"] = f"=AVERAGE(AH{v},AH{vb})"
                cached[f"AM{v}"] = avg
                if efficiency_input:
                    ws[f"AO{v}"] = f"=100*AM{v}/{_num(efficiency_input)}"
                    cached[f"AO{v}"] = 100 * avg / efficiency_input
                if ntc_ref and not ntc_block:
                    ws[f"AN{v}"] = f"=AM{v}-{ntc_ref}"
            if ntc_ref and not ntc_block:
                ws[f"AQ{v}"] = f"=AH{v}-{ntc_ref}"
            for col, nf in (("AL", "0.00"), ("AM", "0.0"), ("AN", "0.00"), ("AO", "0.00"), ("AQ", "0.0")):
                c = ws[f"{col}{v}"]
                c.number_format = nf
                c.alignment = Alignment(horizontal="center")
            ws.row_dimensions[h].height = 17
            ws.row_dimensions[v].height = 17
    return cached


def write_counts_xlsx(analyses: Sequence[Analysis], series_name: Optional[str] = None,
                      exported_at: Optional[str] = None) -> bytes:
    """Counts_<polony date>.xlsx : one sheet per series (+ a Notes sheet)."""
    slides = prepare(analyses, series_name)
    return _counts_xlsx_bytes(slides, exported_at)


def _counts_xlsx_bytes(slides: Sequence[LabSlide], exported_at: Optional[str] = None) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    by_series: dict[str, list[LabSlide]] = {}
    for s in slides:
        by_series.setdefault(s.series.name, []).append(s)
    polonies = sorted({s.polony_date for s in slides})
    extra = []
    if len(polonies) > 1:
        extra.append("slides have different polony dates (" + ", ".join(fmt_dash(d) for d in polonies)
                     + "); the earliest is used in the file name and header")
    eff = {s.analysis.slide.efficiency_input for s in slides if s.analysis.slide.efficiency_input}
    efficiency = None
    if eff:
        efficiency = sorted(eff)[0]
        if len(eff) > 1:
            extra.append("different efficiency_input values were set; the smallest is used: " + _num(efficiency))
    cached: dict[str, dict[str, float]] = {}
    titles: list[str] = []
    for name, sl in by_series.items():
        ser = sl[0].series
        title = _sheet_title(ser.sheet_title or name, titles)
        titles.append(title)
        ws = wb.create_sheet(title)
        ws.sheet_format.defaultRowHeight = 16
        c = _counts_sheet(ws, ser.sheet_title or name, ser.count_type, sl, min(polonies), efficiency)
        _fix_ntc_cache(ws, c)
        cached[title] = c
        if any(_is_ntc(x.sample_name) for x in sl):
            extra.append(f"sheet {title!r}: NTC rows found - 'Average - NTC' (AN) and 'pol/slide -NTC' (AQ) "
                         "subtract the first NTC sample of the sheet")
        else:
            extra.append(f"sheet {title!r}: no NTC sample (slide_id starting 'NTC') in this export - "
                         "AN, AP and AQ are left blank")
    if efficiency is None:
        extra.append("efficiency_input is not set: column AO ('Efficiency average') is blank; AP is never "
                     "filled (its denominator is not defined by the app)")
    _notes_sheet(wb, _notes_rows(slides, "Counts-style", exported_at, extra))
    return _finish(wb, cached)


def _fix_ntc_cache(ws, cached: dict[str, float]) -> None:
    """Fill cached AN/AQ values (needs the NTC cell values, which may sit below the data rows)."""
    # recompute AN/AQ from formulas now that every AH/AM exists
    for row in ws.iter_rows(min_col=_col("AN"), max_col=_col("AQ")):
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("=") and c.column_letter in ("AN", "AQ"):
                m = re.match(r"^=(AM|AH)(\d+)-(AM|AH)(\d+)$", c.value)
                if m:
                    cached[c.coordinate] = cached[f"{m.group(1)}{m.group(2)}"] - cached[f"{m.group(3)}{m.group(4)}"]


# ------------------------------------------------------------------ orchestration used by the API
@dataclass
class LabFile:
    name: str
    data: bytes
    media_type: str


XLSX_MT = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV_MT = "text/csv; charset=utf-8"


def build_files(analyses: Sequence[Analysis], style: str = "t7", series: Optional[str] = None,
                fmt: str = "xlsx", exported_at: Optional[str] = None):
    """Return (files, slides). style 't7' -> one file per (series, sampling date) in the
    requested format (xlsx|csv; fmt 'zip' = both formats); style 'counts' -> one xlsx
    workbook (csv is refused: the Counts layout is a formatted workbook)."""
    if style not in ("t7", "counts"):
        raise LabExportError(422, "invalid_style", "style must be 't7' or 'counts'")
    if fmt not in ("xlsx", "csv", "zip"):
        raise LabExportError(422, "invalid_format", "format must be xlsx, csv or zip")
    if style == "counts":
        if fmt == "csv":
            raise LabExportError(422, "invalid_format", "the Counts-style export is an .xlsx workbook only")
        slides = prepare(analyses, series)
        polony = min(s.polony_date for s in slides)
        return [LabFile(counts_filename(polony), _counts_xlsx_bytes(slides, exported_at), XLSX_MT)], slides
    slides_all = prepare(analyses, series)
    names: list[str] = []
    for s in slides_all:
        if s.series.name not in names:
            names.append(s.series.name)
    files: list[LabFile] = []
    for name in names:
        sl = [s for s in slides_all if s.series.name == name]
        for d in _group_dates(sl):
            grp = [s for s in sl if s.sampling_date == d]
            if fmt in ("xlsx", "zip"):
                files.append(LabFile(t7_filename(name, d, "xlsx"), _t7_xlsx_bytes(grp, name, exported_at), XLSX_MT))
            if fmt in ("csv", "zip"):
                files.append(LabFile(t7_filename(name, d, "csv"), _t7_csv_bytes(grp), CSV_MT))
    return files, slides_all


def package(files: Sequence[LabFile], slides: Sequence[LabSlide], style: str, fmt: str,
            exported_at: Optional[str] = None) -> LabFile:
    """One downloadable file: the file itself, or a zip (with a notes.txt) when several."""
    if len(files) == 1 and fmt != "zip":
        return files[0]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f.name, f.data)
        z.writestr("lab_export_notes.txt", notes_text(slides, "T7-style" if style == "t7" else "Counts-style",
                                                       exported_at))
    return LabFile("lab_export_FINAL.zip", buf.getvalue(), "application/zip")
