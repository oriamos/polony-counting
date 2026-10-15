"""Verbatim CSV import of legacy counts.

Rules (see docs/app/CONTRACTS.md and the user's requirements):
* a blank cell is NOT zero: ``fields`` holds None for blank, 0 for a written 0;
* raw strings are always kept (``raw_fields``; ``meta`` verbatim);
* column names/values are never reinterpreted or compared; unit labels in the
  header are preserved as written even when suspicious;
* T7A/T7B (or any file/slide name) is never mapped to a colour: the colour is
  an explicit user choice stored in ``CsvImport.color_mapping``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re

from polony_app.schemas import CsvImport, CsvRow

_FIELD_RE = re.compile(r"^\s*field\s*(\d+)\s*$", re.IGNORECASE)
_INT_RE = re.compile(r"^[+-]?\d+$")


def decode_csv(data: bytes) -> tuple[str, list[str]]:
    notes: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        notes.append("UTF-8 byte-order mark removed from the first header name")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(enc)
            if enc != "utf-8-sig":
                notes.append("file is not valid UTF-8; decoded as cp1252")
            return text, notes
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), notes + ["decoded as latin-1"]


def parse_csv(data: bytes, filename: str, import_id: str) -> CsvImport:
    text, notes = decode_csv(data)
    sha = hashlib.sha256(data).hexdigest()
    # sniff delimiter conservatively: comma unless the header clearly uses ; or tab
    first = text.split("\n", 1)[0]
    delim = ","
    if first.count(";") > first.count(",") and first.count(";") > 1:
        delim = ";"
        notes.append("delimiter ';' detected")
    elif first.count("\t") > first.count(",") and first.count("\t") > 1:
        delim = "\t"
        notes.append("delimiter TAB detected")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    rows = [r for r in reader]
    if not rows:
        return CsvImport(id=import_id, filename=filename, sha256=sha, header=[], rows=[],
                         notes=notes + ["file is empty"])
    header = rows[0]  # verbatim (no strip: e.g. 'POLONY DATE ' keeps its space)
    field_cols = [(i, int(m.group(1))) for i, h in enumerate(header) if (m := _FIELD_RE.match(h))]
    field_idx = [i for i, _ in field_cols]
    meta_idx = [i for i in range(len(header)) if i not in set(field_idx)]
    slide_col = next((i for i in meta_idx if header[i].strip().lower() == "slide id"), None)
    if slide_col is None:
        slide_col = 0
        notes.append("no 'slide ID' header found; first column used as slide id")
    if not field_idx:
        notes.append("no 'field N' columns found")
    out: list[CsvRow] = []
    n_blank = 0
    n_bad = 0
    for lineno, r in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in r):
            continue  # fully empty line
        if len(r) > len(header):
            extra = r[len(header):]
            if any(c.strip() for c in extra):
                notes.append(f"line {lineno}: {len(extra)} cell(s) beyond the header ignored (non-empty)")
        cells = list(r) + [""] * (len(header) - len(r))
        if len(r) < len(header):
            notes.append(f"line {lineno}: row shorter than header; missing cells treated as blank")
        meta = {header[i]: cells[i] for i in meta_idx}
        raw = [cells[i] for i in field_idx]
        vals: list[int | None] = []
        for j, c in enumerate(raw):
            s = c.strip()
            if s == "":
                vals.append(None)
                n_blank += 1
            elif _INT_RE.match(s):
                vals.append(int(s))
            else:
                vals.append(None)
                n_bad += 1
                notes.append(f"line {lineno}: field cell {j + 1} {c!r} is not an integer; kept raw, value=None")
        out.append(CsvRow(slide_id=cells[slide_col], meta=meta, fields=vals, raw_fields=raw))
    notes.append(
        f"{n_blank} blank field cell(s) kept as None (not measured); they are NOT zeros"
    )
    notes.append("metadata columns and unit labels are preserved verbatim and not validated")
    notes.append("no colour mapping is inferred from the file or slide names; set it explicitly")
    return CsvImport(id=import_id, filename=filename, sha256=sha, header=header, rows=out,
                     color_mapping=None, notes=notes)
