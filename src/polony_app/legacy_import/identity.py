"""Slide / sample identity from file names. Everything here is INFERRED, never confirmed.

Two naming schemes were observed:

* research files (2026): ``<dd-mm-yyyy>_<assay>_<sample>-<replicate>[_<suffix>].tif``, e.g.
  ``29-09-2026_T7_AB_AM-A.tif`` -> assay ``T7_AB``, sample ``AM``, slide replicate ``A``;
  ``..._AS-B_(635_532).tif`` is the rendered RGB composite of ``..._AS-B.tif``;
  ``..._FS-B_D.tif`` has an unexplained suffix ``_D`` (kept verbatim).
* historical training slides: ``<batch>-<sample no.>-<replicate>.tif``,
  e.g. ``1-33-A.tif``, optionally ``3-43-B(1).tif`` (a copy marker).

The lab's CSVs use ``slide ID`` = ``<sample>_<replicate>`` (e.g. ``AM_A``), which is
what :func:`parse_identity` returns as ``slide_id`` for the research scheme.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_RESEARCH = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})_(?P<assay>.+?)_(?P<sample>[A-Za-z0-9]+)-(?P<rep>[A-Za-z])(?P<suffix>(?:_.*)?)$"
)
_LEGACY = re.compile(r"^(?P<batch>\d+)-(?P<sample>\d+)-(?P<rep>[A-Za-z])(?P<suffix>(?:\(\d+\))?)$")
_COMPOSITE = re.compile(r"_\((?P<a>\d{3})_(?P<b>\d{3})\)$")


@dataclass(frozen=True)
class Identity:
    source: str                          # "filename_pattern" | "unknown"
    assay: Optional[str] = None
    specimen_id: Optional[str] = None
    replicate: Optional[str] = None
    slide_id: Optional[str] = None
    file_date: Optional[str] = None
    suffix: Optional[str] = None
    base_stem: str = ""                  # stem without the composite marker
    composite_wavelengths: Optional[tuple[int, int]] = None


def parse_identity(stem: str) -> Identity:
    base = stem
    wl: Optional[tuple[int, int]] = None
    m = _COMPOSITE.search(stem)
    if m:
        wl = (int(m.group("a")), int(m.group("b")))
        base = stem[: m.start()]
    m = _RESEARCH.match(base)
    if m:
        return Identity(
            source="filename_pattern", assay=m.group("assay"), specimen_id=m.group("sample"),
            replicate=m.group("rep"), slide_id=f"{m.group('sample')}_{m.group('rep')}",
            file_date=m.group("date"), suffix=(m.group("suffix") or None), base_stem=base,
            composite_wavelengths=wl)
    m = _LEGACY.match(base)
    if m:
        spec = f"{m.group('batch')}-{m.group('sample')}"
        return Identity(
            source="filename_pattern", assay=f"batch{m.group('batch')}", specimen_id=spec,
            replicate=m.group("rep"), slide_id=f"{spec}_{m.group('rep')}", file_date=None,
            suffix=(m.group("suffix") or None), base_stem=base, composite_wavelengths=wl)
    return Identity(source="unknown", base_stem=base, composite_wavelengths=wl)
