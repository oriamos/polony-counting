"""Evidence that the researcher EXAMINED a (source TIFF, field, page) tuple.

One definition, used by the historical training import (``ml import-legacy``) and by the comparison report
(``fields.service.compare``). The rule is strict and independent for every ``(source TIFF, field, page)`` tuple:

* a field/page is **examined** only with explicit evidence: (1) embedded ImageJ researcher marks on THAT page, or
  (2) an explicit sidecar (lab CSV / XLSX) or manifest entry that states the researcher counted THAT field on THAT page,
  including a stated count of zero;
* the grid overlay alone is NOT evidence; marks on the other page are NOT evidence; application detections are NOT
  evidence; a field without marks is ``unknown`` (never zero) unless a trusted entry states ``count: 0``;
* contradictions (a stated count that differs from the number of marks, a stated zero with marks present) make the
  field/page ``conflict``: excluded, never "repaired".

``unknown`` and ``conflict`` field/pages must never contribute a label, a negative crop, or a metric.

Manifest file (JSON, schema ``polony-examined-fields`` v1)::

    {"schema": "polony-examined-fields", "version": 1,
     "evidence": "T7B_15-09-2026.csv, blank = not counted, 0 = counted zero (confirmed by <name>)",   # required, free text
     "source_kind": "sidecar_csv" | "sidecar_xlsx" | "explicit_manifest",                              # default explicit_manifest
     "examined_by": "<name>", "recorded_on": "2026-10-14",
     "files": [{"file": "29-09-2026_T7_AB_AM-B.tif", "sha256": "<optional>",
                "pages": {"0": [{"counter": 3, "count": 12}, {"row": 1, "col": 2, "count": 0}, {"field": 4}],
                          "1": []}}]}

A field entry names the field by ``counter`` (ImageJ counter), ``field`` (lab field number) or ``row`` + ``col`` (grid cell,
0-based). ``count`` is the researcher's stated number of polonies (``count: 0`` or ``counted_zero: true`` = verified zero).
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError, model_validator

SCHEMA = "polony-examined-fields"
SCHEMA_VERSION = 1

# evidence kinds (``examination_source``)
SRC_EMBEDDED = "embedded_imagej_marks"
SRC_SIDECAR_CSV = "sidecar_csv"
SRC_SIDECAR_XLSX = "sidecar_xlsx"
SRC_MANIFEST = "explicit_manifest"
SOURCE_KINDS = (SRC_SIDECAR_CSV, SRC_SIDECAR_XLSX, SRC_MANIFEST)

# statuses of a (file, field, page) tuple
POSITIVE = "positive_marked"
VERIFIED_ZERO = "verified_zero"
UNKNOWN = "unknown"
CONFLICT = "conflict"
EXAMINED = (POSITIVE, VERIFIED_ZERO)

MODE_MARKED_ONLY = "marked-only"
MODE_MANIFEST = "examined-fields-manifest"
MODES = (MODE_MARKED_ONLY, MODE_MANIFEST)


class ExaminedError(ValueError):
    """A manifest that cannot be trusted (message is meant for the operator)."""


MAX_COUNT = 10 ** 6                    # a stated count above this is a typo, not a count
MAX_INDEX = 10 ** 6                    # counter / field / row / col
MAX_MANIFEST_FILE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32                    # the schema nests 5 levels; anything deeper is not a manifest


class FieldEntry(BaseModel):
    """Strict: integers must be real JSON integers (``true``, ``1.5`` and ``"3"`` are rejected), unknown keys are errors."""
    model_config = ConfigDict(extra="forbid")

    counter: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_INDEX)
    field: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_INDEX)
    row: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_INDEX)
    col: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_INDEX)
    count: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_COUNT)
    counted_zero: StrictBool = False
    note: StrictStr = ""

    @model_validator(mode="after")
    def _check(self):
        keys = [self.counter is not None, self.field is not None, self.row is not None or self.col is not None]
        if sum(keys) != 1:
            raise ValueError("name the field with exactly one of: counter, field, or row+col")
        if (self.row is None) != (self.col is None):
            raise ValueError("a grid cell needs both row and col")
        if self.counted_zero and self.count not in (None, 0):
            raise ValueError("counted_zero contradicts a non-zero count")
        return self

    @property
    def stated_count(self) -> Optional[int]:
        """The researcher's stated count; None when the entry only says 'examined' without a number."""
        if self.count is not None:
            return int(self.count)
        return 0 if self.counted_zero else None

    def key(self) -> tuple:
        if self.counter is not None:
            return ("counter", int(self.counter))
        if self.field is not None:
            return ("field", int(self.field))
        return ("cell", int(self.row), int(self.col))  # type: ignore[arg-type]


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: StrictStr
    sha256: Optional[StrictStr] = None
    pages: dict[str, list[FieldEntry]] = {}

    @model_validator(mode="after")
    def _check(self):
        if not self.file.strip():
            raise ValueError("a file entry needs a file name")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError(f"{self.file}: sha256 must be 64 hex characters")
        return self


class ExaminedManifest(BaseModel):
    schema_name: StrictStr = Field(default=SCHEMA, alias="schema")
    version: StrictInt = SCHEMA_VERSION
    evidence: StrictStr
    source_kind: StrictStr = SRC_MANIFEST
    examined_by: StrictStr = ""
    recorded_on: StrictStr = ""
    files: list[FileEntry] = []
    sha256: str = ""                     # of the manifest file bytes (filled by the loader)
    # written by the /examined-fields endpoint into the stored copy (provenance of the upload); never evidence
    uploaded_name: StrictStr = ""
    uploaded_sha256: StrictStr = ""
    # documentation-only key written by the ``--write-templates`` helper; carries no evidence
    comment: Optional[list[StrictStr] | StrictStr] = Field(default=None, alias="_comment")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="after")
    def _check(self):
        if self.schema_name != SCHEMA:
            raise ValueError(f"schema must be '{SCHEMA}'")
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"unsupported version {self.version} (supported: {SCHEMA_VERSION})")
        if not self.evidence.strip():
            raise ValueError("'evidence' is required: say where the proof that the researcher examined these fields comes from")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
        seen = set()
        for f in self.files:
            for pg, ents in f.pages.items():
                if not re.fullmatch(r"[0-9]{1,4}", pg):
                    raise ValueError(f"{f.file}: page keys must be 0-based integers of 1-4 ASCII digits (TIFF page), got {pg[:20]!r}")
                for e in ents:
                    k = (f.file, int(pg), e.key())
                    if k in seen:
                        raise ValueError(f"{f.file}: page {pg}: field {e.key()} is listed twice")
                    seen.add(k)
        return self

    def entries(self, names: Iterable[str], sha256: Optional[str] = None) -> dict[int, dict[tuple, FieldEntry]]:
        """page -> {field key -> entry} of the file entry matching one of ``names`` (relative path or base name) or the
        sha256; a sha256 given in the manifest must match."""
        want = {n for n in names if n} | {Path(n).name for n in names if n}
        out: dict[int, dict[tuple, FieldEntry]] = {}
        origin: dict[tuple[int, tuple], str] = {}
        for f in self.files:
            hit = f.file in want or Path(f.file).name in want
            if f.sha256 and sha256:
                if f.sha256.lower() == sha256.lower():
                    hit = True
                elif hit:
                    raise ExaminedError(f"{f.file}: the manifest lists sha256 {f.sha256[:12]}... but the file has "
                                        f"{sha256[:12]}...; refusing to apply evidence to a different file")
            if hit:
                for pg, ents in f.pages.items():
                    page = out.setdefault(int(pg), {})
                    for e in ents:
                        k = e.key()
                        if k in page and page[k].stated_count != e.stated_count:
                            raise ExaminedError(
                                f"page {pg}: field {k} is listed by two file entries ({origin[(int(pg), k)]!r} and {f.file!r}) "
                                "that match the same file with different content; refusing to let one silently win")
                        page[k] = e
                        origin[(int(pg), k)] = f.file
        return out


def _no_duplicate_keys(pairs):
    """``object_pairs_hook`` shared with the mapping files of ``ml import-legacy`` (``ml.common.reject_duplicate_pairs``)."""
    from polony_app.ml.common import reject_duplicate_pairs
    return reject_duplicate_pairs(pairs)


def _json_depth_ok(raw: str) -> None:
    """Reject absurdly nested JSON before the (recursive) parser sees it: scan bracket depth outside strings."""
    depth = 0
    in_str = esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ExaminedError(f"not a valid examined-fields manifest: JSON nested deeper than {MAX_JSON_DEPTH} levels")
        elif ch in "]}":
            depth -= 1


def load_manifest(path: str | Path) -> ExaminedManifest:
    """JSON (see the module docstring) or CSV (columns: file, page, counter | field | row+col, count, counted_zero, note)
    with the mandatory evidence text passed in a ``# evidence: ...`` first line. Every failure is an
    :class:`ExaminedError` (file too large, undecodable, too deeply nested, wrong schema, wrong types, unknown keys)."""
    import hashlib
    from polony_app.ml.common import DuplicateKeyError as _DuplicateKey      # lazy: ml imports this module
    p = Path(path)
    try:
        size = p.stat().st_size
        if size > MAX_MANIFEST_FILE_BYTES:
            raise ExaminedError(f"{p}: the manifest is larger than {MAX_MANIFEST_FILE_BYTES // (1024 * 1024)} MB")
        raw = p.read_bytes()
    except OSError as exc:
        raise ExaminedError(f"{p}: cannot read the manifest: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
        if p.suffix.lower() == ".csv":
            m = _from_csv(text)
        else:
            _json_depth_ok(text)
            m = ExaminedManifest.model_validate(json.loads(text, object_pairs_hook=_no_duplicate_keys))
    except ExaminedError:
        raise
    except _DuplicateKey as exc:
        raise ExaminedError(f"{p}: not a valid examined-fields manifest: the key {exc.key!r} appears more than once in one JSON "
                            "object (e.g. 'count' twice); refusing to guess which statement counts") from exc
    except (ValidationError, ValueError, KeyError, TypeError, AttributeError, csv.Error) as exc:
        raise ExaminedError(f"{p}: not a valid examined-fields manifest: {exc}") from exc
    except (RecursionError, MemoryError) as exc:
        raise ExaminedError(f"{p}: not a valid examined-fields manifest: too deeply nested or too large to read") from exc
    m.sha256 = hashlib.sha256(raw).hexdigest()
    return m


CSV_COLUMNS = ("file", "page", "counter", "field", "row", "col", "count", "counted_zero", "note")


def _from_csv(text: str) -> ExaminedManifest:
    lines = text.splitlines()
    evidence, kind = "", SRC_SIDECAR_CSV
    while lines and lines[0].startswith("#"):
        k, _, v = lines.pop(0)[1:].partition(":")
        if k.strip() == "evidence":
            evidence = v.strip()
        elif k.strip() == "source_kind":
            kind = v.strip()
    files: dict[str, dict[str, list[FieldEntry]]] = {}
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    unknown = [c for c in (reader.fieldnames or []) if c not in CSV_COLUMNS]
    if unknown:
        raise ExaminedError(f"unknown CSV column(s) {unknown}; allowed: {list(CSV_COLUMNS)}")
    for r in reader:
        if None in r:
            raise ExaminedError("a CSV row has more cells than the header")
        zero = (r.get("counted_zero") or "").strip().lower()
        if zero not in ("", "0", "1", "true", "false", "yes", "no"):
            raise ExaminedError(f"counted_zero must be one of 1/true/yes/0/false/no/blank, got {zero!r}")

        def num(name):
            v = (r.get(name) or "").strip()
            return int(v) if v != "" else None
        e = FieldEntry(counter=num("counter"), field=num("field"), row=num("row"), col=num("col"), count=num("count"),
                       counted_zero=zero in ("1", "true", "yes"),
                       note=(r.get("note") or "").strip())
        files.setdefault(r["file"].strip(), {}).setdefault(str(int(r["page"])), []).append(e)
    return ExaminedManifest(evidence=evidence, source_kind=kind,
                            files=[FileEntry(file=f, pages=pg) for f, pg in sorted(files.items())])


@dataclass(frozen=True)
class Examination:
    status: str                              # positive_marked | verified_zero | unknown | conflict
    n_marks: int
    sources: tuple = ()                      # evidence kinds that support it (examination_source)
    evidence: str = ""                       # human-readable proof
    reason: str = ""                         # why unknown / conflict
    stated_count: Optional[int] = None

    @property
    def researcher_examined(self) -> bool:
        return self.status in EXAMINED

    @property
    def examination_source(self) -> str:
        return "+".join(self.sources)


def classify(n_marks: int, entry: Optional[FieldEntry], *, mode: str = MODE_MARKED_ONLY,
             manifest: Optional[ExaminedManifest] = None) -> Examination:
    """Examination status of ONE (source TIFF, field, page) tuple.

    ``n_marks`` = usable embedded researcher marks of THAT page inside THAT field (never marks of another page).
    ``entry`` = the manifest/sidecar entry for exactly this tuple (or None).
    ``marked-only``: only embedded marks are evidence (the manifest, if any, may CONTRADICT but never add fields).
    ``examined-fields-manifest``: only tuples listed in the manifest are candidates; marks complete the evidence."""
    if mode not in MODES:
        raise ExaminedError(f"mode must be one of {MODES}")
    kind = (manifest.source_kind if manifest else SRC_MANIFEST)
    ev_text = manifest.evidence if manifest else ""
    stated = entry.stated_count if entry is not None else None
    if mode == MODE_MARKED_ONLY:
        if n_marks == 0:
            return Examination(UNKNOWN, 0, reason="no researcher marks on this page: not proven examined (never read as zero)")
        if stated is not None and stated != n_marks:
            return Examination(CONFLICT, n_marks, (SRC_EMBEDDED, kind), ev_text, f"stated count {stated} differs from {n_marks} "
                               "embedded mark(s): the marks are incomplete or extra", stated)
        return Examination(POSITIVE, n_marks, (SRC_EMBEDDED,) + ((kind,) if entry is not None else ()),
                           f"{n_marks} embedded ImageJ mark(s) on this page" + (f"; {ev_text}" if entry is not None and ev_text else ""),
                           stated_count=stated)
    # examined-fields-manifest
    if entry is None:
        return Examination(UNKNOWN, n_marks, reason="not listed in the examined-fields manifest" + (
            "; marks exist but a field is only evidence when the manifest lists it" if n_marks else ""))
    if n_marks > 0:
        if stated is not None and stated != n_marks:
            return Examination(CONFLICT, n_marks, (SRC_EMBEDDED, kind), ev_text, f"stated count {stated} differs from {n_marks} "
                               "embedded mark(s)", stated)
        return Examination(POSITIVE, n_marks, (SRC_EMBEDDED, kind), f"{n_marks} embedded mark(s); listed in the manifest: {ev_text}",
                           stated_count=stated)
    if stated == 0:
        return Examination(VERIFIED_ZERO, 0, (kind,), f"counted zero: {ev_text}", stated_count=0)
    if stated is None:
        return Examination(UNKNOWN, 0, reason="listed as examined but no count is stated and there are no marks: zero is not proven")
    return Examination(CONFLICT, 0, (kind,), ev_text, f"stated count {stated} but no embedded marks on this page", stated)


@dataclass
class ExaminationTally:
    """Counts of (field, page) tuples by status, for dry-run reports and CSVs."""
    positive_marked: int = 0
    verified_zero: int = 0
    unknown: int = 0
    conflict: int = 0
    reasons: dict = field(default_factory=dict)

    def add(self, ex: Examination) -> None:
        setattr(self, ex.status, getattr(self, ex.status) + 1)
        if ex.reason:
            self.reasons[ex.reason] = self.reasons.get(ex.reason, 0) + 1

    @property
    def eligible_examined(self) -> int:
        return self.positive_marked + self.verified_zero

    @property
    def excluded_unproven(self) -> int:
        return self.unknown + self.conflict

    def as_dict(self) -> dict:
        return {"eligible_examined_fields": self.eligible_examined, "positive_marked_fields": self.positive_marked,
                "verified_zero_fields": self.verified_zero, "unknown_fields_excluded": self.unknown,
                "conflict_fields_excluded": self.conflict, "reasons": dict(sorted(self.reasons.items()))}
