"""Canonical, versioned schema for historical (human-made) polony annotations.

On-disk layout of a dataset directory (all text; nothing here contains pixel data)::

    manifest.json        schema name/version, dataset_id, importer config, counts
    images.jsonl         one ImageRecord per source image (identity, hashes, provenance)
    points.csv           one row per human point (canonical + raw ImageJ coordinates)
    fields.csv           grid cells ("fields") of each image, with polygon + counters
    regions.jsonl        other ROI shapes found (never silently typed as valid/excluded)
    reconciliations.jsonl  comparison of ROI counts with the researcher's CSV rows
    warnings.jsonl       every conversion warning / failure with its source file

Coordinate convention (identical to ``polony_app.schemas``): image pixel coordinates,
``x`` = column, ``y`` = row, origin = centre of the top-left pixel, so pixel (r, c) has
centre (x=c, y=r).  ImageJ's frame is shifted by half a pixel (pixel i spans
[i, i+1), centre i+0.5); the raw ImageJ values are kept next to the canonical ones
(``x_imagej`` / ``y_imagej``) so nothing is lost.

Floats are written with ``repr`` so a write/read round trip is bit-exact.
"""
from __future__ import annotations

import csv
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

SCHEMA_NAME = "polony-historical-dataset"
SCHEMA_VERSION = 1


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class WarningRecord(BaseModel):
    code: str
    severity: Severity
    message: str
    file: Optional[str] = None          # path relative to the import root
    image_id: Optional[str] = None


class PlaneInfo(BaseModel):
    page: int                            # 0-based plane index inside the stored file
    label_hint: Optional[str] = None     # raw metadata text (e.g. "635 [W1]"); a HINT only
    min: int = 0
    max: int = 0
    saturated_fraction: float = 0.0      # fraction of pixels at the dtype maximum


class ImageKind(str, Enum):
    stack = "stack"                      # multi-plane (z-slice or channel) grayscale TIFF
    single_plane = "single_plane"
    rgb_composite = "rgb_composite"      # rendered composite: raw intensities NOT available


class ImageRecord(BaseModel):
    image_id: str                        # "img_" + first 12 hex of sha256
    file_name: str
    rel_path: str                        # relative to the import root (provenance)
    alias_paths: list[str] = []          # byte-identical copies found elsewhere
    sha256: str
    size_bytes: int
    kind: ImageKind
    width: int
    height: int
    n_planes: int
    dtype: str
    bit_depth: int
    axes: str = ""
    planes: list[PlaneInfo] = []
    info_text: Optional[str] = None      # ImageJ "Info" string, verbatim
    scan_settings: dict[str, str] = {}   # raw scanner fields (DateTime, PMT, LaserPower)
    resolution_note: Optional[str] = None  # TIFF resolution tag; informational, never applied
    # --- identity (all INFERRED from the file name unless identity_confirmed) ---
    assay: Optional[str] = None
    specimen_id: Optional[str] = None    # biological sample code, e.g. "AM"
    replicate: Optional[str] = None      # slide replicate letter, e.g. "A"
    slide_id: Optional[str] = None       # lab slide id, e.g. "AM_A"
    file_date: Optional[str] = None      # date in the file name, as written (dd-mm-yyyy)
    filename_suffix: Optional[str] = None
    scan_session: Optional[str] = None   # derived from scanner DateTime date + settings
    identity_source: str = "unknown"     # "filename_pattern" | "unknown" | "manual"
    identity_confirmed: bool = False
    reserved_split: Optional[str] = None  # e.g. "test": never usable for training
    reserved_reason: Optional[str] = None
    # --- composite pairing ---
    composite_of: Optional[str] = None   # image_id of the raw stack this RGB composite renders
    composite_evidence: dict[str, float] = {}  # rank correlations colour plane vs stack page
    has_annotations: bool = False
    n_roi_blobs: int = 0
    grid_found: bool = False


class PointStatus(str, Enum):
    ok = "ok"
    duplicate = "duplicate"              # same plane, within the duplicate tolerance of an earlier point
    out_of_bounds = "out_of_bounds"      # outside the image: kept for provenance, never clamped


class PointLabel(BaseModel):
    label_id: str                        # "<image_id>:<roi_index>:<point_index>"
    image_id: str
    source_file: str                     # file that carries the ROI (image or sidecar), relative path
    roi_source: str                      # "ROI" | "Overlays" | "sidecar"
    roi_index: int
    point_index: int
    x: float                             # canonical (pixel centre = integer)
    y: float
    x_imagej: float                      # raw ImageJ ROI coordinates (pixel edge frame)
    y_imagej: float
    subpixel: bool                       # ROI carried sub-pixel resolution coordinates
    counter: Optional[int] = None        # ImageJ multi-point counter (0..255); see docs
    position: Optional[int] = None       # ImageJ 1-based plane position; 0/None = unspecified
    page: Optional[int] = None           # 0-based plane = position - 1 (None if unspecified)
    field_id: Optional[str] = None       # grid cell containing the point (geometric)
    class_label: Optional[str] = None    # only when the user supplied an explicit class map
    class_confirmed: bool = False
    status: PointStatus = PointStatus.ok


class FieldStatus(str, Enum):
    has_points = "has_points"            # counted: at least one human point inside
    no_points = "no_points"              # NOT counted, or counted with zero: unknowable from the ROI


class FieldRecord(BaseModel):
    field_id: str                        # "<image_id>:r<row>c<col>"
    image_id: str
    grid_row: int
    grid_col: int
    polygon: list[tuple[float, float]]   # 4 corners, canonical coordinates
    counters: list[int] = []             # ImageJ counters whose majority cell this is
    n_points_by_page: dict[str, int] = {}  # "0" -> n; "unspecified" for page None
    status: FieldStatus = FieldStatus.no_points


class RegionRecord(BaseModel):
    region_id: str
    image_id: str
    roi_type: str
    name: Optional[str] = None
    polygon: list[tuple[float, float]]   # canonical coordinates
    role: Optional[str] = None           # "valid" | "excluded" ONLY when named in the import config
    role_source: Optional[str] = None
    position: Optional[int] = None
    source_file: str = ""
    roi_source: str = ""
    roi_index: int = 0


class PlaneMatch(BaseModel):
    position: int                        # ImageJ plane position (1-based)
    roi_total: int
    match: str                           # exact_by_counter_index | exact_by_present_order | total_only | mismatch
    first_mismatch_field: Optional[int] = None


class Reconciliation(BaseModel):
    """Evidence only: how the ROI counts relate to one CSV row. Never used to relabel."""
    image_id: str
    slide_id: str
    csv_file: str
    csv_sha256: str
    series: str                          # e.g. "T7A" (file name prefix; meaning unconfirmed)
    csv_total: int
    csv_counted_fields: int
    csv_blank_fields: int
    planes: list[PlaneMatch] = []


class DatasetManifest(BaseModel):
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION
    dataset_id: str = ""
    created_at: str = ""
    importer: str = "polony_app.legacy_import"
    source_root: str = ""
    config: dict = {}
    counts: dict[str, int] = {}
    notes: list[str] = []


class Dataset(BaseModel):
    manifest: DatasetManifest = Field(default_factory=DatasetManifest)
    images: list[ImageRecord] = []
    points: list[PointLabel] = []
    fields: list[FieldRecord] = []
    regions: list[RegionRecord] = []
    reconciliations: list[Reconciliation] = []
    warnings: list[WarningRecord] = []

    # ---------------------------------------------------------------- queries
    def image(self, image_id: str) -> ImageRecord:
        for im in self.images:
            if im.image_id == image_id:
                return im
        raise KeyError(image_id)

    def points_of(self, image_id: str, *, page: Optional[int] = None, only_ok: bool = True) -> list[PointLabel]:
        return [
            p for p in self.points
            if p.image_id == image_id
            and (not only_ok or p.status == PointStatus.ok)
            and (page is None or p.page == page)
        ]

    def fields_of(self, image_id: str) -> list[FieldRecord]:
        return [f for f in self.fields if f.image_id == image_id]


# ---------------------------------------------------------------- identity / versioning
def compute_dataset_id(ds: Dataset) -> str:
    """Content hash of images + labels (not of created_at or free-text notes)."""
    payload = json.dumps(
        {
            "images": [i.model_dump(mode="json") for i in ds.images],
            "points": [p.model_dump(mode="json") for p in ds.points],
            "fields": [f.model_dump(mode="json") for f in ds.fields],
            "regions": [r.model_dump(mode="json") for r in ds.regions],
        },
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- IO
_POINT_COLS = [
    "label_id", "image_id", "source_file", "roi_source", "roi_index", "point_index",
    "x", "y", "x_imagej", "y_imagej", "subpixel", "counter", "position", "page",
    "field_id", "class_label", "class_confirmed", "status",
]
_FIELD_COLS = ["field_id", "image_id", "grid_row", "grid_col", "polygon", "counters",
               "n_points_by_page", "status"]


def _s(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, Enum):
        return str(v.value)
    return str(v)


def _opt_int(s: str) -> Optional[int]:
    return int(s) if s != "" else None


def _opt_str(s: str) -> Optional[str]:
    return s if s != "" else None


def write_dataset(ds: Dataset, out_dir: Path) -> Path:
    """Write ``ds`` into ``out_dir`` (created; must not already contain a manifest)."""
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already contains a dataset (manifest.json); refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.manifest.dataset_id = ds.manifest.dataset_id or compute_dataset_id(ds)

    def jsonl(name: str, rows: list[BaseModel]) -> None:
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n")

    jsonl("images.jsonl", ds.images)
    jsonl("regions.jsonl", ds.regions)
    jsonl("reconciliations.jsonl", ds.reconciliations)
    jsonl("warnings.jsonl", ds.warnings)
    with open(out_dir / "points.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_POINT_COLS)
        for p in ds.points:
            w.writerow([_s(getattr(p, c)) for c in _POINT_COLS])
    with open(out_dir / "fields.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELD_COLS)
        for fl in ds.fields:
            w.writerow([
                fl.field_id, fl.image_id, fl.grid_row, fl.grid_col,
                json.dumps([list(map(float, q)) for q in fl.polygon]),
                json.dumps(fl.counters), json.dumps(fl.n_points_by_page, sort_keys=True),
                fl.status.value,
            ])
    # manifest last: its presence marks a complete dataset
    (out_dir / "manifest.json").write_text(
        json.dumps(ds.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir


def read_dataset(path: Path) -> Dataset:
    path = Path(path)
    mf = path / "manifest.json"
    if not mf.is_file():
        raise FileNotFoundError(f"{path}: no manifest.json (not a dataset directory, or an incomplete write)")
    manifest = DatasetManifest.model_validate_json(mf.read_text(encoding="utf-8"))
    if manifest.schema_name != SCHEMA_NAME:
        raise ValueError(f"{path}: schema '{manifest.schema_name}' is not '{SCHEMA_NAME}'")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema version {manifest.schema_version} != supported {SCHEMA_VERSION}")

    def jsonl(name: str, model: type[BaseModel]) -> list:
        p = path / name
        if not p.is_file():
            return []
        return [model.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    points: list[PointLabel] = []
    with open(path / "points.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            points.append(PointLabel(
                label_id=r["label_id"], image_id=r["image_id"], source_file=r["source_file"],
                roi_source=r["roi_source"], roi_index=int(r["roi_index"]), point_index=int(r["point_index"]),
                x=float(r["x"]), y=float(r["y"]), x_imagej=float(r["x_imagej"]), y_imagej=float(r["y_imagej"]),
                subpixel=r["subpixel"] == "true", counter=_opt_int(r["counter"]),
                position=_opt_int(r["position"]), page=_opt_int(r["page"]),
                field_id=_opt_str(r["field_id"]), class_label=_opt_str(r["class_label"]),
                class_confirmed=r["class_confirmed"] == "true", status=PointStatus(r["status"]),
            ))
    fields: list[FieldRecord] = []
    with open(path / "fields.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fields.append(FieldRecord(
                field_id=r["field_id"], image_id=r["image_id"], grid_row=int(r["grid_row"]),
                grid_col=int(r["grid_col"]), polygon=[tuple(q) for q in json.loads(r["polygon"])],
                counters=json.loads(r["counters"]), n_points_by_page=json.loads(r["n_points_by_page"]),
                status=FieldStatus(r["status"]),
            ))
    return Dataset(
        manifest=manifest,
        images=jsonl("images.jsonl", ImageRecord),
        points=points, fields=fields,
        regions=jsonl("regions.jsonl", RegionRecord),
        reconciliations=jsonl("reconciliations.jsonl", Reconciliation),
        warnings=jsonl("warnings.jsonl", WarningRecord),
    )
