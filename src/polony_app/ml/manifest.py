"""Typed manifest of a dataset snapshot (``datasets/<version>/manifest.json``) and the test lock."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

SNAPSHOT_SCHEMA = "polony-ml-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
COORDINATE_CONVENTION = ("image pixel coordinates; x = column, y = row; pixel centres at integer coordinates "
                         "(pixel (row r, col c) has its centre at x=c, y=r). Label CSV columns are 'y,x' "
                         "(Spotiflow order) in the coordinates of the exported crop, which is always at NATIVE resolution; "
                         "when the snapshot's downscale d > 1 the training / evaluation step block-mean downsamples the "
                         "crop and maps labels with x_s = (x - (d-1)/2) / d.")
# fields that do not enter the content hash (version itself, wall clock, tool version)
VERSION_EXCLUDES = ("version", "created_at", "app_version")


class SourceAnalysis(BaseModel):
    analysis_id: str
    name: str = ""
    status: str = ""
    review_status: str = ""              # finalized | region_approved | in_review ...
    tiff_sha256: dict[str, str] = {}     # stored file name -> sha256 recomputed from the bytes read
    image_kind: str = ""
    dtype: str = ""
    shape_hw: list[int] = []
    channel_mapping: Optional[dict[str, dict]] = None   # {"red": {"file":..,"page":..}, ...} ONLY when confirmed, else null
    alignment_dy_dx: list[int] = [0, 0]
    reviewers: list[str] = []
    reviewer_identity_source: str = "unknown"
    engine_runs: list[dict] = []         # engine/model versions of the source runs (when the store provides them)
    provenance_engine: str = ""
    identity: dict = {}                  # group identity used for the split (+ whether it is confirmed)
    group_id: Optional[str] = None
    split: Optional[str] = None
    n_regions_total: int = 0
    n_regions_fully_reviewed: int = 0
    n_crops: int = 0
    # per channel role: lo/hi of the PAGE (valid, unsaturated pixels; computed once, before crop / downscale) and how they
    # were obtained. Every crop of that page is normalised with exactly these numbers (train, eval-model, and the app).
    page_normalization: dict[str, dict] = {}
    skipped: Optional[str] = None        # why the whole analysis contributed nothing


class CropRecord(BaseModel):
    crop_id: str
    analysis_id: str
    region_id: str
    region_name: str = ""
    field_index: Optional[int] = None
    page: int
    channel_role: Optional[str] = None   # only from a confirmed mapping
    source_file: str
    source_sha256: str
    box_xyxy: list[int]                  # x0,y0,x1,y1 (exclusive upper bound) in the STORED PAGE's pixel frame
    box_reference_frame: list[int]       # same box in the app's reference (post-alignment) frame
    alignment_dy_dx: list[int] = [0, 0]  # shift that separates the two frames for this page (0,0 unless green was aligned)
    shape_hw: list[int]
    dtype: str
    n_positive: int = 0
    n_rejected: int = 0
    n_edge_dropped: int = 0              # positives whose centre lies outside the exported pixel footprint
    split: str = ""
    group_id: str = ""
    tif_sha256: str = ""
    csv_sha256: str = ""
    norm_lo: Optional[float] = None      # PAGE-level lo/hi of this crop's source page (None = legacy crop: unusable for training)
    norm_hi: Optional[float] = None
    normaliser: str = ""                 # 'valid_unsaturated' | 'none' ('' = legacy)
    downscale: int = 1                   # the factor the snapshot was built for; the exported crop stays at NATIVE resolution
    bbox_of_polygon: bool = False        # True only with --allow-bbox (crop is the bounding box of a non-rectangular region)
    frozen_from: Optional[str] = None    # snapshot version the (locked) test crop was copied from
    path_tif: str = ""
    path_csv: str = ""
    # source-specific provenance that the app path does not have (e.g. ImageJ position / counter / grid cell of a
    # historical crop, label status, coordinate frame). Empty for crops built from reviewed analyses.
    provenance: dict = {}


class SnapshotManifest(BaseModel):
    schema_name: str = SNAPSHOT_SCHEMA
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    version: str = ""
    created_at: str = ""
    app_version: str = ""
    coordinate_convention: str = COORDINATE_CONVENTION
    options: dict = {}
    sources: list[SourceAnalysis] = []
    crops: list[CropRecord] = []
    skipped_crops: list[dict] = []
    regions_summary: dict = {}
    label_counts: dict = {}
    splits: dict = {}                    # seed, level, fractions, group_to_split, ...
    test_lock: Optional[dict] = None     # {"lock_sha256":..., "snapshot":..., "groups":[...]} when a lock was applied
    files: dict[str, str] = {}           # relative path -> sha256 (points.csv, regions.jsonl, crops)
    warnings: list[str] = []
    unresolved_assumptions: list[str] = []
    # summary of a bridge import (e.g. ``ml import-legacy``): files discovered / accepted / rejected / deduplicated, pages,
    # fields and annotations by status, estimates. Empty for snapshots built from reviewed analyses.
    import_summary: dict = {}
    excluded_from_hash: list[str] = Field(default_factory=lambda: list(VERSION_EXCLUDES))

    def hash_payload(self) -> dict:
        d = self.model_dump(mode="json")
        for k in VERSION_EXCLUDES:
            d.pop(k, None)
        # fields added after the first snapshots existed: omitted while empty so older snapshots keep their version hash
        if not d.get("import_summary"):
            d.pop("import_summary", None)
        for c in d.get("crops", []):
            if not c.get("provenance"):
                c.pop("provenance", None)
        return d
