"""Typed manifest of a dataset snapshot (``datasets/<version>/manifest.json``) and the test lock."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

SNAPSHOT_SCHEMA = "polony-ml-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
COORDINATE_CONVENTION = ("image pixel coordinates; x = column, y = row; pixel centres at integer coordinates "
                         "(pixel (row r, col c) has its centre at x=c, y=r). Label CSV columns are 'y,x' "
                         "(Spotiflow order) in the coordinates of the exported crop.")
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
    bbox_of_polygon: bool = False        # True only with --allow-bbox (crop is the bounding box of a non-rectangular region)
    frozen_from: Optional[str] = None    # snapshot version the (locked) test crop was copied from
    path_tif: str = ""
    path_csv: str = ""


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
    excluded_from_hash: list[str] = Field(default_factory=lambda: list(VERSION_EXCLUDES))

    def hash_payload(self) -> dict:
        d = self.model_dump(mode="json")
        for k in VERSION_EXCLUDES:
            d.pop(k, None)
        return d
