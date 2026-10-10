"""Shared typed contracts. OWNED BY THE ORCHESTRATOR: agents must not edit this
file; request changes via the orchestrator. All coordinates are image pixel
coordinates, float, origin = top-left corner of the top-left pixel centre
(i.e. pixel (0,0) has centre (0.0, 0.0)); `x` = column, `y` = row.
Regions use the same space as the (post-registration) reference channel.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- images
class ChannelRole(str, Enum):
    red = "red"
    green = "green"
    ignore = "ignore"


class ChannelInfo(BaseModel):
    index: int                      # position in the source (page / file)
    source_file: str                # stored file name (under data/originals)
    page: int = 0                   # TIFF page inside source_file
    label_hint: Optional[str] = None  # raw metadata text e.g. "635 [W1]"; a HINT only
    role: Optional[ChannelRole] = None  # None until user confirms mapping
    suggested_role: Optional[ChannelRole] = None  # pairing hint only (e.g. upload slot); never confirmed
    role_confirmed: bool = False
    min: int = 0
    max: int = 0
    saturated_fraction: float = 0.0   # fraction of pixels at dtype max


class ImageKind(str, Enum):
    raw_two_channel = "raw_two_channel"      # multipage 2x16-bit
    raw_separate_files = "raw_separate_files"  # red & green in separate TIFFs
    rgb_composite = "rgb_composite"          # composite only: NO raw channels


class ImageMeta(BaseModel):
    kind: ImageKind
    height: int
    width: int
    dtype: str                       # e.g. "uint16"
    bit_depth: int
    sha256: dict[str, str]           # stored_file_name -> sha256 of original bytes
    channels: list[ChannelInfo]
    pixel_size_um: Optional[float] = None   # None = unknown (never fabricated)
    tiff_info_text: Optional[str] = None    # ImageJ "Info" string if present
    embedded_rois: list["EmbeddedRoi"] = []
    warnings: list[str] = []
    composite_only: bool = False     # True => detection of per-channel signal is
                                     # an approximation; UI must say so


class EmbeddedRoi(BaseModel):
    """ROI found inside the TIFF (ImageJ). Meaning is NOT assumed."""
    source: Literal["ROI", "Overlays"]
    roi_type: str
    name: Optional[str] = None
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    n_points: Optional[int] = None
    c_position: int = 0   # raw ImageJ position fields, not interpreted
    z_position: int = 0
    t_position: int = 0


# ---------------------------------------------------------------- regions
class RegionKind(str, Enum):
    counting = "counting"      # valid area to count in (polygon or rect)
    exclusion = "exclusion"    # excluded from counting (edges, debris...)


class Region(BaseModel):
    id: str
    kind: RegionKind
    name: str
    # Closed polygon [[x,y],...]; rectangle = 4 points. Pixel coords.
    polygon: list[tuple[float, float]]
    # Review unit for approval: only `counting` regions are approvable.
    approved: bool = False
    approved_at: Optional[str] = None
    # hash of (polygon, overlapping exclusions, detections state) at approval;
    # used to invalidate approvals when anything affecting the region changes.
    approval_fingerprint: Optional[str] = None
    inspected: bool = False    # user ticked "I inspected this region for omissions"


# ---------------------------------------------------------------- params
class BackgroundParams(BaseModel):
    method: Literal["none", "gaussian_subtract", "white_tophat"] = "gaussian_subtract"
    sigma_px: float = 50.0       # for gaussian_subtract (large vs. spot size)
    radius_px: int = 25          # for white_tophat
    clip_negative: bool = True


class DetectionParams(BaseModel):
    """Per-channel Big-FISH settings; pixel units (no fabricated calibration)."""
    spot_radius_px: float = 5.0          # expected spot radius, pixels
    minimum_distance_px: float = 5.0     # min separation between spots
    threshold: Optional[float] = None    # None => Big-FISH automated threshold
    threshold_scale: float = 1.0         # multiply the auto threshold (manual tune)
    background: BackgroundParams = BackgroundParams()
    dense_decomposition: bool = False    # off by default; see docs
    remove_duplicate_radius_px: float = 3.0
    tile_size: int = 0                   # 0 = whole image; else tiled w/ overlap
    tile_overlap_px: int = 64


class DetectionRequest(BaseModel):
    red: DetectionParams = DetectionParams()
    green: DetectionParams = DetectionParams()
    # optional restriction to a preview window [x0,y0,x1,y1]; None = all
    # counting regions of the analysis.
    preview_window: Optional[tuple[int, int, int, int]] = None
    match_tolerance_px: float = 6.0      # cross-channel association distance


class ReportingMode(str, Enum):
    exclusive = "exclusive"                  # red_only / green_only / yellow
    channel_inclusive = "channel_inclusive"  # red_pos / green_pos incl. dual; yellow = overlap


# ---------------------------------------------------------------- objects
class ObjectClass(str, Enum):
    red = "red"            # red-only
    green = "green"        # green-only
    yellow = "yellow"      # dual-positive (matched red+green evidence)
    uncertain = "uncertain"  # unresolved; never counted in colour totals


class ChannelEvidence(BaseModel):
    detected: bool                      # a Big-FISH spot exists in this channel
    x: Optional[float] = None
    y: Optional[float] = None
    peak_intensity: Optional[float] = None       # background-corrected peak
    local_background: Optional[float] = None
    snr: Optional[float] = None                  # peak / local bg std (measured)
    saturated: bool = False
    spot_id: Optional[str] = None


class Origin(str, Enum):
    auto = "auto"
    manual = "manual"       # added by user
    edited = "edited"       # auto object later moved / reclassified


class ReviewReason(str, Enum):
    weak_signal = "weak_signal"
    ambiguous_match = "ambiguous_match"
    dense_cluster = "dense_cluster"
    saturated = "saturated"
    suspicious_shape = "suspicious_shape"
    near_edge = "near_edge"
    near_threshold = "near_threshold"
    debris_like = "debris_like"   # compact/bright in both scales (scale-ratio filter); flagged, never auto-removed


class DetectedObject(BaseModel):
    id: str
    x: float                            # object position (see docs: mean of
    y: float                            # matched channel positions or the single one)
    cls: ObjectClass
    origin: Origin = Origin.auto
    red: ChannelEvidence = ChannelEvidence(detected=False)
    green: ChannelEvidence = ChannelEvidence(detected=False)
    association_distance_px: Optional[float] = None
    # measured reasons for review; empty => no measurable concern
    review_reasons: list[ReviewReason] = []
    # candidate partner ids when association is ambiguous
    ambiguous_with: list[str] = []
    reviewed: bool = False              # user explicitly looked at / confirmed it
    deleted: bool = False               # soft delete (kept for history/undo)
    region_id: Optional[str] = None     # counting region containing the point
    history: list[str] = []             # human-readable provenance trail


# ---------------------------------------------------------------- review
class ActionType(str, Enum):
    add = "add"
    delete = "delete"
    move = "move"
    reclassify = "reclassify"
    merge = "merge"
    split = "split"
    confirm = "confirm"             # mark reviewed / confirm dual-positive
    accept_region = "accept_region"  # accept all proposals in region
    region_edit = "region_edit"
    region_inspected = "region_inspected"
    approve_region = "approve_region"
    unapprove_region = "unapprove_region"
    rerun_detection = "rerun_detection"
    finalize = "finalize"
    reopen = "reopen"


class ReviewAction(BaseModel):
    seq: int
    ts: str
    type: ActionType
    payload: dict           # type-specific, JSON-serialisable; includes enough
    inverse: dict           # to undo (inverse payload) -> undo/redo stack


# ---------------------------------------------------------------- analysis
class AnalysisStatus(str, Enum):
    created = "created"
    needs_channel_mapping = "needs_channel_mapping"
    ready = "ready"            # mapping confirmed, no detection yet
    detecting = "detecting"
    in_review = "in_review"
    finalized = "finalized"    # all counting regions approved + finalized


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class Job(BaseModel):
    id: str
    analysis_id: str
    status: JobStatus
    progress: float = 0.0           # 0..1
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class SlideMeta(BaseModel):
    """Explicit scientific inputs. None = not provided (gates concentration)."""
    slide_id: Optional[str] = None
    sample_volume_ul: Optional[float] = None
    concentration_factor: Optional[float] = None   # dilution/concentration
    sampled_area_note: Optional[str] = None
    counting_area_px2: Optional[float] = None      # computed from regions
    pixel_size_um: Optional[float] = None
    concentration_formula_confirmed: bool = False  # researcher must confirm formula+geometry; gates ALL concentration output
    csv_color_mapping: Optional[dict[str, str]] = None  # e.g. {"T7A": "red"} ONLY if user set


class Provenance(BaseModel):
    schema_version: int = SCHEMA_VERSION
    app_version: str = ""
    engine: str = "bigfish"
    package_versions: dict[str, str] = {}
    detection_request: Optional[DetectionRequest] = None
    detection_ran_at: Optional[str] = None
    reporting_mode: ReportingMode = ReportingMode.exclusive
    stale_detection: bool = False   # True if params/regions changed since detection


class Analysis(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    status: AnalysisStatus
    image: Optional[ImageMeta] = None
    regions: list[Region] = []
    objects: list[DetectedObject] = []
    reporting_mode: ReportingMode = ReportingMode.exclusive
    detection_request: DetectionRequest = DetectionRequest()
    provenance: Provenance = Provenance()
    slide: SlideMeta = SlideMeta()
    finalized_at: Optional[str] = None
    undo_depth: int = 0
    redo_depth: int = 0
    alignment_dy: int = 0   # integer shift applied to green to match red (explicit user step)
    alignment_dx: int = 0


class Counts(BaseModel):
    """Counts for one scope (region or whole slide). Uncertain are never in colour totals."""
    scope: str                       # region id or "slide"
    red_only: int = 0
    green_only: int = 0
    yellow: int = 0
    uncertain: int = 0
    red_positive: int = 0            # = red_only + yellow
    green_positive: int = 0          # = green_only + yellow
    total_objects_exclusive: int = 0  # red_only+green_only+yellow
    reporting_mode: ReportingMode
    is_final: bool = False


# ---------------------------------------------------------------- CSV import
class CsvFieldCell(BaseModel):
    value: Optional[int]     # None = BLANK (not measured). 0 = measured zero.


class CsvRow(BaseModel):
    slide_id: str
    meta: dict[str, str]     # raw metadata columns, verbatim strings
    fields: list[Optional[int]]  # field 1..N, None = blank cell
    raw_fields: list[str]        # verbatim strings


class CsvImport(BaseModel):
    id: str
    filename: str
    sha256: str
    header: list[str]
    rows: list[CsvRow]
    color_mapping: Optional[str] = None   # user-set only: "red"/"green"/"yellow"/None
    notes: list[str] = []


# ---------------------------------------------------------------- errors
class ApiError(BaseModel):
    code: str
    message: str
    detail: Optional[dict] = None
