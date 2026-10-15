"""Shared typed contracts (pydantic models used by the API, the store and the exports;
mirrored by hand in frontend/src/api/types.ts). All coordinates are image pixel
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
    source_path: Optional[str] = None  # absolute path the scan was opened from ('Open from disk'); None for uploads


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
    field_index: Optional[int] = None   # lab "field" number (1-based) for counting regions; None = not a lab field
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


class EngineName(str, Enum):
    bigfish = "bigfish"
    spotiflow = "spotiflow"
    human = "human"


class EngineChoice(str, Enum):
    bigfish = "bigfish"      # default: current behaviour, Spotiflow never loaded
    spotiflow = "spotiflow"  # Spotiflow only (no consensus)
    hybrid = "hybrid"        # both engines independently + consensus candidates


class ConsensusStatus(str, Enum):
    """Spatial relation of the two engines' detections on one channel page.
    NOT a truth label: agreement is not ground truth."""
    agreement = "agreement"
    bigfish_only = "bigfish_only"
    spotiflow_only = "spotiflow_only"
    human_added = "human_added"
    human_rejected = "human_rejected"   # derived: candidate explicitly rejected (soft-deleted by a user)


class SpotiflowParams(BaseModel):
    model_name: str = "general"                 # official pretrained name or a registered fine-tuned model id
    prob_thresh: Optional[float] = None         # None = the model's own optimised threshold (recorded)
    min_distance: int = 1
    exclude_border: bool = False
    subpix: Optional[bool] = None               # None = model default
    refine_peaks: bool = True                   # quadratic refinement of peaks on the heatmap (lowers localisation error for downscale>1)
    n_tiles: Optional[tuple[int, int]] = None   # None = automatic (max_tile_size)
    max_tile_size: Optional[int] = None
    downscale: int = 1                          # integer image downscale divisor (1,2,4,8) for LARGE blobs: the pretrained models were trained on
                                                # near-diffraction-limited spots; polonies are ~25 px blobs. Default 1 = native scale (unchanged behaviour).
                                                # Spotiflow requires subpix=False when downscale>1 and image sides divisible by it (crop/pad is recorded).
    normalizer: Literal["auto", "none", "valid_unsaturated"] = "valid_unsaturated"   # DEFAULT "valid_unsaturated" (measured on 8 slides: removes a per-slide sensitivity lottery; "auto" = Spotiflow's own percentile normalisation, saturation-sensitive, kept as an explicit legacy option);
                                                                  # "valid_unsaturated" = lo/hi from valid, unsaturated pixels of the page (recorded)  # 'auto' = Spotiflow's percentile normalisation of the page (recorded, never 8-bit conversion)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class DetectionRequest(BaseModel):
    red: DetectionParams = DetectionParams()
    green: DetectionParams = DetectionParams()
    # optional restriction to a preview window [x0,y0,x1,y1]; None = all
    # counting regions of the analysis.
    preview_window: Optional[tuple[int, int, int, int]] = None
    match_tolerance_px: float = 6.0      # cross-channel association distance
    engine: EngineChoice = EngineChoice.bigfish   # default keeps the existing Big-FISH-only behaviour
    spotiflow: SpotiflowParams = SpotiflowParams()
    consensus_tolerance_px: float = 4.0  # same-channel Big-FISH <-> Spotiflow matching tolerance (pixels)


class ReportingMode(str, Enum):
    exclusive = "exclusive"                  # red_only / green_only / yellow
    channel_inclusive = "channel_inclusive"  # red_pos / green_pos incl. dual; yellow = overlap



# ---------------------------------------------------------------- detection engines (two-engine architecture)
class EngineRunInfo(BaseModel):
    """One engine execution on one TIFF page. Raw detections reference run_id. Never mutated afterwards."""
    run_id: str
    analysis_id: str
    image_sha256: dict[str, str] = {}
    engine: EngineName
    engine_version: str = ""
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_checksum: Optional[str] = None        # sha256 of the weights artifact when available
    page: int
    channel_role: Optional[str] = None          # "red"/"green" ONLY from the user-confirmed mapping, else None
    source_file: Optional[str] = None           # stored file the page belongs to (channel index != TIFF page for two single-page files)
    source_page: Optional[int] = None           # real TIFF page inside source_file (``page`` above is the channel index)
    alignment_dy_dx: Optional[tuple[int, int]] = None  # shift applied to the green channel before detection (frame of the raw detections)
    mapping_fingerprint: Optional[str] = None   # hash of the confirmed channel roles in force when the run was made
    params: dict = {}
    preprocessing: dict = {}                    # every normalisation / transformation applied (documented, reversible)
    started_at: str = ""
    finished_at: Optional[str] = None
    device: Optional[str] = None
    duration_s: Optional[float] = None
    status: Literal["ok", "failed", "skipped"] = "ok"
    error: Optional[str] = None                 # clear failure text; a failed run never silently yields an empty result
    n_raw: int = 0
    n_valid: int = 0                            # inside the valid counting area (not excluded)
    notes: list[str] = []


class RawDetection(BaseModel):
    """Engine-specific raw output, preserved forever (auditable), in full-resolution canonical pixel coordinates
    (x = column, y = row, pixel centre at integer coordinates)."""
    id: str
    run_id: str
    analysis_id: str
    engine: EngineName
    page: int
    channel_role: Optional[str] = None
    source_file: Optional[str] = None
    source_page: Optional[int] = None
    frame: Literal["aligned_reference", "stored_page"] = "aligned_reference"   # coordinates are in the registered reference frame (green shifted by the run's alignment)
    x: float
    y: float
    z: Optional[float] = None
    radius_px: Optional[float] = None
    score: Optional[float] = None               # RAW engine score. NOT a calibrated probability / confidence %.
    score_kind: Optional[str] = None            # e.g. "bigfish_log_response_noise_units", "spotiflow_prob_raw"
    valid: bool = True                          # inside the valid counting area
    region_id: Optional[str] = None
    excluded_reason: Optional[str] = None


class ConsensusSummary(BaseModel):
    """Counts for the review summary. Per channel page; sparse channels must not be merged silently."""
    page: Optional[int] = None
    channel_role: Optional[str] = None
    agreement: int = 0
    bigfish_only: int = 0
    spotiflow_only: int = 0
    human_added: int = 0
    human_rejected: int = 0
    confirmed: int = 0
    rejected: int = 0
    unresolved: int = 0


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
    snr: Optional[float] = None                  # (peak - local bg) / local bg robust std (measured)
    saturated: bool = False
    spot_id: Optional[str] = None
    # two-engine provenance (all optional; absent for Big-FISH-only runs)
    consensus_status: Optional[ConsensusStatus] = None
    bigfish_raw_id: Optional[str] = None
    bigfish_x: Optional[float] = None
    bigfish_y: Optional[float] = None
    bigfish_score: Optional[float] = None       # raw LoG response in noise units (not a probability)
    spotiflow_raw_id: Optional[str] = None
    spotiflow_x: Optional[float] = None
    spotiflow_y: Optional[float] = None
    spotiflow_score: Optional[float] = None     # RAW Spotiflow "model score" (heatmap probability); not calibrated
    match_distance_px: Optional[float] = None   # distance between the two engines' points when matched


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
    bigfish_only = "bigfish_only"       # engine disagreement: reviewed before agreements
    spotiflow_only = "spotiflow_only"
    qc_sample = "qc_sample"             # random QC sample of agreement candidates (deterministic seed)
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
    done_with_failures = "done_with_failures"   # batch only: finished, but at least one slide failed (see child jobs)
    failed = "failed"
    cancelled = "cancelled"


class Job(BaseModel):
    id: str
    analysis_id: str                # slide id; a batch uses "experiment:<eid>", model preparation "engine-spotiflow"
    status: JobStatus
    progress: float = 0.0           # 0..1
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    kind: str = "detect"            # "detect" | "batch" | "prepare_model" (see JobKind)
    parent_id: Optional[str] = None       # detect job started by a batch: the batch job id
    experiment_id: Optional[str] = None
    detail: Optional[dict] = None   # batch only: the run plan (slides, skips, overrides)


class LabCountType(str, Enum):
    red_only = "red_only"
    green_only = "green_only"
    yellow = "yellow"
    red_positive = "red_positive"      # red_only + yellow
    green_positive = "green_positive"  # green_only + yellow


class LabSeries(BaseModel):
    """One count series exported in the lab layout (e.g. the researcher's 'T7A' / 'T7B' files).
    The user must choose which count type feeds each series: it is never inferred."""
    name: str                          # file/sheet name, e.g. "T7A"
    count_type: LabCountType
    sheet_title: Optional[str] = None  # title used in the Counts-style workbook (e.g. "Clade A")


class SkippedField(BaseModel):
    """A grid cell the researcher switched off ('not counted'): its geometry is kept so it can be switched back on."""
    field_index: int
    name: str
    polygon: list[tuple[float, float]]


class GridSpec(BaseModel):
    """Counting grid as the researcher generated it (rows x cols cells of cell_w x cell_h px). The cell polygons are
    derived from it (fields/grid.py `spec_cells`, mirrored by frontend/src/lib/labgrid.ts). Effective top-left corner
    = (origin_x + shift_dx, origin_y + shift_dy); `seed` is set when the shift was randomised (reproducible)."""
    rows: int
    cols: int
    cell_w: float
    cell_h: float
    origin_x: float
    origin_y: float
    order: Literal["row-major", "col-major"] = "row-major"   # numbering walks along rows first / down columns first
    corner: Literal["tl", "tr", "bl", "br"] = "tl"           # start corner of the numbering
    first_index: int = 1                                      # field number of the first cell
    seed: Optional[int] = None                                # seed of the random shift (None = not randomised)
    shift_dx: float = 0.0
    shift_dy: float = 0.0


class SlideMeta(BaseModel):
    """Explicit scientific inputs. None = not provided (gates concentration)."""
    slide_id: Optional[str] = None          # lab slide ID, e.g. "AI_A" (sample + "_" + replicate letter)
    sampling_date: Optional[str] = None     # dd/mm/yyyy, as in the lab files
    polony_date: Optional[str] = None       # dd/mm/yyyy
    qc: Optional[int] = None                # lab QC flag (0/1); meaning unconfirmed
    field_area: Optional[float] = None      # lab "field [Pixel^2]" value (template: 2.9)
    grid_area: Optional[float] = None       # lab "grid [Pixel^2]" value (template: 0.1)
    lab_series: list[LabSeries] = []        # count series to export in lab layout
    efficiency_input: Optional[float] = None  # denominator for 'Efficiency' (template: 1500); optional
    sample_volume_ul: Optional[float] = None
    concentration_factor: Optional[float] = None   # dilution/concentration
    sampled_area_note: Optional[str] = None
    counting_area_px2: Optional[float] = None      # computed from regions (net of overlaps)
    sample_total_area_px2: Optional[float] = None  # researcher-supplied: total area (same px units) that the sample volume spreads over; needed for any concentration estimate
    pixel_size_um: Optional[float] = None
    concentration_formula_confirmed: bool = False  # researcher must confirm formula+geometry; gates ALL concentration output
    csv_color_mapping: Optional[dict[str, str]] = None  # e.g. {"T7A": "red"} ONLY if user set
    skipped_fields: list[SkippedField] = []   # grid cells switched off (blank in the CSVs); see /grid-selection
    grid: Optional[GridSpec] = None           # the applied counting grid; set only through PUT /grid


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
    experiment_id: Optional[str] = None     # experiment the slide belongs to (None = loose analysis)
    order_in_experiment: int = 0            # position of the slide in its experiment (1-based once attached)


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
    red_positive_excl_debris: int = 0    # red_positive minus objects carrying review reason debris_like (not a removal: a convenience count)
    green_positive_excl_debris: int = 0
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


# ---------------------------------------------------------------- experiments (stage 3)
class JobKind(str, Enum):
    detect = "detect"
    batch = "batch"                  # one unattended run over the slides of an experiment
    prepare_model = "prepare_model"  # Spotiflow model download / load


class ExperimentStatus(str, Enum):
    open = "open"
    running = "running"              # a batch run is active


class Experiment(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    notes: str = ""
    default_grid: Optional[dict] = None              # opaque grid spec (stage 1 GridSpec), applied by the grid tools
    default_detection: DetectionRequest = DetectionRequest()   # settings of an unattended run
    status: ExperimentStatus = ExperimentStatus.open


class ExperimentSummary(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    notes: str = ""
    status: ExperimentStatus = ExperimentStatus.open
    n_slides: int = 0
    n_detected: int = 0              # slides with a detection result (in review or finalized)
    n_finalized: int = 0
    last_batch: Optional[Job] = None
