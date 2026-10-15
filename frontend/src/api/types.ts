// Hand-mirrored from src/polony_app/schemas.py.
// Coordinates: image pixels, float, pixel (0,0) has its CENTRE at (0,0). x = column, y = row.

export type ChannelRole = 'red' | 'green' | 'ignore';
export type ImageKind = 'raw_two_channel' | 'raw_separate_files' | 'rgb_composite';

export interface ChannelInfo {
  index: number;
  source_file: string;
  page: number;
  label_hint: string | null;
  role: ChannelRole | null;
  suggested_role?: ChannelRole | null;
  role_confirmed: boolean;
  min: number;
  max: number;
  saturated_fraction: number;
}

export interface EmbeddedRoi {
  source: 'ROI' | 'Overlays';
  roi_type: string;
  name: string | null;
  left: number;
  top: number;
  right: number;
  bottom: number;
  n_points: number | null;
  c_position: number;
  z_position: number;
  t_position: number;
}

export interface ImageMeta {
  kind: ImageKind;
  height: number;
  width: number;
  dtype: string;
  bit_depth: number;
  sha256: Record<string, string>;
  channels: ChannelInfo[];
  pixel_size_um: number | null;
  tiff_info_text: string | null;
  embedded_rois: EmbeddedRoi[];
  warnings: string[];
  composite_only: boolean;
  /** absolute path the scan was opened from ('Open from disk'); null/absent for browser uploads */
  source_path?: string | null;
}

export type RegionKind = 'counting' | 'exclusion';

export interface Region {
  id: string;
  kind: RegionKind;
  name: string;
  polygon: [number, number][];
  approved: boolean;
  approved_at: string | null;
  approval_fingerprint: string | null;
  inspected: boolean;
  /** lab "field" number (1-based) for counting regions; null/undefined = not a lab field (blank in the lab export) */
  field_index?: number | null;
}

export interface BackgroundParams {
  method: 'none' | 'gaussian_subtract' | 'white_tophat';
  sigma_px: number;
  radius_px: number;
  clip_negative: boolean;
}

export interface DetectionParams {
  spot_radius_px: number;
  minimum_distance_px: number;
  threshold: number | null;
  threshold_scale: number;
  background: BackgroundParams;
  dense_decomposition: boolean;
  remove_duplicate_radius_px: number;
  tile_size: number;
  tile_overlap_px: number;
}

// ---- two-engine architecture (docs/app/TWO_ENGINE_CONTRACT.md) ----
export type EngineName = 'bigfish' | 'spotiflow' | 'human';
export type EngineChoice = 'bigfish' | 'spotiflow' | 'hybrid';
/** Spatial relation of the two engines' detections. NOT a truth label: agreement is not ground truth. */
export type ConsensusStatus = 'agreement' | 'bigfish_only' | 'spotiflow_only' | 'human_added' | 'human_rejected';

export interface SpotiflowParams {
  model_name: string;
  prob_thresh: number | null;
  min_distance: number;
  exclude_border: boolean;
  subpix: boolean | null;
  n_tiles: [number, number] | null;
  max_tile_size: number | null;
  /** integer image downscale divisor (1, 2, 4, 8) for ~25 px blobs; 1 = native scale (default) */
  downscale: 1 | 2 | 4 | 8;
  /** 'auto' = Spotiflow's own percentile normalisation (legacy; saturation-sensitive); 'valid_unsaturated' = p1/p99.8 of valid, unsaturated page pixels */
  normalizer: 'auto' | 'none' | 'valid_unsaturated';
  device: 'auto' | 'cpu' | 'cuda' | 'mps';
}

export const DEFAULT_SPOTIFLOW: SpotiflowParams = {
  model_name: 'general',
  prob_thresh: null,
  min_distance: 1,
  exclude_border: false,
  subpix: null,
  n_tiles: null,
  max_tile_size: null,
  downscale: 1,
  normalizer: 'auto',
  device: 'auto',
};

export interface DetectionRequest {
  red: DetectionParams;
  green: DetectionParams;
  preview_window: [number, number, number, number] | null;
  match_tolerance_px: number;
  engine?: EngineChoice;
  spotiflow?: SpotiflowParams;
  consensus_tolerance_px?: number;
}

export interface EngineRunInfo {
  run_id: string;
  analysis_id: string;
  image_sha256?: Record<string, string>;
  engine: EngineName;
  engine_version: string;
  model_name: string | null;
  model_version: string | null;
  model_checksum: string | null;
  page: number;
  channel_role: string | null;
  params?: Record<string, unknown>;
  preprocessing?: Record<string, unknown>;
  started_at: string;
  finished_at: string | null;
  device: string | null;
  duration_s: number | null;
  status: 'ok' | 'failed' | 'skipped';
  error: string | null;
  n_raw: number;
  n_valid: number;
  notes: string[];
}

export interface RawDetection {
  id: string;
  run_id: string;
  analysis_id: string;
  engine: EngineName;
  page: number;
  channel_role: string | null;
  x: number;
  y: number;
  z?: number | null;
  radius_px?: number | null;
  /** RAW engine score, never a probability: Big-FISH = LoG response in noise units, Spotiflow = model score */
  score: number | null;
  score_kind: string | null;
  valid: boolean;
  region_id: string | null;
  excluded_reason: string | null;
}

export interface ConsensusSummary {
  page: number | null;
  channel_role: string | null;
  agreement: number;
  bigfish_only: number;
  spotiflow_only: number;
  human_added: number;
  human_rejected: number;
  confirmed: number;
  rejected: number;
  unresolved: number;
}

/** Agreement QC: confirmed/rejected among the SAMPLED agreements and the exact 95 % interval of the agreement error rate. */
export interface QcStatus {
  sample_size: number;
  reviewed: number;
  pending: number;
  confirmed: number;
  rejected: number;
  population_agreements: number;
  seed: number | string | null;
  error_rate: number | null;
  ci95: [number, number] | null;
  interval_method: string;
  n_for_margin_0_05: number;
  n_for_margin_0_05_conservative: number;
  planning_error_rate: number;
  note: string;
}

export interface QcSampleResponse {
  marked: number;
  newly_marked: number;
  eligible: number;
  n: number;
  n_requested: number;
  fraction: number | null;
  seed: number | string;
  population_agreements: number;
  n_for_margin_0_05: number;
  n_for_margin_0_05_conservative: number;
  qc: QcStatus;
  note: string;
}

export interface ConsensusSummaryResponse {
  pages: ConsensusSummary[];
  total: ConsensusSummary;
  qc?: QcStatus;
}

export interface EngineModelInfo {
  name: string;
  cached: boolean;
  version: string | null;
  checksum: string | null;
}

export interface EngineInfo {
  name: EngineName;
  available: boolean;
  reason: string | null;
  version: string | null;
  device?: { cuda?: boolean; mps?: boolean; cpu?: boolean } | null;
  torch_version?: string | null;
  device_selected?: string | null;
  device_note?: string | null;
  models?: EngineModelInfo[];
}

export type ReportingMode = 'exclusive' | 'channel_inclusive';
export type ObjectClass = 'red' | 'green' | 'yellow' | 'uncertain';

export interface ChannelEvidence {
  detected: boolean;
  x: number | null;
  y: number | null;
  peak_intensity: number | null;
  local_background: number | null;
  snr: number | null;
  saturated: boolean;
  spot_id: string | null;
  // two-engine provenance (absent for Big-FISH-only runs)
  consensus_status?: ConsensusStatus | null;
  bigfish_raw_id?: string | null;
  bigfish_x?: number | null;
  bigfish_y?: number | null;
  /** raw LoG response in noise units (not a probability) */
  bigfish_score?: number | null;
  spotiflow_raw_id?: string | null;
  spotiflow_x?: number | null;
  spotiflow_y?: number | null;
  /** raw Spotiflow "model score" (not calibrated, not a confidence) */
  spotiflow_score?: number | null;
  match_distance_px?: number | null;
}

export type Origin = 'auto' | 'manual' | 'edited';

export type ReviewReason =
  | 'weak_signal'
  | 'ambiguous_match'
  | 'dense_cluster'
  | 'saturated'
  | 'suspicious_shape'
  | 'near_edge'
  | 'near_threshold'
  | 'bigfish_only'
  | 'spotiflow_only'
  | 'qc_sample'
  | 'debris_like';

export interface DetectedObject {
  id: string;
  x: number;
  y: number;
  cls: ObjectClass;
  origin: Origin;
  red: ChannelEvidence;
  green: ChannelEvidence;
  association_distance_px: number | null;
  review_reasons: ReviewReason[];
  ambiguous_with: string[];
  reviewed: boolean;
  deleted: boolean;
  region_id: string | null;
  history: string[];
}

export type ActionType =
  | 'add'
  | 'delete'
  | 'move'
  | 'reclassify'
  | 'merge'
  | 'split'
  | 'confirm'
  | 'accept_region'
  | 'region_edit'
  | 'region_inspected'
  | 'approve_region'
  | 'unapprove_region'
  | 'rerun_detection'
  | 'finalize'
  | 'reopen';

export type AnalysisStatus =
  | 'created'
  | 'needs_channel_mapping'
  | 'ready'
  | 'detecting'
  | 'in_review'
  | 'finalized';

/** `done_with_failures`: batch only (finished, at least one slide failed). Terminal: everything but queued/running. */
export type JobStatus = 'queued' | 'running' | 'done' | 'done_with_failures' | 'failed' | 'cancelled';
export type JobKind = 'detect' | 'batch' | 'prepare_model';

/** The plan of a batch job (Job.detail when kind = 'batch'). */
export interface BatchPlanSlide {
  analysis_id: string;
  name: string;
  order: number;
  /** null = runs; 'finalized' | 'already_detected' = skipped */
  skip: string | null;
  override: boolean;
}
export interface BatchDetail {
  slides: BatchPlanSlide[];
  only_missing: boolean;
  default_detection?: Partial<DetectionRequest>;
  overrides?: Record<string, Partial<DetectionRequest>>;
}

export interface Job {
  id: string;
  /** the slide; `experiment:<eid>` for a batch; `engine-spotiflow` for prepare_model */
  analysis_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  kind?: JobKind;
  parent_id?: string | null;
  experiment_id?: string | null;
  detail?: BatchDetail | Record<string, unknown> | null;
}

// ---- experiments (docs/app/EXPERIMENTS_API.md) ----
export type ExperimentStatus = 'open' | 'running';

/** What the experiment page stores in `default_grid` (opaque to the server): the template slide's GridSpec plus the
 * field numbers it skips and the slide it came from. */
export interface ExperimentGrid extends GridSpec {
  skipped_fields?: number[];
  template_id?: string | null;
}

export interface Experiment {
  id: string;
  name: string;
  notes: string;
  created_at: string;
  updated_at: string;
  default_grid: ExperimentGrid | null;
  default_detection: Partial<DetectionRequest>;
  status: ExperimentStatus;
}

export interface ExperimentSummary extends Experiment {
  n_slides: number;
  n_detected: number;
  n_finalized: number;
  last_batch: Job | null;
}

export type SlideExportStatus = 'finalized' | 'approved' | 'draft';

export interface ExperimentSlide {
  id: string;
  name: string;
  order_in_experiment: number;
  status: AnalysisStatus;
  export_status: SlideExportStatus;
  updated_at: string;
  finalized_at: string | null;
  mapping_confirmed: boolean;
  image: { width: number; height: number; kind: ImageKind; dtype: string; n_channels: number } | null;
  has_grid: boolean;
  n_counting_regions: number;
  n_fields: number;
  n_approved: number;
  /** every counting region approved (false without counting regions) */
  approved: boolean;
  review_queue: number;
  detection_ran_at: string | null;
  engine: EngineChoice | null;
  counts: Counts;
  /** newest job of any kind on this slide (a failed batch slide shows its error here) */
  last_job: Job | null;
}

export interface ExperimentView extends Experiment {
  slides: ExperimentSlide[];
  last_batch: Job | null;
}

export type RunSlideStatus = 'pending' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'skipped' | 'not_run';

export interface RunSlide {
  analysis_id: string;
  name: string;
  order: number;
  status: RunSlideStatus;
  skip_reason: string | null;
  override: boolean;
  job_id: string | null;
  progress: number;
  message: string;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunSummary {
  /** excludes skipped */
  total: number;
  done: number;
  failed: number;
  cancelled: number;
  running: number;
  queued: number;
  pending: number;
  not_run: number;
  skipped: number;
}

export interface ExperimentRun {
  batch: Job | null;
  slides: RunSlide[];
  summary: RunSummary | null;
}

export interface ExperimentRunRequest {
  only_missing?: boolean;
  analysis_ids?: string[];
  overrides?: Record<string, Partial<DetectionRequest>>;
}

export interface ExperimentImportResult {
  created: { id: string; name: string; order_in_experiment: number; status: AnalysisStatus; mapping_copied_from: string | null }[];
  errors: { file: string; code: string; message: string }[];
}

export interface ExperimentDeleteResult {
  ok: boolean;
  deleted_slides: string[];
  detached_slides: string[];
}

export type LabCountType = 'red_only' | 'green_only' | 'yellow' | 'red_positive' | 'green_positive';

export interface LabSeries {
  name: string;
  count_type: LabCountType;
  sheet_title?: string | null;
}

export interface SlideMeta {
  slide_id: string | null;
  sampling_date?: string | null; // dd/mm/yyyy
  polony_date?: string | null; // dd/mm/yyyy
  qc?: number | null; // 0/1, meaning unconfirmed
  field_area?: number | null;
  grid_area?: number | null;
  lab_series?: LabSeries[];
  efficiency_input?: number | null;
  sample_volume_ul: number | null;
  concentration_factor: number | null;
  sampled_area_note: string | null;
  counting_area_px2: number | null;
  /** deprecated: no longer used by the lab formula */
  sample_total_area_px2?: number | null;
  concentration_formula_confirmed?: boolean;
  pixel_size_um: number | null;
  csv_color_mapping: Record<string, string> | null;
  /** grid fields switched off for counting (blank in the CSVs); geometry kept so they can be switched back on */
  skipped_fields?: SkippedField[];
  /** the applied counting grid (set only through PUT /grid); null/absent = no grid */
  grid?: GridSpec | null;
}

/** Counting grid as stored on the server (schemas.py GridSpec). Effective origin = origin + shift. */
export interface GridSpec {
  rows: number;
  cols: number;
  cell_w: number;
  cell_h: number;
  origin_x: number;
  origin_y: number;
  order: 'row-major' | 'col-major';
  corner: 'tl' | 'tr' | 'bl' | 'br';
  first_index: number;
  /** seed of the random shift; null = not randomised */
  seed: number | null;
  shift_dx: number;
  shift_dy: number;
}

/** PUT /analyses/{id}/grid. grid=null (without randomise) removes the grid. */
export interface GridRequest {
  grid: GridSpec | null;
  /** server picks a random shift inside the image and records the seed */
  randomise?: boolean;
  seed?: number | null;
  /** drop lab-field regions that are not part of the new grid (default true) */
  replace?: boolean;
  /** clearing: also remove approved grid fields */
  force?: boolean;
}

export interface Provenance {
  schema_version: number;
  app_version: string;
  engine: string;
  package_versions: Record<string, string>;
  detection_request: DetectionRequest | null;
  detection_ran_at: string | null;
  reporting_mode: ReportingMode;
  stale_detection: boolean;
}

export interface Analysis {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: AnalysisStatus;
  image: ImageMeta | null;
  regions: Region[];
  objects: DetectedObject[];
  reporting_mode: ReportingMode;
  detection_request: DetectionRequest;
  provenance: Provenance;
  slide: SlideMeta;
  finalized_at: string | null;
  undo_depth: number;
  redo_depth: number;
  alignment_dy?: number;
  alignment_dx?: number;
  experiment_id?: string | null;
  order_in_experiment?: number;
}

export interface Counts {
  scope: string;
  red_only: number;
  green_only: number;
  yellow: number;
  uncertain: number;
  red_positive: number;
  green_positive: number;
  /** convenience counts: debris_like-flagged objects are NOT removed from the main counts */
  red_positive_excl_debris?: number;
  green_positive_excl_debris?: number;
  total_objects_exclusive: number;
  reporting_mode: ReportingMode;
  is_final: boolean;
}

export interface AnalysisSummary {
  id: string;
  name: string;
  status: AnalysisStatus;
  updated_at: string;
  finalized_at: string | null;
  counts?: Counts | null;
  /** experiment this analysis is a slide of (null = loose analysis) */
  experiment_id?: string | null;
  order_in_experiment?: number;
}

export interface CountsResponse {
  regions: Counts[];
  slide: Counts;
}

export interface ReviewQueueItem {
  object_id: string;
  reasons: ReviewReason[];
  priority: number;
  x: number;
  y: number;
}

export interface CsvRow {
  slide_id: string;
  meta: Record<string, string>;
  fields: (number | null)[];
  raw_fields: string[];
}

export interface CsvImport {
  id: string;
  filename: string;
  sha256: string;
  header: string[];
  rows: CsvRow[];
  color_mapping: string | null;
  notes: string[];
}

export interface ApiErrorBody {
  code: string;
  message: string;
  detail?: Record<string, unknown> | null;
}

export type ImageView = 'red' | 'green' | 'composite' | 'processed_red' | 'processed_green';

export interface PreviewResponse {
  objects: DetectedObject[];
  notes: string[];
}

export interface HealthResponse {
  ok: boolean;
  version: string;
  bigfish_version: string;
}

// ---- researcher reference (marks embedded in the TIFF) ----
export interface ReferencePoint {
  x: number;
  y: number;
  counter: number;
  /** null = mark without a valid plane (missing, 0 or beyond the planes): evidence for no plane */
  plane: number | null;
  field: number | null;
}

export interface ReferenceGridCell {
  field: number;
  polygon: [number, number][];
}

export interface ReferenceGrid {
  source: string;
  origin: [number, number];
  cell: [number, number];
  cols: number;
  rows: number;
  cells: ReferenceGridCell[];
}

export interface ReferenceInfo {
  available: boolean;
  source_file?: string | null;
  points: ReferencePoint[];
  grid: ReferenceGrid | null;
  reference_counts: Record<string, { plane0?: number; plane1?: number; unplaned?: number }>;
  unplaned?: { count: number; fields: number[]; n_without_counter: number };
  planes: { page: number; point_count: number }[];
  warnings: string[];
}

export type ImportMode = 'counted' | 'all';

export interface FieldImportRequest {
  mode: ImportMode;
  shift: [number, number];
  plane: number | null;
  replace: boolean;
}

export type CompareCountType = 'red_positive' | 'green_positive' | 'red_only' | 'green_only' | 'yellow';

export type ExaminedStatus = 'positive_marked' | 'verified_zero' | 'unknown' | 'conflict';

export interface CompareRow {
  field: number;
  /** null when the field was not provably examined on this plane (never read as 0) */
  reference_count: number | null;
  app_count: number | null;
  delta: number | null;
  status: ExaminedStatus;
  examined: boolean;
  examination_source: string;
  region?: string;
  has_reference_marks?: boolean;
  reference_marks?: number;
  reason?: string | null;
}

export interface CompareSummary {
  /** all field regions */
  n_fields: number;
  eligible_examined_fields: number;
  positive_marked_fields: number;
  verified_zero_fields: number;
  unknown_fields_excluded: number;
  conflict_fields_excluded: number;
  /** e.g. 'embedded_imagej_marks', 'embedded_imagej_marks+sidecar_csv', 'none' */
  eligibility_source: string;
  /** over the eligible examined fields only; null when there are none */
  pearson_r: number | null;
  slope: number | null;
  mae: number | null;
  total_reference: number | null;
  total_app: number | null;
  app_total_unknown_fields_not_counted?: number;
  /** marks without a valid plane; their fields are conflict on every plane */
  unplaned_marks?: number;
  unplaned_mark_fields?: number[];
}

export interface CompareResponse {
  rows: CompareRow[];
  summary: CompareSummary;
  plane?: number;
  count_type?: string;
  note?: string;
  warnings?: string[];
}

export interface ExaminedFieldsResult {
  matched_file_entries: number;
  pages: Record<string, number>;
  n_entries: number;
  warnings: string[];
  stored_as?: string;
  evidence?: string;
}

export interface AutoChamberResponse {
  polygon: [number, number][];
  area_px2: number;
  rim_px: number;
  method: string;
  warnings: string[];
  suggested_exclusions: [number, number][][];
}

export interface SkippedField {
  field_index: number;
  name: string;
  polygon: [number, number][];
}

export interface FsListing {
  path: string;
  parent: string | null;
  dirs: string[];
  files: string[];
}

export type CountClass = 'red' | 'green' | 'yellow';

export interface SaveSelection {
  classes: CountClass[];
  label?: string;
}

export interface SaveSlideReport {
  directory: string;
  files: string[];
  slide_id: string;
  warnings: string[];
  rows: { what_counted: string; nofpol: number; nofgrids: number; fields: Record<string, number> }[];
}
