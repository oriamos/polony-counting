# Core contracts (v1)

The original module and REST contracts of the app. Later additions (two engines, lab fields, experiments) are in
[TWO_ENGINE_CONTRACT.md](TWO_ENGINE_CONTRACT.md), [FIELDS.md](FIELDS.md) and [EXPERIMENTS_API.md](EXPERIMENTS_API.md); the OpenAPI
page at `/docs` of a running app is authoritative.

Types: `src/polony_app/schemas.py` (pydantic). Data dir: env
`POLONY_DATA_DIR` (default `./data`): `originals/<analysis_id>/`, `app.sqlite`.

## Python interfaces (Core implements, App consumes)
```python
# imaging/ingest.py
def ingest_files(files: list[tuple[str, Path]], dest_dir: Path) -> ImageMeta
    # files = [(original_filename, path)]. Copies originals (never modifies) into dest_dir
    # under safe names, sha256 each. 1 file w/ 2 pages -> raw_two_channel; 1 RGB -> rgb_composite;
    # 2 single-page files -> raw_separate_files (validates identical shape else ValueError subclass IngestError).
    # channel.role stays None (label_hint filled from metadata e.g. "635 [W1]") unless unambiguous
    # 1-channel-per-file naming is NOT proof -> never auto-confirm.
    # Embedded ImageJ ROI/Overlays parsed to EmbeddedRoi without interpretation.
class IngestError(ValueError): ...
# imaging/io.py
def load_channel(meta: ImageMeta, dest_dir: Path, channel_index: int) -> np.ndarray  # 2D, original dtype
def load_rgb(meta, dest_dir) -> np.ndarray  # HxWx3 for composite-only
# imaging/render.py  (display only; never feeds detection)
def render_png(arr: np.ndarray, lo: float, hi: float, gamma: float = 1.0,
               crop: tuple[int,int,int,int] | None = None, max_dim: int | None = None,
               colormap: Literal["gray","red","green"] = "gray") -> bytes
def percentile_range(arr, lo_pct=0.5, hi_pct=99.8) -> tuple[float,float]
# imaging/preprocess.py
def correct_background(arr: np.ndarray, p: BackgroundParams) -> np.ndarray  # float32
# imaging/registration.py
def check_compatible(red: np.ndarray, green: np.ndarray) -> list[str]  # warnings; raises IngestError on shape mismatch
def estimate_shift(red, green, max_shift_px=20) -> tuple[float,float,float]  # (dy,dx,peak_corr) phase correlation
def apply_shift(arr, dy, dx) -> np.ndarray  # integer-pixel only, explicit user step
# detection/masks.py
def build_valid_mask(shape: tuple[int,int], regions: list[Region]) -> np.ndarray  # bool: union(counting) minus union(exclusion); if no counting regions: whole image minus exclusions
def point_region(x: float, y: float, regions: list[Region]) -> str | None  # id of counting region containing point and not in any exclusion, else None
def region_area_px(region_or_regions, shape) -> float
# detection/engine.py
@dataclass class RawSpot: x: float; y: float; peak: float; background: float; snr: float|None; saturated: bool
class SpotEngine(Protocol): name: str; def version(self)->str; def detect(self, img: np.ndarray, params: DetectionParams, valid_mask: np.ndarray|None) -> list[RawSpot]
# detection/bigfish_engine.py -> class BigFishEngine(SpotEngine)
# detection/matching.py
def associate(red: list[RawSpot], green: list[RawSpot], tolerance_px: float) -> list[DetectedObject]
    # constrained one-to-one (Hungarian on distance, gated by tolerance). Each physical object once.
    # unmatched red -> cls red; unmatched green -> green; matched -> yellow with association_distance_px.
    # ambiguity: if a spot has >1 partner within tolerance whose alternative assignment cost is within
    # margin -> mark review_reasons ambiguous_match + ambiguous_with, cls = uncertain.
# detection/pipeline.py
@dataclass class DetectionResult: objects: list[DetectedObject]; package_versions: dict[str,str]; notes: list[str]
def run_detection(red: np.ndarray, green: np.ndarray, regions: list[Region], request: DetectionRequest,
                  progress: Callable[[float,str],None], cancelled: Callable[[],bool],
                  window: tuple[int,int,int,int]|None = None) -> DetectionResult
    # raises DetectionCancelled. Assign region_id; drop points outside valid mask; compute review_reasons.
```

## REST API (App implements, Frontend consumes). Prefix `/api`; JSON; errors = ApiError body w/ 4xx.
- `GET /api/health` -> `{ok, version, bigfish_version}`
- `POST /api/analyses` multipart: `files` (1–2 TIFF), `name?`, `red_file?`/`green_file?` (optional names for 2-file pairing) -> `Analysis`
- `GET /api/analyses` -> `[{id,name,status,updated_at,finalized_at,counts?:Counts}]`; `GET /api/analyses/{id}` -> `Analysis`; `DELETE /api/analyses/{id}`
- `PUT /api/analyses/{id}/channels` `{mapping:[{index:int, role:"red"|"green"|"ignore"}]}` -> `Analysis` (sets role_confirmed; changing it after detection marks detection stale + clears approvals)
- `POST /api/analyses/{id}/alignment` `{dy:int,dx:int}` or `{estimate:true}` -> `{dy,dx,peak_corr,applied}`
- `GET /api/analyses/{id}/image/{view}` query `lo,hi,gamma,x0,y0,x1,y1,max_dim`; view ∈ `red|green|composite|processed_red|processed_green` -> `image/png` (display only). Headers: `X-Image-Width`,`X-Image-Height`,`X-Crop`,`X-Auto-Lo`,`X-Auto-Hi`.
- `GET /api/analyses/{id}/histogram/{red|green}` -> `{bins,counts,lo,hi,p995}`
- `PUT /api/analyses/{id}/regions` `{regions:Region[]}` -> `Analysis` (invalidates approvals of changed/overlapped regions; marks detection stale if any region changed)
- `PUT /api/analyses/{id}/settings` `{reporting_mode?, slide?:SlideMeta, detection_request?}` -> `Analysis`
- `POST /api/analyses/{id}/detect` `DetectionRequest` -> `Job` (async thread). Auto results replace only origin=auto objects that were never edited/reviewed-by-user; manual/edited objects are preserved; approvals of regions whose detections changed are cleared.
- `POST /api/analyses/{id}/detect/preview` `DetectionRequest` (with preview_window) -> `{objects: DetectedObject[], notes}` (not persisted)
- `GET /api/jobs/{id}` -> `Job`; `POST /api/jobs/{id}/cancel`
- `GET /api/analyses/{id}/counts` -> `{regions: Counts[], slide: Counts}` (never counts deleted; uncertain separate)
- `GET /api/analyses/{id}/review-queue` -> `[{object_id, reasons, priority, x, y}]` sorted
- `GET /api/analyses/{id}/thumb/{object_id}` `?size=64&view=composite` -> png crop around object
- `POST /api/analyses/{id}/actions` `{type, payload}` -> `Analysis`. Payloads:
  add `{x,y,cls}` · delete `{object_ids}` · move `{object_id,x,y}` · reclassify `{object_ids,cls}` ·
  merge `{object_ids}` (>=2 -> one object at centroid, evidence union) ·
  split `{object_id,positions:[[x,y],[x,y]],classes:[cls,cls]}` · confirm `{object_ids}` ·
  accept_region `{region_id}` (marks all non-deleted, non-uncertain objects reviewed; refuses if region not inspected) ·
  region_inspected `{region_id,inspected}` · approve_region `{region_id}` (requires inspected and zero unresolved `uncertain`) · unapprove_region `{region_id}`
- `POST /api/analyses/{id}/undo` / `redo` -> `Analysis`
- `POST /api/analyses/{id}/finalize` -> `Analysis` (requires >=1 counting region, all approved, mapping confirmed, no stale detection)
- `GET /api/analyses/{id}/export?kind=objects|regions|slide|provenance|bundle` -> csv/json/zip. Non-final => filename + column `status=DRAFT` and watermark in bundle README.
- `POST /api/csv-imports` multipart `file` -> `CsvImport`; `GET /api/csv-imports`; `GET /api/csv-imports/{id}`; `PUT /api/csv-imports/{id}` `{color_mapping}`.
- Static: if `frontend/dist` exists, serve at `/`.
