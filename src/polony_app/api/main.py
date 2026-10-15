"""FastAPI application. `create_app(data_dir)` builds an isolated app instance."""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
import json
import os
import re
import shutil
import threading
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import polony_app
from polony_app.export import exporter, lab
from polony_app.review import engine
from polony_app.review import consensus as consensus_mod
from polony_app.review.counts import compute_counts, review_queue
from polony_app.review.state import AnalysisState, ReviewError
from polony_app.schemas import (
    Analysis,
    ChannelRole,
    DetectionRequest,
    EngineName,
    GridSpec,
    JobStatus,
    Region,
    ReportingMode,
    SlideMeta,
)
from polony_app.config import load_settings
from polony_app.store import NotFound, Store, open_store  # noqa: F401
from polony_app.store.csv_import import parse_csv
from polony_app.store.db import new_id
from polony_app.store.jobs import JobRunner

from .images import VIEWS, ImageService, is_view, role_index

ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str | None, fallback: str = "upload") -> str:
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = _SAFE.sub("_", base).lstrip(".")
    return (base or fallback)[:120]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ------------------------------------------------------------------ request models
class OpenPathBody(BaseModel):
    path: str
    name: Optional[str] = None


class GridSelectionBody(BaseModel):
    field_index: int
    counted: bool


class GridBody(BaseModel):
    grid: Optional[GridSpec] = None    # null = remove the grid (its unapproved field regions) and clear the spec
    randomise: bool = False            # server picks a random shift (and a seed unless `seed` is given)
    seed: Optional[int] = None         # with randomise: reproduce a given shift
    replace: bool = True               # drop lab-field regions / skipped fields that are not part of the new grid
    force: bool = False                # clearing: also remove approved grid fields


class CountSelectionBody(BaseModel):
    classes: list[Literal["red", "green", "yellow"]]
    label: Optional[str] = None


class SaveSlideBody(BaseModel):
    selections: list[CountSelectionBody]
    output_dir: Optional[str] = None


class ChannelMap(BaseModel):
    index: int
    role: Literal["red", "green", "ignore"]


class ChannelsBody(BaseModel):
    mapping: list[ChannelMap]


class AlignmentBody(BaseModel):
    dy: Optional[int] = None
    dx: Optional[int] = None
    estimate: bool = False


class RegionsBody(BaseModel):
    regions: list[Region]


class SettingsBody(BaseModel):
    reporting_mode: Optional[ReportingMode] = None
    slide: Optional[SlideMeta] = None
    detection_request: Optional[DetectionRequest] = None


class ActionBody(BaseModel):
    type: str
    payload: dict[str, Any] = {}


class LabExportBody(BaseModel):
    analysis_ids: list[str] = []
    experiment_id: Optional[str] = None     # instead of analysis_ids: every finalized slide of the experiment
    style: Literal["t7", "counts"] = "t7"
    series: Optional[str] = None
    format: Literal["xlsx", "csv", "zip"] = "xlsx"


class FieldsImportBody(BaseModel):
    mode: Literal["counted", "all"]
    shift: tuple[float, float] = (0.0, 0.0)
    plane: Optional[int] = None
    replace: bool = False
    numbering: Literal["index", "ordinal"] = "index"


class AutoChamberBody(BaseModel):
    rim_px: int = 40


class PrepareModelBody(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = "general"


class QcSampleBody(BaseModel):
    n: Optional[int] = None                # fixed per-slide sample size (default 40, clamped to [1, N eligible])
    fraction: Optional[float] = None       # compatibility only: mapped to max(40, fraction * N eligible)
    seed: int = 0


class CsvMapBody(BaseModel):
    color_mapping: Optional[Literal["red", "green", "yellow"]] = None


# --- experiments (request models) ---
class ExperimentBody(BaseModel):
    name: str
    notes: Optional[str] = None
    default_detection: Optional[DetectionRequest] = None
    default_grid: Optional[dict] = None       # opaque grid spec (stage 1), stored as given


class ExperimentUpdateBody(BaseModel):
    """Only the keys present in the request are changed."""
    name: Optional[str] = None
    notes: Optional[str] = None
    default_detection: Optional[DetectionRequest] = None
    default_grid: Optional[dict] = None


class ExperimentSlidesBody(BaseModel):
    analysis_ids: list[str]


class ExperimentRunBody(BaseModel):
    only_missing: bool = False                       # skip slides that already have a detection result
    analysis_ids: Optional[list[str]] = None         # run only these slides (e.g. Retry of failed ones); None = all
    overrides: dict[str, DetectionRequest] = {}      # per-slide settings instead of the experiment default


# ------------------------------------------------------------------ helpers
EXAMINED_FILE = "examined_fields.json"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


def err(status: int, code: str, message: str, detail: dict | None = None) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "detail": detail}, status_code=status)


def unknown_keys(model: type[BaseModel], raw: Any, path: str = "") -> list[str]:
    """Dotted names of keys in `raw` that the pydantic `model` (recursively) does not define.
    schemas.py models ignore extras silently; the API layer refuses them instead (a typo in a
    parameter name must not be accepted and then have no effect)."""
    if not isinstance(raw, dict):
        return []
    out: list[str] = []
    for k, v in raw.items():
        f = model.model_fields.get(k)
        if f is None:
            out.append(f"{path}{k}")
            continue
        sub = f.annotation
        # Optional[X] -> X
        args = [t for t in getattr(sub, "__args__", ()) if t is not type(None)]
        if len(args) == 1 and getattr(sub, "__origin__", None) is not None:
            sub = args[0]
        if isinstance(sub, type) and issubclass(sub, BaseModel):
            out.extend(unknown_keys(sub, v, f"{path}{k}."))
    return out


async def _raw_body(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return None


def _json(model: Any) -> Response:
    return Response(model.model_dump_json(), media_type="application/json")


def _sig(a: Analysis, st: AnalysisState) -> str:
    """Signature of everything detection depends on besides parameters."""
    d = {
        "regions": [(r.id, r.kind.value, [list(p) for p in r.polygon]) for r in a.regions],
        "roles": [(c.index, c.role.value if c.role else None) for c in (a.image.channels if a.image else [])],
        "align": st.alignment,
    }
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _detector_kwargs(fn, aid: str, a: Analysis) -> dict:
    """Optional context the (two-engine) pipeline accepts; passed only if its signature takes it, so an
    older/simpler run_detection keeps working."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    accepts_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    want = {"analysis_id": aid, "image_sha256": dict(a.image.sha256) if a.image else {}}
    # Engine runs are recorded against the channel index of each role (= the TIFF page of a 2-page stack) and the
    # role is passed ONLY from the user-confirmed mapping (never inferred from page order).
    if a.image is not None:
        by_role = {c.role.value: c for c in a.image.channels if c.role is not None and c.role_confirmed}
        if "red" in by_role and "green" in by_role:
            want["pages"] = (int(by_role["red"].index), int(by_role["green"].index))
            want["roles"] = ("red", "green")
    return {k: v for k, v in want.items() if k in params or accepts_kw}


def _engine_warnings(runs: list, notes: list[str]) -> list[str]:
    """Human-readable warnings for the job message: failed/skipped engine runs + pipeline notes."""
    out: list[str] = []
    for r in runs:
        if getattr(r, "status", "ok") in ("failed", "skipped"):
            role = f" ({r.channel_role})" if getattr(r, "channel_role", None) else ""
            out.append(f"WARNING: {r.engine.value} run {r.status} on page {r.page}{role}: {r.error or 'no detail'}")
    for n in notes:
        if isinstance(n, str) and ("warn" in n.lower() or "fail" in n.lower()) and n not in out:
            out.append(n if n.lower().startswith("warn") else f"WARNING: {n}")
    return out


def create_app(data_dir: str | Path | None = None, settings=None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("POLONY_DATA_DIR") or "./data")
    settings = settings or load_settings()
    store = open_store(data_dir, settings)   # SQLite unless a shared database URL is configured
    runner = JobRunner(store)
    images = ImageService(store)
    max_upload = _env_int("POLONY_MAX_UPLOAD_MB", 512) * 1024 * 1024
    max_csv = _env_int("POLONY_MAX_CSV_MB", 20) * 1024 * 1024

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        runner.shutdown()
        store.close()

    app = FastAPI(title="Polony counting", version=polony_app.__version__, lifespan=lifespan)
    app.state.store, app.state.runner, app.state.images = store, runner, images

    # ------------------------------------------------------------- errors
    @app.exception_handler(ReviewError)
    async def _review_error(_: Request, e: ReviewError):
        return err(e.status, e.code, e.message, e.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        errs = [{"loc": list(x.get("loc", [])), "msg": x.get("msg")} for x in e.errors()]
        return err(422, "validation_error", "request validation failed", {"errors": errs})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 413: "too_large"}.get(e.status_code, "http_error")
        return err(e.status_code, code, str(e.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):  # pragma: no cover
        return err(500, "internal_error", f"{type(e).__name__}: {e}")

    def aid_ok(aid: str) -> str:
        if not ID_RE.match(aid):
            raise NotFound("analysis", aid)
        return aid

    api = APIRouter(prefix="/api")

    # ------------------------------------------------------------- health
    @api.get("/health")
    def health():
        try:
            bf = metadata.version("big-fish")
        except metadata.PackageNotFoundError:
            bf = None
        return {"ok": True, "version": polony_app.__version__, "bigfish_version": bf}

    @api.get("/storage")
    def get_storage():
        """Where analyses are stored (no credentials). Never raises: reports connected=false."""
        return store.info()

    # ------------------------------------------------------------- analyses
    def _stage_upload(f: UploadFile, dst: Path, limit: Optional[int], label: str) -> tuple[int, str]:
        """Copy one uploaded file to ``dst`` in 1 MB chunks; returns (size, sha256). 413 above ``limit`` bytes."""
        h = hashlib.sha256()
        size = 0
        with open(dst, "wb") as out:
            while chunk := f.file.read(1 << 20):
                size += len(chunk)
                if limit is not None and size > limit:
                    raise ReviewError("too_large", f"'{label}' exceeds the {limit // 2**20} MB limit", 413)
                h.update(chunk)
                out.write(chunk)
        return size, h.hexdigest()

    def _ingest_upload(files: list[UploadFile], name: Optional[str], red_file: Optional[str] = None,
                       green_file: Optional[str] = None, prepare=None) -> str:
        """THE upload path (POST /analyses and POST /experiments/{eid}/import): stage the TIFF(s), create the
        analysis, ingest, pairing hints, upload manifest. ``prepare(st)`` runs inside the final edit (experiment
        membership, copied channel mapping). Returns the analysis JSON; raises ReviewError and leaves nothing behind."""
        tmp = store.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        aid = None
        try:
            staged: list[tuple[str, Path]] = []
            orig_names_all: list[str] = []
            for i, f in enumerate(files):
                orig = f.filename or f"file{i}.tif"
                sname = safe_filename(orig, f"file{i}.tif")
                if not sname.lower().endswith((".tif", ".tiff")):
                    raise ReviewError("unsupported_file", f"'{sname}': only .tif/.tiff files are accepted", 415)
                dst = tmp / f"{i}_{sname}"
                size, _ = _stage_upload(f, dst, max_upload, sname)
                if size == 0:
                    raise ReviewError("empty_file", f"'{sname}' is empty", 422)
                staged.append((sname, dst))
                orig_names_all.append(orig)
            default_name = Path(staged[0][0]).stem
            aid = store.create_analysis((name or "").strip()[:120] or default_name)
            from polony_app.imaging.ingest import IngestError, ingest_files

            try:
                meta = ingest_files(staged, store.originals_dir(aid))
            except IngestError as e:
                raise ReviewError("ingest_error", str(e), 422)
            except Exception as e:  # noqa: BLE001  unreadable/corrupt TIFF
                raise ReviewError("unreadable_image", f"could not read image: {type(e).__name__}: {e}", 422)
            # optional pairing hints: SUGGESTED roles only, never confirmed
            if len(staged) == 2 and (red_file or green_file):
                names = [s[0] for s in staged]
                orig_names = [safe_filename(f.filename) for f in files]
                for hint, role in ((red_file, ChannelRole.red), (green_file, ChannelRole.green)):
                    if hint:
                        h = safe_filename(hint)
                        if h in orig_names:
                            idx = orig_names.index(h)
                            for c in meta.channels:
                                if c.index == idx:
                                    c.suggested_role = role
                del names
            try:  # original (unsanitised) upload names, for provenance: {stored_name: original_name}
                from polony_app.imaging.ingest import _safe_name

                manifest = {}
                for i_, ((sname_, _), orig_) in enumerate(zip(staged, orig_names_all)):
                    cand = _safe_name(i_, sname_)
                    if (store.originals_dir(aid) / cand).exists():
                        manifest[cand] = orig_
                (store.originals_dir(aid) / "upload_manifest.json").write_text(
                    json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa: BLE001  provenance aid only
                pass
            with store.edit(aid) as st:
                st.analysis.image = meta
                if prepare is not None:
                    prepare(st)
                engine.refresh_status(st)
                return st.analysis.model_dump_json()
        except Exception:
            if aid:
                try:
                    store.delete_analysis(aid)
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @api.post("/analyses")
    async def create_analysis(
        request: Request,
        files: list[UploadFile] = File(...),
        name: Optional[str] = Form(None),
        red_file: Optional[str] = Form(None),
        green_file: Optional[str] = Form(None),
    ):
        if not 1 <= len(files) <= 2:
            return err(422, "invalid_upload", "upload 1 or 2 TIFF files")
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > max_upload * 2 + 1_000_000:
            return err(413, "too_large", f"upload exceeds the {max_upload // 2**20} MB per-file limit")
        out_json = _ingest_upload(files, name, red_file, green_file)
        return Response(out_json, media_type="application/json", status_code=200)

    # ------------------------------------------------------------- open from disk / save next to the original
    @api.get("/fs/list")
    def fs_list(path: Optional[str] = None):
        """Folders and .tif/.tiff files of a local folder (the app runs on the researcher's own machine)."""
        p = Path(path).expanduser() if path else Path.home()
        try:
            p = p.resolve()
            if not p.is_dir():
                return err(422, "not_a_folder", f"not a folder: {p}")
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError as e:
            return err(422, "cannot_list", f"cannot read {p}: {e.strerror or e}")
        dirs = [e.name for e in entries if not e.name.startswith(".") and e.is_dir()]
        files = [e.name for e in entries if e.is_file() and e.suffix.lower() in (".tif", ".tiff")]
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "files": files}

    @api.post("/analyses/open-path")
    def open_path(body: OpenPathBody):
        """Open a TIFF in place: it is copied (read-only) like an upload, but its folder is remembered so that
        Save can write the labelled scan and CSVs next to it."""
        src = Path(body.path).expanduser()
        if not src.is_file():
            return err(422, "file_not_found", f"file not found: {src}")
        if src.suffix.lower() not in (".tif", ".tiff"):
            return err(415, "unsupported_file", f"'{src.name}': only .tif/.tiff files are accepted")
        src = src.resolve()
        if src.stat().st_size > max_upload:
            return err(413, "too_large", f"'{src.name}' exceeds the {max_upload // 2**20} MB limit")
        from polony_app.imaging.ingest import IngestError, _safe_name, ingest_files

        aid = store.create_analysis((body.name or "").strip()[:120] or src.stem)
        try:
            try:
                meta = ingest_files([(safe_filename(src.name, "file0.tif"), src)], store.originals_dir(aid))
            except IngestError as e:
                store.delete_analysis(aid)
                return err(422, "ingest_error", str(e))
            except Exception as e:  # noqa: BLE001  unreadable/corrupt TIFF
                store.delete_analysis(aid)
                return err(422, "unreadable_image", f"could not read image: {type(e).__name__}: {e}")
            meta.source_path = str(src)
            try:
                (store.originals_dir(aid) / "upload_manifest.json").write_text(
                    json.dumps({_safe_name(0, safe_filename(src.name, "file0.tif")): src.name}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
            except Exception:  # noqa: BLE001  provenance aid only
                pass
            with store.edit(aid) as st:
                st.analysis.image = meta
                engine.refresh_status(st)
                out_json = st.analysis.model_dump_json()
            return Response(out_json, media_type="application/json", status_code=200)
        except Exception:
            try:
                store.delete_analysis(aid)
            except Exception:  # noqa: BLE001
                pass
            raise

    @api.put("/analyses/{aid}/grid-selection")
    def put_grid_selection(aid: str, body: GridSelectionBody):
        """Switch one grid field on/off for counting. Off = its region is removed (blank in the CSVs) and its geometry
        is remembered in slide.skipped_fields; on = the region is recreated from that geometry."""
        from polony_app.schemas import RegionKind, SkippedField

        with store.edit(aid_ok(aid)) as st:
            a = st.analysis
            fi = body.field_index
            cur = next((r for r in a.regions if r.kind == RegionKind.counting and r.field_index == fi), None)
            skipped = [s_ for s_ in a.slide.skipped_fields if s_.field_index == fi]
            if body.counted:
                if cur is not None:
                    return _respond(st)
                if not skipped:
                    raise ReviewError("unknown_field", f"field {fi} is not part of the grid", 404)
                sk = skipped[0]
                rid = new_id()
                while any(r.id == rid for r in a.regions):
                    rid = new_id()
                new = Region(id=rid, kind=RegionKind.counting, name=sk.name, field_index=fi,
                             polygon=[tuple(p) for p in sk.polygon])
                regs = [r for r in a.regions if r.kind == RegionKind.counting and r.field_index is not None] + [new]
                regs.sort(key=lambda r: r.field_index or 0)
                regions = regs + [r for r in a.regions if not (r.kind == RegionKind.counting and r.field_index is not None)]
                engine.set_regions(st, regions, {"grid": a.slide.grid, "skipped_fields": [
                    s_ for s_ in a.slide.skipped_fields if s_.field_index != fi]})
            else:
                if cur is None:
                    if skipped:
                        return _respond(st)
                    raise ReviewError("unknown_field", f"field {fi} is not part of the grid", 404)
                sk = SkippedField(field_index=fi, name=cur.name, polygon=[tuple(p) for p in cur.polygon])
                engine.set_regions(st, [r for r in a.regions if r.id != cur.id], {"grid": a.slide.grid, "skipped_fields": [
                    s_ for s_ in a.slide.skipped_fields if s_.field_index != fi] + [sk]})
            return _respond(st)

    @api.put("/analyses/{aid}/grid")
    def put_grid(aid: str, body: GridBody, raw: Any = Depends(_raw_body)):
        """Store the counting-grid spec and (re)create the lab field regions from it in one undoable step.
        Per field index: a field that was switched off stays off (its geometry in slide.skipped_fields follows the
        grid), a field that exists keeps its region id, so its approval survives only if its polygon did not move
        (approval fingerprint). randomise=true: the server shifts the whole grid to a random position inside the
        image (cell size and count unchanged) and records the seed. grid=null: removes the grid."""
        from polony_app.fields import grid as grid_mod
        from polony_app.schemas import RegionKind, SkippedField

        unk = unknown_keys(GridBody, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        with store.edit(aid_ok(aid)) as st:
            engine.guard_editable(st)
            a = st.analysis
            is_field = lambda r: r.kind == RegionKind.counting and r.field_index is not None  # noqa: E731
            others = [r for r in a.regions if not is_field(r)]
            fields = [r for r in a.regions if is_field(r)]
            if body.grid is None and not body.randomise:
                if a.slide.grid is None:
                    return _respond(st)
                own = {fi for fi, *_ in grid_mod.spec_cells(a.slide.grid)}
                approved = sorted(r.field_index for r in fields if r.field_index in own and r.approved)
                if approved and not body.force:
                    return err(409, "grid_fields_approved",
                               f"approved grid field(s) {', '.join(map(str, approved))}: unapprove them first "
                               "(or pass force=true)", {"fields": approved})
                engine.set_regions(st, [r for r in fields if r.field_index not in own] + others, {
                    "grid": None, "skipped_fields": [s_ for s_ in a.slide.skipped_fields if s_.field_index not in own]})
                return _respond(st)
            spec = (body.grid or a.slide.grid)
            if spec is None:
                return err(422, "no_grid", "randomise needs a grid: send one or apply a grid first")
            spec = spec.model_copy(deep=True)
            prob = grid_mod.spec_problem(spec)
            if prob:
                return err(422, "invalid_grid", prob)
            if body.randomise:
                if a.image is None:
                    return err(409, "no_image", "analysis has no image")
                if body.seed is not None and not (0 <= body.seed < 2**32):
                    return err(422, "invalid_grid", "seed must be in 0..2^32-1")
                import secrets

                spec.seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
                spec.shift_dx, spec.shift_dy = grid_mod.random_shift(spec, a.image.width, a.image.height, spec.seed)
            cells = grid_mod.spec_cells(spec)
            own = {fi for fi, *_ in cells}
            by_index: dict[int, Region] = {}
            for r in fields:
                by_index.setdefault(r.field_index, r)          # a duplicated field number: the first one is kept
            skipped = {s_.field_index: s_ for s_ in a.slide.skipped_fields}
            taken = {r.id for r in a.regions}
            grid_regions: list[Region] = []
            new_skipped: list[SkippedField] = []
            for fi, _r, _c, poly in cells:
                cur = by_index.get(fi)
                if cur is not None:
                    grid_regions.append(Region(id=cur.id, kind=RegionKind.counting, name=cur.name, field_index=fi,
                                               polygon=[tuple(p) for p in poly]))
                elif fi in skipped:
                    new_skipped.append(SkippedField(field_index=fi, name=skipped[fi].name, polygon=[tuple(p) for p in poly]))
                else:
                    rid = new_id()
                    while rid in taken:
                        rid = new_id()
                    taken.add(rid)
                    grid_regions.append(Region(id=rid, kind=RegionKind.counting, name=f"Field-{fi}", field_index=fi,
                                               polygon=[tuple(p) for p in poly]))
            kept = [] if body.replace else [r for r in fields if r.field_index not in own]
            kept_skipped = [] if body.replace else [s_ for s_ in a.slide.skipped_fields if s_.field_index not in own]
            regs = sorted(grid_regions + kept, key=lambda r: r.field_index or 0)
            engine.set_regions(st, regs + others, {"grid": spec, "skipped_fields": sorted(
                kept_skipped + new_skipped, key=lambda s_: s_.field_index)})
            return _respond(st)

    @api.post("/analyses/{aid}/save-slide")
    def save_slide_route(aid: str, body: SaveSlideBody):
        """Researcher's Save: <name>_labeled.tif + <name>.csv (+ experiment_counts.csv) next to the original scan."""
        from polony_app.export import slide_save as ss

        sels = []
        for c in body.selections:
            classes = ss.parse_classes(c.classes)
            label = (c.label or "").strip() or ss.default_label(classes)
            sels.append((label, classes))
        with store.view(aid_ok(aid)) as st:
            a = st.analysis
            out_dir = ss.resolve_output_dir(a, body.output_dir)
            ss.count_fields(a, ss.COUNTABLE)   # refuses early when no grid field is selected
            red, green = images.raw_pair(st)
            report = ss.save_slide(a, red, green, sels, out_dir)
        return report

    @api.get("/analyses")
    def list_analyses():
        return store.list_analyses()

    @api.get("/analyses/{aid}")
    def get_analysis(aid: str):
        with store.view(aid_ok(aid)) as st:
            return Response(st.analysis.model_dump_json(), media_type="application/json")

    @api.delete("/analyses/{aid}")
    def delete_analysis(aid: str):
        aid_ok(aid)
        if runner.active_job(aid):
            return err(409, "job_in_progress", "a detection job is running; cancel it first")
        with store.view(aid) as st:
            if st.analysis.finalized_at is not None:
                return err(409, "analysis_finalized", "this analysis is finalized; reopen it before deleting")
        store.delete_analysis(aid)
        images.forget(aid)
        return {"ok": True}

    def _respond(st: AnalysisState, headers: dict | None = None) -> Response:
        return Response(st.analysis.model_dump_json(), media_type="application/json", headers=headers)

    @api.put("/analyses/{aid}/channels")
    def put_channels(aid: str, body: ChannelsBody):
        with store.edit(aid_ok(aid)) as st:
            engine.set_channel_mapping(st, [m.model_dump() for m in body.mapping])
            images.forget(aid)
            return _respond(st)

    @api.post("/analyses/{aid}/alignment")
    def post_alignment(aid: str, body: AlignmentBody):
        aid_ok(aid)
        if body.estimate:
            from polony_app.imaging import registration

            with store.view(aid) as st:
                _need_originals(st)
                red, green = (images.raw(st, "red"), images.raw(st, "green"))
                # estimate on the UNSHIFTED green so the answer is absolute
                cur = st.alignment
            if int(cur.get("dy", 0)) or int(cur.get("dx", 0)):
                from polony_app.imaging import io as imio

                with store.view(aid) as st:
                    gi = role_index(st.analysis, "green")
                    green = imio.load_channel(st.analysis.image, store.originals_dir(aid), gi)
            warns = registration.check_compatible(red, green)
            dy, dx, peak = registration.estimate_shift(red, green, 20)
            return {"dy": dy, "dx": dx, "peak_corr": peak, "applied": False, "warnings": warns}
        if body.dy is None or body.dx is None:
            return err(422, "invalid_alignment", "provide integer dy and dx, or estimate=true")
        if abs(body.dy) > 500 or abs(body.dx) > 500:
            return err(422, "invalid_alignment", "shift larger than 500 px refused")
        with store.edit(aid) as st:
            engine.guard_editable(st)
            im = st.analysis.image
            if im is not None and (abs(body.dy) >= im.height / 2 or abs(body.dx) >= im.width / 2):
                return err(422, "invalid_alignment",
                           f"shift must be smaller than half the image size ({im.height // 2} px vertically, "
                           f"{im.width // 2} px horizontally)")
            role_index(st.analysis, "green")  # mapping must be confirmed
            engine.set_alignment(st, body.dy, body.dx)
            images.forget(aid)
        return {"dy": body.dy, "dx": body.dx, "peak_corr": None, "applied": True}

    # ------------------------------------------------------------- originals (local files)
    def _missing_originals(st: AnalysisState) -> list[dict]:
        im = st.analysis.image
        if im is None:
            return []
        od = store.originals_dir(st.analysis.id)
        return [{"file": n, "sha256": h} for n, h in im.sha256.items() if not (od / n).is_file()]

    def _need_originals(st: AnalysisState) -> None:
        miss = _missing_originals(st)
        if miss:
            raise ReviewError(
                "original_missing",
                "This analysis was created on another computer. Attach the original TIFF(s) to view the image.",
                409, {"files": miss})

    @api.get("/analyses/{aid}/originals")
    def get_originals(aid: str):
        """Which original files this computer lacks (name + sha256 recorded when first imported)."""
        with store.view(aid_ok(aid)) as st:
            return {"missing": _missing_originals(st),
                    "expected": [{"file": n, "sha256": h} for n, h in (st.analysis.image.sha256 if st.analysis.image else {}).items()]}

    @api.post("/analyses/{aid}/originals")
    def attach_originals(aid: str, files: list[UploadFile] = File(...)):
        """Attach original TIFF(s) on this computer. Each file must match a recorded sha256."""
        aid_ok(aid)
        with store.view(aid) as st:
            expected = dict(st.analysis.image.sha256) if st.analysis.image else {}
        if not expected:
            return err(409, "no_image", "analysis has no image")
        by_hash = {h: n for n, h in expected.items()}
        tmp = store.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        try:
            staged: dict[str, Path] = {}
            for i, f in enumerate(files):
                p = tmp / f"{i}.part"
                _, digest = _stage_upload(f, p, None, safe_filename(f.filename))
                if digest not in by_hash:
                    return err(422, "sha256_mismatch",
                               f"'{safe_filename(f.filename)}' is not one of the original files of this analysis "
                               "(its SHA-256 differs from the recorded one); nothing was attached",
                               {"file": safe_filename(f.filename), "sha256": digest,
                                "expected": [{"file": n, "sha256": x} for n, x in expected.items()]})
                staged[by_hash[digest]] = p
            od = store.originals_dir(aid)
            od.mkdir(parents=True, exist_ok=True)
            for name, p in staged.items():
                shutil.move(str(p), str(od / name))
            images.forget(aid)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with store.view(aid) as st:
            return {"attached": sorted(staged), "missing": _missing_originals(st)}

    # ------------------------------------------------------------- images
    def _crop(x0, y0, x1, y1, st):
        vals = [x0, y0, x1, y1]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            raise ReviewError("invalid_crop", "give all of x0,y0,x1,y1 or none", 422)
        if x1 <= x0 or y1 <= y0:
            raise ReviewError("invalid_crop", "crop must have x1>x0 and y1>y0", 422)
        im = st.analysis.image
        c = (max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))
        if c[2] <= c[0] or c[3] <= c[1]:
            raise ReviewError("invalid_crop", f"crop lies outside the {im.width} x {im.height} image", 422)
        return c

    @api.get("/analyses/{aid}/image/{view}")
    def get_image(aid: str, view: str, lo: Optional[float] = None, hi: Optional[float] = None,
                  gamma: float = 1.0, x0: Optional[int] = None, y0: Optional[int] = None,
                  x1: Optional[int] = None, y1: Optional[int] = None,
                  max_dim: Optional[int] = None, colormap: Literal["gray", "red", "green"] = "gray"):
        if not is_view(view):
            return err(422, "invalid_view", f"view must be one of {', '.join(VIEWS)} or ch<index>")
        if max_dim is not None and not 1 <= max_dim <= 8192:
            return err(422, "invalid_param", "max_dim must be in 1..8192")
        with store.view(aid_ok(aid)) as st:
            if st.analysis.image is None:
                return err(409, "no_image", "analysis has no image")
            _need_originals(st)
            crop = _crop(x0, y0, x1, y1, st)
            png, (alo, ahi) = images.render(st, view, lo, hi, gamma, crop, max_dim, colormap)
            im = st.analysis.image
            c = crop or (0, 0, im.width, im.height)
        return Response(png, media_type="image/png", headers={
            "X-Image-Width": str(im.width), "X-Image-Height": str(im.height),
            "X-Crop": ",".join(str(int(v)) for v in c),
            "X-Auto-Lo": repr(float(alo)), "X-Auto-Hi": repr(float(ahi)),
            "Cache-Control": "no-store",
        })

    @api.get("/analyses/{aid}/histogram/{role}")
    def get_histogram(aid: str, role: Literal["red", "green"]):
        with store.view(aid_ok(aid)) as st:
            _need_originals(st)
            return images.histogram(st, role)

    @api.get("/analyses/{aid}/thumb/{object_id}")
    def get_thumb(aid: str, object_id: str, size: int = 64, view: str = "composite"):
        if not 8 <= size <= 512:
            return err(422, "invalid_param", "size must be in 8..512")
        with store.view(aid_ok(aid)) as st:
            o = next((o for o in st.analysis.objects if o.id == object_id), None)
            if o is None:
                raise NotFound("object", object_id)
            _need_originals(st)
            im = st.analysis.image
            half = size // 2
            x0 = int(round(o.x)) - half
            y0 = int(round(o.y)) - half
            x0 = max(0, min(x0, im.width - size)) if im.width >= size else 0
            y0 = max(0, min(y0, im.height - size)) if im.height >= size else 0
            crop = (x0, y0, min(im.width, x0 + size), min(im.height, y0 + size))
            # contrast comes from the whole channel so thumbnails are comparable
            png, _ = images.render(st, view, None, None, 1.0, crop, None)
        return Response(png, media_type="image/png",
                        headers={"X-Crop": ",".join(map(str, crop)), "Cache-Control": "no-store"})

    # ------------------------------------------------------------- regions / settings
    @api.put("/analyses/{aid}/regions")
    def put_regions(aid: str, body: RegionsBody):
        with store.edit(aid_ok(aid)) as st:
            engine.set_regions(st, body.regions)
            return _respond(st)

    # ------------------------------------------------------------- lab fields / reference marks
    def _reference(st: AnalysisState):
        from polony_app.fields.reference import load_reference

        return load_reference(store.originals_dir(st.analysis.id), st.analysis.image)

    @api.get("/analyses/{aid}/reference")
    def get_reference(aid: str):
        with store.view(aid_ok(aid)) as st:
            return _reference(st).to_dict()

    @api.post("/analyses/{aid}/fields/import")
    def post_fields_import(aid: str, body: FieldsImportBody):
        from polony_app.fields import service as fields

        with store.edit(aid_ok(aid)) as st:
            engine.guard_editable(st)
            im = st.analysis.image
            ref = _reference(st)
            if im is None or not ref.available or ref.grid is None:
                return err(422, "no_grid", (ref.reason if not ref.available else None)
                           or "This image has no counting-grid overlay to build fields from.")
            planned, warns = fields.plan_fields(ref, body.mode, body.shift, body.plane, body.numbering,
                                                im.width, im.height)
            regions = fields.apply_fields(st.analysis, planned, body.replace, new_id)
            engine.set_regions(st, regions)
            out = json.loads(st.analysis.model_dump_json())
            out["field_import"] = {"mode": body.mode, "shift": list(body.shift), "plane": body.plane,
                                   "numbering": body.numbering, "created": [p["name"] for p in planned],
                                   "warnings": warns + ref.warnings}
            return JSONResponse(out)

    @api.get("/analyses/{aid}/fields/compare")
    def get_fields_compare(aid: str, plane: int = 0, count_type: str = "red_positive",
                           numbering: Literal["index", "ordinal"] = "index"):
        from polony_app.fields import service as fields

        if plane < 0:
            return err(422, "invalid_param", "plane must be 0 or greater")
        from polony_app.fields import examined

        with store.view(aid_ok(aid)) as st:
            mp = store.originals_dir(aid) / EXAMINED_FILE
            try:
                manifest = examined.load_manifest(mp) if mp.is_file() else None
            except examined.ExaminedError as exc:
                return err(422, "examined_manifest_invalid", str(exc))
            return fields.compare(st.analysis, _reference(st), plane, count_type, numbering, manifest, store.originals_dir(aid))

    @api.post("/analyses/{aid}/examined-fields")
    def post_examined_fields(aid: str, file: UploadFile = File(...)):
        """Attach the researcher's examined-fields manifest (JSON, or CSV with a '# evidence:' first line). It is
        validated, stored as originals/<id>/examined_fields.json and read by every later comparison."""
        from polony_app.fields import examined, service as fields

        name = safe_filename(file.filename or "examined_fields.json")
        suffix = ".csv" if name.lower().endswith(".csv") else ".json"
        raw = file.file.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            return err(422, "examined_manifest_invalid", "the manifest is larger than 2 MB")
        tmp = store.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        try:
            (tmp / f"m{suffix}").write_bytes(raw)
            try:
                manifest = examined.load_manifest(tmp / f"m{suffix}")
            except examined.ExaminedError as exc:
                return err(422, "examined_manifest_invalid", str(exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with store.edit(aid_ok(aid)) as st:
            engine.guard_editable(st)
            try:
                info = fields.summarize_manifest(st.analysis, manifest, store.originals_dir(aid))
            except examined.ExaminedError as exc:
                return err(422, "examined_manifest_mismatch", str(exc))
            od = store.originals_dir(aid)
            od.mkdir(parents=True, exist_ok=True)
            data = json.loads(manifest.model_dump_json(by_alias=True, exclude={"sha256"}))
            data["uploaded_name"] = name
            data["uploaded_sha256"] = manifest.sha256
            (od / EXAMINED_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
            info["stored_as"] = EXAMINED_FILE
            info["evidence"] = manifest.evidence
            return info

    @api.post("/analyses/{aid}/regions/auto-chamber")
    def post_auto_chamber(aid: str, body: AutoChamberBody):
        from polony_app.imaging import chamber, io as imio

        if not 0 <= body.rim_px <= 500:
            return err(422, "invalid_param", "rim_px must be in 0..500")
        with store.view(aid_ok(aid)) as st:
            im = st.analysis.image
            if im is None:
                return err(409, "no_image", "analysis has no image")
            _need_originals(st)
            if im.kind.value == "rgb_composite":
                return err(422, "chamber_unsupported", "Automatic chamber detection needs the raw two-channel image, "
                           "not an RGB composite.")
            od = store.originals_dir(aid)
            chans = [imio.load_channel(im, od, c.index) for c in im.channels]
            nonzero_align = bool(st.analysis.alignment_dy or st.analysis.alignment_dx)
        try:
            res = chamber.auto_chamber(chans, body.rim_px)
        except chamber.ChamberError as e:
            return err(422, "chamber_not_found", str(e))
        if nonzero_align:
            res["warnings"].append("A green alignment shift is set; the polygon is in the unshifted raw frame "
                                   "of the detected channel - check it against the red image.")
        return res

    @api.put("/analyses/{aid}/settings")
    def put_settings(aid: str, body: SettingsBody, raw: Any = Depends(_raw_body)):
        unk = unknown_keys(SettingsBody, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        if body.detection_request is not None and body.detection_request.preview_window is not None:
            return err(422, "invalid_request",
                       "detection_request.preview_window is not a stored setting (only /detect/preview takes it)")
        if body.slide is not None:
            s = body.slide
            for f in ("sample_volume_ul", "concentration_factor", "pixel_size_um"):
                v = getattr(s, f)
                if v is not None and not (v > 0 and v < float("inf")):
                    return err(422, "invalid_slide", f"{f} must be a positive number or null")
            for f in ("field_area", "grid_area", "efficiency_input"):
                v = getattr(s, f)
                if v is not None and not (v > 0 and v < float("inf")):
                    return err(422, "invalid_slide", f"{f} must be a positive number or null")
            names = [x.name for x in s.lab_series]
            if any(not n.strip() for n in names) or len(set(names)) != len(names):
                return err(422, "invalid_slide", "lab_series names must be non-empty and unique")
            if s.qc is not None and s.qc not in (0, 1):
                return err(422, "invalid_slide", "qc must be 0, 1 or null")
            if s.csv_color_mapping:
                bad = [v for v in s.csv_color_mapping.values() if v not in ("red", "green", "yellow")]
                if bad:
                    return err(422, "invalid_slide", "csv_color_mapping values must be red|green|yellow")
        with store.edit(aid_ok(aid)) as st:
            engine.set_settings(st, body.reporting_mode, body.slide, body.detection_request)
            return _respond(st)

    # ------------------------------------------------------------- detection
    def _detection_inputs(st: AnalysisState):
        a = st.analysis
        if a.image is None:
            raise ReviewError("no_image", "analysis has no image", 409)
        if not engine.mapping_confirmed(a):
            raise ReviewError("channel_mapping_unconfirmed",
                              "confirm the red/green channel mapping before detecting", 409)
        _need_originals(st)

    @api.post("/analyses/{aid}/detect")
    def post_detect(aid: str, body: DetectionRequest, raw: Any = Depends(_raw_body)):
        aid_ok(aid)
        unk = unknown_keys(DetectionRequest, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        if body.preview_window is not None:
            return err(422, "invalid_request", "preview_window is only valid for /detect/preview")
        return _json(_start_detection(aid, body))

    def _start_detection(aid: str, body: DetectionRequest, *, parent_id: Optional[str] = None,
                         experiment_id: Optional[str] = None, require_counting: bool = False, then=None):
        """THE detection start, for POST /detect and for every slide of a batch run. Returns the queued Job or
        raises ReviewError (nothing started). ``then(job)`` is called once the job has finished, whatever its
        outcome. ``require_counting``: refuse a slide without a counting region (an unattended run must not
        silently count the whole image)."""

        def work(progress, cancelled):
            try:
                return work_inner(progress, cancelled)
            except BaseException:
                finish(None)  # leave 'detecting' before the job is reported as failed/cancelled
                raise

        def work_inner(progress, cancelled):
            from polony_app.detection import pipeline

            with store.view(aid) as st:
                red, green = images.raw_pair(st)
            progress(0.02, "images loaded")
            res = pipeline.run_detection(red, green, regions, body, progress, cancelled, alignment=align,
                                         **extra_kw)
            if cancelled():
                finish(None)
                return "cancelled"
            runs = list(getattr(res, "engine_runs", None) or [])
            raws = list(getattr(res, "raw_detections", None) or [])
            with store.edit(aid) as st:
                if _sig(st.analysis, st) != start_sig:
                    raise RuntimeError(
                        "regions, channel mapping or alignment changed while detection ran; "
                        "results discarded - run detection again"
                    )
                cur_req = (st.analysis.detection_request.model_dump(mode="json")
                           if st.analysis.detection_request is not None else None)
                # committed only now, and only if settings did not change during the job
                # (otherwise the result is applied but comes out stale)
                if cur_req == start_req:
                    st.analysis.detection_request = body
                engine.apply_detection_result(
                    st, list(res.objects), body, dict(res.package_versions), list(res.notes),
                    engine_name=body.engine.value, cancelled=cancelled,
                )
            # Raw engine output is audit data: appended (never replacing earlier runs) AFTER the analysis
            # commit and outside the Analysis JSON. A failed engine run is stored too (status=failed).
            if runs or raws:
                store.save_engine_results(aid, runs, raws)
            warns = _engine_warnings(runs, list(res.notes))
            msg = f"{len(res.objects)} objects detected"
            return msg + ("; " + " | ".join(warns) if warns else "")

        def finish(job):
            try:
                with store.edit(aid) as st:
                    engine.refresh_status(st, job_running=False)
            except Exception:  # noqa: BLE001
                pass
            if job is not None and then is not None:
                then(job)

        # The active-job check, the 'detecting' flag and runner.submit happen under ONE hold of the
        # analysis lock (the worker's own store access waits for it), so two concurrent requests
        # cannot both pass the check and a refused request can never clear the flag of the
        # accepted job.
        with store.edit(aid) as st:
            engine.guard_editable(st)
            _detection_inputs(st)
            if require_counting and not any(r.kind.value == "counting" for r in st.analysis.regions):
                raise ReviewError("no_counting_region", "no counting region: set a grid or draw a counting region "
                                  "on this slide before running detection", 409)
            if runner.active_job(aid):
                raise ReviewError("job_in_progress", "a detection job is already running", 409)
            start_sig = _sig(st.analysis, st)
            start_req = (st.analysis.detection_request.model_dump(mode="json")
                         if st.analysis.detection_request is not None else None)
            regions = [r.model_copy(deep=True) for r in st.analysis.regions]
            align = (int(st.analysis.alignment_dy), int(st.analysis.alignment_dx))
            try:
                from polony_app.detection import pipeline as _pl

                extra_kw = _detector_kwargs(_pl.run_detection, aid, st.analysis)
            except Exception:  # noqa: BLE001  (the job itself reports an import problem)
                extra_kw = {}
            try:
                job = runner.submit(aid, work, finish, kind="detect", parent_id=parent_id,
                                    experiment_id=experiment_id)
            except RuntimeError:
                raise ReviewError("job_in_progress", "a detection job is already running", 409)
            engine.refresh_status(st, job_running=True)
        return job

    @api.post("/analyses/{aid}/detect/preview")
    def post_preview(aid: str, body: DetectionRequest):
        if body.preview_window is None:
            return err(422, "invalid_request", "preview_window [x0,y0,x1,y1] is required")
        x0, y0, x1, y1 = body.preview_window
        with store.view(aid_ok(aid)) as st:
            _detection_inputs(st)
            im = st.analysis.image
            if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
                return err(422, "invalid_window", "preview_window must lie within the image")
            red, green = images.raw_pair(st)
            regions = [r.model_copy(deep=True) for r in st.analysis.regions]
            align = (int(st.analysis.alignment_dy), int(st.analysis.alignment_dx))
            from polony_app.detection import pipeline as _plp

            extra_kw = _detector_kwargs(_plp.run_detection, aid, st.analysis)
        from polony_app.detection import pipeline

        res = pipeline.run_detection(red, green, regions, body, lambda f, m="": None,
                                     lambda: False, window=body.preview_window, alignment=align,
                                     **extra_kw)
        return {"objects": [o.model_dump(mode="json") for o in res.objects], "notes": list(res.notes)}

    @api.get("/jobs/active")
    def active_jobs():
        return [j.model_dump(mode="json") for j in runner.active_jobs()]

    @api.get("/jobs/{jid}")
    def get_job(jid: str):
        return _json(runner.get(jid))

    @api.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        return _json(runner.cancel(jid))

    # ------------------------------------------------------------- engines (two-engine architecture)
    def _spotiflow_adapter():
        """Lazy: importing the adapter must never be needed for the default Big-FISH path."""
        try:
            import importlib

            return importlib.import_module("polony_app.engines.spotiflow_adapter"), None
        except Exception as e:  # noqa: BLE001  module missing or spotiflow/torch import problem
            return None, f"{type(e).__name__}: {e}"

    def _model_dir() -> Path:
        env = os.environ.get("POLONY_MODEL_DIR")
        if env:
            return Path(env)
        from polony_app.config import home_dir

        return home_dir() / "models" / "spotiflow"

    @api.get("/engines")
    def get_engines():
        try:
            bf = metadata.version("big-fish")
        except metadata.PackageNotFoundError:
            bf = None
        out = [{"name": "bigfish", "available": True, "reason": "", "version": bf,
                "device": {"cuda": False, "mps": False, "cpu": True}, "models": []}]
        ad, why = _spotiflow_adapter()
        sp = {"name": "spotiflow", "available": False, "reason": why or "", "version": None,
              "device": {"cuda": False, "mps": False, "cpu": False}, "models": []}
        if ad is None:
            sp["reason"] = f"Spotiflow adapter is not installed ({why})"
        else:
            try:
                cls = getattr(ad, "SpotiflowEngine", None)
                if not hasattr(ad, "available") and cls is not None:
                    d = cls().describe()    # adapter's own report: availability, version, device, cached models
                    sp.update({"available": bool(d.get("available")), "reason": str(d.get("reason") or ""),
                               "version": d.get("version") or None,
                               "device": {**sp["device"], **(d.get("device") or {})},
                               "models": list(d.get("models") or []),
                               "torch_version": d.get("torch_version") or None,
                               "device_selected": d.get("device_selected") or None,
                               "device_note": d.get("device_note") or None})
                    out.append(sp)
                    return out
                ok = ad.available()
                if isinstance(ok, tuple):
                    sp["available"], sp["reason"] = bool(ok[0]), str(ok[1] or "")
                else:
                    sp["available"] = bool(ok)
                    sp["reason"] = "" if ok else "Spotiflow or torch is not installed"
                for fn, key in (("version", "version"), ("device_info", "device"), ("list_models", "models")):
                    f = getattr(ad, fn, None)
                    if callable(f):
                        try:
                            v = f()
                            if key == "device" and isinstance(v, dict):
                                sp[key] = {**sp[key], **v}
                            elif key == "models":
                                sp[key] = [m if isinstance(m, dict) else
                                           {k: getattr(m, k, None) for k in ("name", "cached", "version", "checksum_sha256")}
                                           for m in (v or [])]
                            elif key == "version":
                                sp[key] = v
                        except Exception:  # noqa: BLE001  informational only
                            pass
            except Exception as e:  # noqa: BLE001
                sp["available"], sp["reason"] = False, f"{type(e).__name__}: {e}"
        out.append(sp)
        return out

    def _prepare_model(name: str, progress) -> str:
        """Download/load one Spotiflow model (no-op when cached). Used by POST /engines/spotiflow/prepare and
        once before the first slide of a batch run that uses Spotiflow."""
        ad, why = _spotiflow_adapter()
        if ad is None or not hasattr(ad, "ensure_model"):
            raise ReviewError("engine_unavailable", f"Spotiflow is not available in this installation ({why})", 409)
        cache = ad.model_cache_dir() if hasattr(ad, "model_cache_dir") else _model_dir()
        info = ad.ensure_model(name, cache, progress)
        cached = getattr(info, "cached", None)
        return (f"model '{name}' ready" + (" (already cached)" if cached else " (downloaded)")
                + (f", sha256 {getattr(info, 'checksum_sha256', '')[:12]}" if getattr(info, "checksum_sha256", None) else ""))

    @api.post("/engines/spotiflow/prepare")
    def post_prepare_spotiflow(body: PrepareModelBody):
        """Download/load a Spotiflow model once (cached afterwards; never re-downloaded) as a Job."""
        ad, why = _spotiflow_adapter()
        if ad is None or not hasattr(ad, "ensure_model"):
            return err(409, "engine_unavailable", f"Spotiflow is not available in this installation ({why})")
        name = body.model_name.strip()
        if not name or len(name) > 200:
            return err(422, "invalid_model_name", "model_name must be a non-empty name")

        def work(progress, cancelled):
            return _prepare_model(name, progress)

        try:
            job = runner.submit("engine-spotiflow", work, kind="prepare_model")
        except RuntimeError:
            raise ReviewError("job_in_progress", "a model preparation job is already running", 409)
        return _json(job)

    @api.get("/analyses/{aid}/engine-runs")
    def get_engine_runs(aid: str):
        aid_ok(aid)
        if not store.exists(aid):
            raise NotFound("analysis", aid)
        return [json.loads(r.model_dump_json()) for r in store.get_engine_runs(aid)]

    @api.get("/analyses/{aid}/raw-detections")
    def get_raw_detections(aid: str, engine: Optional[EngineName] = None, page: Optional[int] = None,
                           run_id: Optional[str] = None, valid: Optional[bool] = None,
                           all_runs: bool = False, limit: int = 5000, offset: int = 0):
        """Raw engine detections. Without ``run_id`` and without ``all_runs=true`` only the LATEST run of each
        (engine, page) is returned (older runs stay stored for audit and are reachable via ``run_id`` or
        ``all_runs=true``), so a rerun never shows stale points on top of the current ones."""
        aid_ok(aid)
        if not store.exists(aid):
            raise NotFound("analysis", aid)
        if not 1 <= limit <= 200_000 or offset < 0:
            return err(422, "invalid_param", "limit must be in 1..200000 and offset >= 0")
        if run_id is not None or all_runs:
            rows = store.get_raw_detections(aid, engine=engine, page=page, run_id=run_id, valid=valid,
                                            limit=limit, offset=offset)
            total = store.count_raw_detections(aid, engine=engine, page=page, run_id=run_id, valid=valid)
        else:
            latest: dict = {}
            for r in sorted(enumerate(store.get_engine_runs(aid)), key=lambda t: (t[1].started_at or "", t[0])):
                if (engine is None or r[1].engine == engine) and (page is None or r[1].page == page):
                    latest[(r[1].engine, r[1].page)] = r[1].run_id   # last one wins = latest
            rows, total = [], 0
            for rid in latest.values():
                total += store.count_raw_detections(aid, engine=engine, page=page, run_id=rid, valid=valid)
                rows += store.get_raw_detections(aid, engine=engine, page=page, run_id=rid, valid=valid)
            rows = rows[offset:offset + limit]
        return Response("[" + ",".join(r.model_dump_json() for r in rows) + "]", media_type="application/json",
                        headers={"X-Total-Count": str(total)})

    @api.get("/analyses/{aid}/consensus/summary")
    def get_consensus_summary(aid: str):
        with store.view(aid_ok(aid)) as st:
            return consensus_mod.consensus_summary(st)

    @api.post("/analyses/{aid}/consensus/qc-sample")
    def post_qc_sample(aid: str, body: QcSampleBody):
        with store.edit(aid_ok(aid)) as st:
            return consensus_mod.mark_qc_sample(st, body.fraction, body.seed, body.n)

    # ------------------------------------------------------------- review
    @api.get("/analyses/{aid}/counts")
    def get_counts(aid: str):
        with store.view(aid_ok(aid)) as st:
            c = compute_counts(st.analysis)
            return {"regions": [x.model_dump(mode="json") for x in c["regions"]],
                    "slide": c["slide"].model_dump(mode="json")}

    @api.get("/analyses/{aid}/review-queue")
    def get_queue(aid: str):
        with store.view(aid_ok(aid)) as st:
            return review_queue(st.analysis)

    @api.post("/analyses/{aid}/actions")
    def post_action(aid: str, body: ActionBody):
        with store.edit(aid_ok(aid)) as st:
            act = engine.apply_action(st, body.type, dict(body.payload))
            created = []
            for k in ("object_id", "result_id"):
                if k in act.payload and body.type in ("add", "merge"):
                    created.append(act.payload[k])
            created += act.payload.get("result_ids", [])
            return _respond(st, {"X-Created-Ids": ",".join(created)} if created else None)

    @api.post("/analyses/{aid}/undo")
    def post_undo(aid: str):
        with store.edit(aid_ok(aid)) as st:
            engine.undo(st)
            return _respond(st)

    @api.post("/analyses/{aid}/redo")
    def post_redo(aid: str):
        with store.edit(aid_ok(aid)) as st:
            engine.redo(st)
            return _respond(st)

    @api.post("/analyses/{aid}/finalize")
    def post_finalize(aid: str):
        with store.edit(aid_ok(aid)) as st:
            if runner.active_job(aid):
                raise ReviewError("job_in_progress", "a detection job is running", 409)
            engine.finalize(st)
            return _respond(st)

    @api.post("/analyses/{aid}/reopen")
    def post_reopen(aid: str):
        with store.edit(aid_ok(aid)) as st:
            engine.reopen(st)
            return _respond(st)

    # ------------------------------------------------------------- export
    EXPORT_KINDS = (*exporter.KINDS, "actions", "history")

    _export_name = exporter.export_filename

    @api.get("/analyses/{aid}/export")
    def get_export(aid: str, kind: str = "bundle"):
        if kind not in EXPORT_KINDS:
            return err(422, "invalid_kind", f"kind must be one of {', '.join(EXPORT_KINDS)}")
        with store.view(aid_ok(aid)) as st:
            a = st.analysis
            fn = _export_name(a, kind)
            od = store.originals_dir(aid)
            if kind == "objects":
                body, mt = exporter.objects_csv(a).encode("utf-8"), "text/csv"
            elif kind == "regions":
                body, mt = exporter.regions_csv(a).encode("utf-8"), "text/csv"
            elif kind == "slide":
                body, mt = exporter.slide_csv(a).encode("utf-8"), "text/csv"
            elif kind == "actions":
                body, mt = exporter.actions_jsonl(st).encode("utf-8"), "application/x-ndjson"
            elif kind == "history":
                body, mt = exporter.objects_history_csv(a).encode("utf-8"), "text/csv"
            elif kind == "provenance":
                body, mt = exporter.provenance_json(st, od, store.get_engine_runs(aid)).encode("utf-8"), "application/json"
            else:
                body, mt = exporter.bundle_zip(st, od, store.get_engine_runs(aid),
                                               store.get_raw_detections(aid)), "application/zip"
            status = exporter.status_label(a)
        return Response(body, media_type=mt, headers={
            "Content-Disposition": f'attachment; filename="{fn}"', "X-Export-Status": status})

    # ------------------------------------------------------------- lab export
    def _lab_response(ids: list[str], style: str, series: Optional[str], fmt: str) -> Response:
        analyses = []
        for i in ids:
            with store.view(aid_ok(i)) as st:
                analyses.append(st.analysis.model_copy(deep=True))
        try:
            files, slides = lab.build_files(analyses, style, series, fmt)
            out = lab.package(files, slides, style, fmt)
        except lab.LabExportError as e:
            return err(e.status, e.code, e.message, e.detail)
        warnings = lab.all_warnings(slides)
        return Response(out.data, media_type=out.media_type, headers={
            "Content-Disposition": f'attachment; filename="{out.name}"', "X-Export-Status": "FINAL",
            "X-Lab-Warnings": str(len(warnings))})

    @api.get("/analyses/{aid}/lab-export")
    def get_lab_export(aid: str, series: Optional[str] = None, format: Literal["xlsx", "csv"] = "xlsx"):
        with store.view(aid_ok(aid)) as st:
            names = [x.name for x in st.analysis.slide.lab_series]
        if series is None:
            if len(names) != 1:
                return err(422, "series_required",
                           "choose the series to export (?series=<name>); defined: " + (", ".join(names) or "none"),
                           {"lab_series": names})
            series = names[0]
        return _lab_response([aid], "t7", series, format)

    @api.post("/lab-export")
    def post_lab_export(body: LabExportBody):
        ids = list(body.analysis_ids)
        if body.experiment_id is not None:
            if ids:
                return err(422, "invalid_request", "give either analysis_ids or experiment_id, not both")
            store.get_experiment(eid_ok(body.experiment_id))
            ids = [i for i in store.experiment_slide_ids(body.experiment_id) if _finalized(i)]
            if not ids:
                return err(409, "no_finalized_slides", "this experiment has no finalized slide to export")
        elif not ids:
            return err(422, "invalid_request", "give analysis_ids or experiment_id")
        return _lab_response(ids, body.style, body.series, body.format)

    # ------------------------------------------------------------- csv imports
    @api.post("/csv-imports")
    async def post_csv(file: UploadFile = File(...)):
        data = await file.read(max_csv + 1)
        if len(data) > max_csv:
            return err(413, "too_large", f"CSV exceeds {max_csv // 2**20} MB")
        if not data.strip():
            return err(422, "empty_file", "CSV file is empty")
        fname = safe_filename(file.filename, "import.csv")
        cid = new_id()
        d = store.csv_dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_bytes(data)  # verbatim copy
        imp = parse_csv(data, fname, cid)
        store.save_csv_import(imp)
        return _json(imp)

    @api.get("/csv-imports")
    def list_csv():
        return [json.loads(c.model_dump_json()) for c in store.list_csv_imports()]

    @api.get("/csv-imports/{cid}")
    def get_csv(cid: str):
        if not ID_RE.match(cid):
            raise NotFound("csv_import", cid)
        return _json(store.get_csv_import(cid))

    @api.put("/csv-imports/{cid}")
    def put_csv(cid: str, body: CsvMapBody):
        if not ID_RE.match(cid):
            raise NotFound("csv_import", cid)
        imp = store.get_csv_import(cid)
        imp.color_mapping = body.color_mapping
        store.save_csv_import(imp)
        return _json(imp)

    # --- experiments -------------------------------------------------------------------------------
    # An experiment groups slides (analyses with experiment_id) in a fixed order. A batch run is one job of
    # kind 'batch' (analysis_id "experiment:<eid>", so the runner allows one active batch per experiment) that
    # starts one ordinary detect job per slide, sequentially, each with parent_id = the batch id.
    from polony_app.export import experiment as exp_export
    from polony_app.schemas import EngineChoice, Experiment, ExperimentStatus, JobKind, RegionKind

    def eid_ok(eid: str) -> str:
        if not ID_RE.match(eid):
            raise NotFound("experiment", eid)
        return eid

    def _batch_key(eid: str) -> str:
        return f"experiment:{eid}"

    def _active_batch(eid: str) -> Optional[str]:
        return runner.active_job(_batch_key(eid))

    def _finalized(aid: str) -> bool:
        with store.view(aid) as st:
            return st.analysis.finalized_at is not None

    def _fresh_job(j):
        """The runner's in-memory copy of a job this computer runs (fresher than the row), else the row."""
        if j.status in (JobStatus.queued, JobStatus.running):
            try:
                return runner.get(j.id)
            except NotFound:
                pass
        return j

    def _job_dict(j) -> Optional[dict]:
        return json.loads(j.model_dump_json()) if j is not None else None

    def _exp_dict(e: Experiment) -> dict:
        d = json.loads(e.model_dump_json())
        d["status"] = ExperimentStatus.running.value if _active_batch(e.id) else ExperimentStatus.open.value
        return d

    def _last_batch(eid: str):
        jobs = store.list_jobs(experiment_id=eid, kind=JobKind.batch.value, limit=1)
        return _fresh_job(jobs[0]) if jobs else None

    def _layout(im) -> Optional[tuple]:
        """What must be identical for a channel mapping to be copied between slides."""
        if im is None:
            return None
        return (im.kind.value, im.dtype, len(im.channels), len({c.source_file for c in im.channels}),
                tuple(c.page for c in im.channels))

    def _slide_row(aid: str) -> dict:
        with store.view(aid) as st:
            a = st.analysis
            regs = [r for r in a.regions if r.kind == RegionKind.counting]
            row = {
                "id": a.id, "name": a.name, "order_in_experiment": a.order_in_experiment, "status": a.status.value,
                "export_status": exp_export.slide_status(a), "updated_at": a.updated_at,
                "finalized_at": a.finalized_at, "mapping_confirmed": engine.mapping_confirmed(a),
                "image": ({"width": a.image.width, "height": a.image.height, "kind": a.image.kind.value,
                           "dtype": a.image.dtype, "n_channels": len(a.image.channels)} if a.image else None),
                "has_grid": getattr(a.slide, "grid", None) is not None,
                "n_counting_regions": len(regs), "n_fields": sum(1 for r in regs if r.field_index is not None),
                "n_approved": sum(1 for r in regs if r.approved),
                "approved": bool(regs) and all(r.approved for r in regs),
                "review_queue": len(review_queue(a)) if a.objects else 0,
                "detection_ran_at": a.provenance.detection_ran_at, "engine": exp_export.slide_engine(a) or None,
                "counts": compute_counts(a)["slide"].model_dump(mode="json"),
            }
        jobs = store.list_jobs(analysis_id=aid, limit=1)
        row["last_job"] = _job_dict(_fresh_job(jobs[0])) if jobs else None
        return row

    def _experiment_view(eid: str) -> dict:
        e = store.get_experiment(eid)
        d = _exp_dict(e)
        d["slides"] = [_slide_row(aid) for aid in store.experiment_slide_ids(eid)]
        d["last_batch"] = _job_dict(_last_batch(eid))
        return d

    def _batch_view(eid: str) -> dict:
        b = _last_batch(eid)
        if b is None:
            return {"batch": None, "slides": [], "summary": None}
        latest: dict = {}
        for c in reversed(store.list_jobs(parent_id=b.id)):   # oldest first: the newest child of a slide wins
            latest[c.analysis_id] = _fresh_job(c)
        active = b.status in (JobStatus.queued, JobStatus.running)
        slides = []
        for p in (b.detail or {}).get("slides", []):
            c = latest.get(p["analysis_id"])
            if p.get("skip"):
                status = "skipped"
            elif c is not None:
                status = c.status.value
            else:
                status = "pending" if active else "not_run"
            slides.append({"analysis_id": p["analysis_id"], "name": p.get("name"), "order": p.get("order"),
                           "status": status, "skip_reason": p.get("skip"), "override": bool(p.get("override")),
                           "job_id": c.id if c else None, "progress": c.progress if c else 0.0,
                           "message": c.message if c else "", "error": c.error if c else None,
                           "started_at": c.started_at if c else None, "finished_at": c.finished_at if c else None})
        summary = {"total": sum(1 for s in slides if s["status"] != "skipped")}
        for k in ("done", "failed", "cancelled", "running", "queued", "pending", "not_run", "skipped"):
            summary[k] = sum(1 for s in slides if s["status"] == k)
        return {"batch": _job_dict(b), "slides": slides, "summary": summary}

    def _set_exp_status(eid: str, status: ExperimentStatus) -> None:
        try:
            e = store.get_experiment(eid)
            e.status = status
            store.save_experiment(e)
        except Exception:  # noqa: BLE001  (deleted meanwhile / database briefly unreachable)
            pass

    def _check_detection(req: Optional[DetectionRequest], raw: Any, where: str) -> Optional[Response]:
        if req is None:
            return None
        unk = unknown_keys(DetectionRequest, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(f"{where}.{u}" for u in unk),
                       {"fields": unk})
        if req.preview_window is not None:
            return err(422, "invalid_request", f"{where}.preview_window is not a run setting")
        return None

    @api.post("/experiments")
    def post_experiment(body: ExperimentBody, raw: Any = Depends(_raw_body)):
        unk = unknown_keys(ExperimentBody, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        name = (body.name or "").strip()
        if not name:
            return err(422, "invalid_name", "the experiment needs a name")
        bad = _check_detection(body.default_detection, (raw or {}).get("default_detection"), "default_detection")
        if bad is not None:
            return bad
        ts = engine.now()
        e = Experiment(id=new_id(), name=name[:120], created_at=ts, updated_at=ts, notes=body.notes or "",
                       default_grid=body.default_grid,
                       default_detection=body.default_detection or DetectionRequest())
        store.create_experiment(e)
        return _experiment_view(e.id)

    @api.get("/experiments")
    def list_experiments():
        by_exp: dict[str, list[dict]] = {}
        for s in store.list_analyses():
            if s.get("experiment_id"):
                by_exp.setdefault(s["experiment_id"], []).append(s)
        out = []
        for e in store.list_experiments():
            sl = by_exp.get(e.id, [])
            d = _exp_dict(e)
            d.update(n_slides=len(sl), n_detected=sum(1 for s in sl if s["status"] in ("in_review", "finalized")),
                     n_finalized=sum(1 for s in sl if s["status"] == "finalized"),
                     last_batch=_job_dict(_last_batch(e.id)))
            out.append(d)
        return out

    @api.get("/experiments/{eid}")
    def get_experiment(eid: str):
        return _experiment_view(eid_ok(eid))

    @api.put("/experiments/{eid}")
    def put_experiment(eid: str, body: ExperimentUpdateBody, raw: Any = Depends(_raw_body)):
        unk = unknown_keys(ExperimentUpdateBody, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        e = store.get_experiment(eid_ok(eid))
        given = set(raw or {}) if isinstance(raw, dict) else set()
        if "name" in given:
            name = (body.name or "").strip()
            if not name:
                return err(422, "invalid_name", "the experiment needs a name")
            e.name = name[:120]
        if "notes" in given:
            e.notes = body.notes or ""
        if "default_detection" in given:
            bad = _check_detection(body.default_detection, raw.get("default_detection"), "default_detection")
            if bad is not None:
                return bad
            e.default_detection = body.default_detection or DetectionRequest()
        if "default_grid" in given:
            e.default_grid = body.default_grid      # null clears it
        e.updated_at = engine.now()
        store.save_experiment(e)
        return _experiment_view(eid)

    @api.delete("/experiments/{eid}")
    def delete_experiment(eid: str, delete_slides: bool = False):
        e = store.get_experiment(eid_ok(eid))
        if _active_batch(eid):
            return err(409, "batch_in_progress", "a batch run of this experiment is active; cancel it first")
        ids = store.experiment_slide_ids(eid)
        if delete_slides:
            busy = [a for a in ids if runner.active_job(a)]
            if busy:
                return err(409, "job_in_progress", "a detection job is running on a slide of this experiment",
                           {"analysis_ids": busy})
            fin = [a for a in ids if _finalized(a)]
            if fin:
                return err(409, "analysis_finalized", "finalized slides are not deleted; reopen them first, or "
                           "delete the experiment without delete_slides (its slides are kept)", {"analysis_ids": fin})
            for a in ids:
                store.delete_analysis(a)
                images.forget(a)
        else:
            for a in ids:
                with store.edit(a) as st:
                    st.analysis.experiment_id = None
                    st.analysis.order_in_experiment = 0
        store.delete_jobs(experiment_id=e.id, kind=JobKind.batch.value)
        store.delete_experiment(e.id)
        return {"ok": True, "deleted_slides": ids if delete_slides else [], "detached_slides": [] if delete_slides else ids}

    def _next_order(eid: str) -> int:
        orders = [s.get("order_in_experiment") or 0 for s in store.list_analyses() if s.get("experiment_id") == eid]
        return max(orders, default=0) + 1

    @api.post("/experiments/{eid}/slides")
    def post_experiment_slides(eid: str, body: ExperimentSlidesBody):
        store.get_experiment(eid_ok(eid))
        ids = list(dict.fromkeys(body.analysis_ids))
        for a in ids:
            aid_ok(a)
            if not store.exists(a):
                raise NotFound("analysis", a)
        other = []
        for a in ids:
            with store.view(a) as st:
                if st.analysis.experiment_id not in (None, eid):
                    other.append({"analysis_id": a, "experiment_id": st.analysis.experiment_id})
        if other:
            return err(409, "in_other_experiment", "these analyses belong to another experiment; remove them "
                       "from it first", {"analyses": other})
        order = _next_order(eid)
        for a in ids:
            with store.edit(a) as st:
                if st.analysis.experiment_id == eid:
                    continue
                st.analysis.experiment_id = eid
                st.analysis.order_in_experiment = order
                order += 1
        return _experiment_view(eid)

    @api.delete("/experiments/{eid}/slides/{aid}")
    def delete_experiment_slide(eid: str, aid: str):
        """Detach one slide (the analysis itself is kept)."""
        store.get_experiment(eid_ok(eid))
        with store.edit(aid_ok(aid)) as st:
            if st.analysis.experiment_id != eid:
                raise ReviewError("not_in_experiment", "this analysis is not part of the experiment", 404)
            st.analysis.experiment_id = None
            st.analysis.order_in_experiment = 0
        return _experiment_view(eid)

    @api.put("/experiments/{eid}/order")
    def put_experiment_order(eid: str, body: ExperimentSlidesBody):
        """Reorder: ``analysis_ids`` must list every slide of the experiment exactly once."""
        store.get_experiment(eid_ok(eid))
        cur = store.experiment_slide_ids(eid)
        if sorted(body.analysis_ids) != sorted(cur):
            return err(422, "invalid_order", "analysis_ids must list every slide of the experiment exactly once",
                       {"analysis_ids": cur})
        for i, a in enumerate(body.analysis_ids, start=1):
            with store.edit(a) as st:
                st.analysis.order_in_experiment = i
        return _experiment_view(eid)

    @api.post("/experiments/{eid}/import")
    def post_experiment_import(eid: str, files: list[UploadFile] = File(...)):
        """Many TIFFs, one slide each, through the same upload path as POST /analyses. The confirmed channel
        mapping of the first confirmed slide is copied to a new slide whose file layout is identical."""
        store.get_experiment(eid_ok(eid))
        template = None
        for a in store.experiment_slide_ids(eid):
            with store.view(a) as st:
                if engine.mapping_confirmed(st.analysis):
                    template = (a, _layout(st.analysis.image),
                                [{"index": c.index, "role": c.role.value} for c in st.analysis.image.channels])
                    break
        order = _next_order(eid)
        created, errors = [], []
        for f in files:
            info = {"mapping_copied_from": None}

            def prepare(st, order=order, info=info):
                st.analysis.experiment_id = eid
                st.analysis.order_in_experiment = order
                if template is not None and _layout(st.analysis.image) == template[1]:
                    engine.set_channel_mapping(st, template[2])
                    info["mapping_copied_from"] = template[0]

            try:
                a = json.loads(_ingest_upload([f], None, prepare=prepare))
            except ReviewError as ex:
                errors.append({"file": safe_filename(f.filename), "code": ex.code, "message": ex.message})
                continue
            order += 1
            created.append({"id": a["id"], "name": a["name"], "order_in_experiment": a["order_in_experiment"],
                            "status": a["status"], "mapping_copied_from": info["mapping_copied_from"]})
        if not created:
            return err(422, "import_failed", "no file could be imported", {"errors": errors})
        return {"created": created, "errors": errors}

    # ............................................................. unattended run
    def _run_batch_slide(bid: str, eid: str, aid: str, req: DetectionRequest) -> JobStatus:
        """One slide of a batch: start its detect job and wait for it. A slide that cannot start is recorded as a
        failed child job carrying the reason, so it shows up like any other failure."""
        finished = threading.Event()
        try:
            job = _start_detection(aid, req, parent_id=bid, experiment_id=eid, require_counting=True,
                                   then=lambda _j: finished.set())
        except Exception as ex:  # noqa: BLE001  ReviewError (mapping, region, finalized, busy) or anything else
            msg = f"{ex.code}: {ex.message}" if isinstance(ex, ReviewError) else f"{type(ex).__name__}: {ex}"
            try:
                store.create_job(aid, kind=JobKind.detect.value, parent_id=bid, experiment_id=eid,
                                 status=JobStatus.failed, error=msg)
            except Exception:  # noqa: BLE001
                pass
            return JobStatus.failed
        while not finished.wait(0.5):
            if runner.get(job.id).status not in (JobStatus.queued, JobStatus.running):
                break   # finished without the callback (should not happen; never wait forever)
        return runner.get(job.id).status

    def _batch_work(eid: str, plan: list[dict], reqs: dict[str, DetectionRequest], started: threading.Event,
                    holder: dict):
        def work(progress, cancelled):
            started.wait(10)
            bid = holder["id"]
            notes: list[str] = []
            models = sorted({r.spotiflow.model_name for r in reqs.values() if r.engine != EngineChoice.bigfish})
            for m in models:   # once, before the first slide, so a download happens now and not in the night
                if cancelled():
                    break
                progress(0.0, f"preparing Spotiflow model '{m}'")
                try:
                    notes.append(_prepare_model(m, lambda f, msg="", m=m: progress(0.0, f"model '{m}': {msg}")))
                except Exception as ex:  # noqa: BLE001  each slide then reports the engine problem itself
                    text = ex.message if isinstance(ex, ReviewError) else f"{type(ex).__name__}: {ex}"
                    notes.append(f"WARNING: Spotiflow model '{m}' could not be prepared: {text}")
            run = [p for p in plan if not p.get("skip")]
            ok = failed = 0
            tail = (" | " + " | ".join(notes)) if notes else ""
            for i, p in enumerate(run):
                if cancelled():
                    return (JobStatus.cancelled,
                            f"cancelled after {i} of {len(run)} slides ({ok} done, {failed} failed){tail}")
                progress(i / len(run), f"slide {i + 1} / {len(run)}: {p['name']} ({ok} done, {failed} failed)")
                if _run_batch_slide(bid, eid, p["analysis_id"], reqs[p["analysis_id"]]) == JobStatus.done:
                    ok += 1
                else:
                    failed += 1
                progress((i + 1) / len(run), f"{ok + failed} / {len(run)} slides ({ok} done, {failed} failed)")
            n_skip = len(plan) - len(run)
            msg = f"{ok} / {len(run)} done, {failed} failed" + (f", {n_skip} skipped" if n_skip else "") + tail
            return (JobStatus.done_with_failures if failed else JobStatus.done, msg)

        return work

    @api.post("/experiments/{eid}/run")
    def post_experiment_run(eid: str, body: ExperimentRunBody, raw: Any = Depends(_raw_body)):
        unk = unknown_keys(ExperimentRunBody, raw)
        if unk:
            return err(422, "unknown_fields", "unknown field(s): " + ", ".join(unk), {"fields": unk})
        e = store.get_experiment(eid_ok(eid))
        if _active_batch(eid):
            return err(409, "batch_in_progress", "a batch run of this experiment is already active")
        ids = store.experiment_slide_ids(eid)
        if body.analysis_ids is not None:
            extra = [a for a in body.analysis_ids if a not in ids]
            if extra:
                return err(422, "not_in_experiment", "analysis_ids must be slides of this experiment",
                           {"analysis_ids": extra})
            want = set(body.analysis_ids)
            ids = [a for a in ids if a in want]
        extra = [a for a in body.overrides if a not in ids]
        if extra:
            return err(422, "not_in_experiment", "overrides must be keyed by slides of this run",
                       {"analysis_ids": extra})
        for a, req in body.overrides.items():
            bad = _check_detection(req, ((raw or {}).get("overrides") or {}).get(a), f"overrides.{a}")
            if bad is not None:
                return bad
        plan, reqs = [], {}
        for a in ids:
            with store.view(a) as st:
                an = st.analysis
                skip = None
                if an.finalized_at is not None:
                    skip = "finalized"
                elif body.only_missing and an.provenance.detection_ran_at is not None:
                    skip = "already_detected"
                plan.append({"analysis_id": a, "name": an.name, "order": an.order_in_experiment, "skip": skip,
                             "override": a in body.overrides})
            if skip is None:
                reqs[a] = body.overrides.get(a) or e.default_detection
        if not reqs:
            return err(422, "nothing_to_run", "no slide to run (all finalized" +
                       (" or already detected)" if body.only_missing else ")"), {"slides": plan})
        detail = {"slides": plan, "only_missing": body.only_missing,
                  "default_detection": e.default_detection.model_dump(mode="json"),
                  "overrides": {a: r.model_dump(mode="json") for a, r in body.overrides.items()}}
        started, holder = threading.Event(), {}
        _set_exp_status(eid, ExperimentStatus.running)
        try:
            job = runner.submit(_batch_key(eid), _batch_work(eid, plan, reqs, started, holder),
                                lambda _j: _set_exp_status(eid, ExperimentStatus.open), kind=JobKind.batch.value,
                                experiment_id=eid, detail=detail, dedicated=True)
        except RuntimeError:
            _set_exp_status(eid, ExperimentStatus.open)
            return err(409, "batch_in_progress", "a batch run of this experiment is already active")
        holder["id"] = job.id
        started.set()
        return _json(job)

    @api.get("/experiments/{eid}/run")
    def get_experiment_run(eid: str):
        store.get_experiment(eid_ok(eid))
        return _batch_view(eid)

    # ............................................................. exports
    def _experiment_slides(eid: str) -> list[Analysis]:
        out = []
        for a in store.experiment_slide_ids(eid):
            with store.view(a) as st:
                out.append(st.analysis.model_copy(deep=True))
        return out

    @api.get("/experiments/{eid}/export.csv")
    def get_experiment_csv(eid: str, series: Optional[list[str]] = Query(None)):
        e = store.get_experiment(eid_ok(eid))
        sel = exp_export.parse_series(series)
        text, included, skipped = exp_export.build_csv(e, _experiment_slides(eid), sel)
        fn = f"{exporter.safe_stem(e.name)}_experiment.csv"
        return Response(text.encode("utf-8-sig"), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="{fn}"', "X-Slides-Included": str(len(included)),
            "X-Slides-Skipped": str(len(skipped))})

    @api.get("/experiments/{eid}/provenance.json")
    def get_experiment_provenance(eid: str, series: Optional[list[str]] = Query(None)):
        e = store.get_experiment(eid_ok(eid))
        sel = exp_export.parse_series(series)
        slides = _experiment_slides(eid)
        _, included, skipped = exp_export.build_csv(e, slides, sel)
        prov = exp_export.provenance(e, [(a, store.get_engine_runs(a.id)) for a in slides], _batch_view(eid),
                                     sel, included, skipped)
        fn = f"{exporter.safe_stem(e.name)}_provenance.json"
        return Response(json.dumps(prov, indent=2, default=str), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{fn}"'})

    app.include_router(api)

    # ------------------------------------------------------------- static frontend
    dist = Path(os.environ.get("POLONY_FRONTEND_DIST") or Path(polony_app.__file__).resolve().parents[2] / "frontend" / "dist")
    if not dist.is_absolute():
        dist = dist.resolve()
    if not (dist / "index.html").exists():  # prebuilt UI shipped with the package (see Makefile `web`)
        dist = Path(polony_app.__file__).resolve().parent / "webdist"
    if (dist / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app
