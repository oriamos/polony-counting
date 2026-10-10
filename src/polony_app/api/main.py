"""FastAPI application. `create_app(data_dir)` builds an isolated app instance."""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
import json
import os
import re
import shutil
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import polony_app
from polony_app.export import exporter
from polony_app.review import engine
from polony_app.review.counts import compute_counts, review_queue
from polony_app.review.state import AnalysisState, ReviewError
from polony_app.schemas import (
    Analysis,
    ChannelRole,
    DetectionRequest,
    JobStatus,
    Region,
    ReportingMode,
    SlideMeta,
)
from polony_app.store import NotFound, Store
from polony_app.store.csv_import import parse_csv
from polony_app.store.db import new_id
from polony_app.store.jobs import JobRunner

from .images import VIEWS, ImageService, role_index

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


class CsvMapBody(BaseModel):
    color_mapping: Optional[Literal["red", "green", "yellow"]] = None


# ------------------------------------------------------------------ helpers
def err(status: int, code: str, message: str, detail: dict | None = None) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "detail": detail}, status_code=status)


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


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("POLONY_DATA_DIR") or "./data")
    store = Store(data_dir)
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

    # ------------------------------------------------------------- analyses
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
        tmp = store.data_dir / "tmp" / uuid.uuid4().hex
        tmp.mkdir(parents=True)
        aid = None
        try:
            staged: list[tuple[str, Path]] = []
            for i, f in enumerate(files):
                orig = f.filename or f"file{i}.tif"
                sname = safe_filename(orig, f"file{i}.tif")
                if not sname.lower().endswith((".tif", ".tiff")):
                    return err(415, "unsupported_file", f"'{sname}': only .tif/.tiff files are accepted")
                dst = tmp / f"{i}_{sname}"
                size = 0
                with open(dst, "wb") as out:
                    while chunk := await f.read(1 << 20):
                        size += len(chunk)
                        if size > max_upload:
                            return err(413, "too_large",
                                       f"'{sname}' exceeds the {max_upload // 2**20} MB limit")
                        out.write(chunk)
                if size == 0:
                    return err(422, "empty_file", f"'{sname}' is empty")
                staged.append((orig_name := sname, dst))
            default_name = Path(staged[0][0]).stem
            aid = store.create_analysis((name or "").strip()[:120] or default_name)
            from polony_app.imaging.ingest import IngestError, ingest_files

            try:
                meta = ingest_files(staged, store.originals_dir(aid))
            except IngestError as e:
                store.delete_analysis(aid)
                return err(422, "ingest_error", str(e))
            except Exception as e:  # noqa: BLE001  unreadable/corrupt TIFF
                store.delete_analysis(aid)
                return err(422, "unreadable_image", f"could not read image: {type(e).__name__}: {e}")
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
            with store.edit(aid) as st:
                st.analysis.image = meta
                engine.refresh_status(st)
                out_json = st.analysis.model_dump_json()
            return Response(out_json, media_type="application/json", status_code=200)
        except Exception:
            if aid:
                try:
                    store.delete_analysis(aid)
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
            role_index(st.analysis, "green")  # mapping must be confirmed
            engine.set_alignment(st, body.dy, body.dx)
            images.forget(aid)
        return {"dy": body.dy, "dx": body.dx, "peak_corr": None, "applied": True}

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
        return (max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))

    @api.get("/analyses/{aid}/image/{view}")
    def get_image(aid: str, view: str, lo: Optional[float] = None, hi: Optional[float] = None,
                  gamma: float = 1.0, x0: Optional[int] = None, y0: Optional[int] = None,
                  x1: Optional[int] = None, y1: Optional[int] = None,
                  max_dim: Optional[int] = None, colormap: Literal["gray", "red", "green"] = "gray"):
        if view not in VIEWS:
            return err(422, "invalid_view", f"view must be one of {', '.join(VIEWS)}")
        if max_dim is not None and not 1 <= max_dim <= 8192:
            return err(422, "invalid_param", "max_dim must be in 1..8192")
        with store.view(aid_ok(aid)) as st:
            if st.analysis.image is None:
                return err(409, "no_image", "analysis has no image")
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
            return images.histogram(st, role)

    @api.get("/analyses/{aid}/thumb/{object_id}")
    def get_thumb(aid: str, object_id: str, size: int = 64, view: str = "composite"):
        if not 8 <= size <= 512:
            return err(422, "invalid_param", "size must be in 8..512")
        with store.view(aid_ok(aid)) as st:
            o = next((o for o in st.analysis.objects if o.id == object_id), None)
            if o is None:
                raise NotFound("object", object_id)
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

    @api.put("/analyses/{aid}/settings")
    def put_settings(aid: str, body: SettingsBody):
        if body.slide is not None:
            s = body.slide
            for f in ("sample_volume_ul", "concentration_factor", "pixel_size_um"):
                v = getattr(s, f)
                if v is not None and not (v > 0 and v < float("inf")):
                    return err(422, "invalid_slide", f"{f} must be a positive number or null")
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

    @api.post("/analyses/{aid}/detect")
    def post_detect(aid: str, body: DetectionRequest):
        aid_ok(aid)
        if body.preview_window is not None:
            return err(422, "invalid_request", "preview_window is only valid for /detect/preview")
        with store.edit(aid) as st:
            engine.guard_editable(st)
            _detection_inputs(st)
            if runner.active_job(aid):
                raise ReviewError("job_in_progress", "a detection job is already running", 409)
            st.analysis.detection_request = body
            start_sig = _sig(st.analysis, st)
            regions = [r.model_copy(deep=True) for r in st.analysis.regions]
            red, green = None, None  # loaded in the worker thread
            engine.refresh_status(st, job_running=True)

        def work(progress, cancelled):
            from polony_app.detection import pipeline

            with store.view(aid) as st:
                red, green = images.raw_pair(st)
            progress(0.02, "images loaded")
            res = pipeline.run_detection(red, green, regions, body, progress, cancelled)
            if cancelled():
                return "cancelled"
            with store.edit(aid) as st:
                if _sig(st.analysis, st) != start_sig:
                    raise RuntimeError(
                        "regions, channel mapping or alignment changed while detection ran; "
                        "results discarded - run detection again"
                    )
                engine.apply_detection_result(
                    st, list(res.objects), body, dict(res.package_versions), list(res.notes)
                )
            return f"{len(res.objects)} objects detected"

        def finish(job):
            try:
                with store.edit(aid) as st:
                    engine.refresh_status(st, job_running=False)
            except Exception:  # noqa: BLE001
                pass

        try:
            job = runner.submit(aid, work, finish)
        except RuntimeError:
            finish(None)
            return err(409, "job_in_progress", "a detection job is already running")
        return _json(job)

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
        from polony_app.detection import pipeline

        res = pipeline.run_detection(red, green, regions, body, lambda f, m="": None,
                                     lambda: False, window=body.preview_window)
        return {"objects": [o.model_dump(mode="json") for o in res.objects], "notes": list(res.notes)}

    @api.get("/jobs/{jid}")
    def get_job(jid: str):
        return _json(runner.get(jid))

    @api.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        return _json(runner.cancel(jid))

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
    @api.get("/analyses/{aid}/export")
    def get_export(aid: str, kind: str = "bundle"):
        if kind not in exporter.KINDS:
            return err(422, "invalid_kind", f"kind must be one of {', '.join(exporter.KINDS)}")
        with store.view(aid_ok(aid)) as st:
            a = st.analysis
            fn = exporter.export_filename(a, kind)
            if kind == "objects":
                body, mt = exporter.objects_csv(a).encode("utf-8"), "text/csv"
            elif kind == "regions":
                body, mt = exporter.regions_csv(a).encode("utf-8"), "text/csv"
            elif kind == "slide":
                body, mt = exporter.slide_csv(a).encode("utf-8"), "text/csv"
            elif kind == "provenance":
                body, mt = exporter.provenance_json(st).encode("utf-8"), "application/json"
            else:
                body, mt = exporter.bundle_zip(st), "application/zip"
            status = exporter.status_label(a)
        return Response(body, media_type=mt, headers={
            "Content-Disposition": f'attachment; filename="{fn}"', "X-Export-Status": status})

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

    app.include_router(api)

    # ------------------------------------------------------------- static frontend
    dist = Path(os.environ.get("POLONY_FRONTEND_DIST") or Path(polony_app.__file__).resolve().parents[2] / "frontend" / "dist")
    if not dist.is_absolute():
        dist = dist.resolve()
    if (dist / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app
