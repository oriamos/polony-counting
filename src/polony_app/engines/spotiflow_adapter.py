"""Spotiflow (learned spot detector) behind the engine-neutral contract.

* ``import polony_app.engines`` never imports spotiflow / torch; they are imported lazily when used.
* The image is passed to the model in its ORIGINAL dtype and values (never an 8-bit conversion). Spotiflow
  itself casts to float32 and (normalizer 'auto') percentile-normalises the array it is given (1 / 99.8 %);
  that normalisation is therefore computed on the model INPUT (the full page, or the padded valid-area
  bounding box when ``crop_to_valid``) and is recorded in ``EngineRunInfo.preprocessing``.
* Score = the model's heatmap value at the detected point (``details.prob``), a RAW model score
  ('spotiflow_prob_raw'), not a calibrated probability.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..detection.errors import DetectionCancelled
from ..schemas import EngineName, EngineRunInfo, SpotiflowParams
from .base import (EngineError, EngineResult, EngineUnavailable, PointValidity, Progress, build_detections,
                   new_run_id, raster_point_validity, utcnow_iso)

SCORE_KIND = "spotiflow_prob_raw"
DEFAULT_MODEL = "general"
CROP_PAD_PX = 32
SUPPORTED_DOWNSCALE = (1, 2, 4, 8)
_SIDECAR = ".polony_checksums.json"
_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
_PARAM_FIELDS = set(SpotiflowParams.model_fields)

__all__ = ["SpotiflowEngine", "ModelInfo", "ensure_model", "model_cache_dir", "list_models", "device_info",
           "SCORE_KIND", "DEFAULT_MODEL"]


# ------------------------------------------------------------------ lazy imports / environment
def _have(name: str) -> bool:
    if name in sys.modules and sys.modules[name] is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch():
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise EngineUnavailable(f"PyTorch is not installed or failed to import ({exc}); "
                                "install the Spotiflow extra (pip install spotiflow, CPU build of torch is enough)") from exc


def _spotiflow_cls():
    try:
        return importlib.import_module("spotiflow.model").Spotiflow
    except Exception as exc:
        raise EngineUnavailable(f"the 'spotiflow' package is not installed or failed to import ({exc}); "
                                "install it with: pip install spotiflow") from exc


def _pkg_version(name: str) -> str:
    try:
        mod = sys.modules.get(name) or importlib.import_module(name)
        v = getattr(mod, "__version__", None)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return ""


def model_cache_dir() -> Path:
    env = os.environ.get("POLONY_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    from ..config import home_dir
    return home_dir() / "models" / "spotiflow"


def device_info() -> dict:
    """{'cuda': bool, 'mps': bool, 'cpu': True}; never raises (torch missing => only cpu)."""
    out = {"cuda": False, "mps": False, "cpu": True}
    try:
        t = importlib.import_module("torch")
        try:
            out["cuda"] = bool(t.cuda.is_available())
        except Exception:
            pass
        try:
            mps = getattr(getattr(t, "backends", None), "mps", None)
            out["mps"] = bool(mps is not None and mps.is_available())
        except Exception:
            pass
    except Exception:
        pass
    return out


def resolve_device(requested: str) -> tuple[str, Optional[str]]:
    """(device, note). 'auto' -> cuda, then mps, then cpu. A requested but unavailable device falls back to
    cpu with a note (never an error)."""
    info = device_info()
    if requested == "auto":
        return ("cuda" if info["cuda"] else "mps" if info["mps"] else "cpu"), None
    if requested == "cpu" or info.get(requested):
        return requested, None
    return "cpu", f"requested device '{requested}' is not available on this machine; fell back to cpu"


# ------------------------------------------------------------------ models + checksums
@dataclass
class ModelInfo:
    name: str
    version: str
    checksum_sha256: str
    path: Path
    cached: bool                       # True = already on disk (no download happened in this call)
    files: dict = field(default_factory=dict)   # weights artifact -> sha256
    prob_thresh: Optional[float] = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_files(folder: Path) -> list[Path]:
    ws = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES)
    return ws or sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


def _folder_checksums(folder: Path) -> dict[str, str]:
    """sha256 of every weights artifact; computed once, then read from a sidecar json keyed by (size, mtime)."""
    side = folder / _SIDECAR
    try:
        cache = json.loads(side.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    out, changed = {}, False
    for p in _weights_files(folder):
        st = p.stat()
        ent = cache.get(p.name)
        if isinstance(ent, dict) and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns \
                and isinstance(ent.get("sha256"), str):
            out[p.name] = ent["sha256"]
        else:
            out[p.name] = _sha256_file(p)
            cache[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": out[p.name]}
            changed = True
    if changed:
        try:
            side.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError:
            pass  # read-only cache: recomputed next time
    return out


def _folder_ready(folder: Path) -> bool:
    return folder.is_dir() and (folder / "config.yaml").is_file() and any(
        p.suffix.lower() in _WEIGHT_SUFFIXES for p in folder.iterdir())


def _release_version(name: str) -> str:
    try:
        from spotiflow.model.pretrained import _REGISTERED
        url = _REGISTERED[name].url
        return "spotiflow-models " + url.rstrip("/").split("/")[-2]
    except Exception:
        return ""


def _resolve_local(model_name: str, cache_dir: Path) -> Optional[Path]:
    p = Path(model_name).expanduser()
    if p.is_dir() and (p / "config.yaml").is_file():
        return p
    q = cache_dir / model_name
    return q if _folder_ready(q) else None


def list_models(cache_dir: Optional[Path] = None) -> list[dict]:
    """Pretrained models known to the installed spotiflow + whether each is cached (no network)."""
    cache_dir = Path(cache_dir) if cache_dir else model_cache_dir()
    names: list[str] = []
    try:
        names = [n for n in importlib.import_module("spotiflow.model.pretrained").list_registered()]
    except Exception:
        names = [DEFAULT_MODEL]
    out = []
    for n in sorted(names):
        folder = cache_dir / n
        ready = _folder_ready(folder)
        ck, ver = None, None
        if ready:
            try:
                cks = _folder_checksums(folder)
                ck = cks.get("best.pt") or next(iter(cks.values()), None)
            except OSError:
                pass
            ver = _release_version(n) or None
        out.append({"name": n, "cached": ready, "version": ver, "checksum": ck})
    return out


def ensure_model(model_name: str = DEFAULT_MODEL, cache_dir: Optional[Path] = None,
                 progress: Progress = lambda f, m: None) -> ModelInfo:
    """Make the model available on disk (download ONCE, never when cached) and return its identity.

    ``model_name``: an official pretrained name ('general', ...), the name of a folder inside the cache dir
    (e.g. a registered fine-tuned model) or a path to a Spotiflow model folder (config.yaml + weights)."""
    cache_dir = Path(cache_dir) if cache_dir else model_cache_dir()
    folder = _resolve_local(model_name, cache_dir)
    cached = folder is not None
    if folder is None:
        _spotiflow_cls()  # clear "not installed" error first
        try:
            from spotiflow.model.pretrained import get_pretrained_model_path, list_registered
        except Exception as exc:
            raise EngineUnavailable(f"cannot import spotiflow's model registry: {exc}") from exc
        if model_name not in list_registered():
            raise EngineUnavailable(f"unknown Spotiflow model '{model_name}' (pretrained: "
                                    f"{', '.join(sorted(list_registered()))}; or give a model folder)")
        progress(0.0, f"downloading Spotiflow model '{model_name}' (once; cached in {cache_dir})")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            folder = Path(get_pretrained_model_path(model_name, cache_dir=cache_dir))
        except Exception as exc:
            raise EngineUnavailable(
                f"could not download Spotiflow model '{model_name}' into {cache_dir}: {exc}. "
                "Check the internet connection, or copy the model folder there manually.") from exc
        if not _folder_ready(folder):
            raise EngineUnavailable(f"downloaded model folder {folder} is incomplete (config.yaml / weights missing)")
        progress(0.9, "model downloaded")
    try:
        files = _folder_checksums(folder)
    except OSError as exc:
        raise EngineError(f"cannot read model files in {folder}: {exc}") from exc
    # the loaded checkpoint is 'best' (Spotiflow default): its hash is the model checksum
    best = next((k for k in files if Path(k).stem == "best"), None)
    if best is not None:
        checksum = files[best]
    else:
        checksum = hashlib.sha256("".join(f"{k}:{files[k]}" for k in sorted(files)).encode()).hexdigest()
    progress(1.0, "model ready")
    return ModelInfo(name=model_name, version=_release_version(model_name) or ("local" if cached else ""),
                     checksum_sha256=checksum, path=folder, cached=cached, files=files)


_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


def _load_model(info: ModelInfo, device: str):
    key = (str(info.path), info.checksum_sha256, device)
    with _MODEL_LOCK:
        m = _MODEL_CACHE.get(key)
        if m is not None:
            return m
        cls = _spotiflow_cls()
        try:
            m = cls.from_folder(str(info.path), map_location=device)
        except Exception as exc:
            raise EngineError(f"failed to load Spotiflow model '{info.name}' from {info.path} "
                              f"on {device}: {type(exc).__name__}: {exc}") from exc
        if len(_MODEL_CACHE) >= 3:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = m
        return m


def clear_model_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


# ------------------------------------------------------------------ engine
class SpotiflowEngine:
    name = EngineName.spotiflow

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir or model_cache_dir()

    def available(self, model_name: str = DEFAULT_MODEL) -> tuple[bool, str]:
        if not _have("spotiflow"):
            return False, "the 'spotiflow' package is not installed (pip install spotiflow)"
        if not _have("torch"):
            return False, "PyTorch is not installed (Spotiflow needs torch; the CPU build is enough)"
        local = _resolve_local(model_name, self.cache_dir)
        if local is not None:
            return True, f"Spotiflow ready; model '{model_name}' cached in {local}"
        return True, (f"Spotiflow installed; model '{model_name}' is not cached yet and will be downloaded once "
                      f"(~90 MB) into {self.cache_dir}")

    def version(self) -> str:
        return _pkg_version("spotiflow")

    def describe(self) -> dict:
        ok, why = self.available()
        return {"name": "spotiflow", "available": ok, "reason": why, "version": self.version() if ok else "",
                "device": device_info(), "models": list_models(self.cache_dir)}

    # ---------------------------------------------------------------- detection
    def detect_page(self, image: np.ndarray, *, analysis_id: str, page: int, channel_role: Optional[str],
                    valid_mask: Optional[np.ndarray], params: dict, image_sha256: dict,
                    progress: Progress = lambda f, m: None, cancelled: Callable[[], bool] = lambda: False,
                    point_validity: Optional[PointValidity] = None, run_id: Optional[str] = None) -> EngineResult:
        params = dict(params or {})
        extras = {"crop_to_valid": bool(params.pop("crop_to_valid", True)),
                  "crop_pad_px": int(params.pop("crop_pad_px", CROP_PAD_PX))}
        try:
            sp = SpotiflowParams(**{k: v for k, v in params.items() if k in _PARAM_FIELDS})
        except Exception as exc:
            raise EngineError(f"invalid Spotiflow parameters: {exc}") from exc
        d = int(sp.downscale)
        if d not in SUPPORTED_DOWNSCALE:
            raise EngineError(f"unsupported Spotiflow downscale {sp.downscale!r}; use one of {list(SUPPORTED_DOWNSCALE)}")
        if d > 1 and sp.subpix:
            raise EngineError("Spotiflow cannot compute sub-pixel positions when downscale > 1; "
                              "set subpix to false/auto or downscale to 1")
        image = np.asarray(image)
        if image.ndim != 2:
            raise EngineError(f"Spotiflow adapter needs a 2-D page, got shape {image.shape}")
        if image.dtype.kind not in "uif":
            raise EngineError(f"unsupported image dtype {image.dtype}")
        if image.dtype.kind == "f" and not np.isfinite(image).all():
            raise EngineError("image contains NaN/inf")
        if valid_mask is not None and valid_mask.shape != image.shape:
            raise EngineError("valid_mask shape differs from image shape")
        h, w = image.shape
        run_id = run_id or new_run_id()
        started = utcnow_iso()
        t0 = time.perf_counter()
        notes: list[str] = []
        if cancelled():
            raise DetectionCancelled()

        ok, why = self.available(sp.model_name)
        if not ok:
            raise EngineUnavailable(why)
        mi = ensure_model(sp.model_name, self.cache_dir, lambda f, m: progress(0.1 * f, m))
        device, dnote = resolve_device(sp.device)
        if dnote:
            notes.append(dnote)

        # ---- crop to the bounding box of the valid area (+ padding), exact integer offset
        x0, y0, x1, y1 = 0, 0, w, h
        if extras["crop_to_valid"] and valid_mask is not None:
            if not valid_mask.any():
                run = EngineRunInfo(
                    run_id=run_id, analysis_id=analysis_id, image_sha256=dict(image_sha256 or {}),
                    engine=EngineName.spotiflow, engine_version=self.version(), model_name=mi.name,
                    model_version=mi.version, model_checksum=mi.checksum_sha256, page=int(page),
                    channel_role=channel_role, params=sp.model_dump(mode="json"), preprocessing={},
                    started_at=started, finished_at=utcnow_iso(), device=device, duration_s=0.0, status="skipped",
                    notes=["no valid pixels in the counting area: nothing to detect"])
                return EngineResult(run=run, detections=[])
            rows, cols = np.nonzero(valid_mask.any(axis=1))[0], np.nonzero(valid_mask.any(axis=0))[0]
            pad = extras["crop_pad_px"]
            x0, x1 = max(0, int(cols[0]) - pad), min(w, int(cols[-1]) + 1 + pad)
            y0, y1 = max(0, int(rows[0]) - pad), min(h, int(rows[-1]) + 1 + pad)
        trimmed_px = {"right": 0, "bottom": 0}
        if d > 1:
            # Spotiflow needs both sides divisible by d: drop the remainder at the bottom/right (never pad)
            trimmed_px = {"right": (x1 - x0) % d, "bottom": (y1 - y0) % d}
            x1 -= trimmed_px["right"]
            y1 -= trimmed_px["bottom"]
            if (x1 - x0) // d < 2 or (y1 - y0) // d < 2:
                raise EngineError(f"image region {x1 - x0}x{y1 - y0} px is too small for downscale {d}")
        sub = np.ascontiguousarray(image[y0:y1, x0:x1])
        if sub.size == 0:
            raise EngineError("empty crop")
        if d > 1:
            if trimmed_px["right"] or trimmed_px["bottom"]:
                notes.append(f"downscale {d}: trimmed {trimmed_px['right']} column(s) at the right and "
                             f"{trimmed_px['bottom']} row(s) at the bottom so the sides are divisible by {d}")

        normalizer = "auto" if sp.normalizer == "auto" else None
        kw = dict(prob_thresh=sp.prob_thresh, n_tiles=sp.n_tiles, max_tile_size=sp.max_tile_size,
                  min_distance=sp.min_distance, exclude_border=sp.exclude_border, subpix=sp.subpix,
                  normalizer=normalizer, verbose=True)
        model_in = sub
        if d > 1:
            # We downscale OURSELVES (exactly what Spotiflow 0.6.5 does for scale=1/d: float32 + scipy zoom order 1)
            # instead of passing scale=: Spotiflow's own `scale` returns points in the ORIGINAL frame on its
            # single-tile path (peaks found on the up-sampled heatmap) but in the DOWNSCALED frame on its tiled
            # path, so the coordinate frame would depend on image size. Predicting on our array always yields
            # downscaled-frame points, mapped back below.
            from scipy.ndimage import zoom
            model_in = zoom(sub.astype(np.float32), 1.0 / d, order=1)
            kw["subpix"] = False

        def wrapper(it, total=None, **_):
            k = 0
            for item in it:
                if cancelled():
                    raise DetectionCancelled()
                progress(0.15 + 0.8 * (k / max(total or 1, 1)), f"Spotiflow tile {k + 1}/{total or '?'}")
                k += 1
                yield item

        pts = prob = None
        used_device = device
        for attempt in (0, 1):
            try:
                model = _load_model(mi, used_device)
                progress(0.12, f"Spotiflow inference on {used_device}")
                pts, details = model.predict(model_in, device=used_device, progress_bar_wrapper=wrapper, **kw)
                prob = getattr(details, "prob", None)
                break
            except DetectionCancelled:
                raise
            except EngineError as exc:
                if used_device != "cpu" and attempt == 0:
                    notes.append(f"model load on {used_device} failed ({exc}); retried on cpu")
                    used_device = "cpu"
                    continue
                raise
            except Exception as exc:
                if used_device != "cpu" and attempt == 0:
                    notes.append(f"inference on {used_device} failed ({type(exc).__name__}: {exc}); retried on cpu")
                    used_device = "cpu"
                    continue
                raise EngineError(f"Spotiflow inference failed on {used_device} for page {page}: "
                                  f"{type(exc).__name__}: {exc}") from exc
        if cancelled():
            raise DetectionCancelled()

        pts = np.asarray(pts, dtype=float).reshape(-1, 2) if pts is not None and len(pts) else np.zeros((0, 2))
        if not np.isfinite(pts).all():
            raise EngineError("Spotiflow returned non-finite coordinates")
        if prob is None or len(prob) != len(pts):
            raise EngineError(f"Spotiflow returned {len(pts)} points but "
                              f"{'no' if prob is None else len(prob)} per-point scores; refusing to guess")
        prob = np.asarray(prob, dtype=float).reshape(-1)
        if d > 1:
            # scipy.ndimage.zoom (grid_mode=False) aligns the first/last pixel CENTRES: downscaled index i sits at
            # i*(n-1)/(m-1) in the cropped full-resolution frame (n = crop side, m = n/d). NOT i*d+(d-1)/2.
            sh, sw = sub.shape
            pts = np.stack([pts[:, 0] * ((sh - 1) / (sh // d - 1)), pts[:, 1] * ((sw - 1) / (sw // d - 1))], 1)
        xs, ys = pts[:, 1] + x0, pts[:, 0] + y0          # crop -> full-resolution frame (exact integer offset)
        validity = point_validity or raster_point_validity(valid_mask, image.shape)
        dets = build_detections(EngineName.spotiflow, run_id, analysis_id, page, channel_role, xs, ys, prob,
                                SCORE_KIND, validity)
        thr_used = sp.prob_thresh
        if thr_used is None:
            pt = getattr(model, "_prob_thresh", None)
            try:
                thr_used = float(pt[0] if isinstance(pt, (list, tuple, np.ndarray)) else pt)
            except (TypeError, ValueError, IndexError):
                thr_used = None
        rp = sp.model_dump(mode="json")
        rp.update(extras)
        rp["prob_thresh_used"] = thr_used
        rp["prob_thresh_source"] = "user" if sp.prob_thresh is not None else "model optimised threshold"
        rp["runtime"] = {"spotiflow": self.version(), "torch": _pkg_version("torch")}
        rp["model_path"] = str(mi.path)
        pre = {"input_dtype": str(image.dtype), "input_shape_hw": [h, w], "crop_to_valid": extras["crop_to_valid"],
               "crop_box_xyxy": [x0, y0, x1, y1],
               "crop_note": "half-open box [x0,x1) x [y0,y1) in full-resolution pixels; full = crop + (x0, y0)",
               "normalizer": ("spotiflow.utils.normalize: percentile 1 / 99.8 of the model input, computed on the crop"
                              if normalizer == "auto" else "none (raw values to the model)"),
               "cast": "float32 inside Spotiflow (lossless for 8/16-bit integers)", "eight_bit_conversion": False,
               "downscale": d, "trimmed_px": trimmed_px,
               "downscaled_shape_hw": list(model_in.shape),
               "downscale_note": ("none (native scale)" if d == 1 else
                                  f"image divided by {d} in the adapter (float32, scipy zoom order 1, align-corners; no pixel invented, bottom/right remainder trimmed); points mapped back with "
                                  f"x_full = x_s*(n-1)/(m-1) + crop offset; localisation accuracy ~ +/-{d / 2:g} px"),
               "page_to_channel": ("confirmed mapping" if channel_role else "unconfirmed (no colour assigned)")}
        run = EngineRunInfo(
            run_id=run_id, analysis_id=analysis_id, image_sha256=dict(image_sha256 or {}),
            engine=EngineName.spotiflow, engine_version=self.version(), model_name=mi.name,
            model_version=mi.version, model_checksum=mi.checksum_sha256, page=int(page),
            channel_role=channel_role, params=rp, preprocessing=pre, started_at=started, finished_at=utcnow_iso(),
            device=used_device, duration_s=time.perf_counter() - t0, status="ok", n_raw=len(dets),
            n_valid=sum(1 for d in dets if d.valid), notes=notes)
        progress(1.0, f"Spotiflow: {len(dets)} raw detection(s)")
        return EngineResult(run=run, detections=dets)
