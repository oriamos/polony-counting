"""Vercel feasibility probe for the polony review app (NOT the app itself).

Endpoints
  GET /api/health   -> trivial liveness, no heavy imports
  GET /api/probe    -> imports the scientific stack + Big-FISH, runs detection on a
                       synthetic 300x300 image (and optionally a full-size 2438x1460
                       one with ?full=1), reports timings / memory / /tmp / bundle size
  GET /api/bg       -> schedules a Starlette BackgroundTask that logs after ?s seconds,
                       to check whether work continues after the response is sent

The optional ``polony_app`` snapshot (detection + imaging + schemas) is copied in by
``deploy/vercel/stage_probe.sh``; the probe works without it.
"""
from __future__ import annotations

import os
import platform
import resource
import shutil
import sys
import time

_T_MODULE = time.time()
_T_PERF = time.perf_counter()

from fastapi import BackgroundTasks, FastAPI  # noqa: E402

app = FastAPI(title="polony-review feasibility probe")
_STATE = {"requests": 0}


def _rss_mb() -> float:
    # ru_maxrss is KiB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _du_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return round(total / 1e6, 1)


def _synthetic(h: int, w: int, n: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    img = rng.normal(300, 20, (h, w))
    ys = rng.uniform(7, h - 8, n)
    xs = rng.uniform(7, w - 8, n)
    yy, xx = np.mgrid[-6:7, -6:7]
    blob = 900 * np.exp(-(yy**2 + xx**2) / (2 * 1.8**2))
    for y, x in zip(ys.astype(int), xs.astype(int)):
        img[y - 6 : y + 7, x - 6 : x + 7] += blob
    return np.clip(img, 0, 65535).astype(np.uint16), n


@app.get("/api/health")
def health():
    _STATE["requests"] += 1
    return {"ok": True, "python": sys.version.split()[0], "requests_this_instance": _STATE["requests"],
            "seconds_since_module_load": round(time.time() - _T_MODULE, 2)}


@app.get("/api/bg")
def bg(background: BackgroundTasks, s: float = 20.0):
    t0 = time.time()

    def later():
        time.sleep(s)
        print(f"POLONY_BG_DONE after {time.time() - t0:.1f}s (requested {s}s)", flush=True)

    background.add_task(later)
    print(f"POLONY_BG_SCHEDULED s={s}", flush=True)
    return {"scheduled": True, "seconds": s}


@app.get("/api/probe")
def probe(full: int = 0):
    _STATE["requests"] += 1
    out: dict = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "requests_this_instance": _STATE["requests"],
        "seconds_since_module_load": round(time.time() - _T_MODULE, 2),
        "rss_peak_mb_before": _rss_mb(),
        "env": {k: os.environ.get(k) for k in ("VERCEL_REGION", "VERCEL_ENV", "AWS_REGION")},
    }
    try:
        with open("/proc/meminfo") as fh:
            out["meminfo"] = [ln.strip() for ln in fh.readlines()[:3]]
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            out["cgroup_memory_max"] = fh.read().strip()
    except OSError:
        pass
    du = shutil.disk_usage("/tmp")
    out["tmp_total_mb"] = round(du.total / 1e6)
    out["tmp_free_mb"] = round(du.free / 1e6)
    # /tmp write test (50 MB)
    t = time.perf_counter()
    try:
        with open("/tmp/probe.bin", "wb") as fh:
            fh.write(os.urandom(50 * 1024 * 1024))
        os.remove("/tmp/probe.bin")
        out["tmp_write_50mb_s"] = round(time.perf_counter() - t, 3)
    except OSError as e:
        out["tmp_write_error"] = repr(e)

    timings = {}
    for mod in ("numpy", "scipy.ndimage", "skimage", "pandas", "sklearn", "matplotlib", "tifffile", "PIL.Image",
                "openpyxl", "roifile", "bigfish.detection"):
        t = time.perf_counter()
        try:
            __import__(mod)
            timings[mod] = round(time.perf_counter() - t, 3)
        except Exception as e:  # report, don't fail
            timings[mod] = f"ERROR {e!r}"
    out["import_s"] = timings
    out["import_total_s"] = round(sum(v for v in timings.values() if isinstance(v, float)), 3)
    try:
        import psycopg  # noqa: F401

        out["psycopg"] = psycopg.__version__
    except Exception as e:
        out["psycopg"] = f"ERROR {e!r}"

    import bigfish
    import bigfish.detection as det
    import numpy as np

    out["bigfish_version"] = bigfish.__version__
    out["numpy_version"] = np.__version__

    img, n = _synthetic(300, 300, 40)
    t = time.perf_counter()
    spots, thr = det.detect_spots(img, threshold=None, return_threshold=True, voxel_size=(100, 100),
                                  spot_radius=(250, 250))
    out["detect_300x300"] = {"s": round(time.perf_counter() - t, 3), "found": int(len(spots)), "planted": n,
                             "threshold": float(thr)}

    if full:
        img, n = _synthetic(2438, 1460, 4000, seed=1)
        t = time.perf_counter()
        spots, thr = det.detect_spots(img, threshold=None, return_threshold=True, voxel_size=(100, 100),
                                      spot_radius=(250, 250))
        out["detect_full_bigfish_2438x1460"] = {"s": round(time.perf_counter() - t, 3), "found": int(len(spots)),
                                               "planted": n}
        try:
            from polony_app.detection.bigfish_engine import BigFishEngine
            from polony_app.schemas import DetectionParams

            eng = BigFishEngine()
            t = time.perf_counter()
            res = eng.detect(img.astype(np.float64), DetectionParams())
            out["detect_full_polony_engine"] = {"s": round(time.perf_counter() - t, 3), "found": len(res)}
        except Exception as e:
            out["detect_full_polony_engine"] = f"ERROR {e!r}"

    out["rss_peak_mb_after"] = _rss_mb()
    sp = [p for p in sys.path if p.endswith("site-packages") and os.path.isdir(p)]
    out["sys_path"] = sys.path
    out["site_packages_mb"] = {p: _du_mb(p) for p in sp}
    out["task_root_mb"] = _du_mb(os.getcwd())
    out["cwd"] = os.getcwd()
    out["total_request_s"] = round(time.perf_counter() - _T_PERF, 2) if _STATE["requests"] == 1 else None
    return out
