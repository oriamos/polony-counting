"""Test doubles + helpers for the APP agent's tests (CORE owns synthetic.py)."""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from polony_app.review.state import AnalysisState
from polony_app.schemas import (
    Analysis,
    AnalysisStatus,
    ChannelInfo,
    ChannelRole,
    ChannelEvidence,
    DetectedObject,
    ImageKind,
    ImageMeta,
    ObjectClass,
    Region,
    RegionKind,
    ReviewReason,
)

H, W = 120, 160


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def counting(rid="c1", poly=None, name=None):
    return Region(id=rid, kind=RegionKind.counting, name=name or rid, polygon=poly or rect(5, 5, 155, 115))


def exclusion(rid="x1", poly=None):
    return Region(id=rid, kind=RegionKind.exclusion, name=rid, polygon=poly or rect(100, 5, 155, 50))


def obj(oid, x, y, cls=ObjectClass.red, **kw) -> DetectedObject:
    dr = cls in (ObjectClass.red, ObjectClass.yellow)
    dg = cls in (ObjectClass.green, ObjectClass.yellow)
    ev_r = ChannelEvidence(detected=dr, x=x, y=y, peak_intensity=100.0, snr=5.0) if dr else ChannelEvidence(detected=False)
    ev_g = ChannelEvidence(detected=dg, x=x, y=y, peak_intensity=90.0, snr=4.0) if dg else ChannelEvidence(detected=False)
    d = dict(id=oid, x=x, y=y, cls=cls, red=ev_r, green=ev_g)
    d.update(kw)
    return DetectedObject(**d)


def make_state(objects=(), regions=(), mapped=True, detected=True) -> AnalysisState:
    ch = [
        ChannelInfo(index=0, source_file="a.tif", page=0, role=ChannelRole.red if mapped else None, role_confirmed=mapped),
        ChannelInfo(index=1, source_file="a.tif", page=1, role=ChannelRole.green if mapped else None, role_confirmed=mapped),
    ]
    img = ImageMeta(kind=ImageKind.raw_two_channel, height=H, width=W, dtype="uint16", bit_depth=16,
                    sha256={"a.tif": "0" * 64}, channels=ch)
    a = Analysis(id="a" * 12, name="t", created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:00Z",
                 status=AnalysisStatus.in_review, image=img, regions=list(regions), objects=list(objects))
    if detected:
        a.provenance.detection_ran_at = "2026-01-01T00:00:00Z"
    st = AnalysisState(analysis=a)
    from polony_app.review.engine import assign_region_ids

    assign_region_ids(a)
    return st


# ----------------------------------------------------------- images
def write_two_channel_tiff(path: Path, seed: int = 0) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """2-page uint16 TIFF with Gaussian blobs; returns (red_xy, green_xy)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    red = rng.normal(200, 10, (H, W))
    green = rng.normal(200, 10, (H, W))
    rpts = [(30.0, 30.0), (80.0, 60.0), (120.0, 90.0)]
    gpts = [(80.0, 61.0), (40.0, 95.0)]
    for (x, y) in rpts:
        red += 3000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))
    for (x, y) in gpts:
        green += 3000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))
    arr = np.stack([red, green]).clip(0, 65535).astype(np.uint16)
    tifffile.imwrite(str(path), arr, photometric="minisblack")
    return rpts, gpts


# ----------------------------------------------------------- fake detection
class FakeDetectionCancelled(Exception):
    pass


FakeDetectionCancelled.__name__ = "DetectionCancelled"


class FakeDetector:
    """Callable replacing detection.pipeline.run_detection."""

    def __init__(self):
        self.objects: list[DetectedObject] = []
        self.slow = False
        self.calls = 0
        self.fail = False
        self.last_window = None

    def __call__(self, red, green, regions, request, progress, cancelled, window=None, alignment=(0, 0), **kw):
        self.calls += 1
        self.last_alignment = alignment
        self.last_window = window
        from polony_app.detection import pipeline

        if self.fail:
            raise RuntimeError("boom")
        n = 0
        while self.slow and n < 400:
            if cancelled():
                raise pipeline.DetectionCancelled()
            progress(min(0.9, n / 400), "working")
            time.sleep(0.01)
            n += 1
        progress(0.9, "done")
        return pipeline.DetectionResult(
            objects=[o.model_copy(deep=True) for o in self.objects],
            package_versions={"big-fish": "fake"},
            notes=["fake detector"],
        )


@pytest.fixture
def detector(monkeypatch):
    """Patch polony_app.detection.pipeline.run_detection with a FakeDetector
    (installing a stub pipeline module if CORE's does not exist yet)."""
    try:
        from polony_app.detection import pipeline  # noqa: F401
    except Exception:
        from dataclasses import dataclass, field

        stub = types.ModuleType("polony_app.detection.pipeline")

        @dataclass
        class DetectionResult:
            objects: list
            package_versions: dict
            notes: list = field(default_factory=list)

        class DetectionCancelled(Exception):
            pass

        stub.DetectionResult, stub.DetectionCancelled = DetectionResult, DetectionCancelled
        stub.run_detection = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "polony_app.detection.pipeline", stub)
        import polony_app.detection as pkg

        monkeypatch.setattr(pkg, "pipeline", stub, raising=False)
    from polony_app.detection import pipeline

    d = FakeDetector()
    monkeypatch.setattr(pipeline, "run_detection", d)
    return d


def wait_job(client, jid, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")
