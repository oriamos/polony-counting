"""Automatic chamber: synthetic stacks, API (preview only) and the real research TIFFs when present."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from legacy_fixtures import REPO, write_stack
from polony_app.api.main import create_app
from polony_app.imaging.chamber import ChamberError, auto_chamber, polygon_area
from polony_app.detection.masks import points_in_polygon

REAL = sorted(p for p in (REPO / "resources" / "research_tiffs").glob("*.tif") if "(" not in p.name)


def synth(h=300, w=400, bubble=True):
    """Clipped bright background outside an ellipse (page 1), dark interior ~500 with a dark crescent."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2) / (w * 0.42)) ** 2 + ((yy - h / 2) / (h * 0.42)) ** 2 < 1
    rng = np.random.default_rng(0)
    p0 = np.where(inside, rng.normal(600, 20, (h, w)), 25000).astype(np.uint16)
    p1 = np.where(inside, rng.normal(600, 20, (h, w)), 65535).astype(np.uint16)
    if bubble:
        dark = inside & (((xx - w / 2) / 60.0) ** 2 + ((yy - h * 0.72) / 25.0) ** 2 < 1)
        p0[dark] = 200
        p1[dark] = 200
    return p0, p1, inside


def test_synthetic_chamber_polygon_matches_eroded_ellipse_and_is_deterministic():
    p0, p1, inside = synth()
    a = auto_chamber([p0, p1], 15)
    b = auto_chamber([p0, p1], 15)
    assert a == b
    assert a["mask_channel_index"] == 1
    poly = np.array(a["polygon"])
    assert polygon_area(a["polygon"]) == pytest.approx(a["area_px2"], rel=0.03)
    # polygon lies inside the ellipse by about the rim
    yy, xx = np.mgrid[0:300, 0:400]
    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(inside)
    dv = d[np.clip(poly[:, 1].round().astype(int), 0, 299), np.clip(poly[:, 0].round().astype(int), 0, 399)]
    assert dv.min() > 12 and dv.max() < 19
    assert a["suggested_exclusions"], "dark crescent should be suggested"
    bx = a["suggested_exclusions"][0]
    cx, cy = np.mean(bx["polygon"], axis=0)
    assert abs(cx - 200) < 25 and abs(cy - 216) < 25


def test_no_bubble_no_suggestion_and_larger_rim_is_smaller():
    p0, p1, _ = synth(bubble=False)
    a, b = auto_chamber([p0, p1], 10), auto_chamber([p0, p1], 40)
    assert a["suggested_exclusions"] == [] and b["area_px2"] < a["area_px2"]


def test_no_clipped_background_is_an_error():
    z = np.full((200, 200), 500, np.uint16)
    with pytest.raises(ChamberError):
        auto_chamber([z, z], 10)


def test_api_is_preview_only(tmp_path):
    p0, p1, _ = synth()
    f = write_stack(tmp_path / "c.tif", [p0, p1])
    with TestClient(create_app(tmp_path / "data")) as c:
        with open(f, "rb") as fh:
            a = c.post("/api/analyses", files=[("files", ("c.tif", fh, "image/tiff"))]).json()
        r = c.post(f"/api/analyses/{a['id']}/regions/auto-chamber", json={"rim_px": 20})
        assert r.status_code == 200, r.text
        j = r.json()
        assert set(j) >= {"polygon", "area_px2", "rim_px", "method", "warnings", "suggested_exclusions"} and j["rim_px"] == 20
        assert c.get(f"/api/analyses/{a['id']}").json()["regions"] == []
        assert c.post(f"/api/analyses/{a['id']}/regions/auto-chamber", json={"rim_px": 9999}).status_code == 422
        flat = write_stack(tmp_path / "f.tif", [np.full((100, 100), 500, np.uint16)] * 2)
        with open(flat, "rb") as fh:
            b = c.post("/api/analyses", files=[("files", ("f.tif", fh, "image/tiff"))]).json()
        assert c.post(f"/api/analyses/{b['id']}/regions/auto-chamber", json={}).status_code == 422


@pytest.mark.skipif(not REAL, reason="resources/research_tiffs not present")
def test_real_slides_chamber_contains_the_marks():
    from polony_app.legacy_import.roi import read_embedded

    for f in REAL:
        a = tifffile.imread(f)
        r = auto_chamber([a[0], a[1]], 40)
        frac = r["chamber_area_px2"] / a[0].size
        assert 0.78 < frac < 0.85, (f.name, frac)
        pts = read_embedded(f).points
        xy = np.array([(p.x - 0.5, p.y - 0.5) for p in pts])
        inside = points_in_polygon(xy[:, 0], xy[:, 1], np.array(r["polygon"])).mean()
        assert inside > 0.95, (f.name, inside)         # 40 px rim drops only the marks nearest the edge
