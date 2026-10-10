"""Integration with CORE's real pipeline (real Big-FISH on a tiny synthetic image)."""
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("bigfish")
pytest.importorskip("polony_app.detection.pipeline")

from fakes import counting, rect, wait_job, write_two_channel_tiff  # noqa: E402
from polony_app.api.main import create_app  # noqa: E402


def test_real_detection_round_trip(tmp_path):
    rpts, gpts = write_two_channel_tiff(tmp_path / "s.tif")
    with TestClient(create_app(tmp_path / "d")) as c:
        with open(tmp_path / "s.tif", "rb") as f:
            a = c.post("/api/analyses", files=[("files", ("s.tif", f, "image/tiff"))]).json()
        aid = a["id"]
        c.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
        c.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1", rect(2, 2, 157, 117)).model_dump(mode="json")]})
        req = {"red": {"spot_radius_px": 2.5, "minimum_distance_px": 5, "threshold": 20, "background": {"method": "gaussian_subtract", "sigma_px": 15}},
               "green": {"spot_radius_px": 2.5, "minimum_distance_px": 5, "threshold": 20, "background": {"method": "gaussian_subtract", "sigma_px": 15}}}
        j = c.post(f"/api/analyses/{aid}/detect", json=req).json()
        j = wait_job(c, j["id"], 60)
        assert j["status"] == "done", j
        a = c.get(f"/api/analyses/{aid}").json()
        assert a["status"] == "in_review" and a["provenance"]["package_versions"]
        counts = c.get(f"/api/analyses/{aid}/counts").json()["slide"]
        # 3 red + 2 green blobs; one red/green pair (80,60)/(80,61) is one yellow object
        assert counts["yellow"] == 1
        assert (counts["red_only"], counts["green_only"], counts["uncertain"]) == (2, 1, 0)
        # processed views use the analysis' background params
        r = c.get(f"/api/analyses/{aid}/image/processed_red", params={"max_dim": 80})
        assert r.status_code == 200
        r = c.post(f"/api/analyses/{aid}/detect/preview", json={**req, "preview_window": [0, 0, 100, 70]})
        assert r.status_code == 200 and len(r.json()["objects"]) >= 2
        z = c.get(f"/api/analyses/{aid}/export", params={"kind": "bundle"})
        assert z.status_code == 200 and z.content[:2] == b"PK"
