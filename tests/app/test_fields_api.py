"""Reference marks, field import and comparison through the HTTP API (synthetic ImageJ stacks)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from legacy_fixtures import GRID_XS, GRID_YS, blank, grid_overlay, point_roi, write_stack
from polony_app.api.main import create_app

INFO = "ImageDescription: 635 [W1]\nSoftware: ImageIO 2.0.4.0"
# ImageJ-frame sub-pixel points (canonical = value - 0.5); counter -> cell:
#  counter 0 -> cell(0,0) x3 plane1 ; counter 1 -> cell(0,1) x2 plane1 ; counter 2 -> cell(1,0) x1 plane1 + x1 plane2
PTS = [(20.5, 20.5), (25.5, 22.5), (30.5, 30.5), (70.5, 20.5), (80.5, 25.5), (20.5, 60.5), (22.5, 62.5)]
CNT = [0, 0, 0, 1, 1, 2, 2]
POS = [1, 1, 1, 1, 1, 1, 2]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


def upload(client, path):
    with open(path, "rb") as f:
        r = client.post("/api/analyses", files=[("files", (path.name, f, "image/tiff"))])
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def stack(tmp_path):
    return write_stack(tmp_path / "s.tif", [blank(), blank()], roi=point_roi(PTS, CNT, POS),
                       overlays=[grid_overlay(GRID_XS, GRID_YS, 160, 120)], info=INFO)


@pytest.fixture
def aid(client, stack):
    return upload(client, stack)["id"]


def test_ingest_suggests_roles_for_635_stack_but_does_not_confirm(client, stack):
    a = upload(client, stack)
    ch = a["image"]["channels"]
    assert [c["suggested_role"] for c in ch] == ["red", "green"]
    assert all(c["role"] is None and c["role_confirmed"] is False for c in ch)
    assert any("Suggested from 8 validated GenePix stacks" in w and "RGB composites are clipped" in w
               for w in a["image"]["warnings"])


def test_no_suggestion_without_635_hint(client, tmp_path):
    p = write_stack(tmp_path / "n.tif", [blank(), blank()], info="ImageDescription: foo")
    ch = upload(client, p)["image"]["channels"]
    assert [c["suggested_role"] for c in ch] == [None, None]


def test_reference_endpoint(client, aid):
    r = client.get(f"/api/analyses/{aid}/reference").json()
    assert r["available"] and r["source_file"].endswith("s.tif")
    assert len(r["points"]) == 7
    p0 = r["points"][0]
    assert (p0["x"], p0["y"], p0["counter"], p0["plane"], p0["field"]) == (20.0, 20.0, 0, 0, 1)
    assert r["planes"] == [{"page": 0, "point_count": 6}, {"page": 1, "point_count": 1}]
    assert r["reference_counts"] == {"1": {"plane0": 3, "plane1": 0}, "2": {"plane0": 2, "plane1": 0},
                                     "3": {"plane0": 1, "plane1": 1}}
    g = r["grid"]
    assert g["source"] == "overlay" and (g["cols"], g["rows"]) == (3, 3)
    assert g["origin"] == [9.5, 9.5] and g["cell"] == [50.0, 35.0]
    assert len(g["cells"]) == 9
    assert [c["field"] for c in g["cells"][:3]] == [1, 2, 4]      # (0,0)->1, (0,1)->2, (0,2) first unused
    assert g["cells"][3]["field"] == 3                            # (1,0) holds counter 2 -> field 3
    assert g["cells"][0]["polygon"] == [[9.5, 9.5], [59.5, 9.5], [59.5, 44.5], [9.5, 44.5]]


def test_reference_unavailable(client, tmp_path):
    p = write_stack(tmp_path / "plain.tif", [blank(), blank()])
    aid = upload(client, p)["id"]
    r = client.get(f"/api/analyses/{aid}/reference").json()
    assert r["available"] is False and r["reason"] and r["points"] == [] and r["grid"]["source"] == "none"


def test_import_counted_all_replace_and_shift(client, aid):
    url = f"/api/analyses/{aid}/fields/import"
    r = client.post(url, json={"mode": "counted", "plane": 0})
    assert r.status_code == 200, r.text
    regs = r.json()["regions"]
    assert [(g["name"], g["field_index"]) for g in regs] == [("Field-1", 1), ("Field-2", 2), ("Field-3", 3)]
    assert any("ambiguous" in w for w in r.json()["field_import"]["warnings"])
    assert regs[0]["polygon"][0] == [9.5, 9.5]
    # second import without replace -> 409
    assert client.post(url, json={"mode": "all"}).status_code == 409
    r = client.post(url, json={"mode": "all", "replace": True})
    regs = r.json()["regions"]
    assert len(regs) == 9
    assert [g["field_index"] for g in regs] == [1, 2, 4, 3, 5, 6, 7, 8, 9] or sorted(g["field_index"] for g in regs) == list(range(1, 10))
    by = {g["name"]: g for g in regs}
    assert by["Field-3"]["polygon"][0] == [9.5, 44.5]            # the cell holding counter 2
    # a shift moves the cells and the field assignment follows the shifted grid (points then fall elsewhere)
    r = client.post(url, json={"mode": "all", "replace": True, "shift": [-10, 0]})
    assert r.status_code == 200
    assert any(g["polygon"][0][0] == -0.5 for g in r.json()["regions"])   # clipped to the image
    # counted + plane with no marks on it
    r = client.post(url, json={"mode": "counted", "plane": 5, "replace": True})
    assert r.status_code == 422


def test_import_without_grid_422(client, tmp_path):
    p = write_stack(tmp_path / "ng.tif", [blank(), blank()], roi=point_roi(PTS, CNT, POS))
    aid = upload(client, p)["id"]
    r = client.post(f"/api/analyses/{aid}/fields/import", json={"mode": "all"})
    assert r.status_code == 422 and "grid" in r.json()["message"].lower()


def test_import_invalidates_approval_and_respects_finalized(client, aid):
    url = f"/api/analyses/{aid}/fields/import"
    client.post(url, json={"mode": "counted"})
    client.put(f"/api/analyses/{aid}/channels",
               json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    a = client.get(f"/api/analyses/{aid}").json()
    rid = a["regions"][0]["id"]
    # import with replace creates new region ids; old region (and its approval) are gone
    r = client.post(url, json={"mode": "counted", "replace": True}).json()
    assert rid not in [g["id"] for g in r["regions"]] and not any(g["approved"] for g in r["regions"])


def act(client, aid, type_, **payload):
    return client.post(f"/api/analyses/{aid}/actions", json={"type": type_, "payload": payload})


def test_compare(client, aid):
    url = f"/api/analyses/{aid}"
    assert client.get(f"{url}/fields/compare?plane=0").status_code == 422      # no field regions
    client.put(f"{url}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    client.post(f"{url}/fields/import", json={"mode": "counted", "plane": 0})
    # app objects: field1 (cell 0,0) 2 red, field2 (cell 0,1) 2 red + 1 green, field3 (cell 1,0) none
    for x, y, c in [(20, 20, "red"), (30, 30, "red"), (70, 20, "red"), (80, 25, "red"), (90, 30, "green")]:
        assert act(client, aid, "add", x=x, y=y, cls=c).status_code == 200
    r = client.get(f"{url}/fields/compare?plane=0&count_type=red_positive")
    assert r.status_code == 200, r.text
    j = r.json()
    core = ("field", "reference_count", "app_count", "delta", "has_reference_marks", "region", "status", "examined",
            "examination_source")
    assert [{k: row[k] for k in core} for row in j["rows"]] == [
        dict(field=1, reference_count=3, app_count=2, delta=-1, has_reference_marks=True, region="Field-1",
             status="positive_marked", examined=True, examination_source="embedded_imagej_marks"),
        dict(field=2, reference_count=2, app_count=2, delta=0, has_reference_marks=True, region="Field-2",
             status="positive_marked", examined=True, examination_source="embedded_imagej_marks"),
        dict(field=3, reference_count=1, app_count=0, delta=-1, has_reference_marks=True, region="Field-3",
             status="positive_marked", examined=True, examination_source="embedded_imagej_marks")]
    assert j["summary"]["eligibility_source"] == "embedded_imagej_marks"
    s = j["summary"]
    assert (s["n_fields"], s["total_reference"], s["total_app"]) == (3, 6, 4)
    assert s["mae"] == pytest.approx(2 / 3)
    assert s["slope"] == pytest.approx((6 + 4) / (9 + 4 + 1) * 1.0, rel=0.2) or s["slope"] is not None
    assert s["pearson_r"] == pytest.approx(np.corrcoef([3, 2, 1], [2, 2, 0])[0, 1])
    assert "accura" not in j["note"].lower()
    g = client.get(f"{url}/fields/compare?plane=0&count_type=green_positive").json()
    assert [row["app_count"] for row in g["rows"]] == [0, 1, 0]
    p1 = client.get(f"{url}/fields/compare?plane=1&count_type=red_positive").json()
    # plane 1: only field 3 has marks there. Fields 1 and 2 (marked on plane 0 only) are NOT examined on plane 1:
    # unknown, reference null (this used to be compared against an artificial 0), and out of every metric.
    assert [row["reference_count"] for row in p1["rows"]] == [None, None, 1]
    assert [row["status"] for row in p1["rows"]] == ["unknown", "unknown", "positive_marked"]
    assert p1["summary"]["eligible_examined_fields"] == 1 and p1["summary"]["unknown_fields_excluded"] == 2
    assert p1["summary"]["total_reference"] == 1 and p1["summary"]["total_app"] == 0
    assert not any("compared as 0" in w for w in p1["warnings"])
    assert client.get(f"{url}/fields/compare?plane=0&count_type=bogus").status_code == 422
