"""Persisted counting grid: PUT /analyses/{aid}/grid (spec round-trip, on/off kept by field index, approvals kept
only for fields that did not move, randomised shift with a recorded seed, clear, finalized analysis refused)."""
import time

import pytest
from fastapi.testclient import TestClient

from fakes import H, W, detector, obj, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.fields import grid as grid_mod
from polony_app.schemas import GridSpec, ObjectClass

# 2 x 3 cells of 40 px at (10, 10) on the 160 x 120 test image: fields 1 2 3 / 4 5 6 (row-major from top-left)
SPEC = {"rows": 2, "cols": 3, "cell_w": 40, "cell_h": 40, "origin_x": 10, "origin_y": 10,
        "order": "row-major", "corner": "tl", "first_index": 1}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


@pytest.fixture
def aid(client, tmp_path):
    p = tmp_path / "slide.tif"
    write_two_channel_tiff(p)
    r = client.post("/api/analyses/open-path", json={"path": str(p)})
    assert r.status_code == 200, r.text
    a = r.json()["id"]
    assert client.put(f"/api/analyses/{a}/channels",
                      json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]}).status_code == 200
    return a


def put_grid(client, aid, **body):
    return client.put(f"/api/analyses/{aid}/grid", json=body)


def fields(a):
    return {r["field_index"]: r for r in a["regions"] if r["kind"] == "counting" and r["field_index"] is not None}


def detect(client, aid, detector, objects):
    detector.objects = objects
    j = client.post(f"/api/analyses/{aid}/detect", json={}).json()
    for _ in range(500):
        s = client.get(f"/api/jobs/{j['id']}").json()["status"]
        if s in ("done", "failed"):
            break
        time.sleep(0.02)
    assert s == "done"


def approve(client, aid, rid):
    for t, p in (("region_inspected", {"region_id": rid, "inspected": True}), ("approve_region", {"region_id": rid})):
        r = client.post(f"/api/analyses/{aid}/actions", json={"type": t, "payload": p})
        assert r.status_code == 200, r.text
    return r.json()


def test_spec_round_trip_and_regions(client, aid):
    a = put_grid(client, aid, grid=SPEC)
    assert a.status_code == 200, a.text
    a = a.json()
    g = a["slide"]["grid"]
    assert {k: g[k] for k in SPEC} == SPEC and g["seed"] is None and (g["shift_dx"], g["shift_dy"]) == (0, 0)
    f = fields(a)
    assert sorted(f) == [1, 2, 3, 4, 5, 6] and f[1]["name"] == "Field-1"
    assert f[5]["polygon"] == [[50, 50], [90, 50], [90, 90], [50, 90]]          # row 1, col 1
    # survives reload, and PUT /settings cannot overwrite it (the grid owns the field regions)
    got = client.get(f"/api/analyses/{aid}").json()
    assert got["slide"]["grid"] == g
    slide = {**got["slide"], "grid": None, "sample_volume_ul": 2.5}
    s = client.put(f"/api/analyses/{aid}/settings", json={"slide": slide}).json()
    assert s["slide"]["grid"] == g and s["slide"]["sample_volume_ul"] == 2.5


def test_numbering_matches_frontend_rule():
    spec = GridSpec(**{**SPEC, "order": "col-major", "corner": "br", "first_index": 3})
    cells = grid_mod.spec_cells(spec)
    assert [(fi, r, c) for fi, r, c, _ in cells] == [(3, 1, 2), (4, 0, 2), (5, 1, 1), (6, 0, 1), (7, 1, 0), (8, 0, 0)]


def test_invalid_specs_refused(client, aid):
    assert put_grid(client, aid, grid={**SPEC, "first_index": 20}).json()["code"] == "invalid_grid"   # 20..25 > 24
    assert put_grid(client, aid, grid={**SPEC, "cell_w": 0}).json()["code"] == "invalid_grid"
    assert put_grid(client, aid, grid={**SPEC, "rows": 0}).json()["code"] == "invalid_grid"
    assert put_grid(client, aid, grid=SPEC, typo=1).json()["code"] == "unknown_fields"
    assert put_grid(client, aid, randomise=True).json()["code"] == "no_grid"


def test_reapply_keeps_skipped_fields(client, aid):
    put_grid(client, aid, grid=SPEC)
    for fi in (2, 6):
        assert client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": fi, "counted": False}).status_code == 200
    a = put_grid(client, aid, grid={**SPEC, "origin_x": 12}).json()
    assert sorted(fields(a)) == [1, 3, 4, 5]
    sk = {s["field_index"]: s for s in a["slide"]["skipped_fields"]}
    assert sorted(sk) == [2, 6]
    assert sk[2]["polygon"][0] == [52, 10]                     # the remembered geometry follows the grid
    # switching it back on recreates it at the new position
    on = client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 2, "counted": True}).json()
    assert fields(on)[2]["polygon"][0] == [52, 10] and [s["field_index"] for s in on["slide"]["skipped_fields"]] == [6]
    # a smaller grid in replace mode drops the fields (and skipped fields) that are no longer part of it
    a = put_grid(client, aid, grid={**SPEC, "rows": 1}).json()
    assert sorted(fields(a)) == [1, 2, 3] and a["slide"]["skipped_fields"] == []


def test_add_mode_keeps_other_lab_fields(client, aid):
    put_grid(client, aid, grid={**SPEC, "rows": 1, "cols": 1, "first_index": 10})
    a = put_grid(client, aid, grid=SPEC, replace=False).json()
    assert sorted(fields(a)) == [1, 2, 3, 4, 5, 6, 10]


def test_moved_field_loses_approval(client, aid, detector):
    a = put_grid(client, aid, grid=SPEC).json()
    f = fields(a)
    detect(client, aid, detector, [obj("o1", 30, 30, ObjectClass.red), obj("o3", 110, 30, ObjectClass.red)])
    approve(client, aid, f[1]["id"])
    a = approve(client, aid, f[3]["id"])
    assert fields(a)[1]["approved"] and fields(a)[3]["approved"]
    # identical re-apply: nothing moved, approvals stay
    a = put_grid(client, aid, grid=SPEC).json()
    assert fields(a)[1]["approved"] and fields(a)[3]["approved"]
    assert fields(a)[1]["id"] == f[1]["id"]
    # 2 columns: fields 1 and 2 stay where they were, field 3 moves to row 1 -> only its approval is lost
    a = put_grid(client, aid, grid={**SPEC, "cols": 2}).json()
    g = fields(a)
    assert sorted(g) == [1, 2, 3, 4]
    assert g[1]["approved"] and g[1]["id"] == f[1]["id"]
    assert not g[3]["approved"] and g[3]["id"] == f[3]["id"] and g[3]["polygon"][0] == [10, 50]
    # clearing refuses while a grid field is approved, unless forced
    r = put_grid(client, aid, grid=None)
    assert r.status_code == 409 and r.json()["code"] == "grid_fields_approved" and r.json()["detail"]["fields"] == [1]
    a = put_grid(client, aid, grid=None, force=True).json()
    assert fields(a) == {} and a["slide"]["grid"] is None


def test_clear_removes_grid_fields_only(client, aid):
    reg = {"id": "chamber", "kind": "counting", "name": "chamber", "polygon": [[0, 0], [150, 0], [150, 110], [0, 110]]}
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [reg]})
    put_grid(client, aid, grid=SPEC)
    client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 4, "counted": False})
    a = put_grid(client, aid, grid=None).json()
    assert [r["id"] for r in a["regions"]] == ["chamber"]
    assert a["slide"]["grid"] is None and a["slide"]["skipped_fields"] == []


def test_randomise_keeps_size_stays_inside_and_is_reproducible(client, aid):
    put_grid(client, aid, grid=SPEC)
    client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 5, "counted": False})
    a = put_grid(client, aid, grid=SPEC, randomise=True, seed=1234).json()
    g = a["slide"]["grid"]
    assert g["seed"] == 1234 and (g["rows"], g["cols"], g["cell_w"], g["cell_h"]) == (2, 3, 40, 40)
    assert (g["origin_x"], g["origin_y"]) == (10, 10)                 # base origin kept, the shift is separate
    assert (g["shift_dx"], g["shift_dy"]) == grid_mod.random_shift(GridSpec(**SPEC), W, H, 1234)
    assert [s["field_index"] for s in a["slide"]["skipped_fields"]] == [5]   # on/off kept by field index
    polys = [r["polygon"] for r in fields(a).values()] + [s["polygon"] for s in a["slide"]["skipped_fields"]]
    assert len(polys) == 6
    for p in polys:
        xs, ys = [q[0] for q in p], [q[1] for q in p]
        assert max(xs) - min(xs) == 40 and max(ys) - min(ys) == 40
        assert min(xs) >= -0.5 and max(xs) <= W - 0.5 and min(ys) >= -0.5 and max(ys) <= H - 0.5
    # same seed -> same shift; the server picks (and records) a seed when none is given
    again = put_grid(client, aid, randomise=True, seed=1234).json()["slide"]["grid"]   # uses the stored spec
    assert (again["shift_dx"], again["shift_dy"]) == (g["shift_dx"], g["shift_dy"])
    auto = put_grid(client, aid, grid=SPEC, randomise=True).json()["slide"]["grid"]
    assert isinstance(auto["seed"], int)
    assert (auto["shift_dx"], auto["shift_dy"]) == grid_mod.random_shift(GridSpec(**SPEC), W, H, auto["seed"])
    # undo restores the previous shift, regions and skipped fields together
    u = client.post(f"/api/analyses/{aid}/undo").json()
    assert u["slide"]["grid"] == again and [s["field_index"] for s in u["slide"]["skipped_fields"]] == [5]


def test_random_shift_bounds_and_seeds():
    spec = GridSpec(**SPEC)
    shifts = {grid_mod.random_shift(spec, W, H, s) for s in range(300)}
    assert len(shifts) > 50
    for dx, dy in shifts:
        assert dx == int(dx) and -10 <= dx <= 29 and -10 <= dy <= 29      # x0 in [0, 39], y0 in [0, 39]
    # grid larger than the image on one axis: no slack there
    wide = GridSpec(**{**SPEC, "cols": 5})                                  # 200 px > 160 px
    assert all(grid_mod.random_shift(wide, W, H, s)[0] == 0 for s in range(20))
    # bit-identical to the TypeScript mulberry32 (frontend/src/lib/labgrid.test.ts uses the same numbers)
    rng = grid_mod.mulberry32(42)
    assert [round(rng(), 10) for _ in range(3)] == [0.6011037519, 0.4482905590, 0.8524657935]
    assert grid_mod.random_shift(spec, W, H, 1234) == (-8, 18)         # = randomShift in labgrid.test.ts


def test_finalized_analysis_refused(client, aid, detector):
    a = put_grid(client, aid, grid={**SPEC, "rows": 1, "cols": 1}).json()
    detect(client, aid, detector, [obj("o1", 30, 30, ObjectClass.red)])
    approve(client, aid, fields(a)[1]["id"])
    assert client.post(f"/api/analyses/{aid}/finalize").status_code == 200
    r = put_grid(client, aid, grid=SPEC, randomise=True)
    assert r.status_code == 409 and r.json()["code"] == "analysis_finalized"
    assert put_grid(client, aid, grid=None, force=True).json()["code"] == "analysis_finalized"
