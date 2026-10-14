"""Open from disk, grid on/off, and the researcher's 'Save': labelled scan + per-slide CSV + experiment CSV."""
import csv
import shutil
import time

import pytest
import tifffile
from fastapi.testclient import TestClient

from fakes import detector, obj, write_two_channel_tiff  # noqa: F401
from labhelp import field_region
from polony_app.api.main import create_app
from polony_app.export import slide_save as ss
from polony_app.schemas import ObjectClass


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


@pytest.fixture
def scan_dir(tmp_path):
    d = tmp_path / "lab" / "slides"
    d.mkdir(parents=True)
    write_two_channel_tiff(d / "AI_A.tif")
    return d


def rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def open_and_prepare(client, scan, detector, fields):
    """fields: {idx: (n_red, n_yellow)}; opens by path, maps channels, puts field regions, fake-detects."""
    r = client.post("/api/analyses/open-path", json={"path": str(scan)})
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    assert r.json()["image"]["source_path"] == str(scan.resolve())
    assert client.put(f"/api/analyses/{aid}/channels",
                      json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]}).status_code == 200
    regs, objs = [], []
    for idx, (nr, ny) in fields.items():
        reg, x0, y0 = field_region(idx)
        regs.append(reg.model_dump(mode="json"))
        objs += [obj(f"r{idx}_{k}", x0 + 1 + k % 12, y0 + 2, ObjectClass.red) for k in range(nr)]
        objs += [obj(f"y{idx}_{k}", x0 + 1 + k % 12, y0 + 8, ObjectClass.yellow) for k in range(ny)]
    assert client.put(f"/api/analyses/{aid}/regions", json={"regions": regs}).status_code == 200
    detector.objects = objs
    j = client.post(f"/api/analyses/{aid}/detect", json={}).json()
    for _ in range(500):
        s = client.get(f"/api/jobs/{j['id']}").json()["status"]
        if s in ("done", "failed"):
            break
        time.sleep(0.02)
    assert s == "done"
    return aid


def test_open_path_errors_and_fs_list(client, scan_dir, tmp_path):
    assert client.post("/api/analyses/open-path", json={"path": str(tmp_path / "nope.tif")}).json()["code"] == "file_not_found"
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    assert client.post("/api/analyses/open-path", json={"path": str(txt)}).status_code == 415
    ls = client.get("/api/fs/list", params={"path": str(scan_dir)}).json()
    assert ls["files"] == ["AI_A.tif"] and ls["dirs"] == [] and ls["parent"] == str(scan_dir.parent.resolve())
    assert client.get("/api/fs/list", params={"path": str(txt)}).status_code == 422


def test_default_labels_and_parse():
    R, G, Y = ObjectClass.red, ObjectClass.green, ObjectClass.yellow
    assert ss.default_label([R]) == "red, not yellow"
    assert ss.default_label([Y]) == "yellow"
    assert ss.default_label([R, Y]) == "red and yellow"
    assert ss.default_label([G]) == "green, not yellow"
    assert ss.default_label([R, G]) == "red and green, not yellow"
    assert ss.default_label([R, G, Y]) == "red, green and yellow"
    assert ss.parse_classes(["yellow", "red", "red"]) == (R, Y)
    for bad in ([], ["uncertain"], ["blue"]):
        with pytest.raises(Exception):
            ss.parse_classes(bad)


def test_save_writes_labeled_scan_and_csvs_next_to_original(client, scan_dir, detector):
    original = (scan_dir / "AI_A.tif").read_bytes()
    aid = open_and_prepare(client, scan_dir / "AI_A.tif", detector, {1: (3, 2), 2: (0, 0), 4: (5, 1)})
    r = client.post(f"/api/analyses/{aid}/save-slide", json={"selections": [{"classes": ["red"]}]})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["directory"] == str(scan_dir.resolve()) and rep["slide_id"] == "AI_A"
    assert sorted(p.name for p in scan_dir.iterdir()) == ["AI_A.csv", "AI_A.tif", "AI_A_labeled.tif", "experiment_counts.csv"]
    assert (scan_dir / "AI_A.tif").read_bytes() == original            # original untouched
    img = tifffile.imread(scan_dir / "AI_A_labeled.tif")
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype.name == "uint8"
    slide = rows(scan_dir / "AI_A.csv")
    assert slide[0] == ["SlideID", "Total number of polonies"] + [f"Field{i}" for i in range(1, 25)]
    # field 1: 3 red-only, field 2: counted but zero, field 3: no region => blank, field 4: 5
    assert slide[1][:6] == ["AI_A", "8", "3", "0", "", "5"] and set(slide[1][6:]) == {""}
    exp = rows(scan_dir / "experiment_counts.csv")
    assert exp[0] == ["Slide name", "What counted", "NofPol", "NofGrids"] + [f"Field{i}" for i in range(1, 25)]
    assert exp[1][:8] == ["AI_A", "red, not yellow", "8", "3", "3", "0", "", "5"]
    assert rep["rows"][0]["nofgrids"] == 3


def test_two_selections_give_two_rows_and_resave_replaces(client, scan_dir, detector):
    aid = open_and_prepare(client, scan_dir / "AI_A.tif", detector, {1: (3, 2), 2: (1, 4)})
    body = {"selections": [{"classes": ["red"]}, {"classes": ["yellow"]}]}
    assert client.post(f"/api/analyses/{aid}/save-slide", json=body).status_code == 200
    exp = rows(scan_dir / "experiment_counts.csv")
    assert [(r[1], r[2]) for r in exp[1:]] == [("red, not yellow", "4"), ("yellow", "6")]
    assert (scan_dir / "AI_A__red-not-yellow.csv").exists() and (scan_dir / "AI_A__yellow.csv").exists()
    # second slide in the same folder appends; re-saving the first one replaces its rows, never duplicates
    shutil.copy(scan_dir / "AI_A.tif", scan_dir / "AI_B.tif")
    bid = open_and_prepare(client, scan_dir / "AI_B.tif", detector, {1: (2, 0)})
    assert client.post(f"/api/analyses/{bid}/save-slide", json={"selections": [{"classes": ["red", "yellow"]}]}).status_code == 200
    assert client.post(f"/api/analyses/{aid}/save-slide", json=body).status_code == 200
    exp = rows(scan_dir / "experiment_counts.csv")
    assert [(r[0], r[1]) for r in exp[1:]] == [("AI_A", "red, not yellow"), ("AI_A", "yellow"), ("AI_B", "red and yellow")]


def test_custom_label_duplicates_and_refusals(client, scan_dir, detector, tmp_path):
    aid = open_and_prepare(client, scan_dir / "AI_A.tif", detector, {1: (1, 1)})
    ok = client.post(f"/api/analyses/{aid}/save-slide", json={"selections": [{"classes": ["red"], "label": "clade B"}]})
    assert ok.status_code == 200 and rows(scan_dir / "experiment_counts.csv")[1][1] == "clade B"
    dup = client.post(f"/api/analyses/{aid}/save-slide",
                      json={"selections": [{"classes": ["red"], "label": "x"}, {"classes": ["yellow"], "label": "x"}]})
    assert dup.status_code == 422 and dup.json()["code"] == "invalid_selection"
    assert client.post(f"/api/analyses/{aid}/save-slide", json={"selections": []}).status_code == 422
    assert client.post(f"/api/analyses/{aid}/save-slide", json={"selections": [{"classes": []}]}).status_code == 422
    # a foreign experiment file is never overwritten
    (scan_dir / "experiment_counts.csv").write_text("a,b\n1,2\n")
    r = client.post(f"/api/analyses/{aid}/save-slide", json={"selections": [{"classes": ["red"]}]})
    assert r.status_code == 409 and r.json()["code"] == "experiment_csv_mismatch"
    assert (scan_dir / "experiment_counts.csv").read_text() == "a,b\n1,2\n"


def test_upload_needs_output_dir_then_saves_there(client, scan_dir, detector, tmp_path):
    with open(scan_dir / "AI_A.tif", "rb") as f:
        aid = client.post("/api/analyses", files=[("files", ("AI_A.tif", f, "image/tiff"))]).json()["id"]
    client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    reg, _, _ = field_region(1)
    client.put(f"/api/analyses/{aid}/regions", json={"regions": [reg.model_dump(mode="json")]})
    body = {"selections": [{"classes": ["red"]}]}
    r = client.post(f"/api/analyses/{aid}/save-slide", json=body)
    assert r.status_code == 422 and r.json()["code"] == "no_output_dir"
    out = tmp_path / "out"
    assert client.post(f"/api/analyses/{aid}/save-slide", json={**body, "output_dir": str(out)}).json()["code"] == "output_dir_missing"
    out.mkdir()
    ok = client.post(f"/api/analyses/{aid}/save-slide", json={**body, "output_dir": str(out)})
    assert ok.status_code == 200 and rows(out / "AI_A.csv")[1][:3] == ["AI_A", "0", "0"]


def test_no_fields_selected_is_refused(client, scan_dir, detector):
    aid = open_and_prepare(client, scan_dir / "AI_A.tif", detector, {1: (1, 0)})
    assert client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 1, "counted": False}).status_code == 200
    r = client.post(f"/api/analyses/{aid}/save-slide", json={"selections": [{"classes": ["red"]}]})
    assert r.status_code == 422 and r.json()["code"] == "no_field_regions"


def test_grid_selection_off_on_roundtrip(client, scan_dir, detector):
    aid = open_and_prepare(client, scan_dir / "AI_A.tif", detector, {1: (3, 0), 2: (2, 0)})
    off = client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 2, "counted": False}).json()
    assert [r["field_index"] for r in off["regions"]] == [1]
    assert [s["field_index"] for s in off["slide"]["skipped_fields"]] == [2]
    assert all(o["region_id"] is None for o in off["objects"] if o["id"].startswith("r2_"))
    body = {"selections": [{"classes": ["red"]}]}
    client.post(f"/api/analyses/{aid}/save-slide", json=body)
    assert rows(scan_dir / "AI_A.csv")[1][:5] == ["AI_A", "3", "3", "", ""]       # field 2 blank, not 0
    on = client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 2, "counted": True}).json()
    assert [r["field_index"] for r in on["regions"]] == [1, 2] and on["slide"]["skipped_fields"] == []
    assert sum(1 for o in on["objects"] if o["region_id"] and o["id"].startswith("r2_")) == 2   # objects re-assigned
    client.post(f"/api/analyses/{aid}/save-slide", json=body)
    assert rows(scan_dir / "AI_A.csv")[1][:5] == ["AI_A", "5", "3", "2", ""]
    # unknown field -> 404
    assert client.put(f"/api/analyses/{aid}/grid-selection", json={"field_index": 9, "counted": True}).status_code == 404
