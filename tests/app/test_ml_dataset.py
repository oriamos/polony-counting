"""Dataset snapshots: reproducibility, immutability, labels, crop rules, 16-bit and CSV order."""
import csv
import os
import stat

import numpy as np
import pytest
import tifffile

from polony_app.ml import dataset as ds
from polony_app.ml.common import MLError, sha256_file
from test_ml_support import Recorder, add_analysis, force_unresolved, new_store, tiff_pages

T0 = "2026-01-01T00:00:00Z"


def opts(**kw):
    kw.setdefault("created_at", T0)
    return ds.BuildOptions(**kw)


def build(store, data_dir, ids, **kw):
    say = Recorder()
    return ds.build_snapshot(store, data_dir, ids, opts(**kw), say=say), say


def read_points(root):
    with open(root / "points.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def one(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide1", slide_id="S1_A", seed=3)
    return store, aid, tmp_path


def test_snapshot_is_reproducible_and_content_addressed(one, tmp_path):
    store, aid, tp = one
    r1, _ = build(store, tp / "d1", [aid])
    r2, _ = build(store, tp / "d2", [aid])
    assert r1.version == r2.version and r1.version.startswith("ds-")
    f1 = sorted(p.relative_to(r1.path).as_posix() for p in r1.path.rglob("*") if p.is_file())
    f2 = sorted(p.relative_to(r2.path).as_posix() for p in r2.path.rglob("*") if p.is_file())
    assert f1 == f2 and "manifest.json" in f1 and "points.csv" in f1 and "regions.jsonl" in f1
    for rel in f1:   # identical files, byte for byte (created_at is injected)
        assert (r1.path / rel).read_bytes() == (r2.path / rel).read_bytes(), rel
    # the wall-clock time does not change the version, a different seed (different content) can
    r3, _ = build(store, tp / "d3", [aid], created_at="2030-05-05T00:00:00Z")
    assert r3.version == r1.version


def test_version_changes_with_content(one):
    store, aid, tp = one
    r1, _ = build(store, tp / "d1", [aid])
    with store.edit(aid) as live:
        from polony_app.review import engine
        engine.apply_action(live, "region_inspected", {"region_id": "f1", "inspected": False})   # un-approve f1
    r2, _ = build(store, tp / "d1", [aid])
    assert r2.version != r1.version
    assert len(r2.manifest.crops) < len(r1.manifest.crops)


def test_never_overwrites_and_detects_tampering(one):
    store, aid, tp = one
    r, _ = build(store, tp / "d", [aid])
    with pytest.raises(MLError, match="already exists"):
        build(store, tp / "d", [aid])
    assert ds.verify_snapshot(tp / "d", r.version) == []
    assert not os.access(r.path / "points.csv", os.W_OK) or os.geteuid() == 0   # read-only (root ignores the bit)
    os.chmod(r.path / "points.csv", stat.S_IWUSR | stat.S_IRUSR)
    with open(r.path / "points.csv", "a") as f:
        f.write("tampered\n")
    assert any("points.csv" in p for p in ds.verify_snapshot(tp / "d", r.version))


def test_positive_and_negative_classification(one):
    store, aid, tp = one
    r, say = build(store, tp / "d", [aid])
    rows = read_points(r.path)
    by = {}
    for x in rows:
        by.setdefault(x["status"], []).append(x)
    assert {x["object_id"] for x in by["rejected"]} == {"r1_1"}
    assert len(by["human_added"]) == 1 and by["human_added"][0]["basis"] == "manual_add"
    conf = {x["object_id"]: x["basis"] for x in by["confirmed"]}
    assert conf["r1_0"] == "object_confirmed" and conf["r1_2"] == "region_approved"
    # label CSVs contain exactly confirmed + human_added, never the rejected point
    n_label = 0
    for c in r.manifest.crops:
        lab = ds.read_labels_csv(r.path / c.path_csv)
        n_label += len(lab)
        assert c.n_positive == len(lab) and c.n_rejected == (1 if c.region_id == "f1" and c.channel_role == "red" else 0)
    assert n_label == sum(1 for x in rows if x["status"] in ("confirmed", "human_added") and x["crop_id"])
    f1red = next(c for c in r.manifest.crops if c.region_id == "f1" and c.channel_role == "red")
    lab = ds.read_labels_csv(r.path / f1red.path_csv)
    assert len(lab) == 3          # r1_0, r1_2, manual add; the deleted r1_1 is NOT a positive
    assert r.manifest.label_counts["points_by_status"]["rejected"] == 1


def test_strict_review_does_not_accept_untouched_proposals(one):
    store, aid, tp = one
    r, _ = build(store, tp / "d", [aid], strict_reviewed=True)
    rows = read_points(r.path)
    assert any(x["status"] == "unreviewed" for x in rows)
    # every crop that contains an unreviewed point is skipped
    assert all(x["crop_id"] == "" for x in rows if x["status"] == "unreviewed")


def test_16bit_preserved_and_csv_is_y_then_x(one):
    store, aid, tp = one
    r, _ = build(store, tp / "d", [aid])
    red, green = tiff_pages(3)
    c = next(c for c in r.manifest.crops if c.region_id == "f1" and c.channel_role == "red")
    img = tifffile.imread(r.path / c.path_tif)
    x0, y0, x1, y1 = c.box_xyxy
    assert (x0, y0, x1, y1) == (5, 5, 75, 60)
    assert img.dtype == np.uint16 and img.shape == (55, 70)
    assert np.array_equal(img, red[y0:y1, x0:x1])           # not one value changed: no 8-bit conversion, no rescale
    assert img.max() > 255
    with open(r.path / c.path_csv) as f:
        assert f.readline().strip() == "y,x"
    lab = ds.read_labels_csv(r.path / c.path_csv)
    # object r1_2 sits at x=60, y=50 -> crop coordinates (y=45, x=55)
    assert any(abs(y - 45.0) < 1e-9 and abs(x - 55.0) < 1e-9 for y, x in lab)
    g = next(c for c in r.manifest.crops if c.region_id == "f1" and c.channel_role == "green")
    assert np.array_equal(tifffile.imread(r.path / g.path_tif), green[5:60, 5:75])
    assert g.page == 1 and c.page == 0
    ys, xs = lab[:, 0], lab[:, 1]
    assert ys.max() < img.shape[0] and xs.max() < img.shape[1]


def test_green_alignment_maps_points_to_the_stored_page(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "al", slide_id="S9_A", seed=5, alignment=(2, 3), green_objects=((30.0, 25.0),))
    r, _ = build(store, tmp_path / "d", [aid])
    _, green = tiff_pages(5)
    g = next(c for c in r.manifest.crops if c.region_id == "f1" and c.channel_role == "green")
    assert g.box_reference_frame == [5, 5, 75, 60] and g.box_xyxy == [2, 3, 72, 58] and g.alignment_dy_dx == [2, 3]
    assert np.array_equal(tifffile.imread(r.path / g.path_tif), green[3:58, 2:72])
    lab = ds.read_labels_csv(r.path / g.path_csv)
    assert lab.tolist() == [[25.0 - 5.0, 30.0 - 5.0]]        # reference frame minus the box origin (shift cancels)
    red = next(c for c in r.manifest.crops if c.region_id == "f1" and c.channel_role == "red")
    assert red.box_xyxy == [5, 5, 75, 60] and red.alignment_dy_dx == [0, 0]


def test_unresolved_point_skips_the_crop(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "u", slide_id="S2_A")
    force_unresolved(store, aid, "r2_0")
    r, say = build(store, tmp_path / "d", [aid])
    ids = {(c.region_id, c.channel_role) for c in r.manifest.crops}
    assert ("f2", "red") not in ids and ("f1", "red") in ids
    assert ("f2", "green") in ids                          # the unresolved point is red-page only
    assert any("unresolved" in w and "skipped" in w for w in r.manifest.warnings)
    rows = read_points(r.path)
    assert [x for x in rows if x["status"] == "unresolved"][0]["crop_id"] == ""
    assert not any(x["status"] == "unresolved" for x in rows if x["status"] in ("confirmed", "human_added"))


def test_crop_overlapping_exclusion_is_skipped(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "x", slide_id="S3_A", exclusion_over_f2=True)
    r, _ = build(store, tmp_path / "d", [aid])
    assert {c.region_id for c in r.manifest.crops} == {"f1"}
    assert any(s["reason"] == "overlaps_exclusion" for s in r.manifest.skipped_crops)


def test_non_rectangular_regions_need_allow_bbox(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "p", slide_id="S4_A", approve=("f1",))
    with store.edit(aid) as live:
        from polony_app.review import engine
        reg = live.analysis.regions[1]
        reg.polygon = [(80, 5), (150, 5), (150, 60), (110, 70), (80, 60)]       # pentagon
        engine.apply_action(live, "region_inspected", {"region_id": "f2", "inspected": True})
        engine.apply_action(live, "approve_region", {"region_id": "f2"})
    r, _ = build(store, tmp_path / "d", [aid])
    assert {c.region_id for c in r.manifest.crops} == {"f1"}
    assert any("not a rectangle" in w for w in r.manifest.warnings)
    r2, _ = build(store, tmp_path / "d2", [aid], allow_bbox=True)
    bb = [c for c in r2.manifest.crops if c.region_id == "f2"]
    assert bb and all(c.bbox_of_polygon for c in bb)
    assert any("BOUNDING BOX" in w for w in r2.manifest.warnings)


def test_unapproved_region_and_unconfirmed_mapping_give_no_crops(tmp_path):
    store = new_store(tmp_path)
    a1 = add_analysis(store, "half", slide_id="S5_A", approve=("f1",))
    r, _ = build(store, tmp_path / "d", [a1])
    assert {c.region_id for c in r.manifest.crops} == {"f1"}
    a2 = add_analysis(store, "nomap", slide_id="S6_A", confirmed_mapping=False, approve=())
    with pytest.raises(MLError, match="none of the selected analyses"):
        build(store, tmp_path / "d9", [a2])


def test_reviewer_identity_from_action_log_or_unknown(tmp_path):
    store = new_store(tmp_path)
    a = add_analysis(store, "r", slide_id="S7_A", reviewer="alice")
    b = add_analysis(store, "u", slide_id="S8_A")
    r, _ = build(store, tmp_path / "d", [a, b])
    src = {s.analysis_id: s for s in r.manifest.sources}
    assert src[a].reviewers == ["alice"] and src[a].reviewer_identity_source == "action_log"
    assert src[b].reviewers == ["unknown"]
    assert any("no reviewer identity" in w for w in r.manifest.warnings)


def test_manifest_contents(one):
    store, aid, tp = one
    r, _ = build(store, tp / "d", [aid])
    m = r.manifest
    s = m.sources[0]
    assert s.tiff_sha256 == {"a.tif": sha256_file(store.originals_dir(aid) / "a.tif")}
    assert s.channel_mapping == {"red": {"file": "a.tif", "page": 0}, "green": {"file": "a.tif", "page": 1}}
    assert "x = column" in m.coordinate_convention and "centre" in m.coordinate_convention.replace("centres", "centre")
    assert s.review_status == "region_approved" and s.n_regions_fully_reviewed == 2
    assert m.files["points.csv"] == sha256_file(r.path / "points.csv")
    assert m.unresolved_assumptions and m.created_at == T0
    # the store has no engine-run API in this fixture only if missing; otherwise runs are listed (possibly empty)
    assert isinstance(s.engine_runs, list)


def test_store_without_engine_api_degrades_with_a_message(one):
    store, aid, tp = one

    class Thin:
        def __init__(self, s):
            self._s = s

        def view(self, aid):
            return self._s.view(aid)

        def exists(self, aid):
            return self._s.exists(aid)

        def originals_dir(self, aid):
            return self._s.originals_dir(aid)

    r, _ = build(Thin(store), tp / "d", [aid])
    assert any("no engine-run API" in w for w in r.manifest.warnings)
    assert r.manifest.crops


def test_dry_run_writes_nothing(one):
    store, aid, tp = one
    r = ds.build_snapshot(store, tp / "d", [aid], opts(), say=Recorder(), dry_run=True)
    assert r.dry_run and r.path is None
    assert ds.list_snapshots(tp / "d") == []
    assert [p for p in (tp / "d" / "datasets").iterdir()] == []        # no staging leftovers
