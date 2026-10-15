"""``ml import-legacy``: only researcher-examined (file, field, page) tuples enter the fine-tuning lifecycle."""
import csv
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import tifffile

from legacy_bridge_fixtures import (CH, CW, GRID_XS, GRID_YS, RED_GREEN, T0, H, W, cell_xy, make_tiff, marks_in, standard_corpus,
                                    write_channel_map, write_examined, write_identity_map)
from legacy_fixtures import grid_overlay, point_roi, write_stack
from polony_app.fields import examined as ex
from polony_app.ml import dataset as ds
from polony_app.ml import legacy_bridge as lb
from polony_app.ml.cli import main
from polony_app.ml.common import MLError
from polony_app.ml.estimates import estimate_training
from test_ml_support import Recorder

FR = {"train": 0.5, "val": 0.5, "test": 0.0}
FR3 = {"train": 0.34, "val": 0.33, "test": 0.33}


def opts(**kw):
    kw.setdefault("created_at", T0)
    kw.setdefault("fractions", FR)
    return lb.BridgeOptions(**kw)


def go(tmp_path, root, *, cm=None, im="auto", dry=False, data=None, examined=None, corpus=None, resume=True, **kw):
    """Run import_legacy; ``cm``: channel-map path (default: page 0 red, page 1 green), ``im``: 'auto' = confirmed identities."""
    tmp_path = Path(tmp_path)
    if cm is None:
        cm = write_channel_map(tmp_path / "channels.json", defaults=RED_GREEN)
    if im == "auto":
        im = write_identity_map(tmp_path / "id.json", corpus) if corpus else None
    data = data or (tmp_path / "data")
    return lb.import_legacy(data, root, opts(**kw), channel_map=cm, identity_map=im, examined_fields=examined, dry_run=dry,
                            resume=resume, say=Recorder())


def crops_of(out, rel=None, page=None, status=None):
    cs = out.result.manifest.crops
    return [c for c in cs if (rel is None or c.source_file == rel) and (page is None or c.page == page)
            and (status is None or c.provenance.get("examination_status") == status)]


def region_ids(crops):
    return sorted(c.region_id for c in crops)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    return root, standard_corpus(root)


def make_one(tmp_path, name="29-09-2026_T7_AB_AM-A.tif", **kw):
    root = tmp_path / "in"
    root.mkdir(parents=True, exist_ok=True)
    make_tiff(root / name, **kw)
    return root


# ================================================================== the scientific rule
def test_marked_only_default_uses_only_marked_field_pages_and_maps_position_to_page(tmp_path, corpus):
    root, files = corpus
    out = go(tmp_path, root, corpus=files)
    m = out.result.manifest
    rel = files["AM-A"]
    assert m.options["import_mode"] == "marked-only" and out.plan.opts.mode == "marked-only"
    # page 0 (ImageJ position 1): three marked cells; page 1 (position 2): ONLY the cell that has marks on page 1
    assert region_ids(crops_of(out, rel, 0)) == ["r0c0", "r0c1", "r1c0"]
    assert region_ids(crops_of(out, rel, 1)) == ["r0c0"]
    for c in m.crops:
        p = c.provenance
        assert p["researcher_examined"] is True and p["examination_status"] == "positive_marked"
        assert p["imagej_position"] == p["page"] + 1 and p["channel_role"] == {0: "red", 1: "green"}[p["page"]]
        assert p["examination_source"] == "embedded_imagej_marks" and p["channel_mapping_origin"] == "default"
        assert c.n_positive == p["n_researcher_marks"] > 0          # marked-only never exports an empty (negative) crop
        assert p["source_tiff"] == c.source_file and p["source_sha256"] == c.source_sha256
    assert lb.verify_examination_provenance(m) == []
    f = m.import_summary["fields"]
    assert f["positive_marked_fields"] == 16 and f["verified_zero_fields"] == 0
    assert f["unknown_fields_excluded"] == 4 * 9 * 2 - 16 and f["excluded_unproven"] == f["unknown_fields_excluded"]
    assert m.import_summary["kind"] == "legacy_imagej" and m.import_summary["annotations"]["labels_exported"] == 44


def test_other_page_marks_and_overlay_cells_never_make_a_field_examined(tmp_path):
    root = make_one(tmp_path, marks=marks_in(1, 1, 3, 0, 1))              # marks on page 0 of cell (1,1) only
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    cs = out.result.manifest.crops
    assert [(c.page, c.region_id) for c in cs] == [(0, "r1c1")]           # page 1 of the same field: unknown, no crop
    s = out.plan.summary["fields"]
    assert s["positive_marked_fields"] == 1 and s["unknown_fields_excluded"] == 17
    # every other (cell, page) tuple, incl. page 1 of the marked field, is unknown and has no labels / crop
    (plan,) = out.plan.plans
    assert {k: t.exam.status for k, t in plan.tuples.items() if t.exam.status != "unknown"} == {(0, 4): "positive_marked"}
    assert all(t.labels == [] for k, t in plan.tuples.items() if k != (0, 4))


def test_every_exported_crop_has_provenance_and_the_gate_refuses_otherwise(tmp_path, corpus):
    root, files = corpus
    out = go(tmp_path, root, corpus=files)
    m = out.result.manifest
    bad = m.model_copy(deep=True)
    bad.crops[0].provenance = {k: v for k, v in bad.crops[0].provenance.items() if k != "researcher_examined"}
    bad.crops[1].provenance["examination_status"] = "unknown"
    bad.crops[2].provenance = {}
    problems = lb.verify_examination_provenance(bad)
    assert len(problems) >= 3 and any("researcher_examined" in p for p in problems)
    cands = [c for p in out.plan.plans for c in p.ext.cands]
    lb.assert_examined(cands)
    cands[0].provenance = dict(cands[0].provenance, researcher_examined=False)
    with pytest.raises(MLError, match="not proven researcher-examined"):
        lb.assert_examined(cands)
    # a snapshot that is not a legacy import is out of scope
    assert lb.verify_examination_provenance({"crops": [{"crop_id": "x"}], "import_summary": {}}) == []


def test_verified_zero_only_from_an_explicit_manifest_and_empty_labels(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 4, 0, 1) + marks_in(0, 1, 2, 1, 1) + marks_in(2, 0, 3, 3, 1)
                    + marks_in(1, 1, 1, 4, 1))
    rel = "29-09-2026_T7_AB_AM-A.tif"
    man = write_examined(tmp_path / "m.json", {rel: {0: [
        {"counter": 0, "count": 4},                      # marks == stated -> positive
        {"row": 2, "col": 2, "count": 0},                # counted zero, no marks -> verified zero (empty labels)
        {"row": 2, "col": 1},                            # listed without a count: zero NOT proven
        {"counter": 1, "count": 5},                      # stated 5 but 2 marks -> conflict
        {"row": 1, "col": 1, "count": 0},                # stated zero but a mark exists -> conflict
    ]}})
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest")
    cs = {c.region_id: c for c in out.result.manifest.crops}
    assert sorted(cs) == ["r0c0", "r2c2"]                # the unlisted marked cell (2,0) is NOT used in manifest mode
    z = cs["r2c2"]
    assert z.n_positive == 0 and z.provenance["examination_status"] == "verified_zero"
    assert z.provenance["examination_source"] == "explicit_manifest" and z.provenance["n_researcher_marks"] == 0
    _img, lab = ds.load_crop(tmp_path / "data", out.result.version, z)
    assert lab.shape == (0, 2) and (tmp_path / "data/datasets" / out.result.version / z.path_csv).read_text() == "y,x\n"
    f = out.plan.summary["fields"]
    assert f["positive_marked_fields"] == 1 and f["verified_zero_fields"] == 1 and f["conflict_fields_excluded"] == 2
    assert f["unknown_fields_excluded"] == 18 - 2 - 2
    reasons = " ".join(f["reasons"])
    assert "no count is stated" in reasons and "not listed in the examined-fields manifest" in reasons
    assert lb.verify_examination_provenance(out.result.manifest) == []


def test_marked_only_ignores_manifest_fields_but_manifest_can_contradict(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1) + marks_in(0, 1, 2, 1, 1))
    rel = "29-09-2026_T7_AB_AM-A.tif"
    man = write_examined(tmp_path / "m.json", {rel: {0: [{"counter": 0, "count": 4},          # contradicts the 3 marks
                                                        {"row": 2, "col": 2, "count": 0}]}})  # zero: cannot ADD fields here
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man)
    assert region_ids(out.result.manifest.crops) == ["r0c1"]
    f = out.plan.summary["fields"]
    assert f["conflict_fields_excluded"] == 1 and f["verified_zero_fields"] == 0


def test_there_is_no_mode_that_reads_unmarked_cells_as_negative():
    from polony_app.ml.cli import build_parser
    with pytest.raises(MLError, match="--mode"):
        opts(mode="all-cells").check()
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["import-legacy", "--input", "x", "--mode", "all-cells"])
    sub = parser._subparsers._group_actions[0].choices["import-legacy"]
    flags = {o for a in sub._actions for o in a.option_strings}
    assert not any(w in f for f in flags for w in ("negative", "all-cells", "unmarked", "empty"))
    assert "--mode" in flags and "--acknowledge-no-counters" in flags


def test_manifest_mode_requires_a_manifest_and_unlisted_marked_fields_are_excluded(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1))
    with pytest.raises(MLError, match="examined-fields"):
        go(tmp_path, root, mode="examined-fields-manifest", dry=True)
    man = write_examined(tmp_path / "m.json", {"29-09-2026_T7_AB_AM-A.tif": {1: [{"counter": 0, "count": 3}]}})
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", dry=True)
    # the entry is for page 1 where cell (0,0) has no marks: stated 3 but no embedded marks -> conflict; page 0 not listed -> unknown
    f = out.plan.summary["fields"]
    assert f["eligible_examined_fields"] == 0 and f["conflict_fields_excluded"] == 1
    assert "no usable crop" in " ".join(out.blockers) or "no crop can be exported" in " ".join(out.blockers)


def test_field_entries_need_the_counter_plus_one_acknowledgement(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1) + marks_in(1, 1, 2, 4, 1))
    rel = "29-09-2026_T7_AB_AM-A.tif"
    man = write_examined(tmp_path / "m.json", {rel: {0: [{"field": 1, "count": 3}, {"field": 5, "count": 2}]}})
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", dry=True)
    assert out.plan.summary["fields"]["eligible_examined_fields"] == 0
    assert len(out.plan.summary["unresolved_manifest_entries"]) == 2
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", dry=True,
             acknowledge_field_equals_counter_plus_one=True)
    f = out.plan.summary["fields"]
    assert f["positive_marked_fields"] == 2 and not out.plan.summary["unresolved_manifest_entries"]
    assert any("counter N-1" in a for a in out.plan.summary["assumptions"])


def test_two_manifest_entries_for_one_cell_conflict(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1))
    rel = "29-09-2026_T7_AB_AM-A.tif"
    man = write_examined(tmp_path / "m.json", {rel: {0: [{"counter": 0, "count": 3}, {"row": 0, "col": 0, "count": 3}]}})
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", dry=True)
    assert out.plan.summary["fields"]["conflict_fields_excluded"] == 1
    assert "same cell" in " ".join(out.plan.summary["fields"]["reasons"])


def test_manifest_sha_mismatch_and_ambiguous_names_are_refused(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "a/x.tif", marks=marks_in(0, 0, 3, 0, 1), seed=1)
    make_tiff(root / "b/x.tif", marks=marks_in(0, 0, 3, 0, 1), seed=2)
    m1 = write_examined(tmp_path / "m1.json", {"x.tif": {0: [{"counter": 0, "count": 3}]}})
    with pytest.raises(MLError, match="ambiguous"):
        go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=m1, mode="examined-fields-manifest", dry=True)
    doc = json.loads(m1.read_text())
    doc["files"][0].update(file="a/x.tif", sha256="0" * 64)
    m2 = tmp_path / "m2.json"
    m2.write_text(json.dumps(doc))
    with pytest.raises(MLError, match="different file"):
        go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=m2, mode="examined-fields-manifest", dry=True)


def test_straddling_points_and_cells_that_only_contain_strays(tmp_path):
    # counter 0 clicks 4 times in cell (0,0) and once across the border into (0,1); (0,1) has no counter of its own
    x_cross, y_cross = cell_xy(0, 1, 0.1, 0.5)
    m = marks_in(0, 0, 4, 0, 1) + [(x_cross, y_cross, 0, 1)]
    root = make_one(tmp_path, marks=m)
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    assert region_ids(out.result.manifest.crops) == ["r0c0"]
    f = out.plan.summary["fields"]
    assert STRAD in " ".join(f["reasons"])
    assert out.plan.summary["annotations"]["dropped_by_reason"] == {"cell_not_examined:straddling_other_counter": 1}
    rows = points_rows(tmp_path / "data", out.result.version)
    dropped = [r for r in rows if r["status"] == "dropped"]
    assert len(dropped) == 1 and dropped[0]["basis"] == "dropped:cell_not_examined:straddling_other_counter"
    assert dropped[0]["region_id"] == "r0c1" and dropped[0]["crop_id"] == ""
    # ... but when (0,1) IS examined through its own counter the stray click is a label of that field (recorded)
    root2 = make_one(tmp_path / "w2", marks=m + marks_in(0, 1, 3, 1, 1))
    out2 = go(tmp_path / "w2", root2, im=None, allow_unconfirmed_identity=True)
    c01 = [c for c in out2.result.manifest.crops if c.region_id == "r0c1"][0]
    assert c01.n_positive == 4 and c01.provenance["n_labels_from_other_counters"] == 1
    assert c01.provenance["n_researcher_marks"] == 3           # evidence = its own counter's marks only
    assert out2.plan.summary["annotations"]["straddling_labels_in_examined_cells"] == 1


STRAD = "only straddling points from another counter"


def test_stated_zero_is_contradicted_by_straddling_points(tmp_path):
    x_cross, y_cross = cell_xy(0, 1, 0.1, 0.5)
    root = make_one(tmp_path, marks=marks_in(0, 0, 4, 0, 1) + [(x_cross, y_cross, 0, 1)])
    man = write_examined(tmp_path / "m.json", {"29-09-2026_T7_AB_AM-A.tif": {0: [{"row": 0, "col": 1, "count": 0}]}})
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", dry=True)
    assert out.plan.summary["fields"]["conflict_fields_excluded"] == 1 and out.plan.summary["fields"]["verified_zero_fields"] == 0


def points_rows(data, version):
    with open(Path(data) / "datasets" / version / "points.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_invalid_points_are_dropped_with_a_reason_never_clamped_never_labels(tmp_path):
    x0, y0 = cell_xy(0, 0, 0.2, 0.2)
    good = marks_in(0, 0, 3, 0, 1)
    bad = [(W + 15.5, 20.5, 0, 1),                  # beyond the image -> out_of_bounds
           (4.5, 5.5, 0, 1),                        # inside the image but left/above the grid -> outside_grid
           (good[0][0] + 0.2, good[0][1] + 0.2, 0, 1),   # within 1 px of an earlier point -> duplicate
           # plane-less marks sit in an UNMARKED cell: inside an examined cell they would make that cell a conflict (F2)
           (*cell_xy(2, 2, 0.3, 0.3), 0, 0),        # position 0 -> plane unspecified
           (*cell_xy(2, 2, 0.7, 0.7), 0, 7)]        # position beyond the number of planes -> plane_out_of_range
    root = make_one(tmp_path, marks=good + bad)
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    a = out.plan.summary["annotations"]
    assert a["points_total"] == 8 and a["labels_exported"] == 3
    assert a["dropped_by_reason"] == {"duplicate": 1, "out_of_bounds": 1, "outside_grid": 1, "plane_unspecified": 1,
                                         "plane_out_of_range": 1}
    (c,) = out.result.manifest.crops
    assert c.n_positive == 3
    rows = points_rows(tmp_path / "data", out.result.version)
    assert Counter(r["status"] for r in rows) == {"confirmed": 3, "dropped": 5}
    assert all(r["crop_id"] == "" and r["basis"].startswith("dropped:") for r in rows if r["status"] == "dropped")
    assert out.result.manifest.label_counts["points_by_status"] == {"confirmed": 3, "dropped": 5}


def test_labels_are_exact_in_the_crop_frame(tmp_path, corpus):
    root, files = corpus
    out = go(tmp_path, root, corpus=files)
    rel = files["AM-A"]
    c = [c for c in crops_of(out, rel, 0) if c.region_id == "r0c1"][0]
    img, lab = ds.load_crop(tmp_path / "data", out.result.version, c)
    ref = tifffile.imread(str(root / rel))[0]
    x0, y0, x1, y1 = c.box_xyxy
    assert (x0, y0, x1, y1) == (int(GRID_XS[1]), int(GRID_YS[0]), int(GRID_XS[2]), int(GRID_YS[1]))
    assert np.array_equal(img, ref[y0:y1, x0:x1]) and img.dtype == np.uint16          # native pixels, dtype unchanged
    exp = sorted((y - 0.5 - y0, x - 0.5 - x0) for x, y, _c, _p in marks_in(0, 1, 3, 1, 1, seed=1))
    assert np.allclose(sorted(map(tuple, lab.tolist())), exp, atol=1e-4)


# ================================================================== scan: rejections
def reject_reasons(out):
    return out.plan.summary["tiffs"]["rejected_files"]


def test_structured_rejections(tmp_path):
    root = tmp_path / "in"
    good = marks_in(0, 0, 3, 0, 1)
    make_tiff(root / "ok.tif", marks=good, seed=1)
    make_tiff(root / "no_roi.tif", marks=[], seed=2)                                         # grid, no points
    make_tiff(root / "no_grid.tif", marks=good, grid=False, seed=3)
    make_tiff(root / "neither.tif", marks=[], grid=False, seed=4)
    make_tiff(root / "two_point_blobs.tif", marks=good, extra_overlays=[point_roi([(20.5, 20.5)], [0], [1])], seed=5)
    make_tiff(root / "two_grids.tif", marks=good, extra_overlays=[grid_overlay(GRID_XS, GRID_YS, W, H)], seed=6)
    make_tiff(root / "corrupt.tif", marks=good, roi=b"this is not an imagej roi at all", seed=7)
    make_tiff(root / "float.tif", marks=good, dtype=np.float32, seed=8)
    from legacy_bridge_fixtures import point_roi_no_counters
    make_tiff(root / "no_counters.tif", roi=point_roi_no_counters([(x, y) for x, y, _c, _p in good], position=1), seed=9)
    tifffile.imwrite(str(root / "rgb.tif"), np.zeros((40, 50, 3), np.uint8), photometric="rgb")
    (root / "broken.tif").write_bytes(b"\x00not a tiff\x00" * 20)
    (root / ".hidden").mkdir()
    make_tiff(root / ".hidden/skip.tif", marks=good, seed=10)
    (root / "x.tif.zip").write_bytes(b"PK\x03\x04")
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, dry=True)
    r = reject_reasons(out)
    assert r["no_roi.tif"] == ["missing_roi"] and r["no_grid.tif"] == ["missing_grid"]
    assert r["neither.tif"] == ["missing_roi", "missing_grid"]
    assert r["two_point_blobs.tif"] == ["multiple_roi_blobs"] and r["two_grids.tif"] == ["multiple_grids"]
    assert r["corrupt.tif"] == ["corrupt_metadata"] and r["float.tif"] == ["unsupported_dtype"]
    assert r["no_counters.tif"] == ["no_counters"] and r["rgb.tif"] == ["unsupported_layout"]
    assert r["broken.tif"] == ["tiff_unreadable"]
    t = out.plan.summary["tiffs"]
    assert t["discovered"] == 11 and t["accepted"] == 1 and t["rejected"] == 10      # hidden dir + archive are not discovered
    assert t["rejected_by_reason"]["missing_roi"] == 2 and "/" not in json.dumps(t["rejected_messages"]["broken.tif"])


def test_no_counters_needs_the_acknowledgement_and_is_recorded(tmp_path):
    from legacy_bridge_fixtures import point_roi_no_counters
    good = [(x, y, None, 1) for x, y, _c, p in marks_in(0, 0, 3, 0, 1)]
    root = make_one(tmp_path, roi=point_roi_no_counters([(x, y) for x, y, _c, _p in good], position=1))
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, dry=True)
    assert out.plan.summary["tiffs"]["rejected_by_reason"] == {"no_counters": 1}
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, acknowledge_no_counters=True)
    (c,) = out.result.manifest.crops
    assert any("no ImageJ counter" in a for a in c.provenance["assumptions"])
    assert any("acknowledge-no-counters" in a for a in out.result.manifest.unresolved_assumptions)


def test_non_sequential_counters_and_missing_counter_numbers(tmp_path):
    m = marks_in(0, 0, 3, 7, 1) + marks_in(0, 1, 2, 200, 1) + marks_in(2, 2, 2, 3, 1)
    root = make_one(tmp_path, marks=m)
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    cs = {c.region_id: c for c in out.result.manifest.crops}
    assert sorted(cs) == ["r0c0", "r0c1", "r2c2"]
    assert cs["r0c0"].provenance["field_counters"] == [7] and cs["r0c0"].field_index == 8
    assert cs["r0c1"].provenance["field_counter"] == 200


def test_byte_identical_duplicates_are_deduplicated_with_alias_recorded(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "b_copy.tif", marks=marks_in(0, 0, 3, 0, 1), seed=5)
    shutil.copy(root / "b_copy.tif", root / "a_first.tif")
    (root / "sub").mkdir()
    shutil.copy(root / "b_copy.tif", root / "sub" / "z.tif")
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    t = out.plan.summary["tiffs"]
    assert t["discovered"] == 3 and t["unique"] == 1 and t["deduplicated"] == 2
    assert t["deduplicated_paths"] == {"a_first.tif": ["b_copy.tif", "sub/z.tif"]}
    (c,) = out.result.manifest.crops
    assert c.source_file == "a_first.tif" and c.provenance["alias_paths"] == ["b_copy.tif", "sub/z.tif"]


# ================================================================== channel map
def test_channel_map_is_required_for_a_real_import_and_reported_in_dry_run(tmp_path, corpus):
    root, files = corpus
    with pytest.raises(MLError, match="channel-map"):
        lb.import_legacy(tmp_path / "d", root, opts(allow_unconfirmed_identity=True), say=Recorder())
    out = lb.import_legacy(tmp_path / "d", root, opts(allow_unconfirmed_identity=True), dry_run=True, say=Recorder())
    assert out.result is None and any("channel-map" in b for b in out.blockers)
    assert out.plan.summary["fields"]["positive_marked_fields"] == 16
    assert not (tmp_path / "d").exists()


def test_channel_map_validation_errors_are_actionable(tmp_path, corpus):
    root, files = corpus
    d = tmp_path
    def doc(**over):
        base = {"schema": "polony-channel-map", "version": 1, "confirmed_by": "E", "confirmed_on": "x", "source_of_truth": "y",
                "defaults": {"pages": {"0": {"role": "red", "train": True}, "1": {"role": "green", "train": True}}}}
        base.update(over)
        p = d / "bad.json"
        p.write_text(json.dumps(base))
        return p
    with pytest.raises(MLError, match="confirmed_by"):
        lb.load_channel_map(doc(confirmed_by=""))
    with pytest.raises(MLError, match="no role"):
        lb.load_channel_map(doc(defaults={"pages": {"0": {"role": None, "train": True}}}))
    with pytest.raises(MLError, match="unique"):
        lb.load_channel_map(doc(defaults={"pages": {"0": {"role": "red", "train": True}, "1": {"role": "red", "train": True}}}))
    with pytest.raises(MLError, match="'train'"):
        lb.load_channel_map(doc(defaults={"pages": {"0": {"role": "red", "train": "yes"}}}))
    with pytest.raises(MLError, match="schema"):
        lb.load_channel_map(doc(schema="other"))
    with pytest.raises(MLError, match="'match' must use only"):
        lb.load_channel_map(doc(rules=[{"id": "r", "match": {"colour": "red"}, "pages": {"0": {"role": "a", "train": True}}}]))
    # F8: no default for 'train', no unknown key at any level (a typo must never silently change what is trained)
    with pytest.raises(MLError, match="no 'train'"):
        lb.load_channel_map(doc(defaults={"pages": {"0": {"role": "red"}}}))
    with pytest.raises(MLError, match="unknown key.*trian"):
        lb.load_channel_map(doc(defaults={"pages": {"0": {"role": "red", "trian": False, "train": True}}}))
    with pytest.raises(MLError, match="unknown key.*defalts"):
        lb.load_channel_map(doc(defalts=1))
    with pytest.raises(MLError, match="unknown key.*mach"):
        lb.load_channel_map(doc(rules=[{"id": "r", "mach": {"glob": "*"}, "match": {"glob": "*"},
                                        "pages": {"0": {"role": "a", "train": True}}}]))
    with pytest.raises(MLError, match="unknown key.*pagez"):
        lb.load_channel_map(doc(files={"a.tif": {"pagez": 1, "pages": {"0": {"role": "a", "train": True}}}}))
    with pytest.raises(MLError, match="unknown key.*idd"):
        lb.load_channel_map(doc(defaults=[{"idd": "x", "pages": {"0": {"role": "a", "train": True}}}]))
    lb.load_channel_map(doc(_comment="comments with a leading underscore are fine"))
    # a template left unfilled is refused
    lb.write_templates(tmp_path / "t", root)
    with pytest.raises(MLError, match="confirmed_by"):
        lb.load_channel_map(tmp_path / "t" / "channels.template.json")


def test_template_default_is_inactive_until_confirmed_then_applies_as_default(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1) + marks_in(0, 0, 2, 0, 2))
    lb.write_templates(tmp_path / "t", root)
    tpl = tmp_path / "t/channels.template.json"
    with pytest.raises(MLError, match="who confirmed this mapping"):
        lb.load_channel_map(tpl)
    doc = json.loads(tpl.read_text())
    doc.update(confirmed_by="E. Researcher", confirmed_on="2026-10-13")
    f = tmp_path / "half.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(MLError, match="source_of_truth"):
        lb.load_channel_map(f)
    doc["source_of_truth"] = "validated on the 8 research stacks"
    f.write_text(json.dumps(doc))
    out = go(tmp_path, root, cm=f, im=None, allow_unconfirmed_identity=True)
    (c,) = out.result.manifest.crops                     # page 1 (green) is mapped but train: false
    assert c.channel_role == "red" and c.provenance["channel_mapping_origin"] == "default"
    assert out.plan.summary["annotations"]["dropped_by_reason"] == {"page_not_trained": 2}


def test_info_label_hint_is_reported_but_never_decides_a_role(tmp_path):
    root = tmp_path / "in"
    for n, hint in (("a.tif", "635 [W1]"), ("b.tif", "635 [W1]"), ("c.tif", "532 [W2]"), ("d.tif", None)):
        make_tiff(root / n, marks=marks_in(0, 0, 3, 0, 1), seed=ord(n[0]),
                  info=(f"ImageDescription: {hint}" if hint else None))
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, dry=True)
    chk = out.plan.summary["info_label_check"]
    assert chk["usual_hint_by_page0_role"] == {"red": "635 [W1]"} and chk["different"] == ["c.tif"] and chk["absent"] == ["d.tif"]
    assert out.plan.summary["channel_mappings"]["origin_counts"] == {"default": 4}     # roles unchanged for the deviating files
    from polony_app.ml.cli_legacy import format_report
    text = "\n".join(format_report(out, tmp_path / "data"))
    assert "deviates: c.tif (different)" in text and "deviates: d.tif (absent)" in text


def test_conflicting_rules_name_both_rules_and_the_file(tmp_path, corpus):
    root, files = corpus
    cm = write_channel_map(tmp_path / "c.json", rules=[
        {"id": "by-glob", "match": {"glob": "*AM-A*"}, "pages": {0: "red", 1: "green"}},
        {"id": "by-assay", "match": {"assay": "T7_AB"}, "pages": {0: "green", 1: "red"}}])
    with pytest.raises(MLError) as e:
        go(tmp_path, root, cm=cm, corpus=files, dry=True)
    msg = str(e.value)
    assert "by-glob" in msg and "by-assay" in msg and files["AM-A"] in msg and "files" in msg


def test_channel_map_precedence_origin_and_unresolved_files(tmp_path):
    root = tmp_path / "in"
    paths = {}
    for k, n in enumerate(["f_file.tif", "r_rule.tif", "d_default.tif", "u_unmapped.tif", "p_bad_page.tif"]):
        make_tiff(root / n, marks=marks_in(0, 0, 3, 0, 1), seed=k + 1, planes=(1 if n == "p_bad_page.tif" else 2))
    cm = write_channel_map(
        tmp_path / "c.json", defaults={0: "red", 1: "green"},
        rules=[{"id": "r1", "match": {"glob": "r_*.tif"}, "pages": {0: "cy5", 1: "cy3"}},
               {"id": "r2", "match": {"glob": "u_*.tif"}, "pages": {0: "cy5", 1: "cy3"}}],
        files={"f_file.tif": {0: "FILE0", 1: ("FILE1", False)}})
    out = go(tmp_path, root, cm=cm, im=None, allow_unconfirmed_identity=True)
    by = {c.source_file: c for c in out.result.manifest.crops}
    assert by["f_file.tif"].channel_role == "FILE0" and by["f_file.tif"].provenance["channel_mapping_origin"] == "file"
    assert by["r_rule.tif"].channel_role == "cy5" and by["r_rule.tif"].provenance["channel_mapping_origin"] == "rule:r1"
    assert by["d_default.tif"].channel_role == "red" and by["d_default.tif"].provenance["channel_mapping_origin"] == "default"
    assert out.plan.summary["channel_mappings"]["origin_counts"] == {"default": 3, "file": 1, "rule:r1": 1, "rule:r2": 1} \
        or "rule:r2" in out.plan.summary["channel_mappings"]["origin_counts"]
    assert out.plan.summary["channel_mappings"]["map"]["sha256"] == lb.load_channel_map(cm).sha256
    # file mapped with a page beyond its planes is rejected, never guessed
    assert reject_reasons(out) == {"p_bad_page.tif": ["channel_mapping_page_out_of_range"]}


def test_unresolved_mapping_rejects_the_file(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "a.tif", marks=marks_in(0, 0, 3, 0, 1), seed=1)
    make_tiff(root / "b.tif", marks=marks_in(0, 0, 3, 0, 1), seed=2)
    cm = write_channel_map(tmp_path / "c.json", files={"a.tif": {0: "red", 1: "green"}})
    out = go(tmp_path, root, cm=cm, im=None, allow_unconfirmed_identity=True)
    assert reject_reasons(out) == {"b.tif": ["channel_mapping_unresolved"]}
    assert [c.source_file for c in out.result.manifest.crops] == ["a.tif", "a.tif"][:len(out.result.manifest.crops)]


def test_train_false_pages_and_roles_are_free_labels_sanitised(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 3, 0, 1) + marks_in(0, 0, 2, 0, 2))
    cm = write_channel_map(tmp_path / "c.json", defaults={0: "Cy5 / red", 1: ("green", False)})
    out = go(tmp_path, root, cm=cm, im=None, allow_unconfirmed_identity=True)
    (c,) = out.result.manifest.crops
    assert c.channel_role == "Cy5 / red" and "Cy5---red" in c.crop_id
    assert out.plan.summary["annotations"]["dropped_by_reason"] == {"page_not_trained": 2}
    assert out.result.manifest.sources[0].channel_mapping == {"Cy5 / red": {"file": c.source_file, "page": 0}}


# ================================================================== identity
def test_inferred_identity_blocks_a_real_import_and_is_reported_in_dry_run(tmp_path, corpus):
    root, files = corpus
    with pytest.raises(MLError, match="UNCONFIRMED"):
        go(tmp_path, root, im=None)
    out = go(tmp_path, root, im=None, dry=True)
    assert out.result is not None and out.result.dry_run
    assert any("UNCONFIRMED" in b for b in out.blockers)
    assert out.plan.summary["identity"]["unconfirmed"] == 4
    assert not (tmp_path / "data").exists()
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True)
    m = out.result.manifest
    assert m.splits["unconfirmed_identity"] is True and "allow_unconfirmed_identity" in m.crops[0].provenance["assumptions"]


def test_identity_map_rows_unconfirmed_ambiguous_and_missing_are_reported(tmp_path, corpus):
    root, files = corpus
    rows = [{"file": files["AM-A"], "specimen_id": "AM", "slide_id": "AM_A", "confirmed": True},
            {"file": files["AM-B"], "specimen_id": "AM", "slide_id": "AM_B", "confirmed": False},
            {"file": "ghost.tif", "specimen_id": "X", "slide_id": "X_A", "confirmed": True}]
    p = tmp_path / "id.json"
    p.write_text(json.dumps(rows))
    tbl = lb.load_identity_map(p)
    out = go(tmp_path, root, im=p, dry=True)
    rep = out.plan.summary["identity"]["map"]
    assert rep["missing_on_disk"] == ["ghost.tif"] and rep["unconfirmed_rows"] == [files["AM-B"]]
    assert sorted(rep["unlisted_files"]) == sorted([files["AS-A"], files["AS-B"]])
    assert out.plan.summary["identity"]["confirmed"] == 1
    # CSV form + unique base name resolution
    c = tmp_path / "id.csv"
    c.write_text("file,specimen_id,slide_id,confirmed\n" + "\n".join(f"{Path(f).name},{k.split('-')[0]},{k.replace('-', '_')},true"
                                                                   for k, f in files.items()) + "\n")
    out = go(tmp_path, root, im=c, dry=True)
    assert out.plan.summary["identity"]["confirmed"] == 4 and not out.blockers
    with pytest.raises(MLError, match="neither specimen_id nor slide_id"):
        (tmp_path / "bad.json").write_text(json.dumps([{"file": "a.tif", "confirmed": True}]))
        lb.load_identity_map(tmp_path / "bad.json")


def test_basename_that_is_not_unique_is_not_applied(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "a/x.tif", marks=marks_in(0, 0, 3, 0, 1), seed=1)
    make_tiff(root / "b/x.tif", marks=marks_in(0, 0, 3, 0, 1), seed=2)
    p = tmp_path / "id.json"
    p.write_text(json.dumps([{"file": "x.tif", "specimen_id": "S", "slide_id": "S_A", "confirmed": True}]))
    out = go(tmp_path, root, im=p, dry=True)
    assert out.plan.summary["identity"]["map"]["ambiguous_basename"] == ["x.tif"] and out.plan.summary["identity"]["confirmed"] == 0


def test_identity_map_keeps_both_slides_of_a_specimen_together(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS"))
    out = go(tmp_path, root, corpus=files, fractions=FR3, seed=3)
    m = out.result.manifest
    spl = m.splits["analysis_to_split"]
    ids = {s.name: s.analysis_id for s in m.sources}
    for sp in ("AM", "AS", "FS"):
        a, b = ids[files[f"{sp}-A"]], ids[files[f"{sp}-B"]]
        assert m.splits["analysis_to_group"][a] == m.splits["analysis_to_group"][b] and spl[a] == spl[b]
    assert sorted({v for v in spl.values()}) == ["test", "train", "val"]
    assert m.splits["unconfirmed_identity"] is False
    assert all(c.group_id == m.splits["analysis_to_group"][c.analysis_id] for c in m.crops)


# ================================================================== dry run
def tree(p):
    return sorted(str(q.relative_to(p)) for q in Path(p).rglob("*")) if Path(p).exists() else None


def test_dry_run_writes_nothing_and_reports_the_excluded_counts(tmp_path, corpus, capsys):
    root, files = corpus
    cm = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    before = tree(tmp_path)
    code = main(["--data-dir", str(tmp_path / "data"), "import-legacy", "--input", str(root), "--channel-map", str(cm),
                 "--dry-run", "--fractions", "0.5,0.5,0"])
    assert code == 0
    text = capsys.readouterr().out
    assert tree(tmp_path) == before and not (tmp_path / "data").exists()
    for needle in ("IMPORT-LEGACY DRY RUN", "SOURCES: discovered 4", "EXCLUDED because the researcher's examination could not be proven: 56",
                   "positive_marked 16", "page 0 / red", "page 1 / green", "GROUPS AND SPLITS", "native snapshot size", "per epoch",
                   "BLOCKERS", "UNCONFIRMED", "next commands", "filename_pattern"):
        assert needle in text, needle


def test_dry_run_exit_codes_malformed_files_refuse(tmp_path, corpus, capsys):
    root, files = corpus
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    code = main(["--data-dir", str(tmp_path / "d"), "import-legacy", "--input", str(root), "--channel-map", str(bad), "--dry-run"])
    assert code == 2 and "REFUSED" in capsys.readouterr().err
    code = main(["--data-dir", str(tmp_path / "d"), "import-legacy", "--input", str(root), "--dry-run",
                 "--mode", "examined-fields-manifest", "--examined-fields", str(bad)])
    assert code == 2
    code = main(["--data-dir", str(tmp_path / "d"), "import-legacy", "--dry-run"])
    assert code == 2 and "--input" in capsys.readouterr().err


def test_write_templates_reads_no_pixels(tmp_path, corpus, monkeypatch):
    root, files = corpus
    monkeypatch.setattr(lb, "read_plane", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pixel read")))
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pixel read")))
    code = main(["import-legacy", "--input", str(root), "--write-templates", str(tmp_path / "tpl")])
    assert code == 0
    ident = json.loads((tmp_path / "tpl/identities.template.json").read_text())["rows"]
    assert len(ident) == 4 and all(r["confirmed"] is False and r["specimen_id"] for r in ident)
    chan = json.loads((tmp_path / "tpl/channels.template.json").read_text())
    assert chan["defaults"]["pages"] == {"0": {"role": "red", "train": True}, "1": {"role": "green", "train": False}}
    assert chan["confirmed_by"] == "" and chan["confirmed_on"] == "" and chan["source_of_truth"] == ""
    exa = json.loads((tmp_path / "tpl/examined-fields.template.json").read_text())
    assert [f["file"] for f in exa["files"]] == sorted(files.values()) and exa["evidence"] == ""
    with pytest.raises(MLError, match="evidence"):
        lb.load_examined(tmp_path / "tpl/examined-fields.template.json")
    # without --input: skeleton templates
    assert main(["import-legacy", "--write-templates", str(tmp_path / "tpl2")]) == 0


# ================================================================== determinism / resume / lifecycle
def snap_digest(data, version):
    root = Path(data) / "datasets" / version
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_identical_inputs_give_identical_snapshots_for_any_worker_count(tmp_path, corpus):
    root, files = corpus
    a = go(tmp_path, root, corpus=files, data=tmp_path / "d1", workers=1)
    b = go(tmp_path, root, corpus=files, data=tmp_path / "d2", workers=3)
    c = go(tmp_path, root, corpus=files, data=tmp_path / "d3", workers=2, resume=False)
    assert a.result.version == b.result.version == c.result.version
    assert snap_digest(tmp_path / "d1", a.result.version) == snap_digest(tmp_path / "d2", b.result.version) \
        == snap_digest(tmp_path / "d3", c.result.version)
    assert ds.verify_snapshot(tmp_path / "d1", a.result.version) == []
    # a different input option changes the version
    d = go(tmp_path, root, corpus=files, data=tmp_path / "d4", seed=5)
    assert d.result.version != a.result.version


def test_interrupted_import_leaves_no_snapshot_and_a_rerun_resumes_identically(tmp_path, corpus, monkeypatch):
    root, files = corpus
    clean = go(tmp_path, root, corpus=files, data=tmp_path / "clean")
    real = ds.write_crop_files
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 9:
            raise OSError("disk went away")
        return real(*a, **k)
    monkeypatch.setattr(ds, "write_crop_files", flaky)
    with pytest.raises(OSError):
        go(tmp_path, root, corpus=files, data=tmp_path / "data")
    assert ds.list_snapshots(tmp_path / "data") == []
    staging = [p for p in (tmp_path / "data/datasets").iterdir() if p.name.startswith(".staging-")]
    assert len(staging) == 1 and any((staging[0] / ".done").iterdir())      # finished sources are kept for the resume
    monkeypatch.setattr(ds, "write_crop_files", real)
    n_before = calls["n"]
    rec = {"n": 0}

    def counting(*a, **k):
        rec["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(ds, "write_crop_files", counting)
    out = go(tmp_path, root, corpus=files, data=tmp_path / "data")
    assert out.result.version == clean.result.version and rec["n"] < 16     # only the unfinished sources were rewritten
    assert snap_digest(tmp_path / "data", out.result.version) == snap_digest(tmp_path / "clean", clean.result.version)
    assert not [p for p in (tmp_path / "data/datasets").iterdir() if p.name.startswith(".staging-")]


def test_source_changed_after_scan_is_refused(tmp_path, corpus, monkeypatch):
    root, files = corpus
    plan = lb.build_plan(root, opts(allow_unconfirmed_identity=True), channel_map=write_channel_map(tmp_path / "c.json", defaults=RED_GREEN),
                         say=Recorder())
    p = root / files["AM-A"]
    os.utime(p, ns=(1, 1))
    with pytest.raises(MLError, match="changed on disk"):
        plan.plans[0].ext.prepare()


def test_snapshot_passes_verify_lock_test_and_train_dry_run(tmp_path, monkeypatch, capsys):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS", "PS", "QS"))
    cm = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    im = write_identity_map(tmp_path / "id.json", files)
    d = tmp_path / "data"
    args = ["--data-dir", str(d), "import-legacy", "--input", str(root), "--channel-map", str(cm), "--identity-map", str(im),
            "--fractions", "0.6,0.2,0.2", "--created-at", T0, "--workers", "2"]
    assert main(args) == 0
    version = ds.list_snapshots(d)[0]
    capsys.readouterr()
    assert main(["--data-dir", str(d), "verify", version]) == 0 and "intact" in capsys.readouterr().out
    assert main(["--data-dir", str(d), "split", "--snapshot", version, "--lock-test"]) == 0
    assert "LOCKED test set" in capsys.readouterr().out
    # a rebuild on top of the lock carries the locked crops over (re-import of the same files lands them in test)
    m = ds.load_manifest(d, version)
    assert {c.split for c in m.crops} == {"train", "val", "test"} and lb.verify_examination_provenance(m) == []
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, dd: {"kind": "folder", "name": "stubbase", "path": "/nonexistent", "registry_id": None})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.load_model", lambda *a, **k: object())
    monkeypatch.setattr("polony_app.ml.spotiflow_io.hardware_info", lambda device="cpu": {"platform": "test", "cuda_available": False})
    code = main(["--data-dir", str(d), "train", "--snapshot", version, "--dry-run", "--allow-small", "--epochs", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "nothing was trained" in out
    # train/val/test all carry the examination provenance and nothing else is in the data
    for c in m.crops:
        assert c.provenance["researcher_examined"] is True


def test_overlay_cells_and_detections_outside_examined_fields_never_enter_the_data(tmp_path, corpus):
    root, files = corpus
    out = go(tmp_path, root, corpus=files)
    m = out.result.manifest
    assert len({(c.source_file, c.page, c.region_id) for c in m.crops}) == len(m.crops) == 16
    rows = points_rows(tmp_path / "data", out.result.version)
    assert len(rows) == 44 and all(r["status"] == "confirmed" and r["crop_id"] for r in rows)
    reg = [json.loads(l) for l in (tmp_path / "data/datasets" / out.result.version / "regions.jsonl").read_text().splitlines()]
    assert len(reg) == 4 * 9                                              # every grid cell stays auditable ...
    n_unknown = 0
    for r in reg:
        for pg, v in r["examination"].items():
            n_unknown += v["status"] == "unknown"
            has_crop = any(k.endswith({"0": ":red", "1": ":green"}[pg]) for k in r["crops"])
            assert has_crop == (v["status"] == "positive_marked")           # ... but only examined tuples have a crop
    assert n_unknown == 4 * 9 * 2 - 16


def test_second_import_after_a_test_lock_keeps_locked_sources_in_test_and_unexported(tmp_path, capsys):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS", "PS", "QS"))
    cm = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    im = write_identity_map(tmp_path / "id.json", files)
    d = tmp_path / "data"
    first = lb.import_legacy(d, root, opts(fractions={"train": 0.6, "val": 0.2, "test": 0.2}), channel_map=cm, identity_map=im,
                             say=Recorder())
    ds.lock_test_set(d, first.result.version)
    n_test = len([c for c in first.result.manifest.crops if c.split == "test"])
    second = lb.import_legacy(d, root, opts(fractions={"train": 0.6, "val": 0.2, "test": 0.2}, seed=9), channel_map=cm,
                              identity_map=im, say=Recorder())
    m = second.result.manifest
    frozen = [c for c in m.crops if c.frozen_from]
    assert len(frozen) == n_test > 0 and all(c.split == "test" for c in frozen)
    assert not [c for c in m.crops if c.split == "test" and not c.frozen_from]
    assert lb.verify_examination_provenance(m) == []                   # carried-over crops keep their provenance
    assert ds.verify_snapshot(d, second.result.version) == []


def test_estimates_follow_the_trainer_formulas():
    e = estimate_training(10, 3, 4, 5, downscale=2, crop_hw=(316, 316))
    assert e["steps_per_epoch"] == 3 and e["optimizer_steps"] == 15 and e["validation_batches"] == 3
    assert e["validation_passes"] == 5 and e["prepared_bytes"] == 13 * 158 * 158 * 4
    e = estimate_training(10, 3, 4, 5, num_train_samples=100)
    assert e["steps_per_epoch"] == 25 and e["optimizer_steps"] == 125 and e["prepared_bytes"] == 13 * 316 * 316 * 4
    e = estimate_training(1, 1, 8, 1, train_shapes_hw=[(35, 50)], val_shapes_hw=[(35, 50)], downscale=4)
    assert e["prepared_bytes"] == 2 * (35 // 4) * (50 // 4) * 4


# ================================================================== lab CSV helper
def test_lab_csv_needs_acknowledgement_and_series_and_converts_to_a_manifest(tmp_path):
    root = make_one(tmp_path, marks=marks_in(0, 0, 4, 0, 1) + marks_in(0, 1, 3, 1, 1) + marks_in(0, 0, 2, 0, 2))
    c = tmp_path / "T7B_15-09-2026.csv"
    c.write_text("slide ID,field 1,field 2,field 3,field 4\nAM_A,4,3,,0\nXX_A,1,1,1,1\n")
    c2 = tmp_path / "T7A_15-09-2026.csv"
    c2.write_text("slide ID,field 1,field 2\nAM_A,2,\n")
    kw = dict(im=None, allow_unconfirmed_identity=True, mode="examined-fields-manifest", dry=True)
    cm = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)

    def run(**o):
        return lb.import_legacy(tmp_path / "d", root, opts(**{k: v for k, v in kw.items() if k not in ("im", "dry")}, **o),
                                channel_map=cm, lab_csv=[c, c2], lab_series=["T7B=0", "T7A=1"], dry_run=True, say=Recorder())
    with pytest.raises(MLError, match="acknowledge-lab-csv-field-numbering"):
        run()
    out = run(acknowledge_lab_csv_field_numbering=True)
    f = out.plan.summary["fields"]
    # page 0: counters 0 (4) and 1 (3) match their marks -> positive; field 4 = counter 3 with count 0 cannot be located
    # (a counter without a single mark has no cell) -> unresolved, never guessed; blank field 3 = not counted -> not an entry
    assert f["positive_marked_fields"] == 3 and f["verified_zero_fields"] == 0
    assert len(out.plan.summary["unresolved_manifest_entries"]) == 1
    lab = out.plan.summary["lab_csv"]
    assert lab["applied_rows"] == 2 and lab["unmatched_rows"] == ["T7B:XX_A"] and lab["series"] == {"T7B": 0, "T7A": 1}
    assert any("lab CSV" in a for a in out.plan.summary["assumptions"]) or any("lab-csv" in a for a in out.plan.summary["assumptions"])
    with pytest.raises(MLError, match="lab-series"):
        lb.import_legacy(tmp_path / "d", root, opts(acknowledge_lab_csv_field_numbering=True, mode="examined-fields-manifest"),
                         channel_map=cm, lab_csv=[c], lab_series=[], dry_run=True, say=Recorder())
    with pytest.raises(MLError, match="not in --lab-series"):
        lb.import_legacy(tmp_path / "d", root, opts(acknowledge_lab_csv_field_numbering=True, mode="examined-fields-manifest"),
                         channel_map=cm, lab_csv=[c], lab_series=["XX=0"], dry_run=True, say=Recorder())


# ================================================================== randomised adversarial tests
def synth_scan(nr, nc, pts):
    """pts: [(cell or -1, counter or -1, page, pstatus)] -> a SourceScan without any file."""
    sc = lb.SourceScan(source_id="img_synth", rel="s.tif", path=Path("s.tif"), sha256="0" * 64, size=0, mtime_ns=0, n_planes=3)
    sc.xs = tuple(float(10 + 50 * k) for k in range(nc + 1))
    sc.ys = tuple(float(10 + 35 * k) for k in range(nr + 1))
    n = len(pts)
    sc.x = np.arange(n, dtype=float)
    sc.y = np.arange(n, dtype=float)
    sc.cell = np.array([p[0] for p in pts], np.int64)
    sc.counter = np.array([p[1] for p in pts], np.int32)
    sc.page = np.array([p[2] for p in pts], np.int32)
    sc.pstatus = np.array([p[3] for p in pts], np.int8)
    maj = {}
    cnt = {}
    for cell, ctr, _pg, st in pts:
        if st == 0 and ctr >= 0 and cell >= 0:
            cnt.setdefault(ctr, Counter())[cell] += 1
    for ctr, c in cnt.items():
        maj[ctr] = c.most_common(1)[0][0]
    sc.counter_cell = maj
    for ctr, cell in maj.items():
        sc.cell_counters.setdefault(cell, []).append(ctr)
    return sc


def random_points(rng, nr, nc, n):
    out = []
    for _ in range(n):
        cell = int(rng.integers(-1, nr * nc))
        out.append((cell, int(rng.integers(-1, 6)), int(rng.integers(-1, 4)), int(rng.choice([0, 0, 0, 0, 1, 2]))))
    return out


def unplaced_cells(sc):
    """cells holding a usable mark whose plane is unspecified (-1) or beyond the file's planes (F2)."""
    return {int(sc.cell[i]) for i in range(sc.n_points)
            if sc.pstatus[i] == 0 and sc.cell[i] >= 0 and not 0 <= sc.page[i] < sc.n_planes}


def check_invariants(sc, pages, res):
    bad_cells = unplaced_cells(sc)
    for (pg, cell), t in res.items():
        if cell in bad_cells:       # a mark of unknown plane may belong to any page: nothing of this cell is proven or zero
            assert t.exam.status == ex.CONFLICT and t.labels == [] and not t.exam.researcher_examined
            continue
        ok_here = [i for i in range(sc.n_points) if sc.page[i] == pg and sc.pstatus[i] == 0 and sc.cell[i] == cell]
        own = [i for i in ok_here if sc.counter[i] < 0 or sc.counter_cell.get(int(sc.counter[i])) == cell]
        if t.exam.status == ex.POSITIVE:
            assert own, "positive without a mark of its own counter inside the cell on THIS page"
            assert set(t.labels) == set(ok_here)
        else:
            assert t.labels == [] and not t.exam.researcher_examined
        if own:
            assert t.exam.status == ex.POSITIVE
        if not own and ok_here:
            assert t.exam.status == ex.UNKNOWN and t.exam.reason == lb.STRADDLE_REASON
        for i in t.labels:
            assert sc.page[i] == pg and sc.cell[i] == cell and sc.pstatus[i] == 0


def test_randomised_eligibility_invariants_and_page_independence():
    rng = np.random.default_rng(1234)
    for _ in range(300):
        nr, nc = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        pts = random_points(rng, nr, nc, int(rng.integers(0, 40)))
        sc = synth_scan(nr, nc, pts)
        res = lb.classify_source(sc, [0, 1, 2], mode="marked-only", manifest=None, entries={})
        assert len(res) == 3 * nr * nc
        check_invariants(sc, [0, 1, 2], res)
        # metamorphic: deleting every point of the OTHER pages never changes a page's tuples
        for page in (0, 1, 2):
            only = synth_scan(nr, nc, [p for p in pts if p[2] == page or not 0 <= p[2] < 3])
            only.counter_cell, only.cell_counters = sc.counter_cell, sc.cell_counters
            r2 = lb.classify_source(only, [page], mode="marked-only", manifest=None, entries={})
            for cell in range(nr * nc):
                assert r2[(page, cell)].exam.status == res[(page, cell)].exam.status
        # manifest mode with nothing listed: nothing is eligible, ever
        none = lb.classify_source(sc, [0, 1], mode="examined-fields-manifest", manifest=None, entries={})
        assert not any(t.exam.researcher_examined for t in none.values())


def test_hypothesis_eligibility_invariants():
    hyp = pytest.importorskip("hypothesis")
    from hypothesis import given, settings, strategies as st
    pt = st.tuples(st.integers(-1, 8), st.integers(-1, 5), st.integers(-1, 3), st.sampled_from([0, 0, 0, 1, 2]))

    @settings(max_examples=150, deadline=None)
    @given(st.lists(pt, max_size=30))
    def prop(points):
        sc = synth_scan(3, 3, points)
        res = lb.classify_source(sc, [0, 1, 2], mode="marked-only", manifest=None, entries={})
        check_invariants(sc, [0, 1, 2], res)
        # the counts the dry run reports are the sum of the tuples, nothing is lost or invented
        tally = ex.ExaminationTally()
        for t in res.values():
            tally.add(t.exam)
        assert tally.eligible_examined + tally.excluded_unproven == 27 and tally.verified_zero == 0
        assert tally.conflict == 3 * len(unplaced_cells(sc))
    prop()


def test_randomised_manifest_zero_requires_no_marks_anywhere_near():
    rng = np.random.default_rng(99)
    man = ex.ExaminedManifest(evidence="t", files=[])
    for _ in range(200):
        nr, nc = 2, 3
        pts = random_points(rng, nr, nc, int(rng.integers(0, 25)))
        sc = synth_scan(nr, nc, pts)
        entries = {(pg, c): ex.FieldEntry(row=c // nc, col=c % nc, count=0) for pg in (0, 1) for c in range(nr * nc)}
        res = lb.classify_source(sc, [0, 1], mode="examined-fields-manifest", manifest=man, entries=entries)
        bad_cells = unplaced_cells(sc)
        for (pg, cell), t in res.items():
            if cell in bad_cells:
                assert t.exam.status == ex.CONFLICT and t.labels == []
                continue
            anything = any(sc.page[i] == pg and sc.pstatus[i] == 0 and (sc.cell[i] == cell or (
                sc.counter[i] >= 0 and sc.counter_cell.get(int(sc.counter[i])) == cell)) for i in range(sc.n_points))
            if t.exam.status == ex.VERIFIED_ZERO:
                assert not anything and t.labels == []
            else:
                assert anything and t.exam.status == ex.CONFLICT


def test_randomised_end_to_end_coordinate_mapping(tmp_path):
    rng = np.random.default_rng(7)
    root = tmp_path / "in"
    expected = {}
    for k in range(4):
        marks = []
        for r in range(3):
            for c in range(3):
                if rng.random() < 0.5:
                    n = int(rng.integers(1, 5))
                    for j in range(n):
                        x = GRID_XS[c] + 2 + float(rng.integers(0, 4 * 48)) / 4 + 0.25
                        y = GRID_YS[r] + 2 + float(rng.integers(0, 4 * 31)) / 4 + 0.25
                        marks.append((x, y, r * 3 + c, int(rng.integers(1, 3))))
        marks = sorted({(round(x, 2), round(y, 2), ctr, pos) for x, y, ctr, pos in marks})
        make_tiff(root / f"f{k}.tif", marks=marks, seed=30 + k)
        expected[f"f{k}.tif"] = marks
    out = go(tmp_path, root, im=None, allow_unconfirmed_identity=True, fractions={"train": 1.0, "val": 0.0, "test": 0.0})
    n_lab = 0
    for c in out.result.manifest.crops:
        img, lab = ds.load_crop(tmp_path / "data", out.result.version, c)
        x0, y0, x1, y1 = c.box_xyxy
        ref = tifffile.imread(str(root / c.source_file))[c.page]
        assert np.array_equal(img, ref[y0:y1, x0:x1])
        r, cc = c.provenance["grid_row"], c.provenance["grid_col"]
        want = sorted((y - 0.5 - y0, x - 0.5 - x0) for x, y, ctr, pos in expected[c.source_file]
                      if pos == c.page + 1 and GRID_XS[cc] <= x < GRID_XS[cc + 1] and GRID_YS[r] <= y < GRID_YS[r + 1])
        # duplicates (<= 1 px apart) are dropped by the importer: compare after the same rule
        got = sorted(map(tuple, lab.tolist()))
        assert len(got) <= len(want) and len(got) == c.n_positive
        for g in got:
            assert any(abs(g[0] - w[0]) < 1e-3 and abs(g[1] - w[1]) < 1e-3 for w in want)
            assert -0.5 <= g[0] < img.shape[0] - 0.5 and -0.5 <= g[1] < img.shape[1] - 0.5
        n_lab += len(got)
    assert n_lab == out.plan.summary["annotations"]["labels_exported"] > 20


# ================================================================== legacy_import perf fix
def test_convert_annotations_is_not_quadratic_and_matches_the_naive_result():
    from polony_app.legacy_import import importer as imp
    from polony_app.legacy_import import roi as roimod
    from polony_app.legacy_import.schema import FieldStatus, ImageKind, ImageRecord
    rng = np.random.default_rng(0)
    n = 60
    xs = [float(10 * k) for k in range(n + 1)]
    ys = [float(10 * k) for k in range(n + 1)]
    rec = ImageRecord(image_id="img_t", file_name="t.tif", rel_path="t.tif", sha256="0" * 64, size_bytes=0, kind=ImageKind.stack,
                      width=10 * n, height=10 * n, n_planes=2, dtype="uint16", bit_depth=16)
    b = roimod.RoiBundle()
    b.grids.append(roimod.RawGrid(0, "Overlays", xs, ys, True, (10.0, 10.0)))
    for k in range(6000):
        b.points.append(roimod.RawPoint(1, k, "ROI", float(rng.uniform(0, 10 * n)), float(rng.uniform(0, 10 * n)), True,
                                        int(rng.integers(0, 200)), int(rng.integers(1, 3))))
    t0 = time.time()
    pts, fields, _ = imp._convert_annotations(rec, b, {}, imp.ImportConfig(), imp._Warn())
    assert time.time() - t0 < 20
    assert len(fields) == n * n
    # reference: naive recomputation of per-cell page counts and counters
    from collections import defaultdict
    ok = [p for p in pts if p.status.value == "ok"]
    cc = defaultdict(Counter)
    for p in ok:
        if p.counter is not None and p.field_id is not None:
            cc[p.counter][p.field_id] += 1
    maj = {c: v.most_common(1)[0][0] for c, v in cc.items()}
    for f in fields:
        cnt = Counter("unspecified" if p.page is None else str(p.page) for p in ok if p.field_id == f.field_id)
        assert f.n_points_by_page == dict(sorted(cnt.items()))
        assert f.counters == sorted(c for c, v in maj.items() if v == f.field_id)
        assert f.status == (FieldStatus.has_points if cnt else FieldStatus.no_points)


# ================================================================== opt-in end-to-end
def _spotiflow_importable():
    try:
        import spotiflow.model  # noqa: F401
        import pytorch_lightning  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _spotiflow_importable(), reason="spotiflow / pytorch-lightning are not installed")
def test_e2e_tiny_finetune_on_an_imported_snapshot(tmp_path, capsys):
    from polony_app.ml import evaluate as ev
    from polony_app.ml.train import TrainOptions, train_candidate
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS", "PS", "QS"))
    cm = write_channel_map(tmp_path / "c.json", defaults={0: "red", 1: ("green", False)})
    im = write_identity_map(tmp_path / "id.json", files)
    d = tmp_path / "data"
    out = lb.import_legacy(d, root, opts(fractions={"train": 0.6, "val": 0.2, "test": 0.2}), channel_map=cm, identity_map=im,
                           say=Recorder())
    ds.lock_test_set(d, out.result.version)
    from spotiflow.model import Spotiflow, SpotiflowModelConfig
    base = tmp_path / "tiny_base"
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(base, which="best")
    e = train_candidate(d, out.result.version, TrainOptions(base=str(base), epochs=2, batch_size=2, crop_size=32, seed=0,
                                                            device="cpu", allow_small=True), say=print)
    assert e["status"] == "candidate" and e["metrics"]["training"]["best_val_loss"] is not None
    pred = ev.make_spotiflow_predictor(e["id"], d, device="cpu")
    res = ev.evaluate_on_snapshot(d, out.result.version, pred, split="test")
    assert res["n_crops"] > 0 and "f1" in res["metrics"]
