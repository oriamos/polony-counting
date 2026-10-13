"""``ml preflight``: a read-only audit of the historical TIFF directory. Fixtures that must PASS, fixtures that must WARN / FAIL
(one injected defect each), the independent cross-check against a deliberately sabotaged bridge, no write into --input, the JSON
schema and the exit codes. Synthetic data only."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from legacy_bridge_fixtures import (GRID_XS, GRID_YS, RED_GREEN, make_tiff, marks_in, standard_corpus, write_channel_map,
                                    write_examined, write_identity_map)
from legacy_fixtures import point_roi, write_stack, textured
from polony_app.ml import legacy_bridge as lb
from polony_app.ml import preflight as pf
from polony_app.ml.cli import main
from polony_app.legacy_import import importer as imp


def tree_hash(root):
    root = Path(root)
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir", p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*"))}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    c = standard_corpus(root)
    cm = write_channel_map(tmp_path / "channels.json", defaults=RED_GREEN)
    im = write_identity_map(tmp_path / "ids.json", c)
    return {"root": root, "corpus": c, "cm": cm, "im": im, "data": tmp_path / "data", "tmp": tmp_path}


def run(capsys, c, *extra, maps=True, json_out=True, sample="all", pixels=False):
    argv = ["--data-dir", str(c["data"]), "preflight", "--input", str(c["root"]), "--no-progress", "--sample", sample]
    if maps:
        argv += ["--channel-map", str(c["cm"]), "--identity-map", str(c["im"])]
    if not pixels:
        argv += ["--no-pixels"]
    if json_out:
        argv += ["--json"]
    argv += list(extra)
    code = main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if json_out else out)


def finding(doc, fid):
    hits = [f for f in doc["findings"] if f["id"] == fid]
    assert hits, f"no finding {fid}: {[f['id'] for f in doc['findings']]}"
    return hits[0]


def levels(doc):
    return {f["id"]: f["level"] for f in doc["findings"]}


# ------------------------------------------------------------------------------------------------ PASS fixtures
def test_clean_corpus_passes_and_schema(corpus, capsys):
    before = tree_hash(corpus["root"])
    code, doc = run(capsys, corpus)
    assert tree_hash(corpus["root"]) == before                      # --input never modified
    assert not (corpus["data"] / "datasets").exists()               # no snapshot
    assert code == 0 and doc["exit_code"] == 0
    assert doc["verdict"] in (pf.V_READY, pf.V_WARN)
    assert doc["schema"] == "polony-preflight" and doc["version"] == 1
    for key in ("verdict", "reasons", "counts", "findings", "checks", "cross_check", "policy", "environment", "timing",
                "next_commands", "n_files", "options"):
        assert key in doc
    assert set(doc["checks"]) == set("ABCDE") and all(doc["checks"][k] for k in "ABCDE")
    for f in doc["findings"]:
        assert set(f) == {"id", "check", "title", "level", "summary", "count", "examples", "detail"}
        assert f["level"] in ("PASS", "WARN", "FAIL", "INFO")
    assert levels(doc)["D.cross_check"] == "PASS"
    cc = doc["cross_check"]
    assert cc["files_compared"] == 4 and cc["n_disagreements"] == 0
    # 3 page-0 cells + 1 page-1 cell per file (both pages trained by RED_GREEN)
    assert doc["policy"]["tuples"]["positive_marked_fields"] == 16
    assert doc["policy"]["annotations"]["labels_exported"] == 4 * 11
    assert levels(doc)["B.truncated"] == "PASS" and levels(doc)["C.channel_map"] == "PASS"
    assert any("--dry-run" in cmd for cmd in doc["next_commands"])


def test_text_output_and_report_dir(corpus, capsys):
    out_dir = corpus["tmp"] / "report"
    code, text = run(capsys, corpus, "--report-dir", str(out_dir), json_out=False)
    assert code == 0 and "VERDICT:" in text and "D  INDEPENDENT CROSS-CHECK" in text
    assert json.loads((out_dir / "preflight-report.json").read_text())["schema"] == "polony-preflight"
    assert "VERDICT" in (out_dir / "preflight-report.txt").read_text()


def test_default_sample_is_deterministic(corpus, capsys):
    _c1, d1 = run(capsys, corpus, sample="2")
    _c2, d2 = run(capsys, corpus, sample="2")
    assert d1["cross_check"]["sample"] == d2["cross_check"]["sample"]
    assert 2 <= d1["cross_check"]["sampled_files"] <= 4


# ------------------------------------------------------------------------------------------------ exit codes
def test_exit_code_2_for_refusals(corpus, capsys):
    assert main(["--data-dir", str(corpus["data"]), "preflight", "--input", str(corpus["tmp"] / "nope")]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert main(["--data-dir", str(corpus["data"]), "preflight", "--input", str(corpus["root"]), "--report-dir",
                 str(corpus["root"] / "rep")]) == 2
    assert not (corpus["root"] / "rep").exists()
    assert main(["--data-dir", str(corpus["data"]), "preflight"]) == 2


def test_exit_code_1_without_maps(corpus, capsys):
    code, doc = run(capsys, corpus, maps=False)
    assert code == 1 and doc["verdict"] == pf.V_NOT
    assert levels(doc)["C.channel_map"] == "FAIL"
    assert any("write-templates" in c for c in doc["next_commands"])


# ------------------------------------------------------------------------------------------------ injected defects (B)
def test_truncated_file_fails(corpus, capsys):
    f = corpus["root"] / corpus["corpus"]["AM-A"]
    b = f.read_bytes()
    f.write_bytes(b[:len(b) // 2])
    code, doc = run(capsys, corpus)
    assert code == 1
    fnd = finding(doc, "B.truncated")
    assert fnd["level"] == "FAIL" and fnd["count"] == 1 and "AM-A" in fnd["examples"][0]


def test_unreadable_and_zero_byte_fail(corpus, capsys):
    (corpus["root"] / "garbage.tif").write_bytes(b"this is not a tiff at all" * 20)
    (corpus["root"] / "empty.tif").write_bytes(b"")
    code, doc = run(capsys, corpus)
    assert code == 1
    assert finding(doc, "B.unreadable")["level"] == "FAIL" and finding(doc, "B.zero_byte")["level"] == "FAIL"


def test_shifted_grid_origin_cluster_warns(corpus, capsys):
    marks = marks_in(0, 0, 3, 0, 1, 5)
    xs = [v - 3 for v in GRID_XS]
    make_tiff(corpus["root"] / "shifted.tif", marks=marks, grid_xs=xs, grid_ys=GRID_YS, seed=9)
    code, doc = run(capsys, corpus)
    f = finding(doc, "B.grid_origin")
    assert f["level"] == "WARN" and any("shifted.tif" in e for e in f["examples"])
    assert len(f["detail"]["origins"]) == 2
    assert "shifted.tif" in f["detail"]["off_modal_files"]
    assert doc["cross_check"]["n_disagreements"] == 0                # the bridge uses the file's own overlay: still agrees


def test_counter_anomaly_and_straddle_warn(corpus, capsys):
    marks = marks_in(0, 0, 2, 5, 1, 1) + marks_in(0, 1, 2, 2, 1, 2) + marks_in(1, 1, 2, 0, 1, 3) + marks_in(1, 1, 1, 5, 1, 4)
    make_tiff(corpus["root"] / "anomaly.tif", marks=marks, seed=7)
    code, doc = run(capsys, corpus)
    assert finding(doc, "B.counters_order")["level"] == "WARN"
    assert finding(doc, "B.counters_span")["level"] == "WARN"
    assert finding(doc, "B.counters_absent")["level"] == "INFO"
    assert doc["cross_check"]["n_disagreements"] == 0


def test_points_outside_image_grid_duplicates_positions(corpus, capsys):
    marks = marks_in(0, 0, 3, 0, 1, 1) + [(200.0, 50.0, 0, 1), (5.0, 5.0, 0, 1), (35.2, 27.2, 0, 1), (35.2, 27.2, 0, 1)]
    marks += [(30.0, 30.0, 0, 4), (31.0, 30.0, 0, 9)]
    make_tiff(corpus["root"] / "points.tif", marks=marks, seed=8)
    code, doc = run(capsys, corpus)
    for fid in ("B.points_outside_image", "B.points_outside_grid", "B.points_duplicate", "B.positions"):
        assert finding(doc, fid)["level"] == "WARN", fid
    # F2: the marks with position 0 / 9 (beyond the 2 planes) put every tuple of their cell into CONFLICT: the independent oracle
    # agrees with the bridge on that, including the drop reason of the position-9 mark (plane_out_of_range)
    assert doc["cross_check"]["disagreements"] == []
    assert doc["policy"]["tuples"]["conflict_fields_excluded"] >= 2


def test_byte_identical_duplicates_warn(corpus, capsys):
    src = corpus["root"] / corpus["corpus"]["AM-A"]
    (corpus["root"] / "copies").mkdir()
    shutil.copy(src, corpus["root"] / "copies" / "AM-A-copy.tif")
    code, doc = run(capsys, corpus)
    f = finding(doc, "B.duplicates")
    assert f["level"] == "WARN" and f["count"] == 1 and "keep" in f["examples"][0]
    assert doc["n_files"]["aliases"] == 1


def test_unreadable_roi_blob_fails(corpus, capsys):
    write_stack(corpus["root"] / "badroi.tif", [textured(120, 160, 3), textured(120, 160, 4)], roi=b"JUNKJUNKJUNKJUNK" * 4)
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "B.roi_parse")["level"] == "FAIL"


def test_names_symlinks_dotdirs_archives(corpus, capsys):
    root = corpus["root"]
    (root / ".hidden").mkdir()
    shutil.copy(root / corpus["corpus"]["AM-A"], root / ".hidden" / "x.tif")
    (root / "a.zip").write_bytes(b"PK")
    shutil.copy(root / corpus["corpus"]["AM-A"], root / "sp ace é.tif")
    outside = corpus["tmp"] / "outside.tif"
    shutil.copy(root / corpus["corpus"]["AS-A"], outside)
    (root / "link.tif").symlink_to(outside)
    (corpus["tmp"] / "elsewhere").mkdir()
    (root / "linkdir").symlink_to(corpus["tmp"] / "elsewhere", target_is_directory=True)
    (root / "Case.tif").write_bytes((root / corpus["corpus"]["AS-B"]).read_bytes())
    (root / "case.TIF").write_bytes((root / corpus["corpus"]["AS-A"]).read_bytes())
    before = tree_hash(root)
    code, doc = run(capsys, corpus)
    assert tree_hash(root) == before
    lv = levels(doc)
    assert lv["B.dot_entries"] == "WARN" and lv["B.archives"] == "WARN" and lv["B.symlink_dirs"] == "WARN"
    assert lv["B.symlink_files"] == "WARN" and "outside" in finding(doc, "B.symlink_files")["summary"]
    assert lv["B.names_spaces"] == "INFO" and lv["B.names_unicode"] == "INFO"
    assert lv["B.names_case"] == "WARN"


def test_non_tiff_scanner_clusters_and_size_outlier(corpus, capsys):
    make_tiff(corpus["root"] / "big.tif", marks=marks_in(0, 0, 3, 0, 1, 1), h=130, w=170, seed=3, info="ImageDescription: 488 [W2]\n")
    code, doc = run(capsys, corpus)
    assert finding(doc, "B.size")["level"] == "WARN" and "big.tif" in finding(doc, "B.size")["examples"][0]


# ------------------------------------------------------------------------------------------------ mapping files (C)
def test_wrong_channel_map_page_fails(corpus, capsys):
    write_channel_map(corpus["cm"], defaults={0: "red", 5: "green"})
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.map_pages")["level"] == "FAIL"


def test_unconfirmed_channel_map_fails(corpus, capsys):
    write_channel_map(corpus["cm"], defaults=RED_GREEN, confirmed_by="")
    code, doc = run(capsys, corpus)
    f = finding(doc, "C.channel_map")
    assert code == 1 and f["level"] == "FAIL" and "UNCONFIRMED" in f["title"]


def test_missing_mapping_stale_and_dead_rules(corpus, capsys):
    write_channel_map(corpus["cm"], rules=[{"id": "only-am", "match": {"glob": "*AM-*.tif"}, "pages": {0: "red"}},
                                           {"id": "dead", "match": {"glob": "nothing*.tif"}, "pages": {0: "red"}}],
                      files={"no/such/file.tif": {0: "red"}})
    code, doc = run(capsys, corpus)
    assert code == 1
    assert finding(doc, "C.map_unmapped")["count"] == 2           # the two AS files
    assert finding(doc, "C.map_stale")["level"] == "FAIL"
    assert finding(doc, "C.map_dead_rules")["level"] == "FAIL"


def test_conflicting_rules_fail(corpus, capsys):
    write_channel_map(corpus["cm"], rules=[{"id": "a", "match": {"glob": "*AM-A*.tif"}, "pages": {0: "red"}},
                                           {"id": "b", "match": {"glob": "*T7_AB*"}, "pages": {0: "green"}}])
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.map_conflicts")["level"] == "FAIL"


def test_identity_problems(corpus, capsys):
    rows = json.loads(corpus["im"].read_text())["rows"]
    rows[0]["confirmed"] = False
    rows.append({"file": "ghost.tif", "specimen_id": "Z", "slide_id": "Z_A", "confirmed": True})
    rows = rows[:-2] + [rows[-2]] + [rows[-1]]
    corpus["im"].write_text(json.dumps({"rows": rows[:3] + [rows[-1]]}))       # one real file left out
    code, doc = run(capsys, corpus)
    lv = levels(doc)
    assert lv["C.identity_unconfirmed"] == "WARN" and lv["C.identity_missing"] == "WARN" and lv["C.identity_unlisted"] == "WARN"
    assert code == 1 and any(f["id"] == "E.blocker" for f in doc["findings"])        # unconfirmed identity blocks the split
    code2, doc2 = run(capsys, corpus, "--allow-unconfirmed-identity")
    assert code2 == 0 and levels(doc2)["E.identity_unconfirmed"] == "WARN"


def test_bad_manifest_and_unresolved_entries(corpus, capsys):
    bad = corpus["tmp"] / "bad.json"
    bad.write_text("{not json")
    code, doc = run(capsys, corpus, "--examined-fields", str(bad))
    assert code == 1 and finding(doc, "C.examined")["level"] == "FAIL"
    ok = write_examined(corpus["tmp"] / "ex.json", {corpus["corpus"]["AM-A"]: {0: [{"counter": 7, "count": 0}, {"field": 3, "count": 0}]},
                                                    "ghost.tif": {0: [{"counter": 1}]}})
    code, doc = run(capsys, corpus, "--examined-fields", str(ok))
    assert code == 1
    assert finding(doc, "C.examined")["level"] == "FAIL"            # ghost.tif is not among the TIFFs
    assert finding(doc, "E.unresolved")["level"] == "FAIL"           # counter 7 absent; field N needs the acknowledgement
    assert doc["cross_check"]["n_disagreements"] == 0


def test_verified_zero_manifest_cross_check(corpus, capsys):
    ex = write_examined(corpus["tmp"] / "ex.json", {corpus["corpus"]["AM-A"]: {0: [{"row": 2, "col": 2, "count": 0}, {"row": 0, "col": 0, "count": 4},
                                                                                  {"row": 0, "col": 1, "count": 9}]}})
    code, doc = run(capsys, corpus, "--examined-fields", str(ex), "--mode", "examined-fields-manifest")
    assert doc["cross_check"]["n_disagreements"] == 0, doc["cross_check"]["disagreements"]
    st = doc["policy"]["tuples"]
    assert st["verified_zero_fields"] == 1 and st["conflict_fields_excluded"] == 1


# ------------------------------------------------------------------------------------------------ the cross-check catches a wrong bridge
def _sabotage(monkeypatch, fn):
    real = lb.build_plan

    def wrapped(*a, **k):
        plan = real(*a, **k)
        fn(plan)
        return plan
    monkeypatch.setattr(lb, "build_plan", wrapped)


def _first_plan(plan):
    return next(p for p in plan.plans if p.scan.n_points)


def _expect_fail(corpus, capsys, kinds):
    code, doc = run(capsys, corpus)
    assert code == 1
    f = finding(doc, "D.cross_check")
    assert f["level"] == "FAIL", f
    got = {d["kind"] for d in doc["cross_check"]["disagreements"]}
    assert got & set(kinds), got
    assert all(d["file"] and d["key"] is not None for d in doc["cross_check"]["disagreements"])


def test_sabotage_move_mark_to_other_cell(corpus, capsys, monkeypatch):
    def fn(plan):
        sc = _first_plan(plan).scan
        sc.cell[0] = (sc.cell[0] + 1) % sc.n_cells
    _sabotage(monkeypatch, fn)
    _expect_fail(corpus, capsys, ["point"])


def test_sabotage_flip_page(corpus, capsys, monkeypatch):
    def fn(plan):
        sc = _first_plan(plan).scan
        sc.page[0] = 1 - sc.page[0]
    _sabotage(monkeypatch, fn)
    _expect_fail(corpus, capsys, ["point"])


def test_sabotage_shift_coordinates_one_pixel(corpus, capsys, monkeypatch):
    def fn(plan):
        sc = _first_plan(plan).scan
        sc.x = sc.x + 1.0
    _sabotage(monkeypatch, fn)
    _expect_fail(corpus, capsys, ["point", "tuple_labels"])


def test_sabotage_drop_a_mark_from_a_tuple_and_crop(corpus, capsys, monkeypatch):
    def fn(plan):
        pl = _first_plan(plan)
        key = next(k for k, t in pl.tuples.items() if t.labels)
        pl.tuples[key].labels.pop()
        c = pl.ext.cands[0]
        c.points = c.points[:-1]
    _sabotage(monkeypatch, fn)
    _expect_fail(corpus, capsys, ["tuple_labels", "crop_points"])


def test_sabotage_status_flip_and_dropped_reason(corpus, capsys, monkeypatch):
    def fn(plan):
        pl = _first_plan(plan)
        key = next(k for k, t in pl.tuples.items() if t.exam.status == "unknown")
        from polony_app.fields import examined as exm
        pl.tuples[key].exam = exm.Examination(exm.POSITIVE, 1, (exm.SRC_EMBEDDED,), "x")
        pl.ext.rows[0]["status"] = "dropped"
        pl.ext.rows[0]["basis"] = "dropped:duplicate"
    _sabotage(monkeypatch, fn)
    _expect_fail(corpus, capsys, ["tuple_status", "point_row"])


def test_sabotage_at_the_source_shifts_every_label(corpus, capsys, monkeypatch):
    real = imp._convert_annotations

    def shifted(rec, bundle, src_files, cfg, warn):
        pts, fields, regions = real(rec, bundle, src_files, cfg, warn)
        for p in pts[:1]:
            p.x = p.x + 1.0
        return pts, fields, regions
    monkeypatch.setattr(imp, "_convert_annotations", shifted)
    _expect_fail(corpus, capsys, ["point", "tuple_labels", "crop_points"])


def test_bridge_crash_is_a_fail(corpus, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(lb, "build_plan", boom)
    code, doc = run(capsys, corpus)
    assert code == 1 and "CRASHED" in finding(doc, "D.plan")["summary"]


def test_dangling_symlink_is_reported(corpus, capsys):
    (corpus["root"] / "dangling.tif").symlink_to(corpus["tmp"] / "missing.tif")
    code, doc = run(capsys, corpus)
    assert code == 0, doc["reasons"]
    assert finding(doc, "B.symlink_files")["detail"]["skipped"] == ["dangling.tif"]


# ------------------------------------------------------------------------------------------------ units
def test_independent_derivation_units():
    rng = np.random.default_rng(0)
    pts = {"x": np.array([35.5, 10.5, 60.5]), "y": np.array([27.5, 10.5, 27.5]), "sub": np.array([True, True, True]),
           "counter": np.array([0, 0, 1], np.int32), "position": np.array([1, 1, 3], np.int32), "blob": np.zeros(3, np.int32)}
    t = pf.point_table(pts, 2, 160, 120, ([10.0, 60.0, 110.0, 160.0], [10.0, 45.0, 80.0, 115.0]))
    assert list(t["cell"]) == [0, 0, 1]           # canonical 35 / 10 / 60: cx=60 is the left edge of cell 1 (x0-0.5=59.5<=60)
    assert list(t["page"]) == [0, 0, -1]          # position 3 is beyond the 2 planes
    cs = pf.counter_stats(np.array([0, 0, 2, 1, 1, 0]))
    assert cs["absent"] == [] and cs["out_of_order"] == [1] and cs["split_runs"] == [0]
    assert pf.pick_sample(["b", "a", "c"], "all", set()) == ["a", "b", "c"]
    assert pf.pick_sample(["b", "a", "c"], "1", {"c"}) in (["a", "c"], ["b", "c"], ["c"])
    inf = pf.parse_info("ImageDescription: 635 [W1]\nDateTime: 2026/09/30 17:10:32\nHostComputer: AxImageIO: PMT=550; LaserPower=1.97;")
    assert inf["label"] == "635 [W1]" and inf["pmt"] == "550" and inf["laser"] == "1.97" and inf["datetime"].startswith("2026-09-30")


def test_bright_check_finds_peaks(tmp_path):
    img = np.full((200, 200), 100, np.uint16)
    pts = [(50, 50), (120, 80), (30, 150), (160, 160), (90, 20), (170, 40), (60, 100), (20, 20)]
    for x, y in pts:
        img[y - 1:y + 2, x - 1:x + 2] = 5000
    import tifffile
    tifffile.imwrite(str(tmp_path / "p.tif"), img)
    r = pf.bright_check(tmp_path / "p.tif", 0, np.array(pts, float), np.random.default_rng(0), (0, 0, 200, 200))
    assert r and r["median_mark_px"] < 2 and r["median_random_px"] > r["median_mark_px"]


def test_unplaced_mark_makes_cell_conflict_independently(corpus, capsys):
    marks = marks_in(0, 0, 3, 0, 1, 1) + [(30.0, 30.0, 0, 0)] + marks_in(1, 1, 2, 1, 1, 2)
    make_tiff(corpus["root"] / "unplaced.tif", marks=marks, seed=12)
    code, doc = run(capsys, corpus)
    assert doc["cross_check"]["n_disagreements"] == 0, doc["cross_check"]["disagreements"]
    assert doc["policy"]["tuples"]["conflict_fields_excluded"] == 2     # cell (0,0) on both trained pages


def test_footprint_rule_on_fractional_grid(corpus, capsys):
    xs = [10.3, 60.6, 110.9, 160.0]
    ys = [10.3, 45.4, 80.5, 115.0]
    marks = [(60.2, 30.0, 0, 1), (60.7, 30.0, 0, 1), (59.9, 46.0, 1, 1), (45.0, 45.9, 1, 1), (45.0, 46.0, 1, 1)]
    make_tiff(corpus["root"] / "frac.tif", marks=marks, grid_xs=xs, grid_ys=ys, seed=13)
    code, doc = run(capsys, corpus)
    assert doc["cross_check"]["n_disagreements"] == 0, doc["cross_check"]["disagreements"]


def test_lohi_recomputed_independently(corpus, capsys):
    code, doc = run(capsys, corpus, pixels=True)
    assert doc["cross_check"]["n_disagreements"] == 0, doc["cross_check"]["disagreements"]
    f = finding(doc, "D.lohi")
    assert f["level"] in ("PASS", "WARN")                 # synthetic noise has no chamber -> documented fallback warning
    assert doc["cross_check"]["lohi"]["rows"]
    for vm in ("grid-cells", "eligible-fields"):
        code, doc = run(capsys, corpus, "--valid-mask", vm, pixels=True)
        assert doc["cross_check"]["n_disagreements"] == 0, (vm, doc["cross_check"]["disagreements"])
        assert finding(doc, "D.lohi")["level"] == "PASS"


def test_lohi_sabotaged_valid_mask_is_caught(corpus, capsys, monkeypatch):
    real = lb.page_bounds

    def wrong(plane, valid=None, percentiles=(1.0, 99.8)):
        b = real(plane, valid, percentiles)
        b["lo"] += 5.0
        return b
    monkeypatch.setattr(lb, "page_bounds", wrong)
    code, doc = run(capsys, corpus, "--valid-mask", "grid-cells", pixels=True)
    assert code == 1 and any(d["kind"] == "lohi" for d in doc["cross_check"]["disagreements"])


def test_strict_loader_rules_are_fails(corpus, capsys):
    cm = json.loads(corpus["cm"].read_text())
    cm["defaults"]["pages"]["0"].pop("train", None)
    corpus["cm"].write_text(json.dumps(cm))
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.channel_map")["level"] == "FAIL"
    write_channel_map(corpus["cm"], defaults=RED_GREEN)
    cm = json.loads(corpus["cm"].read_text())
    cm["bogus_key"] = 1
    corpus["cm"].write_text(json.dumps(cm))
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.channel_map")["level"] == "FAIL"
    write_channel_map(corpus["cm"], defaults=RED_GREEN, files={"./" + corpus["corpus"]["AM-A"]: {0: "red"}})
    code, doc = run(capsys, corpus)
    assert code == 1 and (finding(doc, "C.map_stale")["level"] == "FAIL" or finding(doc, "C.channel_map")["level"] == "FAIL")
    write_channel_map(corpus["cm"], defaults=RED_GREEN)
    rows = json.loads(corpus["im"].read_text())["rows"]
    del rows[0]["confirmed"]
    corpus["im"].write_text(json.dumps({"rows": rows}))
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.identity_map")["level"] == "FAIL"
    rows[0]["confirmed"] = True
    rows[0]["specimen_id"] = None
    corpus["im"].write_text(json.dumps({"rows": rows}))
    code, doc = run(capsys, corpus)
    assert code == 1 and finding(doc, "C.identity_map")["level"] == "FAIL"
    code, doc = run(capsys, corpus, "--level", "slide")
    assert finding(doc, "C.identity_map")["level"] == "PASS"


def test_outside_symlinks_are_reported_and_skipped_by_both(corpus, capsys):
    outside = corpus["tmp"] / "outside.tif"
    shutil.copy(corpus["root"] / corpus["corpus"]["AS-A"], outside)
    (corpus["root"] / "link.tif").symlink_to(outside)
    (corpus["root"] / "broken.tif").symlink_to(corpus["tmp"] / "nowhere.tif")
    code, doc = run(capsys, corpus)
    f = finding(doc, "B.symlink_files")
    assert f["level"] == "WARN" and sorted(f["detail"]["skipped"]) == ["broken.tif", "link.tif"]
    assert doc["cross_check"]["n_disagreements"] == 0 and code == 0


def test_next_commands_mention_idempotent_reimport(corpus, capsys):
    code, doc = run(capsys, corpus)
    assert any("already imported" in c for c in doc["next_commands"])
