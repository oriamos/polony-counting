"""Adversarial tests of the ``ml import-legacy`` bridge: try to break the scientific guarantees (leakage, silent labels).

Everything is derived from INDEPENDENT oracles written in this file (never from the code under test): the fixtures know which
(file, field, page) tuples the researcher marked / listed, where every mark is, which specimens share a slide, ...
Known bugs found while writing these tests are kept as ``xfail(strict=True)`` tests: when the bug is fixed the marker must go.
"""
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import tifffile

from legacy_bridge_fixtures import (GRID_XS, GRID_YS, RED_GREEN, T0, H, W, make_tiff, marks_in, standard_corpus,
                                    write_channel_map, write_examined, write_identity_map)
from legacy_fixtures import grid_overlay, point_roi, write_stack
from polony_app.ml import common as cm
from polony_app.ml import dataset as ds
from polony_app.ml import legacy_bridge as lb
from polony_app.ml.cli import main
from polony_app.ml.common import MLError
from test_ml_support import Recorder

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

FR = {"train": 0.5, "val": 0.5, "test": 0.0}
FR3 = {"train": 0.6, "val": 0.2, "test": 0.2}


# ============================================================================================ helpers
def opts(**kw):
    kw.setdefault("created_at", T0)
    kw.setdefault("fractions", FR)
    return lb.BridgeOptions(**kw)


def run_import(tmp, root, *, data=None, cmap=None, imap=None, examined=None, dry=False, resume=True, **kw):
    tmp = Path(tmp)
    cmap = cmap or write_channel_map(tmp / "channels.json", defaults=RED_GREEN)
    data = data or (tmp / "data")
    return lb.import_legacy(data, root, opts(**kw), channel_map=cmap, identity_map=imap, examined_fields=examined, dry_run=dry,
                            resume=resume, say=Recorder())


def tree(p):
    """Everything below ``p`` (dirs and files, with sizes): to prove a command wrote nothing."""
    p = Path(p)
    return sorted((str(q.relative_to(p)), q.stat().st_size if q.is_file() else -1) for q in p.rglob("*")) if p.exists() else None


def digest(data, version):
    root = Path(data) / "datasets" / version
    return {str(q.relative_to(root)): q.read_bytes() for q in sorted(root.rglob("*")) if q.is_file()}


def unlock(path):
    """Snapshots are read-only on disk: make them writable (tests that tamper on purpose)."""
    for q in [Path(path), *Path(path).rglob("*")]:
        os.chmod(q, 0o755 if q.is_dir() else 0o644)


def cell_of(r, c):
    return r * 3 + c


def spec_marks(marks: dict, seed: int) -> list:
    """{(page, cell): n} -> ImageJ marks with counter == cell index (one counter per cell, all points inside their own cell)."""
    out = []
    for (page, cell), n in sorted(marks.items()):
        if n:
            r, c = divmod(cell, 3)
            out += marks_in(r, c, n, cell, page + 1, seed + 5 * page + cell)
    return out


def entry_dict(cell, count):
    r, c = divmod(cell, 3)
    d = {"row": r, "col": c}
    if count is not None:
        d["count"] = count
    return d


def oracle_status(mode, n, listed, stated):
    """Independent re-statement of the examination rule for one tuple. ``listed``: has a manifest entry, ``stated``: its count."""
    if mode == "marked-only":
        if n == 0:
            return "unknown"
        return "conflict" if stated is not None and stated != n else "positive_marked"
    if not listed:
        return "unknown"
    if n > 0:
        return "conflict" if stated is not None and stated != n else "positive_marked"
    if stated == 0:
        return "verified_zero"
    return "unknown" if stated is None else "conflict"


def random_eligibility_case(rng, n_files):
    """[{'marks': {(page, cell): n}, 'listed': {(page, cell): count|None}}]"""
    files = []
    for _ in range(n_files):
        marks, listed = {}, {}
        for page in (0, 1):
            for cell in range(9):
                if rng.random() < (0.45 if page == 0 else 0.2):
                    marks[(page, cell)] = int(rng.integers(1, 5))
                if rng.random() < 0.35:
                    n = marks.get((page, cell), 0)
                    kind = rng.choice(["none", "zero", "match", "other"])
                    listed[(page, cell)] = {"none": None, "zero": 0, "match": n, "other": n + int(rng.integers(1, 3))}[kind]
        files.append({"marks": marks, "listed": listed})
    return files


def write_case(root, tmp, case, seed=0):
    """Files named so that file-name inference gives each its own specimen; returns {rel: file spec} and the manifest path."""
    out, man = {}, {}
    for i, f in enumerate(case):
        rel = f"29-09-2026_T7_AB_{'ABCDEFGH'[i]}{'ABCDEFGH'[i]}-A.tif"
        make_tiff(root / rel, marks=spec_marks(f["marks"], seed + i), seed=seed + 10 * i + 1)
        out[rel] = f
        pages = defaultdict(list)
        for (page, cell), cnt in sorted(f["listed"].items()):
            pages[page].append(entry_dict(cell, cnt))
        if pages:
            man[rel] = dict(pages)
    path = write_examined(Path(tmp) / "man.json", man) if man else None
    return out, path


def expected_tally(files, mode, accepted=None):
    tally = Counter()
    per_tuple = {}
    for rel, f in files.items():
        if accepted is not None and rel not in accepted:
            continue
        for page in (0, 1):
            for cell in range(9):
                st_ = oracle_status(mode, f["marks"].get((page, cell), 0), (page, cell) in f["listed"], f["listed"].get((page, cell)))
                tally[st_] += 1
                per_tuple[(rel, page, cell)] = st_
    return tally, per_tuple


# ============================================================================================ 1. coordinates
@settings(max_examples=300, deadline=None, derandomize=True)
@given(d=st.sampled_from([1, 2, 4, 8]), h=st.integers(8, 70), w=st.integers(8, 70), ox=st.integers(0, 60), oy=st.integers(0, 60),
       pts=st.lists(st.tuples(st.floats(-3, 75, allow_nan=False), st.floats(-3, 75, allow_nan=False)), max_size=25))
def test_prepare_crop_round_trip_property(d, h, w, ox, oy, pts):
    """Any block grid alignment / trimmed remainder / edge point: kept labels map back EXACTLY, dropped ones are exactly those
    whose pixel footprint lies in a trimmed row/column or outside the crop, and nothing is clamped."""
    if min(h, w) < 2 * d + 7:
        h, w = max(h, 2 * d + 7), max(w, 2 * d + 7)
    rng = np.random.default_rng(h * 1000 + w)
    img = rng.integers(0, 60000, (h, w)).astype(np.uint16)
    yx = np.array([(y, x) for x, y in pts], float).reshape(-1, 2)
    x, lab, dropped, (oy_, ox_) = cm.prepare_crop(img, yx, lo=100.0, hi=50000.0, downscale=d, origin_xy=(ox, oy))
    # the origin of the trimmed crop is on the page-wide block grid (page columns / rows that are multiples of d)
    assert (ox + ox_) % d == 0 and (oy + oy_) % d == 0 and 0 <= ox_ < d and 0 <= oy_ < d
    hs, ws = (h - oy_) // d, (w - ox_) // d
    assert x.shape == (hs, ws) and x.dtype == np.float32
    # oracle: a label is kept iff its pixel centre lies inside the (offset, trimmed) block grid footprint
    keep = [(oy_ - 0.5 <= y < oy_ + d * hs - 0.5) and (ox_ - 0.5 <= xx < ox_ + d * ws - 0.5) for y, xx in yx]
    assert len(lab) == sum(keep) and dropped == len(keep) - sum(keep)
    back = cm.from_downscaled(lab, d) + np.array([oy_, ox_], float)
    assert np.allclose(back, yx[np.array(keep, bool)], atol=1e-4)     # labels are float32
    if len(lab):
        assert (lab[:, 0] >= -0.5).all() and (lab[:, 0] < hs - 0.5).all() and (lab[:, 1] >= -0.5).all() and (lab[:, 1] < ws - 0.5).all()


@settings(max_examples=120, deadline=None, derandomize=True)
@given(d=st.sampled_from([2, 4, 8]), i=st.integers(0, 6), j=st.integers(0, 6))
def test_block_centre_is_the_mean_native_coordinate_of_the_block(d, i, j):
    """Pixel-centre rule: output pixel (i, j) sits at the mean of the native pixel-centre coordinates of its d x d block."""
    ys = np.arange(i * d, i * d + d, dtype=float)
    xs = np.arange(j * d, j * d + d, dtype=float)
    centre = np.array([ys.mean(), xs.mean()])
    assert np.allclose(cm.from_downscaled([i, j], d), centre) and np.allclose(cm.to_downscaled(centre, d), [i, j])


def test_a_bright_spot_stays_under_its_label_after_downscaling():
    """The label of a spot at an arbitrary native sub-pixel position, in a crop at an arbitrary origin, lands on the spot in the
    downscaled network input (centre of mass check) for every factor."""
    rng = np.random.default_rng(3)
    for d in (1, 2, 4, 8):
        for _ in range(25):
            h = w = 80
            ox, oy = int(rng.integers(0, 40)), int(rng.integers(0, 40))
            cy, cx = float(rng.uniform(24, 56)), float(rng.uniform(24, 56))
            yy, xx = np.mgrid[0:h, 0:w]
            img = (1000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2))).astype(np.float32)
            x, lab, dropped, off = cm.prepare_crop(img, [[cy, cx]], lo=0.0, hi=1000.0, downscale=d, origin_xy=(ox, oy))
            assert dropped == 0 and len(lab) == 1
            yy2, xx2 = np.mgrid[0:x.shape[0], 0:x.shape[1]]
            m = x.astype(float)
            com = np.array([(m * yy2).sum() / m.sum(), (m * xx2).sum() / m.sum()])
            assert np.abs(com - lab[0]).max() < 0.05 + 0.02 * d, (d, com, lab)


def _spots_corpus(root, rng, n_files=4):
    """Files whose marks sit on cell borders / image edges / in the remainder rows (crop sizes 50 x 35 are not multiples of 8)."""
    expected = {}
    for k in range(n_files):
        marks = []
        for r in range(3):
            for c in range(3):
                if rng.random() < 0.9:
                    xs = [GRID_XS[c] + 0.5 + 0.05, GRID_XS[c + 1] - 0.5 - 0.05, GRID_XS[c] + 0.5 + rng.uniform(1, 48)]
                    ys = [GRID_YS[r] + 0.5 + 0.05, GRID_YS[r + 1] - 0.5 - 0.05, GRID_YS[r] + 0.5 + rng.uniform(1, 33)]
                    for q in range(3):
                        marks.append((round(float(xs[q]), 3), round(float(ys[(q + 1) % 3]), 3), cell_of(r, c), int(rng.integers(1, 3))))
        marks = sorted({m for m in marks})
        # drop near-duplicates (the importer flags points within 1 px)
        keep = []
        for m in marks:
            if all((m[0] - k_[0]) ** 2 + (m[1] - k_[1]) ** 2 > 4 or m[3] != k_[3] for k_ in keep):
                keep.append(m)
        make_tiff(root / f"29-09-2026_T7_AB_{'XYZW'[k]}S-A.tif", marks=keep, seed=40 + k)
        expected[f"29-09-2026_T7_AB_{'XYZW'[k]}S-A.tif"] = keep
    return expected


@pytest.mark.parametrize("d", [1, 2, 4, 8])
def test_end_to_end_labels_round_trip_through_the_snapshot_and_the_trainer_frame(tmp_path, d):
    """ImageJ mark -> exported CSV label -> prepare_crop (trainer frame) -> mapped back == the original mark (canonical = imagej -
    0.5); labels lost to trimmed remainders / outside the footprint are COUNTED, none is invented, none is moved."""
    rng = np.random.default_rng(100 + d)
    root = tmp_path / "in"
    expected = _spots_corpus(root, rng)
    out = run_import(tmp_path, root, downscale=d, allow_unconfirmed_identity=True, fractions={"train": 1.0, "val": 0.0, "test": 0.0})
    data, ver = tmp_path / "data", out.result.version
    kept_total = dropped_total = 0
    for c in out.result.manifest.crops:
        img, lab = ds.load_crop(data, ver, c)
        x0, y0, x1, y1 = c.box_xyxy
        r, cc = c.provenance["grid_row"], c.provenance["grid_col"]
        want = sorted((y - 0.5 - y0, x - 0.5 - x0) for x, y, ctr, pos in expected[c.source_file]
                      if pos == c.page + 1 and ctr == cell_of(r, cc))
        got = sorted(map(tuple, lab.tolist()))
        # (CSV) the exported labels are exactly the marks of that cell+page (no mark moved, no mark invented, none clamped)
        assert len(got) == len(want) == c.n_positive and c.n_edge_dropped == 0
        for g in got:
            assert any(abs(g[0] - w_[0]) < 1e-3 and abs(g[1] - w_[1]) < 1e-3 for w_ in want)
        x, lab_s, dropped, off = cm.prepare_crop(img, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=d, normaliser=c.normaliser,
                                                 origin_xy=(x0, y0))
        back = cm.from_downscaled(lab_s, d) + np.array(off, float)
        assert len(lab_s) + dropped == len(got)
        # kept labels map back to exactly one original label, within float eps
        for b in back:
            assert min(np.abs(np.array(got) - b).max(axis=1)) < 1e-4
        # dropped ones lie in the trimmed top/left offset or the trimmed bottom/right remainder
        hs, ws = x.shape
        inside = [off[0] - 0.5 <= g[0] < off[0] + d * hs - 0.5 and off[1] - 0.5 <= g[1] < off[1] + d * ws - 0.5 for g in got]
        assert sum(inside) == len(lab_s) and dropped == len(got) - sum(inside)
        kept_total += len(lab_s)
        dropped_total += dropped
        # block alignment: the trimmed crop starts on the page-wide block grid
        assert (x0 + off[1]) % d == 0 and (y0 + off[0]) % d == 0
    assert kept_total > 10
    if d == 1:
        assert dropped_total == 0
    else:
        assert dropped_total >= 0 and kept_total + dropped_total == sum(c.n_positive for c in out.result.manifest.crops)


# ============================================================================================ 2. eligibility
def test_random_eligibility_end_to_end_marked_only_and_manifest(tmp_path):
    """Random marks / manifests: exactly the provable tuples have crops; the status tallies add up; every crop is provenanced."""
    rng = np.random.default_rng(2024)
    n_runs = 0
    for it in range(8):
        work = tmp_path / f"w{it}"
        root = work / "in"
        root.mkdir(parents=True)
        case = random_eligibility_case(rng, int(rng.integers(2, 4)))
        files, man = write_case(root, work, case, seed=7 * it)
        mode = "marked-only" if it % 2 == 0 else "examined-fields-manifest"
        if mode == "examined-fields-manifest" and man is None:
            man = write_examined(work / "man.json", {})
        # a file with no mark at all is rejected in marked-only mode (no ROI): it contributes no tuple
        accepted = {rel for rel, f in files.items() if f["marks"] or mode != "marked-only"}
        if not accepted:
            continue
        out = run_import(work, root, examined=man, mode=mode, allow_unconfirmed_identity=True, dry=True)
        tally, per = expected_tally(files, mode, accepted)
        f = out.plan.summary["fields"]
        assert (f["positive_marked_fields"], f["verified_zero_fields"], f["unknown_fields_excluded"], f["conflict_fields_excluded"]) \
            == (tally["positive_marked"], tally["verified_zero"], tally["unknown"], tally["conflict"])
        assert f["tuples_total"] == 18 * len(accepted) == sum(tally.values())
        assert f["eligible_examined_fields"] + f["excluded_unproven"] == f["tuples_total"]
        # per tuple, from the plan objects
        got = {(p.scan.rel, pg, cell): t.exam.status for p in out.plan.plans for (pg, cell), t in p.tuples.items()}
        assert got == per
        if not (tally["positive_marked"] + tally["verified_zero"]):
            continue
        out = run_import(work, root, examined=man, mode=mode, allow_unconfirmed_identity=True, data=work / "real")
        n_runs += 1
        m = out.result.manifest
        have = {(c.source_file, c.page, c.provenance["grid_row"] * 3 + c.provenance["grid_col"]): c for c in m.crops}
        want = {k: v for k, v in per.items() if v in ("positive_marked", "verified_zero")}
        assert set(have) == set(want), "crops exist for exactly the provable tuples"
        for k, c in have.items():
            assert c.provenance["examination_status"] == want[k] and c.provenance["researcher_examined"] is True
            assert (c.n_positive == 0) == (want[k] == "verified_zero")
            if want[k] == "verified_zero":
                assert (work / "real/datasets" / out.result.version / c.path_csv).read_text() == "y,x\n"
        assert lb.verify_examination_provenance(m) == []
        # nothing outside an examined tuple is a label: confirmed point rows == sum of crop positives
        rows = list(__import__("csv").DictReader(open(work / "real/datasets" / out.result.version / "points.csv", newline="")))
        assert sum(r["status"] == "confirmed" for r in rows) == sum(c.n_positive for c in m.crops)
        for r in rows:
            if r["status"] == "dropped":
                assert r["crop_id"] == ""
    assert n_runs >= 2


def test_marks_on_page_zero_only_never_make_page_one_a_tuple_with_crop_or_label(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    marks = {(0, 0): 3, (0, 4): 2, (0, 8): 4}
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks(marks, 1), seed=2)
    # a manifest listing page 1 fields WITHOUT count, plus a stated count for page 1 that has no marks: still nothing on page 1
    man = write_examined(tmp_path / "m.json", {"29-09-2026_T7_AB_AM-A.tif": {1: [entry_dict(0, None), entry_dict(4, 2)]}})
    for kw in ({}, {"examined": man}):
        out = run_import(tmp_path, root, allow_unconfirmed_identity=True, data=tmp_path / f"d{len(kw)}", **kw)
        assert {c.page for c in out.result.manifest.crops} == {0}
        assert out.plan.summary["annotations"]["labels_exported"] == 9
        rows = list(__import__("csv").DictReader(open(tmp_path / f"d{len(kw)}/datasets" / out.result.version / "points.csv", newline="")))
        assert {r["page"] for r in rows} == {"0"}
    f = out.plan.summary["fields"]
    # marked-only: a manifest can only CONTRADICT marks, it never adds a field (page 1 has no marks -> unknown, never examined)
    assert f["positive_marked_fields"] == 3 and f["conflict_fields_excluded"] == 0 and f["unknown_fields_excluded"] == 18 - 3


def test_grid_overlay_alone_makes_no_crop_in_any_mode(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks({(0, 0): 2}, 1), seed=2)       # 8 other overlay cells, page 1
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True)
    assert [(c.page, c.region_id) for c in out.result.manifest.crops] == [(0, "r0c0")]
    # manifest mode with an empty manifest: nothing at all is eligible -> refused (never "all cells")
    man = write_examined(tmp_path / "m.json", {})
    with pytest.raises(MLError, match="no crop can be exported"):
        run_import(tmp_path, root, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest", data=tmp_path / "d2")


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.lists(st.tuples(st.integers(0, 8), st.integers(0, 1), st.integers(0, 4)), max_size=18, unique_by=lambda t: (t[0], t[1])),
       st.lists(st.tuples(st.integers(0, 8), st.integers(0, 1), st.sampled_from(["none", "zero", "match", "other"])), max_size=18,
                unique_by=lambda t: (t[0], t[1])),
       st.sampled_from(["marked-only", "examined-fields-manifest"]))
def test_hypothesis_status_matches_the_independent_oracle(marks, listed, mode):
    from polony_app.fields import examined as ex
    n = {(pg, cell): k for cell, pg, k in marks if k}
    lst = {}
    for cell, pg, kind in listed:
        k = n.get((pg, cell), 0)
        lst[(pg, cell)] = {"none": None, "zero": 0, "match": k, "other": k + 1}[kind]
    for pg in (0, 1):
        for cell in range(9):
            e = None
            if (pg, cell) in lst:
                e = ex.FieldEntry(row=cell // 3, col=cell % 3, count=lst[(pg, cell)])
            man = ex.ExaminedManifest(evidence="t", files=[])
            got = ex.classify(n.get((pg, cell), 0), e, mode=mode, manifest=man)
            assert got.status == oracle_status(mode, n.get((pg, cell), 0), (pg, cell) in lst, lst.get((pg, cell)))
            assert got.researcher_examined == (got.status in ("positive_marked", "verified_zero"))


def test_hand_removed_provenance_is_detected_and_the_assembler_gate_refuses(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    standard_corpus(root, ("AM",))
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True)
    m = out.result.manifest
    for split in ("train", "val", "test"):
        for c in (c for c in m.crops if c.split == split):
            assert c.provenance["researcher_examined"] is True
    assert lb.verify_examination_provenance(m) == []
    mutations = [("drop flag", lambda p: p.pop("researcher_examined")), ("flag false", lambda p: p.update(researcher_examined=False)),
                 ("flag string", lambda p: p.update(researcher_examined="true")),
                 ("status unknown", lambda p: p.update(examination_status="unknown")),
                 ("status conflict", lambda p: p.update(examination_status="conflict")),
                 ("no source", lambda p: p.pop("examination_source")), ("no sha", lambda p: p.update(source_sha256="")),
                 ("no page", lambda p: p.pop("page")), ("no field id", lambda p: p.pop("field_id")), ("empty", lambda p: p.clear())]
    for name, mutate in mutations:
        for which in (0, len(m.crops) - 1):
            bad = m.model_copy(deep=True)
            mutate(bad.crops[which].provenance)
            assert lb.verify_examination_provenance(bad), name
    # the verified-zero evidence text may not be hand-removed either
    z = m.model_copy(deep=True)
    z.crops[0].provenance.update(examination_status="verified_zero", evidence="")
    assert lb.verify_examination_provenance(z)
    # also through plain dicts (what a JSON reader sees)
    d = m.model_dump(mode="json")
    d["crops"][0]["provenance"] = {k: v for k, v in d["crops"][0]["provenance"].items() if k != "researcher_examined"}
    assert lb.verify_examination_provenance(d)


# ============================================================================================ 3. leakage
def random_identity_corpus(rng, root, tmp, n_spec, confirmed_all=True):
    """Specimens with 1-3 slides, replicates, scan sessions, byte-identical copies under other names. Returns (rows, files)."""
    rows, files, k = [], {}, 0
    for s in range(n_spec):
        sp = f"SP{s}"
        for sl in range(int(rng.integers(1, 4))):
            k += 1
            rel = f"d{k % 3}/scan_{k:02d}.tif"
            make_tiff(root / rel, marks=spec_marks({(0, 0): 3, (0, 1): 2, (1, 0): 2}, k), seed=100 + k)
            row = {"file": rel, "specimen_id": sp, "slide_id": f"{sp}_S{sl}", "assay": "T7", "replicate": "ABC"[sl],
                   "scan_session": f"2026-09-{20 + int(rng.integers(0, 3))}", "confirmed": True}
            rows.append(row)
            files[rel] = row
            if rng.random() < 0.3:             # byte-identical copy under another name, same identity
                alias = f"copies/copy_of_{k:02d}.tif"
                (root / "copies").mkdir(exist_ok=True)
                shutil.copy(root / rel, root / alias)
                rows.append(dict(row, file=alias))
                files[alias] = dict(row, file=alias)
    path = Path(tmp) / "id.json"
    path.write_text(json.dumps({"rows": rows}))
    return path, files


def assert_no_straddle(manifest, ident_of_source, level="auto"):
    """No specimen / slide / group / file bytes in more than one split (specimens are only promised apart at level auto /
    specimen; at level 'slide' a straddling specimen must at least be WARNED about)."""
    splits_by = defaultdict(lambda: defaultdict(set))
    for s in manifest.sources:
        if not s.split:
            continue
        i = ident_of_source(s)
        for key in (("specimen_id", "slide_id") if level in ("auto", "specimen") else ("slide_id",)):
            if i.get(key):
                splits_by[key][i[key]].add(s.split)
        splits_by["group"][s.group_id].add(s.split)
        for sha in s.tiff_sha256.values():
            splits_by["sha"][sha].add(s.split)
    for kind, d in splits_by.items():
        for k, v in d.items():
            assert len(v) == 1, f"{kind} {k} straddles {sorted(v)}"
    by_split_groups = defaultdict(set)
    for c in manifest.crops:
        by_split_groups[c.split].add(c.group_id)
    seen = set()
    for sp, g in by_split_groups.items():
        assert not (seen & g), "a group is in two splits"
        seen |= g
    by_crop_sha = defaultdict(set)
    for c in manifest.crops:
        by_crop_sha[c.source_sha256].add(c.split)
    assert all(len(v) == 1 for v in by_crop_sha.values())


def test_random_identity_maps_never_straddle_splits_and_are_deterministic(tmp_path):
    rng = np.random.default_rng(31)
    for it in range(5):
        work = tmp_path / f"w{it}"
        root = work / "in"
        root.mkdir(parents=True)
        imap, files = random_identity_corpus(rng, root, work, int(rng.integers(3, 6)))
        level = ["auto", "specimen", "auto", "slide", "specimen"][it]
        fr = {"train": 0.5, "val": 0.25, "test": 0.25}
        a = run_import(work, root, imap=imap, fractions=fr, seed=it, level=level, data=work / "a")
        b = run_import(work, root, imap=imap, fractions=fr, seed=it, level=level, data=work / "b", workers=3)
        assert a.result.version == b.result.version and digest(work / "a", a.result.version) == digest(work / "b", b.result.version)
        m = a.result.manifest
        by_rel = {r: v for r, v in files.items()}
        assert_no_straddle(m, lambda s: s.identity, level)
        if level == "slide":
            spl = defaultdict(set)
            for s in m.sources:
                spl[s.identity["specimen_id"]].add(s.split)
            if any(len(v) > 1 for v in spl.values()):
                assert any("straddle" in w for w in m.warnings), "level 'slide' lets a specimen straddle: that must be warned"
        # strict at the specimen level for level auto/specimen: all slides of a specimen together
        if level in ("auto", "specimen"):
            spl = defaultdict(set)
            for s in m.sources:
                if s.split:
                    spl[s.identity["specimen_id"]].add(s.split)
            assert all(len(v) == 1 for v in spl.values())
        # byte-identical copies were collapsed into their original (one source, alias recorded), never a second source
        assert a.plan.summary["tiffs"]["deduplicated"] == sum(1 for r in files if r.startswith("copies/"))
        # a different seed may change the split, never the guarantees
        c = run_import(work, root, imap=imap, fractions=fr, seed=it + 100, level=level, data=work / "c")
        assert_no_straddle(c.result.manifest, lambda s: s.identity, level)


def test_byte_identical_copy_with_a_conflicting_identity_is_refused(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "a/orig.tif", marks=spec_marks({(0, 0): 3}, 1), seed=1)
    shutil.copy(root / "a/orig.tif", root / "b" if (root / "b").mkdir() is None else root)
    shutil.copy(root / "a/orig.tif", root / "b/renamed.tif")
    rows = [{"file": "a/orig.tif", "specimen_id": "S1", "slide_id": "S1_A", "assay": "T7", "confirmed": True},
            {"file": "b/renamed.tif", "specimen_id": "S2", "slide_id": "S2_A", "assay": "T7", "confirmed": True}]
    p = tmp_path / "id.json"
    p.write_text(json.dumps({"rows": rows}))
    with pytest.raises(MLError, match="different identities"):
        run_import(tmp_path, root, imap=p)


def test_too_few_groups_leaves_test_empty_rather_than_filling_it_from_the_same_specimen(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM",))                 # one specimen, two slides
    imap = write_identity_map(tmp_path / "id.json", files)
    out = run_import(tmp_path, root, imap=imap, fractions={"train": 0.6, "val": 0.2, "test": 0.2})
    m = out.result.manifest
    assert len({c.split for c in m.crops}) == 1                           # the single specimen lands in ONE split
    assert {c.split for c in m.crops} & {"train", "val", "test"}
    missing = {"train", "val", "test"} - {c.split for c in m.crops}
    assert len(missing) == 2 and any("no images" in w or "too few" in w or "independent group" in w for w in m.warnings)
    # an empty split is reported, so train refuses (no leakage-free validation) instead of validating on the same specimen
    code = main(["--data-dir", str(tmp_path / "data"), "train", "--snapshot", out.result.version, "--dry-run", "--allow-small"])
    assert code == 2
    # two specimens, three splits requested: still no specimen on both sides
    root2 = tmp_path / "in2"
    root2.mkdir()
    files2 = standard_corpus(root2, ("AM", "AS"))
    out2 = run_import(tmp_path, root2, imap=write_identity_map(tmp_path / "id2.json", files2), data=tmp_path / "d2",
                      fractions={"train": 0.4, "val": 0.3, "test": 0.3})
    assert_no_straddle(out2.result.manifest, lambda s: s.identity)
    assert len({s.split for s in out2.result.manifest.sources}) == 2


def test_unconfirmed_identity_rows_are_refused_without_the_acknowledgement_flag(tmp_path):
    rng = np.random.default_rng(5)
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS"))
    for trial in range(4):
        rows = [{"file": rel, "specimen_id": k.split("-")[0], "slide_id": k.replace("-", "_"), "assay": "T7_AB", "confirmed": True}
                for k, rel in files.items()]
        bad = rng.choice(len(rows), size=int(rng.integers(1, 4)), replace=False)
        for i in bad:
            rows[int(i)]["confirmed"] = [False, "false", "no", "0"][trial % 4]
        p = tmp_path / f"id{trial}.json"
        p.write_text(json.dumps({"rows": rows}))
        data = tmp_path / f"d{trial}"
        with pytest.raises(MLError, match="UNCONFIRMED"):
            run_import(tmp_path, root, imap=p, data=data)
        assert not (data / "datasets").exists() or ds.list_snapshots(data) == []
        # the dry run reports it as a blocker, writes nothing
        dry = run_import(tmp_path, root, imap=p, data=data, dry=True)
        assert any("UNCONFIRMED" in b for b in dry.blockers)
        # with the acknowledgement it is recorded in the manifest
        ok = run_import(tmp_path, root, imap=p, data=data, allow_unconfirmed_identity=True)
        assert ok.result.manifest.splits["unconfirmed_identity"] is True
        assert any("UNCONFIRMED" in a or "unconfirmed" in a.lower() for a in ok.result.manifest.unresolved_assumptions)
    # rows missing from the map (inferred from file names) are equally unconfirmed
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"rows": [{"file": rel, "specimen_id": "S", "slide_id": "S_A", "confirmed": True}
                                      for rel in list(files.values())[:3]]}))
    with pytest.raises(MLError, match="UNCONFIRMED"):
        run_import(tmp_path, root, imap=p, data=tmp_path / "dp")


# ============================================================================================ 4. locked test contamination
def first_locked_import(tmp_path, specimens=("AM", "AS", "FS", "PS", "QS")):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, specimens)
    imap = write_identity_map(tmp_path / "id.json", files)
    d = tmp_path / "data"
    first = run_import(tmp_path, root, imap=imap, fractions=FR3, data=d)
    lock = ds.lock_test_set(d, first.result.version)
    test_src = sorted((s for s in first.result.manifest.sources if s.split == "test"), key=lambda s: s.name)
    return root, files, d, first, lock, test_src


def locked_hashes(d, version):
    m = ds.load_manifest(d, version)
    return {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in m.crops if c.split == "test"}


def second_dir(tmp_path, name="in2"):
    r = tmp_path / name
    r.mkdir()
    return r


def test_other_slide_of_a_locked_specimen_lands_in_test_only_and_is_not_exported(tmp_path):
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    spec = test_src[0].identity["specimen_id"]
    root2 = second_dir(tmp_path)
    make_tiff(root2 / "extra_slide_C.tif", marks=spec_marks({(0, 0): 4, (0, 2): 3}, 9), seed=901)
    make_tiff(root2 / "unrelated.tif", marks=spec_marks({(0, 0): 3, (1, 1): 2}, 5), seed=902)
    rows = [{"file": "extra_slide_C.tif", "specimen_id": spec, "slide_id": f"{spec}_C", "assay": "T7_AB", "confirmed": True},
            {"file": "unrelated.tif", "specimen_id": "NEW1", "slide_id": "NEW1_A", "assay": "T7_AB", "confirmed": True}]
    imap2 = tmp_path / "id2.json"
    imap2.write_text(json.dumps({"rows": rows}))
    before = locked_hashes(d, first.result.version)
    second = run_import(tmp_path, root2, imap=imap2, fractions=FR3, data=d, seed=4)
    m = second.result.manifest
    by_name = {s.name: s for s in m.sources}
    assert by_name["extra_slide_C.tif"].split == "test" and "locked" in (by_name["extra_slide_C.tif"].skipped or "")
    assert not [c for c in m.crops if c.source_file == "extra_slide_C.tif"], "a locked specimen's new slide is never re-exported"
    assert by_name["unrelated.tif"].split in ("train", "val")
    assert [c for c in m.crops if c.split == "test" and not c.frozen_from] == []
    assert {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in m.crops if c.split == "test"} == before
    assert ds.verify_snapshot(d, second.result.version) == []
    assert lb.verify_examination_provenance(m) == []


def test_locked_specimen_under_a_different_file_name_with_the_identity_map_stays_in_test(tmp_path):
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    src = test_src[0]
    root2 = second_dir(tmp_path)
    # different bytes (re-scan), different name, same specimen AND slide per the identity map
    make_tiff(root2 / "rescan_final_v2.tif", marks=spec_marks({(0, 0): 2}, 3), seed=777)
    imap2 = tmp_path / "id2.json"
    imap2.write_text(json.dumps({"rows": [{"file": "rescan_final_v2.tif", "specimen_id": src.identity["specimen_id"],
                                           "slide_id": src.identity["slide_id"], "assay": src.identity["assay"], "confirmed": True}]}))
    only = run_import(tmp_path, root2, imap=imap2, fractions=FR3, data=d, seed=2)      # only locked material: nothing new is exported
    assert {s.name: s.split for s in only.result.manifest.sources} == {"rescan_final_v2.tif": "test"}
    assert all(c.frozen_from for c in only.result.manifest.crops) and not [c for c in only.result.manifest.crops
                                                                          if c.source_file == "rescan_final_v2.tif"]
    # together with new data it is carried along in test only
    make_tiff(root2 / "newer.tif", marks=spec_marks({(0, 0): 2}, 3), seed=778)
    rows = json.loads(imap2.read_text())["rows"] + [{"file": "newer.tif", "specimen_id": "NEWZ", "slide_id": "NEWZ_A", "assay": "T7_AB",
                                                    "confirmed": True}]
    imap2.write_text(json.dumps({"rows": rows}))
    out = run_import(tmp_path, root2, imap=imap2, fractions=FR3, data=d, seed=2)
    names = {s.name: s for s in out.result.manifest.sources}
    assert names["rescan_final_v2.tif"].split == "test" and not [c for c in out.result.manifest.crops
                                                                if c.source_file == "rescan_final_v2.tif"]


def test_renamed_byte_identical_copy_of_a_locked_test_file_is_never_trained_on(tmp_path):
    """(a) a locked test file copied under another name (no identity information at all): it may land in test or be refused, but
    it must NOT be exported to train/val, and it must not break the import of the other new data."""
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    src = test_src[0]
    root2 = second_dir(tmp_path)
    shutil.copy(root / src.name, root2 / "zz_copy_final.tif")
    make_tiff(root2 / "other_new.tif", marks=spec_marks({(0, 0): 3}, 4), seed=910)
    before = locked_hashes(d, first.result.version)
    try:
        out = run_import(tmp_path, root2, fractions=FR3, data=d, allow_unconfirmed_identity=True, seed=3)
    except MLError:
        return                   # a refusal is safe (see the xfail test below for the usability problem it hides)
    m = out.result.manifest
    leaked = [c for c in m.crops if c.source_sha256 == src.tiff_sha256[src.name] and not c.frozen_from]
    assert leaked == [], "byte-identical copy of a locked test image was exported as a live crop"
    assert {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in m.crops if c.split == "test"} == before


def test_renamed_byte_identical_copy_lands_in_test_and_the_rest_imports(tmp_path):
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    src = test_src[0]
    root2 = second_dir(tmp_path)
    shutil.copy(root / src.name, root2 / "zz_copy_final.tif")
    make_tiff(root2 / "other_new.tif", marks=spec_marks({(0, 0): 3}, 4), seed=910)
    out = run_import(tmp_path, root2, fractions=FR3, data=d, allow_unconfirmed_identity=True, seed=3)
    names = {s.name: s for s in out.result.manifest.sources}
    assert names["zz_copy_final.tif"].split == "test"
    assert not [c for c in out.result.manifest.crops if c.source_file == "zz_copy_final.tif"]


def test_train_refuses_a_hand_built_snapshot_with_locked_material_in_train_or_val(tmp_path, monkeypatch, capsys):
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    lockfile = d / "datasets" / "test_lock.json"
    saved = lockfile.read_bytes()
    lockfile.unlink()                                   # "forget" the lock: a snapshot built without it ...
    hand = lb.import_legacy(d, root, opts(fractions={"train": 0.5, "val": 0.5, "test": 0.0}, seed=11),
                            channel_map=tmp_path / "channels.json", identity_map=tmp_path / "id.json", say=Recorder())
    lockfile.write_bytes(saved)                         # ... and the lock comes back
    tr_src = {s.name for s in hand.result.manifest.sources if s.split in ("train", "val")}
    assert {s.name for s in test_src} & tr_src, "fixture: locked specimens are in train/val of the hand-built snapshot"
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, dd: {"kind": "folder", "name": "stub", "path": "/x", "registry_id": None})
    for extra in ([], ["--dry-run"]):
        code = main(["--data-dir", str(d), "train", "--snapshot", hand.result.version, "--allow-small", "--epochs", "1", *extra])
        err = capsys.readouterr().err
        assert code == 2 and "locked test material" in err
    # nothing was registered
    assert not (d / "models" / "registry.json").exists() or not json.loads((d / "models/registry.json").read_text())["models"]


def test_locked_crops_stay_byte_identical_over_many_later_builds(tmp_path):
    root, files, d, first, lock, test_src = first_locked_import(tmp_path)
    before = locked_hashes(d, first.result.version)
    files_before = {c.path_tif: hashlib.sha256((d / "datasets" / first.result.version / c.path_tif).read_bytes()).hexdigest()
                    for c in first.result.manifest.crops if c.split == "test"}
    seen = {first.result.version}
    for k, seed in enumerate((1, 2, 3)):
        root2 = second_dir(tmp_path, f"more{k}")
        make_tiff(root2 / f"m{k}.tif", marks=spec_marks({(0, 0): 3, (0, k + 1): 2}, 20 + k), seed=950 + k)
        imap = tmp_path / f"idm{k}.json"
        imap.write_text(json.dumps({"rows": [{"file": f"m{k}.tif", "specimen_id": f"MORE{k}", "slide_id": f"MORE{k}_A", "assay": "T7_AB",
                                              "confirmed": True}]}))
        out = run_import(tmp_path, root2, imap=imap, fractions=FR3, data=d, seed=seed)
        v = out.result.version
        assert v not in seen
        seen.add(v)
        assert {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in out.result.manifest.crops if c.split == "test"} == before
        for c in out.result.manifest.crops:
            if c.split == "test":
                assert hashlib.sha256((d / "datasets" / v / c.path_tif).read_bytes()).hexdigest() == files_before[c.path_tif]
                assert c.frozen_from == first.result.version
        assert ds.verify_snapshot(d, v) == []
    # the first snapshot itself is untouched
    assert ds.verify_snapshot(d, first.result.version) == []


# ============================================================================================ 5. determinism and integrity
def small_corpus(tmp_path, specimens=("AM", "AS", "FS")):
    root = tmp_path / "in"
    root.mkdir(exist_ok=True)
    files = standard_corpus(root, specimens)
    return root, files, write_identity_map(tmp_path / "id.json", files)


def test_creation_order_and_worker_count_do_not_change_a_byte(tmp_path):
    outs = []
    for k, (rev, workers) in enumerate([(False, 1), (True, 3), (True, 2)]):
        work = tmp_path / f"w{k}"
        root = work / "in"
        root.mkdir(parents=True)
        # the same files, created in a different order with different mtimes
        src = tmp_path / "src"
        if k == 0:
            src.mkdir()
            standard_corpus(src, ("AM", "AS", "FS"))
        names = sorted(p.name for p in src.iterdir())
        for i, n in enumerate(reversed(names) if rev else names):
            shutil.copy(src / n, root / n)
            os.utime(root / n, ns=(10 ** 9 * (i + 1 + 7 * k),) * 2)
        corpus = {n[:-4].split("_")[-1]: n for n in names}
        imap = write_identity_map(work / "id.json", corpus)
        outs.append((work, run_import(work, root, imap=imap, fractions=FR, workers=workers, resume=bool(k % 2))))
    vs = {o.result.version for _w, o in outs}
    assert len(vs) == 1
    ds_ = [digest(w / "data", o.result.version) for w, o in outs]
    assert ds_[0] == ds_[1] == ds_[2]


def test_renaming_directories_changes_discovery_order_but_not_crops_or_splits(tmp_path):
    master = tmp_path / "master"
    base = {"AM-A": 1, "AM-B": 2, "AS-A": 3, "AS-B": 4, "FS-A": 5, "PS-A": 6}
    rows = []
    for name, seed in base.items():
        fn = f"29-09-2026_T7_AB_{name}.tif"
        make_tiff(master / fn, marks=spec_marks({(0, 0): 3, (0, 1): 2, (1, 0): 2}, seed), seed=seed)
        rows.append({"file": fn, "specimen_id": name.split("-")[0], "slide_id": name.replace("-", "_"), "assay": "T7_AB",
                     "confirmed": True})
    results = []
    for k, dirs in enumerate([("a", "b", "c"), ("c", "b", "a"), ("zz", "m", "0")]):
        work = tmp_path / f"w{k}"
        root = work / "in"
        for fn, dr in zip(sorted(p.name for p in master.iterdir()), list(dirs) * 2):
            (root / dr).mkdir(parents=True, exist_ok=True)
            shutil.copy(master / fn, root / dr / fn)
        imap = work / "id.json"
        imap.write_text(json.dumps({"rows": rows}))              # base names are unique: valid for every layout
        out = run_import(work, root, imap=imap, fractions=FR3, seed=3)
        results.append({c.crop_id: (c.split, c.group_id, c.tif_sha256, c.csv_sha256, c.norm_lo, c.norm_hi, c.n_positive)
                        for c in out.result.manifest.crops})
        assert len(results[-1]) == 18
    assert results[0] == results[1] == results[2]


def tamper_one_byte(path):
    unlock(path.parent)
    b = bytearray(path.read_bytes())
    b[len(b) // 2] ^= 0x01
    path.write_bytes(bytes(b))


@pytest.mark.parametrize("what", ["crop_tif", "crop_csv", "points", "regions", "manifest"])
def test_tampering_is_detected_by_verify_and_refused_by_train(tmp_path, what, monkeypatch, capsys):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS", "FS", "PS"))
    out = run_import(tmp_path, root, imap=imap, fractions=FR3)
    d, v = tmp_path / "data", out.result.version
    snap = d / "datasets" / v
    assert main(["--data-dir", str(d), "verify", v]) == 0
    c = next(c for c in out.result.manifest.crops if c.split == "train")
    target = {"crop_tif": snap / c.path_tif, "crop_csv": snap / c.path_csv, "points": snap / "points.csv",
              "regions": snap / "regions.jsonl", "manifest": snap / "manifest.json"}[what]
    if what == "crop_csv":
        c2 = next(c for c in out.result.manifest.crops if c.split == "train" and c.n_positive)
        target = snap / c2.path_csv
    tamper_one_byte(target) if what != "manifest" else (unlock(snap), target.write_text(target.read_text().replace(
        '"n_positive": ', '"n_positive": 1', 1)))
    capsys.readouterr()
    try:
        code = main(["--data-dir", str(d), "verify", v])
    except Exception:                              # an unreadable manifest is also a refusal
        code = 3
    text = capsys.readouterr().out
    assert code == 3 or "PROBLEMS" in text
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, dd: {"kind": "folder", "name": "stub", "path": "/x", "registry_id": None})
    try:
        code = main(["--data-dir", str(d), "train", "--snapshot", v, "--dry-run", "--allow-small"])
    except Exception:
        code = 2
    assert code == 2 and "nothing was trained" not in capsys.readouterr().out
    # a damaged snapshot can never be locked as the test set either
    if what != "manifest":
        with pytest.raises(MLError, match="damaged"):
            ds.lock_test_set(d, v)


def test_dry_run_writes_nothing_not_even_staging_in_every_state(tmp_path, capsys):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS", "FS", "PS"))
    cmap = write_channel_map(tmp_path / "channels.json", defaults=RED_GREEN)
    d = tmp_path / "data"
    for state in ("no data dir", "empty datasets dir", "existing snapshot + lock"):
        if state == "empty datasets dir":
            (d / "datasets").mkdir(parents=True)
        elif state == "existing snapshot + lock":
            first = run_import(tmp_path, root, imap=imap, fractions=FR3, data=d)
            ds.lock_test_set(d, first.result.version)
        before_data, before_in = tree(d), tree(root)
        before_all = tree(tmp_path)
        code = main(["--data-dir", str(d), "import-legacy", "--input", str(root), "--channel-map", str(cmap), "--identity-map", str(imap),
                     "--dry-run", "--fractions", "0.6,0.2,0.2", "--workers", "3"])
        assert code == 0 and "DRY RUN" in capsys.readouterr().out
        assert tree(d) == before_data and tree(root) == before_in and tree(tmp_path) == before_all
        assert not [p for p in tmp_path.rglob("*") if p.name.startswith(".staging")]
        # same through the API (with and without resume) and also for a refused dry run
        lb.import_legacy(d, root, opts(fractions=FR3), channel_map=cmap, identity_map=imap, dry_run=True, resume=False, say=Recorder())
        assert tree(tmp_path) == before_all


@pytest.mark.parametrize("nth", [1, 4, 100])
def test_interrupt_while_writing_the_nth_crop_publishes_nothing_and_a_rerun_matches(tmp_path, monkeypatch, nth):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS", "FS"))
    clean = run_import(tmp_path, root, imap=imap, data=tmp_path / "clean")
    real = ds.write_crop_files
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == nth:
            raise OSError("disk went away")
        return real(*a, **k)
    monkeypatch.setattr(ds, "write_crop_files", flaky)
    d = tmp_path / "data"
    if nth <= len(clean.result.manifest.crops):
        with pytest.raises(OSError):
            run_import(tmp_path, root, imap=imap, data=d)
        assert ds.list_snapshots(d) == []
        assert not [p for p in (d / "datasets").iterdir() if not p.name.startswith(".staging-")]
        # the staging directory is never listed / loadable / lockable / trainable as a snapshot
        for p in (d / "datasets").iterdir():
            assert not (p / "manifest.json").exists()
        assert main(["--data-dir", str(d), "list", "datasets"]) == 0
    monkeypatch.setattr(ds, "write_crop_files", real)
    out = run_import(tmp_path, root, imap=imap, data=d)
    assert out.result.version == clean.result.version
    assert digest(d, out.result.version) == digest(tmp_path / "clean", clean.result.version)
    assert ds.list_snapshots(d) == [out.result.version]
    assert not [p for p in (d / "datasets").iterdir() if p.name.startswith(".staging")]


def test_failure_between_manifest_write_and_publication_still_lists_no_snapshot(tmp_path, monkeypatch):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS"))
    real = os.rename

    def boom(src, dst, *a, **k):
        if ".staging-" in str(src) and "datasets" in str(dst):
            raise OSError("killed during publication")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "rename", boom)
    d = tmp_path / "data"
    with pytest.raises(OSError):
        run_import(tmp_path, root, imap=imap, data=d)
    stage = [p for p in (d / "datasets").iterdir()]
    assert stage and all(p.name.startswith(".staging") for p in stage)
    assert (stage[0] / "manifest.json").is_file()                 # complete manifest INSIDE the staging dir ...
    assert ds.list_snapshots(d) == []                             # ... which is still not a snapshot
    monkeypatch.setattr(os, "rename", real)
    out = run_import(tmp_path, root, imap=imap, data=d)
    assert ds.list_snapshots(d) == [out.result.version]


def test_disk_space_refusal_happens_before_anything_is_written(tmp_path, monkeypatch):
    root, files, imap = small_corpus(tmp_path)
    d = tmp_path / "data"
    before = tree(tmp_path)

    class U:
        total, used, free = 10 ** 9, 10 ** 9 - 1000, 1000
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U)
    with pytest.raises(MLError, match="not enough free disk space"):
        run_import(tmp_path, root, imap=imap, data=d)
    assert [p for p in tree(tmp_path) if p not in before and not p[0].startswith("channels.json")] in ([], [("data", -1)]) \
        or not (d / "datasets").exists() or list((d / "datasets").iterdir()) == []
    assert not list(d.rglob(".staging*")) and not list(d.rglob("*.tif"))
    # CLI: exit 2 with REFUSED, and the dry run reports the blocker but writes nothing
    cmap = tmp_path / "channels.json"
    assert main(["--data-dir", str(d), "import-legacy", "--input", str(root), "--channel-map", str(cmap), "--identity-map", str(imap)]) == 2
    assert not list(d.rglob("*.tif"))


def test_dtype_is_preserved_uint16_beyond_255_and_8_bit_stays_8_bit_never_rescaled(tmp_path):
    root = tmp_path / "in"
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks({(0, 0): 3, (1, 1): 2}, 1), seed=1)               # uint16 up to 60000
    make_tiff(root / "29-09-2026_T7_AB_AS-A.tif", marks=spec_marks({(0, 0): 3, (1, 1): 2}, 2), seed=2, dtype=np.uint8)   # 8-bit
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True)
    seen = set()
    for c in out.result.manifest.crops:
        img, lab = ds.load_crop(tmp_path / "data", out.result.version, c)
        ref = tifffile.imread(str(root / c.source_file))[c.page]
        x0, y0, x1, y1 = c.box_xyxy
        assert img.dtype == ref.dtype == np.dtype(c.dtype) and np.array_equal(img, ref[y0:y1, x0:x1])
        seen.add(str(img.dtype))
        if img.dtype == np.uint16:
            assert img.max() > 255
        else:
            assert img.max() <= 255 and c.norm_hi <= 255 * 1.0001 or c.norm_hi < 256
    assert seen == {"uint16", "uint8"}
    sat = {s.name: {r: v["saturation_level"] for r, v in s.page_normalization.items()} for s in out.result.manifest.sources}
    assert set(sat["29-09-2026_T7_AB_AM-A.tif"].values()) == {65535.0} and set(sat["29-09-2026_T7_AB_AS-A.tif"].values()) == {255.0}
    # unsupported types are refused (not converted)
    make_tiff(root / "29-09-2026_T7_AB_FS-A.tif", marks=spec_marks({(0, 0): 3}, 3), seed=3, dtype=np.float32)
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True, dry=True, data=tmp_path / "d2")
    assert out.plan.summary["tiffs"]["rejected_files"]["29-09-2026_T7_AB_FS-A.tif"] == ["unsupported_dtype"]


# ============================================================================================ 6. normalisation parity
def saturated_stack(root, name, marks, seed, sat_frac=0.0):
    """Two planes with noise and a block of saturated (65535) pixels inside field (0, 0) that would move the 99.8 percentile."""
    rng = np.random.default_rng(seed)
    planes = []
    for k in range(2):
        p = rng.integers(200, 30000, (H, W)).astype(np.uint16)
        if sat_frac:
            p[10:45, 10:60][rng.random((35, 50)) < sat_frac] = 65535
            p[80:115, 110:160][rng.random((35, 50)) < sat_frac] = 65535       # inside an UNexamined cell too
        planes.append(p)
    roi = point_roi([(x, y) for x, y, *_ in marks], [m[2] for m in marks], [m[3] for m in marks])
    write_stack(Path(root) / name, planes, roi=roi, overlays=[grid_overlay(GRID_XS, GRID_YS, W, H)])
    return planes


def independent_valid_mask(shape, cells):
    """Union of the pixel boxes of the examined cells (box = ceil(edge - 0.5): the pixel centres inside the cell)."""
    v = np.zeros(shape, bool)
    for r, c in cells:
        x0, x1 = (math.ceil(GRID_XS[c] - 0.5), math.ceil(GRID_XS[c + 1] - 0.5))
        y0, y1 = (math.ceil(GRID_YS[r] - 0.5), math.ceil(GRID_YS[r + 1] - 0.5))
        v[y0:y1, x0:x1] = True
    return v


def test_page_bounds_in_the_manifest_equal_an_independent_computation_and_exclude_saturation(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    marks = spec_marks({(0, 0): 3, (0, 4): 2, (1, 1): 4}, 1)
    planes = saturated_stack(root, "29-09-2026_T7_AB_AM-A.tif", marks, seed=5, sat_frac=0.35)
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True)
    (src,) = out.result.manifest.sources
    examined = {0: [(0, 0), (1, 1)], 1: [(0, 1)]}                 # cells (row, col) with marks, per page (cell 1 -> (0,1); cell 4 -> (1,1))
    roles = {0: "red", 1: "green"}
    for page, cells in examined.items():
        b = src.page_normalization[roles[page]]
        valid = independent_valid_mask(planes[page].shape, cells)
        want = cm.page_bounds(planes[page], valid, (1.0, 99.8))
        assert b["lo"] == want["lo"] and b["hi"] == want["hi"] and b["n_used"] == want["n_used"]
        assert b["n_saturated"] == want["n_saturated"]
        if page == 0:
            assert b["n_saturated"] > 300, "fixture: the saturated pixels are inside an examined field"
        # saturated pixels are really excluded: including them would change hi
        with_sat = np.percentile(planes[page][valid].astype(float), [1.0, 99.8])
        if b["n_saturated"]:
            assert with_sat[1] > b["hi"] and b["hi"] < 65535
        # the valid mask is the eligible fields only, NOT the whole page
        whole = cm.page_bounds(planes[page], None, (1.0, 99.8))
        assert b["n_valid"] == int(valid.sum()) < whole["n_valid"]
        for c in out.result.manifest.crops:
            if c.page == page:
                assert (c.norm_lo, c.norm_hi) == (want["lo"], want["hi"])      # every crop of the page uses the page's lo/hi


@pytest.mark.parametrize("d", [1, 2, 4])
def test_prepare_crop_is_bit_for_bit_the_trainer_input(tmp_path, d):
    root = tmp_path / "in"
    root.mkdir()
    saturated_stack(root, "29-09-2026_T7_AB_AM-A.tif", spec_marks({(0, 0): 3, (0, 4): 2, (1, 1): 4}, 1), seed=5, sat_frac=0.2)
    saturated_stack(root, "29-09-2026_T7_AB_AS-A.tif", spec_marks({(0, 1): 3, (0, 5): 2, (1, 2): 4}, 2), seed=6, sat_frac=0.0)
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True, downscale=d)
    from polony_app.ml import train as tr
    load_xy = getattr(tr, "_load_xy", None)
    if load_xy is None:
        pytest.skip("train._load_xy was renamed")
    m = out.result.manifest
    crops = [c for c in m.crops if c.split in ("train", "val")]
    xs, spots, n_drop = load_xy(tmp_path / "data", out.result.version, crops, {"downscale": d, "normaliser": "valid_unsaturated"})
    assert len(xs) == len(crops) > 0
    for c, x, s in zip(crops, xs, spots):
        img, lab = ds.load_crop(tmp_path / "data", out.result.version, c)
        # independent formula (float32 arithmetic exactly as documented): (x - lo) / (hi - lo), then block mean with trimmed offset
        lo, hi = np.float32(c.norm_lo), np.float32(float(c.norm_hi) - float(c.norm_lo))
        nrm = ((img.astype(np.float32) - lo) / hi).astype(np.float32)
        oy, ox = (-c.box_xyxy[1]) % d, (-c.box_xyxy[0]) % d
        nrm = nrm[oy:, ox:]
        hs, ws = nrm.shape[0] // d, nrm.shape[1] // d
        want = nrm[:hs * d, :ws * d].reshape(hs, d, ws, d).mean(axis=(1, 3), dtype=np.float32) if d > 1 else nrm
        assert x.dtype == np.float32 and x.shape == want.shape and np.array_equal(x, want)
        y2, l2, dr, off = cm.prepare_crop(img, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=d, origin_xy=c.box_xyxy[:2])
        assert np.array_equal(y2, x) and np.array_equal(l2, s)
        # saturated pixels are inside the crop but produce values > 1 (not clipped), never silently clipped to 1
        if (img == 65535).any():
            assert x.max() > 1.0
    lazy = getattr(tr, "load_lazy", None)               # the memory-saving loader must hand the trainer the very same numbers
    if lazy is not None:
        lz, lspots, ldrop = lazy(tmp_path / "data", out.result.version, crops, {"downscale": d, "normaliser": "valid_unsaturated"})
        assert ldrop == n_drop and all(np.array_equal(a, b) for a, b in zip(lspots, spots))
        assert all(np.array_equal(lz[i], xs[i]) for i in range(len(crops)))


# ============================================================================================ 7. dry-run reporting
def parse_excluded(text):
    m = re.search(r"EXCLUDED because the researcher's examination could not be proven: (\d+) \(unknown (\d+), conflict (\d+)\)", text)
    assert m, text
    return tuple(int(g) for g in m.groups())


@pytest.mark.parametrize("mode", ["marked-only", "examined-fields-manifest"])
def test_dry_run_reports_the_exact_number_of_unproven_tuples(tmp_path, capsys, mode):
    rng = np.random.default_rng(77 if mode == "marked-only" else 78)
    root = tmp_path / "in"
    root.mkdir()
    case = random_eligibility_case(rng, 3)
    for f in case:
        f["marks"].setdefault((0, 0), 2)               # every file has at least one mark (kept in marked-only mode)
    files, man = write_case(root, tmp_path, case, seed=3)
    if man is None:
        man = write_examined(tmp_path / "man.json", {})
    cmap = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    before = tree(tmp_path)
    argv = ["--data-dir", str(tmp_path / "data"), "import-legacy", "--input", str(root), "--channel-map", str(cmap), "--dry-run",
            "--examined-fields", str(man), "--mode", mode, "--allow-unconfirmed-identity"]
    assert main(argv) == 0
    text = capsys.readouterr().out
    tally, _ = expected_tally(files, mode)
    assert parse_excluded(text) == (tally["unknown"] + tally["conflict"], tally["unknown"], tally["conflict"])
    assert f"positive_marked {tally['positive_marked']}, verified_zero {tally['verified_zero']}" in text
    assert f"FIELD/PAGE TUPLES ({sum(tally.values())} in accepted files)" in text
    assert tree(tmp_path) == before
    # the real import prints the same number
    argv = [a for a in argv if a != "--dry-run"]
    if tally["positive_marked"] + tally["verified_zero"]:
        assert main(argv) == 0
        assert f"{tally['unknown'] + tally['conflict']} excluded because examination could not be proven" in capsys.readouterr().out


def test_every_contradiction_class_is_counted_as_conflict_in_the_dry_run(tmp_path, capsys):
    root = tmp_path / "in"
    root.mkdir()
    marks = {(0, 0): 3, (0, 1): 2, (0, 2): 2, (1, 3): 1}
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks(marks, 1), seed=2)
    man = write_examined(tmp_path / "m.json", {"29-09-2026_T7_AB_AM-A.tif": {
        0: [entry_dict(0, 3), entry_dict(1, 5), entry_dict(2, 0), entry_dict(5, 4), entry_dict(6, 0), entry_dict(7, None)],
        1: [entry_dict(3, 2)]}})
    cmap = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    main(["--data-dir", str(tmp_path / "data"), "import-legacy", "--input", str(root), "--channel-map", str(cmap), "--dry-run",
          "--examined-fields", str(man), "--mode", "examined-fields-manifest", "--allow-unconfirmed-identity"])
    # positive: (0,0); verified zero: (0,6); conflict: (0,1) stated 5 vs 2, (0,2) stated 0 vs 2, (0,5) stated 4 vs none, (1,3) stated 2 vs 1
    # unknown: everything else incl. (0,7) listed without a count
    assert parse_excluded(capsys.readouterr().out) == (18 - 2, 18 - 2 - 4, 4)


def test_an_unpublished_staging_directory_is_not_addressable_as_a_snapshot(tmp_path, monkeypatch):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS"))
    real = os.rename

    def boom(src, dst, *a, **k):
        if ".staging-" in str(src) and "datasets" in str(dst):
            raise OSError("killed during publication")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "rename", boom)
    d = tmp_path / "data"
    with pytest.raises(OSError):
        run_import(tmp_path, root, imap=imap, data=d)
    (stage,) = list((d / "datasets").iterdir())
    with pytest.raises(MLError):
        ds.load_manifest(d, stage.name)


# ============================================================================================ more adversarial corpora
def assert_accounting(summary):
    """Every ImageJ point is either an exported label or a counted, reasoned drop: none vanishes, none is invented."""
    a = summary["annotations"]
    assert a["points_total"] == a["labels_exported"] + a["dropped_total"], a
    assert a["dropped_total"] == sum(a["dropped_by_reason"].values())


def random_points_file(rng, n_cells=9):
    """Points with strays, counter-less points, other-plane points, invalid points. Returns (marks, own_counters)
    marks = [(x, y, counter|None, position)] in the ImageJ frame. Every counter has >= 3 own marks so its majority cell is clear."""
    marks, taken = [], []

    def free(x, y):
        return all((x - a) ** 2 + (y - b) ** 2 > 16 for a, b in taken)

    def add(x, y, ctr, pos):
        if free(x, y):
            taken.append((x, y))
            marks.append((round(x, 3), round(y, 3), ctr, pos))
    for cell in range(n_cells):
        r, c = divmod(cell, 3)
        for page in (1, 2):
            if rng.random() < (0.5 if page == 1 else 0.25):
                for _ in range(int(rng.integers(3, 6))):
                    add(GRID_XS[c] + float(rng.uniform(2, 48)), GRID_YS[r] + float(rng.uniform(2, 33)), cell, page)
    owners = {m[2] for m in marks}
    for _ in range(int(rng.integers(0, 8))):                      # strays: geometric cell != counter's cell
        ctr = int(rng.choice(sorted(owners))) if owners else 0
        r, c = divmod(int(rng.integers(0, n_cells)), 3)
        add(GRID_XS[c] + float(rng.uniform(2, 48)), GRID_YS[r] + float(rng.uniform(2, 33)), ctr, int(rng.integers(1, 3)))
    add(3.0, 3.0, sorted(owners)[0] if owners else 0, 1)           # inside the image, outside the grid
    add(W + 12.0, 20.0, sorted(owners)[0] if owners else 0, 1)     # beyond the image
    add(30.0, 30.0, sorted(owners)[0] if owners else 0, 0)         # unspecified plane
    add(31.0, 60.0, sorted(owners)[0] if owners else 0, 9)         # plane that does not exist
    return marks


def geom_cell(x, y):
    for r in range(3):
        for c in range(3):
            if GRID_XS[c] <= x < GRID_XS[c + 1] and GRID_YS[r] <= y < GRID_YS[r + 1]:
                return r * 3 + c
    return None


def test_random_strays_and_invalid_points_match_an_independent_label_oracle(tmp_path):
    rng = np.random.default_rng(555)
    total = 0
    for it in range(6):
        work = tmp_path / f"w{it}"
        root = work / "in"
        root.mkdir(parents=True)
        spec, files = {}, {}
        for k in range(2):
            marks = random_points_file(rng)
            ok = [m for m in marks if m[1] >= 0 and m[0] < W and m[3] in (1, 2)]
            if not any(m[3] in (1, 2) and geom_cell(m[0], m[1]) is not None for m in marks):
                continue
            rel = f"29-09-2026_T7_AB_{'AB'[k]}{'AB'[k]}-A.tif"
            make_tiff(root / rel, marks=marks, seed=it * 10 + k + 1)
            files[rel] = marks
        if not files:
            continue
        out = run_import(work, root, allow_unconfirmed_identity=True, fractions={"train": 1.0, "val": 0.0, "test": 0.0})
        assert_accounting(out.plan.summary)
        # independent oracle
        want = {}
        for rel, marks in files.items():
            pts = [m for m in marks if m[3] in (1, 2) and 0 <= m[0] < W and 0 <= m[1] < H and geom_cell(m[0], m[1]) is not None]
            maj = {}
            for ctr in {m[2] for m in pts}:
                cnt = Counter(geom_cell(m[0], m[1]) for m in pts if m[2] == ctr)
                (c1, n1), *rest = cnt.most_common()
                if not rest or rest[0][1] < n1:
                    maj[ctr] = c1
            for page in (1, 2):
                for cell in range(9):
                    here = [m for m in pts if m[3] == page and geom_cell(m[0], m[1]) == cell]
                    own = [m for m in here if maj.get(m[2]) == cell]
                    if own:
                        want[(rel, page - 1, cell)] = sorted((round(m[1] - 0.5 - math.ceil(GRID_YS[cell // 3] - 0.5), 3),
                                                              round(m[0] - 0.5 - math.ceil(GRID_XS[cell % 3] - 0.5), 3)) for m in here)
        have = {}
        for c in out.result.manifest.crops:
            img, lab = ds.load_crop(tmp_path / "w{}".format(it) / "data", out.result.version, c)
            have[(c.source_file, c.page, c.provenance["grid_row"] * 3 + c.provenance["grid_col"])] = sorted(
                (round(a, 3), round(b, 3)) for a, b in lab.tolist())
        assert set(have) == set(want)
        for k in want:
            assert len(have[k]) == len(want[k])
            assert np.allclose(have[k], want[k], atol=2e-3)
        total += len(have)
        assert lb.verify_examination_provenance(out.result.manifest) == []
    assert total >= 8


def test_irregular_fractional_grids_account_for_every_point(tmp_path):
    """Non-integer grid lines, tiny cells (< min_crop_px), cells leaving the image: nothing is clamped, every point is accounted."""
    rng = np.random.default_rng(808)
    total_crops = total_skipped = 0
    for it in range(6):
        work = tmp_path / f"w{it}"
        root = work / "in"
        root.mkdir(parents=True)
        nx, ny = int(rng.integers(2, 6)), int(rng.integers(2, 5))
        xs = sorted({round(float(v), 2) for v in rng.uniform(3, W + (6 if it % 3 == 0 else -2), nx + 1)})
        ys = sorted({round(float(v), 2) for v in rng.uniform(3, H + (6 if it % 3 == 1 else -2), ny + 1)})
        if len(xs) < 2 or len(ys) < 2:
            continue
        marks = []
        for cell in range((len(xs) - 1) * (len(ys) - 1)):
            r, c = divmod(cell, len(xs) - 1)
            for q in range(3):
                x = xs[c] + (xs[c + 1] - xs[c]) * [0.05, 0.5, 0.95][q] + 0.0123 * q
                y = ys[r] + (ys[r + 1] - ys[r]) * [0.9, 0.5, 0.1][q]
                if 0 < x < W and 0 < y < H:
                    marks.append((round(x, 3), round(y, 3), cell % 200, 1))
        marks = [m for i, m in enumerate(marks) if all((m[0] - o[0]) ** 2 + (m[1] - o[1]) ** 2 > 2.0 for o in marks[:i])]
        if not marks:
            continue
        make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=marks, grid_xs=xs, grid_ys=ys, seed=it)
        out = run_import(work, root, allow_unconfirmed_identity=True, dry=True)
        s = out.plan.summary
        assert_accounting(s)
        assert s["crops"]["created"] + s["crops"]["skipped"] == s["fields"]["eligible_examined_fields"]
        (plan,) = out.plan.plans if out.plan.plans else (None,)
        if plan is None:
            continue
        for cd in plan.ext.cands:
            x0, y0, x1, y1 = cd.box_file
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H and x1 - x0 >= 32 and y1 - y0 >= 32       # never clamped / shrunk
            h, w = cd.shape_hw
            for (py, px) in cd.points:
                assert -0.5 <= py < h - 0.5 and -0.5 <= px < w - 0.5
        total_crops += s["crops"]["created"]
        total_skipped += s["crops"]["skipped"]
        # a skipped examined cell leaves its points counted as dropped, with the reason
        if s["crops"]["skipped"]:
            assert s["annotations"]["dropped_by_reason"].get("crop_not_exported", 0) > 0
    assert total_crops >= 3 and total_skipped >= 1


def test_page_not_trained_leaves_no_crop_label_or_statistics_for_that_page(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks({(0, 0): 3, (1, 0): 4, (1, 5): 2}, 1), seed=1)
    cmap = write_channel_map(tmp_path / "c.json", defaults={0: "red", 1: ("green", False)})
    out = run_import(tmp_path, root, cmap=cmap, allow_unconfirmed_identity=True)
    m = out.result.manifest
    assert {c.page for c in m.crops} == {0} and {c.channel_role for c in m.crops} == {"red"}
    assert set(m.sources[0].page_normalization) == {"red"}
    a = out.plan.summary["annotations"]
    assert a["labels_exported"] == 3 and a["dropped_by_reason"] == {"page_not_trained": 6}
    assert_accounting(out.plan.summary)
    assert out.plan.summary["fields"]["tuples_total"] == 9                # only the trained page has tuples


def test_a_grid_only_file_with_a_stated_zero_gives_a_verified_zero_crop_and_nothing_else(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    make_tiff(root / "29-09-2026_T7_AB_AM-A.tif", marks=spec_marks({(0, 0): 3}, 1), seed=1)
    make_tiff(root / "29-09-2026_T7_AB_AS-A.tif", marks=[], seed=2)        # grid overlay, no ROI at all
    man = write_examined(tmp_path / "m.json", {"29-09-2026_T7_AB_AS-A.tif": {0: [entry_dict(4, 0)]},
                                               "29-09-2026_T7_AB_AM-A.tif": {0: [entry_dict(0, 3)]}})
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True, examined=man, mode="examined-fields-manifest")
    cs = {(c.source_file, c.page, c.region_id): c for c in out.result.manifest.crops}
    assert set(cs) == {("29-09-2026_T7_AB_AM-A.tif", 0, "r0c0"), ("29-09-2026_T7_AB_AS-A.tif", 0, "r1c1")}
    z = cs[("29-09-2026_T7_AB_AS-A.tif", 0, "r1c1")]
    assert z.n_positive == 0 and z.provenance["examination_status"] == "verified_zero" and z.provenance["field_counters"] == []
    img, lab = ds.load_crop(tmp_path / "data", out.result.version, z)
    assert lab.shape == (0, 2) and img.dtype == np.uint16 and z.norm_lo is not None and z.norm_hi > z.norm_lo
    assert_accounting(out.plan.summary)
    # in marked-only mode the unmarked file is a structured rejection, never a source of negatives
    dry = run_import(tmp_path, root, allow_unconfirmed_identity=True, examined=man, dry=True, data=tmp_path / "d1")
    assert dry.plan.summary["tiffs"]["rejected_files"] == {"29-09-2026_T7_AB_AS-A.tif": ["missing_roi"]}


def test_resume_never_trusts_a_damaged_staging_crop(tmp_path, monkeypatch):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS", "FS", "PS"))
    clean = run_import(tmp_path, root, imap=imap, data=tmp_path / "clean")
    real = ds.write_crop_files
    n = {"i": 0}

    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] == 14:
            raise OSError("power cut")
        return real(*a, **k)
    monkeypatch.setattr(ds, "write_crop_files", flaky)
    d = tmp_path / "data"
    with pytest.raises(OSError):
        run_import(tmp_path, root, imap=imap, data=d)
    (stage,) = [p for p in (d / "datasets").iterdir()]
    done = sorted((stage / ".done").iterdir())
    assert len(done) >= 2
    recs = json.loads(done[0].read_text())["crops"]
    tif = stage / recs[0]["path_tif"]
    b = bytearray(tif.read_bytes())
    b[-3] ^= 0xFF
    tif.write_bytes(bytes(b))                                              # flipped byte in a "finished" crop
    (stage / json.loads(done[1].read_text())["crops"][0]["path_csv"]).unlink()      # a vanished label file
    done[-1].write_text("{ this marker is damaged")                       # and a damaged marker
    monkeypatch.setattr(ds, "write_crop_files", real)
    out = run_import(tmp_path, root, imap=imap, data=d)
    assert out.result.version == clean.result.version and ds.verify_snapshot(d, out.result.version) == []
    assert digest(d, out.result.version) == digest(tmp_path / "clean", clean.result.version)


def test_locking_unconfirmed_identities_needs_the_flag_and_nothing_is_locked_otherwise(tmp_path, capsys):
    root = tmp_path / "in"
    root.mkdir()
    standard_corpus(root, ("AM", "AS", "FS", "PS"))
    out = run_import(tmp_path, root, allow_unconfirmed_identity=True, fractions=FR3)
    d = tmp_path / "data"
    assert out.result.manifest.splits["unconfirmed_identity"] is True
    code = main(["--data-dir", str(d), "split", "--snapshot", out.result.version, "--lock-test"])
    assert code == 2 and "UNCONFIRMED" in capsys.readouterr().err
    assert not (d / "datasets" / "test_lock.json").exists()


def test_nan_fractions_are_refused(tmp_path, capsys):
    root, files, imap = small_corpus(tmp_path, ("AM", "AS", "FS", "PS"))
    cmap = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    code = main(["--data-dir", str(tmp_path / "d"), "import-legacy", "--input", str(root), "--channel-map", str(cmap),
                 "--identity-map", str(imap), "--dry-run", "--fractions", "nan,0.5,0.5"])
    assert code == 2
