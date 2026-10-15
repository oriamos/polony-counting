"""Grouped splits, leakage prevention and the locked test set."""
import json
import shutil

import pytest

from polony_app.ml import dataset as ds
from polony_app.ml.common import MLError, sha256_file, lock_file_path
from polony_app.ml.splits import load_lock
from polony_app.ml.train import TrainOptions, plan_training
from test_ml_support import Recorder, add_analysis, new_store

T0 = "2026-01-01T00:00:00Z"
FR = {"train": 0.5, "val": 0.25, "test": 0.25}


def build(store, d, ids, **kw):
    kw.setdefault("created_at", T0)
    kw.setdefault("fractions", FR)
    return ds.build_snapshot(store, d, ids, ds.BuildOptions(**kw), say=Recorder())


def make_world(tmp_path, n=6, replicate_of_first=True):
    store = new_store(tmp_path)
    ids = {}
    for i in range(1, n + 1):
        ids[f"S{i}_A"] = add_analysis(store, f"slide-{i}A", slide_id=f"S{i}_A", seed=i)
    if replicate_of_first:
        ids["S1_B"] = add_analysis(store, "slide-1B", slide_id="S1_B", seed=50)
    return store, ids


def split_of(r):
    return {s.analysis_id: s.split for s in r.manifest.sources}


def test_same_specimen_never_straddles_splits(tmp_path):
    store, ids = make_world(tmp_path)
    for seed in range(6):
        r = build(store, tmp_path / f"d{seed}", sorted(ids.values()), seed=seed)
        sp = split_of(r)
        assert sp[ids["S1_A"]] == sp[ids["S1_B"]]          # replicates of one specimen stay together
        # no group id appears in two splits
        seen = {}
        for c in r.manifest.crops:
            seen.setdefault(c.group_id, set()).add(c.split)
        assert all(len(v) == 1 for v in seen.values())


def test_all_crops_of_a_slide_share_its_split_and_all_splits_filled(tmp_path):
    store, ids = make_world(tmp_path)
    r = build(store, tmp_path / "d", sorted(ids.values()), seed=1)
    by_an = {}
    for c in r.manifest.crops:
        by_an.setdefault(c.analysis_id, set()).add(c.split)
    assert all(len(v) == 1 for v in by_an.values())
    assert {c.split for c in r.manifest.crops} == {"train", "val", "test"}
    for sp in ("train", "val", "test"):
        d = r.path / "spotiflow" / sp
        assert any(d.glob("*.tif")) and len(list(d.glob("*.tif"))) == len(list(d.glob("*.csv")))


def test_split_is_deterministic_for_a_seed(tmp_path):
    store, ids = make_world(tmp_path)
    a = build(store, tmp_path / "d1", sorted(ids.values()), seed=3)
    b = build(store, tmp_path / "d2", sorted(ids.values()), seed=3)
    c = build(store, tmp_path / "d3", sorted(ids.values()), seed=4)
    assert a.version == b.version and split_of(a) == split_of(b)
    assert split_of(a) != split_of(c) or a.version != c.version


def test_filename_inferred_identity_is_refused_unless_allowed(tmp_path):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"29-09-2026_T7_AB_S{i}-A", slide_id=None, seed=i) for i in range(1, 5)]
    with pytest.raises(MLError, match="inferred from file names only"):
        build(store, tmp_path / "d", ids)
    r = build(store, tmp_path / "d", ids, allow_unconfirmed_identity=True)
    assert r.manifest.splits["unconfirmed_identity"] is True
    assert any("UNCONFIRMED" in w for w in r.manifest.warnings)
    assert any("UNCONFIRMED" in a for a in r.manifest.unresolved_assumptions)


def test_identity_map_confirms_identity(tmp_path):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"scan{i}", slide_id=None, seed=i) for i in range(1, 5)]
    imap = {aid: {"specimen_id": f"P{i}", "slide_id": f"P{i}_A", "assay": "T7"} for i, aid in enumerate(ids)}
    r = build(store, tmp_path / "d", ids, identity_map=imap)
    assert r.manifest.splits["unconfirmed_identity"] is False
    # without the map the names 'scan1'... carry no identity at all -> refused
    with pytest.raises(MLError):
        build(store, tmp_path / "d2", ids)


def test_too_few_groups_warns_and_never_fills_splits_from_one_slide(tmp_path):
    store, ids = make_world(tmp_path, n=1, replicate_of_first=False)
    r = build(store, tmp_path / "d", list(ids.values()))
    assert {c.split for c in r.manifest.crops} == {"train"}
    assert any("independent group" in w for w in r.manifest.warnings)
    assert any("NO val crops" in w for w in r.manifest.warnings)


# ---------------------------------------------------------------- locked test set
def test_lock_test_set_then_later_snapshots_keep_it_test_only_and_unchanged(tmp_path):
    store, ids = make_world(tmp_path)
    d = tmp_path / "d"
    s0 = build(store, d, sorted(ids.values()), seed=1)
    test_an = [a for a, sp in split_of(s0).items() if sp == "test"]
    assert test_an
    lock = ds.lock_test_set(d, s0.version)
    assert lock["snapshot"] == s0.version and lock_file_path(d).is_file() and lock["groups"]
    with pytest.raises(MLError, match="already exists"):
        ds.lock_test_set(d, s0.version)
    with pytest.raises(MLError, match="requires --reason"):
        ds.lock_test_set(d, s0.version, replace=True)
    test_files = {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in s0.manifest.crops if c.split == "test"}

    # later corrections on a locked slide + a brand-new replicate of a locked specimen
    from polony_app.review import engine
    with store.edit(test_an[0]) as live:
        engine.apply_action(live, "region_inspected", {"region_id": "f1", "inspected": False})
        engine.apply_action(live, "delete", {"object_ids": ["r1_2"]})
        engine.apply_action(live, "region_inspected", {"region_id": "f1", "inspected": True})
        engine.apply_action(live, "approve_region", {"region_id": "f1"})
    locked_spec = next(s for s in s0.manifest.sources if s.analysis_id == test_an[0]).identity["specimen_id"]
    newrep = add_analysis(store, "new-replicate", slide_id=f"{locked_spec}_C", seed=77)
    others = [a for a in ids.values() if a not in test_an]
    s1 = build(store, d, sorted(others + test_an + [newrep]), seed=9)
    assert s1.version != s0.version
    sp1 = split_of(s1)
    assert sp1[newrep] == "test"                                 # same specimen as a locked image -> test only
    assert not any(c.split == "test" and c.analysis_id in (newrep, *test_an) and not c.frozen_from for c in s1.manifest.crops)
    frozen = {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in s1.manifest.crops if c.split == "test"}
    assert frozen == test_files                                   # locked test labels did not change
    assert all(c.frozen_from == s0.version for c in s1.manifest.crops if c.split == "test")
    for c in s1.manifest.crops:
        if c.split in ("train", "val"):
            assert c.group_id not in lock["groups"]
    assert s1.manifest.test_lock["snapshot"] == s0.version
    assert any("LOCKED test set" in w for w in s1.manifest.warnings)


def test_training_refuses_data_from_locked_groups(tmp_path, monkeypatch):
    store, ids = make_world(tmp_path)
    d = tmp_path / "d"
    s_a = build(store, d, sorted(ids.values()), seed=1)
    # same slides, no test fraction: the future test slides land in train/val of this second snapshot
    s_b = build(store, d, sorted(ids.values()), seed=2, fractions={"train": 0.75, "val": 0.25, "test": 0.0})
    assert s_a.version != s_b.version
    test_a = {s.analysis_id for s in s_a.manifest.sources if s.split == "test"}
    train_b = {c.analysis_id for c in s_b.manifest.crops if c.split in ("train", "val")}
    assert test_a & train_b, "fixture: need a slide that is test in A but train/val in B"
    ds.lock_test_set(d, s_a.version)
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, dd: {"kind": "folder", "name": "x", "path": "/x", "registry_id": None})
    with pytest.raises(MLError, match="locked test"):
        plan_training(d, s_b.version, TrainOptions(base="x"))


def test_lock_requires_confirmed_identities_and_test_crops(tmp_path):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"29-09-2026_T7_AB_S{i}-A", seed=i) for i in range(1, 5)]
    r = build(store, tmp_path / "d", ids, allow_unconfirmed_identity=True)
    if any(c.split == "test" for c in r.manifest.crops):
        with pytest.raises(MLError, match="UNCONFIRMED"):
            ds.lock_test_set(tmp_path / "d", r.version)
    store2, ids2 = make_world(tmp_path / "w", n=1, replicate_of_first=False)
    r2 = build(store2, tmp_path / "w" / "d", list(ids2.values()))
    with pytest.raises(MLError, match="no test crops"):
        ds.lock_test_set(tmp_path / "w" / "d", r2.version)


def test_damaged_locked_snapshot_blocks_new_builds(tmp_path):
    store, ids = make_world(tmp_path)
    d = tmp_path / "d"
    s0 = build(store, d, sorted(ids.values()), seed=1)
    ds.lock_test_set(d, s0.version)
    import os
    c = next(c for c in s0.manifest.crops if c.split == "test")
    p = s0.path / c.path_csv
    os.chmod(p, 0o644)
    p.write_text("y,x\n1.0,1.0\n")
    with pytest.raises(MLError, match="damaged"):
        build(store, d, sorted(ids.values()), seed=2)


# ---------------------------------------------------------------------------- lock matching is identity-robust (review F7)
def _rec(specimen, slide=None, assay=None, sha="a" * 64):
    from polony_app.legacy_import.schema import ImageKind, ImageRecord
    return ImageRecord(image_id="i", file_name="f.tif", rel_path="f.tif", sha256=sha, size_bytes=1, kind=ImageKind.stack, width=1,
                       height=1, n_planes=1, dtype="uint16", bit_depth=16, specimen_id=specimen, slide_id=slide, assay=assay)


def _lock(specimen="AS", slide="AS_A", assay="T7_AB"):
    from polony_app.legacy_import.splits import group_key
    r = _rec(specimen, slide, assay, sha="b" * 64)
    return {"images": [{"sha256": r.sha256, "file_sha256s": [], "keys": {"specimen": group_key(r, "specimen"), "slide": group_key(r, "slide")}}],
            "groups": []}


@pytest.mark.parametrize("specimen,slide,assay", [("AS", "AS_C", "T7_AB"), ("AS", "AS_C", None), ("AS", "AS_C", "OTHER"),
                                                  ("as", "AS_C", "T7_AB"), (" As  ", "AS_C", None), ("XX", "as_a", "OTHER"),
                                                  (None, "AS_A", None)])
def test_locked_reason_ignores_assay_case_and_spacing(specimen, slide, assay):
    from polony_app.ml.splits import locked_reason
    assert locked_reason(_rec(specimen, slide, assay), _lock()) is not None


def test_locked_reason_does_not_lock_unrelated_identities():
    from polony_app.ml.splits import locked_reason
    assert locked_reason(_rec("AM", "AM_A", "T7_AB"), _lock()) is None
    assert locked_reason(_rec(None, None, "T7_AB"), _lock()) is None
    assert locked_reason(_rec("AS", "AS_A"), None) is None
