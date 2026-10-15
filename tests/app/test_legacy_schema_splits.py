"""Canonical dataset round trip, leakage-safe deterministic splitting, exports and adapters."""
import json

import numpy as np
import pytest

from legacy_fixtures import (GRID_XS, GRID_YS, REAL_TEST_IMG, blank, grid_overlay, load_test_img_truth, point_roi,
                             write_stack)
from polony_app.legacy_import import ImportConfig, build_dataset, read_dataset, write_dataset
from polony_app.legacy_import.exports import (counted_field_polygons, export_spotiflow, legacy_squares,
                                             reference_points, score_detections, to_review_seed)
from polony_app.legacy_import.schema import (SCHEMA_VERSION, Dataset, ImageKind, ImageRecord,
                                             compute_dataset_id)
from polony_app.legacy_import.splits import (IdentityNotConfirmed, SplitError, assert_patch_assignment,
                                            assign_splits, split_of_patch, training_images, verify_no_leakage)
from polony_app.schemas import ObjectClass, Origin

OV = grid_overlay(GRID_XS, GRID_YS, 160, 120)


def rec(i, slide, specimen, *, reserved=None, kind=ImageKind.stack, confirmed=False, sha=None, comp_of=None, annotated=True):
    return ImageRecord(
        image_id=f"img_{i:012d}", file_name=f"f{i}.tif", rel_path=f"f{i}.tif", sha256=sha or f"{i:064x}", size_bytes=1,
        kind=kind, width=10, height=10, n_planes=2, dtype="uint16", bit_depth=16, assay="T7_AB",
        specimen_id=specimen, replicate=slide[-1] if slide else None, slide_id=slide,
        identity_source="filename_pattern" if slide else "unknown", identity_confirmed=confirmed,
        reserved_split=reserved, composite_of=comp_of, has_annotations=annotated)


def synthetic(n_specimens=10, confirmed=True):
    ims, i = [], 0
    for s in range(n_specimens):
        for r in "AB":
            ims.append(rec(i, f"S{s}_{r}", f"S{s}", confirmed=confirmed)); i += 1
    return Dataset(images=ims)


# ------------------------------------------------------------ round trip
def test_round_trip_through_canonical_format_is_exact(tmp_path):
    write_stack(tmp_path / "src" / "29-09-2026_T7_AB_AM-A.tif" if (tmp_path / "src").mkdir() is None else None,
                [blank(), blank()],
                roi=point_roi([(20.5, 20.5), (70.123, 50.5), (71.25, 50.75)], [0, 1, 1], [1, 2, 1]), overlays=[OV])
    ds, _ = build_dataset(tmp_path / "src", ImportConfig(created_at="2026-01-01T00:00:00Z", class_map={1: "x"}))
    out = write_dataset(ds, tmp_path / "ds")
    back = read_dataset(out)
    assert back.model_dump() == ds.model_dump()                           # bit-exact floats, enums, optionals, polygons
    assert back.manifest.schema_version == SCHEMA_VERSION and back.manifest.dataset_id == compute_dataset_id(back)
    assert (out / "manifest.json").is_file() and (out / "points.csv").read_text().splitlines()[0].startswith("label_id,")


def test_round_trip_of_an_empty_dataset_and_none_vs_zero(tmp_path):
    ds = Dataset()
    back = read_dataset(write_dataset(ds, tmp_path / "e"))
    assert back.images == [] and back.points == []


def test_write_refuses_to_overwrite_and_read_rejects_foreign_or_incomplete_dirs(tmp_path):
    ds = Dataset()
    write_dataset(ds, tmp_path / "d")
    with pytest.raises(FileExistsError):
        write_dataset(ds, tmp_path / "d")
    with pytest.raises(FileNotFoundError):
        read_dataset(tmp_path / "nothing")
    m = json.loads((tmp_path / "d" / "manifest.json").read_text())
    m["schema_version"] = 99
    (tmp_path / "d" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="schema version"):
        read_dataset(tmp_path / "d")


def test_real_image_survives_round_trip(tmp_path):
    ds, _ = build_dataset(REAL_TEST_IMG, ImportConfig(created_at="2026-01-01T00:00:00Z"))
    back = read_dataset(write_dataset(ds, tmp_path / "r"))
    n_marks = len(load_test_img_truth()["marks"])
    assert back.model_dump() == ds.model_dump() and len(back.points) == n_marks and len(back.fields) == 28


# ------------------------------------------------------------ splitting
def test_split_is_deterministic_and_seed_dependent():
    ds = synthetic()
    a1, a2 = assign_splits(ds, seed=3), assign_splits(ds, seed=3)
    assert a1.image_to_split == a2.image_to_split and a1.dumps() == a2.dumps()
    others = {tuple(sorted(assign_splits(ds, seed=s).image_to_split.items())) for s in range(8)}
    assert len(others) > 1


def test_replicate_slides_of_one_specimen_never_straddle_splits():
    ds = synthetic(12)
    for seed in range(10):
        a = assign_splits(ds, seed=seed)
        by_spec = {}
        for im in ds.images:
            by_spec.setdefault(im.specimen_id, set()).add(a.image_to_split[im.image_id])
        assert all(len(v) == 1 for v in by_spec.values()), seed
        assert verify_no_leakage(ds, a) == []
        assert {"train", "val", "test"} == set(a.image_to_split.values())


def test_slide_level_grouping_is_allowed_but_warns_about_specimen_straddling():
    ds = synthetic(6)
    saw = False
    for seed in range(20):
        a = assign_splits(ds, seed=seed, level="slide")
        assert verify_no_leakage(ds, a) == []                   # slides never straddle
        saw |= any("straddle" in w for w in a.warnings)
    assert saw


def test_patches_inherit_their_images_split():
    ds = synthetic()
    a = assign_splits(ds, seed=1)
    iid = ds.images[0].image_id
    sp = split_of_patch(iid, a)
    assert_patch_assignment([("p1", iid, sp), ("p2", iid, sp)], a)
    other = next(s for s in ("train", "val", "test") if s != sp)
    with pytest.raises(SplitError):
        assert_patch_assignment([("p1", iid, sp), ("p2", iid, other)], a)      # random-patch split would leak
    with pytest.raises(SplitError):
        split_of_patch("img_unknown", a)


def test_unconfirmed_identity_requires_explicit_acknowledgement():
    ds = synthetic(6, confirmed=False)
    with pytest.raises(IdentityNotConfirmed):
        assign_splits(ds, seed=0)
    a = assign_splits(ds, seed=0, allow_unconfirmed=True)
    assert a.unconfirmed_identity and any("UNCONFIRMED" in w for w in a.warnings)
    slides = {i.slide_id for i in ds.images}
    b = assign_splits(ds, seed=0, confirmed_slides=slides)               # researcher confirmed every slide
    assert not b.unconfirmed_identity
    with pytest.raises(IdentityNotConfirmed):
        assign_splits(ds, seed=0, confirmed_slides=set(list(slides)[:-1]))


def test_unknown_identity_images_are_isolated_with_a_warning():
    ds = synthetic(4)
    ds.images.append(rec(900, None, None, confirmed=True))
    a = assign_splits(ds, seed=0)
    assert any("no specimen/slide identity" in w for w in a.warnings)
    assert a.image_to_group["img_000000000900"].startswith("source_image:")


def test_reserved_test_image_and_its_group_are_pinned_to_test():
    ds = synthetic(8)
    ds.images[0].reserved_split = "test"
    ds.images[0].reserved_reason = "historical test"
    sib = ds.images[1]                                                   # same specimen, other replicate
    for seed in range(10):
        a = assign_splits(ds, seed=seed)
        assert a.image_to_split[ds.images[0].image_id] == "test" and a.image_to_split[sib.image_id] == "test"
    ds.images[1].reserved_split = "val"
    with pytest.raises(SplitError, match="reserved to different splits"):
        assign_splits(ds, seed=0)


def test_reserved_images_are_never_training_images():
    ds = synthetic(8)
    ds.images[0].reserved_split = "test"
    a = assign_splits(ds, seed=0)
    train = training_images(ds, a)
    assert ds.images[0] not in train and all(a.image_to_split[i.image_id] == "train" for i in train)


def test_composite_follows_its_raw_stack_and_is_not_training_data():
    ds = synthetic(5)
    st = ds.images[0]
    comp = rec(500, st.slide_id, st.specimen_id, kind=ImageKind.rgb_composite, confirmed=True, comp_of=st.image_id, annotated=False)
    ds.images.append(comp)
    a = assign_splits(ds, seed=2)
    assert a.image_to_split[comp.image_id] == a.image_to_split[st.image_id]
    assert comp not in training_images(ds, a)


def test_identical_files_in_different_slides_are_kept_together():
    ds = synthetic(6)
    ds.images[0].sha256 = ds.images[3].sha256 = "a" * 64          # same bytes, different slide/specimen labels
    for seed in range(10):
        a = assign_splits(ds, seed=seed)
        assert a.image_to_split[ds.images[0].image_id] == a.image_to_split[ds.images[3].image_id]
        assert verify_no_leakage(ds, a) == [] and any("byte-identical" in w for w in a.warnings)


def test_too_few_groups_warns_and_bad_fractions_raise():
    ds = synthetic(2)
    a = assign_splits(ds, seed=0)
    assert any("received no images" in w for w in a.warnings)
    with pytest.raises(SplitError):
        assign_splits(ds, fractions={"train": 0.5, "val": 0.2})
    with pytest.raises(SplitError):
        assign_splits(Dataset())


def test_verify_detects_a_manual_leak():
    ds = synthetic(4)
    a = assign_splits(ds, seed=0)
    i0, i1 = ds.images[0].image_id, ds.images[1].image_id           # same specimen
    a.image_to_split[i1] = "test" if a.image_to_split[i0] != "test" else "train"
    assert any("spans" in p for p in verify_no_leakage(ds, a))


# ------------------------------------------------------------ exports
@pytest.fixture()
def made(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    roi = point_roi([(20.5, 20.5), (30.5, 25.5), (70.5, 50.5)], [0, 0, 1], [1, 1, 1])
    write_stack(src / "29-09-2026_T7_AB_AM-A.tif", [blank(), blank()], roi=roi, overlays=[OV])
    ds, _ = build_dataset(src, ImportConfig(created_at="2026-01-01T00:00:00Z"))
    return ds, src


def test_scoring_ignores_detections_outside_counted_fields(made):
    ds, _ = made
    iid = ds.images[0].image_id
    ref = reference_points(ds, iid, 0)
    assert ref.shape == (3, 2) and len(counted_field_polygons(ds, iid, 0)) == 2
    det = np.vstack([ref + 1.0, [[140.0, 100.0]]])                    # 3 hits + 1 spot in an UNcounted cell
    s = score_detections(ds, iid, 0, det, tolerance_px=3.0)
    assert (s["tp"], s["fp"], s["fn"]) == (3, 0, 0) and s["n_detections_ignored_outside_counted_fields"] == 1
    s2 = score_detections(ds, iid, 0, np.array([[21.0, 21.0], [22.0, 21.0]]), tolerance_px=3.0)   # one-to-one matching
    assert s2["tp"] == 1 and s2["fp"] == 1 and s2["fn"] == 2
    assert score_detections(ds, iid, 0, np.zeros((0, 2)), 3.0)["recall"] == 0.0


def test_duplicates_and_out_of_bounds_never_enter_reference_points(tmp_path):
    src = tmp_path / "s"; src.mkdir()
    write_stack(src / "29-09-2026_T7_AB_AM-A.tif", [blank(), blank()],
                roi=point_roi([(20.5, 20.5), (20.5, 20.5), (400.0, 20.5)], [0, 0, 0], [1, 1, 1]), overlays=[OV])
    ds, _ = build_dataset(src)
    assert len(reference_points(ds, ds.images[0].image_id, 0)) == 1
    assert sorted(p.status.value for p in ds.points) == ["duplicate", "ok", "out_of_bounds"]


def test_spotiflow_export_is_explicit_and_keeps_test_out(made, tmp_path):
    ds, src = made
    ds.images[0].identity_confirmed = True
    a = assign_splits(ds, seed=0, fractions={"train": 1.0, "val": 0.0, "test": 0.0})
    with pytest.raises(SplitError):
        export_spotiflow(ds, tmp_path / "o", a, pages=[])
    m = export_spotiflow(ds, tmp_path / "o", a, pages=[0], source_root=src, write_planes=True)
    (e,) = m["entries"]
    csv_lines = (tmp_path / "o" / e["file"]).read_text().splitlines()
    assert csv_lines[0] == "y,x" and len(csv_lines) == 4 and csv_lines[1].startswith("20.0,20.0")
    assert (tmp_path / "o" / "train" / (e["image_id"] + "__p0.tif")).is_file()
    import tifffile
    assert tifffile.imread(str(tmp_path / "o" / "train" / (e["image_id"] + "__p0.tif"))).dtype == np.uint16
    # nothing for test unless requested; reserved images refuse other splits
    a2 = assign_splits(ds, seed=0, fractions={"train": 0.0, "val": 0.0, "test": 1.0})
    assert export_spotiflow(ds, tmp_path / "o2", a2, pages=[0])["entries"] == []
    ds.images[0].reserved_split = "test"
    a3 = assign_splits(ds, seed=0)
    assert a3.image_to_split[ds.images[0].image_id] == "test"
    assert export_spotiflow(ds, tmp_path / "o3", a3, pages=[0])["entries"] == []


def test_legacy_squares_follow_the_legacy_truncated_box_convention():
    truth = load_test_img_truth()
    ds, _ = build_dataset(REAL_TEST_IMG)
    sq = legacy_squares(ds, ds.images[0].image_id, page=0)
    assert len(sq) == 28 and sum(s["counted"] for s in sq) == len(truth["counted_cells"])
    assert sum(s["n_points"] for s in sq) == len(truth["marks"])
    first = min(sq, key=lambda s: (s["y"], s["x"]))
    assert (first["x"], first["y"], first["size"]) == (158, 158, 316)
    assert {s["size"] for s in sq} <= {315, 316, 317}


def test_review_seed_creates_unclassified_manual_objects(made):
    ds, _ = made
    regions, objs = to_review_seed(ds, ds.images[0].image_id, page=0, ts="T")
    assert len(regions) == 2 and all(r.field_index is None for r in regions)
    assert len(objs) == 3 and {o.cls for o in objs} == {ObjectClass.uncertain} and {o.origin for o in objs} == {Origin.manual}
    assert all(o.region_id in {r.id for r in regions} for o in objs)
