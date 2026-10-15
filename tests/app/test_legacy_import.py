"""End-to-end conversion of historical files: TIFF loading, pairing, labels, warnings, dry run."""
import shutil
import zipfile

import numpy as np
import pytest
import tifffile

from legacy_fixtures import (GRID_XS, GRID_YS, REAL_TEST_IMG, blank, grid_overlay, load_test_img_truth, point_roi,
                             rect_roi, textured, write_stack)
from polony_app.legacy_import import ImportConfig, build_dataset
from polony_app.legacy_import.cli import main as cli_main
from polony_app.legacy_import.identity import parse_identity
from polony_app.legacy_import.schema import FieldStatus, ImageKind, PointStatus

NAME = "29-09-2026_T7_AB_AM-A.tif"
OV = grid_overlay(GRID_XS, GRID_YS, 160, 120)


def codes(ds):
    return {w.code for w in ds.warnings}


def one(tmp_path, planes=None, roi=None, overlays=(OV,), name=NAME, **kw):
    planes = planes if planes is not None else [blank(), blank()]
    write_stack(tmp_path / name, planes, roi=roi, overlays=list(overlays) if overlays else None, **kw)
    return build_dataset(tmp_path)


# ------------------------------------------------------------ TIFF loading
def test_16bit_two_page_stack_keeps_values_and_dtype(tmp_path):
    a, b = textured(120, 160, 1), textured(120, 160, 2)
    ds, rep = one(tmp_path, [a, b])
    (im,) = ds.images
    assert (im.kind, im.dtype, im.bit_depth, im.n_planes, im.height, im.width) == (ImageKind.stack, "uint16", 16, 2, 120, 160)
    assert [p.max for p in im.planes] == [int(a.max()), int(b.max())] and im.planes[0].max > 255   # not squeezed to 8 bit
    assert im.sha256 == __import__("hashlib").sha256((tmp_path / NAME).read_bytes()).hexdigest()
    assert rep.images_loaded == 1 and not rep.images_failed


def test_8bit_stack_is_accepted_with_a_warning(tmp_path):
    ds, _ = one(tmp_path, [blank(dtype=np.uint8, value=7), blank(dtype=np.uint8, value=9)])
    assert ds.images[0].dtype == "uint8" and ds.images[0].bit_depth == 8
    assert "eight_bit_data" in codes(ds)


def test_single_plane_and_three_plane_stacks(tmp_path):
    ds, _ = one(tmp_path, [blank()], name="01-01-2026_T7_AB_AA-A.tif")
    assert ds.images[0].kind == ImageKind.single_plane and ds.images[0].n_planes == 1
    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    ds, _ = one(tmp_path, [blank(), blank(), blank()])
    assert ds.images[0].n_planes == 3 and ds.images[0].kind == ImageKind.stack


def test_float_tiff_is_rejected_not_converted(tmp_path):
    tifffile.imwrite(str(tmp_path / "f.tif"), np.zeros((20, 20), np.float32))
    ds, rep = build_dataset(tmp_path)
    assert ds.images == [] and rep.images_failed == ["f.tif"]
    assert {w.code for w in rep.warnings} == {"unsupported_dtype"} and rep.n_errors == 1


def test_corrupt_tiff_fails_alone_other_files_still_import(tmp_path):
    (tmp_path / "broken.tif").write_bytes(b"II*\x00" + b"\x00" * 50)
    write_stack(tmp_path / NAME, [blank(), blank()], roi=point_roi([(20.5, 20.5)], [0], [1]))
    ds, rep = build_dataset(tmp_path)
    assert [i.file_name for i in ds.images] == [NAME] and rep.images_failed == ["broken.tif"]
    assert "image_unreadable" in {w.code for w in rep.warnings}


def test_archives_are_reported_never_extracted(tmp_path):
    with zipfile.ZipFile(tmp_path / "x.tif.zip", "w") as z:
        z.writestr("x.tif", b"zzz")
    ds, rep = build_dataset(tmp_path)
    assert rep.skipped_archives == ["x.tif.zip"] and not (tmp_path / "x.tif").exists()


def test_byte_identical_copy_is_an_alias_not_a_second_image(tmp_path):
    write_stack(tmp_path / NAME, [blank(), blank()], roi=point_roi([(20.5, 20.5)], [0], [1]))
    shutil.copy(tmp_path / NAME, tmp_path / "copy.tif")
    ds, _ = build_dataset(tmp_path)
    assert len(ds.images) == 1 and ds.images[0].alias_paths == ["copy.tif"] and "duplicate_file" in codes(ds)


# ------------------------------------------------------------ pairing / channels
def _pair(tmp_path, comp_shape=None):
    p0, p1 = textured(120, 160, 3), textured(120, 160, 4)
    write_stack(tmp_path / NAME, [p0, p1], roi=point_roi([(20.5, 20.5)], [0], [1]), overlays=[OV])
    rgb = np.zeros(comp_shape or (120, 160, 3), np.uint8)
    h, w = rgb.shape[:2]
    rgb[..., 0] = (p0[:h, :w] >> 8).astype(np.uint8)     # red plane renders page 0
    rgb[..., 1] = (p1[:h, :w] >> 8).astype(np.uint8)     # green plane renders page 1
    tifffile.imwrite(str(tmp_path / "29-09-2026_T7_AB_AM-A_(635_532).tif"), rgb, photometric="rgb")
    return build_dataset(tmp_path)


def test_rgb_composite_is_paired_with_its_stack_and_red_green_evidence_recorded(tmp_path):
    ds, rep = _pair(tmp_path)
    stack = next(i for i in ds.images if i.kind == ImageKind.stack)
    comp = next(i for i in ds.images if i.kind == ImageKind.rgb_composite)
    assert comp.composite_of == stack.image_id and comp.slide_id == stack.slide_id == "AM_A"
    ev = comp.composite_evidence
    assert ev["rgbR_vs_page0"] > 0.95 > 0.5 > ev["rgbR_vs_page1"]       # red plane <-> page 0
    assert ev["rgbG_vs_page1"] > 0.95 > 0.5 > ev["rgbG_vs_page0"]       # green plane <-> page 1
    assert comp.has_annotations is False and not rep.pairing_issues


def test_composite_with_other_size_is_not_paired(tmp_path):
    ds, rep = _pair(tmp_path, comp_shape=(100, 140, 3))
    comp = next(i for i in ds.images if i.kind == ImageKind.rgb_composite)
    assert comp.composite_of is None and "composite_size_mismatch" in codes(ds) and rep.pairing_issues


def test_composite_without_stack_is_reported(tmp_path):
    tifffile.imwrite(str(tmp_path / "29-09-2026_T7_AB_ZZ-A_(635_532).tif"), np.zeros((50, 60, 3), np.uint8), photometric="rgb")
    ds, rep = build_dataset(tmp_path)
    assert "unpaired_composite" in codes(ds)


def test_two_raw_images_claiming_one_slide_are_ambiguous(tmp_path):
    write_stack(tmp_path / NAME, [blank(), blank()])
    write_stack(tmp_path / "30-09-2026_T7_AB_AM-A.tif", [blank(value=5), blank(value=6)])
    ds, rep = build_dataset(tmp_path)
    assert "ambiguous_slide_images" in codes(ds)


def test_identity_parsing_variants():
    a = parse_identity("29-09-2026_T7_AB_AM-A")
    assert (a.assay, a.specimen_id, a.replicate, a.slide_id, a.file_date) == ("T7_AB", "AM", "A", "AM_A", "29-09-2026")
    b = parse_identity("29-09-2026_T7_AB_FS-B_D")
    assert (b.slide_id, b.suffix) == ("FS_B", "_D")
    c = parse_identity("29-09-2026_T7_AB_AS-B_(635_532)")
    assert c.slide_id == "AS_B" and c.composite_wavelengths == (635, 532) and c.base_stem.endswith("AS-B")
    d = parse_identity("3-43-B(1)")
    assert d.slide_id == "3-43_B" and d.suffix == "(1)"
    assert parse_identity("test_img").source == "unknown"


# ------------------------------------------------------------ labels
def test_orientation_x_is_column_y_is_row(tmp_path):
    img = blank()
    img[50, 20] = 9999                       # row 50, column 20
    ds, _ = one(tmp_path, [img, blank()], roi=point_roi([(20.5, 50.5)], [0], [1]))
    (p,) = ds.points
    assert (p.x, p.y) == (20.0, 50.0)                                 # canonical: x=col, y=row, centre-of-pixel integer
    assert (p.x_imagej, p.y_imagej) == (20.5, 50.5)                   # raw preserved
    arr = tifffile.imread(str(tmp_path / NAME))
    assert arr[0][int(round(p.y)), int(round(p.x))] == 9999


def test_planes_counters_fields_and_no_class_unless_mapped(tmp_path):
    roi = point_roi([(20.5, 20.5), (30.5, 25.5), (70.5, 50.5)], [0, 0, 1], [1, 1, 2])
    ds, _ = one(tmp_path, roi=roi)
    assert [(p.page, p.position, p.counter) for p in ds.points] == [(0, 1, 0), (0, 1, 0), (1, 2, 1)]
    assert all(p.class_label is None for p in ds.points)
    f = {x.grid_row * 3 + x.grid_col: x for x in ds.fields}
    assert len(ds.fields) == 9 and f[0].counters == [0] and f[4].counters == [1]
    assert f[0].n_points_by_page == {"0": 2} and f[4].n_points_by_page == {"1": 1}
    assert [x.status for x in ds.fields].count(FieldStatus.has_points) == 2
    assert f[0].polygon[0] == (9.5, 9.5)                              # ImageJ edge 10 -> canonical 9.5


def test_explicit_class_map_is_applied_only_when_given(tmp_path):
    roi = point_roi([(20.5, 20.5), (70.5, 50.5)], [0, 1], [1, 2])
    write_stack(tmp_path / NAME, [blank(), blank()], roi=roi, overlays=[OV])
    ds, _ = build_dataset(tmp_path, ImportConfig(class_map={1: "clade_B"}))
    assert [p.class_label for p in ds.points] == ["clade_B", None]
    assert all(p.class_confirmed is False for p in ds.points)


def test_out_of_bounds_points_are_flagged_never_clamped(tmp_path):
    roi = point_roi([(20.5, 20.5), (160.0, 50.5), (-1.5, 5.5), (70.5, 130.5)], [0, 0, 0, 0], [1, 1, 1, 1])
    ds, rep = one(tmp_path, roi=roi)
    st = [p.status for p in ds.points]
    assert st == [PointStatus.ok, PointStatus.out_of_bounds, PointStatus.out_of_bounds, PointStatus.out_of_bounds]
    assert (ds.points[1].x_imagej, ds.points[2].x_imagej) == (160.0, -1.5)        # untouched
    assert "points_out_of_bounds" in codes(ds) and rep.labels_not_ok == {"out_of_bounds": 3}


def test_duplicates_and_close_pairs(tmp_path):
    roi = point_roi([(20.5, 20.5), (20.5, 20.5), (21.5, 20.5), (22.5, 20.5), (20.5, 20.5)], [0] * 5, [1, 1, 1, 1, 2])
    ds, _ = one(tmp_path, roi=roi)
    assert [p.status for p in ds.points] == [PointStatus.ok, PointStatus.duplicate, PointStatus.duplicate, PointStatus.ok, PointStatus.ok]
    assert "duplicate_points" in codes(ds) and "overlapping_points" in codes(ds)   # 20.5 vs 22.5: 2px apart
    # the same coordinates on another plane are NOT duplicates
    assert ds.points[4].status == PointStatus.ok


def test_empty_annotations_and_missing_grid(tmp_path):
    ds, rep = one(tmp_path, overlays=None)                    # no ROI at all
    assert "no_annotations" in codes(ds) and ds.points == [] and ds.images[0].has_annotations is False
    shutil.rmtree(tmp_path); tmp_path.mkdir()
    ds, _ = one(tmp_path, roi=point_roi([(20.5, 20.5)], [0], [1]), overlays=None)
    assert "no_grid" in codes(ds) and ds.fields == [] and ds.points[0].field_id is None


def test_points_outside_every_cell_are_warned(tmp_path):
    ds, _ = one(tmp_path, roi=point_roi([(5.5, 5.5), (20.5, 20.5)], [0, 0], [1, 1]))     # x,y<10: left of the grid
    assert "points_outside_grid" in codes(ds) and ds.points[0].field_id is None and ds.points[1].field_id


def test_counter_spanning_two_cells_is_noted(tmp_path):
    ds, _ = one(tmp_path, roi=point_roi([(20.5, 20.5), (70.5, 20.5)], [4, 4], [1, 1]))
    assert "counter_spans_cells" in codes(ds)
    assert sum(1 for f in ds.fields if 4 in f.counters) == 1


def test_regions_unclassified_by_default_excluded_only_by_explicit_name(tmp_path):
    write_stack(tmp_path / NAME, [blank(), blank()], roi=rect_roi(15, 15, 40, 40, "dust"), overlays=[OV])
    ds, _ = build_dataset(tmp_path)
    assert len(ds.regions) == 1 and ds.regions[0].role is None and "unclassified_regions" in codes(ds)
    ds2, rep2 = build_dataset(tmp_path, ImportConfig(excluded_region_names=("dust",)))
    assert ds2.regions[0].role == "excluded" and rep2.excluded_regions == 1
    assert ds2.regions[0].polygon[0] == (14.5, 14.5)


def test_sidecar_roi_next_to_a_plain_tiff(tmp_path):
    tifffile.imwrite(str(tmp_path / NAME), np.stack([blank(), blank()]))
    (tmp_path / (NAME[:-4] + ".roi")).write_bytes(point_roi([(20.5, 20.5)], [0], [1]))
    ds, rep = build_dataset(tmp_path)
    assert len(ds.points) == 1 and ds.points[0].roi_source == "sidecar" and ds.points[0].source_file.endswith(".roi")
    (tmp_path / "orphan.roi").write_bytes(point_roi([(1.5, 1.5)]))
    ds, rep = build_dataset(tmp_path)
    assert "orphan_roi_file" in {w.code for w in rep.warnings}


def test_corrupt_roi_inside_an_otherwise_good_image(tmp_path):
    ds, rep = one(tmp_path, roi=point_roi([(20.5, 20.5)], [0], [1])[:25])
    assert ds.points == [] and rep.n_errors == 1 and ds.images[0].n_roi_blobs == 2


# ------------------------------------------------------------ CSV evidence
def _csv(tmp_path, row):
    hdr = "slide ID,SAMPLING DATE,POLONY DATE ,CONCENTRATION FACTOR,SAMPLE VOLUM[ul],field [Pixel^2],grid [Pixel^2],QC,FAGE ABUNDENCE [fage mL-1]," + ",".join(f"field {i}" for i in range(1, 25))
    (tmp_path / "T7B_15-09-2026.csv").write_text(hdr + "\n" + row + "\n")


def test_csv_reconciliation_exact_blank_and_mismatch(tmp_path):
    pts = [(20.5, 20.5)] * 1 + [(30.5, 22.5), (70.5, 50.5)]        # counter0: 2 pts, counter2: 1 pt (no counter 1)
    roi = point_roi(pts, [0, 0, 2], [1, 1, 1])
    write_stack(tmp_path / NAME, [blank(), blank()], roi=roi, overlays=[OV])
    _csv(tmp_path, "AM_A,15/09/2026,29/09/2026,1,2.5,2.9,0.1,1,1," + "2,0,1" + ",,,,,,,,,,,,,,,,,,,,,")
    ds, _ = build_dataset(tmp_path)
    (rec,) = ds.reconciliations
    assert rec.series == "T7B" and rec.csv_total == 3 and rec.csv_counted_fields == 3
    assert rec.planes[0].match == "exact_by_counter_index" and rec.planes[1].match in ("mismatch", "total_only")
    # a zero-count field exists only in the CSV: a CSV row that disagrees is reported, nothing is changed
    _csv(tmp_path, "AM_A,15/09/2026,29/09/2026,1,2.5,2.9,0.1,1,1," + "2,1,1" + ",,,,,,,,,,,,,,,,,,,,,")
    ds2, _ = build_dataset(tmp_path)
    assert ds2.reconciliations[0].planes[0].match in ("mismatch", "total_only")     # reported, not "fixed"
    assert [p.status for p in ds2.points] == [p.status for p in ds.points]


def test_csv_row_without_image_and_image_without_row(tmp_path):
    write_stack(tmp_path / NAME, [blank(), blank()], roi=point_roi([(20.5, 20.5)], [0], [1]), overlays=[OV])
    _csv(tmp_path, "ZZ_A,15/09/2026,29/09/2026,1,2.5,2.9,0.1,1,1,1," + ",".join([""] * 23))
    ds, _ = build_dataset(tmp_path)
    assert {"csv_rows_without_image", "image_without_csv_row"} <= codes(ds) and ds.reconciliations == []


# ------------------------------------------------------------ example file (synthetic, lab structure)
def test_real_lab_image_is_converted_as_the_audit_describes():
    """The example slide has the lab files' structure; expected values come from its generator's truth."""
    truth = load_test_img_truth()
    n_marks, n_cells = len(truth["marks"]), len(truth["counted_cells"])
    ds, rep = build_dataset(REAL_TEST_IMG)
    (im,) = ds.images
    assert (im.dtype, im.n_planes, im.height, im.width) == ("uint16", 2, 2450, 1438)
    assert im.planes[0].label_hint == "635 [W1]" and im.planes[1].label_hint is None
    assert len(ds.points) == n_marks and {p.position for p in ds.points} == {1} and {p.page for p in ds.points} == {0}
    assert len(ds.fields) == 28 and sum(f.status == FieldStatus.has_points for f in ds.fields) == n_cells
    assert sorted({p.counter for p in ds.points}) == list(range(n_cells))
    # every counter sits in exactly one cell, and it is the cell the generator counted under that counter
    assert all(len(f.counters) <= 1 for f in ds.fields) and sum(len(f.counters) for f in ds.fields) == n_cells
    assert "counter_spans_cells" not in codes(ds)
    cell_of = {c["counter"]: (c["row"], c["col"]) for c in truth["counted_cells"]}
    assert {f.counters[0]: (f.grid_row, f.grid_col) for f in ds.fields if f.counters} == cell_of
    # canonical = ImageJ - 0.5, compared with an independent read
    import roifile
    ref = roifile.roiread(str(REAL_TEST_IMG))[1].subpixel_coordinates.astype(float) - 0.5
    np.testing.assert_allclose([[p.x, p.y] for p in ds.points], ref)
    assert im.reserved_split == "test"                                    # historical test image
    assert rep.n_errors == 0


# ------------------------------------------------------------ dry run / commit
def test_dry_run_writes_nothing_and_commit_is_explicit(tmp_path, capsys):
    src = tmp_path / "src"; src.mkdir()
    write_stack(src / NAME, [blank(), blank()], roi=point_roi([(20.5, 20.5)], [0], [1]), overlays=[OV])
    out = tmp_path / "out"
    assert cli_main(["import", str(src), "--out", str(out), "--created-at", "2026-01-01T00:00:00Z"]) == 0
    assert not out.exists() and "DRY RUN" in capsys.readouterr().out
    assert cli_main(["inspect", str(src)]) == 0 and not out.exists()
    assert cli_main(["import", str(src), "--out", str(out), "--commit", "--created-at", "2026-01-01T00:00:00Z"]) == 0
    assert (out / "manifest.json").is_file() and (out / "points.csv").is_file()
    with pytest.raises(FileExistsError):
        cli_main(["import", str(src), "--out", str(out), "--commit"])
    # the source tree was not modified
    assert sorted(p.name for p in src.iterdir()) == [NAME]


def test_commit_is_refused_when_files_failed(tmp_path, capsys):
    src = tmp_path / "src"; src.mkdir()
    (src / "bad.tif").write_bytes(b"garbage")
    out = tmp_path / "out"
    assert cli_main(["import", str(src), "--out", str(out), "--commit"]) == 2 and not out.exists()
    assert cli_main(["import", str(src), "--out", str(out), "--commit", "--allow-errors"]) == 0 and out.exists()


def test_import_is_reproducible(tmp_path):
    write_stack(tmp_path / NAME, [textured(120, 160, 1), blank()], roi=point_roi([(20.5, 20.5), (70.5, 50.5)], [0, 1], [1, 1]), overlays=[OV])
    cfg = ImportConfig(created_at="2026-01-01T00:00:00Z")
    a, _ = build_dataset(tmp_path, cfg)
    b, _ = build_dataset(tmp_path, cfg)
    assert a.model_dump() == b.model_dump() and a.manifest.dataset_id == b.manifest.dataset_id
