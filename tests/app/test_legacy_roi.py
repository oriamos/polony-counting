"""ImageJ ROI decoding: coordinate frame, orientation, counters/positions, grid, shapes, corrupt input."""
import numpy as np
import pytest
import roifile

from polony_app.legacy_import.roi import read_embedded, read_sidecar, parse_shape_path, to_canonical

from legacy_fixtures import (GRID_XS, GRID_YS, blank, grid_overlay, line_roi, point_roi, rect_roi, write_stack)


def codes(bundle):
    return {i.code for i in bundle.issues}


def test_subpixel_points_are_shifted_by_half_a_pixel_integer_points_are_not():
    assert to_canonical(10.5, subpixel=True) == 10.0      # ImageJ frame: centre of pixel 10 is 10.5
    assert to_canonical(10, subpixel=False) == 10.0       # integer ROI stores the pixel index itself


def test_point_roi_fields_counters_positions(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank(), blank()],
                    roi=point_roi([(20.5, 30.5), (70.5, 50.5), (71.5, 50.5)], [0, 1, 1], [1, 2, 1]))
    b = read_embedded(p)
    assert [(q.x, q.y, q.counter, q.position, q.subpixel) for q in b.points] == [
        (20.5, 30.5, 0, 1, True), (70.5, 50.5, 1, 2, True), (71.5, 50.5, 1, 1, True)]
    assert b.n_blobs == 1 and not b.issues


def test_integer_point_roi_is_flagged_not_subpixel(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank()], roi=point_roi([(20, 30)], [0], [1], subpixel=False))
    (pt,) = read_embedded(p).points
    assert (pt.x, pt.y, pt.subpixel) == (20.0, 30.0, False)


def test_points_without_counters_are_kept_with_info(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank()], roi=point_roi([(20.5, 30.5)]))
    b = read_embedded(p)
    assert b.points[0].counter is None and b.points[0].position is None
    assert "no_counters" in codes(b)


def test_grid_overlay_is_decoded_into_edges(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank()], overlays=[grid_overlay(GRID_XS, GRID_YS, 160, 120)])
    (g,) = read_embedded(p).grids
    assert g.xs == GRID_XS and g.ys == GRID_YS and g.regular
    assert g.spacing == (50.0, 35.0)


def test_irregular_grid_is_warned(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank()], overlays=[grid_overlay([10, 60, 130], GRID_YS, 160, 120)])
    b = read_embedded(p)
    assert "irregular_grid" in codes(b) and not b.grids[0].regular


def test_real_lab_overlay_matches_legacy_grid_numbers():
    from legacy_fixtures import REAL_TEST_IMG, load_test_img_truth
    truth = load_test_img_truth()
    b = read_embedded(REAL_TEST_IMG)
    (g,) = b.grids
    assert len(g.xs) == 5 and len(g.ys) == 8           # 4 x 7 cells
    assert g.xs[0] == 158.0 and g.ys[0] == 158.0
    assert g.spacing[0] == pytest.approx(316.23, abs=0.01)
    assert g.xs == pytest.approx(truth["grid"]["xs"], abs=1e-3) and g.ys == pytest.approx(truth["grid"]["ys"], abs=1e-3)
    assert len(b.points) == len(truth["marks"])
    assert [(p.x, p.y, p.counter, p.position) for p in b.points] == \
           [(m["x_imagej"], m["y_imagej"], m["counter"], m["position"]) for m in truth["marks"]]


def test_area_shapes_and_unsupported_types(tmp_path):
    p = write_stack(tmp_path / "a.tif", [blank()], roi=rect_roi(5, 6, 50, 60, "bad"),
                    overlays=[line_roi()])
    b = read_embedded(p)
    (sh,) = b.shapes
    assert sh.roi_type == "RECT" and sh.name == "bad" and sh.polygon[0] == (5.0, 6.0)
    assert "unsupported_roi_type" in codes(b)


def test_shape_path_parser_rejects_garbage():
    assert parse_shape_path(np.array([0, 1, 2, 1, 3, 4])) == [[(1.0, 2.0), (3.0, 4.0)]]
    with pytest.raises(ValueError):
        parse_shape_path(np.array([9, 1, 2]))
    with pytest.raises(ValueError):
        parse_shape_path(np.array([0, 1]))        # truncated


def test_corrupt_embedded_roi_is_an_error_issue_not_an_exception(tmp_path):
    good = point_roi([(20.5, 30.5)], [0], [1])
    p = write_stack(tmp_path / "a.tif", [blank()], roi=good[:30])      # truncated header
    b = read_embedded(p)
    assert b.points == [] and "roi_unparseable" in codes(b)
    assert any(i.severity == "error" for i in b.issues)


def test_corrupt_and_valid_sidecar_roi(tmp_path):
    (tmp_path / "bad.roi").write_bytes(b"not an roi at all")
    b = read_sidecar(tmp_path / "bad.roi")
    assert "roi_unparseable" in codes(b)
    (tmp_path / "good.roi").write_bytes(point_roi([(5.5, 6.5)], [3], [1]))
    g = read_sidecar(tmp_path / "good.roi")
    assert g.points[0].counter == 3 and g.points[0].roi_source == "sidecar"


def test_zip_roiset_is_read(tmp_path):
    rois = [roifile.ImagejRoi.frombytes(point_roi([(5.5, 6.5)], [0], [1])),
            roifile.ImagejRoi.frombytes(rect_roi(1, 1, 9, 9, "z"))]
    zp = tmp_path / "set.zip"
    roifile.roiwrite(str(zp), rois)
    b = read_sidecar(zp)
    assert len(b.points) == 1 and len(b.shapes) == 1


def test_non_imagej_tiff_has_no_rois(tmp_path):
    import tifffile
    tifffile.imwrite(str(tmp_path / "plain.tif"), blank())
    b = read_embedded(tmp_path / "plain.tif")
    assert b.n_blobs == 0 and not b.points and not b.issues
