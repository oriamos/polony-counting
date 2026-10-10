"""One geometry rule everywhere (detection/masks.py): review area, membership and boundaries agree."""
import numpy as np
import pytest

from fakes import counting, exclusion, make_state, obj, rect
from polony_app.detection.masks import build_valid_mask, point_region, region_area_px, region_areas
from polony_app.review import engine
from polony_app.review.geometry import point_in_polygon, rasterize_polygon

SHAPE = (120, 160)


def A(*regions):
    return make_state([], list(regions)).analysis


def test_rect_10_to_20_has_area_100_everywhere():
    a = A(counting("A", rect(10, 10, 20, 20)))
    assert build_valid_mask(SHAPE, a.regions).sum() == 100
    assert engine.counting_area_px2(a) == 100.0
    assert engine.region_area_px2(a, a.regions[0]) == 100.0
    assert region_area_px(a.regions[0], SHAPE) == 100.0
    assert rasterize_polygon(a.regions[0].polygon, SHAPE).sum() == 100  # review.geometry delegates


@pytest.mark.parametrize("pt,inside", [((10, 15), True), ((20, 15), False), ((15, 20), False), ((15, 10), True),
                                       ((19.99, 15), True), ((9.99, 15), False)])
def test_edge_points_membership_matches_mask(pt, inside):
    a = A(counting("A", rect(10, 10, 20, 20)))
    assert (engine.point_region_id(a, *pt) == "A") is inside
    assert (point_region(pt[0], pt[1], a.regions) == "A") is inside
    assert point_in_polygon(pt[0], pt[1], a.regions[0].polygon) is inside
    if float(pt[0]).is_integer() and float(pt[1]).is_integer():
        assert bool(build_valid_mask(SHAPE, a.regions)[int(pt[1]), int(pt[0])]) is inside


def test_shared_edge_belongs_to_exactly_one_region_and_areas_add_up():
    a = A(counting("A", rect(0, 0, 10, 10)), counting("B", rect(10, 0, 20, 10)))
    assert engine.point_region_id(a, 10, 5) == "B" and engine.point_region_id(a, 9.99, 5) == "A"
    ar = [engine.region_area_px2(a, r) for r in a.regions]
    assert ar == [100.0, 100.0] and engine.counting_area_px2(a) == 200.0


def test_exclusion_edges_half_open():
    a = A(counting("A", rect(0, 0, 30, 30)), exclusion("X", rect(10, 10, 20, 20)))
    assert engine.point_region_id(a, 10, 15) is None      # low edge belongs to the exclusion
    assert engine.point_region_id(a, 20, 15) == "A"       # high edge does not
    assert engine.point_region_id(a, 15, 10) is None and engine.point_region_id(a, 15, 20) == "A"
    m = build_valid_mask(SHAPE, a.regions)
    assert (not m[15, 10]) and m[15, 20] and (not m[10, 15]) and m[20, 15]
    assert engine.counting_area_px2(a) == 800.0 and engine.region_area_px2(a, a.regions[0]) == 800.0
    assert engine.region_area_px2(a, a.regions[1]) == 100.0  # exclusion's own area


def test_overlapping_counting_regions_first_listed_wins_for_area_and_membership():
    a = A(counting("A", rect(0, 0, 20, 20)), counting("B", rect(10, 0, 30, 20)))
    areas = region_areas(a.regions, SHAPE)
    assert areas == {"A": 400.0, "B": 200.0}               # B is net of A (overlap 10x20 counted once)
    assert engine.counting_area_px2(a) == 600.0 == sum(areas.values())
    assert engine.region_area_px2(a, a.regions[1]) == 200.0
    assert engine.point_region_id(a, 15, 5) == "A"          # overlap -> first listed
    assert engine.point_region_id(a, 25, 5) == "B"
    # reversing the order moves the overlap to the other region; the union does not change
    b = A(counting("B", rect(10, 0, 30, 20)), counting("A", rect(0, 0, 20, 20)))
    assert region_areas(b.regions, SHAPE) == {"B": 400.0, "A": 200.0}
    assert engine.point_region_id(b, 15, 5) == "B"


def test_non_integer_grid_cell_area_equals_mask_pixels():
    g = [(158.0, 40.0), (474.23, 40.0), (474.23, 100.0), (158.0, 100.0)]
    a = A(counting("A", g))
    a.image.width, a.image.height = 600, 120
    m = build_valid_mask((120, 600), a.regions)
    assert engine.counting_area_px2(a) == float(m.sum()) == 317 * 60
