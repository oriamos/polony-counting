import numpy as np
from skimage.draw import polygon as sk_polygon

from polony_app.detection.masks import build_valid_mask, point_region, points_regions, region_area_px
from polony_app.schemas import Region, RegionKind


def reg(i, kind, poly):
    return Region(id=i, kind=kind, name=i, polygon=poly)


RECT = [(10, 10), (30, 10), (30, 20), (10, 20)]


def test_rectangle_half_open():
    m = build_valid_mask((40, 50), [reg("c", RegionKind.counting, RECT)])
    assert m.sum() == 20 * 10
    assert m[10, 10] and m[19, 29] and not m[20, 29] and not m[19, 30]


def test_no_counting_means_whole_image_minus_exclusions():
    m = build_valid_mask((40, 50), [reg("e", RegionKind.exclusion, RECT)])
    assert m.sum() == 40 * 50 - 200 and not m[15, 15]
    assert build_valid_mask((5, 6), []).all()


def test_exclusion_and_union():
    regs = [reg("a", RegionKind.counting, RECT), reg("b", RegionKind.counting, [(25, 10), (45, 10), (45, 20), (25, 20)]),
            reg("x", RegionKind.exclusion, [(0, 12), (50, 12), (50, 14), (0, 14)])]
    m = build_valid_mask((40, 50), regs)
    assert not m[12:14].any() and m[10, 40] and m[15, 12]
    assert point_region(12, 15, regs) == "a" and point_region(40, 15, regs) == "b"
    assert point_region(12, 13, regs) is None and point_region(5, 5, regs) is None
    assert point_region(1, 1, [reg("e", RegionKind.exclusion, RECT)]) is None


def test_mask_agrees_with_point_region_and_skimage_interior():
    rng = np.random.default_rng(0)
    for _ in range(5):
        poly = [tuple(p) for p in rng.uniform(5, 95, (7, 2))]
        r = reg("p", RegionKind.counting, poly)
        m = build_valid_mask((100, 100), [r])
        ys, xs = np.mgrid[:100, :100]
        ids = points_regions(xs.ravel(), ys.ravel(), [r])
        assert np.array_equal(np.array([i == "p" for i in ids]).reshape(100, 100), m)
        # skimage polygon fills the same area up to boundary pixels (self-intersections use even-odd too)
        sk = np.zeros((100, 100), bool)
        rr, cc = sk_polygon([p[1] for p in poly], [p[0] for p in poly], (100, 100))
        sk[rr, cc] = True
        assert (m != sk).sum() < 0.08 * max(m.sum(), 1) + 40


def test_triangle_area_and_helpers():
    tri = reg("t", RegionKind.counting, [(0, 0), (100, 0), (0, 100)])
    assert abs(region_area_px(tri, (200, 200)) - 5000) < 150
    assert region_area_px([reg("a", RegionKind.counting, RECT), reg("e", RegionKind.exclusion, [(10, 10), (20, 10), (20, 20), (10, 20)])],
                          (40, 50)) == 100
    assert region_area_px(reg("d", RegionKind.counting, [(1, 1), (2, 2)]), (10, 10)) == 0
    assert region_area_px(reg("c", RegionKind.counting, [(-50, -50), (500, -50), (500, 500), (-50, 500)]), (10, 12)) == 120
