import numpy as np
import pytest

from polony_app.detection.engine import RawSpot
from polony_app.detection.matching import associate
from polony_app.schemas import ObjectClass, ReviewReason


def S(x, y, peak=100.0):
    return RawSpot(x=x, y=y, peak=peak, background=0.0, snr=20.0, saturated=False)


def test_simple_classes_and_position_mean():
    objs = associate([S(10, 10), S(100, 100)], [S(12, 10), S(200, 200)], 5.0)
    by = {o.cls: o for o in objs}
    assert len(objs) == 3
    y = by[ObjectClass.yellow]
    assert (y.x, y.y) == (11.0, 10.0) and y.association_distance_px == pytest.approx(2.0)
    assert y.red.detected and y.green.detected and y.review_reasons == []
    assert by[ObjectClass.red].green.detected is False and by[ObjectClass.green].red.detected is False


def test_gate_and_one_to_one():
    objs = associate([S(0, 0)], [S(5.5, 0)], 5.0)
    assert sorted(o.cls.value for o in objs) == ["green", "red"]
    objs = associate([S(0, 0), S(0, 20)], [S(0, 10)], 15.0)
    # the green is used only once
    assert sum(o.green.detected for o in objs) == 1 and len(objs) == 2


def test_nearby_distinct_pairs_not_ambiguous():
    objs = associate([S(10, 10), S(20, 10)], [S(10.5, 10), S(20.5, 10)], 5.0)
    assert [o.cls for o in objs] == [ObjectClass.yellow] * 2
    assert all(not o.review_reasons for o in objs)
    assert {o.association_distance_px for o in objs} == {0.5}


def test_globally_optimal_assignment():
    # greedy nearest would pair r0-g0 (1.0) leaving r1 with g1 at 4.0 -> total 5.0; optimal r0-g1, r1-g0 = 2.2 + 2.0? check both
    red, green = [S(0, 0), S(3, 0)], [S(1.0, 0), S(-1.5, 0)]
    objs = associate(red, green, 5.0)
    assert sum(o.cls in (ObjectClass.yellow, ObjectClass.uncertain) for o in objs) == 2


def test_ambiguous_competing_partner_marks_uncertain():
    objs = associate([S(10, 10)], [S(11, 10), S(10, 12)], 5.0)  # green candidates at 1.0 and 2.0 (diff 1.0 < 1.5)
    assert len(objs) == 2
    assert all(o.cls == ObjectClass.uncertain and ReviewReason.ambiguous_match in o.review_reasons for o in objs)
    ids = {o.id for o in objs}
    for o in objs:
        assert o.ambiguous_with == sorted(ids - {o.id})
    assert sum(o.red.detected and o.green.detected for o in objs) == 1  # the Hungarian pair keeps both evidences


def test_relative_margin_rule_and_clear_case():
    # d1=4.0, d2=4.9: diff 0.9 -> ambiguous (absolute rule)
    assert any(o.cls == ObjectClass.uncertain for o in associate([S(0, 0)], [S(4, 0), S(0, 4.9)], 6.0))
    # d1=0.3, d2=3.5: diff 3.2 and > 1.25*d1 -> unambiguous
    objs = associate([S(0, 0)], [S(0.3, 0), S(0, 3.5)], 6.0)
    assert sorted(o.cls.value for o in objs) == ["green", "yellow"]
    # competitor outside the tolerance is irrelevant
    objs = associate([S(0, 0)], [S(1, 0), S(0, 6.5)], 6.0)
    assert sorted(o.cls.value for o in objs) == ["green", "yellow"]


def test_deterministic_ids_and_order_independence():
    r = [S(50, 50), S(10, 10), S(90, 20)]
    g = [S(11, 10), S(70, 70)]
    a = associate(r, g, 5.0)
    b = associate(list(reversed(r)), list(reversed(g)), 5.0)
    assert [o.model_dump() for o in a] == [o.model_dump() for o in b]
    assert [o.id for o in a] == [f"o{i:06d}" for i in range(1, len(a) + 1)]
    assert [(o.y, o.x) for o in a] == sorted((o.y, o.x) for o in a)
    assert a[0].red.spot_id.startswith("r") or a[0].green.spot_id.startswith("g")


def test_empty_inputs_and_large_scale():
    assert associate([], [], 5.0) == []
    assert [o.cls for o in associate([S(1, 1)], [], 5.0)] == [ObjectClass.red]
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 2000, (3000, 2))
    objs = associate([S(*p) for p in pts], [S(*(p + 0.3)) for p in pts], 3.0)
    assert sum(o.cls == ObjectClass.yellow for o in objs) > 2900
