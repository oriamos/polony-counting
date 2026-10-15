import random

import numpy as np
import pytest

from polony_app.consensus import match_engines, summarize
from polony_app.schemas import ConsensusStatus as CS
from polony_app.schemas import EngineName, RawDetection


def det(eng, i, x, y, page=0, role="red", valid=True):
    p = "bf" if eng == EngineName.bigfish else "sf"
    return RawDetection(id=f"{p}-p{page}-{i:06d}", run_id="r", analysis_id="a", engine=eng, page=page,
                        channel_role=role, x=x, y=y, valid=valid)


def bf(i, x, y, **k):
    return det(EngineName.bigfish, i, x, y, **k)


def sf(i, x, y, **k):
    return det(EngineName.spotiflow, i, x, y, **k)


def ids(pairs):
    return [(p.status.value, p.bigfish and p.bigfish.id, p.spotiflow and p.spotiflow.id,
             p.distance_px, tuple(p.ambiguous_with)) for p in pairs]


def test_classification_and_distance_preserved_no_averaging():
    pairs = match_engines([bf(1, 10, 10), bf(2, 50, 50)], [sf(1, 10.6, 10.8), sf(2, 90, 90)], 4.0)
    by = {p.status: p for p in pairs}
    a = by[CS.agreement]
    assert a.distance_px == pytest.approx(1.0) and (a.bigfish.x, a.bigfish.y) == (10, 10)
    assert (a.spotiflow.x, a.spotiflow.y) == (10.6, 10.8)          # each side keeps its own position
    assert by[CS.bigfish_only].bigfish.id == "bf-p0-000002" and by[CS.bigfish_only].spotiflow is None
    assert by[CS.spotiflow_only].spotiflow.id == "sf-p0-000002" and by[CS.spotiflow_only].distance_px is None
    assert len(pairs) == 3


def test_tolerance_edge_inclusive():
    assert match_engines([bf(1, 0, 0)], [sf(1, 4.0, 0)], 4.0)[0].status == CS.agreement
    p = match_engines([bf(1, 0, 0)], [sf(1, 4.0001, 0)], 4.0)
    assert sorted(x.status.value for x in p) == ["bigfish_only", "spotiflow_only"]
    assert match_engines([bf(1, 0, 0)], [sf(1, 0, 0)], 0.0)[0].status == CS.agreement


def test_one_to_one_no_duplicate_matching():
    # one bigfish spot between two spotiflow spots: only the closest pairs, the other stays spotiflow_only
    pairs = match_engines([bf(1, 10, 10)], [sf(1, 12, 10), sf(2, 8.5, 10)], 4.0)
    ag = [p for p in pairs if p.status == CS.agreement]
    assert len(ag) == 1 and ag[0].spotiflow.id == "sf-p0-000002" and ag[0].distance_px == pytest.approx(1.5)
    used = [p.spotiflow.id for p in pairs if p.spotiflow] + [p.bigfish.id for p in pairs if p.bigfish]
    assert len(used) == len(set(used)) == 3


def test_max_cardinality_then_min_distance():
    # greedy nearest would pair b1-s1 (1.0) and leave b2 alone; the assignment pairs both (b1-s2, b2-s1)
    pairs = match_engines([bf(1, 0, 0), bf(2, 3.5, 0)], [sf(1, 1.0, 0), sf(2, -2.5, 0)], 4.0)
    assert [p.status for p in pairs] == [CS.agreement, CS.agreement]
    assert sorted((p.bigfish.id, p.spotiflow.id) for p in pairs) == [("bf-p0-000001", "sf-p0-000002"),
                                                                     ("bf-p0-000002", "sf-p0-000001")]


def test_ambiguity_flags_competing_counterpart():
    pairs = match_engines([bf(1, 10, 10)], [sf(1, 10, 11), sf(2, 10, 8.8)], 4.0)
    ag = next(p for p in pairs if p.status == CS.agreement)
    other = next(p for p in pairs if p.status == CS.spotiflow_only)
    assert ag.spotiflow.id == "sf-p0-000002" or ag.spotiflow.id == "sf-p0-000001"
    assert ag.ambiguous_with == [other.spotiflow.id] and other.ambiguous_with == [ag.bigfish.id]


def test_no_ambiguity_when_far_enough():
    # chosen 1.0, alternative 3.5: d2-d1=2.5 >= 1.5 and 3.5 >= 1.25
    pairs = match_engines([bf(1, 10, 10)], [sf(1, 10, 11), sf(2, 13.5, 10)], 4.0)
    assert all(p.ambiguous_with == [] for p in pairs)
    # alternative outside the tolerance is never considered
    pairs = match_engines([bf(1, 10, 10)], [sf(1, 10, 11), sf(2, 10, 14.5)], 4.0)
    assert all(p.ambiguous_with == [] for p in pairs)


def test_deterministic_for_shuffled_inputs():
    rng = np.random.default_rng(3)
    b = [bf(i, *rng.uniform(0, 60, 2)) for i in range(1, 80)]
    s = [sf(i, *rng.uniform(0, 60, 2)) for i in range(1, 80)]
    ref = ids(match_engines(b, s, 3.0))
    for seed in range(5):
        r = random.Random(seed)
        b2, s2 = b[:], s[:]
        r.shuffle(b2); r.shuffle(s2)
        assert ids(match_engines(b2, s2, 3.0)) == ref
    assert any(x[0] == "agreement" for x in ref) and any(x[0] != "agreement" for x in ref)


def test_deterministic_ties_independent_of_order():
    b = [bf(1, 0, 0), bf(2, 2, 0)]
    s = [sf(1, 1, 0), sf(2, 1, 0.0)]            # exact ties
    ref = ids(match_engines(b, s, 3.0))
    assert ids(match_engines(b[::-1], s[::-1], 3.0)) == ref


def test_pages_and_channels_never_merged():
    b = [bf(1, 10, 10, page=0, role="red"), bf(1, 10, 10, page=1, role="green")]
    s = [sf(1, 10, 10, page=1, role="green")]
    pairs = match_engines(b, s, 3.0)
    assert [(p.page, p.channel_role, p.status) for p in pairs] == [
        (0, "red", CS.bigfish_only), (1, "green", CS.agreement)]
    sm = summarize(pairs)
    assert [(x.page, x.channel_role, x.agreement, x.bigfish_only) for x in sm] == [(0, "red", 0, 1), (1, "green", 1, 0)]


def test_empty_inputs_and_negative_tolerance():
    assert match_engines([], [], 4.0) == []
    assert [p.status for p in match_engines([bf(1, 1, 1)], [], 4.0)] == [CS.bigfish_only]
    assert [p.status for p in match_engines([], [sf(1, 1, 1)], 4.0)] == [CS.spotiflow_only]
    with pytest.raises(ValueError):
        match_engines([], [], -1)


def test_large_scale_components_fast():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 3000, (4000, 2))
    b = [bf(i, *p) for i, p in enumerate(pts, 1)]
    s = [sf(i, *(p + rng.normal(0, 0.7, 2))) for i, p in enumerate(pts, 1)]
    pairs = match_engines(b, s, 4.0)
    assert sum(p.status == CS.agreement for p in pairs) > 3900
