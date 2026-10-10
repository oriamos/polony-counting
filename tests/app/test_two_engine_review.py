"""Priorities, QC sample, consensus summary, human_added (pure review layer)."""
import pytest

from fakes import counting, make_state, obj
from polony_app.review import consensus, engine
from polony_app.review.counts import REASON_PRIORITY, review_queue
from polony_app.review.state import ReviewError
from polony_app.schemas import ChannelEvidence, ConsensusStatus as CS, ObjectClass, ReviewReason as R


def ev(status, x=10.0, y=10.0, detected=True):
    return ChannelEvidence(detected=detected, x=x, y=y, peak_intensity=1.0, snr=3.0, consensus_status=status)


def cobj(oid, x, y, status, cls=ObjectClass.red, **kw):
    o = obj(oid, x, y, cls, **kw)
    if cls in (ObjectClass.red, ObjectClass.yellow):
        o.red = ev(status, x, y)
    if cls in (ObjectClass.green, ObjectClass.yellow):
        o.green = ev(status, x, y)
    return o


def test_priorities_disagreements_between_saturated_and_weak_and_qc_last():
    P = REASON_PRIORITY
    assert P[R.saturated] < P[R.bigfish_only] == P[R.spotiflow_only] < P[R.weak_signal]
    assert P[R.ambiguous_match] < P[R.bigfish_only]
    assert P[R.qc_sample] > max(P[r] for r in R if r != R.qc_sample)
    assert set(R) <= set(P)


def test_queue_order_and_unflagged_agreements_not_queued():
    objs = [cobj("agree", 10, 10, CS.agreement),
            cobj("qc", 20, 10, CS.agreement, review_reasons=[R.qc_sample]),
            cobj("bf", 30, 10, CS.bigfish_only, review_reasons=[R.bigfish_only]),
            cobj("sp", 40, 10, CS.spotiflow_only, review_reasons=[R.spotiflow_only]),
            cobj("weak", 50, 10, CS.agreement, review_reasons=[R.weak_signal]),
            cobj("sat", 60, 10, CS.agreement, review_reasons=[R.saturated]),
            cobj("done", 70, 10, CS.bigfish_only, review_reasons=[R.bigfish_only], reviewed=True)]
    q = review_queue(make_state(objs, [counting()]).analysis)
    assert [i["object_id"] for i in q] == ["sat", "bf", "sp", "weak", "qc"]


def test_qc_sample_deterministic_idempotent_and_eligibility():
    objs = [cobj(f"a{i}", 10 + i, 10, CS.agreement) for i in range(100)]
    objs += [cobj("bf", 5, 50, CS.bigfish_only), cobj("rev", 6, 50, CS.agreement, reviewed=True),
             cobj("del", 7, 50, CS.agreement), obj("plain", 8, 50),
             cobj("mixed", 9, 50, CS.agreement, cls=ObjectClass.yellow)]
    objs[-1].green = ev(CS.spotiflow_only)               # one disagreeing channel => not eligible
    st = make_state([o.model_copy(deep=True) for o in objs], [counting()])
    engine.apply_action(st, "delete", {"object_ids": ["del"]})
    n_log = len(st.log)
    r1 = consensus.mark_qc_sample(st, 0.3, 7)
    ids1 = {o.id for o in st.analysis.objects if R.qc_sample in o.review_reasons}
    assert r1["marked"] == len(ids1) == r1["newly_marked"] and 10 < len(ids1) < 60
    assert ids1 <= {f"a{i}" for i in range(100)}         # only unreviewed all-agreement objects
    assert len(st.log) == n_log + 1 and st.log[-1].payload["event"] == "qc_sample"
    # idempotent: same call marks nothing new, no duplicate reasons, no new action
    r2 = consensus.mark_qc_sample(st, 0.3, 7)
    assert r2["marked"] == r1["marked"] and r2["newly_marked"] == 0 and len(st.log) == n_log + 1
    assert all(o.review_reasons.count(R.qc_sample) <= 1 for o in st.analysis.objects)
    # deterministic across fresh states, independent of object order; seed changes the sample
    st2 = make_state(list(reversed([o.model_copy(deep=True) for o in objs])), [counting()])
    engine.apply_action(st2, "delete", {"object_ids": ["del"]})
    consensus.mark_qc_sample(st2, 0.3, 7)
    assert {o.id for o in st2.analysis.objects if R.qc_sample in o.review_reasons} == ids1
    st3 = make_state([o.model_copy(deep=True) for o in objs], [counting()])
    consensus.mark_qc_sample(st3, 0.3, 8)
    assert {o.id for o in st3.analysis.objects if R.qc_sample in o.review_reasons} != ids1
    # queued (below disagreements), then undoable
    assert {i["object_id"] for i in review_queue(st.analysis)} == ids1
    engine.undo(st)
    assert not any(R.qc_sample in o.review_reasons for o in st.analysis.objects)
    for bad in (-0.1, 1.5, "x", True):
        with pytest.raises(ReviewError):
            consensus.mark_qc_sample(st, bad, 1)


def test_qc_fixed_size_sample_and_fraction_compat():
    objs = [cobj(f"a{i}", 10 + i % 100, 10 + 5 * (i // 100), CS.agreement) for i in range(300)]
    mk = lambda: make_state([o.model_copy(deep=True) for o in objs], [counting()])
    r = consensus.mark_qc_sample(mk(), None, 1)
    assert r["marked"] == r["n"] == 40 and r["eligible"] == 300                      # default fixed n = 40
    assert consensus.mark_qc_sample(mk(), None, 1, n=17)["marked"] == 17
    assert consensus.mark_qc_sample(mk(), None, 1, n=1000)["marked"] == 300           # clamped to N eligible
    assert consensus.mark_qc_sample(mk(), None, 1, n=0)["marked"] == 1                # clamped to >= 1
    # compatibility: fraction maps to max(40, ceil(fraction * N)); the old 5 % can no longer shrink the sample
    assert consensus.mark_qc_sample(mk(), 0.05, 1)["marked"] == 40
    assert consensus.mark_qc_sample(mk(), 0.5, 1)["marked"] == 150
    assert consensus.mark_qc_sample(mk(), 0.0, 1)["marked"] == 40
    assert consensus.mark_qc_sample(mk(), 0.9, 1, n=5)["marked"] == 5                 # explicit n wins
    small = make_state([o.model_copy(deep=True) for o in objs[:20]], [counting()])
    assert consensus.mark_qc_sample(small, 1.0, 1)["marked"] == 20
    assert consensus.mark_qc_sample(make_state([], [counting()]), None, 1)["marked"] == 0
    # nested samples: a larger n with the same seed is a superset (one random order)
    s1, s2 = mk(), mk()
    consensus.mark_qc_sample(s1, None, 5, n=20)
    consensus.mark_qc_sample(s2, None, 5, n=60)
    ids = lambda st: {o.id for o in st.analysis.objects if R.qc_sample in o.review_reasons}
    assert ids(s1) < ids(s2)
    with pytest.raises(ReviewError):
        consensus.mark_qc_sample(mk(), None, 1, n=-3)


def test_finite_population_sample_sizes_for_plus_minus_005():
    # p = 0.153 (pooled agreement error), 95 %, finite-population correction
    assert [consensus.n_for_margin(N) for N in (24, 49, 67, 67, 203, 296)] == [22, 40, 51, 51, 101, 120]
    assert consensus.n_for_margin(0) == 0 and consensus.n_for_margin(5) == 5
    assert consensus.n_for_margin(296, p=0.5) > consensus.n_for_margin(296)


def test_clopper_pearson_known_values():
    lo, hi = consensus.clopper_pearson(0, 40)
    assert lo == 0.0 and abs(hi - 0.0881) < 1e-3                 # 1 - 0.025^(1/40)
    lo, hi = consensus.clopper_pearson(6, 40)
    assert abs(lo - 0.0571) < 2e-3 and abs(hi - 0.2984) < 2e-3
    assert consensus.clopper_pearson(40, 40)[1] == 1.0 and consensus.clopper_pearson(0, 0) == (0.0, 1.0)


def test_qc_status_reports_interval_after_review():
    objs = [cobj(f"a{i}", 10 + i, 10, CS.agreement) for i in range(60)]
    st = make_state(objs, [counting()])
    r = consensus.mark_qc_sample(st, None, 3, n=20)
    assert r["qc"]["sample_size"] == 20 and r["qc"]["reviewed"] == 0 and r["qc"]["ci95"] is None
    ids = sorted(o.id for o in st.analysis.objects if R.qc_sample in o.review_reasons)
    engine.apply_action(st, "delete", {"object_ids": ids[:3]})            # 3 sampled agreements rejected
    engine.apply_action(st, "confirm", {"object_ids": ids[3:]})           # 17 confirmed
    q = consensus.consensus_summary(st)["qc"]
    assert (q["sample_size"], q["reviewed"], q["confirmed"], q["rejected"], q["pending"]) == (20, 20, 17, 3, 0)
    lo, hi = consensus.clopper_pearson(3, 20)
    assert q["error_rate"] == 0.15 and q["ci95"] == [lo, hi] and q["seed"] == 3
    assert q["population_agreements"] == 60 and q["n_for_margin_0_05"] == consensus.n_for_margin(60)
    assert "not ground truth" in q["note"] and "NOT reviewed" in q["note"]


def test_manual_add_is_human_added_on_its_channels_and_summary_counts():
    st = make_state([], [counting()])
    for cls in ("red", "green", "yellow"):
        engine.apply_action(st, "add", {"x": 20 + 10 * len(st.analysis.objects), "y": 20, "cls": cls})
    r, g, y = st.analysis.objects
    assert r.red.consensus_status == CS.human_added and r.green.consensus_status is None and not r.red.detected
    assert g.green.consensus_status == CS.human_added and g.red.consensus_status is None
    assert y.red.consensus_status == y.green.consensus_status == CS.human_added
    s = consensus.consensus_summary(st)
    by = {p["channel_role"]: p for p in s["pages"]}
    assert by["red"]["human_added"] == 2 and by["green"]["human_added"] == 2 and s["total"]["human_added"] == 4
    assert s["total"]["confirmed"] == 4 and s["total"]["unresolved"] == 0   # a manual add is reviewed


def test_summary_scenarios_per_channel():
    objs = [cobj("ag", 10, 10, CS.agreement),
            cobj("ag2", 20, 10, CS.agreement),
            cobj("bf", 30, 10, CS.bigfish_only, review_reasons=[R.bigfish_only]),
            cobj("sp", 40, 10, CS.spotiflow_only, ObjectClass.green, review_reasons=[R.spotiflow_only]),
            cobj("rej", 50, 10, CS.bigfish_only),
            cobj("unc", 60, 10, CS.agreement, ObjectClass.uncertain),
            cobj("dual", 70, 10, CS.agreement, ObjectClass.yellow),
            obj("legacy", 80, 10)]
    objs[5].red = ev(CS.agreement)                        # uncertain object with red evidence
    st = make_state(objs, [counting()])
    engine.apply_action(st, "confirm", {"object_ids": ["ag"]})
    engine.apply_action(st, "delete", {"object_ids": ["rej"]})
    s = consensus.consensus_summary(st)
    red, green = s["pages"]
    assert (red["channel_role"], red["page"], green["channel_role"], green["page"]) == ("red", 0, "green", 1)
    assert red["agreement"] == 4 and red["bigfish_only"] == 1 and red["spotiflow_only"] == 0
    assert red["human_rejected"] == 1 and red["rejected"] == 1
    assert red["confirmed"] == 1                           # only the confirmed one
    # ag2, bf, unc (uncertain), dual, and the status-less legacy object (a detected candidate)
    assert red["unresolved"] == 5
    assert green["agreement"] == 1 and green["spotiflow_only"] == 1 and green["unresolved"] == 2
    tot = s["total"]
    assert tot["agreement"] == 5 and tot["human_rejected"] == 1
    # human_rejected is derived, never stored; objects keep their engine status
    assert [o.red.consensus_status for o in st.analysis.objects if o.id == "rej"] == [CS.bigfish_only]
    # undo of the delete restores the candidate (no stored rejection)
    engine.undo(st)
    assert consensus.consensus_summary(st)["total"]["human_rejected"] == 0


def test_merge_deletions_are_not_rejections():
    objs = [cobj("m1", 10, 10, CS.agreement), cobj("m2", 12, 10, CS.agreement)]
    st = make_state(objs, [counting()])
    engine.apply_action(st, "merge", {"object_ids": ["m1", "m2"]})
    s = consensus.consensus_summary(st)
    assert s["total"]["human_rejected"] == 0 and s["total"]["rejected"] == 0


def test_debris_like_bigfish_only_is_queued_with_saturated_priority():
    from polony_app.review.counts import object_priority
    both = cobj("both", 10, 10, CS.bigfish_only, review_reasons=[R.bigfish_only, R.debris_like])
    bf = cobj("bf", 11, 10, CS.bigfish_only, review_reasons=[R.bigfish_only])
    deb = cobj("deb", 12, 10, CS.agreement, review_reasons=[R.debris_like])
    assert object_priority(both) == 1 and object_priority(bf) == 2 and object_priority(deb) == 4
