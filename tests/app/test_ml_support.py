"""Helpers for the ML tests (no tests here): reviewed analyses in a real SQLite store, with real 16-bit TIFFs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from fakes import counting, exclusion, make_state, obj, rect
from polony_app.review import engine
from polony_app.review.engine import region_fingerprint
from polony_app.schemas import ObjectClass, SlideMeta
from polony_app.store import Store

H, W = 120, 160


def tiff_pages(seed: int):
    rng = np.random.default_rng(seed)
    red = rng.integers(100, 60000, (H, W)).astype(np.uint16)   # full 16-bit range: any 8-bit conversion would show
    green = rng.integers(50, 65000, (H, W)).astype(np.uint16)
    return red, green


def new_store(tmp_path) -> Store:
    return Store(tmp_path / "data")


def add_analysis(store: Store, name: str, *, slide_id=None, seed=1, approve=("f1", "f2"), confirmed_mapping=True,
                 f1_objects=((20.0, 20.0), (40.0, 30.0), (60.0, 50.0)), f2_objects=((100.0, 20.0), (120.0, 40.0)),
                 exclusion_over_f2=False, reviewer=None, alignment=(0, 0), green_objects=((30.0, 25.0),), pages=None,
                 extra_deleted=()) -> str:
    """Analysis with two rectangular fields f1 (5,5)-(75,60) and f2 (80,5)-(150,60).
    f1: red objects (first one confirmed individually, one deleted by the user, one manual add), green objects.
    Returns the analysis id."""
    aid = store.create_analysis(name)
    red, green = pages if pages is not None else tiff_pages(seed)
    path = store.originals_dir(aid) / "a.tif"
    tifffile.imwrite(str(path), np.stack([red, green]), photometric="minisblack")
    from polony_app.store.db import Store as _S  # noqa: F401
    from polony_app.review.state import AnalysisState  # noqa: F401
    from polony_app.imaging.io import read_plane  # noqa: F401

    f1 = counting("f1", rect(5, 5, 75, 60), name="Field-1")
    f1.field_index = 1
    f2 = counting("f2", rect(80, 5, 150, 60), name="Field-2")
    f2.field_index = 2
    regions = [f1, f2] + ([exclusion("x1", rect(90, 10, 110, 30))] if exclusion_over_f2 else [])
    objs = [obj(f"r1_{i}", x, y, ObjectClass.red) for i, (x, y) in enumerate(f1_objects)]
    objs += [obj(f"g1_{i}", x, y, ObjectClass.green) for i, (x, y) in enumerate(green_objects)]
    objs += [obj(f"r2_{i}", x, y, ObjectClass.red) for i, (x, y) in enumerate(f2_objects)]
    st = make_state(objs, regions, mapped=confirmed_mapping)
    a = st.analysis
    a.id = aid
    a.name = name
    a.alignment_dy, a.alignment_dx = alignment
    a.image.channels[0].source_file = a.image.channels[1].source_file = "a.tif"
    from polony_app.ml.common import sha256_file
    a.image.sha256 = {"a.tif": sha256_file(path)}
    if slide_id:
        a.slide = SlideMeta(slide_id=slide_id)
    with store.edit(aid) as live:
        live.analysis = a
        live.next_seq = 1
        engine.assign_region_ids(live.analysis)
        engine.apply_action(live, "confirm", {"object_ids": ["r1_0"]})
        engine.apply_action(live, "delete", {"object_ids": ["r1_1"]})
        engine.apply_action(live, "add", {"x": 50.0, "y": 40.0, "cls": "red"})
        for oid in extra_deleted:
            engine.apply_action(live, "delete", {"object_ids": [oid]})
        if reviewer:
            for act in live.log:
                act.payload["reviewer"] = reviewer
        for rid in approve:
            engine.apply_action(live, "region_inspected", {"region_id": rid, "inspected": True})
            engine.apply_action(live, "approve_region", {"region_id": rid})
    return aid


def force_unresolved(store: Store, aid: str, oid: str = "r2_0"):
    """Make an object uncertain AFTER approval and refresh the approval fingerprint (so the region still counts as
    fully reviewed): the situation 'unlabelled / unresolved point inside an approved region'."""
    with store.edit(aid) as live:
        a = live.analysis
        o = next(o for o in a.objects if o.id == oid)
        o.cls = ObjectClass.uncertain
        for r in a.regions:
            if r.approved:
                r.approval_fingerprint = region_fingerprint(a, r)


class Recorder:
    """say() collecting lines."""
    def __init__(self):
        self.lines = []

    def __call__(self, s):
        self.lines.append(str(s))

    def text(self):
        return "\n".join(self.lines)
