"""Review engine: applies review actions to an AnalysisState, producing
inverse (undo) payloads, approval fingerprints and history strings.

Pure: no I/O, no clocks other than `now()` (overridable for tests), no FastAPI.

Undo model: every undoable action stores, in ``ReviewAction.inverse``,
``{"op": "restore", "before": <snapshot>, "after": <snapshot>}``. A snapshot
holds the full JSON of every object/region/provenance touched by the action
(``None`` = did not exist). Undo applies ``before``, redo applies ``after``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from polony_app.schemas import (
    ActionType,
    Analysis,
    AnalysisStatus,
    ChannelEvidence,
    ConsensusStatus,
    DetectedObject,
    DetectionRequest,
    ObjectClass,
    Origin,
    Region,
    RegionKind,
    ReviewAction,
)

from ..detection import masks as _masks
from .geometry import bbox, polygon_area, polygons_overlap, validate_polygon
from .state import AnalysisState, DetectionCancelled, ReviewError

RERUN_MERGE_RADIUS_PX = 3.0
MAX_UNDO = 1000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _short(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _unique_id(prefix: str, taken: set[str]) -> str:
    """Fresh id not in `taken` (loops until unique), and reserves it."""
    while True:
        i = _short(prefix)
        if i not in taken:
            taken.add(i)
            return i


def _taken_ids(tx: "Tx") -> set[str]:
    return {o.id for o in tx.a.objects} | set(tx.before_objs)


# ------------------------------------------------------------------ guards
def guard_editable(state: AnalysisState) -> None:
    if state.analysis.finalized_at is not None:
        raise ReviewError(
            "analysis_finalized",
            "This analysis is finalized and read-only. Reopen it "
            "(POST /api/analyses/{id}/reopen) to make changes.",
            409,
        )


def mapping_confirmed(a: Analysis) -> bool:
    if a.image is None:
        return False
    chans = a.image.channels
    if not chans or not all(c.role_confirmed and c.role is not None for c in chans):
        return False
    roles = [c.role.value for c in chans]
    return roles.count("red") == 1 and roles.count("green") == 1


# ------------------------------------------------------------------ geometry / regions
def _counting(a: Analysis) -> list[Region]:
    return [r for r in a.regions if r.kind == RegionKind.counting]


def _exclusions(a: Analysis) -> list[Region]:
    return [r for r in a.regions if r.kind == RegionKind.exclusion]


def point_region_id(a: Analysis, x: float, y: float) -> str | None:
    """Region id of a point under the single geometry rule (detection/masks.point_region):
    half-open pixel-centre rule, first listed counting region wins, exclusions veto."""
    if not _counting(a):
        return None
    return _masks.point_region(x, y, a.regions)


def assign_region_ids(a: Analysis, objs: Iterable[DetectedObject] | None = None) -> None:
    for o in objs if objs is not None else a.objects:
        o.region_id = point_region_id(a, o.x, o.y)


def _round_poly(poly) -> list[list[float]]:
    return [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in poly]


def region_objects_fingerprint(a: Analysis, region_id: str) -> str:
    objs = sorted(
        (o.id, round(o.x, 3), round(o.y, 3), o.cls.value)
        for o in a.objects
        if not o.deleted and o.region_id == region_id
    )
    return hashlib.sha256(json.dumps(objs).encode()).hexdigest()


def _roles(a: Analysis) -> list[list]:
    return [
        [c.index, c.role.value if c.role else None, bool(c.role_confirmed)]
        for c in (a.image.channels if a.image else [])
    ]


def input_signature(a: Analysis, request: DetectionRequest | None = None) -> dict:
    """Everything a detection result depends on: the request, the regions, the channel
    roles and the green->red alignment. Recorded at detection time (AnalysisState.detection_sig);
    `provenance.stale_detection` is derived by comparing the CURRENT signature with it."""
    req = (request if request is not None else a.detection_request).model_dump(mode="json")
    regions = hashlib.sha256(
        json.dumps(
            [[r.id, r.kind.value, _round_poly(r.polygon)] for r in a.regions], sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "request": req,
        "regions_sha256": regions,
        "channel_roles": _roles(a),
        "alignment": [int(a.alignment_dy), int(a.alignment_dx)],
    }


def request_differs(a: Analysis) -> bool:
    """True if the parameters currently set differ from those the shown detection used."""
    ran = a.provenance.detection_request
    if a.provenance.detection_ran_at is None or ran is None:  # nothing recorded to compare with
        return False
    return ran.model_dump(mode="json") != a.detection_request.model_dump(mode="json")


def compute_stale(a: Analysis, recorded_sig: dict | None, flag: bool = False) -> bool:
    if a.provenance.detection_ran_at is None:
        return False
    if request_differs(a):
        return True
    if recorded_sig is None:  # legacy state without a recorded signature: keep the stored flag
        return bool(flag)
    return recorded_sig != input_signature(a)


def recompute_stale(state: AnalysisState) -> bool:
    a = state.analysis
    a.provenance.stale_detection = compute_stale(a, state.detection_sig, a.provenance.stale_detection)
    return a.provenance.stale_detection


def migrate_legacy_state(state: AnalysisState) -> bool:
    """States persisted before `detection_sig` existed carry only the stored stale flag, which
    compute_stale would trust forever (an edit could never make them stale). On load:
      * detection ran, stored flag False -> the CURRENT inputs are by definition the ones the
        shown result was produced from: synthesise the signature from them (+ the recorded
        request) so later edits are detected;
      * stored flag True -> keep it (sticky) until detection is re-run, which records a real
        signature.
    Returns True if a signature was synthesised. See docs/app/RERUN_POLICY.md."""
    a = state.analysis
    if state.detection_sig is not None or a.provenance.detection_ran_at is None:
        return False
    if a.provenance.stale_detection:
        return False
    state.detection_sig = input_signature(a, a.provenance.detection_request)
    return True


def region_fingerprint(a: Analysis, region: Region) -> str:
    """Hash of (region polygon, overlapping exclusion polygons, non-deleted
    objects inside). Anything affecting the count inside the region changes it."""
    excl = sorted(
        (e.id, _round_poly(e.polygon))
        for e in _exclusions(a)
        if polygons_overlap(region.polygon, e.polygon)
    )
    payload = {
        "region": [region.id, region.kind.value, _round_poly(region.polygon)],
        "exclusions": excl,
        "objects": region_objects_fingerprint(a, region.id),
        # a different channel mapping / alignment changes what the objects mean
        "inputs": {"roles": _roles(a), "alignment": [int(a.alignment_dy), int(a.alignment_dx)]},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ transactions
class Tx:
    """Collects before/after snapshots for one action."""

    def __init__(self, state: AnalysisState):
        self.state = state
        self.a = state.analysis
        self.before_objs: dict[str, dict | None] = {}
        self.before_regions: list[dict] | None = None
        self.before_prov: dict | None = None
        self.before_grid: dict | None = None
        self.before_book: dict | None = None
        self.cleared_approvals: list[str] = []

    def touch(self, obj_id: str) -> DetectedObject:
        o = self.get(obj_id)
        if obj_id not in self.before_objs:
            self.before_objs[obj_id] = o.model_dump(mode="json")
        return o

    def touch_obj(self, o: DetectedObject) -> None:
        if o.id not in self.before_objs:
            self.before_objs[o.id] = o.model_dump(mode="json")

    def new(self, o: DetectedObject) -> None:
        if o.id not in self.before_objs:  # (an id removed earlier in this tx keeps its snapshot)
            self.before_objs[o.id] = None
        self.a.objects.append(o)

    def remove(self, o: DetectedObject) -> None:
        self.touch_obj(o)
        self.a.objects.remove(o)

    def touch_regions(self) -> None:
        if self.before_regions is None:
            self.before_regions = [r.model_dump(mode="json") for r in self.a.regions]

    def _grid(self) -> dict:
        sl = self.a.slide
        return {"grid": sl.grid.model_dump(mode="json") if sl.grid is not None else None,
                "skipped_fields": [s_.model_dump(mode="json") for s_ in sl.skipped_fields]}

    def touch_grid(self) -> None:
        """Snapshot slide.grid + slide.skipped_fields (grid on/off state) so undo restores them with the regions."""
        if self.before_grid is None:
            self.before_grid = self._grid()

    def touch_prov(self) -> None:
        if self.before_prov is None:
            self.before_prov = self.a.provenance.model_dump(mode="json")
        self.touch_book()

    def _book(self) -> dict:
        return {
            "detection_sig": copy.deepcopy(self.state.detection_sig),
            "user_deleted": list(self.state.user_deleted),
        }

    def touch_book(self) -> None:
        if self.before_book is None:
            self.before_book = self._book()

    def get(self, obj_id: str) -> DetectedObject:
        for o in self.a.objects:
            if o.id == obj_id:
                return o
        raise ReviewError("object_not_found", f"object '{obj_id}' not found", 404)

    def get_live(self, obj_id: str) -> DetectedObject:
        o = self.get(obj_id)
        if o.deleted:
            raise ReviewError("object_deleted", f"object '{obj_id}' is deleted", 409)
        return o

    def revalidate(self) -> None:
        self.cleared_approvals += revalidate_approvals(self.a, self)

    def _after_snapshot(self) -> dict:
        by_id = {o.id: o for o in self.a.objects}
        return {
            "objects": {
                i: (by_id[i].model_dump(mode="json") if i in by_id else None)
                for i in self.before_objs
            },
            "regions": [r.model_dump(mode="json") for r in self.a.regions]
            if self.before_regions is not None
            else None,
            "provenance": self.a.provenance.model_dump(mode="json")
            if self.before_prov is not None
            else None,
            "bookkeeping": self._book() if self.before_book is not None else None,
            "grid": self._grid() if self.before_grid is not None else None,
        }

    def commit(self, type_: ActionType, payload: dict, undoable: bool = True) -> ReviewAction:
        st = self.state
        inverse = {
            "op": "restore",
            "before": {
                "objects": self.before_objs,
                "regions": self.before_regions,
                "provenance": self.before_prov,
                "bookkeeping": self.before_book,
                "grid": self.before_grid,
            },
            "after": self._after_snapshot(),
        }
        payload = dict(payload)
        if self.cleared_approvals:
            payload["approvals_cleared"] = sorted(set(self.cleared_approvals))
        act = ReviewAction(seq=st.next_seq, ts=now(), type=type_, payload=payload, inverse=inverse)
        st.next_seq += 1
        st.log.append(act)
        if undoable:
            st.undo.append(act.seq)
            if len(st.undo) > MAX_UNDO:
                st.undo = st.undo[-MAX_UNDO:]
            st.redo = []
        self.a.updated_at = act.ts
        st.sync_depths()
        refresh_status(st)
        return act


def _apply_snapshot(state: AnalysisState, snap: dict) -> None:
    a = state.analysis
    objs = snap.get("objects") or {}
    index = {o.id: i for i, o in enumerate(a.objects)}
    appended: list[DetectedObject] = []
    to_remove: set[str] = set()
    for oid, data in objs.items():
        if data is None:
            to_remove.add(oid)
        elif oid in index:
            a.objects[index[oid]] = DetectedObject.model_validate(data)
        else:
            appended.append(DetectedObject.model_validate(data))
    if to_remove:
        a.objects = [o for o in a.objects if o.id not in to_remove]
    a.objects.extend(appended)
    if snap.get("regions") is not None:
        a.regions = [Region.model_validate(r) for r in snap["regions"]]
    if snap.get("provenance") is not None:
        from polony_app.schemas import Provenance

        a.provenance = Provenance.model_validate(snap["provenance"])
    grid = snap.get("grid")
    if grid is not None:
        from polony_app.schemas import GridSpec, SkippedField

        a.slide.grid = GridSpec.model_validate(grid["grid"]) if grid.get("grid") is not None else None
        a.slide.skipped_fields = [SkippedField.model_validate(x) for x in grid.get("skipped_fields", [])]
    book = snap.get("bookkeeping")
    if book is not None:
        state.detection_sig = copy.deepcopy(book.get("detection_sig"))
        state.user_deleted = list(book.get("user_deleted", []))


def revalidate_approvals(a: Analysis, tx: Tx | None = None) -> list[str]:
    """Clear approvals whose fingerprint no longer matches. Returns cleared ids."""
    cleared = []
    for r in a.regions:
        if not r.approved:
            continue
        ok = (
            r.kind == RegionKind.counting
            and r.approval_fingerprint is not None
            and r.approval_fingerprint == region_fingerprint(a, r)
        )
        if not ok:
            if tx is not None:
                tx.touch_regions()
            r.approved = False
            r.approved_at = None
            r.approval_fingerprint = None
            cleared.append(r.id)
    return cleared


def refresh_status(state: AnalysisState, job_running: bool | None = None) -> None:
    """Recompute staleness and status. `job_running` True/False sets the persisted
    `state.job_active` flag; None keeps it, so an edit made while a detection job runs can
    never move 'detecting' to 'in_review'."""
    a = state.analysis
    if job_running is not None:
        state.job_active = bool(job_running)
    recompute_stale(state)
    if a.finalized_at is not None:
        a.status = AnalysisStatus.finalized
    elif state.job_active:
        a.status = AnalysisStatus.detecting
    elif a.image is None:
        a.status = AnalysisStatus.created
    elif not mapping_confirmed(a):
        a.status = AnalysisStatus.needs_channel_mapping
    elif a.provenance.detection_ran_at is not None or a.objects:
        a.status = AnalysisStatus.in_review
    else:
        a.status = AnalysisStatus.ready


# ------------------------------------------------------------------ payload helpers
def _need(payload: dict, key: str) -> Any:
    if not isinstance(payload, dict) or key not in payload:
        raise ReviewError("invalid_payload", f"payload is missing '{key}'", 422)
    return payload[key]


def _ids(payload: dict, key: str = "object_ids", min_n: int = 1) -> list[str]:
    ids = _need(payload, key)
    if not isinstance(ids, list) or len(ids) < min_n or not all(isinstance(i, str) for i in ids):
        raise ReviewError("invalid_payload", f"'{key}' must be a list of >= {min_n} ids", 422)
    if len(set(ids)) != len(ids):
        raise ReviewError("invalid_payload", f"'{key}' has duplicates", 422)
    return ids


def _cls(v: Any) -> ObjectClass:
    try:
        return ObjectClass(v)
    except Exception:
        raise ReviewError(
            "invalid_payload", f"invalid class {v!r}; use red|green|yellow|uncertain", 422
        )


def _xy(a: Analysis, x: Any, y: Any) -> tuple[float, float]:
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        raise ReviewError("invalid_payload", "x and y must be numbers", 422)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ReviewError("invalid_payload", "x and y must be finite", 422)
    if a.image is not None and not (
        -0.5 <= x <= a.image.width - 0.5 and -0.5 <= y <= a.image.height - 0.5
    ):
        raise ReviewError(
            "out_of_image", f"position ({x:.1f},{y:.1f}) is outside the image", 422
        )
    return x, y


def _region(a: Analysis, rid: Any, counting_only: bool = True) -> Region:
    for r in a.regions:
        if r.id == rid:
            if counting_only and r.kind != RegionKind.counting:
                raise ReviewError("not_counting_region", f"region '{rid}' is not a counting region", 422)
            return r
    raise ReviewError("region_not_found", f"region '{rid}' not found", 404)


def _hist(o: DetectedObject, msg: str, ts: str) -> None:
    o.history = list(o.history) + [f"{ts} {msg}"]


def _check_position_counted(a: Analysis, x: float, y: float) -> str | None:
    rid = point_region_id(a, x, y)
    if _counting(a) and rid is None:
        raise ReviewError(
            "outside_counting_region",
            "position is outside every counting region (or inside an exclusion); "
            "objects there are never counted",
            422,
        )
    return rid


def _channels_of(o: DetectedObject) -> tuple[bool, bool]:
    r = o.cls in (ObjectClass.red, ObjectClass.yellow) or (
        o.cls == ObjectClass.uncertain and o.red.detected
    )
    g = o.cls in (ObjectClass.green, ObjectClass.yellow) or (
        o.cls == ObjectClass.uncertain and o.green.detected
    )
    return r, g


# ------------------------------------------------------------------ actions
def apply_action(state: AnalysisState, type_: ActionType | str, payload: dict) -> ReviewAction:
    """Apply one user action. Raises ReviewError when refused (state unchanged
    for refusals raised before mutation; callers evict/reload on exceptions)."""
    try:
        type_ = ActionType(type_)
    except ValueError:
        raise ReviewError("invalid_action", f"unknown action type {type_!r}", 422)
    handler = _HANDLERS.get(type_)
    if handler is None:
        raise ReviewError(
            "invalid_action", f"action '{type_.value}' cannot be submitted directly", 422
        )
    guard_editable(state)
    a = state.analysis
    if not isinstance(payload, dict):
        raise ReviewError("invalid_payload", "payload must be an object", 422)
    tx = Tx(state)
    ts = now()
    handler(tx, a, payload, ts)
    tx.revalidate()
    return tx.commit(type_, payload)


def _h_add(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    x, y = _xy(a, _need(p, "x"), _need(p, "y"))
    cls = _cls(_need(p, "cls"))
    rid = _check_position_counted(a, x, y)
    # A manual add is, by definition, a human-added candidate: ALWAYS recorded (whatever engine ran)
    # on the channel evidence of the colour(s) the user chose. `detected` stays False: no engine saw it.
    ha = ChannelEvidence(detected=False, consensus_status=ConsensusStatus.human_added)
    o = DetectedObject(
        id=_unique_id("m", _taken_ids(tx)),
        x=x,
        y=y,
        cls=cls,
        red=ha.model_copy() if cls in (ObjectClass.red, ObjectClass.yellow) else ChannelEvidence(detected=False),
        green=ha.model_copy() if cls in (ObjectClass.green, ObjectClass.yellow) else ChannelEvidence(detected=False),
        origin=Origin.manual,
        region_id=rid,
        reviewed=True,
        history=[f"{ts} added manually as {cls.value} at ({x:.1f},{y:.1f})"],
    )
    tx.new(o)
    p["object_id"] = o.id  # recorded so redo/inverse is traceable


def _merge_constituents(a: Analysis, o: DetectedObject, seen: set[str] | None = None) -> list[str]:
    """Ids of all objects merged (transitively) into `o`, read from its 'merged from' history."""
    seen = seen if seen is not None else {o.id}
    by_id = {x.id: x for x in a.objects}
    out: list[str] = []
    for h in o.history:
        _, sep, rest = h.partition(" merged from ")
        if not sep:
            continue
        for cid in (c.strip() for c in rest.split(",")):
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
                c = by_id.get(cid)
                if c is not None:
                    out.extend(_merge_constituents(a, c, seen))
    return out


def _h_delete(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    ids = _ids(p)
    objs = [tx.get_live(oid) for oid in ids]  # validate all before mutating
    tx.touch_book()
    for o in objs:
        tx.touch_obj(o)
        o.deleted = True
        _hist(o, "deleted by user", ts)
        # explicit marker (snapshotted for undo/redo): this deletion was a user decision
        if o.id not in tx.state.user_deleted:
            tx.state.user_deleted.append(o.id)
        # deleting a MERGED object is a decision about every spot it was made of: explicit
        # tombstone anchors for all constituents (their positions and evidence positions)
        for cid in _merge_constituents(a, o):
            if cid not in tx.state.user_deleted and any(x.id == cid for x in a.objects):
                tx.state.user_deleted.append(cid)


def _h_move(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    oid = _need(p, "object_id")
    x, y = _xy(a, _need(p, "x"), _need(p, "y"))
    o = tx.get_live(oid)
    rid = _check_position_counted(a, x, y)
    tx.touch_obj(o)
    _hist(o, f"moved from ({o.x:.1f},{o.y:.1f}) to ({x:.1f},{y:.1f})", ts)
    o.x, o.y = x, y
    o.region_id = rid
    o.reviewed = True
    if o.origin == Origin.auto:
        o.origin = Origin.edited


def _h_reclassify(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    cls = _cls(_need(p, "cls"))
    for oid in _ids(p):
        o = tx.get_live(oid)
        tx.touch_obj(o)
        _hist(o, f"reclassified {o.cls.value} -> {cls.value}", ts)
        o.cls = cls
        o.reviewed = True
        if o.origin == Origin.auto:
            o.origin = Origin.edited


def _h_merge(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    ids = _ids(p, min_n=2)
    objs = [tx.get_live(i) for i in ids]
    cx = sum(o.x for o in objs) / len(objs)
    cy = sum(o.y for o in objs) / len(objs)
    rid = _check_position_counted(a, cx, cy)
    has_r = has_g = False
    for o in objs:
        r, g = _channels_of(o)
        has_r |= r
        has_g |= g
    cls = (
        ObjectClass.yellow
        if has_r and has_g
        else ObjectClass.red
        if has_r
        else ObjectClass.green
        if has_g
        else ObjectClass.uncertain
    )

    def union(ch: str) -> ChannelEvidence:
        ev = [getattr(o, ch) for o in objs if getattr(o, ch).detected]
        if not ev:
            return ChannelEvidence(detected=False)
        best = max(ev, key=lambda e: (e.peak_intensity is not None, e.peak_intensity or 0.0))
        out = best.model_copy(deep=True)
        out.saturated = any(e.saturated for e in ev)
        return out

    reasons = []
    for o in objs:
        for r in o.review_reasons:
            if r not in reasons:
                reasons.append(r)
    new = DetectedObject(
        id=_unique_id("e", _taken_ids(tx)),
        x=cx,
        y=cy,
        cls=cls,
        origin=Origin.manual if all(o.origin == Origin.manual for o in objs) else Origin.edited,
        red=union("red"),
        green=union("green"),
        review_reasons=reasons,
        reviewed=True,
        region_id=rid,
        history=[f"{ts} merged from {', '.join(ids)}"],
    )
    for o in objs:
        tx.touch_obj(o)
        o.deleted = True
        _hist(o, f"merged into {new.id}", ts)
    tx.new(new)
    p["result_id"] = new.id


def _h_split(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    oid = _need(p, "object_id")
    pos = _need(p, "positions")
    classes = _need(p, "classes")
    if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pos)):
        raise ReviewError("invalid_payload", "'positions' must be [[x,y],[x,y]]", 422)
    if not (isinstance(classes, list) and len(classes) == 2):
        raise ReviewError("invalid_payload", "'classes' must have two entries", 422)
    o = tx.get_live(oid)
    pts = [_xy(a, q[0], q[1]) for q in pos]
    cl = [_cls(c) for c in classes]
    rids = [_check_position_counted(a, *q) for q in pts]
    tx.touch_obj(o)
    o.deleted = True
    new_ids = []
    taken = _taken_ids(tx)
    for (x, y), c, rid in zip(pts, cl, rids):
        # a child keeps only the evidence that matches its class
        keep_r = c != ObjectClass.green
        keep_g = c != ObjectClass.red
        n = DetectedObject(
            id=_unique_id("e", taken),
            x=x,
            y=y,
            cls=c,
            origin=Origin.manual if o.origin == Origin.manual else Origin.edited,
            red=o.red.model_copy(deep=True) if keep_r else ChannelEvidence(detected=False),
            green=o.green.model_copy(deep=True) if keep_g else ChannelEvidence(detected=False),
            review_reasons=list(o.review_reasons),
            reviewed=True,
            region_id=rid,
            history=[f"{ts} split from {o.id}"],
        )
        tx.new(n)
        new_ids.append(n.id)
    _hist(o, f"split into {', '.join(new_ids)}", ts)
    p["result_ids"] = new_ids


def _h_confirm(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    for oid in _ids(p):
        o = tx.get_live(oid)
        tx.touch_obj(o)
        if not o.reviewed:
            o.reviewed = True
            _hist(o, "confirmed by user", ts)


def _h_region_inspected(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    r = _region(a, _need(p, "region_id"))
    val = _need(p, "inspected")
    if not isinstance(val, bool):
        raise ReviewError("invalid_payload", "'inspected' must be boolean", 422)
    tx.touch_regions()
    r.inspected = val
    if not val and r.approved:
        r.approved, r.approved_at, r.approval_fingerprint = False, None, None
        tx.cleared_approvals.append(r.id)


def _h_accept_region(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    r = _region(a, _need(p, "region_id"))
    if not r.inspected:
        raise ReviewError(
            "region_not_inspected",
            f"region '{r.name}' must be marked as inspected before accepting its proposals",
            409,
        )
    n = 0
    for o in a.objects:
        if o.region_id == r.id and not o.deleted and o.cls != ObjectClass.uncertain and not o.reviewed:
            tx.touch_obj(o)
            o.reviewed = True
            _hist(o, "accepted with region", ts)
            n += 1
    p["accepted"] = n


def _h_approve_region(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    r = _region(a, _need(p, "region_id"))
    if not r.inspected:
        raise ReviewError(
            "region_not_inspected", f"region '{r.name}' has not been marked as inspected", 409
        )
    unresolved = [
        o.id for o in a.objects if o.region_id == r.id and not o.deleted and o.cls == ObjectClass.uncertain
    ]
    if unresolved:
        raise ReviewError(
            "unresolved_uncertain",
            f"region '{r.name}' still has {len(unresolved)} unresolved uncertain object(s)",
            409,
            {"object_ids": unresolved},
        )
    if recompute_stale(tx.state):
        raise ReviewError(
            "stale_detection",
            "detection is stale (parameters, regions, channels or alignment changed); re-run detection first",
            409,
        )
    has_obj = any(o.region_id == r.id and not o.deleted for o in a.objects)
    if a.provenance.detection_ran_at is None and not has_obj:
        raise ReviewError(
            "no_detection", "no detection has been run and the region has no objects", 409
        )
    tx.touch_regions()
    r.approved = True
    r.approved_at = ts
    r.approval_fingerprint = region_fingerprint(a, r)


def _h_unapprove_region(tx: Tx, a: Analysis, p: dict, ts: str) -> None:
    r = _region(a, _need(p, "region_id"))
    tx.touch_regions()
    r.approved, r.approved_at, r.approval_fingerprint = False, None, None


_HANDLERS: dict[ActionType, Callable[[Tx, Analysis, dict, str], None]] = {
    ActionType.add: _h_add,
    ActionType.delete: _h_delete,
    ActionType.move: _h_move,
    ActionType.reclassify: _h_reclassify,
    ActionType.merge: _h_merge,
    ActionType.split: _h_split,
    ActionType.confirm: _h_confirm,
    ActionType.accept_region: _h_accept_region,
    ActionType.region_inspected: _h_region_inspected,
    ActionType.approve_region: _h_approve_region,
    ActionType.unapprove_region: _h_unapprove_region,
}


# ------------------------------------------------------------------ undo / redo
def undo(state: AnalysisState) -> ReviewAction:
    guard_editable(state)
    if not state.undo:
        raise ReviewError("nothing_to_undo", "nothing to undo", 409)
    seq = state.undo.pop()
    act = state.action_by_seq(seq)
    _apply_snapshot(state, act.inverse["before"])
    state.redo.append(seq)
    state.undo_performed += 1
    _log_event(state, "undo", {"undone_seq": seq, "action_type": act.type.value})
    _after_history_move(state)
    return act


def redo(state: AnalysisState) -> ReviewAction:
    guard_editable(state)
    if not state.redo:
        raise ReviewError("nothing_to_redo", "nothing to redo", 409)
    seq = state.redo.pop()
    act = state.action_by_seq(seq)
    _apply_snapshot(state, act.inverse["after"])
    state.undo.append(seq)
    state.redo_performed += 1
    _log_event(state, "redo", {"redone_seq": seq, "action_type": act.type.value})
    _after_history_move(state)
    return act


def _after_history_move(state: AnalysisState) -> None:
    a = state.analysis
    revalidate_approvals(a)  # a restored approval is kept only if still valid
    update_counting_area(a)
    a.updated_at = now()
    state.sync_depths()
    refresh_status(state)


# ------------------------------------------------------------------ regions (PUT /regions)
def set_regions(state: AnalysisState, new_regions: list[Region], grid_state: dict | None = None) -> ReviewAction | None:
    """Replace the regions (one undoable action). `grid_state` = {"grid": GridSpec | None, "skipped_fields": [...]}
    sets the slide's grid spec and switched-off fields in the same action, so undo/redo restore them too."""
    guard_editable(state)
    a = state.analysis
    seen: set[str] = set()
    for r in new_regions:
        if not r.id or r.id in seen:
            raise ReviewError("invalid_regions", f"region id '{r.id}' is empty or duplicated", 422)
        seen.add(r.id)
        err = validate_polygon(r.polygon)
        if err:
            raise ReviewError("invalid_regions", f"region '{r.name}': {err}", 422)
    old = {r.id: r for r in a.regions}
    merged: list[Region] = []
    changed = False
    for r in new_regions:
        o = old.get(r.id)
        # approval/inspection are server-controlled: client values are ignored
        nr = Region(id=r.id, kind=r.kind, name=r.name, polygon=[tuple(p) for p in r.polygon],
                    field_index=r.field_index if r.kind == RegionKind.counting else None)
        if o is not None:
            nr.approved, nr.approved_at = o.approved, o.approved_at
            nr.approval_fingerprint, nr.inspected = o.approval_fingerprint, o.inspected
            if o.kind != nr.kind or _round_poly(o.polygon) != _round_poly(nr.polygon):
                changed = True
                nr.inspected = False
        else:
            changed = True
        merged.append(nr)
    if set(old) - seen:
        changed = True
    tx = Tx(state)
    tx.touch_regions()
    a.regions = merged
    if grid_state is not None:
        tx.touch_grid()
        a.slide.grid = grid_state["grid"]
        a.slide.skipped_fields = list(grid_state["skipped_fields"])
    if changed:
        for o in a.objects:
            nid = point_region_id(a, o.x, o.y)
            if nid != o.region_id:
                tx.touch_obj(o)
                o.region_id = nid
    update_counting_area(a)
    tx.revalidate()
    act = tx.commit(
        ActionType.region_edit,
        {"regions": [r.id for r in merged], "geometry_changed": changed},
        undoable=True,
    )
    return act


def counting_area_px2(a: Analysis) -> float | None:
    """Area (px^2) of union(counting) minus union(exclusion) under the single geometry
    rule; None w/o regions/image."""
    if a.image is None or not _counting(a):
        return None
    return _masks.region_area_px(list(a.regions), (a.image.height, a.image.width))


def region_area_px2(a: Analysis, region: Region) -> float | None:
    """Pixel area of one region under the single rule: counting regions are net of all
    exclusions AND of earlier-listed counting regions (first listed wins, consistent with
    membership), so per-region areas add up to counting_area_px2."""
    if a.image is None:
        return None
    return _masks.region_areas(list(a.regions), (a.image.height, a.image.width)).get(region.id)


def update_counting_area(a: Analysis) -> None:
    a.slide.counting_area_px2 = counting_area_px2(a)


# ------------------------------------------------------------------ non-undoable events
def _log_event(state: AnalysisState, kind: str, detail: dict) -> None:
    ts = now()
    state.events.append({"ts": ts, "event": kind, **detail})
    state.analysis.updated_at = ts


def _invalidate_for_input_change(state: AnalysisState, why: str, detail: dict | None = None) -> None:
    """Inputs of detection (channel mapping / alignment) changed. Whenever there is anything
    that depends on them - a detection result, any object (also manual), any approval or
    inspection mark - approvals and inspection marks are cleared and the undo history is
    dropped (an undo would resurrect state that belongs to the old inputs). Staleness is
    derived (refresh_status). The region fingerprint also contains roles + alignment, so
    an approval could not survive this even if this function were bypassed."""
    a = state.analysis
    has_dependents = (
        a.provenance.detection_ran_at is not None
        or bool(a.objects)
        or any(r.approved or r.inspected for r in a.regions)
    )
    cleared: list[str] = []
    if has_dependents:
        for r in a.regions:
            if r.approved:
                cleared.append(r.id)
            r.approved, r.approved_at, r.approval_fingerprint = False, None, None
            r.inspected = False
        state.undo, state.redo = [], []
    _log_event(state, why, {**(detail or {}), "approvals_cleared": cleared,
                            "had_dependents": has_dependents})
    state.sync_depths()


def _swap_red_green_evidence(state: AnalysisState, ts: str) -> int:
    """Confirmed role swap (red channel <-> green channel): what the objects' 'red' and
    'green' fields hold now belongs to the other colour. Mirror red<->green classes, swap
    the evidence, keep yellow/uncertain classes, and force re-review. See RERUN_POLICY.md."""
    n = 0
    for o in state.analysis.objects:
        if o.deleted:
            continue
        o.red, o.green = o.green, o.red
        old = o.cls
        if o.cls == ObjectClass.red:
            o.cls = ObjectClass.green
        elif o.cls == ObjectClass.green:
            o.cls = ObjectClass.red
        o.reviewed = False
        _hist(o, f"red/green channel roles swapped: class {old.value} -> {o.cls.value}, "
                 "evidence swapped, needs re-review", ts)
        n += 1
    return n


def set_channel_mapping(state: AnalysisState, mapping: list[dict]) -> bool:
    """Returns True if roles changed. Requires every channel mapped; exactly one red and one green."""
    guard_editable(state)
    a = state.analysis
    if a.image is None:
        raise ReviewError("no_image", "analysis has no image", 409)
    by_index = {c.index: c for c in a.image.channels}
    roles: dict[int, str] = {}
    for m in mapping:
        i = m.get("index")
        role = m.get("role")
        if i not in by_index:
            raise ReviewError("invalid_mapping", f"unknown channel index {i!r}", 422)
        if role not in ("red", "green", "ignore"):
            raise ReviewError("invalid_mapping", f"invalid role {role!r}", 422)
        if i in roles:
            raise ReviewError("invalid_mapping", f"channel {i} mapped twice", 422)
        roles[i] = role
    if set(roles) != set(by_index):
        raise ReviewError("invalid_mapping", "every channel must be assigned a role", 422)
    vals = list(roles.values())
    if vals.count("red") != 1 or vals.count("green") != 1:
        raise ReviewError(
            "invalid_mapping", "exactly one channel must be red and one green", 422
        )
    was_confirmed = mapping_confirmed(a)
    old_roles = {i: (c.role.value if c.role else None) for i, c in by_index.items()}
    changed = (not was_confirmed) or any(old_roles[i] != r for i, r in roles.items())
    old_red = next((i for i, r in old_roles.items() if r == "red"), None)
    old_green = next((i for i, r in old_roles.items() if r == "green"), None)
    new_red = next(i for i, r in roles.items() if r == "red")
    new_green = next(i for i, r in roles.items() if r == "green")
    is_swap = was_confirmed and old_red == new_green and old_green == new_red
    from polony_app.schemas import ChannelRole

    for i, r in roles.items():
        by_index[i].role = ChannelRole(r)
        by_index[i].role_confirmed = True
    if changed:
        detail: dict = {"roles": {str(k): v for k, v in roles.items()},
                        "previous_roles": {str(k): v for k, v in old_roles.items()},
                        "was_confirmed": was_confirmed}
        if is_swap:
            detail["red_green_swapped"] = _swap_red_green_evidence(state, now())
            # The alignment is "shift applied to GREEN to match RED". After a swap the physical
            # channels trade roles, so the same correction is the inverse shift.
            prev = {"dy": int(a.alignment_dy), "dx": int(a.alignment_dx)}
            state.alignment = {"dy": -prev["dy"], "dx": -prev["dx"]}
            detail["alignment_negated"] = {"previous": prev, "now": dict(state.alignment)}
        _invalidate_for_input_change(
            state, "channel_mapping_changed" if was_confirmed else "channel_mapping_confirmed", detail
        )
    a.provenance.reporting_mode = a.reporting_mode
    refresh_status(state)
    return changed


def set_alignment(state: AnalysisState, dy: int, dx: int) -> bool:
    guard_editable(state)
    cur = state.alignment
    if int(cur.get("dy", 0)) == int(dy) and int(cur.get("dx", 0)) == int(dx):
        return False
    state.alignment = {"dy": int(dy), "dx": int(dx)}
    _invalidate_for_input_change(
        state, "alignment_changed",
        {"dy": int(dy), "dx": int(dx), "previous": {"dy": int(cur.get("dy", 0)), "dx": int(cur.get("dx", 0))}},
    )
    refresh_status(state)
    return True


def set_settings(state: AnalysisState, reporting_mode=None, slide=None, detection_request=None) -> None:
    guard_editable(state)
    a = state.analysis
    if reporting_mode is not None:
        a.reporting_mode = reporting_mode
        a.provenance.reporting_mode = reporting_mode
    if slide is not None:
        keep_area, keep_grid = a.slide.counting_area_px2, a.slide.grid
        a.slide = slide.model_copy(deep=True)
        a.slide.counting_area_px2 = keep_area  # computed from regions, not user-set
        a.slide.grid = keep_grid               # set only through PUT /grid (it owns the field regions)
    if detection_request is not None:
        a.detection_request = detection_request
    # staleness is derived: setting the parameters back to the ones used clears it again
    recompute_stale(state)
    a.updated_at = now()
    _log_event(state, "settings_changed", {})


# ------------------------------------------------------------------ re-detection merge
def apply_detection_result(
    state: AnalysisState,
    new_objects: list[DetectedObject],
    request: DetectionRequest,
    package_versions: dict[str, str],
    notes: list[str],
    engine_name: str = "bigfish",
    cancelled: Callable[[], bool] | None = None,
) -> ReviewAction:
    """Merge a detection run into the analysis per docs/app/RERUN_POLICY.md.

    Call it inside the store's edit lock and pass `cancelled`: it is checked HERE, before
    anything is mutated, so a cancelled job commits nothing (raises DetectionCancelled)."""
    guard_editable(state)
    if cancelled is not None and cancelled():
        raise DetectionCancelled("cancelled before commit")
    a = state.analysis
    tx = Tx(state)
    ts = now()
    r2 = RERUN_MERGE_RADIUS_PX ** 2

    user_deleted = set(state.user_deleted)

    def is_tombstone(o: DetectedObject) -> bool:
        # explicit marker set by the `delete` action (any origin); merge/split deletions are not
        return o.deleted and o.id in user_deleted

    live = [o for o in a.objects if not o.deleted]
    replaceable = [o for o in live if o.origin == Origin.auto and not o.reviewed]
    preserved = [o for o in live if not (o.origin == Origin.auto and not o.reviewed)]
    tombstones = [o for o in a.objects if is_tombstone(o)]
    anchors = [(o.x, o.y, "preserved") for o in preserved] + [(o.x, o.y, "tombstone") for o in tombstones]
    # a deleted object that had been moved still suppresses re-detection where it was detected
    for o in tombstones:
        for ev in (o.red, o.green):
            if ev.detected and ev.x is not None and ev.y is not None:
                anchors.append((ev.x, ev.y, "tombstone"))

    before_fp = {r.id: region_objects_fingerprint(a, r.id) for r in _counting(a)}

    for o in replaceable:
        tx.remove(o)

    existing_ids = {o.id for o in a.objects}
    idmap: dict[str, str] = {}
    kept: list[DetectedObject] = []
    dropped_pres = dropped_tomb = dropped_out = 0
    for n in new_objects:
        n = n.model_copy(deep=True)
        n.origin = Origin.auto
        n.deleted = False
        n.region_id = point_region_id(a, n.x, n.y)
        if _counting(a) and n.region_id is None:
            dropped_out += 1
            continue
        hit = next(
            (k for (ax, ay, k) in anchors if (n.x - ax) ** 2 + (n.y - ay) ** 2 <= r2), None
        )
        if hit == "preserved":
            dropped_pres += 1
            continue
        if hit == "tombstone":
            dropped_tomb += 1
            continue
        old_id = n.id
        if n.id in existing_ids or not n.id:
            n.id = _unique_id("a", existing_ids)
        existing_ids.add(n.id)
        idmap[old_id] = n.id
        n.history = list(n.history) + [f"{ts} created by detection"]
        kept.append(n)
    for n in kept:
        n.ambiguous_with = [idmap[i] for i in n.ambiguous_with if i in idmap]
        tx.new(n)

    tx.touch_prov()
    p = a.provenance
    p.detection_request = request.model_copy(deep=True)
    p.detection_ran_at = ts
    p.package_versions = dict(package_versions)
    p.engine = engine_name
    p.reporting_mode = a.reporting_mode
    # signature of what this result was produced from (the `request` actually used + current
    # regions/roles/alignment). Staleness is derived against the CURRENT analysis settings in
    # refresh_status (called by commit): a settings change during the job => stale.
    state.detection_sig = input_signature(a, request)
    recompute_stale(state)

    # inspection of omissions is void for any region whose objects changed
    for r in _counting(a):
        if before_fp.get(r.id) != region_objects_fingerprint(a, r.id) and r.inspected:
            tx.touch_regions()
            r.inspected = False
    tx.revalidate()
    summary = {
        "replaced_auto_objects": len(replaceable),
        "new_auto_objects": len(kept),
        "preserved_objects": len(preserved),
        "dropped_near_preserved": dropped_pres,
        "dropped_near_deleted": dropped_tomb,
        "dropped_outside_valid_area": dropped_out,
        "merge_radius_px": RERUN_MERGE_RADIUS_PX,
        "notes": list(notes),
    }
    return tx.commit(ActionType.rerun_detection, summary)


# ------------------------------------------------------------------ finalize / reopen
def finalize_problems(x: "Analysis | AnalysisState") -> list[tuple[str, str]]:
    """Why the analysis cannot be finalized. Accepts the state (preferred: also compares the
    recorded detection-input signature) or a bare Analysis."""
    state = x if isinstance(x, AnalysisState) else None
    a = state.analysis if state is not None else x
    probs = []
    counting = _counting(a)
    if not counting:
        probs.append(("no_counting_region", "define at least one counting region"))
    if not mapping_confirmed(a):
        probs.append(("channel_mapping_unconfirmed", "channel mapping has not been confirmed"))
    sig = state.detection_sig if state is not None else None
    if request_differs(a):
        probs.append(("stale_detection",
                      "the detection parameters now set differ from those the shown result was "
                      "produced with; re-run detection (or set them back)"))
    elif a.provenance.stale_detection or (
        state is not None and compute_stale(a, sig, a.provenance.stale_detection)
    ):
        probs.append(("stale_detection", "detection is stale; re-run detection"))
    if a.provenance.detection_ran_at is None and not any(not o.deleted for o in a.objects):
        probs.append(("no_detection", "no detection has been run and there are no objects"))
    unapproved = [r.name for r in counting if not r.approved]
    if unapproved:
        probs.append(("regions_not_approved", "regions not approved: " + ", ".join(unapproved)))
    outdated = [
        r.name for r in counting
        if r.approved and r.approval_fingerprint != region_fingerprint(a, r)
    ]
    if outdated:
        probs.append(("approval_outdated",
                      "approval no longer matches the region content: " + ", ".join(outdated)))
    return probs


def finalize(state: AnalysisState) -> ReviewAction:
    guard_editable(state)
    a = state.analysis
    probs = finalize_problems(state)
    if probs:
        raise ReviewError(
            probs[0][0],
            "cannot finalize: " + "; ".join(m for _, m in probs),
            409,
            {"problems": [{"code": c, "message": m} for c, m in probs]},
        )
    ts = now()
    a.finalized_at = ts
    a.updated_at = ts
    act = ReviewAction(seq=state.next_seq, ts=ts, type=ActionType.finalize, payload={}, inverse={})
    state.next_seq += 1
    state.log.append(act)
    state.undo, state.redo = [], []
    state.sync_depths()
    refresh_status(state)
    return act


def reopen(state: AnalysisState) -> None:
    """Return a finalized analysis to review. Approvals are KEPT (their
    fingerprints still guard them: any later edit inside an approved region
    clears that approval). finalized_at is cleared; the finalize/reopen are in the log."""
    a = state.analysis
    if a.finalized_at is None:
        raise ReviewError("not_finalized", "analysis is not finalized", 409)
    ts = now()
    act = ReviewAction(
        seq=state.next_seq,
        ts=ts,
        type=ActionType.reopen,
        payload={"previous_finalized_at": a.finalized_at},
        inverse={},
    )
    state.next_seq += 1
    state.log.append(act)
    a.finalized_at = None
    a.updated_at = ts
    refresh_status(state)
    if a.status == AnalysisStatus.finalized:  # pragma: no cover
        a.status = AnalysisStatus.in_review
    if a.status in (AnalysisStatus.ready, AnalysisStatus.created):
        a.status = AnalysisStatus.in_review


def action_log_summary(state: AnalysisState) -> dict:
    by_type: dict[str, int] = {}
    for a in state.log:
        key = a.type.value
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "total_actions": len(state.log),
        "by_type": by_type,
        "undo_performed": state.undo_performed,
        "redo_performed": state.redo_performed,
        "first_action_at": state.log[0].ts if state.log else None,
        "last_action_at": state.log[-1].ts if state.log else None,
        "events": list(state.events),  # complete, never truncated
        "rerun_summaries": [a.payload for a in state.log if a.type == ActionType.rerun_detection],
    }
