"""Mutable per-analysis state: the Analysis plus review bookkeeping that the
shared schema has no field for (action log, undo/redo stacks, alignment)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polony_app.schemas import Analysis, ReviewAction


class ReviewError(Exception):
    """Raised for refused operations. Maps to an ApiError response."""

    def __init__(self, code: str, message: str, status: int = 409, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


class DetectionCancelled(Exception):
    """Raised by apply_detection_result when the job was cancelled before the commit
    (the job runner recognises this class by name -> job ends as 'cancelled', nothing committed)."""


@dataclass
class AnalysisState:
    analysis: Analysis
    log: list[ReviewAction] = field(default_factory=list)
    undo: list[int] = field(default_factory=list)   # seqs, last = next to undo
    redo: list[int] = field(default_factory=list)
    next_seq: int = 1
    undo_performed: int = 0
    redo_performed: int = 0
    prev_status: str | None = None
    events: list[dict] = field(default_factory=list)  # non-undoable events (mapping, reopen...)
    # --- engine bookkeeping (the shared schema has no field for these) ---
    # Inputs (request, regions, channel roles, alignment) that the CURRENT detection result was
    # produced from. Staleness is DERIVED: stale <=> current inputs != this signature.
    detection_sig: dict | None = None
    # Ids of objects the user deleted with the `delete` action (explicit marker; any origin).
    # Only these soft-deleted objects act as tombstones on re-detection.
    user_deleted: list[str] = field(default_factory=list)
    # A detection job is running (persisted; cleared by refresh_status(job_running=False)).
    job_active: bool = False

    @property
    def alignment(self) -> dict[str, Any]:
        return {"dy": self.analysis.alignment_dy, "dx": self.analysis.alignment_dx}

    @alignment.setter
    def alignment(self, v: dict[str, Any]) -> None:
        self.analysis.alignment_dy = int(v.get("dy", 0))
        self.analysis.alignment_dx = int(v.get("dx", 0))

    def action_by_seq(self, seq: int) -> ReviewAction:
        for a in reversed(self.log):
            if a.seq == seq:
                return a
        raise KeyError(seq)

    def sync_depths(self) -> None:
        self.analysis.undo_depth = len(self.undo)
        self.analysis.redo_depth = len(self.redo)

    def to_json(self) -> dict:
        return {
            "analysis": self.analysis.model_dump(mode="json"),
            "undo": self.undo,
            "redo": self.redo,
            "next_seq": self.next_seq,
            "undo_performed": self.undo_performed,
            "redo_performed": self.redo_performed,
            "prev_status": self.prev_status,
            "events": self.events,
            "detection_sig": self.detection_sig,
            "user_deleted": self.user_deleted,
            "job_active": self.job_active,
        }

    @classmethod
    def from_json(cls, d: dict, log: list[ReviewAction] | None = None) -> "AnalysisState":
        st = cls(analysis=Analysis.model_validate(d["analysis"]))
        st.log = list(log or [])
        st.undo = list(d.get("undo", []))
        st.redo = list(d.get("redo", []))
        st.next_seq = d.get("next_seq", 1)
        if "alignment" in d and not (st.analysis.alignment_dy or st.analysis.alignment_dx):
            st.alignment = d["alignment"]  # legacy persisted form
        st.undo_performed = d.get("undo_performed", 0)
        st.redo_performed = d.get("redo_performed", 0)
        st.prev_status = d.get("prev_status")
        st.events = d.get("events", [])
        st.detection_sig = d.get("detection_sig")
        st.user_deleted = list(d.get("user_deleted", []))
        st.job_active = bool(d.get("job_active", False))
        return st
