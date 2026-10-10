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


@dataclass
class AnalysisState:
    analysis: Analysis
    log: list[ReviewAction] = field(default_factory=list)
    undo: list[int] = field(default_factory=list)   # seqs, last = next to undo
    redo: list[int] = field(default_factory=list)
    next_seq: int = 1
    alignment: dict[str, Any] = field(default_factory=lambda: {"dy": 0, "dx": 0})
    undo_performed: int = 0
    redo_performed: int = 0
    prev_status: str | None = None
    events: list[dict] = field(default_factory=list)  # non-undoable events (mapping, reopen...)

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
            "alignment": self.alignment,
            "undo_performed": self.undo_performed,
            "redo_performed": self.redo_performed,
            "prev_status": self.prev_status,
            "events": self.events,
        }

    @classmethod
    def from_json(cls, d: dict, log: list[ReviewAction] | None = None) -> "AnalysisState":
        st = cls(analysis=Analysis.model_validate(d["analysis"]))
        st.log = list(log or [])
        st.undo = list(d.get("undo", []))
        st.redo = list(d.get("redo", []))
        st.next_seq = d.get("next_seq", 1)
        st.alignment = d.get("alignment", {"dy": 0, "dx": 0})
        st.undo_performed = d.get("undo_performed", 0)
        st.redo_performed = d.get("redo_performed", 0)
        st.prev_status = d.get("prev_status")
        st.events = d.get("events", [])
        return st
