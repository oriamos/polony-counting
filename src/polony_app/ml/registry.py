"""Model registry: ``<data_dir>/models/registry.json`` (atomic writes, cross-process file lock).

Entry fields: id, kind (pretrained | finetuned), base, dataset_version, split membership, hyperparameters, seed, duration,
hardware, metrics, weights sha256, status (candidate | active | retired), promoted_at, previous_active, evaluations
(on the locked test set), promotion record.

The registry never trains and never promotes by itself: ``promote`` and ``rollback`` are explicit calls.
``resolve_model`` is the single function the engines use to turn ``SpotiflowParams.model_name`` into an artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .common import MLError, atomic_write_text, file_lock, models_dir, pretty_json, registry_path, sha256_file, utc_now

REGISTRY_SCHEMA = 1
DEFAULT_PRETRAINED = "general"
STATUSES = ("candidate", "active", "retired")
PRIMARY_METRIC = "f1"


def pretrained_id(name: str) -> str:
    return f"pretrained-{name}"


def _empty() -> dict:
    return {"schema": REGISTRY_SCHEMA, "active": None, "models": {}, "log": []}


@dataclass
class ModelRef:
    """What the engines need to load a model."""
    name: str                           # as requested
    kind: str                           # "pretrained" | "finetuned"
    registry_id: Optional[str]
    path: Optional[Path]                # folder with config.yaml + weights (None for official pretrained names)
    weights_sha256: Optional[str] = None
    status: Optional[str] = None
    base: Optional[str] = None
    dataset_version: Optional[str] = None
    info: dict = field(default_factory=dict)


class Registry:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.path = registry_path(self.data_dir)
        self.lock_path = self.path.with_suffix(".lock")

    # ---------------------------------------------------------------- io
    def load(self) -> dict:
        if not self.path.is_file():
            return _empty()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MLError(f"{self.path} is unreadable ({exc}); not touching it")
        if d.get("schema") != REGISTRY_SCHEMA:
            raise MLError(f"{self.path}: unsupported registry schema {d.get('schema')!r}")
        return d

    def _save(self, d: dict) -> None:
        atomic_write_text(self.path, pretty_json(d))

    def _mutate(self, fn):
        with file_lock(self.lock_path):
            d = self.load()
            out = fn(d)
            self._save(d)
            return out

    # ---------------------------------------------------------------- queries
    def get(self, model_id: str) -> dict:
        d = self.load()
        if model_id not in d["models"]:
            raise MLError(f"model '{model_id}' is not in the registry ({self.path})"
                          + (f"; known: {', '.join(sorted(d['models']))}" if d["models"] else "; the registry is empty"))
        return d["models"][model_id]

    def list(self) -> list[dict]:
        d = self.load()
        return [dict(e, is_active=(e["id"] == d["active"])) for e in sorted(d["models"].values(), key=lambda e: e["created_at"])]

    def active_id(self) -> Optional[str]:
        return self.load()["active"]

    # ---------------------------------------------------------------- mutations
    def add(self, entry: dict) -> dict:
        def fn(d):
            if entry["id"] in d["models"]:
                raise MLError(f"model id '{entry['id']}' already exists; registry entries are never overwritten")
            e = dict(entry)
            e.setdefault("status", "candidate")
            e.setdefault("created_at", utc_now())
            e.setdefault("evaluations", [])
            e.setdefault("promoted_at", None)
            e.setdefault("previous_active", None)
            d["models"][e["id"]] = e
            d["log"].append({"at": utc_now(), "event": "add", "id": e["id"]})
            return e
        return self._mutate(fn)

    def ensure_pretrained(self, name: str) -> dict:
        """Entry for an official pretrained model (created on first use; status retired = not in service by promotion)."""
        mid = pretrained_id(name)

        def fn(d):
            if mid not in d["models"]:
                d["models"][mid] = {"id": mid, "kind": "pretrained", "base": None, "name": name, "dataset_version": None,
                                    "status": "retired", "created_at": utc_now(), "evaluations": [], "promoted_at": None,
                                    "previous_active": None, "weights_sha256": None, "artifact": None,
                                    "notes": "official pretrained model (not trained here)"}
                d["log"].append({"at": utc_now(), "event": "add", "id": mid})
            return d["models"][mid]
        return self._mutate(fn)

    def add_evaluation(self, model_id: str, evaluation: dict) -> None:
        def fn(d):
            if model_id not in d["models"]:
                raise MLError(f"model '{model_id}' is not in the registry")
            ev = d["models"][model_id].setdefault("evaluations", [])
            ev[:] = [x for x in ev if x.get("lock_sha256") != evaluation.get("lock_sha256")
                     or x.get("params") != evaluation.get("params")]
            ev.append(evaluation)
            d["log"].append({"at": utc_now(), "event": "evaluate", "id": model_id,
                             "lock_sha256": evaluation.get("lock_sha256")})
        self._mutate(fn)

    def evaluation_on(self, model_id: str, lock_sha256: str) -> Optional[dict]:
        evs = [e for e in self.get(model_id).get("evaluations", []) if e.get("lock_sha256") == lock_sha256]
        return evs[-1] if evs else None

    def promote(self, model_id: str, *, lock_sha256: Optional[str] = None, force: bool = False, reason: str = "") -> dict:
        """Make ``model_id`` the active model. Requires a locked-test evaluation of the candidate AND of the model it
        replaces (the active one, else the app default pretrained 'general'), and that the candidate is not worse on F1.
        ``force`` bypasses these checks but needs a recorded reason."""
        if force and not reason.strip():
            raise MLError("--force requires --reason (the override is recorded in the registry)")

        def fn(d):
            if model_id not in d["models"]:
                raise MLError(f"model '{model_id}' is not in the registry")
            cand = d["models"][model_id]
            if cand["status"] == "active":
                raise MLError(f"model '{model_id}' is already the active model")
            cur = d["active"]
            base_id = cur or pretrained_id(DEFAULT_PRETRAINED)
            problems: list[str] = []
            lock = lock_sha256
            if not lock:
                problems.append("no locked test set exists (ml split --lock-test); promotion needs evaluation on it")
            comparison: dict = {"candidate": model_id, "baseline": base_id, "lock_sha256": lock}
            if lock:
                ev_c = _find_eval(cand, lock)
                if ev_c is None:
                    problems.append(f"'{model_id}' has no evaluation on the current locked test set "
                                    f"(run: ml eval-model --model {model_id})")
                base = d["models"].get(base_id)
                ev_b = _find_eval(base, lock) if base else None
                if ev_b is None:
                    problems.append(f"the model it would replace ('{base_id}') has no evaluation on the locked test set, so "
                                    f"there is nothing to compare against (run: ml eval-model --model {base_id})")
                if ev_c and ev_b:
                    mc, mb = ev_c["metrics"], ev_b["metrics"]
                    comparison.update({"candidate_metrics": mc, "baseline_metrics": mb,
                                       "delta": {k: (None if mc.get(k) is None or mb.get(k) is None else mc[k] - mb[k])
                                                 for k in ("precision", "recall", "f1", "mean_match_distance_px")}})
                    fc, fb = mc.get(PRIMARY_METRIC), mb.get(PRIMARY_METRIC)
                    if fc is None or fb is None:
                        problems.append("the evaluations contain no F1 value to compare")
                    elif fc < fb:
                        problems.append(f"the candidate is WORSE than '{base_id}' on the locked test set "
                                        f"(F1 {fc:.4f} < {fb:.4f})")
            if problems and not force:
                raise MLError("promotion refused: " + "; ".join(problems) + ". (--force --reason overrides, and is recorded.)")
            if cur and cur in d["models"]:
                d["models"][cur]["status"] = "retired"
            cand["status"] = "active"
            cand["promoted_at"] = utc_now()
            cand["previous_active"] = cur
            cand["promotion"] = {"forced": bool(problems and force), "reason": reason, "overridden_checks": problems if force else [],
                                 "comparison": comparison}
            d["active"] = model_id
            d["log"].append({"at": utc_now(), "event": "promote", "id": model_id, "previous_active": cur,
                             "forced": bool(problems and force), "reason": reason})
            return cand
        return self._mutate(fn)

    def rollback(self) -> dict:
        """Restore the model that was active before the current one (None = back to the app default)."""
        def fn(d):
            cur = d["active"]
            if not cur:
                raise MLError("nothing to roll back: no model has been promoted (the app default is in use)")
            e = d["models"][cur]
            prev = e.get("previous_active")
            e["status"] = "retired"
            e["rolled_back_at"] = utc_now()
            d["active"] = prev
            if prev and prev in d["models"]:
                d["models"][prev]["status"] = "active"
            d["log"].append({"at": utc_now(), "event": "rollback", "from": cur, "to": prev})
            return {"rolled_back": cur, "restored": prev}
        return self._mutate(fn)


def _find_eval(entry: Optional[dict], lock_sha256: str) -> Optional[dict]:
    if not entry:
        return None
    evs = [e for e in entry.get("evaluations", []) if e.get("lock_sha256") == lock_sha256 and e.get("metrics")]
    return evs[-1] if evs else None


# ---------------------------------------------------------------------- engine entry point
def resolve_model(name: str, data_dir: str | Path) -> ModelRef:
    """Turn ``SpotiflowParams.model_name`` into an artifact.

    * a registry id (candidate, active or retired)   -> kind 'finetuned', ``path`` = the weights folder
    * ``"active"``                                   -> the promoted model, or the default pretrained one when none
    * anything else                                  -> kind 'pretrained' (an official name or a folder path); nothing is
                                                        looked up or downloaded here
    Raises :class:`MLError` when a registry entry exists but its files are missing or were modified."""
    reg = Registry(data_dir)
    d = reg.load()
    if name == "active":
        if d["active"] is None:
            return ModelRef(DEFAULT_PRETRAINED, "pretrained", None, None, info={"note": "no promoted model; default pretrained"})
        name = d["active"]
    e = d["models"].get(name)
    if e is None or e.get("kind") == "pretrained":
        n = e["name"] if e else name
        return ModelRef(n, "pretrained", pretrained_id(n) if e else None, None, status=(e or {}).get("status"))
    art = e.get("artifact")
    if not art:
        raise MLError(f"registry model '{name}' has no weights artifact")
    path = models_dir(data_dir) / art
    w = path / e.get("weights_file", "best.pt")
    if not w.is_file():
        raise MLError(f"weights of registry model '{name}' are missing: {w}")
    if e.get("weights_sha256") and sha256_file(w) != e["weights_sha256"]:
        raise MLError(f"weights of registry model '{name}' do not match the recorded sha256 (file modified): {w}")
    return ModelRef(name, "finetuned", name, path, weights_sha256=e.get("weights_sha256"), status=e.get("status"),
                    base=e.get("base"), dataset_version=e.get("dataset_version"),
                    info={"seed": e.get("seed"), "hyperparameters": e.get("hyperparameters")})
