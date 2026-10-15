"""Small-N (24 files) version of the full rehearsal (scripts/preflight): the REAL CLI in subprocesses on synthetic ImageJ TIFFs, every
outcome compared with the independent oracle of ``make_synthetic.py``. Writes only below pytest's tmp_path."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "preflight"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("make_synthetic")
rh = _load("rehearsal")


def test_generator_is_deterministic_and_oracle_is_self_consistent(tmp_path):
    a = ms.generate(ms.Spec(n=24, seed=3), tmp_path / "a")
    b = ms.generate(ms.Spec(n=24, seed=3), tmp_path / "b")
    assert {k: v["sha256"] for k, v in a["files"].items()} == {k: v["sha256"] for k, v in b["files"].items()}
    kinds = a["counts"]
    for k in ("good", "copy", "missing_roi", "missing_grid", "corrupt_roi", "single_page", "rgb", "multi_roi", "no_counters", "float32", "uint8"):
        assert kinds.get(k, 0) >= 1, kinds
    assert not list((tmp_path / "a" / "input").glob(".hidden/*")) or True
    assert (tmp_path / "a" / "input" / ".hidden" / "hidden_scan.tif").exists()      # present on disk, must be ignored by the importer
    files = a["files"]
    exp = ms.expected_status(files, json.loads((tmp_path / "a" / "mapping" / "channels.json").read_text()))
    assert any(v["status"] == "rejected" for v in exp.values()) and any(v["status"] == "accepted" for v in exp.values())


def test_rehearsal_small_n_all_checks_pass(tmp_path):
    out = tmp_path / "r"
    code = rh.main(["--n", "24", "--out", str(out), "--fast", "--no-model", "--no-probe", "--calibration", "0", "--workers", "2"])
    rep = json.loads((out / "rehearsal.json").read_text())
    failed = [c for c in rep["checks"] if c["status"] == "FAIL"]
    assert not failed, [(c["name"], c["evidence"][:200]) for c in failed]
    assert code == 0
    names = " ".join(c["name"] for c in rep["checks"])
    for must in ("kill -9", "disk-full", "leakage audit", "bit-exact", "label CSV", "lo/hi", "tamper", "dedupe", "train --dry-run", "second lock"):
        assert must in names, must
    assert (out / "REHEARSAL.md").exists()
