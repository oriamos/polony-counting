"""The whole researcher workflow on a tiny synthetic corpus, driven through the CLI exactly as a user would:

    import-legacy --dry-run -> import-legacy -> verify -> split -> split --lock-test -> train --dry-run -> train (2 epochs, CPU)
    -> eval-model <candidate> -> eval-model general -> compare -> promote (refusals and acceptance, never forced) -> rollback

No network: the official pretrained model 'general' cannot be downloaded in the test, so ``load_model('general')`` is redirected to a
second, randomly initialised tiny model folder (the same trick the existing e2e tests use for the base). A 2-epoch model on random
weights has no meaningful F1, therefore the promotion logic is exercised with CONTROLLED evaluation records written through the public
``Registry.add_evaluation`` API (the real evaluations are in the registry too, and checked for consistency).
"""
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from legacy_bridge_fixtures import GRID_XS, GRID_YS, RED_GREEN, T0, H, W, marks_in, write_channel_map
from legacy_fixtures import grid_overlay, point_roi, write_stack
from polony_app.ml import dataset as ds
from polony_app.ml import legacy_bridge as lb
from polony_app.ml.cli import main
from polony_app.ml.registry import Registry
from polony_app.ml.splits import load_lock

BUDGET_S = 90.0


def _ml_importable():
    try:
        import pytorch_lightning  # noqa: F401
        import spotiflow.model  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ml_importable(), reason="spotiflow / torch / pytorch-lightning cannot be imported")

# 6 two-page stacks over 4 specimens (AM x2, AS, FS x2, PS)
SLIDES = {"AM-A": 1, "AM-B": 2, "AS-A": 3, "FS-A": 4, "FS-B": 5, "PS-A": 6}
MARKED = {0: ((0, 0, 4, 0), (0, 1, 3, 1), (1, 0, 3, 2)), 1: ((0, 0, 3, 0),)}      # page -> (row, col, n, counter)


def spot_stack(path: Path, seed: int):
    """Two 16-bit pages with real bright spots at every ImageJ mark (so a model has something to learn), grid overlay + ROI."""
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(seed)
    marks, planes = [], []
    for page in (0, 1):
        img = rng.normal(500, 30, (H, W))
        for (r, c, n, ctr) in MARKED[page]:
            for (x, y, _c, _p) in marks_in(r, c, n, ctr, page + 1, seed + 3 * page):
                marks.append((x, y, ctr, page + 1))
                img += 9000 * np.exp(-((xx - (x - 0.5)) ** 2 + (yy - (y - 0.5)) ** 2) / (2 * 2.0 ** 2))
        planes.append(np.clip(img, 0, 65535).astype(np.uint16))
    roi = point_roi([(m[0], m[1]) for m in marks], [m[2] for m in marks], [m[3] for m in marks])
    write_stack(path, planes, roi=roi, overlays=[grid_overlay(GRID_XS, GRID_YS, W, H)])
    return len(marks)


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def tree_hashes(root: Path):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def test_full_legacy_workflow_from_tiffs_to_a_guarded_promotion(tmp_path, capsys, monkeypatch):
    timings = {}

    def timed(name):
        class T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                timings[name] = time.perf_counter() - self.t
        return T()

    t_start = time.perf_counter()
    root, d = tmp_path / "scans", tmp_path / "data"
    root.mkdir()
    n_marks = 0
    rows = []
    for name, seed in SLIDES.items():
        rel = f"29-09-2026_T7_AB_{name}.tif"
        n_marks = spot_stack(root / rel, seed)
        rows.append({"file": rel, "specimen_id": name.split("-")[0], "slide_id": name.replace("-", "_"), "assay": "T7_AB",
                     "replicate": name.split("-")[1], "scan_session": "2026-09-30", "confirmed": True})
    imap = tmp_path / "identities.json"
    imap.write_text(json.dumps({"rows": rows}, indent=2))
    cmap = write_channel_map(tmp_path / "channels.json", defaults=RED_GREEN)
    n_examined_per_file = sum(len(v) for v in MARKED.values())                    # 4 (field, page) tuples with marks
    n_tuples = len(SLIDES) * 9 * 2
    n_unproven = n_tuples - len(SLIDES) * n_examined_per_file
    common = ["--input", root, "--channel-map", cmap, "--identity-map", imap, "--fractions", "0.5,0.25,0.25", "--seed", 1,
              "--created-at", T0, "--batch-size", 2, "--epochs", 2]

    # ---------------------------------------------------------------- 1. dry run: nothing written, exact numbers
    before = tree_hashes(tmp_path)
    with timed("import --dry-run"):
        code, out, err = run(capsys, "--data-dir", d, "import-legacy", *common, "--dry-run")
    assert code == 0, err
    assert tree_hashes(tmp_path) == before and not d.exists(), "a dry run wrote something"
    m = re.search(r"EXCLUDED because the researcher's examination could not be proven: (\d+) \(unknown (\d+), conflict (\d+)\)", out)
    assert m and tuple(map(int, m.groups())) == (n_unproven, n_unproven, 0), out
    assert f"positive_marked {len(SLIDES) * n_examined_per_file}" in out and "no blockers" in out
    assert "SOURCES: discovered 6, unique 6, accepted 6, rejected 0" in out and "4 independent group(s)" in out

    # ---------------------------------------------------------------- 2. real import
    with timed("import-legacy"):
        code, out, err = run(capsys, "--data-dir", d, "import-legacy", *common, "--workers", 2)
    assert code == 0, err
    assert f"{n_unproven} excluded because examination could not be proven" in out
    (version,) = ds.list_snapshots(d)
    man = ds.load_manifest(d, version)
    by_split = {s: [c for c in man.crops if c.split == s] for s in ("train", "val", "test")}
    assert all(by_split.values()), {k: len(v) for k, v in by_split.items()}                # all three splits populated
    assert len(man.crops) == len(SLIDES) * n_examined_per_file and lb.verify_examination_provenance(man) == []
    assert all(c.provenance["researcher_examined"] is True and c.norm_lo is not None for c in man.crops)
    spec_split = {}
    for s in man.sources:
        spec_split.setdefault(s.identity["specimen_id"], set()).add(s.split)
    assert all(len(v) == 1 for v in spec_split.values()), spec_split                        # no specimen straddles

    # ---------------------------------------------------------------- 3. verify, split, lock
    with timed("verify"):
        code, out, err = run(capsys, "--data-dir", d, "verify", version)
    assert code == 0 and "intact" in out
    code, out, err = run(capsys, "--data-dir", d, "split", "--snapshot", version)
    assert code == 0 and not (d / "datasets" / "test_lock.json").exists()
    with timed("split --lock-test"):
        code, out, err = run(capsys, "--data-dir", d, "split", "--snapshot", version, "--lock-test")
    assert code == 0 and "LOCKED test set" in out
    lock = load_lock(d)
    assert lock["snapshot"] == version and set(lock["groups"]) == {c.group_id for c in by_split["test"]}
    test_before = {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in by_split["test"]}
    # a second lock is refused (permanent)
    code, out, err = run(capsys, "--data-dir", d, "split", "--snapshot", version, "--lock-test")
    assert code == 2 and "permanent" in err

    # ---------------------------------------------------------------- 4. models: a tiny trainable base and a stand-in for 'general'
    from spotiflow.model import Spotiflow, SpotiflowModelConfig
    base = tmp_path / "tiny_base"
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(base, which="best")
    stand_in = tmp_path / "general_stand_in"
    import torch
    torch.manual_seed(123)
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(stand_in, which="best")
    from polony_app.ml import spotiflow_io as sio
    real_load = sio.load_model

    def load_model(b, data_dir, **kw):
        return real_load(str(stand_in) if b == "general" else b, data_dir, **kw)       # no download, ever
    monkeypatch.setattr(sio, "load_model", load_model)

    # ---------------------------------------------------------------- 5. train --dry-run writes nothing
    targs = ["--snapshot", version, "--base", base, "--epochs", 2, "--batch-size", 2, "--crop-size", 32, "--allow-small",
             "--device", "cpu"]
    before = tree_hashes(tmp_path)
    with timed("train --dry-run"):
        code, out, err = run(capsys, "--data-dir", d, "train", *targs, "--dry-run")
    assert code == 0, err
    assert "nothing was trained" in out and tree_hashes(tmp_path) == before and not (d / "models").exists()

    # ---------------------------------------------------------------- 6. the tiny CPU fine-tune
    with timed("train (2 epochs, cpu)"):
        code, out, err = run(capsys, "--data-dir", d, "train", *targs)
    assert code == 0, err
    mid = re.search(r"model id: (\S+)", out).group(1)
    reg = Registry(d)
    entry = reg.get(mid)
    assert entry["status"] == "candidate" and reg.active_id() is None
    assert entry["metrics"]["training"]["best_val_loss"] is not None and entry["dataset_version"] == version
    sm = entry["split_membership"]
    test_ids = {c.crop_id for c in by_split["test"]}
    assert not (set(sm["train"]) | set(sm["val"])) & test_ids, "test crops leaked into the fine-tune"
    assert not set(sm["train_groups"]) & set(lock["groups"]) and not set(sm["val_groups"]) & set(lock["groups"])
    assert sm["test_excluded"] == len(by_split["test"]) and sm["lock_sha256"] == lock["lock_sha256"]
    assert set(sm["train"]) == {c.crop_id for c in by_split["train"]} and set(sm["val"]) == {c.crop_id for c in by_split["val"]}

    # ---------------------------------------------------------------- 7. promotion is impossible before the evaluations
    code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    assert code == 2 and "has no evaluation on the current locked test set" in err
    assert Registry(d).active_id() is None

    # ---------------------------------------------------------------- 8. eval-model candidate AND general, on the locked set
    with timed("eval-model candidate"):
        code, out, err = run(capsys, "--data-dir", d, "eval-model", "--model", mid, "--device", "cpu")
    assert code == 0, err
    code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    assert code == 2 and "nothing to compare against" in err and "general" in err           # the model it replaces was not evaluated
    with timed("eval-model general"):
        code, out, err = run(capsys, "--data-dir", d, "eval-model", "--model", "general", "--device", "cpu")
    assert code == 0, err
    reg = Registry(d)
    ev_c = reg.evaluation_on(mid, lock["lock_sha256"])
    gid = next(e["id"] for e in reg.list() if e["id"] != mid)
    ev_g = reg.evaluation_on(gid, lock["lock_sha256"])
    for ev in (ev_c, ev_g):
        assert ev["split"] == "test" and ev["snapshot"] == version and ev["n_crops"] == len(by_split["test"])
        assert ev["lock_sha256"] == lock["lock_sha256"] and ev["metrics"]["n_fields"] == len(by_split["test"])
        assert set(ev["metrics"]["by_tolerance"]) == {"5", "10", "15"}
        assert ev["config"] == {"downscale": 1, "normaliser": "valid_unsaturated"}

    # ---------------------------------------------------------------- 9. compare
    code, out, err = run(capsys, "--data-dir", d, "compare", "--models", mid, "general")
    assert code == 0, err
    assert mid in out and gid in out and "| model |" in out and "F1 @10 px" in out
    code, out, err = run(capsys, "--data-dir", d, "compare", "--models", mid, "nonexistent-model")
    assert code == 2

    # ---------------------------------------------------------------- 10. promote: the real evaluations, consistent either way
    f1 = lambda ev: ev["metrics"]["by_tolerance"]["10"].get("f1")           # noqa: E731
    code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    if f1(ev_c) is None or f1(ev_g) is None:
        assert code == 2 and "no F1 value" in err
    elif f1(ev_c) < f1(ev_g):
        assert code == 2 and "WORSE" in err
    else:
        assert code == 0 and Registry(d).active_id() == mid
        code, out, err = run(capsys, "--data-dir", d, "rollback")
        assert code == 0
    assert "--force" in err or code == 0
    assert Registry(d).active_id() is None

    # ---------------------------------------------------------------- 11. promote with CONTROLLED numbers (never --force)
    def controlled(model_id, base_ev, f1_value, tag):
        ev = json.loads(json.dumps(base_ev))
        for k, v in ev["metrics"]["by_tolerance"].items():
            v.update(f1=f1_value, precision=f1_value, recall=f1_value)
        ev["metrics"].update(f1=f1_value, precision=f1_value, recall=f1_value)
        ev["params"] = dict(ev["params"], controlled=tag)                 # a different params key: it is added, not replaced
        Registry(d).add_evaluation(model_id, ev)

    controlled(gid, ev_g, 0.60, "baseline")
    controlled(mid, ev_c, 0.40, "worse")
    code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    assert code == 2 and "WORSE" in err and Registry(d).active_id() is None
    # a different app configuration is refused; acknowledging it does NOT override the missing / worse evidence
    code, out, err = run(capsys, "--data-dir", d, "promote", mid, "--app-downscale", 2)
    assert code == 2 and "configuration" in err and Registry(d).active_id() is None
    code, out, err = run(capsys, "--data-dir", d, "promote", mid, "--app-downscale", 2, "--acknowledge-config")
    assert code == 2 and ("WORSE" in err or "has no evaluation" in err) and Registry(d).active_id() is None
    # --force without a reason is itself refused
    code, out, err = run(capsys, "--data-dir", d, "promote", mid, "--force")
    assert code == 2 and "--reason" in err and Registry(d).active_id() is None
    controlled(mid, ev_c, 0.75, "better")
    with timed("promote"):
        code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    assert code == 0, err
    reg = Registry(d)
    assert reg.active_id() == mid and reg.get(mid)["promotion"]["forced"] is False and reg.get(mid)["promotion"]["overridden_checks"] == []
    assert abs(reg.get(mid)["promotion"]["comparison"]["delta"]["f1"] - 0.15) < 1e-9
    code, out, err = run(capsys, "--data-dir", d, "promote", mid)
    assert code == 2 and "already the active" in err
    code, out, err = run(capsys, "--data-dir", d, "rollback")
    assert code == 0 and Registry(d).active_id() is None

    # ---------------------------------------------------------------- 12. the locked test set never moved
    assert ds.verify_snapshot(d, version) == []
    assert {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in ds.load_manifest(d, version).crops if c.split == "test"} == test_before
    assert load_lock(d)["lock_sha256"] == lock["lock_sha256"]
    # a LATER import (new data) on top of the lock carries the very same test crops, byte for byte, and train still refuses nothing
    more = tmp_path / "more"
    more.mkdir()
    spot_stack(more / "29-10-2026_T7_AB_QS-A.tif", 77)
    rows2 = [{"file": "29-10-2026_T7_AB_QS-A.tif", "specimen_id": "QS", "slide_id": "QS_A", "assay": "T7_AB", "confirmed": True}]
    imap2 = tmp_path / "identities2.json"
    imap2.write_text(json.dumps({"rows": rows2}))
    code, out, err = run(capsys, "--data-dir", d, "import-legacy", "--input", more, "--channel-map", cmap, "--identity-map", imap2,
                         "--fractions", "0.5,0.5,0", "--created-at", T0)
    assert code == 0, err
    v2 = [v for v in ds.list_snapshots(d) if v != version][0]
    m2 = ds.load_manifest(d, v2)
    assert {c.crop_id: (c.tif_sha256, c.csv_sha256) for c in m2.crops if c.split == "test"} == test_before
    assert all(c.frozen_from == version for c in m2.crops if c.split == "test")
    assert {c.split for c in m2.crops if not c.frozen_from} <= {"train", "val"}

    total = time.perf_counter() - t_start
    print("WORKFLOW TIMINGS (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()) + f"; total {total:.1f}")
    assert total < BUDGET_S, timings
