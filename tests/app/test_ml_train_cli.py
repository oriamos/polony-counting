"""Training plan / refusals / registry candidate (stubbed fit), CLI smoke tests, and the opt-in end-to-end run."""
import json
import os
import time

import numpy as np
import pytest
import tifffile

from polony_app.ml import dataset as ds
from polony_app.ml.cli import main
from polony_app.ml.common import MLError, models_dir
from polony_app.ml.registry import Registry, resolve_model
from polony_app.ml.train import TrainOptions, plan_training, train_candidate
from test_ml_support import Recorder, add_analysis, new_store

T0 = "2026-01-01T00:00:00Z"
TV = {"train": 0.6, "val": 0.4, "test": 0.0}


def snapshot(tmp_path, n=5, fractions=None, **kw):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, n + 1)]
    r = ds.build_snapshot(store, tmp_path / "data", ids, ds.BuildOptions(created_at=T0, fractions=fractions or TV, **kw),
                          say=Recorder())
    return store, ids, r


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, d: {"kind": "folder", "name": "stubbase", "path": "/nonexistent", "registry_id": None})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.load_model", lambda *a, **k: object())
    monkeypatch.setattr("polony_app.ml.spotiflow_io.hardware_info", lambda device="cpu": {"platform": "test", "cuda_available": False})


def test_dry_run_validates_without_training_or_writing(tmp_path, stub_models):
    _, _, r = snapshot(tmp_path)
    say = Recorder()
    out = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=3, allow_small=True), dry_run=True, say=say)
    assert out["dry_run"] and out["n_train"] > 0 and out["n_val"] > 0 and out["train_config"]["num_epochs"] == 3
    assert "nothing was trained" in say.text()
    assert not (tmp_path / "data" / "models").exists() or list((tmp_path / "data" / "models").iterdir()) == []
    assert Registry(tmp_path / "data").list() == []


def test_refuses_without_valid_train_or_val_crops(tmp_path, stub_models):
    _, _, r = snapshot(tmp_path, n=1, fractions={"train": 1.0, "val": 0.0, "test": 0.0})
    with pytest.raises(MLError, match="no valid VALIDATION crops"):
        plan_training(tmp_path / "data", r.version, TrainOptions())
    store = new_store(tmp_path / "w")
    a = add_analysis(store, "x", slide_id="S1_A", approve=())
    with pytest.raises(MLError):
        ds.build_snapshot(store, tmp_path / "w" / "data", [a], ds.BuildOptions(created_at=T0), say=Recorder())


def test_refuses_bad_hyperparameters_and_damaged_snapshot(tmp_path, stub_models):
    _, _, r = snapshot(tmp_path)
    with pytest.raises(MLError, match="invalid hyper-parameters"):
        plan_training(tmp_path / "data", r.version, TrainOptions(epochs=0))
    p = r.path / "points.csv"
    os.chmod(p, 0o644)
    p.write_text("x")
    with pytest.raises(MLError, match="integrity"):
        plan_training(tmp_path / "data", r.version, TrainOptions())


def test_unknown_snapshot_message(tmp_path):
    with pytest.raises(MLError, match="not found"):
        plan_training(tmp_path / "data", "ds-nope", TrainOptions())


def test_train_records_a_candidate_with_full_provenance(tmp_path, stub_models):
    _, _, r = snapshot(tmp_path)
    seen = {}

    def fit(model, train, val, save_dir, cfg, device, logger):
        seen.update(n_train=len(train[0]), n_val=len(val[0]), cfg=cfg, device=device)
        assert train[1][0].ndim == 2 and train[1][0].shape[1] == 2
        assert train[0][0].dtype == np.float32 and train[0][0].max() < 4 and train[0][0].min() > -1   # page-normalised, not raw uint16
        save_dir.mkdir(parents=True)
        (save_dir / "best.pt").write_bytes(b"weights-v1")
        (save_dir / "config.yaml").write_text("sigma: 1.0\n")

    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, seed=7, allow_small=True), say=Recorder(), fit_fn=fit)
    assert e["status"] == "candidate" and e["kind"] == "finetuned" and e["dataset_version"] == r.version
    assert e["seed"] == 7 and e["hyperparameters"]["num_epochs"] == 2 and e["hyperparameters"]["device"] == "cpu"
    assert e["base"]["name"] == "stubbase" and e["duration_s"] >= 0 and e["hardware"]["platform"] == "test"
    assert len(e["weights_sha256"]) == 64 and e["split_membership"]["train"] and e["split_membership"]["val"]
    assert set(e["split_membership"]["train"]).isdisjoint(e["split_membership"]["val"])
    assert seen["n_train"] == len(e["split_membership"]["train"]) and seen["cfg"]["finetuned_from"] == "stubbase"
    reg = Registry(tmp_path / "data")
    assert reg.get(e["id"])["status"] == "candidate" and reg.active_id() is None      # never auto-promoted
    assert resolve_model(e["id"], tmp_path / "data").path == models_dir(tmp_path / "data") / e["id"]
    assert not [p for p in models_dir(tmp_path / "data").iterdir() if p.name.startswith(".staging")]


def test_failed_training_leaves_nothing_behind(tmp_path, stub_models):
    _, _, r = snapshot(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", allow_small=True), say=Recorder(), fit_fn=boom)
    assert Registry(tmp_path / "data").list() == []
    assert not [p for p in models_dir(tmp_path / "data").iterdir()]


# ---------------------------------------------------------------- CLI
def run(capsys, *argv):
    code = main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_build_list_show_verify_split(tmp_path, capsys):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, 7)]
    d = str(tmp_path / "data")
    code, out, err = run(capsys, "--data-dir", d, "list", "datasets")
    assert code == 0 and "no dataset snapshots" in out
    code, out, err = run(capsys, "--data-dir", d, "list", "models")
    assert code == 0 and "registry is empty" in out
    code, out, err = run(capsys, "--data-dir", d, "dataset", "build", "--analyses", *ids, "--fractions", "0.5,0.25,0.25",
                         "--created-at", T0)
    assert code == 0, err
    assert "created snapshot ds-" in out and "dataset version: ds-" in out and "WARNING" in out
    version = [l for l in out.splitlines() if l.startswith("dataset version:")][0].split()[-1]
    code, out, err = run(capsys, "--data-dir", d, "build", "--analyses", *ids, "--fractions", "0.5,0.25,0.25", "--created-at", T0)
    assert code == 2 and "already exists" in err                                  # immutability through the CLI
    code, out, _ = run(capsys, "--data-dir", d, "list", "datasets")
    assert version in out
    assert run(capsys, "--data-dir", d, "verify", version)[1].strip() == "intact"
    assert json.loads(run(capsys, "--data-dir", d, "show", version)[1])["version"] == version
    code, out, err = run(capsys, "--data-dir", d, "split", "--snapshot", version, "--lock-test")
    assert code == 0 and "LOCKED test set" in out and (tmp_path / "data" / "datasets" / "test_lock.json").is_file()
    code, out, err = run(capsys, "--data-dir", d, "split", "--snapshot", version, "--lock-test")
    assert code == 2 and "already exists" in err


def test_cli_refusals_exit_2_with_a_message(tmp_path, capsys):
    d = str(tmp_path / "data")
    code, out, err = run(capsys, "--data-dir", d, "promote", "ft-x")
    assert code == 2 and err.startswith("REFUSED:")
    code, out, err = run(capsys, "--data-dir", d, "rollback")
    assert code == 2 and "nothing to roll back" in err
    code, out, err = run(capsys, "--data-dir", d, "eval-model", "--model", "general")
    assert code == 2 and "no locked test set" in err
    code, out, err = run(capsys, "--data-dir", d, "train", "--snapshot", "ds-none", "--dry-run")
    assert code == 2 and "not found" in err
    code, out, err = run(capsys, "--data-dir", d, "build", "--analyses", "nonexistent")
    assert code == 2 and "not found" in err


def test_cli_train_dry_run(tmp_path, capsys, stub_models):
    _, _, r = snapshot(tmp_path)
    code, out, err = run(capsys, "--data-dir", str(tmp_path / "data"), "train", "--snapshot", r.version, "--base", "x",
                         "--epochs", "2", "--dry-run", "--allow-small")
    assert code == 0, err
    assert "[dry-run]" in out and "nothing was trained" in out and "ok: snapshot" in out


def test_cli_evaluate_writes_markdown_and_json(tmp_path, capsys):
    store = new_store(tmp_path)
    aid = add_analysis(store, "s", slide_id="S1_A")
    out_md = tmp_path / "rep" / "report.md"
    code, out, err = run(capsys, "--data-dir", str(tmp_path / "data"), "evaluate", "--predictions", aid, "--reference", "analysis",
                         "--out", str(out_md))
    assert code == 0, err
    md = out_md.read_text()
    assert "# Evaluation report" in md and "preliminary" in md
    rep = json.loads(out_md.with_suffix(".json").read_text())
    assert rep["reference"] == "analysis" and rep["tolerance_px"] == 10.0 and rep["tolerances_px"] == [5.0, 10.0, 15.0] and "union_before_review" in rep["configs"]


# ---------------------------------------------------------------- opt-in end-to-end (real Spotiflow, CPU)
@pytest.mark.skipif(os.environ.get("POLONY_ML_E2E") != "1", reason="set POLONY_ML_E2E=1 to run a real 2-epoch CPU fine-tuning")
def test_e2e_tiny_finetune_on_cpu(tmp_path, capsys):
    from polony_app.ml import evaluate as ev
    yy, xx = np.mgrid[0:120, 0:160]

    def pages(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(2):
            img = rng.normal(500, 30, (120, 160))
            for (x, y) in [(20, 20), (40, 30), (60, 50), (100, 20), (120, 40), (50, 40)]:
                img += 8000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
            out.append(np.clip(img, 0, 65535).astype(np.uint16))
        return tuple(out)

    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i, pages=pages(i)) for i in range(1, 7)]
    d = tmp_path / "data"
    r = ds.build_snapshot(store, d, ids, ds.BuildOptions(created_at=T0, fractions={"train": 0.5, "val": 0.25, "test": 0.25}),
                          say=Recorder())
    ds.lock_test_set(d, r.version)
    # a tiny randomly initialised Spotiflow stands in for the pretrained base (no download in the test)
    from spotiflow.model import Spotiflow, SpotiflowModelConfig
    base = tmp_path / "tiny_base"
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(base, which="best")
    t0 = time.time()
    e = train_candidate(d, r.version, TrainOptions(base=str(base), epochs=2, batch_size=2, crop_size=64, seed=0, device="cpu",
                                                   allow_small=True),
                        say=print)
    dt = time.time() - t0
    assert e["status"] == "candidate" and len(e["weights_sha256"]) == 64
    assert e["metrics"]["training"]["best_val_loss"] is not None and e["hardware"]["cuda_available"] in (True, False)
    ref = resolve_model(e["id"], d)
    assert (ref.path / "best.pt").is_file() and (ref.path / "config.yaml").is_file() and (ref.path / "thresholds.yaml").is_file()
    pred = ev.make_spotiflow_predictor(e["id"], d, device="cpu")
    res = ev.evaluate_on_snapshot(d, r.version, pred, split="test")
    assert res["n_crops"] > 0 and "f1" in res["metrics"]
    # CLI: evaluate the candidate on the locked test set, then promotion is refused (the default model was not evaluated)
    code, out, err = run(capsys, "--data-dir", str(d), "eval-model", "--model", e["id"])
    assert code == 0, err
    assert Registry(d).get(e["id"])["evaluations"][0]["metrics"]["n_fields"] == res["metrics"]["n_fields"]
    code, out, err = run(capsys, "--data-dir", str(d), "promote", e["id"])
    assert code == 2 and "nothing to compare against" in err
    assert Registry(d).active_id() is None
    print(f"E2E fine-tune OK in {dt:.1f}s; test metrics {res['metrics']['precision']}, {res['metrics']['recall']}")
