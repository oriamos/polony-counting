"""Device resolution (CUDA / ROCm / MPS / CPU), hardware record, ``gpu-smoke``, the scale printed by ``train --dry-run``,
lazy loading, validation subsampling and the examination-provenance gate of ``train``."""
import copy
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from legacy_bridge_fixtures import RED_GREEN, T0 as LT0, standard_corpus, write_channel_map, write_identity_map
from polony_app.ml import dataset as ds
from polony_app.ml import estimates as est
from polony_app.ml import legacy_bridge as lb
from polony_app.ml import spotiflow_io as sio
from polony_app.ml import train as tr
from polony_app.ml.cli import main
from polony_app.ml.common import MLError, models_dir
from polony_app.ml.registry import Registry
from polony_app.ml.train import TrainOptions, plan_training, train_candidate
from test_ml_support import Recorder, add_analysis, new_store

T0 = "2026-01-01T00:00:00Z"
TV = {"train": 0.6, "val": 0.4, "test": 0.0}


# ----------------------------------------------------------------------------------------------- fake torch objects
def fake_torch(*, hip=None, cuda=None, available=False, mps=False, name="AMD Radeon 8060S", arch="gfx1151", mem=64 * 1024 ** 3,
               n=1):
    props = SimpleNamespace(name=name, total_memory=mem, gcnArchName=arch)
    if arch is None:
        del props.gcnArchName
    return SimpleNamespace(
        __version__="2.9.0+fake", version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: n if available else 0,
                             get_device_name=lambda i=0: name, get_device_properties=lambda i=0: props),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


CPU_ONLY = fake_torch()
HIP_NO_DEVICE = fake_torch(hip="6.4.1")
CUDA_NO_DEVICE = fake_torch(cuda="12.4")
HIP_OK = fake_torch(hip="6.4.1", available=True)
CUDA_OK = fake_torch(cuda="12.4", available=True, name="NVIDIA RTX", arch=None)


def test_pick_device_cpu_and_unknown():
    assert sio.pick_device("cpu", torch_mod=CPU_ONLY) == "cpu"
    with pytest.raises(MLError, match="unknown --device"):
        sio.pick_device("tpu")


@pytest.mark.parametrize("req", ["cuda", "rocm"])
def test_explicit_gpu_on_a_cpu_only_torch_is_an_error_not_a_fallback(req):
    with pytest.raises(MLError) as e:
        sio.pick_device(req, torch_mod=CPU_ONLY)
    msg = str(e.value)
    assert "CPU-only build" in msg and "no silent fallback" in msg and "Nothing was started" in msg


def test_hip_build_without_a_visible_device_is_distinguished_and_only_lists_things_to_check():
    with pytest.raises(MLError) as e:
        sio.pick_device("rocm", torch_mod=HIP_NO_DEVICE)
    msg = str(e.value)
    assert "ROCm/HIP build" in msg and "torch.version.hip=6.4.1" in msg and "alias of 'cuda'" in msg
    assert "HSA_OVERRIDE_GFX_VERSION" in msg and "nothing here claims it works" in msg
    assert "CPU-only" not in msg and "CUDA build" not in msg
    # the linux access facts are CHECKS, never a claimed fix
    diag = sio.gpu_diagnosis(HIP_NO_DEVICE)
    assert diag["kind"] == "hip_no_device" and all(isinstance(h, str) for h in diag["hints"])
    assert not any("fixes" in h or "will fix" in h for h in diag["hints"])


def test_kfd_notes_are_reported_when_missing(monkeypatch):
    monkeypatch.setattr(sio.os.path, "exists", lambda p: False if p == "/dev/kfd" else os.path.exists(p))
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    msg = str(pytest.raises(MLError, sio.pick_device, "cuda", HIP_NO_DEVICE).value)
    assert "/dev/kfd does not exist" in msg and "HIP_VISIBLE_DEVICES='3'" in msg


def test_cuda_build_without_a_device():
    with pytest.raises(MLError) as e:
        sio.pick_device("cuda", torch_mod=CUDA_NO_DEVICE)
    assert "CUDA build" in str(e.value) and "nvidia-smi" in str(e.value) and "torch.version.hip" not in str(e.value)


@pytest.mark.parametrize("torch_mod", [HIP_OK, CUDA_OK])
def test_available_gpu_resolves_to_cuda_for_cuda_rocm_and_auto(torch_mod):
    assert [sio.pick_device(r, torch_mod=torch_mod) for r in ("cuda", "rocm", "auto")] == ["cuda"] * 3


def test_auto_falls_back_to_cpu_with_a_printed_notice_and_prefers_mps_over_cpu():
    say = Recorder()
    assert sio.pick_device("auto", torch_mod=HIP_NO_DEVICE, say=say) == "cpu"
    assert "NOTICE" in say.text() and "ROCm/HIP build" in say.text()
    say = Recorder()
    assert sio.pick_device("auto", torch_mod=CPU_ONLY, say=say) == "cpu" and "CPU-only build" in say.text()
    assert sio.pick_device("auto", torch_mod=fake_torch(mps=True)) == "mps"
    assert sio.pick_device("mps", torch_mod=fake_torch(mps=True)) == "mps"
    with pytest.raises(MLError, match="mps"):
        sio.pick_device("mps", torch_mod=CPU_ONLY)


# ------------------------------------------------------------------------------------------------- hardware_info
def test_hardware_info_records_hip_gpu_facts_and_requested_vs_resolved():
    h = sio.hardware_info("cuda", requested="rocm", torch_mod=HIP_OK)
    assert h["torch"] == "2.9.0+fake" and h["torch_hip"] == "6.4.1" and h["torch_cuda"] is None and h["cuda_available"] is True
    assert h["device_names"] == ["AMD Radeon 8060S"] and h["selected_device_name"] == "AMD Radeon 8060S"
    assert h["device_total_memory_bytes"] == 64 * 1024 ** 3 and h["gcn_arch"] == "gfx1151"
    assert h["requested_device"] == "rocm" and h["resolved_device"] == "cuda"
    assert h["cuda_device"] == "AMD Radeon 8060S"                                  # pre-existing key kept
    for k in ("platform", "python", "cpu_count", "spotiflow", "lightning", "mps_available"):
        assert k in h
    assert h["spotiflow"] and h["lightning"]                                       # installed in the test environment


def test_hardware_info_cuda_build_has_no_gcn_arch_and_no_gpu_means_no_gpu_fields():
    h = sio.hardware_info("cuda:0", torch_mod=CUDA_OK)
    assert h["torch_cuda"] == "12.4" and h["torch_hip"] is None and "gcn_arch" not in h and h["device_total_memory_bytes"]
    h = sio.hardware_info("cpu", requested="auto", torch_mod=HIP_NO_DEVICE)
    assert h["cuda_available"] is False and h["torch_hip"] == "6.4.1"
    assert "device_names" not in h and "device_total_memory_bytes" not in h
    assert (h["requested_device"], h["resolved_device"]) == ("auto", "cpu")


def test_hardware_info_survives_a_broken_gpu_query_and_a_missing_torch(monkeypatch):
    bad = fake_torch(hip="6", available=True)
    bad.cuda.get_device_properties = lambda i=0: (_ for _ in ()).throw(RuntimeError("hipErrorNoDevice"))
    assert "hipErrorNoDevice" in sio.hardware_info("cuda", torch_mod=bad)["gpu_query_error"]
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: None)
    h = sio.hardware_info("cpu")
    assert h["torch"] is None and h["cuda_available"] is False


# ---------------------------------------------------------------------------------------------------- gpu-smoke
real_gpu = pytest.mark.skipif(sio._torch_module() is not None and sio._gpu_available(sio._torch_module()),
                              reason="this machine has a GPU: the 'no GPU' paths are tested on GPU-less machines")


def test_gpu_smoke_auto_on_cpu_reports_cpu_only_and_exits_0(capsys):
    code = main(["gpu-smoke", "--size", "64", "--steps", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "CPU only" in out and "matmul+backward" in out and "conv2d forward+backward" in out
    assert "no GPU was used" in out and "requested_device: auto" in out and "torch_hip" in out


@real_gpu
@pytest.mark.parametrize("dev", ["cuda", "rocm"])
def test_gpu_smoke_explicit_gpu_that_is_missing_exits_2_with_the_reason(capsys, dev):
    code = main(["gpu-smoke", "--device", dev, "--size", "64"])
    out = capsys.readouterr().out
    assert code == 2 and "FAILED" in out and "no silent fallback" in out and f"requested_device: {dev}" in out


@real_gpu
def test_gpu_smoke_reports_the_failing_stage_when_the_gpu_claims_to_exist_but_fails(monkeypatch, capsys):
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: HIP_OK if m is None else m)
    code = main(["gpu-smoke", "--device", "rocm", "--size", "64"])
    out = capsys.readouterr().out
    assert code == 2 and "[FAIL] matmul+backward" in out and "gpu-smoke: FAILED - stage 'matmul+backward' failed on cuda" in out


def test_gpu_smoke_function_never_raises_and_checks_against_the_cpu():
    r = pytest.importorskip("polony_app.ml.cli_devices").smoke("cpu", size=48, steps=2)
    assert r["ok"] and [s["name"] for s in r["stages"]] == ["matmul+backward", "conv2d forward+backward"]


# --------------------------------------------------------------------------------------------- snapshot helpers
def snapshot(tmp_path, n=5, **kw):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, n + 1)]
    return ds.build_snapshot(store, tmp_path / "data", ids, ds.BuildOptions(created_at=T0, fractions=TV, **kw), say=Recorder())


def legacy_snapshot(tmp_path, downscale=1):
    root = tmp_path / "in"
    root.mkdir()
    files = standard_corpus(root, ("AM", "AS", "FS", "PS", "QS"))
    cm = write_channel_map(tmp_path / "c.json", defaults=RED_GREEN)
    im = write_identity_map(tmp_path / "id.json", files)
    out = lb.import_legacy(tmp_path / "data", root, lb.BridgeOptions(created_at=LT0, fractions={"train": 0.6, "val": 0.4, "test": 0.0},
                                                                     downscale=downscale),
                           channel_map=cm, identity_map=im, say=Recorder())
    return out.result


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, d: {"kind": "folder", "name": "stubbase", "path": "/nonexistent", "registry_id": None})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.load_model", lambda *a, **k: object())
    monkeypatch.setattr("polony_app.ml.spotiflow_io.hardware_info", lambda device="cpu": {"platform": "test", "cuda_available": False})


def fit_stub(model, train, val, save_dir, cfg, device, logger):
    save_dir.mkdir(parents=True)
    (save_dir / "best.pt").write_bytes(os.urandom(16))          # distinct weights per call (identical ones are refused)
    (save_dir / "config.yaml").write_text("sigma: 1.0\n")


# ------------------------------------------------------------------------------------------- dry run prints the scale
def test_dry_run_prints_the_actual_scale_and_recommendations(tmp_path, stub_models):
    r = snapshot(tmp_path)
    m = ds.load_manifest(tmp_path / "data", r.version)
    n_tr, n_va = (sum(1 for c in m.crops if c.split == s) for s in ("train", "val"))
    say = Recorder()
    out = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=10, batch_size=4, allow_small=True,
                                                                      early_stopping_patience=3), dry_run=True, say=say)
    t = say.text()
    e = out["scale"]["estimate"]
    assert e["n_train_crops"] == n_tr and e["n_val_crops"] == n_va and e["steps_per_epoch"] == -(-n_tr // 4)
    assert e["optimizer_steps"] == e["steps_per_epoch"] * 10 and e["validation_batches_total"] == n_va * 10
    for needle in ("SCALE OF THIS RUN", f"sources (source images): {len(m.sources)} in the manifest",
                   f"eligible crops: {len(m.crops)}", f"{n_tr} train crop(s), {n_va} val crop(s)",
                   f"steps per epoch: {e['steps_per_epoch']}   estimated optimizer steps: {e['optimizer_steps']}",
                   f"{n_va} batch(es) per pass (Spotiflow validates with batch size 1) x 10 pass(es)",
                   "prepared in-memory size", "available RAM", "load mode: eager", "native snapshot size on disk",
                   "Training will not start because --dry-run was supplied.", "nothing was trained", "RECOMMENDATIONS",
                   "--epochs 1", "--epochs 10 --early-stopping-patience 3", "--num-train-samples N bounds an epoch",
                   "eval-model for the candidate and for 'general'"):
        assert needle in t, needle
    assert out["scale"]["native_snapshot_bytes"] > 0 and tr.format_scale  # real bytes on disk
    assert not (tmp_path / "data" / "models").exists() or not list((tmp_path / "data" / "models").iterdir())


def test_dry_run_num_train_samples_bounds_the_epoch(tmp_path, stub_models):
    r = snapshot(tmp_path)
    say = Recorder()
    out = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=3, batch_size=4, num_train_samples=10,
                                                                      allow_small=True), dry_run=True, say=say)
    e = out["scale"]["estimate"]
    assert e["samples_per_epoch"] == 10 and e["steps_per_epoch"] == 3 and e["optimizer_steps"] == 9
    assert "num_train_samples: 10  -> 10 sample(s) drawn per epoch" in say.text()


def test_dry_run_fails_early_for_an_unavailable_explicit_gpu(tmp_path, stub_models, monkeypatch):
    r = snapshot(tmp_path)
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: HIP_NO_DEVICE)
    with pytest.raises(MLError, match="ROCm/HIP build"):
        train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", device="rocm", allow_small=True), dry_run=True,
                        say=Recorder())


def test_cli_train_dry_run_new_flags(tmp_path, capsys, stub_models):
    r = snapshot(tmp_path)
    code = main(["--data-dir", str(tmp_path / "data"), "train", "--snapshot", r.version, "--base", "x", "--epochs", "2", "--dry-run",
                 "--allow-small", "--device", "cpu", "--num-workers", "2", "--load-mode", "lazy", "--max-val-crops", "1",
                 "--num-train-samples", "8"])
    out = capsys.readouterr().out
    assert code == 0 and "load mode: lazy (--load-mode lazy)" in out and "workers: 2" in out and "--max-val-crops 1" in out


# -------------------------------------------------------------------- device handling inside train (stubbed fit)
def test_train_with_rocm_alias_runs_on_cuda_and_records_requested_and_resolved(tmp_path, stub_models, monkeypatch):
    r = snapshot(tmp_path)
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: HIP_OK)
    seen = {}

    def fit(model, train, val, save_dir, cfg, device, logger):
        seen["device"] = device
        fit_stub(model, train, val, save_dir, cfg, device, logger)

    say = Recorder()
    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, device="rocm", allow_small=True), say=say,
                        fit_fn=fit)
    assert seen["device"] == "cuda"
    assert e["hyperparameters"]["device"] == "cuda" and e["hyperparameters"]["device_requested"] == "rocm"
    assert e["hardware"]["requested_device"] == "rocm" and e["hardware"]["resolved_device"] == "cuda"
    assert "alias of 'cuda'" in say.text()


def test_train_explicit_gpu_missing_trains_nothing_and_registers_nothing(tmp_path, stub_models, monkeypatch):
    r = snapshot(tmp_path)
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: CPU_ONLY)
    called = []
    with pytest.raises(MLError, match="CPU-only build"):
        train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", device="cuda", allow_small=True), say=Recorder(),
                        fit_fn=lambda *a: called.append(1))
    assert not called and Registry(tmp_path / "data").list() == []


def test_train_auto_without_a_gpu_falls_back_to_cpu_with_a_notice(tmp_path, stub_models, monkeypatch):
    r = snapshot(tmp_path)
    monkeypatch.setattr(sio, "_torch_module", lambda m=None: HIP_NO_DEVICE)
    say = Recorder()
    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, device="auto", allow_small=True), say=say,
                        fit_fn=fit_stub)
    assert e["hyperparameters"]["device"] == "cpu" and e["hyperparameters"]["device_requested"] == "auto"
    assert "NOTICE: --device auto uses the CPU" in say.text()


def test_train_prints_the_next_commands_and_registers_the_best_checkpoint(tmp_path, stub_models):
    r = snapshot(tmp_path)
    say = Recorder()
    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, allow_small=True), say=say, fit_fn=fit_stub)
    t = say.text()
    assert e["weights_file"] == "best.pt" and "BEST validation checkpoint" in t
    assert f"eval-model --model {e['id']}" in t and "eval-model --model general" in t
    assert f"compare --models {e['id']} general" in t and f"promote {e['id']}" in t and "only if the comparison supports it" in t
    assert Registry(tmp_path / "data").active_id() is None


def test_num_workers_and_deferred_normalisation_are_forwarded_to_fit(tmp_path, monkeypatch):
    r = snapshot(tmp_path)
    got = {}

    class FakeModel:
        def fit(self, tri, trs, vai, vas, **kw):
            got.update(kw, n=len(tri), types=type(tri).__name__)
            sd = kw["save_dir"]
            sd.mkdir(parents=True)
            (sd / "best.pt").write_bytes(os.urandom(16))

    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, d: {"kind": "folder", "name": "stubbase", "path": "/x", "registry_id": None})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.load_model", lambda *a, **k: FakeModel())
    for mode, defer in (("eager", False), ("lazy", True)):
        got.clear()
        train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, num_workers=3, load_mode=mode, allow_small=True,
                                                                   seed=hash(mode) % 1000), say=Recorder())
        assert got["num_workers"] == 3 and got["deterministic"] is True and got["normalizer"] is None
        assert ("defer_normalization" in got) is defer and (got["types"] == "LazyCropImages") is defer
        assert "num_workers" not in got["train_config"] and "max_val_crops" not in got["train_config"]


# --------------------------------------------------------------------------- memory guard and load-mode selection
def test_choose_load_mode_thresholds():
    G = 1024 ** 3
    assert est.choose_load_mode(1 * G, 64 * G)[0] == "eager"
    assert est.choose_load_mode(9 * G, 512 * G)[0] == "lazy"                  # > 8 GiB
    assert est.choose_load_mode(5 * G, 16 * G)[0] == "lazy"                   # > 25 % of the RAM
    assert est.choose_load_mode(1 * G, None)[0] == "eager"
    assert est.choose_load_mode(9 * G, 512 * G, "eager")[0] == "eager" and est.choose_load_mode(1, 10 ** 12, "lazy")[0] == "lazy"
    assert est.format_bytes(1536 * 1024 ** 2) == "1.5 GiB" and est.format_bytes(None) == "unknown"
    assert (est.available_memory_bytes() or 1) > 0


def test_eager_that_exceeds_ram_is_refused_early_unless_allowed_and_auto_goes_lazy(tmp_path, stub_models, monkeypatch):
    r = snapshot(tmp_path)
    monkeypatch.setattr(est, "available_memory_bytes", lambda: 1000)             # absurdly small
    base = dict(base="x", epochs=2, allow_small=True)
    with pytest.raises(MLError, match="exceeds the available RAM.*--load-mode lazy.*--allow-large-memory"):
        train_candidate(tmp_path / "data", r.version, TrainOptions(load_mode="eager", **base), say=Recorder(), fit_fn=fit_stub)
    assert Registry(tmp_path / "data").list() == []
    e = train_candidate(tmp_path / "data", r.version, TrainOptions(load_mode="eager", allow_large_memory=True, **base),
                        say=Recorder(), fit_fn=fit_stub)
    assert e["hyperparameters"]["load_mode"] == "eager" and e["hyperparameters"]["allow_large_memory"] is True
    e2 = train_candidate(tmp_path / "data", r.version, TrainOptions(load_mode="auto", seed=5, **base), say=Recorder(), fit_fn=fit_stub)
    assert e2["hyperparameters"]["load_mode"] == "lazy" and e2["hyperparameters"]["load_mode_requested"] == "auto"


# ------------------------------------------------------------------------------------ lazy == eager, bit for bit
@pytest.mark.parametrize("downscale", [1, 2])
@pytest.mark.parametrize("kind", ["reviewed", "legacy"])
def test_lazy_inputs_and_labels_equal_the_eager_path_bit_for_bit(tmp_path, kind, downscale):
    r = snapshot(tmp_path, downscale=downscale) if kind == "reviewed" else legacy_snapshot(tmp_path, downscale)
    m = ds.load_manifest(tmp_path / "data", r.version)
    crops = [c for c in m.crops if c.split in ("train", "val")]
    assert len(crops) >= 4
    cfg = {"downscale": downscale, "normaliser": "valid_unsaturated"}
    ei, es, ed = tr._load_xy(tmp_path / "data", r.version, crops, cfg)
    li, ls, ld = tr.load_lazy(tmp_path / "data", r.version, crops, cfg, cache_mb=1)
    assert ed == ld and len(li) == len(ei)
    for i in range(len(crops)):
        assert li[i].dtype == ei[i].dtype == np.float32 and np.array_equal(li[i], ei[i]) and li[i].shape == ei[i].shape
        assert np.array_equal(ls[i], es[i]) and ls[i].dtype == es[i].dtype
        assert tr.prepared_hw(crops[i], downscale) == ei[i].shape
    assert [x.shape for x in li] == [x.shape for x in ei]


def test_lazy_sequence_cache_is_bounded_iteration_is_free_and_it_pickles(tmp_path):
    r = snapshot(tmp_path)
    m = ds.load_manifest(tmp_path / "data", r.version)
    crops = [c for c in m.crops if c.split == "train"]
    cfg = {"downscale": 1, "normaliser": "valid_unsaturated"}
    one = int(np.prod(tr.prepared_hw(crops[0], 1))) * 4
    seq, _, _ = tr.load_lazy(tmp_path / "data", r.version, crops, cfg, cache_mb=1)
    seq._cache_bytes = 2 * one + 1                                            # room for two crops
    assert [x.shape for x in seq] and seq.loads == 0                          # iterating (shape scans) reads nothing
    assert all(x.strides == (0, 0) for x in seq)
    a = seq[0]
    assert seq.loads == 1 and seq[0] is a and seq.loads == 1 and seq.hits == 1  # repeated access is a cache hit
    assert not a.flags.writeable
    for i in range(len(seq)):
        seq[i]
    assert seq._cache_used <= seq._cache_bytes and len(seq._cache) <= 2
    assert seq[-1].shape == seq[len(seq) - 1].shape and len(seq[0:2]) == 2
    with pytest.raises(IndexError):
        seq[len(seq)]
    seq[1] = np.zeros((70, 70), np.float32)                                    # Spotiflow pads small crops by assignment
    assert seq[1].shape == (70, 70) and list(seq)[1].shape == (70, 70)
    clone = pickle.loads(pickle.dumps(seq))
    assert clone.loads == 0 and len(clone._cache) == 0 and np.array_equal(clone[0], seq[0]) and clone[1].shape == (70, 70)


def test_lazy_detects_a_snapshot_that_does_not_match_its_manifest(tmp_path):
    r = snapshot(tmp_path)
    m = ds.load_manifest(tmp_path / "data", r.version)
    crops = [c for c in m.crops if c.split == "train"][:2]
    seq, _, _ = tr.load_lazy(tmp_path / "data", r.version, crops, {"downscale": 1, "normaliser": "valid_unsaturated"})
    seq._shapes[0] = (3, 3)
    with pytest.raises(MLError, match="manifest says"):
        seq[0]


# ---------------------------------------------------------------- real tiny Spotiflow fit: lazy works end to end
def tiny_base(tmp_path):
    from spotiflow.model import Spotiflow, SpotiflowModelConfig
    base = tmp_path / "tiny_base"
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(base, which="best")
    return base


def real_fit(d, version, base, **kw):
    return train_candidate(d, version, TrainOptions(base=str(base), epochs=2, batch_size=2, crop_size=32, seed=0, device="cpu",
                                                    allow_small=True, **kw), say=Recorder())


def test_real_tiny_fit_lazy_equals_eager_and_workers_run(tmp_path):
    import shutil
    r = legacy_snapshot(tmp_path)
    base = tiny_base(tmp_path)
    runs = {}
    for name, kw in (("eager", {"load_mode": "eager"}), ("lazy", {"load_mode": "lazy"}),
                     ("workers", {"load_mode": "lazy", "num_workers": 2})):
        d = tmp_path / f"data_{name}"
        shutil.copytree(tmp_path / "data", d)                                    # a fresh registry per run
        runs[name] = real_fit(d, r.version, base, **kw)
    assert runs["lazy"]["hyperparameters"]["load_mode"] == "lazy" and runs["eager"]["hyperparameters"]["load_mode"] == "eager"
    # same data in the same order + same seed -> the same training, eager or lazy
    assert runs["eager"]["weights_sha256"] == runs["lazy"]["weights_sha256"]
    assert runs["eager"]["metrics"]["training"]["val_loss_curve"] == runs["lazy"]["metrics"]["training"]["val_loss_curve"]
    w = runs["workers"]                                                           # forked worker processes: must simply work
    assert w["hyperparameters"]["num_workers"] == 2 and w["metrics"]["training"]["best_val_loss"] is not None


def test_spotiflow_validation_only_scores_the_first_item_of_a_batch_so_val_batch_size_is_not_changed():
    """Why there is no --val-batch-size: Spotiflow 0.6.5's validation_step keeps ``img[0]`` / ``pts[0]`` of every batch for the
    epoch-end F1 and the logged loss is weighted by the TRAIN batch size, so a validation batch > 1 would silently change
    val_f1 and drop samples from it. If this test fails after a Spotiflow upgrade, re-evaluate the option."""
    import inspect
    from spotiflow.model.trainer import SpotiflowTrainingWrapper
    src = inspect.getsource(SpotiflowTrainingWrapper.validation_step) + inspect.getsource(SpotiflowTrainingWrapper.generate_dataloaders)
    assert 'batch["pts"][0]' in src and "img[0]" in src and "batch_size=1" in src


# ------------------------------------------------------------------------------------------- --max-val-crops
def test_subsample_val_is_deterministic_stable_and_independent_of_input_order():
    crops = [SimpleNamespace(crop_id=f"c{i}", group_id="g") for i in range(20)]
    kept, excl = tr.subsample_val(crops, 5)
    kept2, excl2 = tr.subsample_val(list(reversed(crops)), 5)
    assert [c.crop_id for c in kept] == [c.crop_id for c in kept2] and excl == excl2
    assert len(kept) == 5 and len(excl) == 15 and set(excl).isdisjoint(c.crop_id for c in kept)
    import hashlib
    assert kept[0].crop_id == min((c.crop_id for c in crops), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    assert [c.crop_id for c in tr.subsample_val(crops, 7)[0][:5]] == [c.crop_id for c in kept]      # a prefix: stable


def test_max_val_crops_is_recorded_warned_and_default_uses_everything(tmp_path, stub_models):
    r = snapshot(tmp_path, n=8)
    full = plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))
    assert full.val_total == len(full.val) >= 3 and not full.val_excluded
    sub = plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True, max_val_crops=2))
    assert len(sub.val) == 2 and len(sub.val_excluded) == len(full.val) - 2
    assert any("--max-val-crops" in w and "2 of" in w for w in sub.warnings)
    assert plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True, max_val_crops=999)).val_excluded == []
    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=2, allow_small=True, max_val_crops=2),
                        say=Recorder(), fit_fn=fit_stub)
    sm = e["split_membership"]
    assert len(sm["val"]) == 2 and sm["val_excluded_by_max_val_crops"] == sub.val_excluded
    assert e["hyperparameters"]["max_val_crops"] == 2 and e["hyperparameters"]["n_val_total"] == len(full.val)
    assert any("--max-val-crops" in w for w in e["warnings"])
    with pytest.raises(MLError, match="invalid options"):
        plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True, max_val_crops=0))


# ------------------------------------------------------------------------------- examination provenance gate
def test_a_normal_legacy_snapshot_passes_the_gate(tmp_path, stub_models):
    r = legacy_snapshot(tmp_path)
    plan = plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))
    assert any("researcher_examined provenance" in c for c in plan.checks)
    assert lb.verify_examination_provenance(plan.manifest) == [] and plan.manifest.import_summary


def edited(monkeypatch, tmp_path, version, edit):
    m = ds.load_manifest(tmp_path / "data", version).model_copy(deep=True)
    edit(m)
    monkeypatch.setattr(tr, "load_manifest", lambda *a, **k: m)
    monkeypatch.setattr(tr, "verify_snapshot", lambda *a, **k: [])
    return m


def test_a_bridge_import_with_a_crop_that_is_not_researcher_examined_is_refused(tmp_path, stub_models, monkeypatch):
    r = legacy_snapshot(tmp_path)

    def edit(m):
        m.crops[0].provenance = dict(m.crops[0].provenance, researcher_examined=False)

    edited(monkeypatch, tmp_path, r.version, edit)
    with pytest.raises(MLError) as e:
        plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))
    assert "not proven researcher-examined" in str(e.value) and "researcher_examined" in str(e.value)


def test_a_bridge_import_whose_crop_lost_its_provenance_is_refused_in_every_split(tmp_path, stub_models, monkeypatch):
    r = legacy_snapshot(tmp_path)
    m = ds.load_manifest(tmp_path / "data", r.version)
    for split in ("train", "val"):
        idx = next(i for i, c in enumerate(m.crops) if c.split == split)

        def edit(mm, idx=idx):
            mm.crops[idx].provenance = {}

        edited(monkeypatch, tmp_path, r.version, edit)
        with pytest.raises(MLError, match="not proven researcher-examined"):
            plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))


def test_a_stripped_import_summary_does_not_hide_missing_examination(tmp_path, stub_models, monkeypatch):
    r = legacy_snapshot(tmp_path)

    def edit(m):
        m.import_summary = {}
        m.crops[0].provenance = {"researcher_examined": False}

    edited(monkeypatch, tmp_path, r.version, edit)
    with pytest.raises(MLError, match="not proven researcher-examined"):
        plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))


def test_snapshots_built_from_reviewed_analyses_are_not_subject_to_the_bridge_gate(tmp_path, stub_models):
    r = snapshot(tmp_path)
    plan = plan_training(tmp_path / "data", r.version, TrainOptions(allow_small=True))
    assert any("no bridge-import provenance" in c for c in plan.checks)
