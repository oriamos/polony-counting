"""Spotiflow adapter with a fake spotiflow module: normalisation of output, 16-bit preservation, crop mapping,
validity flags, checksum, failures. No model download, no GPU."""
import json
import sys

import numpy as np
import pytest

from polony_app.detection.errors import DetectionCancelled
from polony_app.engines import EngineError, EngineUnavailable, SpotiflowEngine
from polony_app.engines import spotiflow_adapter as sa
from polony_app.schemas import EngineName

try:
    from tests.app.engines_fakes import FakeSpotiflow
except ImportError:  # pragma: no cover
    from engines_fakes import FakeSpotiflow

NOP = lambda f, m: None
NEVER = lambda: False


def call(eng, img, valid=None, params=None, page=0, role="red", **kw):
    return eng.detect_page(img, analysis_id="a1", page=page, channel_role=role, valid_mask=valid,
                           params=params or {}, image_sha256={"f": "abc"}, progress=NOP, cancelled=NEVER, **kw)


def test_import_without_torch_does_not_import_torch():
    import subprocess
    code = ("import sys, polony_app.engines; assert 'torch' not in sys.modules and 'spotiflow' not in sys.modules")
    import os
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_available_explains_missing_package(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_have", lambda n: n != "spotiflow")
    ok, why = SpotiflowEngine(tmp_path).available()
    assert not ok and "spotiflow" in why and "pip install" in why
    monkeypatch.setattr(sa, "_have", lambda n: n != "torch")
    ok, why = SpotiflowEngine(tmp_path).available()
    assert not ok and "PyTorch" in why


def test_available_reports_model_cache_state(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    ok, why = fs.engine().available()
    assert ok and "cached" in why and "downloaded" not in why
    ok, why = SpotiflowEngine(tmp_path / "empty").available()
    assert ok and "not cached yet" in why


def test_output_normalisation_and_16bit_preserved(monkeypatch, tmp_path):
    img = np.full((120, 150), 1000, np.uint16)
    img[60, 70] = 65535          # extreme values must reach the model untouched
    img[10:12, 5:7] = 4321
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[60.0, 70.0], [10.25, 5.75]]),
                                                                  np.array([0.9, 0.6])))
    res = call(fs.engine(), img)
    (c,) = fs.calls
    assert c["dtype"] == np.uint16 and np.array_equal(c["values"], img)       # original dtype AND values
    assert c["kw"]["normalizer"] == "auto" and c["kw"]["prob_thresh"] is None   # model's own threshold
    d = res.detections
    assert [x.id for x in d] == ["sf-p0-000001", "sf-p0-000002"]               # (y, x) order, deterministic
    assert (d[0].x, d[0].y) == (5.75, 10.25) and (d[1].x, d[1].y) == (70.0, 60.0)  # (y,x) -> x,y
    assert [x.score for x in d] == [0.6, 0.9] and {x.score_kind for x in d} == {"spotiflow_prob_raw"}
    assert all(x.engine == EngineName.spotiflow and x.run_id == res.run.run_id and x.valid for x in d)
    r = res.run
    assert r.status == "ok" and r.n_raw == 2 and r.n_valid == 2 and r.page == 0 and r.channel_role == "red"
    assert r.model_name == "general" and len(r.model_checksum) == 64 and r.engine_version == "9.9.9-fake"
    assert r.params["prob_thresh_used"] == 0.5 and r.params["runtime"]["torch"] == "0.0-fake"
    assert r.preprocessing["eight_bit_conversion"] is False and r.preprocessing["input_dtype"] == "uint16"
    assert r.device == "cpu" and r.image_sha256 == {"f": "abc"}
    json.dumps(r.model_dump(mode="json"))


def test_normalizer_none_is_passed(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
    res = call(fs.engine(), np.zeros((50, 50), np.uint8), params={"normalizer": "none", "prob_thresh": 0.3})
    assert fs.calls[0]["kw"]["normalizer"] is None and fs.calls[0]["kw"]["prob_thresh"] == 0.3
    assert res.run.params["prob_thresh_used"] == 0.3 and res.detections == [] and res.run.status == "ok"


def test_crop_to_valid_bbox_maps_back_exactly(monkeypatch, tmp_path):
    h, w = 300, 400
    img = np.full((h, w), 500, np.uint16)
    valid = np.zeros((h, w), bool)
    valid[100:150, 200:260] = True           # bbox rows 100..149, cols 200..259
    # the model sees the padded crop: rows 68..181, cols 168..291  -> point at crop (y=32, x=32) = full (100, 200)
    seen = {}

    def pts(a):
        seen["shape"] = a.shape
        return np.array([[32.0, 32.0], [32.0 + 49, 32.0 + 59]]), np.array([0.8, 0.7])

    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=pts)
    res = call(fs.engine(), img, valid)
    assert seen["shape"] == (150 - 100 + 64, 260 - 200 + 64)
    assert res.run.preprocessing["crop_box_xyxy"] == [168, 68, 292, 182]
    d = res.detections
    assert (d[0].x, d[0].y) == (200.0, 100.0) and d[0].valid          # first valid pixel: no off-by-one
    assert (d[1].x, d[1].y) == (259.0, 149.0) and d[1].valid           # last valid pixel
    assert fs.calls[0]["values"].shape == seen["shape"]


def test_crop_clipped_at_image_border_and_disabled(monkeypatch, tmp_path):
    img = np.zeros((100, 100), np.uint16)
    valid = np.zeros((100, 100), bool)
    valid[0:10, 0:10] = True
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[3.0, 4.0]]), np.array([0.7])))
    res = call(fs.engine(), img, valid)
    assert res.run.preprocessing["crop_box_xyxy"] == [0, 0, 42, 42]
    assert (res.detections[0].x, res.detections[0].y) == (4.0, 3.0)
    res = call(fs.engine(), img, valid, params={"crop_to_valid": False})
    assert res.run.preprocessing["crop_box_xyxy"] == [0, 0, 100, 100]


def test_invalid_detections_kept_and_flagged(monkeypatch, tmp_path):
    img = np.zeros((100, 100), np.uint16)
    valid = np.zeros((100, 100), bool)
    valid[20:60, 20:60] = True
    pts = np.array([[30.0, 30.0], [10.0, 10.0], [59.4, 59.4], [59.6, 30.0]])
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (pts - 0, np.full(4, 0.9)))
    res = call(fs.engine(), img, valid, params={"crop_to_valid": False})
    by = {(d.x, d.y): d for d in res.detections}
    assert by[(30.0, 30.0)].valid and by[(30.0, 30.0)].excluded_reason is None
    assert not by[(10.0, 10.0)].valid and by[(10.0, 10.0)].excluded_reason
    assert by[(59.4, 59.4)].valid and not by[(30.0, 59.6)].valid        # nearest-pixel fallback rule
    assert res.run.n_raw == 4 and res.run.n_valid == 2


def test_custom_point_validity_is_used(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[5.0, 5.0]]), np.array([0.9])))

    def never(xs, ys):
        return np.zeros(len(xs), bool), [None] * len(xs), ["custom"] * len(xs)

    res = call(fs.engine(), np.zeros((20, 20), np.uint16), point_validity=never)
    assert not res.detections[0].valid and res.detections[0].excluded_reason == "custom"


def test_empty_valid_mask_is_skipped_not_silent(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    res = call(fs.engine(), np.zeros((20, 20), np.uint16), np.zeros((20, 20), bool))
    assert res.run.status == "skipped" and res.detections == [] and res.run.notes and not fs.calls


def test_model_load_failure_is_engine_error(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, fail_load=RuntimeError("corrupt checkpoint"))
    with pytest.raises(EngineError, match="corrupt checkpoint"):
        call(fs.engine(), np.zeros((20, 20), np.uint16))


def test_inference_failure_is_engine_error(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, fail_predict=ValueError("bad tile"))
    with pytest.raises(EngineError, match="inference failed.*bad tile"):
        call(fs.engine(), np.zeros((20, 20), np.uint16))


def test_mismatched_scores_refused(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([0.5])))
    with pytest.raises(EngineError, match="per-point scores"):
        call(fs.engine(), np.zeros((20, 20), np.uint16))


def test_unavailable_package_raises_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "_have", lambda n: False)
    with pytest.raises(EngineUnavailable):
        call(SpotiflowEngine(tmp_path), np.zeros((20, 20), np.uint16))


def test_bad_inputs(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    eng = fs.engine()
    with pytest.raises(EngineError):
        call(eng, np.zeros((3, 20, 20), np.uint16))
    with pytest.raises(EngineError):
        call(eng, np.full((20, 20), np.nan, np.float32))
    with pytest.raises(EngineError):
        call(eng, np.zeros((20, 20), np.uint16), np.zeros((5, 5), bool))


def test_cancellation_between_tiles(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    fs.n_tiles_fake = 6
    n = {"i": 0}

    def cancelled():
        n["i"] += 1
        return n["i"] > 4

    with pytest.raises(DetectionCancelled):
        fs.engine().detect_page(np.zeros((40, 40), np.uint16), analysis_id="a", page=0, channel_role=None,
                                valid_mask=None, params={}, image_sha256={}, progress=NOP, cancelled=cancelled)


def test_device_selection_and_fallback(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, cuda=False)
    res = call(fs.engine(), np.zeros((20, 20), np.uint16), params={"device": "cuda"})
    assert res.run.device == "cpu" and any("not available" in n for n in res.run.notes)
    assert fs.loads[-1]["device"] == "cpu"
    fs2 = FakeSpotiflow(monkeypatch, tmp_path / "x", cuda=True)
    res = call(fs2.engine(), np.zeros((20, 20), np.uint16))        # auto -> cuda
    assert res.run.device == "cuda"
    # cuda inference error -> retried on cpu and recorded
    fs3 = FakeSpotiflow(monkeypatch, tmp_path / "y", cuda=True)
    orig = fs3.points_fn
    state = {"n": 0}

    def flaky(a):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return np.zeros((0, 2)), np.zeros(0)

    fs3.points_fn = flaky
    res = call(fs3.engine(), np.zeros((20, 20), np.uint16))
    assert res.run.device == "cpu" and any("retried on cpu" in n for n in res.run.notes)


def test_page_and_role_recorded_per_page(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[2.0, 3.0]]), np.array([0.6])))
    eng = fs.engine()
    a = call(eng, np.zeros((20, 20), np.uint16), page=0, role=None)
    b = call(eng, np.zeros((20, 20), np.uint16), page=1, role="green")
    assert a.run.page == 0 and a.run.channel_role is None and a.detections[0].id == "sf-p0-000001"
    assert b.run.page == 1 and b.run.channel_role == "green" and b.detections[0].id == "sf-p1-000001"
    assert a.run.run_id != b.run.run_id and a.run.preprocessing["page_to_channel"].startswith("unconfirmed")


# ------------------------------------------------------------------ model cache + checksums
def test_ensure_model_checksum_stable_cached_sidecar(monkeypatch, tmp_path):
    import hashlib
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    prog = []
    m1 = sa.ensure_model("general", fs.cache_dir, lambda f, m: prog.append((f, m)))
    assert m1.cached and m1.checksum_sha256 == hashlib.sha256(b"fake-weights-best").hexdigest()
    assert set(m1.files) == {"best.pt", "last.pt"} and prog[-1][0] == 1.0
    side = fs.cache_dir / "general" / sa._SIDECAR
    assert side.is_file()
    # second call: served from the sidecar (hashing is not repeated)
    monkeypatch.setattr(sa, "_sha256_file", lambda p: (_ for _ in ()).throw(AssertionError("rehashed")))
    m2 = sa.ensure_model("general", fs.cache_dir)
    assert m2.checksum_sha256 == m1.checksum_sha256


def test_ensure_model_never_downloads_when_cached(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("download attempted")

    monkeypatch.setattr(sys.modules["spotiflow.model.pretrained"], "get_pretrained_model_path", boom)
    assert sa.ensure_model("general", fs.cache_dir).cached


def test_ensure_model_downloads_once_and_reports(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    cache = tmp_path / "fresh"
    n = {"dl": 0}

    def fake_dl(name, cache_dir=None):
        n["dl"] += 1
        d = cache_dir / name
        d.mkdir(parents=True)
        (d / "config.yaml").write_text("x")
        (d / "best.pt").write_bytes(b"w")
        return d

    monkeypatch.setattr(sys.modules["spotiflow.model.pretrained"], "get_pretrained_model_path", fake_dl)
    prog = []
    m = sa.ensure_model("general", cache, lambda f, s: prog.append(s))
    assert not m.cached and n["dl"] == 1 and any("downloading" in s for s in prog)
    assert sa.ensure_model("general", cache).cached and n["dl"] == 1


def test_ensure_model_unknown_and_download_failure(monkeypatch, tmp_path):
    FakeSpotiflow(monkeypatch, tmp_path)
    with pytest.raises(EngineUnavailable, match="unknown Spotiflow model"):
        sa.ensure_model("nope", tmp_path / "c")

    def bad(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(sys.modules["spotiflow.model.pretrained"], "get_pretrained_model_path", bad)
    with pytest.raises(EngineUnavailable, match="network down"):
        sa.ensure_model("general", tmp_path / "c2")


def test_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("POLONY_MODEL_DIR", str(tmp_path / "m"))
    assert sa.model_cache_dir() == tmp_path / "m"
    monkeypatch.delenv("POLONY_MODEL_DIR")
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "home"))
    assert sa.model_cache_dir() == tmp_path / "home" / "models" / "spotiflow"


def test_describe_and_list_models(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    d = fs.engine().describe()
    assert d["available"] and d["device"]["cpu"] and {m["name"]: m["cached"] for m in d["models"]} == {
        "general": True, "hybiss": False}


# ------------------------------------------------------------------ downscale (large blobs)
def test_downscale_1_passes_no_scale_kwargs(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[5.0, 7.0]]), np.array([0.9])))
    res = call(fs.engine(), np.zeros((40, 41), np.uint16), params={"downscale": 1})
    kw = fs.calls[0]["kw"]
    assert "scale" not in kw and kw["subpix"] is None and fs.calls[0]["dtype"] == np.uint16
    assert (res.detections[0].x, res.detections[0].y) == (7.0, 5.0)
    assert res.run.preprocessing["downscale"] == 1 and res.run.preprocessing["trimmed_px"] == {"right": 0, "bottom": 0}


def test_downscale_input_trim_and_coordinate_mapping(monkeypatch, tmp_path):
    img = np.arange(103 * 202, dtype=np.uint16).reshape(103, 202)      # 202 % 4 = 2, 103 % 4 = 3
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[0.0, 0.0], [24.0, 49.0], [10.0, 20.0]]),
                                                                  np.array([0.9, 0.8, 0.7])))
    res = call(fs.engine(), img, params={"downscale": 4, "prob_thresh": 0.1})
    c = fs.calls[0]
    from scipy.ndimage import zoom
    assert "scale" not in c["kw"] and c["kw"]["subpix"] is False and c["kw"]["prob_thresh"] == 0.1
    # the adapter downscales (frame-unambiguous): model input = zoom(float32 trimmed crop, 1/4, order=1)
    assert c["shape"] == (25, 50) and c["dtype"] == np.float32
    assert np.array_equal(c["values"], zoom(img[:100, :200].astype(np.float32), 0.25, order=1))
    pre = res.run.preprocessing
    assert pre["downscaled_shape_hw"] == [25, 50]
    assert pre["downscale"] == 4 and pre["trimmed_px"] == {"right": 2, "bottom": 3}
    assert pre["crop_box_xyxy"] == [0, 0, 200, 100] and any("trimmed" in n for n in res.run.notes)
    got = {(round(d.x, 6), round(d.y, 6)) for d in res.detections}
    # align-corners: first/last downscaled pixel centres coincide with first/last full-res pixel centres
    assert (0.0, 0.0) in got and (199.0, 99.0) in got
    assert any(abs(x - 20 * 199 / 49) < 1e-5 and abs(y - 10 * 99 / 24) < 1e-5 for x, y in got)


def test_downscale_with_crop_offset(monkeypatch, tmp_path):
    img = np.zeros((300, 400), np.uint16)
    valid = np.zeros((300, 400), bool)
    valid[100:150, 200:260] = True
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[0.0, 0.0]]), np.array([0.9])))
    res = call(fs.engine(), img, valid, params={"downscale": 2})
    # crop 168..292 x 68..182 (124 x 114), both even -> nothing trimmed; point (0,0) = crop origin
    assert res.run.preprocessing["crop_box_xyxy"] == [168, 68, 292, 182]
    assert (res.detections[0].x, res.detections[0].y) == (168.0, 68.0)


@pytest.mark.parametrize("bad", [0, 3, 16, -2])
def test_downscale_unsupported_is_explicit(monkeypatch, tmp_path, bad):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    with pytest.raises(EngineError, match="unsupported Spotiflow downscale"):
        call(fs.engine(), np.zeros((64, 64), np.uint16), params={"downscale": bad})
    assert not fs.calls


def test_downscale_rejects_subpix_and_tiny_images(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    with pytest.raises(EngineError, match="sub-pixel"):
        call(fs.engine(), np.zeros((64, 64), np.uint16), params={"downscale": 2, "subpix": True})
    with pytest.raises(EngineError, match="too small"):
        call(fs.engine(), np.zeros((10, 64), np.uint16), params={"downscale": 8})
    assert not fs.calls


def test_downscale_mapping_matches_scipy_zoom_convention():
    """The mapping used by the adapter inverts scipy.ndimage.zoom(order=1) (what Spotiflow 0.6.5 uses)."""
    from scipy.ndimage import zoom
    n = 400
    yy, xx = np.mgrid[:n, :n]
    for d in (2, 4, 8):
        m = n // d
        for cx in (100.0, 100.5, 201.3):
            img = np.exp(-((xx - cx) ** 2 + (yy - 200.0) ** 2) / (2 * 6.0 ** 2)).astype(np.float32)
            z = zoom(img, (1 / d, 1 / d), order=1)
            cs = (z.sum(0) * np.arange(m)).sum() / z.sum(0).sum()
            assert abs(cs * (n - 1) / (m - 1) - cx) < 0.02
