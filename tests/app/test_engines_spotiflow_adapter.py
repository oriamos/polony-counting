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


@pytest.mark.parametrize("device,msg,expected", [
    ("mps", "NotImplementedError: The operator 'aten::foo' is not currently implemented for the MPS device.", True),
    ("mps", "MPSNDArray: failed assertion", True),
    ("mps", "Invalid buffer size: 20.00 GiB", True),
    ("cuda", "CUDA error: an illegal memory access", True),
    ("cuda", "CUDA out of memory. Tried to allocate 2 GiB", True),
    ("mps", "expected 2D input, got shape (3,)", False),
    ("mps", "temps and attempts", False),
    ("cpu", "MPS CUDA error out of memory", False),
    ("mps", "", False),
])
def test_is_device_failure(device, msg, expected):
    assert sa.is_device_failure(device, msg) is expected


def test_mps_runtime_failure_falls_back_to_cpu_once_per_process(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, mps=True)
    calls = []

    def f(a):
        calls.append(1)
        if len(calls) == 1:
            raise NotImplementedError("The operator 'aten::x' is not currently implemented for the MPS device.\nmore")
        return np.array([[2.0, 3.0]]), np.array([0.7])

    fs.points_fn = f
    eng = fs.engine()
    res = call(eng, np.zeros((20, 20), np.uint16))
    assert res.run.device == "cpu" and len(res.detections) == 1
    fb = res.run.preprocessing["device_fallback"]
    assert fb["from"] == "mps" and fb["to"] == "cpu" and fb["error"].endswith("MPS device.") and "\n" not in fb["error"]
    assert "MPS" in sa.device_note() and "MPS" in eng.describe()["device_note"]
    assert eng.describe()["device_selected"] == "cpu"
    # the next slide goes straight to cpu: no second failed attempt, still recorded
    res2 = call(eng, np.zeros((20, 20), np.uint16))
    assert len(calls) == 3 and res2.run.device == "cpu" and res2.run.preprocessing["device_fallback"]["from"] == "mps"


def test_unrelated_inference_error_is_not_retried(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, mps=True)

    def boom(a):
        raise ValueError("bad input shape")

    fs.points_fn = boom
    with pytest.raises(EngineError, match="bad input shape"):
        call(fs.engine(), np.zeros((20, 20), np.uint16))
    assert sa.device_note() is None


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
    # the adapter downscales (frame-unambiguous): model input = block mean (area) of the float32 trimmed crop
    # (25, 50) blocks, reflect-padded by us on the right/bottom to a multiple of the model stride (16): (32, 64)
    assert c["shape"] == (32, 64) and c["dtype"] == np.float32
    bm = img[:100, :200].astype(np.float32).reshape(25, 4, 50, 4).mean((1, 3))
    assert np.allclose(c["values"][:25, :50], bm) and np.allclose(c["values"][25:, :50], bm[-2:-9:-1][:7])
    pre = res.run.preprocessing
    assert pre["downscaled_shape_hw"] == [25, 50]
    assert pre["downscale"] == 4 and pre["trimmed_px"] == {"right": 2, "bottom": 3}
    assert pre["crop_box_xyxy"] == [0, 0, 200, 100] and any("block grid" in n for n in res.run.notes)
    got = {(round(d.x, 6), round(d.y, 6)) for d in res.detections}
    # block-mean: block i covers pixels i*d..i*d+d-1, centre i*d + (d-1)/2
    assert (1.5, 1.5) in got and (49 * 4 + 1.5, 24 * 4 + 1.5) in got and (81.5, 41.5) in got


def test_downscale_with_crop_offset(monkeypatch, tmp_path):
    img = np.zeros((300, 400), np.uint16)
    valid = np.zeros((300, 400), bool)
    valid[100:150, 200:260] = True
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.array([[0.0, 0.0]]), np.array([0.9])))
    res = call(fs.engine(), img, valid, params={"downscale": 2})
    # padded box 168..292 x 68..182 extended outward to multiples of 16*d = 32: 160..320 x 64..192;
    # point (0,0) = block centre (d-1)/2 = 0.5 from the origin
    assert res.run.preprocessing["crop_box_xyxy"] == [160, 64, 320, 192]
    assert (res.detections[0].x, res.detections[0].y) == (160.5, 64.5)


def test_downscale_crop_origin_is_anchored_to_the_block_grid(monkeypatch, tmp_path):
    """Whatever the counting area, the crop origin is a multiple of 16*d (extended outward inside the image) so the
    block grid does not depend on where the area starts."""
    img = np.zeros((300, 400), np.uint16)
    for d in (2, 4, 8):
        for ox in (0, 1, 2, 3):
            valid = np.zeros(img.shape, bool)
            valid[103 + ox:151 + ox, 203 + ox:259 + ox] = True
            fs = FakeSpotiflow(monkeypatch, tmp_path / f"{d}{ox}", points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
            res = call(fs.engine(), img, valid, params={"downscale": d})
            x0, y0, x1, y1 = res.run.preprocessing["crop_box_xyxy"]
            assert x0 % (16 * d) == 0 and y0 % (16 * d) == 0 and (x1 - x0) % d == 0 and (y1 - y0) % d == 0
            assert x0 <= 203 + ox - 32 and y0 <= 103 + ox - 32 and x1 >= 259 + ox + 32 - d and y1 >= 151 + ox + 32 - d


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


@pytest.mark.parametrize("d", [2, 4, 8])
def test_block_mean_centre_mapping_is_unbiased_for_subpixel_gaussians(d):
    """Model-free: the centroid of the block-mean image mapped with i*d + (d-1)/2 recovers the true centre."""
    n = 400
    yy, xx = np.mgrid[:n, :n]
    worst = 0.0
    for cx, cy in ((100.0, 200.0), (100.5, 150.25), (201.3, 99.75), (57.9, 301.1)):
        img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2)).astype(np.float32)
        z = img.reshape(n // d, d, n // d, d).mean((1, 3))
        idx = np.arange(n // d)
        gx = (z.sum(0) * idx).sum() / z.sum() * d + (d - 1) / 2
        gy = (z.sum(1) * idx).sum() / z.sum() * d + (d - 1) / 2
        worst = max(worst, abs(gx - cx), abs(gy - cy))
    assert worst < 0.02, worst


def test_block_mean_end_to_end_with_a_fake_peak_finder(monkeypatch, tmp_path):
    """Full path: a bright blob at a known sub-pixel position, the fake model returns the argmax of the downscaled
    input; the mapped position is within d/2 px of the truth and (for a blob centred on a block) unbiased."""
    n = 256
    yy, xx = np.mgrid[:n, :n]
    for d in (2, 4, 8):
        cx, cy = 96.0 + (d - 1) / 2, 136.0 + (d - 1) / 2     # exactly a block centre
        img = (1000 + 30000 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2))).astype(np.uint16)

        def peak(a):
            r, c = np.unravel_index(np.argmax(a), a.shape)
            return np.array([[float(r), float(c)]]), np.array([0.9])
        fs = FakeSpotiflow(monkeypatch, tmp_path / f"e{d}", points_fn=peak)
        res = call(fs.engine(), img, params={"downscale": d})
        assert abs(res.detections[0].x - cx) <= 1e-6 + 0 and abs(res.detections[0].y - cy) <= 1e-6


# ------------------------------------------------------------------ normaliser 'valid_unsaturated'
def _norm_image():
    rng = np.random.default_rng(0)
    img = rng.integers(300, 3000, size=(200, 240)).astype(np.uint16)
    img[:, :30] = 65535                      # saturated rim OUTSIDE the counting area
    img[100:104, 100:110] = 65535            # saturated debris INSIDE the counting area
    valid = np.zeros(img.shape, bool)
    valid[10:190, 60:230] = True
    return img, valid


def test_valid_unsaturated_normaliser_excludes_saturated_and_outside_pixels(monkeypatch, tmp_path):
    img, valid = _norm_image()
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
    res = call(fs.engine(), img, valid, params={"normalizer": "valid_unsaturated", "prob_thresh": 0.2})
    norm = fs.calls[0]["kw"]["normalizer"]
    assert callable(norm)
    sel = valid & (img < 65535)
    lo, hi = np.percentile(img[sel].astype(np.float64), [1, 99.8])
    pre = res.run.preprocessing
    st = pre["normalizer_stats"]
    assert pre["normalizer_mode"] == "valid_unsaturated"
    assert abs(st["lo"] - lo) < 1e-9 and abs(st["hi"] - hi) < 1e-9 and hi < 5000     # not 65535
    assert st["n_used"] == int(sel.sum()) and st["n_valid_saturated"] == 40 and st["n_valid"] == int(valid.sum())
    assert res.run.params["normalizer"] == "valid_unsaturated"
    out = norm(np.array([lo, hi], np.float32))
    assert np.allclose(out, [0.0, 1.0], atol=1e-4) and out.dtype == np.float32


def test_valid_unsaturated_range_is_independent_of_crop_and_applied_identically_when_downscaled(monkeypatch, tmp_path):
    img, valid = _norm_image()
    ranges = []
    for kwp in ({"crop_to_valid": True}, {"crop_to_valid": False}, {"downscale": 4}, {"downscale": 2}):
        fs = FakeSpotiflow(monkeypatch, tmp_path / str(len(ranges)), points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
        res = call(fs.engine(), img, valid, params={"normalizer": "valid_unsaturated", **kwp})
        st = res.run.preprocessing["normalizer_stats"]
        ranges.append((st["lo"], st["hi"]))
    assert len(set(ranges)) == 1


def test_valid_unsaturated_fallback_and_float_images(monkeypatch, tmp_path):
    img = np.full((100, 100), 65535, np.uint16)
    img[:20, :20] = np.arange(400, dtype=np.uint16).reshape(20, 20) + 100
    valid = np.ones(img.shape, bool)
    fs = FakeSpotiflow(monkeypatch, tmp_path / "a", points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
    res = call(fs.engine(), img, valid, params={"normalizer": "valid_unsaturated"})
    assert any("fallback" in n for n in res.run.notes)
    fl = np.random.default_rng(1).random((80, 80)).astype(np.float32)
    fs = FakeSpotiflow(monkeypatch, tmp_path / "b", points_fn=lambda a: (np.zeros((0, 2)), np.zeros(0)))
    res = call(fs.engine(), fl, None, params={"normalizer": "valid_unsaturated"})
    assert res.run.preprocessing["normalizer_stats"]["saturation_level"] is None


def test_unknown_normalizer_is_rejected(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    with pytest.raises(EngineError, match="normalizer"):
        call(fs.engine(), np.zeros((64, 64), np.uint16), params={"normalizer": "magic"})


def test_auto_normaliser_is_documented_as_legacy_saturation_sensitive(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    res = call(fs.engine(), np.zeros((64, 64), np.uint16), params={})
    assert "legacy" in res.run.preprocessing["normalizer"] and fs.calls[0]["kw"]["normalizer"] == "auto"


def test_quadratic_peak_refinement_recovers_subpixel_offsets():
    from polony_app.engines.spotiflow_adapter import _refine_peaks
    yy, xx = np.mgrid[:20, :20]
    hm = np.exp(-((xx - 9.3) ** 2 + (yy - 11.4) ** 2) / (2 * 2.0 ** 2))
    pts, n = _refine_peaks(np.array([[11.0, 9.0]]), hm)
    assert n == 1 and abs(pts[0, 0] - 11.4) < 0.05 and abs(pts[0, 1] - 9.3) < 0.05
    same, n0 = _refine_peaks(np.array([[0.0, 5.0]]), hm)             # border: untouched
    assert n0 == 0 and same[0, 0] == 0.0
