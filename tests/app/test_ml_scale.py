"""Scale and normalisation of the ML pipeline: downscale mapping, page-level lo/hi, training at the model's configuration,
the small-data safeguard, best/last epoch reporting and eval-model / promotion at the model's own configuration."""
import json
import os

import numpy as np
import pytest
import tifffile

from polony_app.ml import dataset as ds
from polony_app.ml.cli import main
from polony_app.ml import common as C
from polony_app.ml.common import MLError, models_dir, sha256_file
from polony_app.ml.registry import Registry, resolve_model
from polony_app.ml.train import TrainOptions, plan_training, train_candidate
from test_ml_support import Recorder, add_analysis, new_store, tiff_pages

T0 = "2026-01-01T00:00:00Z"
TV = {"train": 0.6, "val": 0.4, "test": 0.0}
TVT = {"train": 0.5, "val": 0.25, "test": 0.25}


# ---------------------------------------------------------------- pure helpers, hand-computed
def test_block_mean_and_pixel_centre_label_mapping_hand_computed():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = C.block_mean_downscale(img, 2)
    assert out.dtype == np.float32 and out.tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert C.block_mean_downscale(img, 1) is not None and C.block_mean_downscale(img, 1).tolist() == img.tolist()
    # remainder rows / columns are trimmed at the bottom / right, never padded
    assert C.block_mean_downscale(np.ones((7, 9), np.float32), 4).shape == (1, 2)
    # x_s = (x - (d-1)/2) / d  (y, x order preserved)
    assert C.to_downscaled([[1.5, 1.5]], 4).tolist() == [[0.0, 0.0]]          # centre of the first 4x4 block
    assert C.to_downscaled([[5.5, 9.5]], 4).tolist() == [[1.0, 2.0]]
    assert C.to_downscaled([[0.5, 2.5]], 2).tolist() == [[0.0, 1.0]]
    assert C.to_downscaled([[3.0, 7.0]], 1).tolist() == [[3.0, 7.0]]
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 100, (20, 2))
    for d in (1, 2, 4, 8):
        assert np.allclose(C.from_downscaled(C.to_downscaled(p, d), d), p)
    # a symmetric spot centred at native (y, x) = (13.5, 9.5) is centred at the mapped position of the block mean
    yy, xx = np.mgrid[0:32, 0:32]
    blob = np.exp(-((yy - 13.5) ** 2 + (xx - 9.5) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
    sm = C.block_mean_downscale(blob, 4)
    gy, gx = np.mgrid[0:8, 0:8]
    cy, cx = (sm * gy).sum() / sm.sum(), (sm * gx).sum() / sm.sum()
    assert np.allclose(C.to_downscaled([[13.5, 9.5]], 4)[0], [cy, cx], atol=1e-4)
    with pytest.raises(MLError):
        C.check_downscale(3)
    with pytest.raises(MLError):
        C.block_mean_downscale(img, 16)


def test_block_mean_equals_the_adapters_block_mean_and_page_bounds_equal_its_normaliser():
    adapter = pytest.importorskip("polony_app.engines.spotiflow_adapter")
    rng = np.random.default_rng(1)
    page = rng.integers(0, 65536, (96, 128)).astype(np.uint16)
    page[:10] = 65535                                                        # saturated rim
    valid = np.zeros(page.shape, bool)
    valid[5:90, 8:120] = True
    for d in (2, 4, 8):
        h, w = page.shape[0] // d * d, page.shape[1] // d * d
        ref = page[:h, :w].reshape(h // d, d, w // d, d).astype(np.float32).mean(axis=(1, 3), dtype=np.float32)
        assert np.array_equal(C.block_mean_downscale(page.astype(np.float32), d), ref)     # same expression as the adapter's
    fn = getattr(adapter, "_valid_unsaturated_range", None)
    if fn is None:
        pytest.skip("adapter has no normaliser helper")
    lo, hi, _ = fn(page, valid)
    b = C.page_bounds(page, valid)
    assert (b["lo"], b["hi"]) == (lo, hi)
    assert b["n_saturated"] > 0 and b["n_used"] == b["n_valid"] - b["n_saturated"]


def test_page_bounds_uses_valid_unsaturated_pixels_only():
    page = np.full((100, 100), 1000, np.uint16)
    page[:, :50] = np.arange(1, 5001, dtype=np.uint16).reshape(100, 50)   # 1..5000 in the left half
    page[:, 50:] = 65535                                                    # right half saturated
    valid = np.ones(page.shape, bool)
    valid[:20] = False                                                      # top rows are outside the counting area
    b = C.page_bounds(page, valid)
    vals = page[20:, :50].astype(float)
    assert b["lo"] == pytest.approx(np.percentile(vals, 1)) and b["hi"] == pytest.approx(np.percentile(vals, 99.8))
    assert b["hi"] < 5001 and b["n_saturated"] == 80 * 50 and b["n_used"] == 80 * 50 and b["fallback"] is None
    # a plain per-image percentile (what the app used to feed) would sit at the saturation level
    assert np.percentile(page, 99.8) == 65535
    # almost everything saturated: documented fallback, never a silent wrong number
    allsat = np.full((60, 60), 65535, np.uint16)
    allsat[0, :5] = [1, 2, 3, 4, 5]
    fb = C.page_bounds(allsat)
    assert fb["fallback"] and "whole page" in fb["fallback"] or "valid pixels" in fb["fallback"]
    # float pages have no saturation level
    assert C.page_bounds(np.random.default_rng(0).random((50, 50)))["n_saturated"] == 0


def test_normalise_image_is_linear_in_page_lo_hi_and_refuses_legacy_crops():
    x = np.array([[100, 200], [300, 400]], np.uint16)
    n = C.normalise_image(x, 100.0, 300.0)
    assert n.dtype == np.float32 and n.tolist() == [[0.0, 0.5], [1.0, 1.5]]                 # not clipped
    assert C.normalise_image(x, None, None, "none").tolist() == x.astype(np.float32).tolist()
    with pytest.raises(MLError, match="rebuild the snapshot"):
        C.normalise_image(x, None, None)


def test_prepare_crop_aligns_the_block_grid_to_the_page_and_maps_labels():
    img = np.arange(11 * 13, dtype=np.uint16).reshape(11, 13)
    # crop origin (x0, y0) = (5, 3): page-grid d=4 starts at page column 8 / row 4 -> trim 3 columns and 1 row
    x, lab, dropped, off = C.prepare_crop(img, [[1.0 + 1.5, 3.0 + 1.5]], lo=0.0, hi=1.0, downscale=4, origin_xy=(5, 3))
    assert off == (1, 3) and x.shape == (10 // 4, 10 // 4)
    assert lab.tolist() == [[0.0, 0.0]] and dropped == 0               # (y=2.5, x=4.5) native = centre of the first kept block
    # a label in the trimmed remainder falls outside and is counted
    _, lab2, dropped2, _ = C.prepare_crop(img, [[10.5, 12.5]], lo=0.0, hi=1.0, downscale=4, origin_xy=(5, 3))
    assert lab2.shape == (0, 2) and dropped2 == 1


# ---------------------------------------------------------------- dataset build: lo/hi per page, downscale in the manifest
def build(tmp_path, n=2, **kw):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, n + 1)]
    r = ds.build_snapshot(store, tmp_path / "data", ids, ds.BuildOptions(created_at=T0, fractions=kw.pop("fractions", TV), **kw),
                          say=Recorder())
    return store, ids, r


def test_manifest_has_one_lo_hi_per_source_page_shared_by_all_its_crops(tmp_path):
    from polony_app.detection.masks import build_valid_mask
    store = new_store(tmp_path)
    red, green = tiff_pages(3)
    red[:30, :60] = 65535                                         # a saturated patch inside field 1
    aid = add_analysis(store, "s1", slide_id="S1_A", pages=(red, green))
    r = ds.build_snapshot(store, tmp_path / "data", [aid], ds.BuildOptions(created_at=T0, fractions={"train": 1, "val": 0, "test": 0}),
                          say=Recorder())
    m = r.manifest
    src = m.sources[0]
    assert set(src.page_normalization) == {"red", "green"}
    with store.view(aid) as st:
        regions = st.analysis.regions
    valid = build_valid_mask(red.shape, regions)
    exp = C.page_bounds(red, valid)
    pn = src.page_normalization["red"]
    assert (pn["lo"], pn["hi"]) == (exp["lo"], exp["hi"]) and pn["n_saturated"] == int((valid & (red == 65535)).sum()) > 0
    assert pn["page"] == 0 and pn["policy"] == C.LO_HI_POLICY and pn["mode"] == "valid_unsaturated"
    # the page statistic differs from what each crop would give on its own (the old per-crop behaviour)
    red_crops = [c for c in m.crops if c.channel_role == "red"]
    assert len(red_crops) == 2 and {(c.norm_lo, c.norm_hi) for c in red_crops} == {(exp["lo"], exp["hi"])}
    for c in red_crops:
        img, _ = ds.load_crop(tmp_path / "data", r.version, c)
        assert np.percentile(img, 99.8) != pytest.approx(exp["hi"])
        assert img.dtype == np.uint16 and list(img.shape) == c.shape_hw          # exported at NATIVE resolution
    gc = [c for c in m.crops if c.channel_role == "green"]
    assert {(c.norm_lo, c.norm_hi) for c in gc} == {(src.page_normalization["green"]["lo"], src.page_normalization["green"]["hi"])}
    assert ds.verify_snapshot(tmp_path / "data", r.version) == []


def test_green_valid_mask_follows_the_alignment_shift(tmp_path):
    store = new_store(tmp_path)
    red, green = tiff_pages(4)
    green[:, :3] = 0                                              # columns that the shift moves out of the frame
    aid = add_analysis(store, "s1", slide_id="S1_A", pages=(red, green), alignment=(2, 3))
    r = ds.build_snapshot(store, tmp_path / "data", [aid], ds.BuildOptions(created_at=T0, fractions={"train": 1, "val": 0, "test": 0}),
                          say=Recorder())
    with store.view(aid) as st:
        a = st.analysis.model_copy(deep=True)
    mask = ds._stored_valid_mask(a, "green", green.shape)
    # stored pixel (r, c) is reference pixel (r + 2, c + 3): rows/cols that fall out of the frame are invalid
    assert not mask[-2:].any() and not mask[:, -3:].any()
    assert r.manifest.sources[0].page_normalization["green"]["valid_mask"].endswith("alignment shift")


def test_downscale_and_normaliser_are_part_of_the_snapshot(tmp_path):
    _, _, r1 = build(tmp_path / "a", downscale=1)
    _, _, r4 = build(tmp_path / "b", downscale=4)
    assert r4.manifest.options["downscale"] == 4 and r4.manifest.options["normaliser"] == "valid_unsaturated"
    assert r4.manifest.options["lo_hi_policy"] == C.LO_HI_POLICY and r4.manifest.options["percentiles"] == [1.0, 99.8]
    assert all(c.downscale == 4 and c.normaliser == "valid_unsaturated" for c in r4.manifest.crops)
    assert r1.version != r4.version                               # different training view = different snapshot
    # crops stay native, whatever the downscale
    assert [c.shape_hw for c in r4.manifest.crops] == [c.shape_hw for c in r1.manifest.crops]
    with pytest.raises(MLError, match="downscale must be one of"):
        build(tmp_path / "c", downscale=3)
    with pytest.raises(MLError, match="normaliser must be one of"):
        build(tmp_path / "d", normaliser="auto")


def test_unmarked_raw_proposals_are_counted_in_the_manifest(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "s1", slide_id="S1_A")

    class RawStore:
        def __init__(self, s):
            self._s = s

        def __getattr__(self, k):
            return getattr(self._s, k)

        def get_raw_detections(self, a):
            pts = [(20.0, 20.0), (60.0, 50.0), (50.0, 40.0), (21.0, 21.0), (59.0, 51.0), (30.0, 45.0), (10.0, 52.0), (70.0, 12.0), (40.0, 50.0)]
            return [{"id": f"d{i}", "x": x, "y": y, "channel_role": "red", "page": 0, "score": 0.9 - 0.01 * i, "valid": True,
                     "engine": "spotiflow", "region_id": None, "score_kind": "heatmap"} for i, (x, y) in enumerate(pts)]

    r = ds.build_snapshot(RawStore(store), tmp_path / "data", [aid],
                          ds.BuildOptions(created_at=T0, fractions={"train": 1, "val": 0, "test": 0}), say=Recorder())
    u = r.manifest.label_counts["unmarked_raw_proposals"]
    assert u["n_raw_proposals"] == 9 and u["n_unmarked"] == 4 and u["n_confirmed_proposals"] == 5   # (30,45) (10,52) (70,12) (40,50)
    assert u["n_unmarked_bright"] is not None and "lower bound" in u["note"]
    assert "THIS annotator's marks" in r.manifest.label_counts["precision_label"]
    # without raw detections the count is absent, not zero
    _, _, r0 = build(tmp_path / "x")
    assert r0.manifest.label_counts["unmarked_raw_proposals"] is None


# ---------------------------------------------------------------- training at the snapshot's configuration
@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, d: {"kind": "folder", "name": "stubbase", "path": "/nonexistent", "registry_id": None})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.load_model", lambda *a, **k: object())
    monkeypatch.setattr("polony_app.ml.spotiflow_io.hardware_info", lambda device="cpu": {"platform": "test", "cuda_available": False})


def write_weights(save_dir, curve=None, best=True, last=True):
    save_dir.mkdir(parents=True)
    if best:
        (save_dir / "best.pt").write_bytes(b"best")
    if last:
        (save_dir / "last.pt").write_bytes(b"last")
    (save_dir / "config.yaml").write_text("sigma: 1.0\n")
    if curve:
        log = save_dir.parent / "logs" / "fit" / "version_0"
        log.mkdir(parents=True)
        (log / "metrics.csv").write_text("epoch,train_loss,val_loss\n" + "".join(f"{e},0.5,{v}\n" for e, v in curve))


SMALL_OK = dict(allow_small=True)


def test_training_data_is_normalised_with_the_page_lo_hi_downscaled_and_labels_mapped(tmp_path, stub_models):
    _, _, r = build(tmp_path, n=4, downscale=4)
    seen = {}

    def fit(model, train, val, save_dir, cfg, device, logger):
        seen.update(train=train, val=val, cfg=cfg)
        write_weights(save_dir)

    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", epochs=1, **SMALL_OK), say=Recorder(), fit_fn=fit)
    plan = plan_training(tmp_path / "data", r.version, TrainOptions(base="x", **SMALL_OK))
    imgs, spots = seen["train"]
    assert len(imgs) == len(plan.train)
    for im, sp, c in zip(imgs, spots, plan.train):
        raw, lab = ds.load_crop(tmp_path / "data", r.version, c)
        want, want_lab, _, _ = C.prepare_crop(raw, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=4, origin_xy=c.box_xyxy[:2])
        assert im.dtype == np.float32 and np.array_equal(im, want) and np.allclose(sp, want_lab)
        assert im.shape[0] <= raw.shape[0] // 4 and im.shape[1] <= raw.shape[1] // 4
        # the SAME page lo/hi for every crop of the page: the per-page statistic, not the crop's own percentiles
        assert (c.norm_lo, c.norm_hi) == (r.manifest.sources[[s.analysis_id for s in r.manifest.sources].index(c.analysis_id)]
                                          .page_normalization[c.channel_role]["lo"],
                                          r.manifest.sources[[s.analysis_id for s in r.manifest.sources].index(c.analysis_id)]
                                          .page_normalization[c.channel_role]["hi"])
    assert seen["cfg"]["downscale"] == 4 and seen["cfg"]["normaliser"] == "valid_unsaturated"
    # the registry entry carries the configuration the model was trained at
    assert e["inference_config"] == {"downscale": 4, "normaliser": "valid_unsaturated"}
    assert e["lo_hi_policy"] == C.LO_HI_POLICY and e["page_normalization"]
    assert resolve_model(e["id"], tmp_path / "data").trained_downscale == 4


def test_small_training_sets_are_refused_unless_allowed(tmp_path, stub_models):
    _, _, r = build(tmp_path, n=4)                                         # 2-3 independent training groups
    with pytest.raises(MLError, match=r"too small.*independent training group.*--allow-small"):
        plan_training(tmp_path / "data", r.version, TrainOptions(base="x"))
    with pytest.raises(MLError, match="training crop"):
        plan_training(tmp_path / "data", r.version, TrainOptions(base="x", min_train_groups=1))
    plan = plan_training(tmp_path / "data", r.version, TrainOptions(base="x", allow_small=True))
    assert any("--allow-small: training anyway" in w for w in plan.warnings)
    plan = plan_training(tmp_path / "data", r.version, TrainOptions(base="x", min_train_groups=1, min_train_crops=1))
    assert not any("too small" in w for w in plan.warnings)


def test_best_epoch_is_registered_and_last_epoch_is_reported_when_validation_diverges(tmp_path, stub_models):
    _, _, r = build(tmp_path, n=4)
    say = Recorder()

    def fit(model, train, val, save_dir, cfg, device, logger):
        write_weights(save_dir, curve=[(0, 1.0), (1, 0.5), (2, 2.0)])

    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", **SMALL_OK), say=say, fit_fn=fit)
    t = e["metrics"]["training"]
    assert e["weights_file"] == "best.pt" and t["weights_epoch"] == "best"
    assert (t["best_val_loss"], t["best_epoch"], t["last_val_loss"], t["last_epoch"]) == (0.5, 1, 2.0, 2)
    assert t["diverged"] is True and any("DIVERGED" in w for w in e["warnings"]) and "DIVERGED" in say.text()
    assert e["hyperparameters"]["allow_small"] is True


def test_diverged_run_without_best_weights_is_refused_not_silently_last_pt(tmp_path, stub_models):
    _, _, r = build(tmp_path, n=4)

    def fit(model, train, val, save_dir, cfg, device, logger):
        write_weights(save_dir, curve=[(0, 1.0), (1, 0.5), (2, 2.0)], best=False)

    with pytest.raises(MLError, match="diverged.*refusing to register the last epoch"):
        train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", **SMALL_OK), say=Recorder(), fit_fn=fit)
    assert Registry(tmp_path / "data").list() == []

    def fit_ok(model, train, val, save_dir, cfg, device, logger):
        write_weights(save_dir, curve=[(0, 1.0), (1, 0.9)], best=False)

    e = train_candidate(tmp_path / "data", r.version, TrainOptions(base="x", **SMALL_OK), say=Recorder(), fit_fn=fit_ok)
    assert e["weights_file"] == "last.pt" and e["metrics"]["training"]["diverged"] is False
    assert any("no best.pt" in w for w in e["warnings"])


def test_legacy_snapshot_without_page_statistics_is_refused(tmp_path, stub_models):
    _, _, r = build(tmp_path, n=4)
    m = ds.load_manifest(tmp_path / "data", r.version)
    for c in m.crops:
        c.norm_lo = c.norm_hi = None
    import polony_app.ml.train as T
    orig = T.load_manifest
    T.load_manifest = lambda *a, **k: m
    try:
        with pytest.raises(MLError, match="before page-level normalisation"):
            plan_training(tmp_path / "data", r.version, TrainOptions(base="x", **SMALL_OK))
    finally:
        T.load_manifest = orig


# ---------------------------------------------------------------- CLI: eval-model at the model's configuration, promotion
def run(capsys, *argv):
    code = main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_build_downscale_and_train_safeguard(tmp_path, capsys, stub_models):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, 5)]
    d = str(tmp_path / "data")
    code, out, err = run(capsys, "--data-dir", d, "build", "--analyses", *ids, "--fractions", "0.6,0.4,0", "--created-at", T0,
                         "--downscale", "4")
    assert code == 0, err
    version = [l for l in out.splitlines() if l.startswith("dataset version:")][0].split()[-1]
    assert json.loads(run(capsys, "--data-dir", d, "show", version)[1])["options"]["downscale"] == 4
    code, out, err = run(capsys, "--data-dir", d, "train", "--snapshot", version, "--base", "x", "--dry-run")
    assert code == 2 and "--allow-small" in err
    code, out, err = run(capsys, "--data-dir", d, "train", "--snapshot", version, "--base", "x", "--dry-run", "--allow-small")
    assert code == 0, err
    assert "downscale 4" in out and "--allow-small: training anyway" in out


def register_ft(data_dir, mid, config):
    reg = Registry(data_dir)
    d = models_dir(data_dir) / mid
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"w")
    reg.add({"id": mid, "kind": "finetuned", "base": {"name": "general"}, "dataset_version": "ds-1", "artifact": mid,
             "weights_file": "best.pt", "weights_sha256": sha256_file(d / "best.pt"), "inference_config": config})
    return reg


def test_eval_model_runs_a_finetuned_model_at_its_own_configuration(tmp_path, capsys, monkeypatch):
    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i) for i in range(1, 7)]
    d = tmp_path / "data"
    r = ds.build_snapshot(store, d, ids, ds.BuildOptions(created_at=T0, fractions=TVT, downscale=4), say=Recorder())
    ds.lock_test_set(d, r.version)
    reg = register_ft(d, "ft-4", {"downscale": 4, "normaliser": "valid_unsaturated"})
    shapes = []

    def factory(base, data_dir, *, device="cpu", prob_thresh=None, downscale=1):
        shapes.append(("factory", base, downscale))
        return lambda x: (shapes.append(("x", x.shape, x.dtype)), np.zeros((0, 2)))[1]

    monkeypatch.setattr("polony_app.ml.evaluate.make_spotiflow_predictor", factory)
    code, out, err = run(capsys, "--data-dir", str(d), "eval-model", "--model", "ft-4", "--downscale", "1")
    assert code == 2 and "trained at downscale 4" in err and "different downscale" in err          # never native
    code, out, err = run(capsys, "--data-dir", str(d), "eval-model", "--model", "ft-4")
    assert code == 0, err
    assert ("factory", "ft-4", 4) in shapes
    xs = [s for s in shapes if s[0] == "x"]
    assert xs and all(s[2] == np.float32 and s[1][0] <= 120 // 4 and s[1][1] <= 160 // 4 for s in xs)
    ev = Registry(d).get("ft-4")["evaluations"][0]
    assert ev["config"] == {"downscale": 4, "normaliser": "valid_unsaturated"} and ev["params"]["downscale"] == 4
    assert ev["tolerance_px"] == 10.0 and set(ev["metrics"]["by_tolerance"]) == {"5", "10", "15"}
    assert "precision vs annotator's marks" in out and "@15 px" in out
    # an official pretrained model is evaluated at the downscale the researcher names, and that is recorded
    code, out, err = run(capsys, "--data-dir", str(d), "eval-model", "--model", "general", "--downscale", "2")
    assert code == 0, err
    assert Registry(d).get("pretrained-general")["evaluations"][0]["config"]["downscale"] == 2
    # promotion: the candidate (÷4) differs from the app's configuration (÷1, auto): refused until acknowledged
    code, out, err = run(capsys, "--data-dir", str(d), "promote", "ft-4")
    assert code == 2 and "app's current inference configuration" in err
    code, out, err = run(capsys, "--data-dir", str(d), "promote", "ft-4", "--app-downscale", "4", "--app-normaliser", "valid_unsaturated")
    assert code == 2 and "nothing to compare against" in err and "--downscale 4" in err           # baseline needs ITS config too
    code, out, err = run(capsys, "--data-dir", str(d), "eval-model", "--model", "general", "--downscale", "4")
    assert code == 0, err
    code, out, err = run(capsys, "--data-dir", str(d), "promote", "ft-4", "--app-downscale", "4", "--app-normaliser", "valid_unsaturated")
    assert code == 2 and "no F1 value" in err          # configuration checks passed; the stub predicts nothing, so F1 is undefined


@pytest.mark.skipif(os.environ.get("POLONY_ML_E2E") != "1", reason="opt-in")
def test_e2e_downscaled_finetune_runs_on_cpu(tmp_path):
    from spotiflow.model import Spotiflow, SpotiflowModelConfig
    yy, xx = np.mgrid[0:120, 0:160]

    def pages(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(2):
            img = rng.normal(500, 30, (120, 160))
            for (x, y) in [(20, 20), (40, 30), (60, 50), (100, 20), (120, 40), (50, 40)]:
                img += 8000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 4.0 ** 2))
            out.append(np.clip(img, 0, 65535).astype(np.uint16))
        return tuple(out)

    store = new_store(tmp_path)
    ids = [add_analysis(store, f"s{i}", slide_id=f"S{i}_A", seed=i, pages=pages(i)) for i in range(1, 7)]
    d = tmp_path / "data"
    r = ds.build_snapshot(store, d, ids, ds.BuildOptions(created_at=T0, fractions=TVT, downscale=2), say=Recorder())
    base = tmp_path / "tiny"
    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(base, which="best")
    e = train_candidate(d, r.version, TrainOptions(base=str(base), epochs=2, batch_size=2, crop_size=32, device="cpu",
                                                   allow_small=True), say=print)
    assert e["inference_config"]["downscale"] == 2
    from polony_app.ml import evaluate as ev
    pred = ev.make_spotiflow_predictor(e["id"], d, device="cpu", downscale=2)     # normalizer=None, subpix off, input already prepared
    res = ev.evaluate_on_snapshot(d, r.version, pred, split="val", downscale=2)
    assert res["config"]["downscale"] == 2 and res["n_crops"] > 0 and set(res["metrics"]["by_tolerance"]) == {"5", "10", "15"}
