import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from polony_app.imaging.embedded import embedded_points
from polony_app.imaging.ingest import IngestError, ingest_files, parse_info_label_hint
from polony_app.imaging.io import load_channel, load_rgb
from polony_app.schemas import ImageKind

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn

REAL = Path(__file__).resolve().parents[2] / "resources/dataset_files/test/test_img.tif"   # synthetic example
REAL_TRUTH = REAL.with_name("test_img.truth.json")                                         # its generator's truth


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def scene():
    return syn.make_scene((120, 150), [(50, 50, 3000)], [(80, 60, 3000)], seed=5)


def test_two_channel_imagej_preserves_pixels_and_source(tmp_path, scene):
    src = syn.write_two_channel(tmp_path / "a.tif", scene.red, scene.green, info="ImageDescription: 635 [W1]\nSoftware: x\n",
                                roi_points=np.array([[10.5, 20.5], [30, 40]]))
    before = sha(src)
    mtime = src.stat().st_mtime_ns
    dest = tmp_path / "store"
    meta = ingest_files([("My File (1).tif", src)], dest)
    assert sha(src) == before and src.stat().st_mtime_ns == mtime
    assert meta.kind == ImageKind.raw_two_channel and not meta.composite_only
    assert (meta.height, meta.width) == scene.red.shape
    assert meta.dtype == "uint16" and meta.bit_depth == 16
    stored = list(dest.iterdir())
    assert len(stored) == 1 and meta.sha256[stored[0].name] == before
    r, g = load_channel(meta, dest, 0), load_channel(meta, dest, 1)
    assert r.dtype == np.uint16 and np.array_equal(r, scene.red) and np.array_equal(g, scene.green)
    assert meta.channels[0].label_hint == "635 [W1]"
    assert meta.channels[1].label_hint is None
    assert all(c.role is None and not c.role_confirmed for c in meta.channels)
    assert meta.pixel_size_um is None
    assert meta.embedded_rois and meta.embedded_rois[0].roi_type == "POINT" and meta.embedded_rois[0].n_points == 2
    assert embedded_points(src) == [(10.5, 20.5), (30.0, 40.0)]
    assert meta.channels[0].max == int(scene.red.max())


def test_plain_multipage_tiff(tmp_path, scene):
    src = syn.write_two_channel(tmp_path / "p.tif", scene.red, scene.green, imagej=False)
    meta = ingest_files([("p.tif", src)], tmp_path / "s")
    assert meta.kind == ImageKind.raw_two_channel
    assert np.array_equal(load_channel(meta, tmp_path / "s", 1), scene.green)
    assert all(c.role is None for c in meta.channels)


def test_separate_files_never_confirm_roles_even_with_color_names(tmp_path, scene):
    a = syn.write_single(tmp_path / "red.tif", scene.red)
    b = syn.write_single(tmp_path / "green.tif", scene.green)
    meta = ingest_files([("red.tif", a), ("green.tif", b)], tmp_path / "s")
    assert meta.kind == ImageKind.raw_separate_files
    assert all(c.role is None and not c.role_confirmed for c in meta.channels)
    assert len(meta.sha256) == 2
    assert np.array_equal(load_channel(meta, tmp_path / "s", 0), scene.red)


def test_rgb_composite(tmp_path, scene):
    src = syn.write_rgb(tmp_path / "c.tif", scene.red, scene.green)
    meta = ingest_files([("c.tif", src)], tmp_path / "s")
    assert meta.kind == ImageKind.rgb_composite and meta.composite_only
    assert any("raw per-channel" in w for w in meta.warnings)
    rgb = load_rgb(meta, tmp_path / "s")
    assert rgb.shape == scene.red.shape + (3,) and rgb.dtype == np.uint8
    assert all(c.role is None for c in meta.channels)
    assert np.array_equal(load_channel(meta, tmp_path / "s", 0), rgb[..., 0])


def test_shape_mismatch_is_actionable(tmp_path):
    a = syn.write_single(tmp_path / "a.tif", np.zeros((50, 60), np.uint16))
    b = syn.write_single(tmp_path / "b.tif", np.zeros((50, 70), np.uint16))
    with pytest.raises(IngestError, match="never resamples"):
        ingest_files([("a.tif", a), ("b.tif", b)], tmp_path / "s")


@pytest.mark.parametrize("arr,msg", [
    (np.zeros((40, 40), np.float32), "not supported"),
    (np.zeros((3, 40, 40), np.uint16), "planes"),
])
def test_bad_inputs(tmp_path, arr, msg):
    p = tmp_path / "x.tif"
    tifffile.imwrite(str(p), arr, photometric="minisblack")
    other = syn.write_single(tmp_path / "y.tif", np.zeros((40, 40), np.uint16))
    with pytest.raises(IngestError, match=msg):
        ingest_files([("x.tif", p)], tmp_path / "s")
    assert other.exists()


def test_single_channel_single_file_and_too_many(tmp_path):
    p = syn.write_single(tmp_path / "x.tif", np.zeros((40, 40), np.uint16))
    with pytest.raises(IngestError, match="single-channel"):
        ingest_files([("x.tif", p)], tmp_path / "s")
    with pytest.raises(IngestError, match="at most 2"):
        ingest_files([("x.tif", p)] * 3, tmp_path / "s")
    with pytest.raises(IngestError):
        ingest_files([], tmp_path / "s")


def test_saturation_stats(tmp_path):
    a = np.full((40, 50), 100, np.uint16)
    a[:10] = 65535
    p = syn.write_two_channel(tmp_path / "x.tif", a, np.zeros_like(a))
    meta = ingest_files([("x.tif", p)], tmp_path / "s")
    assert meta.channels[0].saturated_fraction == pytest.approx(0.25)
    assert (meta.channels[0].min, meta.channels[0].max) == (100, 65535)
    assert any("saturated" in w for w in meta.warnings)


def test_info_hint_parser():
    assert parse_info_label_hint("Foo: 1\nImageDescription: 635 [W1]\n") == "635 [W1]"
    assert parse_info_label_hint(None) is None


@pytest.mark.skipif(not REAL.exists(), reason="real example image missing")
def test_real_example(tmp_path):
    before = sha(REAL)
    meta = ingest_files([("test_img.tif", REAL)], tmp_path / "s")
    assert sha(REAL) == before
    assert (meta.height, meta.width) == (2450, 1438) and meta.dtype == "uint16"
    assert meta.channels[0].label_hint == "635 [W1]"
    assert all(c.role is None and not c.role_confirmed for c in meta.channels)
    kinds = {(r.source, r.roi_type) for r in meta.embedded_rois}
    assert ("ROI", "POINT") in kinds
    truth = json.loads(REAL_TRUTH.read_text())
    n_marks, bb = len(truth["marks"]), truth["roi_bbox"]
    roi = next(r for r in meta.embedded_rois if r.source == "ROI")
    assert (roi.left, roi.top, roi.right, roi.bottom, roi.n_points) == (bb["left"], bb["top"], bb["right"], bb["bottom"], n_marks)
    assert len(embedded_points(REAL)) == n_marks
    assert meta.pixel_size_um is None and any("1000" in w for w in meta.warnings)
    ref = tifffile.imread(str(REAL))
    assert np.array_equal(load_channel(meta, tmp_path / "s", 1), ref[1])
    # the committed TIFF is exactly what resources/dataset_files/test/make_test_img.py writes
    assert hashlib.sha256(np.ascontiguousarray(ref, dtype="<u2").tobytes()).hexdigest() == truth["pixels_sha256"]


@pytest.mark.skipif(not REAL.exists(), reason="real example image missing")
def test_example_image_carries_no_instrument_metadata():
    """The example is synthetic: only generic ImageJ metadata, no scanner settings, dates or identifiers."""
    with tifffile.TiffFile(str(REAL)) as t:
        text = repr(t.imagej_metadata) + "".join(str(tag.value) for p in t.pages for tag in p.tags.values()
                                                 if tag.name not in ("StripOffsets", "StripByteCounts"))
    for word in ("AxImageIO", "PMT", "LaserPower", "DateTime", "Barcode", "HostComputer", "Artist"):
        assert word not in text, word
