import io

import numpy as np
import pytest
from PIL import Image

from polony_app.imaging.preprocess import correct_background, gaussian_background
from polony_app.imaging.render import percentile_range, render_composite_png, render_png
from polony_app.schemas import BackgroundParams

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn


def decode(b):
    return np.asarray(Image.open(io.BytesIO(b)))


def test_render_window_gamma_crop_maxdim():
    a = np.tile(np.arange(0, 1000, 10, dtype=np.uint16), (50, 1))  # 50x100
    im = decode(render_png(a, 100, 900))
    assert im.shape == (50, 100) and im[0, 0] == 0 and im[0, -1] == 255
    g = decode(render_png(a, 100, 900, gamma=2.0))
    assert g[0, 50] > im[0, 50]  # gamma>1 brightens
    c = decode(render_png(a, 0, 1000, crop=(10, 5, 30, 25)))
    assert c.shape == (20, 20)
    s = decode(render_png(a, 0, 1000, max_dim=25))
    assert max(s.shape) == 25
    r = decode(render_png(a, 0, 1000, colormap="red"))
    assert r.shape == (50, 100, 3) and r[..., 1:].max() == 0
    with pytest.raises(ValueError):
        render_png(a, 0, 1, crop=(500, 500, 600, 600))
    assert decode(render_composite_png(a, a, (0, 1000), (0, 1000))).shape == (50, 100, 3)


def test_render_does_not_modify_input():
    a = (np.arange(100, dtype=np.uint16).reshape(10, 10))
    b = a.copy()
    render_png(a, 0, 50, 0.5)
    assert np.array_equal(a, b)


def test_percentile_range():
    lo, hi = percentile_range(np.arange(1000), 1, 99)
    assert 5 < lo < 15 and 980 < hi < 995
    assert percentile_range(np.zeros((4, 4)))[1] > 0


def test_gaussian_subtract_removes_uneven_background():
    rng = np.random.default_rng(0)
    bg = syn.uneven_background((300, 400))
    img = bg + rng.normal(0, 10, bg.shape)
    img[100:105, 200:205] += 3000  # a spot must not lift the estimate
    out = correct_background(img.astype(np.float32), BackgroundParams(sigma_px=40, clip_negative=False))
    assert out.dtype == np.float32
    quiet = np.ones(out.shape, bool); quiet[95:110, 195:210] = False
    assert abs(out[quiet].mean()) < 100 and out[quiet].std() < 0.5 * bg[quiet].std()  # large-scale bump is partially kept by any finite sigma
    assert (bg.max() - bg.min()) > 1500
    assert out[100:105, 200:205].mean() > 2500


def test_clip_negative_and_none():
    img = np.random.default_rng(1).normal(1000, 20, (80, 90)).astype(np.float32)
    clipped = correct_background(img, BackgroundParams(sigma_px=10, clip_negative=True))
    assert clipped.min() >= 0
    same = correct_background(img.astype(np.uint16), BackgroundParams(method="none"))
    assert same.dtype == np.float32 and np.array_equal(same, img.astype(np.uint16).astype(np.float32))


def test_mask_aware_background_ignores_bright_outside():
    from polony_app.imaging.preprocess import _masked_gaussian_direct
    img = np.full((200, 200), 1000.0, np.float32)
    img[:, :40] = 60000
    valid = np.ones(img.shape, bool); valid[:, :40] = False
    plain = _masked_gaussian_direct(img, np.ones_like(img), 30.0)
    masked = _masked_gaussian_direct(img, valid.astype(np.float32), 30.0)
    assert abs(masked[:, 60:] - 1000).max() < 1e-2 and abs(plain[:, 60:] - 1000).max() > 50
    # full path with noise
    noisy = img + np.random.default_rng(0).normal(0, 10, img.shape).astype(np.float32)
    out = correct_background(noisy, BackgroundParams(sigma_px=30, clip_negative=False), valid)
    assert abs(out[:, 60:].mean()) < 1.0


def test_fast_path_close_to_direct():
    from polony_app.imaging.preprocess import _masked_gaussian, _masked_gaussian_direct
    bg = syn.uneven_background((200, 260)).astype(np.float32)
    w = np.ones_like(bg)
    assert np.abs(_masked_gaussian(bg, w, 40.0) - _masked_gaussian_direct(bg, w, 40.0)).max() < 5.0


def test_white_tophat_matches_exact_disk():
    from scipy import ndimage as ndi
    from skimage.morphology import disk
    img = np.random.default_rng(2).normal(100, 30, (70, 80)).astype(np.float32)
    out = correct_background(img, BackgroundParams(method="white_tophat", radius_px=6))
    exact = img - ndi.grey_opening(img, footprint=disk(6))
    assert np.allclose(out, exact, atol=1e-4)
    assert out.min() >= 0
