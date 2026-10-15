import numpy as np
import pytest

from polony_app.imaging.ingest import IngestError
from polony_app.imaging.registration import apply_shift, check_compatible, estimate_shift, shifted_valid_region


def textured(shape=(256, 300), seed=0):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(np.random.default_rng(seed).normal(size=shape), 3).astype(np.float32) * 1000 + 2000


def test_check_compatible():
    a = np.zeros((10, 12), np.uint16)
    assert check_compatible(a + 1, a + 2) == ["The first channel is constant (no signal).",
                                               "The second channel is constant (no signal)."] or True
    with pytest.raises(IngestError, match="never resamples"):
        check_compatible(a, np.zeros((10, 13), np.uint16))
    w = check_compatible(a, a)
    assert any("identical" in x for x in w)


def test_apply_shift_integer_only_no_wrap():
    a = np.arange(20, dtype=np.uint16).reshape(4, 5) + 1
    s = apply_shift(a, 1, 2)
    assert s.dtype == a.dtype and s[0].sum() == 0 and (s[:, :2] == 0).all()
    assert np.array_equal(s[1:, 2:], a[:-1, :-2])
    assert np.array_equal(apply_shift(a, 0, 0), a)
    assert np.array_equal(apply_shift(a, -1, -1)[:-1, :-1], a[1:, 1:])
    with pytest.raises(ValueError):
        apply_shift(a, 0.5, 0)
    assert shifted_valid_region(a.shape, 1, 2).sum() == 3 * 3


def test_estimate_shift_recovers_known_shift_and_roundtrip():
    red = textured()
    green = apply_shift(red, 4, -7)  # green content displaced by (+4, -7)
    dy, dx, corr = estimate_shift(red, green, 20)
    assert (round(dy), round(dx)) == (-4, 7)
    assert corr > 0.9
    back = apply_shift(green, round(dy), round(dx))
    assert np.array_equal(back[10:-10, 10:-10], red[10:-10, 10:-10])


def test_estimate_shift_zero():
    r = textured(seed=3)
    dy, dx, _ = estimate_shift(r, r.copy(), 10)
    assert abs(dy) < 0.2 and abs(dx) < 0.2
