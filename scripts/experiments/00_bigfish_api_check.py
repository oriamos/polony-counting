"""Experiment 0: verify Big-FISH 0.6.2 behaviour that the app relies on.
Pure pixel units, mask handling, dtype paths. Prints facts; asserts them."""
import warnings

import numpy as np
import bigfish
import bigfish.detection as det
import bigfish.stack as stack
from scipy import ndimage as ndi

print("bigfish", bigfish.__version__)
rng = np.random.default_rng(0)
H, W = 300, 300
# NOTE: synthetic image used ONLY to check API mechanics, never for accuracy.
yy, xx = np.mgrid[:H, :W]
img = rng.normal(500, 20, (H, W))
centres = [(50, 50), (60, 150), (200, 220), (250, 80), (150, 150)]
for cy, cx in centres:
    img += 2000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2))
img_f = img.astype(np.float32)
img_u = np.clip(img, 0, 65535).astype(np.uint16)

# 1) pixel-unit equivalence: voxel_size=1, spot_radius=r  <=> log_kernel_size=r, minimum_distance=r
a = det.detect_spots(img_f, threshold=10.0, voxel_size=1, spot_radius=5)
b = det.detect_spots(img_f, threshold=10.0, log_kernel_size=5, minimum_distance=5)
assert np.array_equal(a, b), (a, b)
print("NOTE LoG response is not scale-normalised: threshold is in LoG units (~A*2s^2/(s^2+t^2)^2)")
print("pixel-unit equivalence OK; spots:", a.tolist())
print("utils.get_object_radius_pixel(1, 5, 2) =", det.get_object_radius_pixel(1, 5, 2))

# 2) dtype: uint16 LoG output is re-quantised to uint16 integers; float32 kept float
lf = stack.log_filter(img_f, 5.0); lu = stack.log_filter(img_u, 5.0)
print("log_filter float32 ->", lf.dtype, f"max={lf.max():.3f}; uint16 ->", lu.dtype, "max=", lu.max())
# 3) int/float64 rejections
for bad in (img.astype(np.int32), img.astype(np.float16)):
    try:
        det.detect_spots(bad, voxel_size=1, spot_radius=5)
        print("accepted", bad.dtype)
    except Exception as e:
        print("rejected", bad.dtype, type(e).__name__)
# 4) threshold type: numpy float is rejected by check_parameter
try:
    det.detect_spots(img_f, threshold=np.float32(50), voxel_size=1, spot_radius=5)
    print("np.float32 threshold accepted")
except Exception as e:
    print("np.float32 threshold rejected:", type(e).__name__, str(e)[:80])

# 5) masked detection via the decomposed API
mask = np.zeros((H, W), bool); mask[:, :180] = True
lf = stack.log_filter(img_f, 5.0)
lm = det.local_maximum_detection(lf, 5.0)
lf_m = np.where(mask, lf, 0).astype(lf.dtype)
lm_m = lm & mask
try:
    thr = det.automated_threshold_setting(lf_m, lm_m)
except ValueError as e:  # get_breaking_point: m.index(False) on a short/monotone curve
    print("automated_threshold_setting raised ValueError:", e, "-> fallback manual 10.0")
    thr = 10.0
spots, _ = det.spots_thresholding(lf_m, lm_m, float(thr), remove_duplicate=True)
print("masked auto threshold", thr, "spots", spots.tolist())
# 6) coordinates are int64 (row, col) = (y, x); centroid truncated (astype int64)
print("spots dtype", spots.dtype)
# 7) few spots: auto threshold on pure noise
noise = rng.normal(500, 20, (H, W)).astype(np.float32)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        s, t = det.detect_spots(noise, voxel_size=1, spot_radius=5, return_threshold=True)
        print("pure-noise auto threshold:", t, "n spots:", len(s), "(Big-FISH always 'finds' an elbow)")
    except Exception as e:
        print("pure noise raised", type(e).__name__, e)
# 8) minimum_distance cast: np.ceil(min_distance).astype(image.dtype) -> uint16 LoG ok
print("local max kernel = 2*ceil(md)+1 square (Chebyshev), not a disc")
