"""Big-FISH 0.6.2 spot engine (2-D).

Verified against the installed source (bigfish.detection.detect_spots):
* images must be uint8/uint16/float32/float64; we always feed **float32** (the
  background-corrected image). uint16 would be cast through a lossy rescale in
  ``stack.log_filter`` and clip the negatives; float32 is passed through as is.
* Big-FISH's ``spot_radius`` (with ``voxel_size``) is converted to pixels and
  used directly as the LoG sigma AND the local-maximum window. Its docs call it
  "radius" but it is the gaussian standard deviation of the spot in pixels. We
  call with ``voxel_size=(1,1)``, ``spot_radius=(r,r)`` and ALSO the explicit
  ``log_kernel_size=(r,r)``, ``minimum_distance=(d,d)`` so that
  ``DetectionParams.minimum_distance_px`` is honoured independently of r (when
  all four are given Big-FISH uses the kernel/min-distance pair).
* ``threshold`` is in LoG-response units (of the float image we pass). With
  ``threshold=None`` Big-FISH selects it automatically at the kink of the
  spot-count-vs-threshold curve. We compute that value ourselves with
  ``stack.log_filter`` + ``local_maximum_detection`` + ``automated_threshold_setting``
  (what detect_spots does internally, but without building a Python list of
  every pixel -- unusable for big images), multiply by ``threshold_scale``, and pass
  the result as a *manual* threshold. A test checks equality with
  ``detect_spots(..., threshold=None, return_threshold=True)``.

Masking choice: pixels outside the valid mask are replaced by the median of
the valid pixels (~0 after background correction) BEFORE the LoG/threshold
computation, so bright structures outside the counting area (saturated slide
edge, debris) create neither spots nor a threshold bias, and there is no
artificial step edge. Spots whose peak pixel is outside the mask are dropped.
Spots next to the mask boundary are kept but measured only on valid pixels and
flagged ``near_edge`` downstream.

Tiling: the threshold is computed ONCE on the whole (masked) image, never per
tile -- a per-tile automated threshold depends on that tile's content (an empty
tile would pick a noise threshold), which would make results depend on the tile
grid. Tiles are cores of ``tile_size`` with a halo of >= ``tile_overlap_px``
(raised to the LoG kernel + local-max support); a spot belongs to the tile whose
CORE contains its integer peak pixel, so each spot is reported exactly once and
the result is deterministic and (up to float rounding) identical to the untiled run.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..schemas import DetectionParams
from .engine import RawSpot
from .errors import DetectionCancelled

_MAD_TO_SIGMA = 1.4826


class BigFishEngine:
    name = "bigfish"

    def __init__(self, annulus_inner: float = 2.0, annulus_outer: float = 3.5):
        """Local background annulus radii as multiples of the spot radius."""
        self.annulus_inner = annulus_inner
        self.annulus_outer = annulus_outer
        self.last_info: dict = {}

    def version(self) -> str:
        import bigfish
        return bigfish.__version__

    # ------------------------------------------------------------ public
    def detect(self, img, params: DetectionParams, valid_mask=None, raw=None, progress=None, cancelled=None,
               measure=None):
        spots, info = self.detect_with_info(img, params, valid_mask, raw, progress, cancelled, measure)
        self.last_info = info
        return spots

    # ------------------------------------------------------------ pieces
    @staticmethod
    def _prepare(img: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
        a = np.array(img, dtype=np.float32, copy=True, order="C")
        if not np.isfinite(a).all():
            raise ValueError("image contains NaN/inf")
        if valid is not None and not valid.all():
            fill = float(np.median(a[valid])) if valid.any() else 0.0
            a[~valid] = fill
        return a

    @staticmethod
    def log_filtered(img: np.ndarray, sigma: float) -> np.ndarray:
        from bigfish import stack
        return stack.log_filter(img, (sigma, sigma))

    def auto_threshold(self, filled: np.ndarray, params: DetectionParams,
                       valid: Optional[np.ndarray]) -> Optional[float]:
        """Big-FISH automated threshold (before threshold_scale)."""
        from bigfish.detection import automated_threshold_setting, local_maximum_detection
        r, d = float(params.spot_radius_px), float(params.minimum_distance_px)
        filt = self.log_filtered(filled, r)
        lm = local_maximum_detection(filt, (d, d))
        if valid is not None:
            lm &= valid
        if not lm.any():
            return None
        try:
            thr = automated_threshold_setting(filt, lm)
        except ValueError:  # curve without a kink (e.g. one candidate)
            return None
        return None if thr is None else float(thr)

    @staticmethod
    def _detect_coords(image: np.ndarray, thr: float, params: DetectionParams) -> np.ndarray:
        from bigfish.detection import detect_spots
        r, d = float(params.spot_radius_px), float(params.minimum_distance_px)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "No spots were detected"
            sp = detect_spots(image, threshold=float(thr), remove_duplicate=True, return_threshold=False,
                              voxel_size=(1, 1), spot_radius=(r, r), log_kernel_size=(r, r),
                              minimum_distance=(d, d))
        return np.asarray(sp, dtype=np.int64).reshape(-1, 2)

    # ------------------------------------------------------------ main
    def detect_with_info(self, img, params: DetectionParams, valid_mask=None, raw=None,
                         progress=None, cancelled=None, measure=None):
        progress = progress or (lambda f, m: None)
        cancelled = cancelled or (lambda: False)
        if img.ndim != 2:
            raise ValueError("BigFishEngine.detect expects a 2-D image")
        if valid_mask is not None and valid_mask.shape != img.shape:
            raise ValueError("valid_mask shape differs from image shape")
        if raw is not None and raw.shape != img.shape:
            raise ValueError("raw shape differs from image shape")
        if measure is not None and measure.shape != img.shape:
            raise ValueError("measure shape differs from image shape")
        valid = None if valid_mask is None else np.asarray(valid_mask, bool)
        info: dict = {"notes": [], "threshold": None, "auto_threshold": None, "n_tiles": 1}
        h, w = img.shape
        if valid is not None and not valid.any():
            info["notes"].append("no valid pixels: nothing to detect")
            return [], info

        r, md = float(params.spot_radius_px), float(params.minimum_distance_px)
        filled = self._prepare(img, valid)
        progress(0.0, "threshold")
        if cancelled():
            raise DetectionCancelled()

        if params.threshold is not None:
            thr = float(params.threshold)
            info["notes"].append("manual threshold used as given (threshold_scale ignored)")
        else:
            auto = self.auto_threshold(filled, params, valid)
            info["auto_threshold"] = auto
            if auto is None:
                info["notes"].append("automatic threshold failed (no usable spot-count curve); no spots reported")
                return [], info
            thr = auto * float(params.threshold_scale)
        info["threshold"] = thr
        if cancelled():
            raise DetectionCancelled()

        # ---- coordinates (whole image or tiles)
        ts = int(params.tile_size)
        if ts <= 0 or ts >= max(h, w):
            progress(0.2, "detecting")
            coords = self._detect_coords(filled, thr, params)
        else:
            support = int(math.ceil(4 * r)) + int(math.ceil(md)) + 2
            halo = max(int(params.tile_overlap_px), support)
            if halo != params.tile_overlap_px:
                info["notes"].append(f"tile overlap raised from {params.tile_overlap_px} to {halo}px "
                                     "(LoG kernel + local-maximum support)")
            ys, xs = list(range(0, h, ts)), list(range(0, w, ts))
            info["n_tiles"] = len(ys) * len(xs)
            parts = []
            k = 0
            for y0 in ys:
                for x0 in xs:
                    k += 1
                    if cancelled():
                        raise DetectionCancelled()
                    progress(0.2 + 0.5 * k / info["n_tiles"], f"tile {k}/{info['n_tiles']}")
                    y1, x1 = min(y0 + ts, h), min(x0 + ts, w)
                    if valid is not None and not valid[y0:y1, x0:x1].any():
                        continue
                    py0, py1 = max(0, y0 - halo), min(h, y1 + halo)
                    px0, px1 = max(0, x0 - halo), min(w, x1 + halo)
                    c = self._detect_coords(np.ascontiguousarray(filled[py0:py1, px0:px1]), thr, params)
                    if c.size == 0:
                        continue
                    c = c + np.array([py0, px0])
                    keep = (c[:, 0] >= y0) & (c[:, 0] < y1) & (c[:, 1] >= x0) & (c[:, 1] < x1)
                    parts.append(c[keep])
            coords = np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))] if coords.size else coords

        if valid is not None and coords.size:
            coords = coords[valid[coords[:, 0], coords[:, 1]]]
        if cancelled():
            raise DetectionCancelled()

        # ---- optional dense decomposition (global: needs all spots)
        simulated = np.zeros(len(coords), bool)
        if params.dense_decomposition and len(coords) >= 2:
            progress(0.75, "dense decomposition")
            try:
                from bigfish.detection import decompose_dense
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new, regions, _ref = decompose_dense(filled, coords, voxel_size=(1, 1), spot_radius=(r, r))
                new = np.asarray(new, dtype=np.int64).reshape(-1, 2)
                old = {(int(a), int(b)) for a, b in coords}
                add = np.array([p for p in new if (int(p[0]), int(p[1])) not in old],
                               dtype=np.int64).reshape(-1, 2)
                if valid is not None and add.size:
                    add = add[valid[add[:, 0], add[:, 1]]]
                if add.size:
                    coords = np.concatenate([coords, add])
                    simulated = np.concatenate([simulated, np.ones(len(add), bool)])
                    o = np.lexsort((coords[:, 1], coords[:, 0]))
                    coords, simulated = coords[o], simulated[o]
                info["notes"].append(f"dense decomposition added {int(simulated.sum())} simulated spot(s) "
                                     f"in {len(regions)} dense region(s)")
            except Exception as exc:  # decomposition is best-effort
                info["notes"].append(f"dense decomposition failed ({type(exc).__name__}: {exc}); skipped")

        progress(0.85, "measuring")
        spots = self._measure(filled, valid, coords, simulated, params, raw, thr, measure)
        spots = self._suppress(spots, params.remove_duplicate_radius_px)
        spots.sort(key=lambda s: (s.y, s.x))
        progress(1.0, "done")
        return spots, info

    # ------------------------------------------------------------ measurement
    def _measure(self, img, valid, coords, simulated, params, raw, thr, measure=None) -> list[RawSpot]:
        if len(coords) == 0:
            return []
        h, w = img.shape
        r = float(params.spot_radius_px)
        rin = max(self.annulus_inner * r, 3.0)
        rout = max(self.annulus_outer * r, rin + 3.0)
        R = int(math.ceil(rout))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr2 = yy * yy + xx * xx
        ann = (rr2 >= rin * rin) & (rr2 <= rout * rout)
        pimg = np.pad(img, R, mode="constant", constant_values=np.nan)  # detection image (masked pixels filled)
        pmeas = pimg if measure is None else np.pad(np.asarray(measure, np.float32), R, mode="constant",
                                                    constant_values=np.nan)
        pval = None
        if valid is not None:
            pval = np.pad(valid, R, mode="constant", constant_values=False)
        # LoG support for per-spot response/sub-pixel position
        K = int(4.0 * r + 0.5) + 2
        hc = max(1, int(round(r / 2)))
        he = int(math.ceil(2.5 * r)) + 1
        he = min(he, R)
        dmax = None
        if raw is not None and raw.dtype.kind in "ui":
            dmax = np.iinfo(raw.dtype).max
        rs = int(math.ceil(max(r, 1.0)))
        sy, sx = np.mgrid[-rs:rs + 1, -rs:rs + 1]
        sdisk = (sy * sy + sx * sx) <= max(r, 1.0) ** 2

        out: list[RawSpot] = []
        for (y, x), sim in zip(coords, simulated):
            y, x = int(y), int(x)
            win = pimg[y:y + 2 * R + 1, x:x + 2 * R + 1]
            mwin = pmeas[y:y + 2 * R + 1, x:x + 2 * R + 1]
            use = ann & np.isfinite(mwin)
            if pval is not None:
                use &= pval[y:y + 2 * R + 1, x:x + 2 * R + 1]
            vals = mwin[use]
            if vals.size >= 12:
                bg = float(np.median(vals))
                std = _MAD_TO_SIGMA * float(np.median(np.abs(vals - bg)))
            else:
                bg, std = float("nan"), 0.0
            c3 = mwin[R - 1:R + 2, R - 1:R + 2]
            peak = float(np.nanmax(c3))
            snr = None
            if not math.isnan(bg) and std > 1e-9 * max(1.0, abs(peak)):
                snr = float((peak - bg) / std)
            # LoG response and sub-pixel refinement on the matched-filter output
            y0c, y1c, x0c, x1c = max(0, y - K), min(h, y + K + 1), max(0, x - K), min(w, x + K + 1)
            crop = img[y0c:y1c, x0c:x1c]
            pad = ((max(0, K - y), max(0, y + K + 1 - h)), (max(0, K - x), max(0, x + K + 1 - w)))
            if any(p for pp in pad for p in pp):
                crop = np.pad(crop, pad, mode="symmetric")
            lg = np.clip(-ndi.gaussian_laplace(crop, sigma=r, mode="reflect"), 0, None)
            L = lg[K - 1:K + 2, K - 1:K + 2]
            resp = float(L[1, 1])
            dy = dx = 0.0
            if L[1, 1] >= L.max() and L[1, 1] > 0:
                dy = _parab(L[0, 1], L[1, 1], L[2, 1])
                dx = _parab(L[1, 0], L[1, 1], L[1, 2])
            # elongation of the half-max blob
            elong = None
            sub = win[R - he:R + he + 1, R - he:R + he + 1]
            bgz = 0.0 if math.isnan(bg) else bg
            d = np.nan_to_num(sub - bgz, nan=0.0)
            pk = peak - bgz
            if pk > 0:
                m = d >= 0.5 * pk
                lab, _n = ndi.label(m)
                c3d = d[he - 1:he + 2, he - 1:he + 2]
                iy, ix = np.unravel_index(int(np.argmax(c3d)), c3d.shape)
                lbl = lab[he - 1 + iy, he - 1 + ix]
                if lbl:
                    comp = lab == lbl
                    if comp.sum() >= 6:
                        gy, gx = np.mgrid[-he:he + 1, -he:he + 1]
                        wt = d[comp]
                        px_, py_ = gx[comp], gy[comp]
                        sw = wt.sum()
                        mx, my = (wt * px_).sum() / sw, (wt * py_).sum() / sw
                        cxx = (wt * (px_ - mx) ** 2).sum() / sw + 1 / 12
                        cyy = (wt * (py_ - my) ** 2).sum() / sw + 1 / 12
                        cxy = (wt * (px_ - mx) * (py_ - my)).sum() / sw
                        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
                        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
                        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-9)
                        elong = float(math.sqrt(l1 / l2))
            sat = False
            if dmax is not None:
                a0, a1, b0, b1 = max(0, y - rs), min(h, y + rs + 1), max(0, x - rs), min(w, x + rs + 1)
                dm = sdisk[a0 - (y - rs):a1 - (y - rs), b0 - (x - rs):b1 - (x - rs)]
                sat = bool((raw[a0:a1, b0:b1][dm] >= dmax).any())
            out.append(RawSpot(x=float(x + dx), y=float(y + dy), peak=peak, background=bg, snr=snr,
                               saturated=sat, log_response=resp, threshold=float(thr),
                               elongation=elong, simulated=bool(sim)))
        return out

    @staticmethod
    def _suppress(spots: list[RawSpot], radius: float) -> list[RawSpot]:
        """Greedy: strongest LoG response first (ties by y,x); drop weaker spots
        closer than ``radius``. Deterministic."""
        if radius <= 0 or len(spots) < 2:
            return spots
        order = sorted(range(len(spots)), key=lambda i: (-(spots[i].log_response or 0.0), spots[i].y, spots[i].x))
        xy = np.array([[s.x, s.y] for s in spots])
        tree = cKDTree(xy)
        dead = np.zeros(len(spots), bool)
        for i in order:
            if dead[i]:
                continue
            for j in tree.query_ball_point(xy[i], radius):
                if j != i:
                    dead[j] = True
        return [s for s, dd in zip(spots, dead) if not dd]


def _parab(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0.0 if d == 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
