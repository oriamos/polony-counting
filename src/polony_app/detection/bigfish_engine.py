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
* Big-FISH's ``threshold`` is in LoG-response units (of the float image we pass). THIS APP
  expresses every threshold in NOISE UNITS instead: LoG response / noise_sigma, with
  noise_sigma = 1.4826*MAD of the signed LoG response over valid pixels (background-corrected
  image). Reason: Big-FISH's elbow search uses an absolute candidate grid (integer steps up to the
  99.9999th percentile), so on raw units the automatic threshold -- and the detections -- changed
  with image brightness/exposure. ``DetectionParams.threshold`` (manual) is in the same noise
  units; the absolute LoG threshold used internally is ``threshold * noise_sigma``. With
  ``threshold=None`` Big-FISH selects it automatically at the kink of the
  spot-count-vs-threshold curve of the noise-normalised response. We compute that value ourselves with
  ``stack.log_filter`` + ``local_maximum_detection`` + ``automated_threshold_setting``
  (what detect_spots does internally, but without building a Python list of
  every pixel -- unusable for big images), multiply by ``threshold_scale``, and pass
  the result as a *manual* threshold. A test checks equality with
  ``detect_spots(..., threshold=None, return_threshold=True)`` (``auto_threshold`` without
  ``noise_sigma`` is exactly Big-FISH's raw-unit value; the app multiplies nothing by it).

Masking (SCIENCE_NOTES 1.4): the LoG is computed on the REAL background-corrected image
and its response is set to zero outside the valid mask; local maxima of the real response
are restricted to the mask (+1 px: the final keep/drop decision is made by the pipeline on
the refined centre with the single region rule) and the automatic threshold is derived from
maxima strictly inside the mask only. A spot centred outside the mask is a maximum outside
and is not produced, and a spot next to the boundary keeps an undistorted position (no
median-fill step edge). Pixels are replaced by the median of the valid pixels only when
they are farther than 4 sigma from the mask (cannot influence valid responses; saves
nothing but is harmless) or when they are SATURATED and outside the mask (rim glare must
not leak into valid pixels through the LoG kernel). Spots next to the boundary are flagged
``near_edge`` downstream.

Saturated plateaus: a clipped spot is a flat plateau whose LoG response is a ring (several
local maxima). Connected saturated components (raw >= dtype max, inside the mask) are
labelled and all spots within the component dilated by the spot radius are merged into ONE
spot at the plateau centroid, flagged saturated; a plateau much larger than the spot
(equivalent diameter > PLATEAU_SHAPE_FACTOR x spot radius), an elongated one (plateau
elongation > PLATEAU_ELONGATION_MAX) or one that carried more than one LoG maximum is also flagged
suspicious_shape (debris, or touching saturated polonies that cannot be resolved from clipped data;
the merge is kept, not split). Merged spots have snr = None (clipped peak, annulus possibly inside
the plateau), so weak_signal / near_threshold are not evaluated for them.

Automatic threshold is computed once on the whole (masked) image per channel
(``estimate_threshold``) and can be supplied back through ``threshold_info`` so that
previews and tiles reuse the identical value.

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
NOISE_K = 5.0
PLATEAU_ELONGATION_MAX = 1.3  # merged saturated plateau elongated beyond this => suspicious_shape
PLATEAU_SHAPE_FACTOR = 6.0   # saturated plateau diameter > this x spot radius => suspicious_shape


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
               measure=None, threshold_info=None):
        spots, info = self.detect_with_info(img, params, valid_mask, raw, progress, cancelled, measure,
                                            threshold_info)
        self.last_info = info
        return spots

    # ------------------------------------------------------------ pieces
    @staticmethod
    def _prepare(img: np.ndarray, valid: Optional[np.ndarray], raw: Optional[np.ndarray] = None,
                 sigma: float = 5.0) -> np.ndarray:
        """float32 copy of the detection image. NOT median-filled inside the LoG support
        of the mask: only pixels > 4 sigma away from the mask and saturated pixels outside
        the mask are replaced (by the median of the valid pixels)."""
        a = np.array(img, dtype=np.float32, copy=True, order="C")
        if not np.isfinite(a).all():
            raise ValueError("image contains NaN/inf")
        if valid is not None and not valid.all():
            if not valid.any():
                return a
            fill = float(np.median(a[valid]))
            far = ndi.distance_transform_edt(~valid) > (4.0 * sigma + 1.0)
            repl = far
            if raw is not None and raw.dtype.kind in "ui":
                repl = repl | ((raw >= np.iinfo(raw.dtype).max) & ~valid)
            a[repl] = fill
        return a

    @staticmethod
    def log_filtered(img: np.ndarray, sigma: float) -> np.ndarray:
        from bigfish import stack
        return stack.log_filter(img, (sigma, sigma))

    @staticmethod
    def _log_maxima(image: np.ndarray, valid: Optional[np.ndarray], params: DetectionParams,
                    extended: bool = False):
        """(LoG response with the outside of the mask zeroed, local maxima inside the mask).

        ``extended=True`` accepts maxima up to 1 px outside the mask: the integer maximum of
        a spot centred at x=99.6 lies on pixel 100 (outside a region [0,100)) although its
        refined centre is inside. The final keep/drop decision is made on the refined
        centre by the pipeline (single region rule). The automatic threshold always uses
        the strict (inside-only) maxima."""
        from bigfish.detection import local_maximum_detection
        r, d = float(params.spot_radius_px), float(params.minimum_distance_px)
        filt = BigFishEngine.log_filtered(image, r)
        # local maxima of the REAL response (no artificial edge at the mask boundary: a spot
        # centred just outside the mask is a maximum outside, not at the boundary pixel
        # inside), then restricted to the mask; thresholds see the zeroed response.
        lm = local_maximum_detection(filt, (d, d))
        if valid is not None:
            keep = ndi.binary_dilation(valid, structure=np.ones((3, 3), bool)) if extended else valid
            filt = np.where(keep, filt, np.zeros((), filt.dtype))
            lm &= keep
        return filt, lm

    def auto_threshold(self, filled: np.ndarray, params: DetectionParams,
                       valid: Optional[np.ndarray], noise_sigma: Optional[float] = None) -> Optional[float]:
        """Big-FISH automated threshold (before threshold_scale), from local maxima
        INSIDE the mask of the real LoG response. With ``noise_sigma`` the response is divided
        by it first, so the result is in NOISE UNITS (Big-FISH's candidate grid is absolute --
        integer steps up to the 99.9999th percentile -- so on raw LoG units the elbow moved with
        image brightness; in noise units it does not). Without it: raw LoG units (as Big-FISH)."""
        from bigfish.detection import automated_threshold_setting
        filt, lm = self._log_maxima(filled, valid, params)
        if not lm.any():
            return None
        if noise_sigma is not None:
            filt = (filt / np.float32(noise_sigma)).astype(np.float32)
        try:
            thr = automated_threshold_setting(filt, lm)
        except ValueError:  # curve without a kink (e.g. one candidate)
            return None
        return None if thr is None else float(thr)

    @staticmethod
    def noise_sigma(filled: np.ndarray, params: DetectionParams, valid: Optional[np.ndarray]) -> Optional[float]:
        """Robust noise sigma (1.4826*MAD) of the SIGNED -LoG response over valid pixels of the
        (background-corrected) image: the unit of every threshold. None if undefined/zero."""
        r = float(params.spot_radius_px)
        resp = -ndi.gaussian_laplace(np.asarray(filled, np.float32), sigma=r, mode="reflect")
        vals = resp[valid] if valid is not None else resp.ravel()
        if vals.size == 0:
            return None
        med = float(np.median(vals))
        sig = _MAD_TO_SIGMA * float(np.median(np.abs(vals - med)))
        scale = float(np.max(np.abs(filled))) if np.size(filled) else 0.0
        return sig if (math.isfinite(sig) and sig > 1e-9 * max(scale, 1e-30)) else None

    def estimate_threshold(self, img, params: DetectionParams, valid_mask=None, raw=None, measure=None,
                           with_auto: bool = True) -> dict:
        """Automatic threshold of the WHOLE masked image (before ``threshold_scale``), in NOISE
        UNITS: Big-FISH's elbow on (LoG response / noise_sigma), noise_sigma = 1.4826*MAD of the
        LoG response over valid pixels (computed on ``measure``, the unclipped corrected image,
        when given). An absolute LoG threshold is ``threshold * noise_sigma``.

        Returns {"auto": float|None, "base": float|None, "noise_sigma": float|None,
        "fallback": bool, "notes": [...]} where ``base`` is the Big-FISH value or, if unusable,
        NOISE_K. The pipeline caches this and passes it back as ``threshold_info`` so that
        preview windows and tiles use exactly the value (and sigma) of the full run."""
        valid = None if valid_mask is None else np.asarray(valid_mask, bool)
        out: dict = {"auto": None, "base": None, "noise_sigma": None, "fallback": False, "noise_regime": False,
                     "notes": []}
        if valid is not None and not valid.any():
            out["notes"].append("no valid pixels: nothing to detect")
            return out
        r = float(params.spot_radius_px)
        filled = self._prepare(img, valid, raw, r)
        src = filled if measure is None else self._prepare(np.asarray(measure, np.float32), valid, raw, r)
        sigma = self.noise_sigma(src, params, valid)
        out["noise_sigma"] = sigma
        if sigma is None:
            out["notes"].append("LoG response has zero noise (flat image?): automatic threshold undefined; "
                                "no spots reported")
            return out
        if not with_auto:  # manual threshold: only the noise unit is needed
            return out
        auto = None
        try:
            auto = self.auto_threshold(filled, params, valid, noise_sigma=sigma)
        except ValueError as exc:  # bigfish get_breaking_point on a short/monotone curve
            out["notes"].append(f"Big-FISH automatic threshold raised ({exc})")
        out["auto"] = auto
        if auto is not None and 0 < auto < NOISE_K:
            # the elbow of the spot-count curve lies where noise maxima still dominate (a local
            # maximum of pure LoG noise rarely exceeds ~4 sigma): there is no separable population
            # of real spots (empty/near-empty channel, or the curve has no kink)
            out["noise_regime"] = True
            out["notes"].append(
                f"WARNING automatic threshold ({auto:.3g} noise units) is in the noise regime (< {NOISE_K:g}): "
                "this channel may contain no (or very few) real spots; using "
                f"{NOISE_K:g} noise units instead. Confirm the channel visually or set a threshold manually.")
            out["fallback"] = True
            out["base"] = NOISE_K
        elif auto is None or auto <= 0:
            out["notes"].append(
                f"WARNING automatic threshold unusable ({'none' if auto is None else auto}); fell back to "
                f"{NOISE_K:g} noise units (robust sigma of the LoG response = {sigma:.4g}). "
                "Set a threshold manually if unsure.")
            out["fallback"] = True
            out["auto"] = None
            out["base"] = NOISE_K
        else:
            out["base"] = auto
        return out

    @staticmethod
    def _detect_coords(image: np.ndarray, thr: float, params: DetectionParams,
                       valid: Optional[np.ndarray] = None) -> np.ndarray:
        from bigfish.detection import spots_thresholding
        filt, lm = BigFishEngine._log_maxima(image, valid, params, extended=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "No spots were detected"
            sp, _ = spots_thresholding(filt, lm, float(thr), remove_duplicate=True)
        return np.asarray(sp, dtype=np.int64).reshape(-1, 2)

    # ------------------------------------------------------------ main
    def detect_with_info(self, img, params: DetectionParams, valid_mask=None, raw=None,
                         progress=None, cancelled=None, measure=None, threshold_info=None):
        """``threshold_info``: result of ``estimate_threshold`` computed on the full image;
        when given it is used instead of estimating from ``img`` (preview windows)."""
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
        info: dict = {"notes": [], "threshold": None, "auto_threshold": None, "n_tiles": 1,
                      "threshold_info": None}
        h, w = img.shape
        if valid is not None and not valid.any():
            info["notes"].append("no valid pixels: nothing to detect")
            return [], info

        r, md = float(params.spot_radius_px), float(params.minimum_distance_px)
        filled = self._prepare(img, valid, raw, r)
        progress(0.0, "threshold")
        if cancelled():
            raise DetectionCancelled()

        ti = threshold_info if threshold_info is not None else \
            self.estimate_threshold(img, params, valid, raw, measure, with_auto=params.threshold is None)
        info["threshold_info"] = ti
        sigma = ti.get("noise_sigma")
        info["noise_sigma"] = sigma
        if params.threshold is not None:
            thr = float(params.threshold)  # manual: noise units, same as the automatic value
            info["notes"].append("manual threshold used as given, in noise units (threshold_scale ignored)")
            if sigma is None:
                info["notes"] += list(ti.get("notes", []))
                return [], info
        else:
            info["notes"] += list(ti.get("notes", []))
            info["auto_threshold"] = ti.get("auto")
            info["fallback_threshold"] = bool(ti.get("fallback"))
            info["noise_regime"] = bool(ti.get("noise_regime"))
            if ti.get("base") is None or sigma is None:
                return [], info
            thr = float(ti["base"]) * float(params.threshold_scale)
        info["threshold"] = thr                       # noise units
        thr_abs = thr * float(sigma)                  # LoG-response units (what the spots are compared with)
        info["threshold_abs"] = thr_abs
        if cancelled():
            raise DetectionCancelled()

        # ---- coordinates (whole image or tiles)
        ts = int(params.tile_size)
        if ts <= 0 or ts >= max(h, w):
            progress(0.2, "detecting")
            coords = self._detect_coords(filled, thr_abs, params, valid)
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
                    vt = None if valid is None else valid[py0:py1, px0:px1]
                    c = self._detect_coords(np.ascontiguousarray(filled[py0:py1, px0:px1]), thr_abs, params, vt)
                    if c.size == 0:
                        continue
                    c = c + np.array([py0, px0])
                    keep = (c[:, 0] >= y0) & (c[:, 0] < y1) & (c[:, 1] >= x0) & (c[:, 1] < x1)
                    parts.append(c[keep])
            coords = np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))] if coords.size else coords

        if valid is not None and coords.size:  # (maxima already restricted to the mask +1 px)
            keep = ndi.binary_dilation(valid, structure=np.ones((3, 3), bool))
            coords = coords[keep[coords[:, 0], coords[:, 1]]]
        if cancelled():
            raise DetectionCancelled()

        # ---- saturated plateaus: one spot per connected saturated component
        overrides: dict = {}
        if raw is not None and len(coords):
            coords, overrides, pnote = self._merge_plateaus(coords, raw, valid, r)
            if pnote:
                info["notes"].append(pnote)

        # ---- optional dense decomposition (global: needs all spots)
        simulated = np.zeros(len(coords), bool)
        if params.dense_decomposition and len(coords) >= 2:
            progress(0.75, "dense decomposition")
            try:
                from bigfish.detection import decompose_dense
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # decompose_dense crashes on float images (np.iinfo on float dtype): requantise to uint16
                    q = np.clip(np.rint(filled), 0, 65535).astype(np.uint16)
                    new, regions, _ref = decompose_dense(q, coords, voxel_size=1, spot_radius=r)
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
                info["notes"].append("WARNING dense decomposition assumes identical gaussian spots; on the real test image it split bright single polonies into many spots")
                info["notes"].append(f"dense decomposition added {int(simulated.sum())} simulated spot(s) "
                                     f"in {len(regions)} dense region(s)")
            except Exception as exc:  # decomposition is best-effort
                info["notes"].append(f"dense decomposition failed ({type(exc).__name__}: {exc}); skipped")

        progress(0.85, "measuring")
        spots = self._measure(filled, valid, coords, simulated, params, raw, thr_abs, measure, overrides)
        spots = self._suppress(spots, params.remove_duplicate_radius_px)
        spots.sort(key=lambda s: (s.y, s.x))
        progress(1.0, "done")
        return spots, info

    # ------------------------------------------------------------ saturated plateaus
    @staticmethod
    def _merge_plateaus(coords: np.ndarray, raw: np.ndarray, valid: Optional[np.ndarray], r: float):
        """Merge all spots within a saturated component dilated by the spot radius into one.

        Returns (coords, overrides, note). ``overrides[(iy, ix)]`` carries the plateau
        centroid (x, y float), area, elongation and the oversize flag for the merged
        spot whose integer measurement position is (iy, ix)."""
        if raw.dtype.kind not in "ui" or len(coords) == 0:
            return coords, {}, None
        sat = raw >= np.iinfo(raw.dtype).max
        if valid is not None:
            sat &= valid
        if not sat.any():
            return coords, {}, None
        h, w = raw.shape
        lab, n = ndi.label(sat, structure=np.ones((3, 3), bool))
        rr = int(math.ceil(r)) + 1
        member: dict[int, list[int]] = {}
        for k, (y, x) in enumerate(coords):
            y0, y1, x0, x1 = max(0, y - rr), min(h, y + rr + 1), max(0, x - rr), min(w, x + rr + 1)
            win = lab[y0:y1, x0:x1]
            if not win.any():
                continue
            yy, xx = np.nonzero(win)
            d2 = (yy + y0 - y) ** 2 + (xx + x0 - x) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= r * r:
                member.setdefault(int(win[yy[j], xx[j]]), []).append(k)
        if not member:
            return coords, {}, None
        slices = ndi.find_objects(lab)
        drop = np.zeros(len(coords), bool)
        new_rows: list[list[int]] = []
        overrides: dict = {}
        n_merged = 0
        for l, ks in member.items():
            sl = slices[l - 1]
            yy, xx = np.nonzero(lab[sl] == l)
            yy = yy + sl[0].start
            xx = xx + sl[1].start
            area = int(yy.size)
            cy, cx = float(yy.mean()), float(xx.mean())
            cxx = float(((xx - cx) ** 2).mean()) + 1 / 12
            cyy = float(((yy - cy) ** 2).mean()) + 1 / 12
            cxy = float(((xx - cx) * (yy - cy)).mean())
            tr, det = cxx + cyy, cxx * cyy - cxy * cxy
            disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
            elong = float(math.sqrt((tr / 2 + disc) / max(tr / 2 - disc, 1e-9)))
            diam = 2.0 * math.sqrt(area / math.pi)
            iy, ix = int(min(max(round(cy), 0), h - 1)), int(min(max(round(cx), 0), w - 1))
            drop[ks] = True
            n_merged += len(ks)
            new_rows.append([iy, ix])
            # suspicious merge (kept as ONE spot, flagged for review): plateau much larger than a spot,
            # elongated plateau (touching polonies give a dumbbell/peanut), or >1 LoG maximum on it.
            # Splitting by distance-transform maxima is NOT attempted: clipped data carry no information
            # on how the true intensity divides, so the reviewer decides.
            suspicious = bool(diam > PLATEAU_SHAPE_FACTOR * max(r, 1.0) or elong > PLATEAU_ELONGATION_MAX
                              or len(ks) > 1)
            overrides[(iy, ix)] = dict(x=cx, y=cy, area=area, elongation=elong, oversize=suspicious)
        coords = np.concatenate([coords[~drop], np.array(new_rows, dtype=np.int64).reshape(-1, 2)])
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        note = (f"{n_merged} spot(s) on {len(member)} saturated plateau(s) merged into {len(member)} "
                "spot(s) at the plateau centroid (flagged saturated)")
        return coords, overrides, note

    # ------------------------------------------------------------ measurement
    def _measure(self, img, valid, coords, simulated, params, raw, thr, measure=None,
                 overrides=None) -> list[RawSpot]:
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
        sig_small, sig_large = (0.3 * r, 0.4 * r), (1.0 * r, 1.4 * r)
        K = int(4.0 * sig_large[1] + 0.5) + 2
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
            c3_max = float(np.nanmax(c3))
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
            # peak = measured image interpolated AT the refined centre (not the max of the
            # 3x3 block, which is biased upward by noise: at true A=0 it gives SNR ~ +1.5)
            peak = _interp_at(mwin, R + dy, R + dx, c3_max)
            snr = None
            if not math.isnan(bg) and std > 1e-9 * max(1.0, abs(peak)):
                snr = float((peak - bg) / std)
            # scale-selection ratio (debris filter, flag only): max over small sigmas of
            # s^2 * -LoG divided by the same over large sigmas, max in a 5x5 window.
            def _norm_max(sg):
                return float((-(sg * sg) * ndi.gaussian_laplace(crop, sigma=sg, mode="reflect"))
                             [K - 2:K + 3, K - 2:K + 3].max())
            small = max(_norm_max(sg) for sg in sig_small)
            large = max(_norm_max(sg) for sg in sig_large)
            ratio = float(small / max(large, 1e-6))
            # elongation of the half-max blob
            elong = None
            sub = win[R - he:R + he + 1, R - he:R + he + 1]
            bgz = 0.0 if math.isnan(bg) else bg
            d = np.nan_to_num(sub - bgz, nan=0.0)
            pk = c3_max - bgz
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
            sx_, sy_ = float(x + dx), float(y + dy)
            plateau_area, oversize = None, False
            ov = (overrides or {}).get((y, x))
            if ov is not None:  # merged saturated plateau: centroid, plateau shape
                sx_, sy_, sat = ov["x"], ov["y"], True
                elong, plateau_area, oversize = ov["elongation"], ov["area"], ov["oversize"]
                peak = c3_max  # clipped plateau: the interpolated value adds nothing
                snr = None  # clipped peak (a lower bound) and the annulus may lie inside the plateau: SNR undefined
            out.append(RawSpot(x=sx_, y=sy_, peak=peak, background=bg, snr=snr,
                               saturated=sat, log_response=resp, threshold=float(thr),
                               elongation=elong, scale_ratio=ratio, simulated=bool(sim),
                               plateau_area_px=plateau_area, plateau_oversize=oversize))
        return out

    @staticmethod
    def _suppress(spots: list[RawSpot], radius: float) -> list[RawSpot]:
        """Greedy: strongest LoG response first (ties by y,x); drop weaker spots
        closer than ``radius``. Deterministic."""
        if radius <= 0 or len(spots) < 2:
            return spots
        order = sorted(range(len(spots)), key=lambda i: (spots[i].plateau_area_px is None,
                                                          -(spots[i].log_response or 0.0), spots[i].y, spots[i].x))
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


def _interp_at(win: np.ndarray, fy: float, fx: float, fallback: float) -> float:
    """Bilinear value of ``win`` at the (float) position (fy, fx); NaN neighbours are
    skipped (weights renormalised); ``fallback`` if all four are NaN."""
    y0, x0 = int(math.floor(fy)), int(math.floor(fx))
    ty, tx = fy - y0, fx - x0
    num = den = 0.0
    for yy, wy in ((y0, 1 - ty), (y0 + 1, ty)):
        for xx, wx in ((x0, 1 - tx), (x0 + 1, tx)):
            if 0 <= yy < win.shape[0] and 0 <= xx < win.shape[1]:
                v = float(win[yy, xx])
                if not math.isnan(v) and wy * wx > 0:
                    num += wy * wx * v
                    den += wy * wx
    return num / den if den > 0 else fallback


def _parab(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0.0 if d == 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
