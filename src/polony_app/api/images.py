"""Image access + display rendering for the API (display never feeds detection)."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

from polony_app.review.state import AnalysisState, ReviewError
from polony_app.schemas import Analysis, ChannelRole, ImageKind

VIEWS = ("red", "green", "composite", "processed_red", "processed_green")


class _LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get_or_make(self, key, make: Callable[[], object]):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
        val = make()
        with self.lock:
            self.d[key] = val
            self.d.move_to_end(key)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
        return val

    def drop_prefix(self, prefix: str) -> None:
        with self.lock:
            for k in [k for k in self.d if k[0] == prefix]:
                del self.d[k]


def role_index(a: Analysis, role: str) -> int:
    if a.image is None:
        raise ReviewError("no_image", "analysis has no image", 409)
    for c in a.image.channels:
        if c.role is not None and c.role.value == role and c.role_confirmed:
            return c.index
    raise ReviewError(
        "channel_mapping_unconfirmed",
        f"no confirmed '{role}' channel: confirm the channel mapping first (PUT /channels)",
        409,
    )


class ImageService:
    def __init__(self, store, cap: int = 6):
        self.store = store
        self._raw = _LRU(cap)
        self._proc = _LRU(cap)
        self._stats = _LRU(16)

    def forget(self, aid: str) -> None:
        for c in (self._raw, self._proc, self._stats):
            c.drop_prefix(aid)

    # -- arrays
    def raw_key(self, st: AnalysisState, role: str):
        a = st.analysis
        dy, dx = int(st.alignment.get("dy", 0)), int(st.alignment.get("dx", 0))
        return (a.id, "raw", role, role_index(a, role), (dy, dx) if role == "green" else (0, 0))

    def proc_key(self, st: AnalysisState, role: str):
        a = st.analysis
        params = (a.detection_request.red if role == "red" else a.detection_request.green).background
        return (a.id, "proc", role, params.model_dump_json(),
                tuple(int(st.alignment.get(k, 0)) for k in ("dy", "dx")))

    def raw(self, st: AnalysisState, role: str) -> np.ndarray:
        """Channel array in original dtype; green gets the user-set integer shift."""
        from polony_app.imaging import io as imio
        from polony_app.imaging import registration

        a = st.analysis
        idx = role_index(a, role)
        key = self.raw_key(st, role)
        shift = key[4]

        def make():
            arr = imio.load_channel(a.image, self.store.originals_dir(a.id), idx)
            if shift != (0, 0):
                arr = registration.apply_shift(arr, *shift)
            return arr

        return self._raw.get_or_make(key, make)

    def raw_pair(self, st: AnalysisState) -> tuple[np.ndarray, np.ndarray]:
        return self.raw(st, "red"), self.raw(st, "green")

    def processed(self, st: AnalysisState, role: str) -> np.ndarray:
        from polony_app.imaging.preprocess import correct_background

        a = st.analysis
        params = (a.detection_request.red if role == "red" else a.detection_request.green).background
        key = self.proc_key(st, role)
        return self._proc.get_or_make(
            key, lambda: np.asarray(correct_background(self.raw(st, role), params), dtype=np.float32)
        )

    def auto_range(self, key, arr: np.ndarray) -> tuple[float, float]:
        from polony_app.imaging.render import percentile_range

        return self._stats.get_or_make(key, lambda: percentile_range(arr))

    # -- rendering
    def render(self, st: AnalysisState, view: str, lo, hi, gamma: float, crop, max_dim,
               colormap: str = "gray") -> tuple[bytes, tuple[float, float]]:
        from polony_app.imaging import render as R
        from polony_app.imaging.io import load_rgb

        a = st.analysis
        if view not in VIEWS:
            raise ReviewError("invalid_view", f"view must be one of {', '.join(VIEWS)}", 422)
        if gamma <= 0:
            raise ReviewError("invalid_param", "gamma must be > 0", 422)
        if view == "composite":
            if a.image.kind == ImageKind.rgb_composite:
                rgb = load_rgb(a.image, self.store.originals_dir(a.id))
                png = R.render_rgb_png(rgb, lo, hi, gamma, crop, max_dim)
                return png, (0.0, float(np.iinfo(rgb.dtype).max) if rgb.dtype.kind in "ui" else 1.0)
            red, green = self.raw(st, "red"), self.raw(st, "green")
            kr = self.raw_key(st, "red")
            kg = self.raw_key(st, "green")
            rr = (lo, hi) if lo is not None and hi is not None else self.auto_range(kr, red)
            gr = (lo, hi) if lo is not None and hi is not None else self.auto_range(kg, green)
            return R.render_composite_png(red, green, rr, gr, gamma, crop, max_dim), rr
        role = "red" if view.endswith("red") else "green"
        if view.startswith("processed"):
            arr = self.processed(st, role)
        else:
            arr = self.raw(st, role)
        auto = self.auto_range(
            self.proc_key(st, role) if view.startswith("processed") else self.raw_key(st, role), arr
        )
        l = auto[0] if lo is None else lo
        h = auto[1] if hi is None else hi
        return R.render_png(arr, l, h, gamma, crop, max_dim, colormap), auto  # type: ignore[arg-type]

    def histogram(self, st: AnalysisState, role: str, bins: int = 256) -> dict:
        arr = self.raw(st, role)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
        return {
            "bins": [float(e) for e in edges[:-1]],
            "bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
            "lo": lo,
            "hi": hi,
            "p995": float(np.percentile(arr, 99.5)),
        }
