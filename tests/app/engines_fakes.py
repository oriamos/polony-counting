"""Fake spotiflow / torch modules for the normal test suite (no model download, no GPU)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def local_max_points(img: np.ndarray, rel_thresh: float = 800.0):
    """(y, x) int points of bright blobs on the given array + a fake 'probability' in (0, 1)."""
    a = np.asarray(img, np.float64)
    sm = ndi.gaussian_filter(a, 2.0)
    bgd = ndi.uniform_filter(sm, 41)
    d = sm - bgd
    pk = (d == ndi.maximum_filter(d, size=9)) & (d > rel_thresh)
    ys, xs = np.nonzero(pk)
    prob = np.clip(d[ys, xs] / (d[ys, xs] + 1000.0), 0.01, 0.99)
    return np.stack([ys, xs], 1).astype(float), prob


class FakeModel:
    _prob_thresh = [0.5]

    def __init__(self, owner):
        self.owner = owner

    def predict(self, img, **kw):
        o = self.owner
        o.calls.append({"dtype": img.dtype, "shape": img.shape, "values": np.array(img, copy=True), "kw": dict(kw)})
        if o.fail_predict is not None:
            raise o.fail_predict
        if o.points_fn is not None:
            pts, prob = o.points_fn(img)
        else:
            pts, prob = local_max_points(img)
        wrapper = kw.get("progress_bar_wrapper")
        if wrapper is not None and o.n_tiles_fake:
            for _ in wrapper(iter(range(o.n_tiles_fake)), total=o.n_tiles_fake):
                pass
        return np.asarray(pts, float).reshape(-1, 2), types.SimpleNamespace(prob=np.asarray(prob, float))


class FakeSpotiflow:
    """Installs fake modules into sys.modules (via monkeypatch) and records every model.predict call."""

    def __init__(self, monkeypatch, tmp_path: Path, points_fn=None, fail_load=None, fail_predict=None,
                 cuda=False, mps=False):
        self.calls: list[dict] = []
        self.loads: list[dict] = []
        self.points_fn = points_fn
        self.fail_predict = fail_predict
        self.fail_load = fail_load
        self.n_tiles_fake = 0
        self.cache_dir = tmp_path / "models"
        model_dir = self.cache_dir / "general"
        model_dir.mkdir(parents=True)
        (model_dir / "config.yaml").write_text("fake: true\n")
        (model_dir / "best.pt").write_bytes(b"fake-weights-best")
        (model_dir / "last.pt").write_bytes(b"fake-weights-last")
        owner = self

        class Spotiflow:
            @classmethod
            def from_folder(cls, path, map_location="cpu", **kw):
                owner.loads.append({"path": str(path), "device": map_location})
                if owner.fail_load is not None:
                    raise owner.fail_load
                return FakeModel(owner)

        mod = types.ModuleType("spotiflow")
        mod.__version__ = "9.9.9-fake"
        model = types.ModuleType("spotiflow.model")
        model.Spotiflow = Spotiflow
        pre = types.ModuleType("spotiflow.model.pretrained")
        pre.list_registered = lambda: ["general", "hybiss"]
        pre._REGISTERED = {}
        pre.get_pretrained_model_path = lambda name, cache_dir=None: Path(cache_dir) / name
        mod.model = model
        model.pretrained = pre
        torch = types.ModuleType("torch")
        torch.__version__ = "0.0-fake"
        torch.Tensor = type("Tensor", (), {})      # scipy / sklearn probe sys.modules["torch"].Tensor lazily
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        for n, m in (("spotiflow", mod), ("spotiflow.model", model), ("spotiflow.model.pretrained", pre),
                     ("torch", torch)):
            monkeypatch.setitem(sys.modules, n, m)
        from polony_app.engines import spotiflow_adapter
        spotiflow_adapter.clear_model_cache()
        spotiflow_adapter.reset_device_fallback()

    def engine(self):
        from polony_app.engines import SpotiflowEngine
        return SpotiflowEngine(cache_dir=self.cache_dir)
