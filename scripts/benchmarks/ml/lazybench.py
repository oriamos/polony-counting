"""Eager vs lazy (polony_app.ml.train.LazyCropImages) end-to-end Spotiflow.fit on a synthetic crop set. CPU, tiny U-Net.
usage: lazybench.py MODE(eager|lazy) NTRAIN NVAL CACHE_MB"""
import sys, os, time, resource, tempfile, pathlib, numpy as np, tifffile, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from polony_app.ml import train as tr
from polony_app.ml.common import prepare_crop
from spotiflow.model import Spotiflow, SpotiflowModelConfig
import pytorch_lightning as pl
mode, ntr, nva, cache = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
root = pathlib.Path(tempfile.mkdtemp()); rng = np.random.default_rng(0)
H = 256; items, labs = [], []
for i in range(ntr + nva):
    im = (rng.random((H, H)) * 2000 + 500).astype("uint16")
    pts = rng.integers(8, H - 8, (20, 2)).astype("float32")
    for y, x in pts: im[int(y)-1:int(y)+2, int(x)-1:int(x)+2] += 20000
    tifffile.imwrite(root / f"c{i}.tif", im); items.append((f"c{i}.tif", 400.0, 22000.0, (0, 0))); labs.append(pts)
d = 2
def lazy(sl):
    sub = items[sl]
    shapes = [(H // d, H // d)] * len(sub)
    return tr.LazyCropImages(str(root), sub, shapes, d, "valid_unsaturated", labs[sl], cache * 1024**2)
def eager(sl):
    xs = []
    for it, lab in zip(items[sl], labs[sl]):
        x, l, _, _ = prepare_crop(tifffile.imread(root / it[0]), lab, lo=it[1], hi=it[2], downscale=d, origin_xy=it[3]); xs.append(x)
    return xs
sl_tr, sl_va = slice(0, ntr), slice(ntr, ntr + nva)
def spots(sl):
    return [prepare_crop(np.zeros((H, H), "uint16"), l, lo=0., hi=1., downscale=d)[1] for l in labs[sl]]
pl.seed_everything(0, workers=True)
tri, vai = (lazy(sl_tr), lazy(sl_va)) if mode == "lazy" else (eager(sl_tr), eager(sl_va))
m = Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=3, sigma=1.0))
extra = {"defer_normalization": True} if mode == "lazy" else {}
t = time.time()
m.fit(tri, spots(sl_tr), vai, spots(sl_va), save_dir=root / "w", train_config={"num_epochs": 2, "batch_size": 4, "crop_size": 64, "optimize_threshold": False},
      device="cpu", logger=pl.loggers.CSVLogger(str(root / "l")), deterministic=True, normalizer=None, **extra)
dt = time.time() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if mode == "lazy":
    print(f"RESULT lazy ntr={ntr} nva={nva} cache={cache}MB fit={dt:.1f}s peakRSS={rss:.0f}MB train loads={tri.loads} hits={tri.hits} | val loads={vai.loads} hits={vai.hits}  (2 epochs: {2*ntr} train + {2*nva} val accesses + sanity)")
else:
    print(f"RESULT eager ntr={ntr} nva={nva} fit={dt:.1f}s peakRSS={rss:.0f}MB")
