import sys, os, resource, tempfile, pathlib, numpy as np, tifffile, psutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from polony_app.ml import train as tr
from polony_app.ml.common import prepare_crop
mode, n, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root = pathlib.Path(tempfile.mkdtemp()); rng = np.random.default_rng(0)
labs = [np.zeros((0, 2), "float32")] * n
base = (rng.random((H, H)) * 2000 + 500).astype("uint16")
for i in range(n): tifffile.imwrite(root / f"c{i}.tif", base)
items = [(f"c{i}.tif", 400.0, 2500.0, (0, 0)) for i in range(n)]
p = psutil.Process(); r0 = p.memory_info().rss
if mode == "eager":
    xs = [prepare_crop(tifffile.imread(root / it[0]), l, lo=it[1], hi=it[2])[0] for it, l in zip(items, labs)]
else:
    seq = tr.LazyCropImages(str(root), items, [(H, H)] * n, 1, "valid_unsaturated", labs, 256 * 1024**2)
    for _ in seq: pass
    for i in rng.integers(0, n, 400): seq[int(i)]
print(f"RESULT {mode} n={n} {H}x{H} float32 data={n*H*H*4/2**20:.0f} MiB  RSS growth={(p.memory_info().rss - r0)/2**20:.0f} MiB")
