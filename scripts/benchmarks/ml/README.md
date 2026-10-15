# ML scale benchmarks (CPU, synthetic data)

Reproducible, small measurements behind the choices in `docs/app/LEGACY_FINETUNING.md` (no GPU numbers: none were measured).

* `membench.py MODE N H` (`MODE` = `eager|lazy`): resident-memory growth of N prepared HxH float32 crops loaded eagerly vs through the bounded
  `LazyCropImages` (256 MiB LRU). Example: `PYTHONPATH=src python scripts/benchmarks/ml/membench.py eager 1500 512`.
* `lazybench.py MODE NTRAIN NVAL CACHE_MB`: end-to-end tiny `Spotiflow.fit` (2 epochs, tiny U-Net) with eager vs lazy inputs: time, peak RSS and
  the number of crop reads.

Measured on a 4-core CPU with torch 2.14.1+cpu / Spotiflow 0.6.5 (400 train + 100 val crops of 128x128, tiny U-Net, 2 epochs):
eager 24.3 s / 603 MB; lazy (256 MB cache) 27.3 s / 607 MB; 1,500 crops of 512x512 float32 (1,500 MiB): eager RSS +1,502 MiB, lazy +259 MiB.
At small data the process memory is dominated by the torch/Spotiflow stack, so lazy loading only matters for large crop counts; it is chosen automatically
above 8 GiB (or 25 % of available RAM) of prepared data (`--load-mode`).
