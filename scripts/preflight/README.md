# Preflight rehearsal for the historical-TIFF fine-tuning workflow

Dress rehearsal of `docs/app/LEGACY_FINETUNING.md` at the scale of the real job (~1,500 files) on **synthetic** data with an
**independent oracle**, so every guarantee is verified before the operator runs the real directory. Nothing here reads or writes
real data, and nothing is ever written into the repository (generated TIFFs/snapshots live under `--out`).

* `make_synthetic.py` - deterministic (seeded) generator of N two-page uint16 ImageJ TIFFs shaped like the lab's scans (ROI point blob with
  counters and positions, composite line-grid overlay of 4x7 cells, clipped scanner background on page 1) at a reduced pixel size
  (`--cell-px`, default 64; `--real-size` = 2438x1460 with 316.23 px cells), in dated sub-directories, ~N/2 specimens with 1-3 replicates, plus
  injected pathologies with exact counts (byte-identical copies, missing ROI/overlay, corrupt ROI bytes, 8-bit, float32, single page, RGB
  composite, two ROI blobs, no counters, files under dot directories, unicode/space names, non-TIFF files, a zip, a symlinked directory,
  re-scans with the same base name). Per file random but recorded: origin shift, marks per field, counters (absent / out of order), straddling
  clicks, boundary clicks, out-of-grid / out-of-bounds points, positions 0/3, page-1 marks, duplicates and near duplicates, integer (non-sub-pixel)
  ROIs, files without a chamber. It writes `oracle.json` computed ONLY from the generator's parameters (the rules of the guide re-stated in
  `derive()`; never from `polony_app.ml.legacy_bridge`), confirmed `mapping/channels.json` + `identities.json` + `examined-fields.json`
  (verified zeros, deliberate conflicts, unresolved entries) and `mapping/errors/*` (deliberately broken mapping files).
* `rehearsal.py` - drives the real CLI (`python -m polony_app.ml ...`, one subprocess per command) through write-templates, refusals, dry-run,
  import with workers 1 and N, verify (+ tampering), split, lock-test, an import of extra data after the lock, a real `kill -9` + resume,
  disk-full, `train --dry-run`, a tiny CPU fine-tune on a small separate snapshot, eval-model, compare, and a real-size calibration batch. It
  writes `REHEARSAL.md` and `rehearsal.json` (each check PASS / FAIL / KNOWN_FAIL / SKIP with evidence, per-command wall time and peak RSS,
  disk, crops/s, projection for the real directory) into `--out`; exit code 1 if any check FAILs.

## Run

```bash
# full rehearsal (about 25-40 min on 4 cores, ~6 GB of disk with --keep):
PYTHONPATH=src python3 scripts/preflight/rehearsal.py --n 1500 --cell-px 64 --workers 4 --out /big/disk/rehearsal --keep
# fast smoke (about 1 minute), also run by the test suite:
python -m pytest tests/app/test_preflight_rehearsal.py -q
PYTHONPATH=src python3 scripts/preflight/rehearsal.py --n 24 --out /tmp/rh24 --fast --no-model --no-probe --calibration 0 --workers 2
# only the generator:
PYTHONPATH=src python3 scripts/preflight/make_synthetic.py --out /tmp/synth --n 1500 --cell-px 64 --seed 1 --workers 4
```

Options: `--no-model` (skip the fine-tune stage; it needs torch/spotiflow and uses a tiny random base model and a random stand-in for `general`
through a subprocess wrapper, no download), `--calibration K` (K real-size files, 0 = skip), `--project-files 1500`, `--pixel-files K` (limit the
bit-exact comparison to K files), `--fast`, `--keep`.

## What is injected into the subprocesses (only through `_cli_wrapper.py`, never in the product)
`REHEARSAL_DISK_FREE` (patches `shutil.disk_usage`), `REHEARSAL_SLOW_WRITE_MS` (so the `kill -9` has a window), `REHEARSAL_ENOSPC_AFTER`
(an `OSError(ENOSPC)` after k crop writes), `REHEARSAL_GENERAL_STAND_IN` (redirects `load_model('general')` offline).

## Not covered
Real scanner data and real annotator behaviour; the real GPU (ROCm) speed; filesystems with different semantics (network shares, case-insensitive
names); concurrent writers on one data directory; a torn power-off (only `kill -9` is simulated); the pretrained `general` model quality (a random
stand-in is evaluated, so F1 values are meaningless - the mechanics are what is checked).
