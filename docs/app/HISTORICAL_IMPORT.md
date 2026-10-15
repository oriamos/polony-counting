# Importing the lab's historical annotations (`polony_app.legacy_import`)

File layouts and conventions: [DATA_FORMATS.md](DATA_FORMATS.md). Requires `pip install -r requirements-app.txt` (incl. `imagecodecs`).
Run from the repository root with `PYTHONPATH=src`.

## 1. Preview (nothing is written)

```bash
PYTHONPATH=src python -m polony_app.legacy_import inspect  /path/to/folder_or_file.tif
PYTHONPATH=src python -m polony_app.legacy_import import   /path/to/folder --out data/historical/2026-09-29      # dry run: no --commit
```
The report lists files discovered, slides reconstructed, images loaded/failed, ROI files converted, labels per plane, grid fields, regions
(excluded regions only if named), pairing problems, signal evidence and every warning. Exit code 1/2 signals errors.
`*.tif.zip` archives are reported, never extracted: extract them to their own directory first.

Useful options: `--class-map 1=<label>` (explicit; default: no class), `--excluded-name NAME` / `--valid-name NAME`, `--reserve 'GLOB_OR_SHA=test'`,
`--duplicate-tol 1.0`, `--created-at 2026-10-10T00:00:00Z` (reproducible manifest), `--json`.

## 2. Real import

```bash
PYTHONPATH=src python -m polony_app.legacy_import import /path/to/folder --out data/historical/2026-09-29 --commit
```
Refuses to overwrite an existing dataset, and refuses to commit when files failed (unless `--allow-errors`). Output = canonical dataset
(schema `polony-historical-dataset` v1; see [DATA_FORMATS.md §7](DATA_FORMATS.md#7-canonical-historical-dataset-import-output)). Images are referenced by path + sha256, **not copied**.

## 3. Splits (leakage-safe)

```bash
PYTHONPATH=src python -m polony_app.legacy_import split data/historical/2026-09-29 --seed 0 --dry-run
```
Exit code 3 = identity is inferred from file names only; get the researcher's confirmation and pass
`--confirmed-slides confirmed.json` (`{"confirmed_slides": ["AM_A", "AM_B"], "confirmed_by": "..."}`), or consciously use `--allow-unconfirmed-identity`.
Writes `<dataset>/splits/<name>.json`. Images under a `test` directory (e.g. `test_img.tif`) are pinned to `test`.

## 4. Consumers

```python
from polony_app.legacy_import import read_dataset
from polony_app.legacy_import.exports import score_detections, export_spotiflow, legacy_squares, to_review_seed
```
* Big-FISH evaluation: `score_detections(ds, image_id, page, detections_xy, tolerance_px)` - scores only inside counted fields.
* Spotiflow fine-tuning data: `python -m polony_app.legacy_import export-spotiflow DS --splits-file DS/splits/default.json --pages 0 --out OUT`
  (explicit planes; `y,x` CSV + counted-field polygons; `test` is not exported unless requested).
* Fixed-size grid squares: `legacy_squares(ds, image_id, page=0)`.
* Review sessions: `to_review_seed(ds, image_id, page)` -> regions + unclassified manual objects.

Train only inside the counted fields; the rest of the image is unlabelled, not background.

## 5. Training on the historical annotations

For Spotiflow fine-tuning use `python -m polony_app.ml import-legacy` (it reads the TIFFs directly, keeps only fields the researcher examined per page, and builds a snapshot for the ML lifecycle): [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md). `export-spotiflow` above does not provide snapshots, the locked test set or the examined-field evidence.
