# Fine-tuning Spotiflow from reviewed corrections

Everything is **explicit**. The app never trains, evaluates or switches a model by itself. Code: `src/polony_app/ml/`,
CLI: `python -m polony_app.ml <command>` (`--data-dir` defaults to the app's data directory / `POLONY_DATA_DIR`).
Neither engine is ground truth; the *human-reviewed final points* are the only labels used, and they carry the reviewer's biases (see Limitations).

```
reviewed analyses ──build──► immutable dataset snapshot ──split/--lock-test──► locked test set
                                   │                                              │
                                   └──train (candidate)──► registry ──eval-model──┤
                                                              └── compare ── promote / rollback (explicit)
```

## 1. Collect corrections
Review analyses as usual. A region counts as **fully reviewed** only when a person approved it and nothing changed since (the
approval fingerprint still matches). Colour/page is used only from a **confirmed channel mapping**; without it an analysis yields no crops.

## 2. Build a snapshot — `build`
```
python -m polony_app.ml build --analyses <id> <id> ... [--seed 0] [--fractions 0.7,0.15,0.15] [--identity-map ids.json]
                              [--allow-unconfirmed-identity] [--allow-bbox] [--strict-reviewed] [--require-finalized] [--dry-run]
```
Writes `<data_dir>/datasets/<version>/` (read-only, never overwritten; `version = ds-` + hash of the manifest content, which lists the
sha256 of every file — identical inputs give an identical version and byte-identical files; the creation time and app version are
recorded but not hashed):

| file | content |
|---|---|
| `manifest.json` | version, created, app version, options/seed, source analyses (+ TIFF sha256, review status, **reviewer identity** from the action log else `unknown` with a warning, engine/model versions of the source runs, channel mapping **only if confirmed else null**, alignment), crops (box in the stored page and in the reference frame, counts, sha256), skipped crops, label counts, split assignment, test-lock reference, warnings, **unresolved assumptions** |
| `points.csv` | every point with `status`: `confirmed`, `human_added`, `rejected`, `unresolved`, `unreviewed`, `raw_proposal` (+ engine, model score, consensus status, page, role, split, crop id). Coordinates: x = column, y = row, pixel centres at integer coordinates |
| `regions.jsonl` | every region (counting/exclusion, polygon, approved, fully reviewed, rectangular, crop or skip reason per page) |
| `spotiflow/{train,val,test}/<id>.tif` + `<id>.csv` | crops cut from the **original page** (original dtype, no 8-bit conversion, no rescaling, single page, uncompressed) and their labels (columns `y,x`, crop coordinates) |

Rules:
* **Positives** = objects a human confirmed, moved, reclassified or added. In a region the human *approved*, untouched proposals are taken as confirmed
  (approval attests that the remaining proposals are the final set; basis `region_approved`). `--strict-reviewed` turns this off: such points become
  `unreviewed` and the crop is skipped. Rejected, unresolved and unreviewed points are **never** positives. Raw engine proposals are listed, never positives.
* **Crops** only from rectangular, fully reviewed regions (fields). A crop is skipped (with a warning) if it overlaps an exclusion region, contains an
  unresolved point (unlabelled real spots would teach the model that spots are background), is smaller than 32 px, or leaves the stored image.
  Non-rectangular regions are skipped unless `--allow-bbox` (the bounding box is cut; spots outside the polygon are unlabelled — warned).
* Green is aligned to red by an integer shift: its crop box and points are mapped back to the **stored page** (`box_xyxy` vs `box_reference_frame`).
* The tool uses the app store read-only (`open_store`). Raw engine proposals and engine/model versions need `get_engine_runs/get_raw_detections`; if the store
  lacks them the snapshot says so in its warnings.

## 3. Splits and the locked test set — `split`
Crops inherit the split of their **analysis**; analyses are grouped specimen → slide → source file with `legacy_import.splits` (replicates of one specimen
stay together; byte-identical files tie groups). Identity comes from, in order: `--identity-map` (confirmed), the slide id typed in the app (`<sample>_<replicate>`,
specimen derived by that lab convention — listed as an assumption), or the file name (**inferred, refused** unless `--allow-unconfirmed-identity`, which is recorded).
Deterministic for a seed. With too few independent groups a split stays empty (never filled from the same slide) and the snapshot says so.

`python -m polony_app.ml split --snapshot <v>` prints the group → split table. `... --lock-test` writes `<data_dir>/datasets/test_lock.json`
(test group ids, the specimen/slide/file keys, the test snapshot version and the hash of its test files). From then on:
* every later `build` puts analyses of a locked specimen/slide or with identical file bytes into **test only**, does **not** re-export them from the live analysis
  and copies the locked test crops verbatim from the locked snapshot (hash-checked) — new corrections never change test labels;
* no new test groups are created (the test fraction moves to train/val);
* `train` **refuses** any snapshot whose train/val crops belong to a locked group or file;
* re-locking needs `--replace-lock --reason "..."` and is logged.

## 4. Train a candidate — `train`
```
python -m polony_app.ml train --snapshot <v> --base general|<registry id>|<model folder> [--epochs 30] [--lr 1e-4] [--batch-size 4]
                              [--crop-size 256] [--seed 0] [--device cpu|auto|cuda|mps] [--dry-run]
```
`--dry-run` validates snapshot integrity, the lock, train/val crops, hyper-parameters and the base model, then stops. Training is refused without valid train **and** validation crops.
Fine-tuning continues from the base weights: `Spotiflow.from_pretrained/from_folder(..., inference_mode=False)` then `fit(...)` (what `spotiflow-train --finetune-from` does).
Official pretrained names are downloaded by Spotiflow on first use. The result goes to `<data_dir>/models/<id>/` and the registry with status **candidate**: base model,
dataset version, split membership (crop ids and groups), hyper-parameters, seed, duration, hardware, validation losses, weights sha256. The app does not use it.
Defaults (CPU, lr 1e-4, 30 epochs) are starting points, not tuned values.

## 5. Evaluate on the locked test set — `eval-model`, `compare`
`eval-model --model <id|general>` runs the model on the locked test crops and stores precision/recall/F1/matching distance in the registry, tied to the lock hash. Evaluate the
candidate **and** the model it would replace (the active one, else `general`). `compare --models A B` shows them side by side. Results are labelled `preliminary` when thin (see [EVALUATION.md](EVALUATION.md)).

## 6. Promote / roll back — explicit
`promote <id>` requires evaluations of the candidate and of the model it replaces on the **current** lock, and refuses a candidate with lower F1. `--force --reason "..."` overrides and is recorded.
`rollback` restores the previously active model (or the app default). The app resolves `SpotiflowParams.model_name` through `polony_app.ml.registry.resolve_model(name, data_dir)`:
a registry id → its weights folder (sha256 re-checked), `"active"` → the promoted model, anything else → an official pretrained name.

## Hardware
CPU is the default and enough for small corrections sets (the opt-in test fine-tunes a tiny model for 2 epochs in about a second). Real pretrained-size models on many crops are slower on CPU; `--device auto` uses CUDA/MPS when present.
Hardware and library versions are recorded per model.

## Limitations
* **One annotator / one session** is the usual case: the labels reflect that person's choices and fatigue; model gains can be bias learnt, not accuracy. Annotator count is read from the action log; if absent it is `unknown`.
* **Unlabelled areas**: only fully reviewed fields are cropped, but a human can still miss real spots inside them; those become false "background".
* Approved regions with untouched proposals are accepted as confirmed (`--strict-reviewed` avoids this at the price of fewer crops).
* Spotiflow normalises each image by percentiles; crops are normalised per crop while detection normalises the whole page — a small train/inference mismatch.
* Few groups ⇒ small, noisy test set; metrics are `preliminary`. Nothing here proves a fine-tuned model generalises to new scanners or samples.
* Reference marks of the researcher (ROI) are not used as training labels here; only app-reviewed points are.
* **Never auto-train**: there is no scheduler, hook or background job. A person runs each step.
