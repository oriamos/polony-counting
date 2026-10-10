# Legacy repository audit (kirill-push/polony-counting)

Status: forensic audit + integration record. Every statement is tagged **[confirmed]** (read in code or
reproduced on real files in this session), **[inferred]** (consistent with evidence, not proven) or
**[open]** (needs the researcher). Method: all code paths were read; the legacy ROI/TIFF behaviour was
re-derived by running `tifffile`/`roifile` on the real files; nothing was trained.

## 0. Scope, premises that did not hold, and limits

* **The "old" and the "current" repositories share one `main`.** `oriamos/polony-counting@main` is byte-identical
  to `kirill-push/polony-counting@main` (`65cb245`, v0.6.5, 2024-01-31; `diff -rq` empty, 263 commits). The "current system"
  (Big-FISH engine, review app) lives on the branch `claude/polony-hitl-app` (`src/polony_app`). No Spotiflow engine exists
  in either (`grep -ri spotiflow` is empty): the Spotiflow work here is the *dataset export* only. This work is
  built on `claude/polony-hitl-app` (branch `claude/legacy-import`); `main` was not touched.
* The legacy datasets (four Google-Drive zips, ids in `config.yaml`) and the W&B runs are **not available**; only
  `resources/dataset_files/test/test_img.tif`, the two checkpoints and the 2026 research samples are.
* **Research samples** exist on the branch `data/research-samples-2026-09-29` (8 zipped TIFFs + 2 CSVs, committed
  by the repository owner). They were extracted in a scratch directory and are *not* added to this branch.
* **Legacy GitHub issues**: the API is not reachable for the legacy repo from this session; the public page lists
  52 issues, only the first 25 *titles* were read (#32-#88: config defaults, docstrings, classifier, CSV output, UI
  for testing, "more augmentations"). No issue describes a scientific data problem. [partial]
* `docs/app/LEGACY_AUDIT.md` (earlier work) already audits the *prediction* path (8 findings, e.g. 28 vs 24 fields,
  5 uL hard-code). This document does not repeat them; it covers data/ROI/TIFF/dataset construction.
* Baseline test status before any change: `PYTHONPATH=src pytest tests/app` = 220 passed, **14 failed**, 2 skipped. The 14
  failures are all lab-export tests (`formula cell F6 not found in T7B`, openpyxl/xlsx) and are unrelated to this work (§H).

## A. File and data inventory

| Format | Purpose | Reader / writer | Important assumptions | Reusable? |
|---|---|---|---|---|
| ImageJ 2-page `uint16` TIFF (`ZYX`, `slices=2`) with embedded `ROI` + `Overlays` | The scanner image **and** Elena's annotation in one file | legacy: `PIL` (`read_tiff`) + `roifile.roiread(path)[1]`/`[0]`; new: `tifffile` + `roifile` | 2 equal-size planes; ROI blob = points, Overlays blob = grid; legacy relies on list order `[0]`=grid, `[1]`=points | Behaviour only. Rewritten (`legacy_import/roi.py`) without index assumptions |
| `uint8` RGB TIFF, LZW (`…_(635_532).tif`) | Rendered composite of the same slide | not read by legacy code (README mentions it) | name suffix = two wavelengths; **needs `imagecodecs`** (LZW) | Yes, as pairing input; evidence below |
| Standalone `.roi` / `.zip` RoiSet | Not observed in the data | `roifile.roiread` | - | Supported by the new reader (sidecar, same stem), untested on real files |
| Lab CSV `T7A_15-09-2026.csv`, `T7B_…csv` (24 rows, `field 1…24`) | Per-field counts + abundance | legacy `save_predictions_to_csv` (writer); `pd.read_csv` (efficacy); new `store/csv_import.py` | blank = not counted, 0 = counted zero; header has typos kept verbatim | `csv_import.py` reused as is |
| `Counts_30-09-2026.xlsx`, T7*.xlsx | Lab summary workbooks | openpyxl (new export only) | see `docs/app/LAB_EXPORT.md` | not part of the import |
| HDF5 `polony/{train,valid}.h5` (`images`, `labels`, `n_points`, `path`, `class`) | Legacy training squares | `make_dataset.py` (writer), `PolonyDataset` | 316x316 squares, 2 planes, density label x100 | Do not reuse (§F) |
| `path_dict.json` | square-index -> source path | `make_dataset.py` | contains the author's Windows temp paths, 5 entries (`1-33-A.tif`…) | Reference only: shows the old naming `<batch>-<n>-<A/B>.tif` under `polony/<zip>/slides/` |
| `*.pth` (`unet_49_1.7496`, `classifier_57_0.8896`, `test`) | Weights | `torch.save(state_dict)` | DataParallel (`module.` prefix); `test.pth` is byte-identical to `unet_49…` (sha256 `9ed3a3cd…`) | Not copied; see §G |
| `config.yaml` / `config_kaggle.yaml` | Parameters | PyYAML | `img_size [2450,1438]`, grid origin 158, `number_of_lines [8,5]`, `mean/std` | Values are facts about *one* scan layout, not defaults |
| **New canonical dataset** (this work) | Versioned labels + provenance | `legacy_import/schema.py` | §D | - |

## B. TIFF conventions

Verified on `test_img.tif` (2023 scan) and the seven 2026 research slides. [confirmed] unless marked.

* **Structure.** ImageJ-written TIFF, 2 IFDs, `uint16`, uncompressed, `photometric=minisblack`. `tifffile` reports
  `series[0].shape == (2, H, W)`, `axes == 'ZYX'` and `imagej_metadata['slices'] == 2` - the two planes are
  stored as **z-slices, not as ImageJ channels**. Sizes: `test_img` 2450 x 1438 (H x W); research slides 2438 x 1460.
  The legacy config hard-codes `[2450, 1438]` and so does not describe the 2026 files.
* **Plane/channel interpretation.** The only metadata label is `ImageDescription: 635 [W1]` (ImageJ `Info`), attached
  to the stack once; page 1 has no label. 635 nm is the red laser line, but this is a hint, not a role.
  Evidence for **page 0 = 635 / red**: (a) in the 2026 composite `AS-B_(635_532).tif`, the composite R plane correlates with
  page 0 (Spearman 0.996) and the G plane with page 1 (0.999), with zero spatial shift (FFT cross-correlation peak at (0,0));
  (b) points placed on ImageJ position 1 sit on page-0 signal (median 3018 vs 615 on page 1 in `AI-B`; 9x in `test_img`).
  Legacy code names page 0 "RGB" (`img = imgs[0]`, `# load an RGB image`) - **wrong** for these files: there is no RGB
  page in the stack. The composite is a *separate* file. The importer records the correlations in
  `composite_evidence` but never writes a colour role; the app's rule stays "user confirms the mapping".
* **Red/green/yellow.** Not encoded in any file. Colours come only from the composite's channel names `(635_532)` and
  the workbook header "Cy5 red & Cy3 green (yellow counts)" [inferred]. No file marks a point as yellow.
* **RGB / red / green files for one slide.** The README promises "one RGB + two B/W images per slide". Reality in the data:
  one 2-plane stack **plus optionally** one RGB composite named `<stack stem>_(635_532).tif`. No separate red/green
  single-plane files were observed. Pairing rule implemented: same directory, same base stem after removing
  `_(ddd_ddd)`, identical `H x W`; anything else is reported, never guessed.
* **Orientation / scaling / cropping.** Images are portrait (H > W). Nothing in the legacy code rotates, flips or crops the
  full image; the only transforms are squares cut from the grid and an optional `np.transpose(square)` for 1-channel
  squares with `new_size` (a latent bug: the label is not transposed, `square_2c` is not either - unused with the default 2 channels).
  The grid origin differs between files (see §C), so squares are positioned from each file's own overlay.
* **Pixel size.** `XResolution = 100000/1000000` px per `µm` (2026) and `1000000000/1000000` px per `cm` (2023)
  both mean **10 µm/px** if trusted. Shown as `resolution_note`, never applied. [open: Elena]
* **Normalisation.** Legacy `read_tiff` returns `float32(np.array(frame)) / 255` regardless of dtype: a 16-bit
  value 65535 becomes **257.0**, so values run 0-257, not 0-1. `MEAN=[12.69, 2.48]`, `STD=[13.35, 2.45]` (config) do not match
  `test_img` (page means 48.7 / 44.9 after the same division), so they come from another dataset or from the squares only [unverified].
  No 16-bit information is *lost* by the division (float32 keeps it), but nothing is clipped either, and the 8-bit RGB
  composite would be on a different scale.
* **Risk of losing 16-bit information.** (1) Any `Image.convert`/8-bit export (the composite is 8-bit: 18-19 % of the pixels of each colour plane are at 255); (2) `PIL` reads these frames as big-endian `>u2` (mode `I;16B`, checked), and `cv2.resize` in
  `reshape_numpy` on float32 is the only resampling and is a no-op for 316 -> 316.
  The new importer reads with `tifffile`, keeps dtype, refuses non-`uint8/uint16`, never rescales or resamples.
* **Saturation.** Page 1 has 16-18 % of pixels at 65535 in every file (background outside the gel), page 0 0.5-14 %.
* **Missing codec.** The LZW composite cannot be read without `imagecodecs`; `requirements-app.txt` lacked it
  (added). Without it the importer reports `image_unreadable` with the install hint.

## C. ImageJ ROI conventions

* **ROI types actually encountered.** Two blobs per file (`tifffile.imagej_metadata`):
  `ROI` = `POINT` (type 10), `options=5248` (SUB_PIXEL_RESOLUTION | …), 61-416 points; `Overlays` = `RECT` (type 1)
  composite shape (`multi_coordinates`, 13 sub-paths = 5 vertical + 8 horizontal lines, each `[0,x1,y1,1,x2,y2]`).
  No polygon, freehand, oval or any exclusion shape exists in any file.
* **Decoding.** `roifile` gives `subpixel_coordinates` (float32, ImageJ frame), `counters` (uint8 0-255) and
  `counter_positions` (uint32). Legacy `get_roi_coordinates` returns `subpixel_coordinates[counter_positions == channel]` -
  "channel" is the ImageJ **position = 1-based plane index** the point was clicked on.
* **x/y, origin.** `x` = column, `y` = row, origin top-left, zero-based pixel index = `floor(coord)`. ImageJ's frame has pixel *i*
  spanning [i, i+1): every sub-pixel coordinate in the data has a fractional part in {.5, .17, .83} (= .5 plus multiples of 1/3,
  i.e. clicks at 300 % zoom), 830 of 832 on `AI-B`; the exceptions are exactly one integer-valued point per file (in `test_img` it is the
  first point). Canonical coordinates (the app's: pixel centre = integer) are therefore `coord - 0.5`; ROIs *without* the
  sub-pixel flag store the pixel index itself and are not shifted. The half-pixel cannot be verified from pixels
  (median click-to-peak distance is 4-6 px), so it rests on ImageJ's definition. Legacy code used `int(x)` (floor), which is the same pixel.
* **Counter = field.** In all 8 annotated files the counter id is the **grid cell**: each counter's points fall in a single
  cell, cells are visited in a serpentine (boustrophedon) order, counter 0 = first cell counted. Exceptions: `AI-B` has 3 counters
  with one point each across a cell border (clicks near lines) and one point outside the grid, `AS-B` counter 11 lies in cell (row 0, col 2) amid rows 2-3 (an out-of-order counter).
  The counter is therefore Elena's **field number**, not a class. Legacy `grid_to_squares` used the counter to rescue points that
  were clicked across a border and **clamped** out-of-square points silently (`bring_back_points`); the importer flags, never clamps.
* **Points vs grid vs excluded areas.** Distinguished purely by blob and type: `Overlays` composite line shape -> grid; `ROI` POINT -> points;
  any other area ROI -> `RegionRecord(role=None)`, becomes `excluded`/`valid` **only** if its ROI name is listed explicitly
  (`--excluded-name`). Legacy had no concept of excluded areas; "invalid PCR regions" are encoded nowhere in the files.
* **Grid -> image mapping.** Vertical lines x = 158.0, 474.23, 790.46, 1106.68, 1422.91 and horizontal lines y = 158.0 … 2371.59 (step 316.23 px;
  4 x 7 = 28 cells) in all files except `FS-B_D` (x = 3.0 + k*316.23, y = 123.0 + k*316.23). The legacy config/predict code hard-codes origin 158,
  size 316 and 8x5 lines, so it would mis-place that slide; `grid_to_squares` itself reads the lines from the file and truncates them with
  `astype(int)` (size 316.23 -> 316, accumulating up to ~1 px by the last cell).
  Cell membership: `line_k <= coord < line_k+1` in the ImageJ frame.
* **Unsupported / ambiguous.** (1) A cell with no points is "not counted" **or** "counted, zero" - the ROI cannot say; the CSV can
  (zeros in the middle of a run: `AM_B` field 4, `AS_A` field 8). Legacy treated every empty cell as class 0 ("no polonies") for the classifier.
  (2) Counter 255+ would wrap (uint8). (3) Positions > number of planes, counters of different length from the points, composite shapes that are
  not grids, LINE/POLYLINE/ANGLE types: reported and ignored. (4) Points on position 2 are not reliably on plane-1 signal (`AI-B`: x1.46 vs the other
  plane; `FS-B_D`'s single such point is 3.9x brighter on plane 0): the plane a point was placed on is *where it was clicked*, not proof of what it shows.

### Evidence: ROI points vs the researcher's CSV (all 5 research slides with CSV rows)

Computed by the importer (`reconciliations.jsonl`); nothing is relabelled from it.

| slide | T7B CSV total | ROI position 1 | T7B vs pos 1 | T7A CSV total | ROI position 2 | T7A vs pos 2 |
|---|---|---|---|---|---|---|
| AI_B | 402 | 402 | exact per field (counter i = field i+1) | 14 | 14 | exact |
| AM_A | 100 | 100 | exact | 4 | 0 | **CSV has 1,1,1,1 in fields 1-4; no plane-2 points saved in the TIFF** |
| AM_B | 101 | 103 | fields 1-19 equal; CSV omits empty counter 19 and counters 22-23 (1 point each) | 2 | 2 | exact |
| AS_A | 82 | 82 | exact | 3 | 0 | **CSV 1,1,1 vs no plane-2 points** |
| AS_B | 60 | 60 | exact | 1 | 1 | exact |

So `T7B` is Elena's count of **plane 1 (page 0, 635 nm) points** and `T7A` of **plane 2 (page 1) points**, field by field in 7 of 10 comparisons
(3 comparisons disagree: `AM_A`/`AS_A` T7A and `AM_B` T7B), and every disagreement is a CSV value that has no counterpart in the saved TIFF (or the reverse for 2 points). [inferred - this
is a strong pattern, not a statement from Elena]. FS_A/FS_B have no CSV rows.

## D. Dataset reconstruction

**Implemented:** `src/polony_app/legacy_import/` (`python -m polony_app.legacy_import`). Source files are only read.

Pipeline: discover -> load every TIFF exactly as stored -> parse ROI blobs -> convert -> assign grid cells -> compare with CSV -> report -> (optionally) write.

Canonical dataset (schema `polony-historical-dataset`, version 1; layout in `legacy_import/schema.py`):

| File | One row per | Key fields |
|---|---|---|
| `manifest.json` | dataset | `schema_name/version`, `dataset_id` (content hash of images+labels), `created_at`, importer config, counts, notes |
| `images.jsonl` | source image | `image_id` (`img_`+sha256[:12]), `rel_path`, `sha256`, `kind` (stack/single_plane/rgb_composite), `height,width,n_planes,dtype,bit_depth,axes`, per-plane `label_hint,min,max,saturated_fraction`, `info_text` (verbatim), `scan_settings`, `scan_session`, identity (`assay,specimen_id,replicate,slide_id,file_date,identity_source,identity_confirmed`), `reserved_split`, `composite_of`, `composite_evidence`, `alias_paths` |
| `points.csv` | human point | `label_id`, `image_id`, `source_file`, `roi_source` (ROI/Overlays/sidecar), `roi_index,point_index`, `x,y` (canonical), `x_imagej,y_imagej` (raw), `subpixel`, `counter`, `position`, `page`, `field_id`, `class_label`, `class_confirmed`, `status` (ok/duplicate/out_of_bounds) |
| `fields.csv` | grid cell | `field_id`, `grid_row,grid_col`, `polygon` (canonical corners), `counters`, `n_points_by_page`, `status` (has_points / no_points = unknowable) |
| `regions.jsonl` | other ROI shape | `roi_type,name,polygon,role` (None unless named in config), `position` |
| `reconciliations.jsonl` | CSV row x image | `series,csv_total,csv_counted_fields,csv_blank_fields`, per plane `match` |
| `warnings.jsonl` | warning/failure | `code,severity,message,file,image_id` |

Mapping of the requested items: original image identity = `images.jsonl` (`sha256`, `rel_path`); slide/experiment = `assay, specimen_id,
replicate, slide_id` (**inferred** from the file name, `identity_confirmed=false`); channel = `page` (+ raw `label_hint`; roles absent by design);
human-confirmed coordinates = `points.csv` (*human-made*, not *human-confirmed*: no review flag exists in the historical data, hence `class_confirmed=false`);
class/colour = `class_label` only from an explicit `--class-map`; valid and excluded regions = counted `fields` (valid by counting evidence) and
`regions` (excluded only when named); source ROI/CSV file = `source_file` / `reconciliations.csv_file`; provenance = hashes, `roi_index/point_index`, raw coordinates; warnings = `warnings.jsonl`.

Guarantees tested: dtype/values preserved; no clamping; duplicates (<1 px, same plane) flagged; round trip bit-exact; same inputs -> same `dataset_id`;
files are never written next to the sources; an existing dataset directory is never overwritten; archives (`*.tif.zip`) are listed, never extracted.

**Splitting** (`legacy_import/splits.py`): the unit is a group of images, never patches. Strongest available identity per image:
specimen (`assay|sample`, so slides A and B of one sample stay together) > slide > scan session > source image. Rules: unconfirmed (file-name-inferred)
identity **raises** `IdentityNotConfirmed` unless the researcher's slide list is supplied (`--confirmed-slides`) or `--allow-unconfirmed-identity` is given (then recorded in
the split file); images with no recoverable identity get a per-file group plus a warning; byte-identical files are merged into one group; reserved images (anything under a
`test` directory, or `--reserve`) pin their whole group to that split (`test_img.tif` -> test); an RGB composite follows its stack; patches take the split of their image
(`split_of_patch`, `assert_patch_assignment`); `verify_no_leakage` is run on every result. Deterministic (seeded SHA-256 ordering). With 4 specimens (plus an unidentified image) the requested
70/15/15 cannot give every split data; the tool then warns (`received no images`).

Legacy split, for comparison [confirmed in `make_dataset.py`]: squares of all images, files sorted by name, **first 90 % of the squares = train, rest = valid**, no shuffle,
no group logic, so the boundary slide is cut across train/valid, and the "validation" set also selects checkpoints (no independent test set).

## E. Scientific workflow

Confirmed (code, files, CSV/xlsx evidence) vs inferred, in order:

1. **Scanner.** Axon/GenePix-type scanner (`Software: ImageIO 2.0.4.0`, `AxImageIO: …PMT=550; LaserPower=1.97`) writes a 2-plane 16-bit TIFF at 10 µm/px (if the tag is right), 635 nm first. A rendered 8-bit composite
   can be exported as `<name>_(635_532).tif`. Scan day/settings differ from the file-name date (files dated 29-09, scanned 30-09: hybridisation on the 30th per `Counts_30-09-2026.xlsx`). [confirmed: metadata; inferred: meaning of dates]
2. **Naming/grouping.** `<dd-mm-yyyy>_<assay>_<sample>-<A|B>[_suffix].tif`; CSV `slide ID = <sample>_<A|B>`; two slides per sample. Unexplained: suffix `_D` (`FS-B_D`), sample codes (AI/AS/AM/PI/PS/PM/HI/HS/HM/QI/QS/QM = "phage & dilution"), FS = controls (no CSV rows). [inferred]
3. **ImageJ/Fiji.** Slide opened as a stack; a line-grid overlay is placed (28 cells, 316.23 px = 0.1 cm² at 10 µm/px; origin differs per slide); the Multi-point tool with **counters** is used, the counter incremented per
   field; positions 1/2 = the slice on view. Annotation is saved *into the TIFF*. [confirmed from the files]
4. **Grid / valid region.** Only fields that were counted carry points; their choice rule (inside the gel, not on the rim, how many) is unknown. 6-22 fields with points per slide, up to 24 in the CSV (`field 1…24`). [open]
5. **Manual counting.** One click per polony. Clicks are imprecise (median 4-6 px from the intensity peak). Overlapping polonies, debris and rim artefacts: rule unknown. [open]
6. **Red/green/yellow.** `Counts_30-09-2026.xlsx` header "Cy5 red & Cy3 green (yellow counts)" and the evidence above suggest: T7B = counts on plane 1, T7A = counts on plane 2. Whether a "yellow" polony is counted in both, and whether
   T7A/T7B are clades A/B, colours or replicates, is unconfirmed. No matching rule between red and green points exists in the legacy code (the legacy model only ever learned plane-1 points, `channel=1`).
7. **CSV.** One row per slide, `field 1…24` = counts per counted field, blank = not counted. `FAGE ABUNDENCE [fage mL-1] = mean(non-blank fields) * (field/grid) / SAMPLE VOLUM[ul] * 1000` with `field=2.9`, `grid=0.1` (cm², not pixel² as the header says), reproduces the CSV to 4e-10 (see `docs/app/SCIENCE_NOTES.md` §5).
8. **Biological result.** Legacy `efficacy/polony_to_phage.py`: abundance x mean of hard-coded "slopes" (T4: 6 values, T7: 5 values, T7c: unimplemented `Ellipsis` placeholder) with a bootstrap CI (`n=10000`, `alpha` used as significance level). The origin of the slopes and their units are undocumented; the 2026 lab sheets do not use them. [open]

### Questions for Elena (all still open)

1. Do `T7A`/`T7B` mean clade A/B, red/green or replicate? Are positions 1 and 2 in ImageJ really "T7B" and "T7A" (the data match in 7 of 10 comparisons)?
2. `AM_A` (CSV 1,1,1,1) and `AS_A` (CSV 1,1,1): where do those T7A counts come from, since the saved TIFFs hold no position-2 points?
3. `AM_B`: CSV stops at 21 fields and omits counters 19, 22 and 23 (two single-point fields); intentional?
4. Is a field without points "not counted" or "counted, zero"? How do you decide which fields to count (inside gel, rim, number)?
5. What defines a yellow polony (both channels, distance, size)? Are yellow ones counted in both series? Is the column `yellow` of the workbook a manual tally?
6. Are excluded/invalid PCR regions ever drawn? How (ROI name/type)? None exist in these 8 files.
7. What do the sample codes, the `_D` suffix and `FS` mean? Are A/B technical replicates of one specimen (we group them) or independent?
8. Is the pixel size 10 µm and the chamber 2.9 cm²? Is "field/grid" in cm²? Which volume (2.5 vs 5 µL) was used for which historical file?
9. Where did the T4/T7 slopes come from and may they be used?
10. May the research TIFFs, CSVs and `test_img.tif` be redistributed (repository, training of third-party models)? Are the legacy Drive zips (4 archives, 100+ slides) still available, with their file names `1-33-A.tif`-style?
11. Which slides were used to *test* the old model? (`test_img.tif` is the only known one.)

## F. Reusable components

| Component (legacy path) | Verdict | Why |
|---|---|---|
| `data/utils.py::get_roi_coordinates` (counters + positions) | **Port and rewrite with tests** | The idea (position = plane, counter = field) is right; implementation indexes `roiread(path)[1]` and has 4 near-identical branches. Rewritten as `legacy_import/roi.py` (order-independent, never raises on one bad blob) |
| `data/utils.py::grid_to_squares` (grid -> squares) | **Behavioural reference only** | Reads lines correctly, but truncates to int, clamps points silently, drops squares with 1 point or "2 types of points" (a rule with no documented scientific basis), conflates empty with counted-zero, couples to 2-plane + `channel=1` |
| `bring_back_points` | **Do not reuse** | Silently moves labels |
| `read_tiff` | **Do not reuse** | `/255` irrespective of dtype; PIL multi-frame loop; replaced by `imaging.io.read_stack` (already in the app) |
| `create_density_roi` | **Use only as behavioural reference** | Gaussian sigma 1, 100 per point, `int()` truncation, returns `False` on error. Not needed by Big-FISH/Spotiflow |
| `delete_duplicates` (md5 of pixels, **deletes** files) | **Do not reuse** | Destructive. New: sha256 of bytes, alias recorded, nothing deleted |
| `remove_img_without_roi` (deletes files) / `get_and_unzip` (gdown) | **Do not reuse** | Destructive / network download of unverifiable archives |
| `make_dataset.py` (HDF5, 90/10 sequential split, `path_dict`) | **Do not reuse** | Leaky split, absolute paths, float labels |
| `models/*` (U-Net, classifier, Looper) | **Use only as behavioural reference** | See §G |
| `efficacy/polony_to_phage.py` | **Use only as behavioural reference** | Unknown slope provenance; T7c stub |
| `store/csv_import.py` (current app) | **Reuse as is** | Already blank-vs-zero correct; used for reconciliation |
| `imaging/io.py`, `imaging/ingest.py` helpers (current app) | **Reuse as is** | Dtype-preserving; `channel_stats`, `parse_info_label_hint` imported |
| Filename conventions (`path_dict.json`, 2026 files) | **Port and rewrite with tests** | `legacy_import/identity.py` (both schemes; inferred only) |

No legacy code was copied. `legacy_import/ATTRIBUTION.md` records the behavioural references (repository URL, files, licence MIT, what was changed).
The checkpoints and `path_dict.json` were not copied or modified; the legacy tree on `main` (which already contains them) is untouched.

## G. Old-model assessment

* **Architecture [confirmed].** `UNet` (64 filters, 3 levels, 2 convs/level, BatchNorm, 2 input channels, no residual) -> 1-channel density map; `Classifier` = ResNet18 (ImageNet weights, 2-channel `conv1`, 1 logit) for "square has polonies".
  Checkpoints inspected without torch (restricted unpickler): UNet `block1.0.0.weight (64,2,3,3)`, `density_pred (1,64,1,1)`, 85 tensors; classifier `net.conv1.weight (64,2,7,7)`, `fc (1,512)`, 122 tensors; both with `module.` prefix; shapes match `models.py`.
* **Training target.** Per 316x316 square: Gaussian-smoothed (sigma 1) delta map, 100 per point, from **plane-1 points only**; count = sum/100. Squares with 0 points are skipped for density (`mode="density"`) but kept (class 0) for the classifier.
* **Patch generation.** The 28 grid cells of each file; skips squares with exactly 1 point or with several counter ids of similar frequency; points outside are clamped; optional resize (no-op at 316).
* **Preprocessing.** `/255`, `Normalize(MEAN, STD)` with the train-set channel means (`mean_std` averages per-image means/stds); flips p=0.5 (train only).
* **Inference output.** Per square `{result = sum(density)//100, class, probs, density}`, CSV via `save_predictions_to_csv` (broken, see `LEGACY_AUDIT.md`).
* **Metrics.** MAE/MARE/MSE of counts per square (regression), accuracy/precision/recall/F1/AUC (classifier), logged to W&B. Checkpoint names carry the validation value (`unet_49_1.7496` = epoch 49, 1.7496, probably MAE per square [inferred]; `classifier_57_0.8896` epoch 57 F1 0.8896 [inferred]). Selection and "validation" use the same set.
* **Split.** Sequential 90/10 by squares (see §D) - no slide, specimen or session grouping.
* **Limitations [confirmed in code].** Trained on one plane; labels are human clicks; empty cells assumed negative; no yellow logic; fixed 316 px squares and the 158 px grid origin (wrong for `FS-B_D`); unit mismatches in the predict path.
* **Can it run as a historical benchmark?** *In principle yes*: architectures and checkpoints are compatible and `legacy_squares()` exports the counted cells with the legacy box convention. It was **not run** here (no `torch`, and the predict
  entry points are broken, `LEGACY_AUDIT.md` #1-#5), so **no accuracy statement about it is made**. Requirements for a fair run: fix the predict path in a separate harness, apply `Normalize` with the constants above on `float32(page)/255`, and score only on slides never used by the old training (unknown; its training zips are unavailable, so contamination of any 2026 slide cannot be excluded - the 2026 slides postdate the 2024 repository, which makes them plausible held-out data).

## H. Integration into the current system

Added (nothing existing was modified except two dependency lines):

* `src/polony_app/legacy_import/` - `roi.py`, `identity.py`, `schema.py`, `importer.py`, `splits.py`, `exports.py`, `cli.py`.
* `tests/app/test_legacy_{roi,import,schema_splits}.py`, `tests/app/legacy_fixtures.py` (63 tests; synthetic fixtures built with `tifffile`/`roifile`, one real file: `test_img.tif`).
* `requirements-app.txt`, `pyproject.toml`: `imagecodecs` (LZW composites).
* Docs: this file, `docs/legacy-import.md` (usage), `src/polony_app/legacy_import/ATTRIBUTION.md`.

Adapters: `reference_points`, `score_detections` (Hungarian one-to-one, **only inside counted fields**) for Big-FISH evaluation; `export_spotiflow` (`y,x` CSV, counted-field polygons, explicit pages, `test` never exported unless named) -
the exact Spotiflow folder layout was not verified against an installed Spotiflow [unverified]; `legacy_squares` for the old model; `to_review_seed` (counted cells as `counting` regions, points as manual `uncertain` objects; no colour, `field_index=None`).

Test status: `PYTHONPATH=src pytest tests/app` -> **283 passed, 14 failed (unchanged pre-existing lab-export failures), 2 skipped**. The legacy `tests/test_data|test_models` need torch/network and were not run.

## I. Remaining scientific and data-integrity risks

1. All identity (specimen/slide) is inferred from file names; splits on it are unconfirmed by default (warned/blocked).
2. Only 8 annotated images, 4 identifiable specimens (+ `test_img`): any split is tiny; the only pre-2026 image is a test fixture.
3. Annotation = clicks, not verified detections; unlabelled polonies outside counted fields are not negatives. Always train/score inside `counted_fields`.
4. Empty cells are ambiguous (not counted vs zero); CSV zeros are the only evidence.
5. Plane meaning (position 2) and red/green/yellow semantics are unconfirmed; no class label is assigned without `--class-map`, and even then `class_confirmed=false`.
6. Half-pixel offset of ImageJ coordinates is by definition, not measurable here (clicks are ~5 px imprecise).
7. Legacy training data are unavailable: contamination of the 2026 slides by the old model cannot be ruled out from the repository alone.
