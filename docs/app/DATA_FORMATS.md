# The lab's data formats (TIFF, ImageJ ROI, grid, CSV, workbook)

Reference for how the lab's scanner images, ImageJ annotations and count tables are structured, and for the canonical
dataset that [`polony_app.legacy_import`](HISTORICAL_IMPORT.md) builds from them. Statements describe the files as they
are (the annotated research slides of 29-09-2026 and `test_img.tif`); points that the files cannot settle are listed under
[Unconfirmed](#unconfirmed).

## 1. Files at a glance

| Format | Content | Notes |
|---|---|---|
| ImageJ 2-page `uint16` TIFF (`ZYX`, `slices=2`) with embedded `ROI` + `Overlays` | the scanner image **and** the researcher's annotation in one file | 2 equal-size planes; `ROI` blob = points, `Overlays` blob = counting grid |
| `uint8` RGB TIFF, LZW (`<stem>_(635_532).tif`) | rendered composite of the same slide | name suffix = the two wavelengths; reading it needs `imagecodecs` (LZW) |
| Standalone `.roi` / `.zip` RoiSet | not present in the lab's data | read as a sidecar with the same stem, if one ever appears |
| Lab CSV `T7A_<dd-mm-yyyy>.csv`, `T7B_<dd-mm-yyyy>.csv` | per-field counts and abundance, one row per slide | blank = not counted, 0 = counted zero; header typos are kept verbatim |
| `Counts_<dd-mm-yyyy>.xlsx`, `T7*.xlsx` | lab summary workbooks | layout in [LAB_EXPORT.md](LAB_EXPORT.md) |
| `*.tif.zip` | zipped TIFFs | listed by the importer, never extracted |

## 2. ImageJ TIFF stacks

* **Layout.** ImageJ-written TIFF, 2 IFDs (pages), `uint16`, uncompressed, `photometric=minisblack`. `tifffile` reports
  `series[0].shape == (2, H, W)`, `axes == 'ZYX'` and `imagej_metadata['slices'] == 2`: the two planes are **z-slices,
  not ImageJ channels**. There is no RGB page in the stack; the composite is a separate file.
* **Size and orientation.** Portrait (H > W). the synthetic `test_img.tif` is 2450 x 1438 (H x W); the 2026 research
  slides are 2438 x 1460. The size is not fixed across scans, so nothing may assume one.
* **Info string.** The only plane label is `ImageDescription: 635 [W1]` (the ImageJ `Info` text), attached to the stack
  once; page 1 carries no label. 635 nm is the red laser line. The label is a hint, not a role.
* **Scanner metadata.** Axon/GenePix-type scanner tags (`Software: ImageIO 2.0.4.0`, `AxImageIO: …PMT=550; LaserPower=1.97`).
  The scan date in the metadata can differ from the date in the file name (files named 29-09 were scanned on 30-09).
* **Plane 0 = 635 nm (red).** In a stack/composite pair the composite R plane correlates with page 0 (Spearman 0.996)
  and the G plane with page 1 (0.999), with zero spatial shift (FFT cross-correlation peak at (0, 0)); points placed on
  ImageJ position 1 sit on page-0 signal (several times brighter than on page 1). The importer records this as
  `composite_evidence` but never writes a colour role: the channel mapping is confirmed by the user.
* **Red / green / yellow** are not encoded in any file. Colour names come only from the composite suffix `(635_532)` and
  the workbook header "Cy5 red & Cy3 green (yellow counts)". No file marks a point as yellow.
* **Resolution tags.** `XResolution = 100000/1000000` px per `µm` (2026 files) and `1000000000/1000000` px per `cm`
  (2023 file). Both mean **10 µm/px** if trusted. The importer shows this as `resolution_note` and never applies it.
* **Composite pairing.** One slide = one 2-plane stack plus, optionally, one RGB composite `<stack stem>_(635_532).tif`.
  No separate single-plane red/green files exist. Pairing rule: same directory, same base stem after removing
  `_(ddd_ddd)`, identical `H x W`; anything else is reported, never guessed.
* **Value range and saturation.** Full 16-bit range. Page 1 has 16-18 % of pixels at 65535 (background outside the gel),
  page 0 0.5-14 %. The 8-bit composite has 18-19 % of the pixels of each colour plane at 255, so it is not a substitute
  for the stack. Read with `tifffile`, keep the dtype, never rescale, down-convert or resample.

## 3. Embedded ImageJ annotations (`ROI` and `Overlays`)

Each annotated TIFF carries two blobs in `tifffile`'s `imagej_metadata`:

| Blob | ROI type | Content |
|---|---|---|
| `ROI` | `POINT` (type 10), `options=5248` (includes `SUB_PIXEL_RESOLUTION`) | the researcher's clicks: 61-416 points per file |
| `Overlays` | `RECT` (type 1) composite shape (`multi_coordinates`) | the counting grid: 13 sub-paths = 5 vertical + 8 horizontal lines |

No polygon, freehand, oval or other area shape exists in the lab's files; excluded / invalid PCR regions are not drawn.

* **Decoding.** `roifile` yields per point `subpixel_coordinates` (float32, ImageJ frame), `counters` (uint8, 0-255) and
  `counter_positions` (uint32).
* **Position = plane.** The *position* is the 1-based plane index the point was clicked on (1 = page 0, 2 = page 1). It
  records where the click was made, not what the point shows: points on position 2 are not reliably on page-1 signal.
* **Counter = field number.** The counter id is the **grid cell** being counted: each counter's points fall in one cell,
  cells are visited in a serpentine (boustrophedon) order and counter 0 is the first cell counted. The counter is the
  researcher's field number (counter *i* = CSV `field i+1`), not a class. Isolated exceptions occur: a click just across
  a cell border, a point outside the grid, or an out-of-order counter. The importer flags such points; it never moves
  or clamps them.
* **Coordinates.** `x` = column, `y` = row, origin top-left. In the ImageJ frame pixel *i* spans [i, i+1); the
  zero-based pixel index is `floor(coord)`. Sub-pixel coordinates have fractional parts .5, .17 or .83 (.5 plus
  multiples of 1/3: clicks at 300 % zoom), except one integer-valued point per file. The app's canonical frame
  (pixel centre = integer) is therefore `coord - 0.5` for sub-pixel ROIs; ROIs without the sub-pixel flag store the
  pixel index and are not shifted. Clicks lie a median 4-6 px from the intensity peak.
* **Other ROI shapes**, if present, become `regions` with `role=None`; they count as excluded or valid **only** when their
  ROI name is passed explicitly (`--excluded-name` / `--valid-name`).
* **Not supported, reported and ignored:** counters above 255 (uint8 wrap), positions greater than the number of planes,
  counter arrays of a different length from the points, composite shapes that are not grids, `LINE` / `POLYLINE` / `ANGLE`.

## 4. Counting grid

* **Geometry.** Vertical lines at x = 158.0, 474.23, 790.46, 1106.68, 1422.91 and horizontal lines from y = 158.0 to
  2371.59, spacing **316.23 px** (≈ 316): 5 x 8 lines, 4 x 7 = **28 cells**. At 10 µm/px a cell is 0.1 cm².
* **Origin varies per slide.** Most files use origin (158, 158); `FS-B_D` uses x = 3.0 + k·316.23, y = 123.0 + k·316.23.
  The grid must therefore be read from each file's own overlay, never from fixed constants (origin 158, size 316,
  8 x 5 lines).
* **Line parsing.** Each sub-path of the composite shape is `[0, x1, y1, 1, x2, y2]` (moveTo, lineTo). A segment with
  `x1 == x2` is a vertical line at that x, one with `y1 == y2` a horizontal line at that y. Keep the float values;
  truncating to int (316.23 -> 316) accumulates up to ~1 px by the last cell.
* **Cell membership and numbering.** A point belongs to cell (row *r*, col *c*) when `line_c <= x < line_c+1` and
  `line_r <= y < line_r+1` in the ImageJ frame. Cells are identified by `grid_row, grid_col` (row-major, top-left = (0, 0));
  the researcher's field number is the counter, not a fixed cell index.
* **Counted fields.** Only fields that were counted carry points: 6-22 cells per slide, up to 24 in the CSV
  (`field 1…24`). A cell with no points is either "not counted" or "counted, zero"; the ROI cannot tell, the CSV can.

## 5. File naming

| Scheme | Pattern | Example |
|---|---|---|
| 2026 research slides | `<dd-mm-yyyy>_<assay>_<sample>-<A\|B>[_suffix].tif`, e.g. `29-09-2026_T7_AB_<slide>.tif` | `29-09-2026_T7_AB_AM-A.tif`, `…_FS-B_D.tif` |
| Composite of a slide | `<stack stem>_(635_532).tif` | `…_AS-B_(635_532).tif` |
| Older scans | `<batch>-<n>-<A\|B>.tif` | `1-33-A.tif` |

* The CSV slide id is `<sample>_<A|B>` (`AM-A.tif` -> `AM_A`); there are two slides (A, B) per sample.
* Sample codes (`AI`, `AS`, `AM`, `PI`, `PS`, `PM`, `HI`, `HS`, `HM`, `QI`, `QS`, `QM`) encode phage and dilution; `FS` slides
  are controls and have no CSV rows. Identity derived from a file name is **inferred** (`identity_confirmed=false`).

## 6. CSV and workbook conventions

* **Series.** One CSV per series and date: `T7A_<date>.csv`, `T7B_<date>.csv`; one row per slide (`slide ID`),
  columns `field 1…24` = count per counted field. **Blank = not counted, `0` = counted, none found**; a zero in the middle
  of a run is a real zero. Header typos (`FAGE ABUNDENCE`, `SAMPLE VOLUM`) are part of the format.
* **Abundance.** `FAGE ABUNDENCE [fage mL-1] = mean(non-blank fields) * (field / grid) / SAMPLE VOLUM[ul] * 1000` with
  `field = 2.9` and `grid = 0.1`, both in cm² (the header says pixel²); this reproduces the CSV values to 4e-10. See
  [SCIENCE_NOTES.md](SCIENCE_NOTES.md).
* **Relation to the ROI.** `T7B` counts the points on **position 1** (page 0, 635 nm) and `T7A` the points on
  **position 2** (page 1), field by field. In the research samples this holds exactly in 7 of 10 slide/series pairs;
  each mismatch is a CSV value with no matching points in the saved TIFF (or 2 extra single-point fields the CSV omits).
  The importer records this comparison in `reconciliations.jsonl` and never relabels from it.
* **Workbooks.** `Counts_<date>.xlsx` summarises the series per slide under the header "Cy5 red & Cy3 green (yellow
  counts)" and carries the hybridisation date. Its layout and the lab-format export are in [LAB_EXPORT.md](LAB_EXPORT.md).

## 7. Canonical historical dataset (import output)

Schema `polony-historical-dataset`, version 1 (layout in `src/polony_app/legacy_import/schema.py`). Images are referenced
by path + sha256, not copied.

| File | One row per | Key fields |
|---|---|---|
| `manifest.json` | dataset | `schema_name/version`, `dataset_id` (content hash of images + labels), `created_at`, importer config, counts, notes |
| `images.jsonl` | source image | `image_id` (`img_` + sha256[:12]), `rel_path`, `sha256`, `kind` (stack / single_plane / rgb_composite), `height, width, n_planes, dtype, bit_depth, axes`, per plane `label_hint, min, max, saturated_fraction`, `info_text` (verbatim), `scan_settings`, `scan_session`, identity (`assay, specimen_id, replicate, slide_id, file_date, identity_source, identity_confirmed`), `reserved_split`, `composite_of`, `composite_evidence`, `alias_paths` |
| `points.csv` | human point | `label_id`, `image_id`, `source_file`, `roi_source` (ROI / Overlays / sidecar), `roi_index, point_index`, `x, y` (canonical), `x_imagej, y_imagej` (raw), `subpixel`, `counter`, `position`, `page`, `field_id`, `class_label`, `class_confirmed`, `status` (ok / duplicate / out_of_bounds) |
| `fields.csv` | grid cell | `field_id`, `grid_row, grid_col`, `polygon` (canonical corners), `counters`, `n_points_by_page`, `status` (has_points / no_points = unknowable) |
| `regions.jsonl` | other ROI shape | `roi_type, name, polygon, role` (None unless named in the config), `position` |
| `reconciliations.jsonl` | CSV row x image | `series, csv_total, csv_counted_fields, csv_blank_fields`, per plane `match` |
| `warnings.jsonl` | warning / failure | `code, severity, message, file, image_id` |
| `splits/<name>.json` | split | written by `split`: `seed, level, fractions, unconfirmed_identity, warnings, group_to_split, image_to_group, image_to_split` |

What each field means:

* **Image identity** = `sha256` + `rel_path`; byte-identical files are merged (`alias_paths`), nothing is deleted.
* **Slide / experiment** = `assay, specimen_id, replicate, slide_id`, inferred from the file name (`identity_confirmed=false`)
  unless a confirmed slide list is supplied.
* **Channel** = `page` plus the raw `label_hint`; no colour role is stored.
* **Coordinates** are human-made clicks, not human-confirmed detections: `class_confirmed=false` always, and `class_label`
  is set only from an explicit `--class-map`.
* **Valid / excluded regions** = counted `fields` (valid by counting evidence) and `regions` (excluded only when named).
* **Provenance** = hashes, `roi_index / point_index`, raw ImageJ coordinates, `source_file`, `reconciliations.csv_file`.

Guarantees: dtype and values preserved; no clamping; duplicates (< 1 px apart on the same plane) flagged; round trip
bit-exact; same inputs -> same `dataset_id`; nothing written next to the sources; an existing dataset directory is never
overwritten.

**Splits.** The unit is a group of images, never patches. Group by the strongest identity available: specimen
(`assay|sample`, so slides A and B of one sample stay together) > slide > scan session > source image. Inferred identity
blocks splitting unless confirmed (`--confirmed-slides`) or explicitly allowed (`--allow-unconfirmed-identity`, recorded in
the split file). Images under a `test` directory or named by `--reserve` pin their whole group to that split; a composite
follows its stack; patches take their image's split. Assignment is deterministic (seeded SHA-256 ordering) and checked for
leakage. Training and scoring use only the counted fields: the rest of an image is unlabelled, not background.

## Unconfirmed

* What `T7A` / `T7B` denote (clade, colour or replicate), and whether positions 1 / 2 always map to `T7B` / `T7A`.
* Whether a field without points was not counted or counted as zero, and how the counted fields are chosen (inside the
  gel, away from the rim, how many).
* What defines a yellow polony (both channels, distance, size), whether it is counted in both series, and whether the
  workbook's yellow column is a manual tally.
* Whether excluded / invalid PCR regions are ever drawn, and with which ROI name or type.
* The meaning of the `_D` suffix and of the sample codes; whether slides A and B are technical replicates of one specimen.
* The pixel size (10 µm per the resolution tag), the chamber area (2.9 cm²), and the sample volume (2.5 vs 5 µL) used for
  each historical file.
* The counting rule for overlapping polonies, debris and rim artefacts.
* The source of the few CSV counts that have no matching points in the saved TIFFs (`AM_A`, `AS_A` T7A; `AM_B` fields 19, 22, 23).
