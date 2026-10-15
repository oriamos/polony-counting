# Real-data validation (researcher TIFFs, T7 AB assay, slides 29-09-2026)

Status: complete (stages 1-5, 2026-10-10). Inputs are the researcher's 8 raw two-page slides
(`resources/research_tiffs/`, git-excluded, never committed) plus 7 RGB composites (GenePix exports of 7 of
these slides), and
`Counts_30-09-2026.xlsx` / `T7A_15-09-2026.csv` / `T7B_15-09-2026.csv`. All scripts are in
`scripts/validation/` and are run as `python -I <script> <paths...>` (inputs are untrusted).
Caveat for every number below: 8 slides, one annotator, one scan session; mappings are inferences from
the data unless stated as read from metadata.

---

## Stage 1 (2026-10-10): file facts, embedded annotations, channel identity, chamber

Scripts: `stage1_facts.py <tiff_dir> <Counts.xlsx> <out>` (facts, ROI vs sheet, RGB correlation),
`chamber.py <out> <raw.tif...>` (chamber mask, rim profile, per-cell table), `crop_view.py` (zoomed crops).

### 1.1 Raw files (all 8 identical in structure)

* ImageJ 1.54g hyperstack, 2 pages (`slices=2`), uint16, 2438 x 1460 (H x W), uncompressed, no `Labels`.
  Resolution tag 10 px/unit, unit µm (i.e. 10 µm/px as declared by the scanner export).
* Page-level TIFF tags carry **no** DateTime and no wavelength. The only channel text is the ImageJ `Info`
  block (one per file, copied from the scanner's first image): `ImageDescription: 635 [W1]`,
  `Software: ImageIO 2.0.4.0` (Axon/GenePix), `DateTime: 2026/09/30 17:01–18:00`. Nothing says `532`/`W2`.
* Scanner settings in `Info.HostComputer`: AI/AM/AS slides `PMT=550; LaserPower=1.97`;
  **FS slides `PMT=500; LaserPower=3.30`** (different scan settings), all `FullSize=2200,7200`.
* Saturation (value 65535): page 0 0.1–1.6 % of pixels for AI/AM/AS, **12–13 % for FS**; page 1 ~18 % in
  every file — this is the scanner background **outside** the chamber, which is clipped in page 1
  (page 0 is ~20–30k there). Inside the chamber both pages sit at ~250–1000 counts (FS page 0 ~1.5–2.5k).
* Pixel correlation page0 vs page1 (log, 4x binned, whole frame): 0.93–0.985, dominated by the
  inside/outside step, not by polonies.

### 1.2 Embedded annotations = the researcher's counts (key finding)

* `ROI` = one ImageJ multi-point ROI per file (counter/“type” per point + `counter_positions` = slice).
* `Overlays` = one composite ROI of 13 lines = the counting grid: verticals at x = 158 + k·316.23
  (k = 0..4), horizontals at y = 158 + k·316.23 (k = 0..7) → **4 columns x 7 rows = 28 cells, 316.23 px**,
  origin (158, 158) in 7 files; the overlay stored in FS-B_D is itself a shifted grid (origin about (0, 124), see stage 4).
* **Points on slice 1 (page 0) reproduce the 'Clade B' sheet / T7B CSV exactly**, one ImageJ counter
  per field: AI-A (332 pts), AI-B (402), AM-A (100), AS-A (82), AS-B (60) are exact matches of the
  per-field series. AM-B: 103 pts vs sheet 101 — fields 1–19 identical, the tail differs
  (ROI counters 20..23 = 3,3,1,1 vs sheet fields 20..21 = 3,3; the file was probably saved after edits).
* **Points on slice 2 (page 1) reproduce the 'Clade A' sheet / T7A CSV**: AI-B (14 pts, exact per field),
  AM-B (2 pts, fields 1 and 12, exact), AS-B (1 pt, field 11, exact). These are the 3 slides whose page-1 points match the Clade A sheet; FS-B_D carries one more page-1 point (field 5, stage 4), so 4 of the 8 slides have any page-1 marks. AI-A / AM-A / AS-A carry no slice-2
  points although the sheet has Clade A counts 4 / 4 / 3 (those annotations were not saved in the file).
* Counter index → field: counter k = field k+1 in all slides **except AI-A**, where counter 3 is unused
  and counters 4..18 = fields 4..18 (ordinal, not index). Fields with 0 count have no counter (AS-A field 8).
* Each counter's points lie in exactly one overlay cell (AI-A, AM-A, AM-B, AS-A, AS-B: 100 %;
  AI-B: 3 stray points of 416 one cell over). So **the field → grid-cell mapping can be read from the file;
  it does not need to be guessed.** The order is a boustrophedon over the rows but it is **not
  consistent between slides**: row 0 is visited (c2,c1) in AI-A/AI-B/AM-B and (c1,c2) in AM-A/AS-A;
  AI-B visits r0c1 as field 13; AS-B visits r0c2 as field 12; AI-A visits r4c3 as field 13 between
  r3c3 and r3c2. No fixed ordering (row/column-major, snake, any corner) reproduces all slides.
* Sheet blanks: fields beyond the series length are cells that were not counted. Counted cells are always
  inside the chamber; never-counted cells are the 4 corner cells (r0c0, r0c3, r6c0, r6c3; 17–68 %
  chamber coverage) and, slide-dependent, rows 5–6 (a dark bubble region at the bottom: background
  page 0 ~200–370 vs 450–900 elsewhere, visible in every AI/AM/AS overview).
* **FS-A / FS-B_D do not match the 'Controls' sheet.** The Controls rows (header dated 13-05-2026) are
  FS A = [0], FS B = [1], NTC A = [0] (one field each). The FS TIFFs were scanned 2026-09-30 with other
  scanner settings and carry 182 pts / 7 fields (FS-A: 24,34,24,18,26,28,28) and 151 + 1 pts / 6 fields
  (FS-B_D) — no row of any sheet or CSV matches them. FS-B_D's counters each straddle 2–4 cells of the
  (158, 158) grid of the other slides: FS-B_D was counted on a shifted grid, and the overlay stored in that
  file is itself that shifted grid (see stage 4).

### 1.3 Channel identity from the RGB composites (evidence, not filenames)

Each `*_(635_532).tif` is a GenePix RGB export (LZW, uint8, same 2438 x 1460 frame, DateTime 0000).
Spearman correlation of 4x-binned planes vs raw pages (7 slides):

| plane | vs page 0 | vs page 1 |
|---|---|---|
| R | **0.995–0.997** | 0.85–0.97 |
| G | 0.85–0.97 | **0.998–1.000** |
| B | 0.67–0.68 | 0.68 |

Phase correlation R↔page0 and G↔page1: shift (0, 0) px in all 7. Transfer function: plane ≈ 2.76·√raw − c
(R² 0.998–1.000, linear R² 0.87–0.96), so the composite clips at raw ≈ 8.6–16k (page 0 values at which R
reaches 255: p5 = 11–17k). B is a constant-ish mask of the clipped outside region. Conclusion:
**page 0 = red = 635 nm (`W1`, matches the Info text) = the channel the researcher counted for Clade B /
T7B; page 1 = green = 532 nm = Clade A / T7A.** The composites lose dynamic range (square-root display,
clipping) and should not be used for detection.

### 1.4 Chamber geometry and mask

The chamber is a stadium-shaped region (x ≈ 32–57 → 1446–1459, y ≈ 13–24 → 2399–2410; 81.5–81.8 % of the
frame) surrounded by the clipped scanner background. Mask (`chamber.py`): 8x-binned fraction of page-1
pixels ≥ 60000 > 0.5 → outside; largest inside component, holes filled; boundary refined at full res
(5 px median of page 1 < 60000). All 1330 embedded points fall inside the mask (min distance to the edge
24–29 px for AI/AM/AS, 138/162 px for FS). Radial profile from the edge: 0–10 px bright transition,
10–50 px a bright gasket ring (in FS a band of saturated debris, 1–4 % of page-0 pixels clipped up to
50 px), 55–70 px a dark trough, plateau from ~80 px. **Counting region = chamber eroded by 50 px**
(drops 1–8 researcher points per slide that sit 24–50 px from the edge; see stage 2 for the exact count).
Inner dark bubble (rows 5–6) is inside the mask; it is not excluded by the mask, only by the grid choice.

Stage-1 summary: annotations embedded in the TIFFs ARE the sheet counts (page 0 → Clade B, page 1 →
Clade A); field→cell mapping is per-slide and recoverable from ROI counters; RGB R=page0, G=page1;
chamber mask robust from page-1 clipping; FS slides are not the Controls rows.

---

## Stage 2 (2026-10-10): the app pipeline on AI-A, AI-B, AM-A, AM-B

Script: `run_app.py <out> 50 3,4,5 <raw.tif...>` (calls `polony_app.imaging.ingest.ingest_files` and
`polony_app.detection.pipeline.run_detection` read-only; results cached as JSON), `analyze.py` (numbers),
`render_overlay.py` / `missed.py` (pictures, failure characterisation). Setup: red = page 0, green =
page 1 (stage 1 evidence); one counting region = chamber mask eroded by 50 px (polygon from the mask
contour); app defaults except `spot_radius_px` ∈ {3,4,5} (both channels); sensitivity run with
`threshold_scale = 0.5` at r = 4, 5. Recall/precision: one-to-one matching of the researcher's slice-1
points to detections within 15 px, both restricted to the cells the researcher counted (detections in
uncounted cells are not false positives); points in the 50-px rim are excluded (3–9 per slide).

| slide | setting | runtime s | objects (whole chamber) | red-only | green-only | yellow | uncertain | flagged | researcher pts (all/in region) | red-positive recall / precision | pts with no detection of any class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AI-A | r3 | 11.0 | 376 | 320 | 19 | 2 | 35 | 300 | 332/327 | 0.67 / 0.88 | 99 |
| AI-A | r4 | 11.5 | 342 | 296 | 20 | 2 | 24 | 147 | 332/327 | 0.66 / 0.94 | 101 |
| AI-A | r5 | 10.8 | 333 | 286 | 29 | 3 | 15 | 90 | 332/327 | 0.65 / 0.95 | 105 |
| AI-A | r5, thr x0.5 | 11.1 | 429 | 341 | 55 | 3 | 30 | 155 | 332/327 | 0.77 / 0.94 | 65 |
| AI-B | r3 | 10.6 | 500 | 403 | 53 | 3 | 41 | 471 | 402/393 | 0.73 / 0.92 | 98 |
| AI-B | r4 | 11.2 | 483 | 411 | 41 | 3 | 28 | 360 | 402/393 | 0.76 / 0.94 | 86 |
| AI-B | r5 | 11.3 | 422 | 382 | 22 | 2 | 16 | 200 | 402/393 | 0.72 / 0.94 | 102 |
| AI-B | r5, thr x0.5 | 11.7 | 516 | 445 | 38 | 4 | 29 | 260 | 402/393 | 0.81 / 0.94 | 66 |
| AM-A | r3 | 10.7 | 115 | 82 | 2 | 0 | 31 | 110 | 100/97 | 0.52 / 0.82 | 44 |
| AM-A | r4 | 10.3 | 153 | 108 | 12 | 0 | 33 | 108 | 100/97 | 0.71 / 0.88 | 24 |
| AM-A | r5 | 10.3 | 148 | 114 | 8 | 0 | 26 | 69 | 100/97 | 0.74 / 0.88 | 22 |
| AM-A | r5, thr x0.5 | 10.6 | 194 | 133 | 15 | 1 | 45 | 114 | 100/97 | 0.77 / 0.84 | 17 |
| AM-B | r3 | 10.3 | 111 | 65 | 27 | 1 | 18 | 108 | 103/100 | 0.45 / 0.73 | 54 |
| AM-B | r4 | 11.8 | 117 | 85 | 13 | 0 | 19 | 88 | 103/100 | 0.63 / 0.79 | 36 |
| AM-B | r5 | 9.9 | 126 | 92 | 15 | 0 | 19 | 71 | 103/100 | 0.71 / 0.82 | 28 |
| AM-B | r5, thr x0.5 | 10.3 | 172 | 106 | 23 | 2 | 41 | 112 | 103/100 | 0.79 / 0.77 | 20 |

(r4 with thr x0.5: recall 0.79 / 0.85 / 0.75 / 0.75, precision 0.92 / 0.93 / 0.83 / 0.76.)
Runtime ≈ 10–12 s per slide (full 2438 x 1460, both channels, shared CPU). Ingest warnings: the
single `635 [W1]` hint for the whole stack, saturation warnings for both channels (page 1 ≈ 18 % clipped
= outside the chamber).

What the overlays show (`render_overlay.py`, looked at full-slide and 3x crops of all 4 slides):
* **Polonies are large**: half-max radius 6–8 px (≈ 12–16 px diameter, larger for bright ones), diffuse.
  r = 3 under-segments dim ones (AM recall 0.45–0.52); r = 5 is the best default of the three.
* **Main miss = dim polonies the researcher counts** (missed points: median peak ≈ 750–1300 counts on a
  ~500–700 background, local SNR ≈ 5–7; 11–40 per slide have SNR < 5 at a 6-px peak). The automatic
  threshold is set by the many bright polonies (31 noise units on AI-A); halving it recovers +5 to
  +12 recall points with little precision loss. No missed point had a saturated core.
* **False positives are point-like specks**: half-max radius ≈ 1 px, peak 9–13k (dust / hot pixels),
  plus green fibres. 70–90 % of the red-positive false positives already carry `debris_like`; only
  0–3 per slide carry no reason. So "red-positive minus debris_like" is close to what the researcher counts.
* **Green channel (page 1) is essentially empty**: the app warns "73–100 % of green spots are
  debris-like … this channel may be empty" on every slide (correct warning); green detections are
  fibres/specks, 12–43 per slide vs 0–14 researcher Clade-A points.
* A large out-of-focus red haze (AM-B centre) is correctly not detected and not counted by the researcher.
* Review load: 30–90 % of objects carry a reason (weak_signal dominates at r3/r4 — 208/376 on AI-A r3);
  at r5 71–200 flagged per slide. Of the matched (true) red objects at r4, 27–54 % are flagged, mostly
  weak_signal — the queue is long but not wrong for these dim polonies.

## Stage 3 (2026-10-10): researcher comparison for AI-A, AI-B, AM-A, AM-B

Grid: from the Overlays (4 x 7, 316.23 px, origin 158,158; stage 1). Field → cell mapping: read from the
embedded counters (`analyze.py field_cells`: 'index' mapping agrees on 17/17, 17/17, 19/21 fields for
AI-B, AM-A, AM-B; AI-A needs the 'ordinal' mapping, 18/18).

Fixed orderings tested over the cells ≥ 85 % inside the chamber (row/column-major and snake, from each
corner = 16 orderings): best correct-cell hits 7/18 (AI-A, row-snake from top-left), 3/17 (AI-B),
14/17 (AM-A, row-snake top-left), 19/21 (AM-B, row-snake top-right). **No fixed ordering is consistent;
the hypothesis "Field-n follows a fixed grid order" is rejected.** With the embedded mapping the per-field
agreement is much higher (AI-A red-positive r = 0.75 vs 0.44 with the best fixed ordering; AI-B 0.85 vs −0.23).

Per-field agreement, detections in the field's cell vs the sheet (r = Pearson, k = least-squares slope
through 0 of detected vs researcher, MAE in polonies/field):

| slide (fields) | count type, setting | vs Clade B (T7B) | vs Clade A (T7A) |
|---|---|---|---|
| AI-A (18) | red_positive r5 | r 0.75, k 0.67, MAE 6.0 (332 → 224) | r 0.37 (sheet total 4) |
| | red_positive r5 thr x0.5 | r 0.86, k 0.80, MAE 3.6 (332 → 267) | |
| | green_positive r5 | r −0.09, k 0.07 | r 0.11, total 4 → 24 |
| AI-B (17) | red_positive r5 | r 0.87, k 0.75, MAE 6.0 (402 → 300) | r 0.11 |
| | red_positive r5 thr x0.5 | r 0.86, k 0.84, MAE 3.6 (402 → 340) | |
| | green_positive r5 | r 0.50, k 0.04 | r 0.23, total 14 → 17 (5/14 points matched at r4) |
| AM-A (17) | red_positive r5 | r 0.71, k 0.81, MAE 1.2 (100 → 82) | r −0.02 |
| | red_positive r5 thr x0.5 | r 0.79, k 0.89, MAE 1.0 (100 → 89) | |
| | green_positive r5 | r −0.08 | r 0.10, total 4 → 7 |
| AM-B (19 mapped) | red_positive r5 | r 0.66, k 0.82, MAE 1.2 (98 → 85) | r 0.00 |
| | red_positive r5 thr x0.5 | r 0.81, k 0.94, MAE 1.0 (98 → 97) | |
| | green_positive r5 | r −0.43 | r 0.57, total 2 → 14 |

red_only ≈ red_positive (yellow is 0–3 per slide). `all` classes and `red_detected_any` are not better
than red_positive (they add green specks/uncertain objects). Point level: see the stage-2 table.

Hypotheses:
* **Clade B / T7B = red channel = page 0 (635 nm): supported** (exact slice-1 point identity with the
  sheet; per-field r 0.66–0.87 at defaults, 0.79–0.86 at half threshold; green_positive does not track it).
* **Clade A / T7A = green channel = page 1 (532 nm): supported by the annotations** (slice-2 points =
  Clade A sheet in AI-B/AM-B/AS-B) but **detection agreement is inconclusive**: 0–14 events per slide;
  the app's green counts are dominated by specks/fibres (2–7x the sheet totals, 5/14 points matched).
* "Yellow" (dual) as a count type: not used by the researcher on these slides (0–3 app yellows/slide;
  one bright AI-A polony called yellow was not marked by the researcher at all) — inconclusive.
* **Blank sheet fields = cells outside the chamber: partly.** The 4 corner cells (17–68 % chamber) are
  never counted, but 2–7 cells that are ≥ 85 % inside the chamber are also never counted (rows 5–6, the
  dark bubble region in AI/AM; r4c0/r4c1 in AM-A). A field with 0 polonies has no counter, so its cell
  must be inferred by elimination (AM-B field 4 = r1c1; AS-A field 8 = r2c1). The stop rule (how many
  fields) is not derivable from the data (17–22 fields, 60–402 polonies).
* Systematic undercount by the app at defaults (slope 0.67–0.82) is driven by dim polonies, not by
  saturation or merging.

---

## Stage 4 (2026-10-10): AS-A, AS-B, FS-A, FS-B_D

Same setup as stage 2 (rim 50 px; r4, r5, r5 with threshold x0.5).

| slide | setting | runtime s | objects | red-only | green-only | yellow | uncertain | flagged | researcher pts | red-pos recall / precision |
|---|---|---|---|---|---|---|---|---|---|---|
| AS-A | r4 | 10.1 | 95 | 52 | 21 | 1 | 21 | 70 | 82/77 | 0.56 / 0.91 |
| AS-A | r5 | 9.8 | 103 | 71 | 11 | 0 | 21 | 54 | 82/77 | 0.70 / 0.87 |
| AS-A | r5 thr x0.5 | 11.1 | 157 | 95 | 22 | 0 | 40 | 104 | 82/77 | 0.83 / 0.84 |
| AS-B | r4 | 11.2 | 94 | 42 | 24 | 0 | 28 | 78 | 60/59 | 0.39 / 0.68 |
| AS-B | r5 | 9.6 | 93 | 47 | 18 | 0 | 28 | 65 | 60/59 | 0.51 / 0.77 |
| AS-B | r5 thr x0.5 | 10.8 | 143 | 66 | 34 | 0 | 43 | 104 | 60/59 | 0.73 / 0.80 |
| FS-A | r4 | 11.3 | 746 | 635 | 50 | 4 | 57 | 643 | 182/182 | 0.80 / (n/a) |
| FS-A | r5 | 13.2 | 658 | 581 | 38 | 2 | 37 | 457 | 182/182 | 0.80 / (n/a) |
| FS-A | r5 thr x0.5 | 13.8 | 813 | 689 | 57 | 6 | 61 | 583 | 182/182 | 0.86 / (n/a) |
| FS-B_D | r5 | 11.3 | 466 | 427 | 13 | 1 | 25 | 282 | 151/151 | 0.86 / (n/a) |
| FS-B_D | r5 thr x0.5 | 12.4 | 582 | 514 | 29 | 1 | 38 | 371 | 151/151 | 0.91 / (n/a) |

FS recall is against all red-positive detections: cell-restricted precision (stages 2–3) is not given for FS,
and for FS-B_D `analyze.py`'s cell-restricted numbers, 0.52 / 0.71, are an artefact of using the (158, 158)
grid, which is not the grid FS-B_D was counted on.

Per-field (embedded mapping, Clade B): AS-A red_positive r5 r = 0.83, k = 0.73, MAE 1.3 (82 → 62);
thr x0.5 r = 0.88, k = 0.90, MAE 1.0 (82 → 76). AS-B r5 r = 0.77, k = 0.67, MAE 1.2 (60 → 39);
thr x0.5 r = 0.92, k = 0.94, MAE 0.6 (60 → 54). green_positive vs Clade B r = −0.35 … −0.02.
Clade A: AS-A sheet 3 (no slice-2 points stored), app green_positive 9–20; AS-B sheet 1, its one slice-2
point is matched by a green detection (1/1), app green_positive 12–20. Best fixed ordering hits only
5/22 (AS-A) and 2/18 (AS-B) fields — confirms stage 3.

FS slides:
* **FS are not the Controls rows.** Controls sheet: FS A = 0, FS B = 1 (one field each, slides dated
  13-05-2026). The FS TIFFs (scanned 2026-09-30) hold ~580–690 red-positive objects in the chamber and
  ~26 researcher points per field (FS-A 7 fields = 182; FS-B_D 6 fields = 151 + 1 slice-2 point in field 5).
  They match no sheet/CSV row; their counts are only in the TIFFs. Whatever FS is, it is strongly
  positive in the 635 channel, not a negative control on this date.
* **FS-B_D was counted on a shifted grid**: the counters are 100 % single-cell for a 316.23-px grid at
  offset ≈ (0, 124) px (x lines ≈ 316/632/948/1264, y lines ≈ 440/756/1072/1388/1704), i.e. shifted by
  ≈ (−158, −34) from the (158, 158) grid of the other seven slides; each field's points span ≈ 260–308 px.
  The overlay stored in the FS-B_D file is itself this shifted grid (origin (3, 123) when read from the file,
  see [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md)); it is the (158, 158) grid that does not describe FS-B_D's fields.
* **Saturation / density**: FS were scanned brighter (PMT 500, laser 3.30 vs 550 / 1.97); page-0 median
  in the counting region 1.6k vs 0.5k; page 0 is clipped over 64–67 % of the outside (vs 6–7 %), and the
  rim carries a band of saturated debris (stage 1). Inside the counting region saturation is still small
  (0.03 % of pixels, 50–56 clipped blobs; AI/AS 0.00–0.002 %) but 16–23 spots/slide are saturated
  plateaus merged by the app, and polonies are bigger (missed points: half-max radius 9–10 px vs 5–6 in
  AI/AS) and denser (touching pairs; `suspicious_shape` on 123–269 objects). Red debris "worms"
  (fibre-like streaks, top half of FS-A) produce many elongated detections — flagged
  `suspicious_shape`, not removed.
* The AS slides are sparse (60–82 researcher polonies) and have the same failure modes as AI/AM
  (dim polonies missed at default threshold; specks as false positives, mostly `debris_like`;
  red-channel debris warning on AS-B: 51–56 % of red spots debris-like).

## Stage 5 (2026-10-10): conclusions

### What these images required of the app (ordered by importance)

Status: items 1, 2, 3, 5 and 6 are implemented: embedded reference marks and field mapping, field import with a user-settable shift,
lab-layout field export, the red/green channel suggestion, the `*_excl_debris` counts and the automatic chamber. Behaviour and endpoints
are documented in [FIELDS.md](FIELDS.md) (export layout: [LAB_EXPORT.md](LAB_EXPORT.md)). Items 4 and 7 remain recommendations (the
per-channel sensitivity preset, and a separate review queue and larger radii for bright scans, are not implemented). The text of each
item is the requirement as derived from the data.

1. **Read the embedded researcher annotations as fields** (implemented, FIELDS.md). The ImageJ multi-point ROI is the lab's
   ground truth: one counter per field, `counter_positions` = slice (1 = page 0 = Clade B, 2 = page 1 =
   Clade A), points inside one grid cell. At the time of this analysis the app only listed embedded ROIs as metadata.
   Import them as (a) reference points for an agreement view and (b) the field → cell mapping (counter k →
   Field-(k+1); warn when the ordinal and index readings disagree, as in AI-A; infer zero-count fields by
   elimination and ask). No fixed field ordering works (best 2/18 … 19/21 correct).
2. **Grid-field export design.** Import the Overlays grid (4 x 7, 316.23 px, origin 158,158) as
   candidate counting regions with `field_index`, but let the user shift it (FS-B_D was counted on a grid
   offset by ≈ (−158, −34)) and choose which cells are fields. Export per-field counts in the sheet's layout
   (Field-1..24, blank = not counted, 0 = counted empty) with polonies/slide = mean·29 and
   polonies/ml = ·1000/volume·dilution, per channel (Clade B = red, Clade A = green) — never by grid order.
3. **Channel handling.** For 2-page ImageJ stacks from GenePix/ImageIO with `Info: 635 [W1]`, suggest
   page 0 = red/635 and page 1 = green/532 (evidence: R↔page 0 / G↔page 1 Spearman ≥ 0.995, zero shift,
   and the researcher's slice-1/slice-2 points), still requiring confirmation. Count types to offer:
   red-positive and green-positive per channel; yellow is unused on these slides. When an RGB composite
   is uploaded, say explicitly that it is a √-scaled export clipped at raw ≈ 9–16k (R² 0.998 for
   2.76·√raw) and prefer the raw stack.
4. **Detection defaults for these slides.** spot_radius 5 px (half-max radius of polonies 6–8 px; r3 is
   clearly worse). The automatic threshold is too high for the dim polonies the researcher counts:
   threshold_scale 0.5 raises recall from 0.51–0.80 to 0.73–0.91 at precision 0.77–0.94 and brings
   per-field slopes from 0.67–0.82 to 0.80–0.94 (r 0.79–0.92). Expose a per-channel sensitivity preset
   and show the threshold relative to the faint population; validate on more slides before changing the
   global default.
5. **Debris.** False positives are ≈ 1-px specks (peak 8–13k) and green fibres; 70–90 % already get
   `debris_like`. Offer a count "red-positive excluding debris_like" (or a one-click "reject all
   debris-like in this region") rather than auto-removal; the per-channel "mostly debris-like / channel may
   be empty" warning fired correctly on every green channel.
6. **Chamber mask.** A one-click counting region from page-1 clipping (stage 1.4) works on all 8 slides
   (81.5–81.8 % of the frame, all 1 330 points inside). Rim: the researcher counts to 24 px from the edge;
   a 50-px rim drops 1–9 points/slide but removes the bright gasket ring and FS rim debris — make the rim
   width a visible parameter (default 30–50 px) and show what it removes. The dark bubble (rows 5–6)
   needs an exclusion region or a "not counted" field state.
7. **Saturation/brighter scans (FS).** Merged saturated plateaus (16–23/slide) and touching big polonies
   need their own review queue; spot radius may need 7–9 px on such scans (not tested here).

### Questions for the researcher
* What decides how many fields are counted (17–22 here) and which cells are skipped (rows 5–6, r4c0/r4c1
  in AM-A)? Is the dark region at the bottom a bubble/dry area?
* Do you count only the 635 (red) channel for Clade B and 532 (green) for Clade A? Are dual-colour
  polonies counted in both, either or neither?
* What are the FS slides (they are not the 13-05-2026 Controls rows), what is `_D`, and why was FS-B_D
  counted on a shifted grid?
* AI-A/AM-A/AS-A: the Clade A counts (4/4/3) are not in the TIFF — were they counted on another file?
* Are the faintest marked spots (≈ +50–300 counts over background) polonies you want counted?

### Honest limits
8 slides from one scan session, one annotator, no repeat counts (no inter-annotator variability to
compare against). Clade A has 0–14 events per slide, too few for any per-field statistic. The channel and
field mappings are inferences from correlations and point identity (strong, but not stated by the lab).
The 15-px matching and the "counted cells" restriction are analysis choices. Only radii 3/4/5 and
threshold_scale 0.5 were tested (CPU budget); r ≥ 6 and other background settings are untested.
Stage-1 summary numbers were obtained with `stage1_facts.py`; the detection results of stages 2-4 are cached
JSON files outside the repository and can be regenerated with `run_app.py`.
