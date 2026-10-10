# Scientific open questions

These are **decisions the researcher has to make or confirm**. They are not software defects: the app deliberately refuses to guess any of them (it exports raw counts and leaves
concentration cells empty until the inputs are confirmed). Software problems and legacy-code problems are in [LEGACY_AUDIT.md](LEGACY_AUDIT.md). Detailed background and numbers: [SCIENCE_NOTES.md](SCIENCE_NOTES.md).

## Data limitation (read first)

* The research TIFFs named in the task (`29-09-2026_T7_AB_*.tif`, e.g. `..._AM-A.tif`, described as 1460 x 2438) were **NOT available** in the workspace. Nothing below could be checked against the pixels of those slides.
* The only image is `resources/dataset_files/test/test_img.tif`: 2450 x 1438 (the brief says 1460 x 2438; the dimensions differ), 2 pages, uint16, **red-only content** (page 1 has no signal at the marked polonies). One annotator's 77 embedded point marks serve as reference.
* The CSVs (`T7A_15-09-2026.csv`, `T7B_15-09-2026.csv`), their xlsx versions and `Counts_30-09-2026.xlsx` were supplied as uploads; they are not part of the repository.
* Consequently there is **no accuracy validation** of the detector (see item 11) and no validation of yellow or green calls.

## Evidence from Counts_30-09-2026.xlsx (unconfirmed interpretation)

Read by the orchestrator; it is an interpretation of a spreadsheet, not confirmed by the researcher, and not checked against any image.

1. Sheets `Clade B`, `Clade A`, `Controls`. Title "T7 AB assay", "SLIDES FROM 29-09-2026, HYBRIDIZATION 30-09-2026". The `Controls` sheet is headed "Acr-C1-84 /C1-288 10/15 uM", slides from 13-05-2026 (FS, NTC).
2. The column header reads "Cy5 red & Cy3 green (yellow counts)". Columns: assay; Phage & dilution (AI, AS, AM, PI, PS, PM, HI, HS, HM, QI, QS, QM); Slide number (A/B); `yellow`; "summ / slide"; "well area/grid area" (= 29, i.e. 2.9 / 0.1); "total yellow polonies / grid"; then Field-1 ... Field-N.
3. The field values of `Clade B` equal the T7B CSV rows exactly (e.g. AI A: 13, 29, 15, 15, ...; sum 332) and `Clade A` equals the T7A CSV (AI A: 1, 3, 0, ...; sum 4). So **T7A / T7B most plausibly mean Clade A / Clade B counts of the T7 AB assay, not red / green colours**, and the A/B slide letter would be the replicate. Unconfirmed: ask the researcher.
4. The sheet header calls the counted quantity "yellow counts" (Cy5 red & Cy3 green). The reported quantity per clade may therefore be the **dual-positive (yellow) polonies per grid field**, which makes the yellow-convention question (item 1) central: is the value per clade yellow only, or are red / green / yellow reported separately?
5. A separate `yellow` column holds small values (e.g. 3 for AI/A in both sheets, 0 for AS/A and AM/A) of unknown meaning, perhaps a per-slide manual count of yellow objects. Ask.
6. Field = grid cell; "well area/grid area = 29". `test_img.tif` has a 4 x 7 = 28-cell grid in its Overlays (316.23 px spacing) while the sheets list Field-1 ... Field-N with varying N (blank = not counted). The exact cell numbering and which cells are valid are still unknown.
7. File names such as `29-09-2026_T7_AB_AM-A.tif` presumably read: phage T7, assay "AB", sample AM, slide A; `FS-A` / `FS-B_D` come from the Controls sheet (FS = ?, NTC = no-template control).

These items are **questions, not facts**.

## Questions

1. **Yellow reporting convention.** Is a yellow polony (red and green signal) reported as its own category (exclusive: red-only, green-only, yellow) or counted in both channels (inclusive: red+ = red-only + yellow, green+ = green-only + yellow)? The app exports both, always labelled, and defaults to exclusive for the display only. *Unconfirmed.* See also point 4 of the xlsx evidence.
2. **Meaning of T7A / T7B.** Clades A and B of one assay (xlsx evidence), two targets with two dyes, or two replicates? The app never maps names to colours; the CSV colour mapping must be set by hand.
3. **Field geometry and numbering.** The legacy code generates 28 squares (ids 0-27, a 4 x 7 grid starting at pixel 158 with 316 px squares) but its export reads ids 1-24, and the CSVs have fields 1-24 (up to N). Which squares are eligible (fully inside the chamber, off the rim, not in a dark crescent)? In what order are they numbered? Are partial squares counted? On test_img 17 of 28 squares contain marks. The app uses free-form counting regions instead of a fixed grid.
4. **Blank versus zero.** The CSV formula excludes blank cells from the mean and includes zeros (blank = "not counted", 0 = "counted, none"). The app preserves the distinction on import. Is this the intended meaning, and what decides how many fields are counted (6-24 per row)?
5. **Units "Pixel^2" of 2.9 and 0.1.** The headers say `field [Pixel^2]` / `grid [Pixel^2]`, but 2.9 and 0.1 are almost certainly cm^2 (the embedded grid is 316.23 px = 0.316 cm at 1000 px/cm, so 0.100 cm^2; the chamber is about 2.8 cm^2). Is "field" the whole gel/chamber and "grid" one counted square?
6. **QC = 0.** Rows in T7B have QC = 0 (and an abundance is still computed). What does QC mean, and are such rows excluded from summaries? The app does not use QC.
7. **Concentration formula, geometry and factor.** The CSV reproduces `abundance = mean(non-blank fields) * (field / grid) / SAMPLE VOLUME[ul] * 1000` (here `mean * 11600`). The app proposes an area-scaled formula that needs the total sample area; it is flagged UNCONFIRMED and gated by `concentration_formula_confirmed`. Open: does the sample volume spread over exactly the "field" area; is mean-per-square extrapolation or count / counted-area preferred (different for partial squares)? Does the **concentration factor** multiply or divide, and is it applied before or after the volume (it is 1 in all supplied rows; the legacy code uses 1.1 for T4 and 1.0 for T7/T7c).
8. **Sample volume: 5 uL (legacy) versus 2.5 uL (CSV).** The legacy code hardcodes 5.0; the supplied CSVs use 2.5 (with 5.0 the factor would be 5800 instead of 11600). Which historical files used which value?
9. **Pixel size.** The TIFF tag says 1000 px/cm (10 um/px) with unit cm. This is **unconfirmed** and the app does not apply it; distances, radii and areas are in pixels. Is it the true scanner pixel size?
10. **Spot radius choice.** The default sigma of 5 px comes from one image and one scanner setting; other slides may be larger (about 8-12 px in screenshots, not measured). The count depends strongly on it (radius 5 -> 108 objects, 3 -> 207, 2 -> 331 on test_img). Who chooses it per slide and by what criterion? The app asks you to choose it by looking at a preview.
11. **Debris criteria.** The debris-like flag (compact and sharper than a typical polony; threshold ratio 1.0) is an unvalidated heuristic. On test_img it separated many unmarked bright specks from marked polonies, and every would-be yellow pair was compact and bright in both channels. Can a real polony be that small (early colonies)? Is there a minimum size? Are polonies in the dark crescent, near the rim or on worm-like artefacts counted?
12. **No accuracy validation.** One red-only image, one annotator, parameters tuned on the same image (optimistic). The manual marks are imprecise (median 6.5 px from the nearest LoG maximum), so reported recall/precision on it is only indicative. No recall/precision, false-yellow rate or inter-annotator variability is known for any real green or yellow data.
13. **Bleed-through and registration.** Red->green bleed-through on test_img was negligible (ratio about 0.0004), but green->red could not be measured, and bleed-through at other PMT/laser settings is unknown. Red/green registration residual was estimated at about 1-3 px with an unreliable phase-correlation estimate; the match tolerance of 6 px depends on it. A per-slide bleed-through / registration check is recommended but not implemented.
14. **Which page is red?** `test_img.tif` carries one wavelength label (`635 [W1]`) for two pages; the mapping could not be confirmed from the data, so the app asks you each time.
15. **Legacy 5 uL, T7c.** T7c efficacy and its concentration factor are not implemented in the legacy code (see LEGACY_AUDIT.md); the biology is the researcher's to specify.
