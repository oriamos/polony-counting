# Lab-format export & lab formulas

Derived from the lab's template workbooks (`tests/test_data/lab_templates`): `T7A_template.xlsx`, `T7B_template.xlsx`
(+ CSVs) and `Counts_template.xlsx` (formulas read with openpyxl `data_only=False`).

## T7-style table (one sheet / CSV per count series)
Header row (exact text, including the trailing space after `POLONY DATE` and the typos):
`slide ID | SAMPLING DATE | POLONY DATE␠ | CONCENTRATION FACTOR | SAMPLE VOLUM[ul] | field [Pixel^2] | grid [Pixel^2] | QC | FAGE ABUNDENCE [fage mL-1] | field 1 … field 24`
Columns A–AG. One row per slide. Dates are real Excel dates (XLSX) / `dd/mm/yyyy` (CSV).
Column I formula (row n): `=AVERAGE(Jn:AGn)*(Fn/Gn)/En*1000` — AVERAGE ignores BLANK cells and counts zeros.
(The lab's own formula does NOT multiply by the concentration factor in column D.)
CSV: same header, abundance as the computed number (full float precision), blanks empty.
Fields: counting regions with `field_index` 1..24; no region ⇒ blank cell; region with 0 objects ⇒ 0.
Template sheet names: series name (e.g. `T7B`). File name: `<series>_<sampling dd-mm-yyyy>.xlsx|csv`.

## Counts-style workbook (`Counts_<polony date>.xlsx`, one sheet per series, `sheet_title`)
Per sample pair (replicates A,B; paired by `slide_id` = `<sample>_<A|B>`), 2 rows per slide:
- header row: C=sampling date, D=replicate letter, F=`=SUM(H{v}:AE{v})`, H..AE=`Field-1`..`Field-24`
- value row v: G=29 (`=field_area/grid_area` value, written as number), H..AE = field counts (blank stays blank),
  AG=`=AVERAGE(H{v}:AE{v})`, AH=`=AG{v}*G{v}` (polonies/slide), AI=dilution (concentration_factor),
  AJ=sample volume µL, AK=`=(AH{v})*1000/AJ{v}*AI{v}` (polonies/ml),
  on replicate A: AL=`=100*ABS(AH{vA}-AH{vB})/AVERAGE(AH{vA},AH{vB})`, AM=`=AVERAGE(AH{vA},AH{vB})`,
  AO=`=100*AM/{efficiency_input}` only if efficiency_input set.
- NTC: if a finalized analysis with slide_id starting `NTC` exists in the batch: AN=`=AM-<NTC AH>`, AQ=`=AH-<NTC AH>`; else leave blank (NEVER write `#REF!`).
- Template column headers row 4: `average number of polonies/grid | polonies/ slide | dilution | sample volume/ slide | polonies/ ml | %v diff btw slides | Average | Average - NTC | Efficiency average | Efficiency average-NTC | pol/slide -NTC`.
- Lab constants seen in the template (NOT defaults): volume 2.5056 = 5.4*11.6/25 (CSV uses 2.5); 29 = field/grid; 1500/1000 efficiency denominators.

## App rules
- Lab export available only for FINALIZED analyses (batch endpoint refuses drafts, listing blockers).
- Needs: `lab_series` (user-chosen count type per series: never inferred from T7A/T7B or colours), counting regions with `field_index`, sample_volume_ul, field_area, grid_area, concentration_factor, slide_id, sampling_date, polony_date, qc. Missing ⇒ 422 listing missing items.
- A field region partially covered by exclusions is flagged in a Notes sheet / README (count is over the remaining area).
- Fields > 24 refused (template columns end at field 24) with a clear message.
- Concentration in slide.csv: `abundance = mean(field counts) * (field_area/grid_area) / sample_volume_ul * 1000` (the lab formula, no concentration factor, as in the T7 sheets); `concentration_formula_confirmed` means "field geometry and series mapping confirmed". `sample_total_area_px2` is not used.
- Endpoints: `GET /api/analyses/{id}/lab-export?series=<name>&format=xlsx|csv` (single slide), `POST /api/lab-export` body `{analysis_ids:[...], style:"t7"|"counts", series:"T7A"|null, format:"xlsx|csv"}` → file or zip.
- UI: grid-field generator (rows × cols, origin, cell size, numbering order), field index editing per region, lab metadata form, series mapping, Lab export panel (multi-select finalized analyses).
