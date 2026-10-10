# Lab fields, reference marks and the automatic chamber

Backend: `src/polony_app/fields/` (reference, grid, service), `src/polony_app/imaging/chamber.py`, routes in `api/main.py`.
Background and numbers: [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md). The ImageJ ROI reader is the one of
`polony_app.legacy_import.roi`; canonical coordinates = ImageJ value - 0.5 (sub-pixel points, grid edges).

## Reference marks (`GET /api/analyses/{id}/reference`)
Read-only view of what the TIFF carries: the multi-point ROI (ImageJ *counter* = field, *position* = plane/page, 1-based in
the file, 0-based `plane` in the API) and the Overlays line grid. Only two-page ImageJ stacks are read. Response:
`available`, `source_file`, `points[{x,y,counter,plane,field,field_ordinal}]`, `grid{source,origin,cell,cols,rows,cells[{field,polygon,row,col,mapped}]}`,
`reference_counts{"<field>":{"plane0":n,"plane1":n}}`, `planes[{page,point_count}]`, `warnings`; `reason` when unavailable.
Which plane is which colour is never stated here (validated stacks: plane 0 = red/635 = 'Clade B', plane 1 = green/532 = 'Clade A';
inferred from composite correlation and sheet agreement, not confirmed by the lab).

**Field numbering.** `field = counter + 1` ('index', default). On the 8 validated slides this equals the sheet's Field-N order for
every slide except AI-A, where counter 3 is absent and the sheet continues without a gap: there `field_ordinal` (rank among the counters
present + 1) reproduces the sheet exactly (checked on the real AI-A file: 13,29,15,15,16,25,23,15,12,15,22,16,15,22,13,19,23,24).
A missing counter is either a zero-mark field (AM-B) or a skipped number (AI-A); the file cannot tell, so a warning is given whenever counters
are not contiguous. `numbering: "index" | "ordinal"` is accepted by import and compare.
`cells[].field` = the field whose marks fall in that cell, other cells take the lowest unused numbers in reading order (`mapped:false`).

## Field regions (`POST /api/analyses/{id}/fields/import`)
Body `{mode: counted|all, shift:[dx,dy]=[0,0], plane: int|null, replace: bool=false, numbering}`. Creates counting regions `Field-N`
(`field_index=N`) from the grid cells, shifted by `shift` and clipped to the image. `counted`: only fields that have marks on `plane` (any plane
if null), placed in the cell holding most of their marks. `all`: every cell (fields keep their mapped number, the rest are numbered
in reading order; cells numbered above 24 are named `Cell-N` without a field number because the lab template has 24 fields).
**Ambiguity:** a field without marks may be 'not counted' or 'counted, zero' - the file cannot say; the response warns about it.
Existing `Field-*`/`Cell-*` regions: 409 `fields_exist` unless `replace=true`. Goes through `engine.set_regions`: finalized -> 409,
replaced regions lose approval/inspection, objects are re-assigned. Field regions are placed *before* other regions: when regions overlap
the first listed counting region owns the object, so a 'Chamber' region must come after the fields (done here). 422 when the image has no grid.
Response = the Analysis plus `field_import{created, warnings, ...}`.
Shift: the stored overlay does not always describe what was counted (FS-B_D was counted on a grid ~(-158,-34) px away); marks are re-assigned
to cells of the shifted grid, so a wrong shift shows up as warnings ('marks lie in another cell').

## Comparison (`GET /api/analyses/{id}/fields/compare?plane=&count_type=`)
Per field region (counting regions with a `field_index`): reference count (marks of `plane`) vs the app's non-deleted objects of
`count_type` (red_positive, green_positive, red_only, green_only, yellow), `delta = app - reference`; `summary{n_fields, pearson_r, slope
(least squares through 0), mae, total_reference, total_app}`. Neutral: the marks are the researcher's counts, not a measured truth, and
differences can come from either side. Fields without marks are compared as 0 and listed in `warnings`. 422 without field regions.

## Automatic chamber (`POST /api/analyses/{id}/regions/auto-chamber`, preview only)
`{rim_px=40}` -> `{polygon, area_px2, rim_px, method, warnings, suggested_exclusions[{polygon,area_px2,reason}]}`; nothing is stored (apply it
with `PUT /regions` as a counting region 'Chamber'). Method: the scanner background outside the chamber is clipped in one channel (page 1:
~18 % of the frame); the channel with the most clipped pixels (2-90 %) gives the outside, chamber = largest inside component, eroded by
`rim_px` (Euclidean). Deterministic, ~3 s on 2438 x 1460. 422 `chamber_not_found` when no clipped background exists. Bubble suggestion: 16-px blocks
whose 20th-percentile background is below 60 % of the chamber median, >= 1 % of the area; a hint only. On the 8 real slides: chamber 81.5-81.8 % of
the frame, 95-100 % of the researcher's marks inside the 40-px counting area, bubble suggestion at the bottom rows (0-3 marks inside).

## Other additions
* Ingest: a two-page ImageJ stack whose Info hint contains '635' gets `suggested_role` red (page 0) / green (page 1); `role` stays null and
  `role_confirmed` false; a warning says the suggestion comes from 8 validated GenePix stacks and that RGB composites are clipped.
* `Counts.red_positive_excl_debris` / `green_positive_excl_debris` = positive count minus non-deleted objects flagged `debris_like`
  (yellow counts in both); in `regions.csv` / `slide.csv`. Convenience count; flagged objects remain counted elsewhere.
