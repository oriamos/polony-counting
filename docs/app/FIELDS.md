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
**Ambiguity:** a field without marks may be 'not counted' or 'counted, zero' - the file cannot say. Such a field/plane is 'unknown' and is excluded from the comparison (see below); the response warns about it.
Existing `Field-*`/`Cell-*` regions: 409 `fields_exist` unless `replace=true`. Goes through `engine.set_regions`: finalized -> 409,
replaced regions lose approval/inspection, objects are re-assigned. Field regions are placed *before* other regions: when regions overlap
the first listed counting region owns the object, so a 'Chamber' region must come after the fields (done here). 422 when the image has no grid.
Response = the Analysis plus `field_import{created, warnings, ...}`.
Shift: the grid at the default origin (158, 158) does not describe every file. FS-B_D was counted on a grid shifted by about (-158,-34) px relative to the
other slides, and the overlay stored in that file is itself that shifted grid. `shift` moves the grid; marks are re-assigned
to cells of the shifted grid, so a wrong shift shows up as warnings ('marks lie in another cell').

## Comparison (`GET /api/analyses/{id}/fields/compare?plane=&count_type=`)
Per field region (counting regions with a `field_index`): app count (non-deleted objects of `count_type`: red_positive, green_positive,
red_only, green_only, yellow) next to the researcher's reference, but **only for field/plane pairs the researcher provably examined**.
The rule is applied independently to every (source TIFF, field, plane) tuple, with the shared definition in `fields/examined.py`:

* **examined** = embedded ImageJ marks on THAT plane in THAT field (`positive_marked`), or an entry of the examined-fields manifest /
  sidecar stating the count of that field on that plane, including a counted zero (`verified_zero`, reference 0 ONLY then);
* a field without marks on the plane is `unknown` (reference `null`, never 0); marks on the other plane, the grid, and the app's own
  detections are never evidence. A field imported because it has marks on page 0 is therefore unknown on page 1 unless page 1 has marks there;
* a manifest count that differs from the number of marks (or a stated zero where marks exist) is a `conflict`: excluded, not repaired;
  an entry that names the field without a count and there are no marks stays `unknown`.

Rows: `{field, region, status, examined, examination_source, reference_count (null if not examined), app_count, delta (null if not
examined), has_reference_marks, reference_marks, reason}`. `summary`: `n_fields` (all field regions), `eligible_examined_fields`,
`positive_marked_fields`, `verified_zero_fields`, `unknown_fields_excluded`, `conflict_fields_excluded`, `eligibility_source`
(`embedded_imagej_marks`, `embedded_imagej_marks+sidecar_csv`, ... or `none`), and over the eligible fields only `pearson_r`, `slope`
(least squares through 0), `mae`, `total_reference`, `total_app`; all five are `null` when no field is eligible (with a warning), never
0/NaN. `app_total_unknown_fields_not_counted` shows what the app found in excluded fields; it enters no total. Neutral: the marks are
the researcher's counts, not a measured truth, and differences can come from either side. 422 without field regions.

### Examined-fields manifest (`POST /api/analyses/{id}/examined-fields`, multipart `file`)
JSON (`schema: polony-examined-fields`, v1) or CSV (`# evidence: ...` first line; columns `file,page,counter|field|row+col,count,counted_zero,note`),
see `fields/examined.py`. It is validated (422 `examined_manifest_invalid` when malformed, `examined_manifest_mismatch` when its sha256 differs
from the file), stored next to the originals as `examined_fields.json` (replaces a previous one; 409 when the analysis is finalized) and
returns `{matched_file_entries, pages: {page: n_entries}, n_entries, warnings, stored_as, evidence}`. The file entry is found by the stored or
uploaded file name, or by sha256; if several file entries match the same file with DIFFERENT content the manifest is refused (422), never
"last wins". Entries are resolved to a Field-* region by `field` (= the lab field number; trusted only because the manifest author stated it),
`counter` or `row`+`col` (cell of the overlay grid, numbered as in `/reference`). A `counter` entry resolves ONLY through that counter's actual
points in the TIFF; a counter with no point stays unresolved (warning naming the entry) and is never guessed as field counter+1, because lab
sheets number fields without gaps and an absent counter (AM-B: 3 and 19) would shift every later field. Unresolved entries are warnings in
`compare`, not silently dropped. Every later `compare` reads the stored manifest.
Strict schema: integers must be real integers (`true`, `1.5`, `"3"` rejected), `counter/field/row/col` >= 0, `count` 0..1,000,000, unknown keys
(manifest, file entry, field entry, CSV column) are errors; files > 2 MB (endpoint; 8 MB for the CLI loader) and JSON nested deeper than
32 levels are rejected with 422, never a 500.

### Marks without a plane
A mark whose ImageJ position is missing, 0 ("all planes") or beyond the file's planes has `plane: null` in `/reference` and is evidence for NO
plane (`reference_counts[field].unplaned` counts them; `unplaned{count,fields}`). In `compare`, every field containing such a mark is `conflict`
on ALL planes ("mark with unspecified plane inside this field"), even when a manifest states a count, and a leading `WARNING:` plus
`summary.unplaned_marks` / `unplaned_mark_fields` are returned (shown as an alert in the UI). The training import drops these marks too.

## Automatic chamber (`POST /api/analyses/{id}/regions/auto-chamber`, preview only)
`{rim_px=40}` -> `{polygon, area_px2, rim_px, method, warnings, suggested_exclusions[{polygon,area_px2,reason}]}`; nothing is stored (apply it
with `PUT /regions` as a counting region 'Chamber'). Method: the scanner background outside the chamber is clipped in one channel (page 1:
~18 % of the frame); the channel with the most clipped pixels (2-90 %) gives the outside, chamber = largest inside component, eroded by
`rim_px` (Euclidean). Deterministic, ~3 s on 2438 x 1460. 422 `chamber_not_found` when no clipped background exists. Bubble suggestion: 16-px blocks
whose 20th-percentile background is below 60 % of the chamber median, >= 1 % of the area; a hint only. On the 8 real slides: chamber 81.5-81.8 % of
the frame, 95-100 % of the researcher's marks inside the 40-px counting area, bubble suggestion at the bottom rows (0-3 marks inside).

## Generated grid (`PUT /api/analyses/{id}/grid`)
The grid made with **Generate grid…** is stored as `slide.grid` (`GridSpec`: `rows, cols, cell_w, cell_h, origin_x, origin_y,
order (row-major|col-major), corner (tl|tr|bl|br), first_index, seed, shift_dx, shift_dy`). Cell polygons are derived from it
(`fields/grid.py spec_cells`, mirrored by `frontend/src/lib/labgrid.ts`): effective origin = origin + shift, not clipped.
Body `{grid, randomise=false, seed=null, replace=true, force=false}`, response = the analysis. The call stores the spec and
rebuilds the lab field regions in one undoable action (undo also restores `slide.grid` and `slide.skipped_fields`):
a field that exists keeps its region id (approval survives only if the polygon is unchanged, via the approval fingerprint),
a switched-off field stays off with its remembered geometry moved along, new numbers are created on. `replace=false` keeps
lab fields whose numbers are not in the grid. `randomise=true` (grid omitted = the stored one): the server draws a seed
(or uses `seed`) and sets `shift_dx/dy` = whole pixels, uniform over the room left between the grid and the image
border ([-0.5, W-0.5]), 0 on an axis where the grid is larger than the image; generator = mulberry32, identical in Python
and TypeScript. `grid=null` removes the grid's field regions and skipped fields and clears the spec (409
`grid_fields_approved` while one is approved, unless `force`). 422 `invalid_grid` (sizes, field numbers > 24),
409 `analysis_finalized`. `PUT /settings` cannot change `slide.grid`.

Provenance: `provenance.json` -> `slide.grid` holds the spec; a randomised grid is reproducible from `origin_x/y`, `seed`
and the image size (`random_shift(spec, W, H, seed)`).

## Other additions
* Ingest: a two-page ImageJ stack whose Info hint contains '635' gets `suggested_role` red (page 0) / green (page 1); `role` stays null and
  `role_confirmed` false; a warning says the suggestion comes from 8 validated GenePix stacks and that RGB composites are clipped.
* `Counts.red_positive_excl_debris` / `green_positive_excl_debris` = positive count minus non-deleted objects flagged `debris_like`
  (yellow counts in both); in `regions.csv` / `slide.csv`. Convenience count; flagged objects remain counted elsewhere.
