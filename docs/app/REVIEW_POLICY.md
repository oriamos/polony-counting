# Review, approval and finalize rules (backend)

* **Actions** (`POST /actions`) follow CONTRACTS.md. Every undoable action stores
  a before/after snapshot of what it touched; undo/redo stacks are persisted and
  survive restarts. Deletes are soft. `confirm`, `accept_region` mark objects
  reviewed; `move`/`reclassify`/`merge`/`split` also mark reviewed and set
  `origin=edited` on automatic objects.
* **Uncertain objects** are never part of colour totals (`red_only`, `green_only`,
  `yellow`, `red_positive`, `green_positive`, `total_objects_exclusive`); they are
  counted separately and block region approval until reclassified or deleted.
* **Counted objects**: non-deleted objects inside a counting region and outside every
  exclusion. If no counting region exists every non-deleted object counts. An object
  can only be added/moved inside a counting region when regions exist.
* **Approval fingerprint** = sha256 of (region polygon, the polygons of the
  exclusions overlapping it, ids/positions/classes of its non-deleted objects, the channel
  roles and the green->red alignment).
  After every mutation approvals whose fingerprint changed are cleared
  (add, move, delete, reclassify, merge, split, re-detection, region/exclusion edits,
  channel-mapping and alignment changes).
  `reviewed` flags are not part of it. Undo that restores an identical state restores
  a valid approval. `finalize` additionally re-checks every approval against its fingerprint.
* `accept_region` and `approve_region` require the region to be marked inspected;
  `approve_region` additionally requires zero unresolved uncertain objects, non-stale
  detection and that detection ran (or objects exist). `region_inspected=false` clears
  an approval.
* **Stale detection** (`provenance.stale_detection`) is DERIVED, never merely a stored flag:
  at detection time the engine records a signature (`AnalysisState.detection_sig`: the request
  actually used, a hash of the regions, the channel roles, the alignment). Stale <=> the
  current signature differs from the recorded one, or the parameters currently set differ
  from `provenance.detection_request`. It is recomputed in `refresh_status` after every
  mutation including undo/redo (undo cannot erase staleness caused by a non-undoable settings
  change), a settings change while a job ran leaves the result stale, and setting the
  parameters / regions / alignment back to the used ones clears it. `finalize` checks the
  request equality explicitly in addition.
* **Channel mapping / alignment change** (any time something depends on it: a detection result,
  any object including manual ones, any approval or inspection mark - detection need not have
  run): all approvals and inspection marks are cleared, the undo/redo history is dropped,
  the event (with the cleared approvals) is logged. A confirmed red<->green swap also mirrors
  object classes and evidence (see RERUN_POLICY.md).
* **Status while a job runs**: `AnalysisState.job_active` (persisted; set/cleared by
  `refresh_status(job_running=True/False)`). While set, edits cannot move `detecting` to
  `in_review`; orphan recovery at start-up clears it.
* **Finalize** requires >= 1 counting region, all counting regions approved, confirmed
  channel mapping, non-stale detection. After finalize every mutating endpoint (actions,
  undo/redo, regions, channels, alignment, settings, detect) returns 409
  `analysis_finalized`; exports are labelled FINAL.
* **Reopen** (`POST /api/analyses/{id}/reopen`): status returns to `in_review`,
  `finalized_at` is cleared, **approvals are kept** (their fingerprints still protect
  them: any later edit inside an approved region clears just that approval, so
  re-finalizing needs those regions re-approved). Finalize and reopen are in the action
  log / provenance summary. The undo history is not restored.
* **Reopen** is logged as `ActionType.reopen`; alignment lives on `Analysis.alignment_dy/dx`;
  upload pairing hints (`red_file`/`green_file`) only set `ChannelInfo.suggested_role` - `role`
  stays None and unconfirmed until `PUT /channels`.
* **Concentration / lab abundance** is gated first by `SlideMeta.concentration_formula_confirmed`
  (default false; it now means "field geometry and series mapping confirmed by the researcher").
  Unless it is true AND `sample_volume_ul`, `field_area`, `grid_area`, at least one `lab_series`,
  at least one counting region with a `field_index` (1..24, unique) and a confirmed channel mapping
  are present, every estimate cell is empty and `concentration_status` lists what is missing
  ("inputs incomplete: ..."). The formula is the researcher's own (T7 sheets):
  `abundance = mean(field counts) * (field_area/grid_area) / sample_volume_ul * 1000`, mean over
  the fields that exist (blank = no region, 0 = region with zero objects). The concentration
  factor is **not** part of it (it is exported as a column; the Counts-style workbook multiplies
  by it in AK). The count type of every series (red_only / green_only / yellow / red_positive /
  green_positive) is chosen by the user in `lab_series`, never inferred. `sample_total_area_px2`
  is no longer used. No confidence interval is ever exported.
* **Lab export** (`GET /api/analyses/{id}/lab-export`, `POST /api/lab-export`; details in
  `LAB_EXPORT.md`) works on FINALIZED analyses only (409 `not_finalized` otherwise) and answers 422
  `lab_export_incomplete` with the exact missing items per analysis. A field region that exclusions
  cover partly is counted over the remaining area only and flagged in the Notes sheet; a field
  that is covered completely is left blank and flagged. Counting regions without a `field_index`
  are not in the lab export (warned).
