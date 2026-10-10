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
  exclusions overlapping it, ids/positions/classes of its non-deleted objects).
  After every mutation approvals whose fingerprint changed are cleared
  (add, move, delete, reclassify, merge, split, re-detection, region/exclusion edits).
  `reviewed` flags are not part of it. Undo that restores an identical state restores
  a valid approval.
* `accept_region` and `approve_region` require the region to be marked inspected;
  `approve_region` additionally requires zero unresolved uncertain objects, non-stale
  detection and that detection ran (or objects exist). `region_inspected=false` clears
  an approval.
* **Stale detection** (`provenance.stale_detection`) is set when, after a detection ran,
  any region changes, the channel mapping changes, the alignment changes, or the
  detection parameters in settings differ from those used. Mapping/alignment changes
  also clear all approvals and the undo history.
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
* **Concentration** is gated first by `SlideMeta.concentration_formula_confirmed` (default
  false). Unless it is true AND `sample_volume_ul`, `concentration_factor`, a counting area
  (net of exclusions and of overlaps between counting regions), `sample_total_area_px2` and a
  confirmed mapping are all present, every concentration cell is empty and
  `concentration_status` lists what is missing ("inputs incomplete: ..."). The built-in
  formula is only a PROPOSAL, flagged UNCONFIRMED in the export README and the CSV:
  `conc_per_ml = (n_counted / counting_area_px2) * sample_total_area_px2 /
  (sample_volume_ul/1000) * concentration_factor` (assumes the counted area is representative
  of the whole sample area; the same count in a 4x larger counting area gives 1/4 of the
  concentration). The legacy convention
  `abundance = mean(fields)*(field/grid)/volume_ul*1000` is quoted for the researcher to compare.
  No confidence interval is ever exported.
