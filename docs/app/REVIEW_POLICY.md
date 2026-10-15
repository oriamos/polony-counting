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

## Two-engine review (Big-FISH + Spotiflow)

Contract: `TWO_ENGINE_CONTRACT.md`. **Agreement is not truth; a model score is not a probability;
nothing is ever auto-rejected or auto-deleted because one engine missed it.**

* **Candidate status** lives on the channel evidence (`ChannelEvidence.consensus_status`):
  `agreement | bigfish_only | spotiflow_only | human_added`. `human_added` is set on the channel
  evidence of EVERY manual `add` (red -> red evidence, green -> green, yellow -> both), whichever
  engine ran, because it is simply true; `detected` stays false (no engine saw it).
  `human_rejected` is **derived**, never stored: a candidate whose object was soft-deleted with the
  `delete` action (tombstone, `AnalysisState.user_deleted`). Deletions caused by merge/split are not
  rejections. Undoing the delete removes the derived rejection.
* **Queue order** (`review/counts.py`, lower first): ambiguous_match 0, saturated 1,
  **bigfish_only / spotiflow_only 2**, weak_signal / near_threshold 3, debris_like 4, dense_cluster 5,
  suspicious_shape 6, near_edge 7, **qc_sample 8**. Unflagged agreements are not queued. Exception: an object that
  is both `bigfish_only` and `debris_like` is queued at priority 1 (with saturated): on 7 research slides 97 % of
  these (333 of 345) had no researcher mark within 10 px, against 18 % of the other Big-FISH-only objects (size /
  compactness separates specks from real dim spots better than disagreement alone). Never auto-removed.
* **QC sample** `POST /consensus/qc-sample {n, seed, fraction?}`: a FIXED-SIZE sample, not a fraction. A 5 %
  sample cannot bound the agreement error (pooled agreement false-positive rate 0.153; 5 % of 24-296 agreements
  is 1-15 objects). Eligible = counted, non-deleted, unreviewed objects with at least one detected channel whose
  every detected channel evidence is `agreement`. `n` defaults to 40 and is clamped to `[1, N_eligible]`; the
  sample is the `n` eligible objects with the smallest `sha256("<seed>:<object id>")` (first 8 bytes / 2^64), a
  pure function of (seed, id) (independent of object order, process and RNG state; a larger `n` with the same seed is
  a superset). `fraction` is kept for compatibility and maps to `max(40, ceil(fraction * N_eligible))`; an explicit
  `n` wins. It only ADDS `ReviewReason.qc_sample` (idempotent; the response reports `marked`, `newly_marked`,
  `eligible`, `n`, `n_requested`, `population_agreements`, the `qc` status below, and a note). One undoable action,
  logged as `region_edit` with `payload.event == "qc_sample"` (no new ActionType). Draw it once per slide with one
  seed: the interval below is valid for a single random sample.
* **QC result** (`qc` in the sample response, `GET /consensus/summary` and `provenance.json` `agreement_qc`): among
  the sampled agreements, `confirmed` (reviewed, kept) vs `rejected` (deleted by the user), the error rate
  `rejected / (confirmed + rejected)` and its exact **Clopper-Pearson 95 % interval** (null until something is
  reviewed), the sample size, and the sample sizes needed for a margin of +/-0.05 on this slide's agreement
  population N: `n = ceil(N n0 / (N + n0 - 1))`, `n0 = 1.96^2 p (1 - p) / 0.05^2`, with the measured p = 0.153
  (N = 24, 49, 67, 203, 296 -> 22, 40, 51, 101, 120) and the conservative p = 0.5. The interval bounds the error
  rate of the UNSAMPLED agreements only under random sampling and only for agreements present when it was drawn;
  it says nothing about spots both engines missed. Agreement is not ground truth.
* **Summary** `GET /consensus/summary` -> `{pages:[ConsensusSummary per red/green page], total, qc}`.
  Per channel: the four status counts are LIVE candidates; `human_rejected` = `rejected` = user-deleted
  candidates (their original engine status is not counted again); `confirmed` = live and reviewed
  (class not uncertain); `unresolved` = live and (unreviewed or uncertain). Candidates are channel
  evidence that an engine detected or a human added, so status-less legacy evidence counts in
  confirmed/unresolved only. `total` is the plain sum; channels are never merged silently.
* **Audit**: engine runs and raw detections are stored outside the Analysis JSON, appended per detect
  job and never deleted by re-detection (`GET /engine-runs`, `GET /raw-detections`). A failed Spotiflow
  run in hybrid mode is stored (`status=failed`, error) and surfaced as a WARNING in the job message;
  the job still succeeds with the Big-FISH result. Staleness is derived from the whole
  `DetectionRequest` (engine choice, Spotiflow params incl. model name, consensus tolerance).
