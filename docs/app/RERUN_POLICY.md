# Re-detection merge policy

What happens to existing objects when detection is run again
(`POST /api/analyses/{id}/detect`, implemented in
`polony_app.review.engine.apply_detection_result`).

## Classes of existing objects
| Existing object | On re-run |
|---|---|
| `origin=auto`, not reviewed, not deleted | **replaced** by the new automatic results |
| `origin=manual` (added by user), `origin=edited` (moved / reclassified / merged / split) | **preserved untouched** |
| `origin=auto` and `reviewed=true` (confirmed / accepted by the user) | **preserved untouched** |
| soft-deleted by the user (`delete` action), **any origin** (auto, reviewed, edited, manual) | kept as a *tombstone* (history) and **suppresses** new detections at its position and at the positions of its detected evidence (so a moved-then-deleted object still blocks the spot where it was detected) |
| soft-deleted by merge/split | kept for history, no effect (not a tombstone) - unless the merged object is later deleted by the user, see next row |
| a user-deleted **merged** object | the delete also records every (transitive) constituent of the merge ("merged from" history) as an explicit tombstone anchor in the analysis bookkeeping (`user_deleted`), so a rerun does not bring back ANY of the spots the merged object was made of (positions and evidence positions). Snapshotted with the delete action: undo/redo/restart safe |

**How "deleted by the user" is recorded.** `DetectedObject` has no field for it and the
schema is shared, so the engine keeps an explicit marker in its own bookkeeping:
`AnalysisState.user_deleted` (list of object ids, persisted with the state). The `delete`
action appends the id; the list is part of every action's undo/redo snapshot, so undoing a
delete removes the marker and redoing restores it. Tombstones are exactly the soft-deleted
objects whose id is in that list - no history-string parsing. (Analyses persisted before
this change have no markers; their earlier deletes do not suppress re-detection.)

## Merge rule
A new automatic object is **dropped** when it lies within **3 px**
(`RERUN_MERGE_RADIUS_PX`, Euclidean distance between object positions) of

* a preserved object (manual / edited / reviewed), or
* a tombstone (an auto object the user deleted).

It is also dropped if it falls outside every counting region (or inside an
exclusion) when counting regions exist. All numbers are recorded in the
`rerun_detection` entry of the action log (`replaced_auto_objects`,
`new_auto_objects`, `preserved_objects`, `dropped_near_preserved`,
`dropped_near_deleted`, `dropped_outside_valid_area`) and in `provenance.json`.
New objects whose id collides with a preserved id are given a fresh id
(`ambiguous_with` references are remapped).

## Approvals
* `region.approved` is cleared for every counting region whose approval
  fingerprint no longer matches (see REVIEW_POLICY.md) - i.e. whenever the set,
  positions or classes of its non-deleted objects changed.
* `region.inspected` is reset to `false` for a region whose objects changed:
  an earlier "I inspected this region for omissions" no longer describes the
  new result.
* A re-run that reproduces exactly the same objects in a region (same ids,
  positions, classes) leaves that region's approval valid (nothing changed).
* A re-run records the request actually used, time, package versions and the input signature
  (request, regions, channel roles, alignment). `provenance.stale_detection` is derived by comparing
  that signature with the CURRENT settings, so if the settings were changed while the job ran the
  result is stale (see REVIEW_POLICY.md). It is one undoable action (undo restores the previous objects,
  regions and provenance exactly).
* A re-run is refused while the analysis is finalized (reopen first), when the
  channel mapping is unconfirmed, or while another job for the analysis is running.
  If regions, channel mapping or alignment change while a job runs, the result is
  discarded and the job fails with an explanatory message.

## Analyses saved by an older version
States persisted before the input signature was recorded have no `detection_sig`. On load:
if detection had run and the stored stale flag is false, the signature is synthesised from the
current inputs (request recorded in the provenance, regions, roles, alignment), so every later
change makes the result stale as usual; if the stored flag is true it stays stale (sticky) until
detection is re-run, which records a real signature. Tombstones of objects deleted before
`user_deleted` existed are not known and do not suppress re-detected spots.

## Concurrent detect requests
Checking for a running job, setting the `detecting` status and submitting the job happen under one
hold of the analysis lock: of two simultaneous `POST /detect` exactly one is accepted, the other
gets 409 `job_in_progress` and never touches the status of the accepted job.

## Red/green role swap
A confirmed swap of the red and green channels (`PUT /channels`: the channel that was red
becomes green and vice versa) changes the meaning of every stored `red`/`green` evidence.
At the moment of the swap, for every non-deleted object (any origin):

* class `red` <-> `green` are mirrored; `yellow` and `uncertain` keep their class;
* the `red` and `green` evidence are exchanged;
* `reviewed=false` (needs re-review), a history entry is added;
* all approvals and inspection marks are cleared, the undo history is dropped, the detection
  is stale, and the event (`channel_mapping_changed`, `red_green_swapped=<n objects>`) is logged;
* the green->red alignment shift (defined as "shift applied to GREEN to match RED") is **negated**
  (dy, dx -> -dy, -dx), because the physical channel the shift applied to is now the red one. The
  event carries `alignment_negated={previous:{dy,dx}, now:{dy,dx}}`. Detection and the image views
  both read the corrected shift from the analysis. Swapping back restores the original shift.

Deleted objects are left untouched. A re-run afterwards is consistent: unreviewed automatic
objects are replaced by results made with the new mapping; manual/edited objects are preserved
with their already mirrored class (new detections within 3 px of them are dropped). Other
mapping changes (e.g. a different channel becomes red, ignore <-> role) cannot be mirrored: they
only invalidate (stale, approvals cleared); objects keep their classes until re-detection.

## Cancellation
`apply_detection_result(..., cancelled=...)` checks the cancel flag inside the same critical
section as the commit and before any mutation: a cancelled job commits nothing (raises
`DetectionCancelled`, job ends `cancelled`). The job runner decides the final status from what
the job function did (return value `"cancelled"` / raised cancel -> cancelled); a function that
returned a result is `done` even if a cancel request arrived afterwards.
