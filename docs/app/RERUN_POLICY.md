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
| soft-deleted by the user (`delete` action) | kept as a *tombstone* (history) and **suppresses** new detections at that spot |
| soft-deleted by merge/split | kept for history, no effect |

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
* A re-run clears `provenance.stale_detection` and records the request, time and
  package versions. It is one undoable action (undo restores the previous objects,
  regions and provenance exactly).
* A re-run is refused while the analysis is finalized (reopen first), when the
  channel mapping is unconfirmed, or while another job for the analysis is running.
  If regions, channel mapping or alignment change while a job runs, the result is
  discarded and the job fails with an explanatory message.
