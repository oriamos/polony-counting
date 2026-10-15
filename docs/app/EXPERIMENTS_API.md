# Experiments API

Contract between the experiment page and the backend. All paths are under `/api`. Errors use the usual
body `{"code": "...", "message": "...", "detail": {...} | null}`. Ids are 12 lowercase hex characters; a malformed or
unknown experiment id gives `404 experiment_not_found`. `DetectionRequest` is the same object as
`POST /analyses/{aid}/detect` takes (`engine`, `red`, `green`, `spotiflow`, ...); it is abbreviated `{DetectionRequest}`
below. Unknown keys in a JSON body are refused with `422 unknown_fields`.

An experiment is an ordered list of slides. A slide is an ordinary analysis with `experiment_id` and
`order_in_experiment` set (both also appear in `GET /analyses/{aid}` and in each row of `GET /analyses`; loose
analyses have `experiment_id: null`, `order_in_experiment: 0`). Everything else about a slide (channels, regions,
grid, detect, review, finalize, save) uses the existing analysis endpoints.

## Experiment object

```json
{
  "id": "a0d1001dcb99", "name": "Clade run 12", "notes": "T7 phage",
  "created_at": "2026-10-15T16:29:58.213+00:00", "updated_at": "2026-10-15T16:29:58.213+00:00",
  "default_grid": null,
  "default_detection": {DetectionRequest},
  "status": "open"
}
```

* `status`: `open` | `running` (a batch run of this experiment is active, on this or another computer).
* `default_grid`: opaque JSON (the stage 1 `GridSpec`); stored and returned as given, never interpreted by the backend.
* `default_detection`: the settings an unattended run uses for every slide without an override.

## Endpoints

### `POST /experiments`
Body: `{"name": "Clade run 12", "notes": "T7 phage", "default_detection": {DetectionRequest}, "default_grid": {...}}`
(only `name` is required). Returns the experiment view (as `GET /experiments/{eid}`). `422 invalid_name` for an empty
name; `422 invalid_request` if `default_detection.preview_window` is set.

### `GET /experiments`
List, most recently updated first. Each item is the experiment object plus:
```json
{ "...": "...", "n_slides": 12, "n_detected": 9, "n_finalized": 2, "last_batch": {Job} | null }
```
`n_detected` = slides in status `in_review` or `finalized`.

### `GET /experiments/{eid}`
The experiment object plus `slides` (in experiment order) and `last_batch` (the newest batch job or null):
```json
{
  "...experiment fields...": "",
  "last_batch": {Job},
  "slides": [{
    "id": "7081d7622e73", "name": "slide_0", "order_in_experiment": 1,
    "status": "in_review",                 // analysis status: created|needs_channel_mapping|ready|detecting|in_review|finalized
    "export_status": "draft",              // finalized | approved (all counting regions approved) | draft
    "updated_at": "...", "finalized_at": null,
    "mapping_confirmed": true,
    "image": {"width": 160, "height": 120, "kind": "raw_two_channel", "dtype": "uint16", "n_channels": 2},
    "has_grid": false,                     // slide.grid is set (stage 1)
    "n_counting_regions": 1, "n_fields": 1, "n_approved": 0, "approved": false,
    "review_queue": 0,                     // objects waiting in the review queue
    "detection_ran_at": "...", "engine": "bigfish",   // null until detected
    "counts": {"scope": "slide", "red_only": 1, "green_only": 1, "yellow": 0, "uncertain": 0, "red_positive": 1,
               "green_positive": 1, "total_objects_exclusive": 2, "...": "same as GET /analyses/{aid}/counts slide"},
    "last_job": {Job} | null               // newest job of any kind on this slide (a failed batch slide shows its error here)
  }]
}
```

### `PUT /experiments/{eid}`
Body: any of `name`, `notes`, `default_detection`, `default_grid`. Only the keys present change; `"default_grid": null`
clears it, `"default_detection": null` resets to defaults. Returns the experiment view. A change does not affect a
batch that is already running.

### `DELETE /experiments/{eid}[?delete_slides=true]`
* `409 batch_in_progress` while a batch run is active.
* Default: the slides are detached (kept as loose analyses). `{"ok": true, "deleted_slides": [], "detached_slides": [ids]}`
* `delete_slides=true`: the slide analyses are deleted too; refused with `409 job_in_progress` if a slide has a running
  job, `409 analysis_finalized` (`detail.analysis_ids`) if a slide is finalized; nothing is deleted then.
  Returns `{"ok": true, "deleted_slides": [ids], "detached_slides": []}`.

### `POST /experiments/{eid}/import` (multipart)
Field `files`, repeated: one TIFF per slide (one 2-page 2-channel TIFF, or an RGB composite; slides made of two
single-channel files are not supported through this endpoint, use `POST /analyses` and then `/slides`). Each file goes
through exactly the same path as `POST /analyses` (size limit, ingest, upload manifest). Slides are appended in the
order of the files. If the experiment already has a slide with a confirmed channel mapping, the first such slide (in
experiment order) is the template: its mapping is applied (as a normal confirmed mapping) to every new slide whose file
layout is identical (image kind, dtype, number of channels, number of files, page of each channel); other slides stay
`needs_channel_mapping`.
```json
{ "created": [{"id": "86a36156e843", "name": "slide_1", "order_in_experiment": 2, "status": "ready",
               "mapping_copied_from": "7081d7622e73"}],
  "errors":  [{"file": "notes.txt", "code": "unsupported_file", "message": "'notes.txt': only .tif/.tiff files are accepted"}] }
```
A bad file does not stop the others. If no file could be imported: `422 import_failed` with `detail.errors`.

### `POST /experiments/{eid}/slides`
Attach existing analyses: `{"analysis_ids": ["..."]}`, appended in that order. Already attached ones are left as
they are. `409 in_other_experiment` (`detail.analyses`) if one belongs to another experiment. Returns the experiment view.

### `DELETE /experiments/{eid}/slides/{aid}`
Detach one slide (the analysis is kept). `404 not_in_experiment`. Returns the experiment view.

### `PUT /experiments/{eid}/order`
`{"analysis_ids": [...]}` listing every slide exactly once, in the new order (`422 invalid_order` otherwise).
Returns the experiment view.

### `POST /experiments/{eid}/run`
Starts the unattended run ("run and go to sleep"). Body (all optional):
```json
{ "only_missing": false,                  // true: skip slides that already have a detection result
  "analysis_ids": ["..."],                 // only these slides (e.g. Retry of the red rows); default all
  "overrides": {"<aid>": {DetectionRequest}} }   // per-slide settings instead of default_detection
```
Returns the batch `Job` (status `running`). Finalized slides are always skipped. Errors: `409 batch_in_progress`,
`422 not_in_experiment`, `422 invalid_request` (an override with `preview_window`), `422 nothing_to_run`
(`detail.slides` = the plan: everything finalized / already detected).

Behaviour:
* If any slide uses Spotiflow (`engine` `spotiflow` or `hybrid`), each needed model is prepared once before the first
  slide (same code as `POST /engines/spotiflow/prepare`). A preparation failure is a warning in the batch message;
  the run continues and each slide then reports its own engine problem.
* Slides run one at a time, in experiment order, each as an ordinary detect job (`parent_id` = batch id).
* **Skip and continue**: a slide that fails (no confirmed mapping, no counting region, engine error, finalized/busy) is
  recorded as a failed child job with the error text and the next slide starts. Refusals before detection have
  errors of the form `"<code>: <message>"`, e.g. `"no_counting_region: no counting region: ..."`,
  `"channel_mapping_unconfirmed: confirm the red/green channel mapping before detecting"`; engine errors look like
  `"RuntimeError: ..."`.
* Cancel: `POST /jobs/{batch_id}/cancel`. The slide being detected finishes; no further slide starts; the batch ends
  `cancelled` with a message `cancelled after 1 of 3 slides (1 done, 0 failed)`.
* Progress is in the `jobs` table: the run survives a closed browser. Batch `progress` = finished slides / slides to
  run; `message` e.g. `"slide 2 / 3: slide_1 (1 done, 0 failed)"`, at the end `"2 / 3 done, 1 failed, 1 skipped"`.
* Server restart during a run: the batch becomes `failed` with error
  `"server restarted: the batch was interrupted; run it again with only_missing=true to finish the remaining slides"`;
  the slide that was running gets the usual "interrupted: the application was restarted ..." failure. With the shared
  Postgres database the batch keeps a heartbeat like every job, so another computer sees it running and can cancel it.

### `GET /experiments/{eid}/run`
The current or last batch with per-slide status (poll this every 1-2 s while `batch.status` is `queued`/`running`):
```json
{
  "batch": {Job},          // null if the experiment never ran
  "slides": [{
    "analysis_id": "7081d7622e73", "name": "slide_0", "order": 1,
    "status": "done",       // pending|queued|running|done|failed|cancelled|skipped|not_run
    "skip_reason": null,    // "finalized" | "already_detected" when status = skipped
    "override": false, "job_id": "4cbda338aaaf", "progress": 1.0,
    "message": "2 objects detected", "error": null, "started_at": "...", "finished_at": "..."
  }],
  "summary": {"total": 3, "done": 2, "failed": 1, "cancelled": 0, "running": 0, "queued": 0, "pending": 0,
              "not_run": 0, "skipped": 0}   // total excludes skipped
}
```
`pending` = not started yet in an active batch; `not_run` = the batch ended (cancelled / restarted) before reaching it.
Without a run: `{"batch": null, "slides": [], "summary": null}`.

### `GET /experiments/{eid}/export.csv[?series=red&series=green&series=yellow]`
One CSV for the experiment, UTF-8 with BOM (Excel), file name `<experiment>_experiment.csv`. One row per slide per
count series. Each `series` value is a comma-separated set of classes that are summed (`red`, `green`, `yellow`; e.g.
`series=red,yellow` = red-positive); default: three series `red`, `green`, `yellow`. Columns:

| Column | Meaning |
|---|---|
| `Slide name` | scan file stem when opened from disk, else the analysis name (as `experiment_counts.csv`) |
| `What counted` | `red, not yellow` / `green, not yellow` / `yellow` / `red and yellow` ... (same labels as Save) |
| `NofPol` | sum of the field columns |
| `NofGrids` | number of counted fields |
| `Field1` .. `Field24` | count per grid field; `0` = counted, none found; blank = field not counted |
| `Experiment` | experiment name |
| `Status` | `finalized` / `approved` / `draft` |
| `Engine` | `bigfish` / `spotiflow` / `hybrid` |

The first 28 columns are exactly `experiment_counts.csv` (`export/slide_save.py`). Only slides with a detection result
and numbered grid fields are written; headers `X-Slides-Included` / `X-Slides-Skipped` give the numbers and the
provenance lists the skipped slides with the reason. `422 invalid_selection` for an unknown class.

### `GET /experiments/{eid}/provenance.json[?series=...]`
`<experiment>_provenance.json`, to keep next to the CSV:
```json
{ "exported_at": "...", "app_version": "...", "schema_version": 1,
  "experiment": {"id": "...", "name": "...", "notes": "...", "created_at": "...", "updated_at": "...",
                 "default_grid": null, "default_detection": {DetectionRequest}},
  "csv": {"header": ["Slide name", "..."], "series": [{"what_counted": "red, not yellow", "classes": ["red"]}],
          "included_slides": [{"analysis_id": "...", "name": "...", "slide_name": "...", "status": "draft"}],
          "skipped_slides": [{"analysis_id": "...", "name": "slide_2", "reason": "no_detection"}]},
  "last_batch": {"job_id": "...", "status": "done_with_failures", "message": "...", "error": null,
                 "started_at": "...", "finished_at": "...", "summary": {...}} ,
  "slides": [{"analysis_id": "...", "name": "...", "slide_name": "...", "order_in_experiment": 1, "status": "draft",
              "finalized_at": null, "image_sha256": {"0_slide_0.tif": "..."},
              "grid": null,                       // slide.grid (stage 1) when set: spec incl. seed / shift
              "counted_fields": [1, 2], "skipped_fields": [],
              "engine": "bigfish", "detection_ran_at": "...", "detection_request": {DetectionRequest},
              "engine_runs": [{"run_id": "...", "engine": "bigfish", "page": 0, "channel_role": "red",
                               "model_name": null, "model_version": null, "model_checksum": null,
                               "params": {...thresholds...}, "device": null, "status": "ok", "error": null,
                               "started_at": "..."}],
              "batch": {"batch_job_id": "...", "job_id": "...", "status": "done", "error": null}}] }
```

### `POST /lab-export` (existing, extended)
Body may give `"experiment_id": "<eid>"` instead of `analysis_ids`: all finalized slides of the experiment, in
experiment order. `422 invalid_request` if both or neither are given, `409 no_finalized_slides`. Everything else
(`style`, `series`, `format`, response) is unchanged.

## Jobs

`Job` (from `GET /jobs/{jid}`, the run endpoints and `last_job` / `last_batch`) gained fields:
```json
{ "id": "...", "analysis_id": "...", "status": "...", "progress": 0.0, "message": "", "error": null,
  "started_at": null, "finished_at": null,
  "kind": "detect", "parent_id": null, "experiment_id": null, "detail": null }
```

| `kind` | `analysis_id` | notes |
|---|---|---|
| `detect` | the slide | a slide of a batch has `parent_id` = batch id and `experiment_id` |
| `batch` | `experiment:<eid>` | `detail` = the plan: `{"slides": [{"analysis_id","name","order","skip","override"}], "only_missing", "default_detection", "overrides"}` |
| `prepare_model` | `engine-spotiflow` | `POST /engines/spotiflow/prepare` |

Statuses: `queued`, `running`, `done`, `failed`, `cancelled`, and new **`done_with_failures`** (batch only: finished,
at least one slide failed). A batch is never `failed` because of a slide; `failed` for a batch means the run itself
was interrupted (server restart, stale heartbeat on the shared database). Terminal states for polling:
`done`, `done_with_failures`, `failed`, `cancelled`.
