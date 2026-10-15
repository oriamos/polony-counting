# Implementation plan: lab feedback round 1

Source: Elena's feedback after the hackathon (grid by clicking, randomise grid, Spotiflow on Mac, work organised by
experiment, standalone program). Decisions already taken by the project owner: **a grid is set per image, with an
option to apply one grid to every image of the experiment**; **an overnight run skips a failed slide and continues**.
Decision still open: the exact column layout of the experiment CSV (a default is proposed in stage 3, step 3.6).

Order: stage 1 (grid clicking + randomise), stage 2 (Mac), stage 3 (experiments), stage 4 (desktop window).
Stage 2 runs in parallel with the others because it is mostly waiting on a Mac to test with.
Every stage ends with: backend tests (`make app-test`), `npm test` and the mock e2e in `frontend/`, `make web`
(the committed `src/polony_app/webdist/` must be rebuilt), a short entry in `USER_GUIDE.md`.

What already exists and is reused (verified in the code, not assumed):

| Piece | Where | Reused by |
|---|---|---|
| Grid generator (rows, cols, cell size, origin, numbering order, start corner) | `frontend/src/lib/labgrid.ts` (`GridSpec`, `generateGrid`, `defaultGrid`), `RegionsTab.tsx` (`GridGenerator`) | 1 |
| Switch a grid field on/off, geometry remembered in `slide.skipped_fields` | `PUT /analyses/{aid}/grid-selection`, `SlideMeta.skipped_fields` | 1, 3 |
| Grid preview cells drawn over the image | `store.gridPreview`, `Overlay.tsx` | 1 |
| Integer grid shift | `src/polony_app/fields/grid.py` `Grid.shifted` | 1 |
| Batch upload + detect many files, summary CSV | `BatchPanel.tsx`, `lib/batch.ts` | 3 |
| Threaded job runner with progress, cancel, one active job per analysis, heartbeat for the shared DB | `store/jobs.py` | 3 |
| Per-slide save: labelled TIFF, per-slide CSV, `experiment_counts.csv` upsert next to the scan | `export/slide_save.py`, `POST /analyses/{aid}/save-slide` | 3 |
| Lab-format XLSX/CSV export of finalized analyses | `export/lab.py`, `POST /lab-export` | 3 |
| Device detection cuda / mps / cpu with fallback | `engines/spotiflow_adapter.py` | 2 |
| Installer AI extra with per-platform torch plan | `install/ai_extra.py` (`plan`) | 2 |
| Desktop launcher (detached server, pid file, icon, `stop`, `status`) | `src/polony_app/launcher.py`, `install/` | 4 |

---

## Stage 1: pick grid fields by clicking, randomise the grid (about 1 day)

Goal: the researcher generates a grid, clicks the fields she wants counted (instead of drawing rectangles), and can
shuffle the grid position with one button. Field size never changes; only the position does.

### 1.1 Persist the grid specification (backend, small)
Today only the resulting regions are stored; the `GridSpec` lives in React state and is lost on reload, so neither
"randomise" nor "same grid on every image" (stage 3) can be reproduced.
* `schemas.py`: add `GridSpec` (rows, cols, cell_w, cell_h, origin_x, origin_y, order, corner, first_index,
  `seed: Optional[int]`, `shift_dx`, `shift_dy`) and `SlideMeta.grid: Optional[GridSpec]`.
* `PUT /analyses/{aid}/grid` (new): stores the spec and (re)creates the field regions from it in one step, keeping the
  on/off choice of fields that still exist (by field index) and the approvals of fields whose polygon did not move.
  Moving a grid after approval invalidates the approval fingerprint, exactly as editing a region does today.
* Record the action in the undo log (`engine.set_regions` already does).
* Tests: `tests/app/test_grid.py` round-trip, re-apply keeps skipped fields, a moved field loses approval.

### 1.2 Click a cell to switch it on/off (frontend)
* New tool `'grid-pick'` in `store.ts` / `Viewer.tsx` `onPointerDown`: a click inside a grid cell calls the existing
  `PUT /grid-selection` with `counted = !current`. Cells outside the image or fully inside an exclusion mask are not
  selectable (reuse `exclusionCoverage`).
* `Overlay.tsx`: draw every cell of the stored grid (not only the preview): selected = solid outline + field number,
  unselected = dotted grey, hover = highlight. The current `gridPreview` becomes the "grid not yet applied" state.
* `RegionsTab.tsx` `GridGenerator`: after **Apply**, the tool switches to `grid-pick` automatically and the help line
  says "Click fields to count them; click again to skip". Keyboard: `A` select all, `N` none, `I` invert.
* The old rectangle/polygon tools stay for exclusions and for non-grid slides.
* Tests: `labgrid.test.ts` (cell hit test), e2e mock: generate grid, click two cells, reload, both still selected.

### 1.3 Randomise button
* `labgrid.ts`: `randomShift(spec, imageW, imageH, rng)`: picks `(dx, dy)` uniformly inside the slack between the grid
  bounding box and the image border (so the whole grid stays on the image; if the grid is larger than the image the
  slack is 0 on that axis). Cell size and count never change. The seed is stored in `GridSpec.seed` so a slide can be
  reproduced and the provenance export can state "grid randomised, seed N".
* Button **Randomise** next to Apply; **Undo** reverts (undo log). Selected field indices are kept (same row/col),
  but their approvals are invalidated because the pixels changed.
* Open question for Elena (does not block): should the shift also be allowed in integer multiples of the cell size
  only, or any pixel offset? Default: any pixel offset.

### 1.4 Done when
Generate grid, randomise, click 12 cells, detect, approve, Save: `experiment_counts.csv` has 12 field columns filled
and the rest blank; reload the page and the grid and selection are unchanged.

---

## Stage 2: Spotiflow on Mac (about 0.5 day of work, plus waiting for a test Mac)

Facts from the code: Apple Silicon is already supported (`ai_extra.py` installs PyPI torch 2.14.1 and uses MPS). Intel
Macs are refused because PyTorch stopped publishing macOS x86_64 wheels after torch 2.2.2. The installer therefore
says "Intel Macs: not available" and the app keeps working with Big-FISH only.

### 2.1 Find out which Mac Elena has
Apple menu > About This Mac > "Chip" (Apple M1/M2/M3/M4) or "Processor" (Intel). This decides the path below.

### 2.2 Apple Silicon and it still fails
Then it is a bug, not a platform gap. Ask for `~/PolonyReview/logs/server.log` and the output of
`~/PolonyReview/bin/polony-review status`. Likely causes, each with a fix:
* MPS op not implemented in Spotiflow's model (error contains `MPSNDArray` or `not currently implemented for the MPS
  device`): `spotiflow_adapter.py` catches the error on first inference and retries on `cpu`, recording
  `device_fallback` in the run notes (today a requested-but-unavailable device falls back; a device that exists but
  fails at run time does not).
* Gatekeeper / quarantine on the downloaded wheels: add the quarantine removal to the `--with-ai` path of
  `install.sh` (it is already done for the `.app`).
* `--with-ai` was never run: the UI shows "Spotiflow not installed" and the install command to re-run.

### 2.3 Intel Mac
Pin an older torch pair for `Darwin x86_64` only: `torch==2.2.2`, `torchvision==0.17.2` (last macOS x86_64 wheels,
Python 3.10-3.12 only; Python 3.13 has no wheel, so the installer must select 3.12 on this platform). Spotiflow 0.6.5
does not pin a torch version (checked its wheel metadata), but it pulls `lightning` and `torchvision` and newer
`lightning` releases may require torch >= 2.x features; if resolution fails, pin `lightning<2.4`.
* `ai_extra.py plan()`: new branch for Darwin + x86_64 returning `supported=True`, the pins above, `device="cpu"`,
  label "Intel Mac: last PyTorch release for this platform (CPU only, slower)".
* `install.sh`: on Intel Mac prefer Python 3.12 when downloading via `uv`.
* Test: `tests/app/test_installer_*` unit test of `plan("Darwin", "x86_64", ...)`; real check only on a Mac or in the
  `install-test.yml` workflow (GitHub macOS runners are Apple Silicon now, so Intel can only be tested on a real
  machine: Elena's).
* Expectation to state honestly: CPU inference on an Intel Mac is slow (minutes per slide); the overnight batch of
  stage 3 is what makes this acceptable.

### 2.4 Done when
`polony-review status` on her Mac prints `spotiflow <ver>, torch <ver>, device mps|cpu` and a hybrid detection
finishes on one slide.

---

## Stage 3: work organised by experiment (about 3-4 days)

Goal: create an experiment, import all its slides, set a grid on each (or one grid for all), run detection on
everything unattended, review the counts the next day, export one CSV for the whole experiment.

### 3.1 Data model (backend)
* New table `experiments` (id, name, created_at, updated_at, notes, `default_grid` JSON, `default_detection` JSON,
  `status`). Column `experiment_id TEXT` on `analyses` (nullable: old analyses stay valid). Same in `store/pg.py` for the
  shared Postgres option, with a migration that adds the column.
* `Analysis.experiment_id: Optional[str]`, `Analysis.order_in_experiment: int`.
* Endpoints: `POST/GET /experiments`, `GET /experiments/{eid}` (slides with status, counts summary, last job),
  `PUT /experiments/{eid}` (name, notes, default grid / detection settings), `DELETE` (refuses while a run is active),
  `POST /experiments/{eid}/import` (many files, reuses the upload path of `POST /analyses` + `/originals`;
  channel mapping copied from the first confirmed slide when the file layout is identical, otherwise left for the
  user), `GET /experiments/{eid}/export.csv`.

### 3.2 Grid per slide, with "apply to all"
* Each slide keeps its own `SlideMeta.grid` (stage 1). The experiment page has **Apply this grid to all slides**: copies
  the spec (same size, same origin, same selected field indices) to every slide that has no grid yet, or to all slides
  after a confirmation ("3 slides already have a grid and 2 of them are approved; overwrite?"). Slides with a different
  image size get the grid clamped and are flagged.
* **Randomise on all** does the same with a fresh seed per slide (independent shifts, as a blind design wants).

### 3.3 Unattended run ("run and go to sleep")
* `POST /experiments/{eid}/run` creates one *batch job* (new `jobs.kind = 'batch'`, `parent_id` on child jobs) which
  enqueues one detection job per slide in order. The existing `JobRunner` already caps concurrency (2 workers) and
  forbids two jobs on one analysis; the batch job only sequences them.
* **Skip and continue** (decided): a slide whose detection fails (bad mapping, missing grid, engine error) is marked
  `failed` with the error text; the batch continues; the experiment page shows a red row with **Retry**. The batch
  ends `done_with_failures` rather than `failed`.
* Progress is persisted (`jobs` table), so closing the browser or the laptop lid (server keeps running; the launcher
  never idles it out) does not lose it. Reopening the experiment shows "7 / 12 done, 1 failed".
* **Cancel** stops after the current slide. Optional: **Pause after current** for a machine that is needed.
* Detection settings for the run: the experiment defaults (engine, downscale, thresholds); a slide may override.
* `prepare-model` runs once before the batch so the model download happens up front, not in the night.

### 3.4 Review the next morning
* Experiment page: a table, one row per slide: thumbnail, status, red / green / yellow totals, number of fields counted,
  number of objects in the review queue, approve state. Click a row = open the slide as today. **Next unreviewed**
  button (`]` key) walks the experiment in order.
* Approve stays per field; **Finalize experiment** finalizes every approved slide and refuses while a slide is
  unreviewed, listing them.

### 3.5 One CSV per experiment
* `GET /experiments/{eid}/export.csv` writes **one row per slide per count series**, same columns as the existing
  `experiment_counts.csv` (`Slide name, What counted, NofPol, NofGrids, Field1..Field24`), plus `Experiment`, `Status`
  (finalized / approved / draft) and `Engine`. Provenance JSON (model id, thresholds, grid seed per slide) goes next
  to it as `<experiment>_provenance.json`, like the per-analysis export.
* The lab-format XLSX (`POST /lab-export`) accepts an `experiment_id` and takes all finalized slides of it.

### 3.6 Open decision: CSV layout
Default proposed above = the researcher's own `experiment_counts.csv` layout, which `slide_save.py` already produces,
so nothing she does today changes. Alternative: one row per slide with the three colour series side by side. Ask Elena
with a sample file of each; switching later is a one-function change in `export/`.

### 3.7 Frontend
* New left-rail section **Experiments** above the analyses list; **New experiment** dialog (name, notes, engine
  defaults); drag-and-drop of many TIFFs onto the experiment; the current `BatchPanel` becomes the import step of an
  experiment and is removed as a separate panel.
* Experiment page component (`ExperimentPanel.tsx`): slide table, grid tools (3.2), Run / Cancel / Retry, export
  buttons, progress bar from the batch job.

### 3.8 Tests
Backend: experiment CRUD, import of 3 files, batch run with one slide forced to fail (skip + continue +
`done_with_failures`), cancel between slides, export CSV content and provenance, PG migration. Frontend: vitest for
the slide table state; mock e2e: create experiment, import, apply grid to all, run, one failure shown, export.

### 3.9 Done when
Elena creates an experiment with 10 slides, applies one grid, randomises, runs hybrid, closes the laptop lid, comes
back, sees 9 done / 1 failed with a clear reason, reviews, finalizes, downloads one CSV with 10 slides.

---

## Stage 4: a separate program instead of a browser tab (about 1-2 days for the window, more for a single-file installer)

Goal: Polony Review opens in its own window with its own icon and does not look like a web page. The server and the
UI stay exactly as they are; only the shell changes.

### 4.1 Native window around the existing UI (do this first)
* Add `pywebview` to `requirements-app.txt` (pure Python, uses the OS web view: WebKit on macOS, WebView2 on Windows,
  GTK WebKit on Linux). `launcher.py launch` starts the server as today and, when `pywebview` imports, opens
  `http://127.0.0.1:<port>` in a `webview.create_window("Polony Review", ...)` instead of `webbrowser.open`.
  Fallback to the browser when the web view is missing (old Linux without GTK WebKit) with a one-line notice.
* Window close = `polony-review stop` only if no job is running; otherwise a dialog "A detection is running; keep the
  server running in the background?" (the launcher already refuses to kill a running detection).
* File dialogs: the existing **Open from disk** (`/fs/list`) keeps working; pywebview adds a native file picker, wire
  it to the same endpoint.
* The desktop icon generated by the installer (`install/make_icons.py`) already exists; point the `.app` / `.lnk` /
  `.desktop` at `launcher.py launch` (they already do), so no installer change beyond the new dependency.
* Tests: `launcher --self-test` gains a `--window` smoke that creates and closes the window in CI where a display
  exists (Windows/macOS runners), skipped on headless Linux.

### 4.2 Optional later: single-file installer
PyInstaller / briefcase bundle with Python inside, per OS. Heavy (PyTorch adds 0.8-2.5 GB, code signing needed on
macOS to avoid Gatekeeper warnings, Windows SmartScreen for unsigned exe). Only worth it if the lab needs to install
on machines without internet; the current one-command installer already gives an icon and needs no admin rights.

### 4.3 Done when
Double-click the Desktop icon on Windows and macOS: a window titled Polony Review opens with the app, no browser
chrome; closing it stops the server when idle.

---

## Cross-cutting

* **Docs**: `USER_GUIDE.md` sections "Grid" (click, randomise), "Experiments" (create, import, run overnight, review,
  export), `INSTALL.md` (Intel Mac row, window mode), `FIELDS.md` (grid spec and seed in provenance).
* **Provenance**: grid seed and shift, experiment id, batch job id and per-slide status go into the export JSON; a
  randomised grid is reproducible from the seed.
* **Shared DB**: every new table/column also in `store/pg.py`; the batch job uses the existing heartbeat so a second
  computer sees the run.
* **Out of scope here** (tracked in `OPEN_ITEMS_AND_HACKATHON.md`): model registry wiring, locked-test enforcement,
  Vercel port.

## Timeline (one developer)

| Week | Work |
|---|---|
| 1 | Stage 1 (1 day), stage 2 code (0.5 day), stage 3 backend: data model, import, batch run (2.5 days) |
| 2 | Stage 3 frontend + export + tests (2.5 days), stage 4.1 window (1.5 days), docs, release to the lab |
| waiting | Stage 2 verification on Elena's Mac; CSV layout decision (3.6) |
