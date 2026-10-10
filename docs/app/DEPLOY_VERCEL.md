# Deploying the polony review app to Vercel: feasibility (phase 1)

Status: **feasible, with caveats.** A real probe ran on Vercel on 2026-10-10 with the full
scientific stack (Big-FISH 0.6.2, numpy, scipy, scikit-image, scikit-learn, pandas,
matplotlib, tifffile, Pillow, openpyxl, roifile, psycopg). It ran Big-FISH detection on a
full-size 2438x1460 image in about 1.4 s. The port itself (Postgres, Blob, job model) has
not been done yet. This document records the evidence and proposes an architecture.

## 1. What was deployed

| Item | Value |
|---|---|
| Vercel team | `rbmc85wmc2-9475` (`team_dd86eRFmBMZrZ4LcHCgUk7W5`), Hobby plan |
| Project | `polony-review` (`prj_mNvxsAMr5be89bkPA15yhddVT4WH`), framework preset `fastapi` |
| Deployments | `dpl_8PA3bypDC5ZuoRNoCmwvrYJE3U3x` (production target, first deploy), `dpl_C6afT7VgZXgyfKqYTuTiaxWZxTCX` (preview, forced rebuild without cache) |
| Source | `deploy/vercel/probe/` (`app.py`, `requirements.txt`, `.python-version`, `vercel.json`), uploaded through the Vercel API. Nothing was pushed to git. No TIFFs and no `resources/` were included. |
| Protection | Vercel Authentication (`ssoProtection`) **enabled, `deploymentType: "all"`**. Password protection is off and there are no trusted IPs. No env vars or secrets were added. |

### Protection incident (fixed)

The project was created with the team default, `ssoProtection.deploymentType =
"all_except_custom_domains"`. Under that setting the production alias
**`polony-review.vercel.app` answered `200` without authentication.** Vercel counts that alias
as a "custom domain". The other `*.vercel.app` URLs redirected to SSO.

The setting was changed to `"all"`, which is stricter; protection was never disabled. All
four hostnames now answer `302 -> vercel.com/sso-api`. The exposure window was about 4
minutes (roughly 09:58:40 to 10:02:40 UTC). Only the probe was exposed: no data, no secrets,
nothing beyond the probe JSON. Runtime logs show only my own requests and Vercel's
deploy-time screenshot requests (`/`, `/favicon.*`).

**Keep `ssoProtection.deploymentType = "all"` and never attach a custom domain** without
revisiting this setting.

Testing used temporary share links from `get_access_to_vercel_url`, which expire after
23 h (on 2026-10-11).

## 2. Platform limits: documented and measured

Sources (fetched 2026-10-10):
- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/functions/runtimes/python
- https://vercel.com/docs/functions/runtimes (file system)
- https://vercel.com/docs/fluid-compute
- https://vercel.com/docs/frameworks/backend/fastapi
- https://vercel.com/docs/vercel-blob/using-blob-sdk
- https://vercel.com/docs/vercel-blob/client-upload
- https://vercel.com/docs/vercel-blob/private-storage
- https://vercel.com/docs/vercel-blob/vercel-signed-urls
- https://vercel.com/docs/functions/functions-api-reference/vercel-sdk-python

| Limit | Docs (Hobby) | Measured on the probe |
|---|---|---|
| Python bundle, uncompressed | 500 MB. "Large functions" beta allows up to 5 GB (Fluid + Active CPU; `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` for existing projects). | Build log: `Bundle size (375.43 MB) exceeds the standard size; optimizing dependencies.` At runtime the deps are **installed into `/tmp/_vc_deps` on every cold instance** (`Runtime dependencies installed in 3.4-3.7 s`). Measured 388.9 MB in `/tmp/_vc_deps` and 186.6 MB in `/var/task`. |
| Memory / CPU | 2 GB / 1 vCPU (Hobby default and maximum) | `MemTotal` 2.33 GB. `os.cpu_count()` = 2 and sched affinity = 2, but Hobby is billed and throttled as 1 vCPU. |
| maxDuration | 300 s default and maximum (Fluid) | `vercel.json` sets 300. A BackgroundTask of 240 s and one of 330 s: see §2.1. |
| Request/response body | **4.5 MB** (`413 FUNCTION_PAYLOAD_TOO_LARGE`) | Not tested. A raw channel TIFF is 14 MB, so uploads must bypass the function. |
| `/tmp` | "writable `/tmp` scratch space up to 500 MB" | 550 MB total but only **127 MB free**, because the runtime deps occupy `/tmp`. 50 MB write in 0.29 s. |
| Background work after the response | `waitUntil` is documented for Node.js/Edge only. The Python SDK (`vercel.functions`) has no `wait_until`. | **Starlette `BackgroundTasks` keep running after the response**: response in 0.2 s, `POLONY_BG_DONE after 45.0s` in the runtime logs. See §2.1 for the limit. |
| Concurrency | Fluid: several invocations share one instance (Python supported) | 6 parallel full-size probes **all landed on one instance**. Peak RSS 1.31 GB; Big-FISH time rose from 1.4 s to 6.3 s each. |
| Cold start | Production pre-warming; bytecode is precompiled when space allows | Both deployments were pre-warmed by Vercel's screenshot bot. Cold-instance cost observed in logs: about 3.6 s dependency install, plus about 0.5 s FastAPI import, plus **2.6 s** first-use heavy imports (sklearn 0.89 s, scipy 0.43 s, pandas 0.36 s, bigfish 0.33 s, ...). **Estimated first request on a cold instance: about 7-8 s.** Warm `/api/health`: 0.2 s round-trip. |
| Python | 3.12 (default), 3.13, 3.14. Version comes from `.python-version` or `pyproject.toml`. | 3.12.14, Amazon Linux 2 (glibc 2.34), region `iad1` |

### 2.1 Background task beyond maxDuration

Two background tasks were scheduled at 10:02:08 UTC: 240 s and 330 s. Result:
- 240 s task: `POLONY_BG_DONE after 240.0s` at 10:06:08. It completed.
- 330 s task: `Vercel Runtime Timeout Error: Task timed out after 300 seconds` at 10:07:07.

**Background work is allowed, but the response time plus the background work must fit in
maxDuration (300 s on Hobby).** A detection job that may exceed about 280 s must be split
into resumable steps.

### 2.2 Detection timings (probe, warm instance)

| Workload | Vercel | Local dev box |
|---|---|---|
| `bigfish.detection.detect_spots`, 300x300 synthetic, 40 spots | 0.025-0.03 s (40/40 found) | 0.085 s |
| `detect_spots`, 2438x1460 synthetic, 4000 spots | **1.36-1.39 s** (3815 found) | 2.7 s |
| `polony_app` `BigFishEngine.detect`, same 2438x1460 image (sub-pixel, measurement, plateau merge) | not deployed (see note) | 15.3 s, so roughly 8 s on Vercel |

Peak RSS for one full-size detection: 434 MB.

Note: the probe did not ship the `polony_app` snapshot, because inline API upload of about
50 KB of source was skipped. `deploy/vercel/stage_probe.sh` stages it, and the probe's
`?full=1` path runs `BigFishEngine` when the snapshot is present.

The ~8 s estimate scales the local engine time by the measured Vercel/local ratio for
`detect_spots` (about 0.5). A two-channel analysis is therefore about 15-20 s. That is well
inside 300 s, as long as the work is not starved by concurrent jobs on the same instance.

## 3. Dependency set and size

Trimmed set: `deploy/vercel/probe/requirements.txt`, pinned to the versions the app is
already tested with locally.

Measured in a clean Python 3.12 venv (`uv pip install --no-cache`):

| Set | Unpacked site-packages |
|---|---|
| fastapi, pydantic, numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib, big-fish 0.6.2, tifffile, roifile, Pillow, openpyxl | 373 MB |
| + `psycopg[binary]` | 393 MB (Vercel reported 375.4 MB) |
| + `imagecodecs` | +86 MB (481 MB). **Dropped.** |
| + `vercel` Python SDK (Blob, OIDC) | +45 MB (zstandard 23 MB, cryptography 15 MB, ...). **Avoid**: call the Blob REST API with `httpx` instead, or accept the size. |

Other dependency findings:
- `pip` is not pulled in by big-fish 0.6.2. `uvicorn` is not needed on Vercel.
  `joblib` warns that there is no `/dev/shm` and runs serially, which is harmless.
- **imagecodecs and LZW.** The raw channel TIFFs (14 MB, uint16, 2 pages, ImageJ) are
  uncompressed (`compression=1`). The `*_(635_532).tif` RGB composites (3-4 MB) are **LZW**
  (`compression=5`). tifffile without imagecodecs raises
  `ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package`. Pillow 12 decodes the
  same files correctly (`(2438, 1460, 3) uint8`).
  - Recommendation: add a small fallback in `polony_app/imaging/io.py`: on that `ValueError`,
    read the composite with `PIL.Image.open`. Do not ship imagecodecs.
- Static/bytecode: Vercel compiles `.pyc` "when space allows". With the dependency
  optimisation above, the bundle fits.

Headroom: about 105 MB to the 500 MB documented limit. Because of the runtime-install
optimisation, **the binding constraint is really `/tmp`**: 550 MB minus about 389 MB of
deps leaves 127 MB of scratch space per instance. Every dependency added to the bundle
reduces that free space.

## 4. Repository layout (avoiding the legacy root `pyproject.toml`)

Vercel picks dependencies from `requirements.txt`, `pyproject.toml` (with or without
`uv.lock`) or `Pipfile` **in the project root directory**. The repo root `pyproject.toml` is
a Poetry file that pins `torch 2.0.0+cu118`, so it must never be the deploy root.

Chosen layout (verified by this probe): the deploy root is a **self-contained staging
directory** containing only
`app.py`, `requirements.txt`, `.python-version`, `vercel.json`, a copy of `polony_app/` and,
for the port, `public/` holding the built frontend (`frontend/dist`). It is assembled by
`deploy/vercel/stage_probe.sh` (to become `stage_app.sh` for the port). Two ways to deploy it:
- **API/CLI upload of the staged directory.** This is what was used here; it needs no git
  push. With a token: `vercel deploy <staged dir> --scope rbmc85wmc2-9475`.
- **Git integration.** Set the project `rootDirectory = deploy/vercel/app`, commit the
  staged sources there, or use a `[tool.vercel.scripts] build` / `buildCommand` that copies
  `../../src/polony_app`. This also needs `sourceFilesOutsideRootDirectory`. It is not
  verified yet and needs a push, which was not done.

Frontend: put the Vite build in `public/` (served from the CDN; do not `app.mount` it), or
use `app.frontend("/", directory="dist")`, which applies an index.html fallback for SPA
navigation. Either way the CDN serves the assets and API routes take priority.

## 5. Proposed architecture for the port

```
browser (React, served from CDN behind Vercel Authentication)
  |-- PUT TIFF -> Vercel Blob (private store)          [bypasses the 4.5 MB limit]
  |-- /api/*  -> FastAPI function (Python 3.12, 300 s, 2 GB)
                   |-- Postgres (Neon, psycopg 3, pooled DATABASE_URL)
                   |-- Blob GET (server side, no size limit) -> /tmp cache
```

### 5.1 Database: Neon Postgres replacing SQLite

- `store/db.py` (279 lines) already keeps JSON documents in four tables (`analyses`,
  `actions`, `jobs`, `csv_imports`). Port these to Postgres with `JSONB` `data` columns:
  - `INSERT OR REPLACE` becomes `INSERT ... ON CONFLICT DO UPDATE`.
  - The `PRAGMA`s go away.
  - Use `psycopg` 3 with short-lived connections, or `psycopg_pool` with `max_size` of about 3.
    The pooled Neon endpoint (`-pooler` host) is required because Fluid instances scale out.
- Env vars from the Neon Marketplace integration: `DATABASE_URL` (pooled) and
  `DATABASE_URL_UNPOOLED`. Use the unpooled one for migrations only.
  - Keep `POLONY_DATA_DIR` pointing at `/tmp/polony` for caches only.
- Action log ordering (`seq`) and undo need row-level locking (`SELECT ... FOR UPDATE`) or a
  unique `(analysis_id, seq)` constraint with retry. With SQLite this was serialised by a
  process-local lock, which no longer holds across instances.

### 5.2 Uploads: Vercel Blob, private store

- Uploads cannot go through the function: 14 MB TIFFs against a 4.5 MB body limit.
- Blob multipart parts must be at least 5 MB (except the last), so proxying chunks through
  the function into one multipart upload is impossible.
- **The Python SDK (`vercel` 0.11.6) has no `handle_upload` / `generate_client_token` /
  `presign`**. It only does server-side `put`/`get`/`head`/`list`/`copy`/`delete` and
  multipart. Client-token minting is JS-only (`@vercel/blob/client` `handleUpload`, which
  needs `BLOB_READ_WRITE_TOKEN`; or `handleUploadPresigned` with `issueSignedToken`, which
  is OIDC-compatible).

Options, in order of preference:
1. **Small Node route** (`api/blob-upload.mjs` with `handleUploadPresigned`) alongside the
   Python app via Vercel *Services*. Services is a multi-runtime project mode, "Beta,
   available on all plans", configured with a `services` key in `vercel.json`
   (https://vercel.com/docs/services). The browser uses
   `@vercel/blob/client` `uploadPresigned(..., {access: 'private', multipart: true})`.
   - The upload-completed callback calls the Python API, or the browser calls it after
     `upload()` resolves, to register the analysis.
   - Auth: `getSignedToken` must authorise the request. Behind Vercel Authentication, only
     team members reach the route.
   - Needs verification: the Services setup itself and how Deployment Protection treats the
     Blob callback. Use the presigned flow, or have the browser confirm, to avoid callbacks
     into a protected deployment.
2. **Python-only fallback.** The browser slices the file into 4 MB chunks and `POST`s each
   to `/api/uploads/{id}/chunk/{n}`. The function `put`s each chunk as its own private blob
   (`uploads/{id}/{n}`). On `/complete` the function downloads the chunks server side,
   concatenates them in `/tmp` (14 MB, fine), `put`s the merged original, and deletes the
   chunks. This costs one function invocation per 4 MB but needs no JS server code.
3. Re-implementing the client-token HMAC format in Python. The format is undocumented, so
   this is **not recommended**.

Private blobs are read server-side with `get(pathname, access='private')` (OIDC on Vercel,
or `BLOB_READ_WRITE_TOKEN`). Env vars: `BLOB_STORE_ID`, `VERCEL_OIDC_TOKEN` (automatic),
`BLOB_READ_WRITE_TOKEN`, `BLOB_WEBHOOK_PUBLIC_KEY`. To keep the bundle small, call the
Blob HTTP API through `httpx` rather than adding the 45 MB `vercel` SDK, or accept the SDK
if the `/tmp` budget allows.

### 5.3 Detection jobs

- `store/jobs.py` currently uses an in-process `ThreadPoolExecutor` with `threading.Event`
  cancellation. That does not survive across instances.
- New model:
  - `POST /api/analyses/{id}/detect` inserts a `jobs` row (`queued`), returns its id, and
    runs the detection in a Starlette `BackgroundTask` in the same invocation (measured to
    keep running after the response; bounded by maxDuration, see §2.1).
  - Alternatively, run detection synchronously within the request (about 15-20 s for two
    full-size channels) while the UI polls. The background-task variant gives a nicer UX.
  - Progress is written to the `jobs` row every few percent, so `GET /api/jobs/{id}` works
    from any instance.
  - Cancel sets `jobs.cancel_requested = true`. The engine's existing `cancelled()` callback
    polls that flag, throttled to about 1/s.
  - Stale jobs (`running` with a heartbeat older than 330 s) are marked `failed` on read.
- Concurrency risk: Fluid packs concurrent requests into one 2 GB instance, and 6 parallel
  probes reached 1.3 GB. Cap concurrent detections per instance with a module-level
  semaphore of 1-2, and/or reject with `429` when more than N jobs are `running` in the DB.

### 5.4 Render/thumbnail caching

- The originals cache lives at `/tmp/polony/originals/<sha>.tif` and the render/thumb cache
  at `/tmp/polony/render/...`, both with an LRU budget of about **80 MB**. Only 127 MB of
  `/tmp` is free.
- On a miss, re-download from Blob, which is fast from `iad1`.
- Rendered PNGs can carry `Cache-Control: private, max-age=...` so the browser caches them.
  Do not use the CDN cache for protected data.

### 5.5 Env vars (proposed)

| Var | Source |
|---|---|
| `DATABASE_URL`, `DATABASE_URL_UNPOOLED` | Neon Marketplace integration (or the Neon connector) |
| `BLOB_STORE_ID`, `BLOB_READ_WRITE_TOKEN`, `BLOB_WEBHOOK_PUBLIC_KEY` | Blob store connected to the project |
| `POLONY_DATA_DIR=/tmp/polony` | plain |
| `POLONY_MAX_UPLOAD_MB=100` | plain (enforced at token minting) |
| `POLONY_MAX_CONCURRENT_JOBS=2` | plain |

## 6. Effort and risks (honest estimate)

| Work item | Estimate |
|---|---|
| Postgres store (`db.py`, `jobs.py`) + migrations + tests against a Neon branch | 1.5-2 days |
| Blob upload path: Node presigned route + frontend `uploadPresigned` (or the Python chunk fallback) | 1-1.5 days |
| Job model rewrite (DB progress, cancel flag, background task, stale detection) | 1 day |
| `/tmp` cache layer + LZW Pillow fallback + staging script + vercel.json | 0.5 day |
| Deploy, verify, e2e against the protected preview | 0.5-1 day |
| **Total** | **about 4.5-6 days** |

Risks:
- **Hobby has 1 vCPU / 2 GB**, shared by all concurrent requests on an instance. Several
  reviewers running detection at once will slow down or OOM unless capped. Hobby is also
  for non-commercial use per Vercel's terms; lab use may need Pro (which gives 4 GB / 2 vCPU
  and 800 s).
- `/tmp` headroom of 127 MB is small and shrinks with every dependency.
- "Optimizing dependencies" (runtime install into `/tmp`) is undocumented behaviour. It
  could change, and it adds about 3.6 s to every cold start.
- The background-task-after-response behaviour is observed, not documented for Python.
  If it breaks, fall back to synchronous detection within the request.
- Vercel Authentication means every user needs a Vercel account with access. On Hobby
  that is the owner plus **at most one external user per account**, who is granted access
  through an "access request" and sees the latest deployment of a branch. Shareable links
  also work, but anyone holding the link gets in. Source:
  https://vercel.com/docs/deployment-protection/methods-to-protect-deployments/vercel-authentication
  **This is the biggest practical blocker for a lab with more than two reviewers**:
  team/Viewer seats require Pro. Hobby is also "non-commercial, personal use only".
  Source: https://vercel.com/docs/plans/hobby
- Neon free-tier compute auto-suspends, adding about 0.5-1 s on the first query after idle.

## 7. What the user still needs to do

1. Decide on access. On Hobby, the owner plus one external Vercel user (via an access
   request) can pass Vercel Authentication. For more reviewers, upgrade the team to Pro;
   Viewer seats are free on Pro.
2. Provision Postgres: add **Neon via the Vercel Marketplace** (Project -> Storage -> Neon),
   which injects `DATABASE_URL*`, or authorise the Neon connector so the agent can create
   the DB and set env vars.
3. Create a **private Blob store** and connect it to `polony-review` (Storage -> Blob ->
   Private), for Production and Preview.
4. Optional: give the agent a way to deploy the staged directory (Vercel token for the CLI,
   or approve a git branch + `rootDirectory`).
