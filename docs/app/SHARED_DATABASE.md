# Shared database (Neon / PostgreSQL)

By default the app keeps everything on the researcher's computer (SQLite, `data/app.sqlite`).
If a **database URL** is configured, the same app stores analyses in a shared PostgreSQL
database (Neon) instead, so several people / computers see the same analyses, counts and
review history.  Nothing else changes in the UI; the header shows `Local only` or
`Shared database (Neon): connected`.

## What is shared and what is not

| Shared (in the database)                                   | Stays on each computer                      |
|------------------------------------------------------------|---------------------------------------------|
| analyses, objects, regions, counts, undo/redo, action log  | original TIFF files (`data/originals/<id>/`) |
| detection jobs (progress, cancel), CSV imports             | the exports you download                    |
| per-file **SHA-256** of each original, machine registry    | the app, its logs, `config.json`            |

If you open an analysis made on another computer you get a banner *"This analysis was created on
another computer. Attach the original TIFF(s) to view the image"* listing the expected file names
and SHA-256.  **Attach** accepts only files whose SHA-256 equals the recorded one (a different or
modified file is rejected, whatever its name), and stores them under the recorded name.
Counts, tables, CSV/lab exports work without the image; image views, thumbnails, histograms and
detection return `409 original_missing` until the file is attached.

## Configuring a computer

Priority: environment variable, then `config.json`.  No URL = local SQLite.

1. Environment: `POLONY_DATABASE_URL=postgresql://<role>:<password>@<host>/<db>?sslmode=require`
2. File `~/PolonyReview/config.json` (Windows: `%LOCALAPPDATA%\PolonyReview\config.json`), **chmod 600**:
   ```json
   { "database_url": "postgresql://<role>:<password>@<host>/<db>?sslmode=require" }
   ```
   Other keys written by the app/launcher (`data_dir`, `port`, `machine_id`) are preserved.
3. Optional: `POLONY_DATABASE_SCHEMA` / `"database_schema"` (default `polony`).

`machine_id` (random UUID) is generated on first use and stored in `config.json`; it tags every
action and job so the app knows which jobs belong to this computer.  Do not copy `config.json`
between computers (or delete the `machine_id` line after copying).

There is deliberately no `--database-url` flag (secrets on a command line show up in process lists).
The installer/launcher should prompt for the URL once and write it with `save_config_file`
(`polony_app.config`, writes atomically with mode 600).

## Administrator: handing out the connection string

The lab administrator creates a restricted application role (the app only touches its own schema, `polony`) and
hands the pooled connection string to each researcher through a password manager or another one-to-one secure
channel: never by archived e-mail or chat, never into git, tickets or screenshots. Never hand out the owner role.
The first start creates the tables inside schema `polony` (idempotent, versioned migrations in `polony.schema_version`);
the role needs `USAGE, CREATE` on that schema only.

### Recommended hardening

* **IP allow-list** (in the provider's console): list the lab/institute egress IPs (and
  VPN).  Remember to update it when a laptop moves networks.
* **Rotate the password** when a person leaves or a laptop is lost, and at least once a year:
  reset the role's password in the provider's console,
  then distribute the new string and update `config.json` on every computer.
* Keep `config.json` mode 600; the app never logs the URL (errors show `postgresql://user:***@host/db`).
* Use `sslmode=require` (Neon enforces TLS anyway).
* Take a Neon snapshot / rely on history retention before bulk operations; use a branch for experiments.
* Roles are shared by all researchers: the database cannot tell *people* apart, only computers
  (`machine_id`).  If per-person accounts are needed, create one role per person and grant them the
  schema (`GRANT USAGE, CREATE ON SCHEMA polony`; later roles do not own the tables, so create tables
  once as `polony_app` and `GRANT ALL ON ALL TABLES IN SCHEMA polony`).

## Behaviour and guarantees

* **No silent fallback.**  If a URL is configured but the database cannot be reached the app still
  starts, `GET /api/storage` reports `connected:false` + the (redacted) error, and every data
  request answers `503 {"code":"database_unreachable"}` until the connection returns.  Nothing is
  written to local SQLite in that case.
* **Concurrent editing.**  Every change is one transaction holding a row lock on the analysis
  (`SELECT ... FOR UPDATE`), so two people editing the same analysis are serialised, nothing is lost,
  object ids stay unique, and undo/redo stacks are consistent (they are part of the persisted state).
  The API returns the new analysis after every edit; other computers see a change at their next
  request (reads are served from a cache that is re-validated every 1.5 s, `POLONY_PG_VIEW_TTL`).
  One person's *undo* undoes the shared stack's last action, whoever made it - coordinate when two
  people review the same analysis at the same time.
* **Jobs.**  One detection job per analysis across all computers (partial unique index).  A running
  job writes a heartbeat every 5 s.  On start a computer fails only jobs it owns itself; jobs of other
  computers are failed only when their heartbeat is older than 120 s.  Cancel from any computer sets
  a flag in the database that the running job picks up within ~5 s.
* **Round trips.**  A read served from cache costs 0, a cache re-validation 1, an edit 2 (statements
  are pipelined).  Measured through a TCP proxy with an injected network delay: edit about 75 ms at
  30 ms round trip, 215 ms at 100 ms round trip; cache re-validation = 1 round trip.
* Large state: an analysis state (objects) is rewritten on each edit, same as SQLite.

## Limitations

* Originals are local; every researcher who wants the pixels must hold the TIFF (same SHA-256).
  Exports that embed original files (`bundle` provenance hashing) read the local copy.
* Neon free plan: 0.5 GB storage per project, compute scales to zero after 5 minutes idle; the first
  request after a pause takes about 0.5-3 s while the compute starts (the app shows the normal
  spinner; `GET /api/storage` latency shows it).  Each analysis is a few hundred KB to a few MB.
* Schema upgrades: a newer app may migrate the schema; do not run mixed app versions for long.
* Deleting an analysis removes it for everybody (the local originals of other computers are kept).

## Tests

`pytest tests/app` runs on SQLite.  With `POLONY_TEST_DATABASE_URL` set (a **dev branch** URL of the
restricted role) the same API/store suites also run on PostgreSQL, plus the two-computer tests
(`test_store_pg.py`, `test_store_api.py`).  Every test creates and drops its own schema
`test_<uuid>`; the URL is read only from the environment and is never printed or stored.
The role needs `CREATE` on the *dev branch* database for this (`GRANT CREATE ON DATABASE neondb
TO polony_app` on the dev branch only - not on `main`).
