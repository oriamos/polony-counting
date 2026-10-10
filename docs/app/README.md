# Polony Review App

A local, human-in-the-loop web app for reviewing and counting red / green / yellow polonies on two-channel TIFF scans. A classical spot detector (Big-FISH, Laplacian of Gaussian)
proposes positions, you inspect, correct and approve them per region, and the app exports counts together with full provenance. Code: `src/polony_app` (FastAPI + SQLite) and `frontend/` (React + TypeScript).
It does **not** need torch, and it does not use the legacy neural-network code in `src/polony` (see [LEGACY_AUDIT.md](LEGACY_AUDIT.md)).

> **Status / honesty note.** Validated on one real red-only image (`resources/dataset_files/test/test_img.tif`, 2450 x 1438, 2 pages uint16) and synthetic data. The research TIFFs named in the task
> (`29-09-2026_T7_AB_*.tif`, described as 1460 x 2438) were not available, so no accuracy on them is claimed. Defaults and review cut-offs are unvalidated starting points. Concentration output is off until the researcher confirms the formula.

## Quick start (one command)

```bash
make app            # creates .venv, installs requirements-app.txt, builds the frontend if npm exists, starts the server
# open http://127.0.0.1:8000     (API docs: /docs, health: /api/health)
```

Requirements: Python >= 3.10 (tested with 3.13) and, optionally, Node/npm to build the UI (without npm the API still works, and an existing `frontend/dist` is served).
Options: `PORT=8001 make app`, `APP_DATA_DIR=/path make app`, or `make app APP_PY=python` to install into / use your current interpreter instead of `.venv`.

| Command | Does |
|---|---|
| `make app-install` | create `.venv` (if needed) and `pip install -r requirements-app.txt` (no torch) |
| `make app-run` | build the frontend (if npm) and start the server on port 8000 |
| `make app-test` | run `tests/app` (192 tests) |
| `make app` | install + run |

Equivalent without make: `pip install -r requirements-app.txt`, then `PYTHONPATH=src python -m polony_app --data-dir ./data --port 8000`. Poetry users: `poetry install --with app` (group `app`).
Data (originals, SQLite database) live in `./data` (git-ignored; override with `POLONY_DATA_DIR`). Uploads are limited to 512 MB per file (`POLONY_MAX_UPLOAD_MB`).
Frontend development: see `frontend/README.md` (`npm run dev`, mock API, vitest, e2e).

Verified from a fresh venv (Python 3.13): `pip install -r requirements-app.txt` (no torch), `PYTHONPATH=src pytest tests/app` = 192 passed, `make app` serves `/api/health` and `/`.

## Documents

| Document | Content |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | the whole workflow, shortcuts, exports, draft vs final, counting conventions |
| [PARAMETERS.md](PARAMETERS.md) | every detection parameter and review cut-off, exactly |
| [SCIENTIFIC_OPEN_QUESTIONS.md](SCIENTIFIC_OPEN_QUESTIONS.md) | what the researcher must still decide or confirm |
| [LEGACY_AUDIT.md](LEGACY_AUDIT.md) | verified problems of the legacy pipeline and what the app does instead |
| [SCIENCE_NOTES.md](SCIENCE_NOTES.md) | experiments, Big-FISH facts, failure modes |
| [REVIEW_POLICY.md](REVIEW_POLICY.md), [RERUN_POLICY.md](RERUN_POLICY.md), [CONTRACTS.md](CONTRACTS.md) | approval/finalize rules, re-detection merge rules, API contracts |

## Principles

Every automatic result is a proposal; uncertain objects are never counted; colour roles are confirmed by the user, never inferred; pixel size is never assumed; approvals are invalidated by any later change; every export says DRAFT or FINAL; provenance (parameters, versions, hashes, actions) is exported with the counts.
