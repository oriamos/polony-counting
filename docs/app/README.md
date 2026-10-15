# Polony Review App

A local, human-in-the-loop app (own desktop window, or a browser tab) for reviewing and counting red / green / yellow polonies on two-channel TIFF scans. A classical spot detector (Big-FISH, Laplacian of Gaussian)
proposes positions, you inspect, correct and approve them per region, and the app exports counts together with full provenance. Code: `src/polony_app` (FastAPI + SQLite) and `frontend/` (React + TypeScript).
The base install does **not** need torch; the optional Spotiflow engine does (`--with-ai`).

> **Status / honesty note.** Validated end-to-end on 8 real two-channel slides against the researcher's marks ([ALL_SLIDES_EVAL.md](ALL_SLIDES_EVAL.md), [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md)). Accuracy is partial and varies by slide; defaults and review cut-offs are working values, not final. Everything still open: [ROADMAP.md](ROADMAP.md).

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
| `make app-test` | run `tests/app` (backend suite) |
| `make app` | install + run |

Equivalent without make: `pip install -r requirements-app.txt`, then `PYTHONPATH=src python -m polony_app --data-dir ./data --port 8000`.
Data (originals, SQLite database) live in `./data` (git-ignored; override with `POLONY_DATA_DIR`). Uploads are limited to 512 MB per file (`POLONY_MAX_UPLOAD_MB`).
Frontend development: see `frontend/README.md` (`npm run dev`, mock API, vitest, e2e).

Verified from a fresh venv (Python 3.13): `pip install -r requirements-app.txt` (no torch), `PYTHONPATH=src pytest tests/app` (latest full run: 1237 passed / 140 skipped; 10 tests need torch/spotiflow or a GPU), `make app` serves `/api/health` and `/`.

## Documents

| Document | Content |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | the whole workflow: one slide (grid by clicking, Randomise, review, Save) and experiments (overnight run, one CSV), shortcuts, exports, counting conventions |
| [PARAMETERS.md](PARAMETERS.md) | every detection parameter and review cut-off, exactly |
| [SCIENTIFIC_OPEN_QUESTIONS.md](SCIENTIFIC_OPEN_QUESTIONS.md) | what the researcher must still decide or confirm |
| [SCIENCE_NOTES.md](SCIENCE_NOTES.md) | experiments, Big-FISH facts, failure modes |
| [ROADMAP.md](ROADMAP.md) | what is done, known gaps, open scientific points |
| [INSTALL.md](INSTALL.md), [SHARED_DATABASE.md](SHARED_DATABASE.md) | one-command install, optional shared Postgres database |
| [LAB_EXPORT.md](LAB_EXPORT.md), [FIELDS.md](FIELDS.md) | lab-format export and field/reference handling |
| [EXPERIMENTS_API.md](EXPERIMENTS_API.md) | experiments: import, unattended batch run, experiment CSV / provenance (API contract) |
| [TWO_ENGINE.md](TWO_ENGINE.md), [TWO_ENGINE_CONTRACT.md](TWO_ENGINE_CONTRACT.md), [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md) | Big-FISH + Spotiflow, consensus, contracts |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md), [FINE_TUNING.md](FINE_TUNING.md), [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md), [EVALUATION.md](EVALUATION.md), [ALL_SLIDES_EVAL.md](ALL_SLIDES_EVAL.md) | dataset snapshots, controlled fine-tuning, fine-tuning on the historical ImageJ TIFFs, evaluation |
| [DATA_FORMATS.md](DATA_FORMATS.md), [HISTORICAL_IMPORT.md](HISTORICAL_IMPORT.md) | the lab's TIFF / ROI / workbook conventions; importing historical annotations |
| [REVIEW_POLICY.md](REVIEW_POLICY.md), [RERUN_POLICY.md](RERUN_POLICY.md), [CONTRACTS.md](CONTRACTS.md) | approval/finalize rules, re-detection merge rules, API contracts |

## Principles

Every automatic result is a proposal; uncertain objects are never counted; colour roles are confirmed by the user, never inferred; pixel size is never assumed; approvals are invalidated by any later change; every export says DRAFT or FINAL; provenance (parameters, versions, hashes, actions) is exported with the counts.
