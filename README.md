[![codecov](https://codecov.io/gh/oriamos/polony-counting/graph/badge.svg?token=3XYNQ0GYTB)](https://codecov.io/gh/oriamos/polony-counting)
![GitHub release (with filter)](https://img.shields.io/github/v/release/oriamos/polony-counting?sort=semver&color=brightgreen)
[![Linting](https://github.com/oriamos/polony-counting/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/oriamos/polony-counting/actions/workflows/lint.yml)
[![Testing](https://github.com/oriamos/polony-counting/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/oriamos/polony-counting/actions/workflows/test.yml)
[![Documentation](https://github.com/oriamos/polony-counting/actions/workflows/pages.yml/badge.svg)](https://github.com/oriamos/polony-counting/actions/workflows/pages.yml)

> **Hackathon winner.** This project won the hackathon. The final state lives on `main`: the Polony Review app, the fine-tuned Spotiflow model (`artifacts/ft-7e4c89cb-8edc09e3`), the locked test-set manifest and the comparison scripts (`scripts/compare_to_researcher.py`, `scripts/calibration/`). Hebrew briefing: [docs/app/HACKATHON_BRIEF_HE.pdf](docs/app/HACKATHON_BRIEF_HE.pdf). Repository layout and branch status: [docs/BRANCHES.md](docs/BRANCHES.md).

## Install Polony Review (one command, no admin rights)

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/oriamos/polony-counting/main/install/install.ps1 | iex
```
**macOS / Linux** (Terminal):
```bash
curl -fsSL https://raw.githubusercontent.com/oriamos/polony-counting/main/install/install.sh | bash
```
This installs the app into your user folder (about 650 MB, downloads Python 3.12 via `uv` only if no Python 3.10-3.13 is present)
and puts a **Polony Review** icon on your Desktop. Re-run the command to update. Optional AI detector (Spotiflow + PyTorch, +0.8-2.5 GB): add `--with-ai` (`-WithAi` on Windows). The app has no login and listens on `127.0.0.1` only.
Details, uninstall, troubleshooting and test status: [docs/app/INSTALL.md](docs/app/INSTALL.md).

## Where to find what

| I want to... | Read |
|---|---|
| Install the app | [docs/app/INSTALL.md](docs/app/INSTALL.md) (or the one-command installer above) |
| Use the app (upload, review, export) | [docs/app/USER_GUIDE.md](docs/app/USER_GUIDE.md) |
| Train / fine-tune a model | [docs/app/FINE_TUNING.md](docs/app/FINE_TUNING.md) (from reviewed corrections), [docs/app/LEGACY_FINETUNING.md](docs/app/LEGACY_FINETUNING.md) (from the historical ImageJ TIFFs) |
| Evaluate a model | [docs/app/EVALUATION.md](docs/app/EVALUATION.md) |

## Polony Review App (new)

A local human-in-the-loop web app (FastAPI + React) for reviewing and approving red / green / yellow polony counts with full provenance. Two independent detectors: Big-FISH (classical, no torch) and optional Spotiflow (AI, needs the `--with-ai` extra); a deterministic consensus between them; a lab-format Excel/CSV export; optional shared Postgres database.

```bash
make app        # venv + dependencies + frontend build (if npm) + server on http://127.0.0.1:8000
make app-test   # backend tests (552 passed / 85 skipped in the last full run; skips need Postgres or the real model)
```

Docs: [docs/app/README.md](docs/app/README.md) (start here), [user guide](docs/app/USER_GUIDE.md), [parameters](docs/app/PARAMETERS.md),
[open scientific questions](docs/app/SCIENTIFIC_OPEN_QUESTIONS.md).
**Status:** end-to-end validated on 8 real two-channel slides against the researcher's own marks (Big-FISH and Spotiflow); accuracy is partial (see [ALL_SLIDES_EVAL.md](docs/app/ALL_SLIDES_EVAL.md)) and defaults are not final. **Open items and the hackathon task list: [docs/app/OPEN_ITEMS_AND_HACKATHON.md](docs/app/OPEN_ITEMS_AND_HACKATHON.md).**

## Background

Polony is a solid-phase single-molecule PCR method (Mitra & Church, 1999), adapted at the Lindell lab, Technion, to quantify marine viruses of a given clade in field samples. After the wet-lab stage, each slide is scanned (one RGB image plus red and green channel images) and the polonies (red, green, yellow spots) are counted. Polony Review automates detection and keeps a researcher in the loop to review and approve every count.

## Feedback

Open an issue on the GitHub repository for bugs or suggestions.

## License

MIT, see [LICENSE](LICENSE).
