[![codecov](https://codecov.io/gh/oriamos/polony-counting/graph/badge.svg?token=3XYNQ0GYTB)](https://codecov.io/gh/oriamos/polony-counting)
![GitHub release (with filter)](https://img.shields.io/github/v/release/oriamos/polony-counting?sort=semver&color=brightgreen)
[![Linting](https://github.com/oriamos/polony-counting/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/oriamos/polony-counting/actions/workflows/lint.yml)
[![Testing](https://github.com/oriamos/polony-counting/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/oriamos/polony-counting/actions/workflows/test.yml)
[![Documentation](https://github.com/oriamos/polony-counting/actions/workflows/pages.yml/badge.svg)](https://github.com/oriamos/polony-counting/actions/workflows/pages.yml)

# Polony Review

A desktop program for counting red / green / yellow polonies on two-channel microscope scans, with a researcher in the loop.
Two independent detectors propose spots (Big-FISH, classical; Spotiflow, AI, optional), you review and approve per counting
field, and the app writes the counts in the lab's own CSV / Excel layouts with full provenance. Everything runs on your own
computer; nothing is uploaded.

> **Hackathon winner.** The final state lives on `main`: the app, the fine-tuned Spotiflow model (`artifacts/ft-7e4c89cb-8edc09e3`), the locked test-set manifest and the comparison scripts (`scripts/compare_to_researcher.py`, `scripts/calibration/`). Hebrew briefing: [docs/app/HACKATHON_BRIEF_HE.pdf](docs/app/HACKATHON_BRIEF_HE.pdf). Branch history: [docs/BRANCHES.md](docs/BRANCHES.md).

## Install (one command, no admin rights)

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/oriamos/polony-counting/main/install/install.ps1 | iex
```
**macOS / Linux** (Terminal):
```bash
curl -fsSL https://raw.githubusercontent.com/oriamos/polony-counting/main/install/install.sh | bash
```

This installs the app into your user folder (about 650 MB; downloads Python 3.12 only if no Python 3.10-3.13 is present) and
puts a **Polony Review** icon on your Desktop. The icon opens the app in its own window. Re-run the command to update.

* **AI detector** (Spotiflow + PyTorch, +0.8-2.5 GB): add `--with-ai` (`-WithAi` on Windows). On Apple Silicon Macs it uses the
  GPU (MPS) and falls back to the CPU by itself when an operation is missing. Intel Macs: Big-FISH only.
* Details, uninstall, troubleshooting (including Spotiflow on Mac): [docs/app/INSTALL.md](docs/app/INSTALL.md).

## What you can do

| | |
|---|---|
| **Review one slide** | open a TIFF, confirm which channel is red / green, generate the counting grid, **click the fields to count**, run detection, review the queue, approve, save `<name>_labeled.tif` + CSV next to the scan |
| **Randomise the grid** | one button moves the whole grid to a random position (same field size), seed recorded for reproducibility |
| **Run a whole experiment overnight** | create an experiment, drop all its slides in, apply one grid to all (or randomise on all), press Run and go home; a failed slide is skipped and the run continues. Next morning: review slide by slide, finalize, **download one CSV for the experiment** |
| **Export in the lab's formats** | `experiment_counts.csv`, T7-style and Counts-style Excel workbooks with the lab formula, plus provenance (parameters, model, seeds, hashes, every action) |
| **Train / evaluate a model** | dataset snapshots from reviewed slides, controlled fine-tuning, locked test set |

## Where to find what

| I want to... | Read |
|---|---|
| Install, update, uninstall, fix an install | [docs/app/INSTALL.md](docs/app/INSTALL.md) |
| Use the app, step by step | [docs/app/USER_GUIDE.md](docs/app/USER_GUIDE.md) (experiments: section 12; shortcuts: section 17) |
| Understand a parameter or a review flag | [docs/app/PARAMETERS.md](docs/app/PARAMETERS.md) |
| Train / fine-tune a model | [docs/app/FINE_TUNING.md](docs/app/FINE_TUNING.md) (from reviewed corrections), [docs/app/LEGACY_FINETUNING.md](docs/app/LEGACY_FINETUNING.md) (from the historical ImageJ TIFFs) |
| Evaluate a model | [docs/app/EVALUATION.md](docs/app/EVALUATION.md), results so far: [docs/app/ALL_SLIDES_EVAL.md](docs/app/ALL_SLIDES_EVAL.md) |
| Call the API (experiments, batch runs) | [docs/app/EXPERIMENTS_API.md](docs/app/EXPERIMENTS_API.md), live docs at `/docs` while the app runs |
| See what is still open | [docs/app/OPEN_ITEMS_AND_HACKATHON.md](docs/app/OPEN_ITEMS_AND_HACKATHON.md), [docs/app/SCIENTIFIC_OPEN_QUESTIONS.md](docs/app/SCIENTIFIC_OPEN_QUESTIONS.md) |
| All documents | [docs/app/README.md](docs/app/README.md) |

## For developers

```bash
make app        # venv + dependencies + frontend build (if npm) + server on http://127.0.0.1:8000
make app-test   # backend tests: 1237 passed / 140 skipped; 10 need torch/spotiflow or a GPU
make web        # rebuild the committed UI in src/polony_app/webdist after a frontend change
cd frontend && npm test && npm run e2e    # vitest + mock end-to-end (248 checks)
```

Backend: `src/polony_app` (FastAPI, SQLite, optional shared Postgres). Frontend: `frontend/` (React + TypeScript).
**Status:** validated end to end on 8 real two-channel slides against the researcher's own marks; accuracy is partial
([ALL_SLIDES_EVAL.md](docs/app/ALL_SLIDES_EVAL.md)) and defaults are not final. Uncertain objects are never counted, nothing
is counted as final without a person's approval, and no model is trained or switched automatically.

## Background

Polony is a solid-phase single-molecule PCR method (Mitra & Church, 1999), adapted at the Lindell lab, Technion, to quantify marine viruses of a given clade in field samples. After the wet-lab stage, each slide is scanned (one RGB image plus red and green channel images) and the polonies (red, green, yellow spots) are counted per grid field. Polony Review automates the detection and keeps the researcher in charge of every count.

## Feedback

Open an issue on the GitHub repository for bugs or suggestions.

## License

MIT, see [LICENSE](LICENSE).
