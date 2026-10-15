# Branch history

The repository is consolidated on **`main`**; every other branch is inactive (no further work happens there).
The branches below still exist on GitHub and can all be deleted safely.

| Branch | Content | State at archive time |
|---|---|---|
| `claude/polony-hitl-app` | Polony Review app (Big-FISH / Spotiflow, FastAPI + React), installers | fully merged into `main` |
| `claude/legacy-import` | legacy dataset import and fine-tuning bridge | fully merged into `main` |
| `docs/hackathon-brief-hitl` | Hebrew hackathon briefing PDF | PDF copied to `docs/app/HACKATHON_BRIEF_HE.pdf` on `main` |
| `data/research-samples-2026-09-29` | eight research TIFFs (zipped) and two reference CSVs | **not merged**, data not needed; PR #1 closed |

## Repository map

| Path | What |
|---|---|
| `src/polony_app` | the Polony Review app backend (FastAPI + SQLite) and ML tooling |
| `frontend/` | React + TypeScript UI |
| `src/polony` | legacy neural-network pipeline (known defects, see `docs/legacy-repo-audit.md`) |
| `artifacts/` | winning fine-tuned model, model registry, locked test-set manifest |
| `scripts/` | runbooks, calibration sweeps, comparison to the researcher's marks |
| `install/` | one-command installers (Windows / macOS / Linux) |
| `docs/app/` | app documentation (start at `docs/app/README.md`) |
