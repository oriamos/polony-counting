# Branch history

The repository is consolidated on **`main`**. All other branches were archived as git tags
(`archive/<branch-name>`) and then deleted, so nothing was lost. To restore one:

```bash
git fetch --tags
git checkout -b <branch-name> archive/<branch-name>
```

| Archived tag | Former branch | Content | State at archive time |
|---|---|---|---|
| `archive/claude/polony-hitl-app` | `claude/polony-hitl-app` | Polony Review app (Big-FISH / Spotiflow, FastAPI + React), installers | fully merged into `main` |
| `archive/claude/legacy-import` | `claude/legacy-import` | legacy dataset import and fine-tuning bridge | fully merged into `main` |
| `archive/docs/hackathon-brief-hitl` | `docs/hackathon-brief-hitl` | Hebrew hackathon briefing PDF | PDF copied to `docs/app/HACKATHON_BRIEF_HE.pdf` on `main` |
| `archive/data/research-samples-2026-09-29` | `data/research-samples-2026-09-29` | eight research TIFFs (zipped) and two reference CSVs | **not merged** (large data); PR #1 closed unmerged |

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
