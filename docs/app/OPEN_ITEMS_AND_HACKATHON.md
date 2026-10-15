# Open items and hackathon task list

State at the time of writing: development paused until the hackathon. Everything below is **what is not done or not decided**; nothing here is a claim of completion.
Branch: `main` (`oriamos/polony-counting`; the former working branch `claude/polony-hitl-app` was merged and archived, see [../BRANCHES.md](../BRANCHES.md)). Last full test run: backend 552 passed / 85 skipped, vitest 32/32, mock e2e 203/203, real-model tests pass with `POLONY_SPOTIFLOW_E2E=1`.

## 1. What works (verified)

Upload TIFF, channel mapping, regions, Big-FISH detection, optional Spotiflow detection (CPU, ~21-57 s hybrid), consensus (agreement / bigfish_only / spotiflow_only), review UI with undo/redo, approval, finalize, exports with provenance, lab-format export, shared-Postgres option, one-command installer, dataset snapshot (`ml build`), 8-slide comparison against the researcher's marks.

## 2. Known defects (blockers for any "trained model" claim)

1. **Model registry is not wired into the app.** `resolve_model` is only called inside `polony_app.ml`; `SpotiflowEngine` and `GET /api/engines` do not use it, so a promoted model would never be used. 
2. **Locked test set is not enforced at promotion.** Promotion must verify `lock_sha256`, group-disjointness of train/test, and that `eval-model` checked `test_content_sha256`.
3. **Leakage / identity**: dataset sources should be keyed by content hash (`SourceInfo.sha256`), identity keys should be assay-agnostic, a missing `test_lock.json` should warn.
4. **Checksum tamper**: hash `best.pt` on each run/process, not once.
5. **Provenance gaps**: record real `source_file` / `source_page`, alignment frame and mapping fingerprint on runs, raw detections and exports, and use them in dataset/evaluate.
6. Smaller: `_h_split` evidence/labels; `human_added` through merge; `human_rejected` / agreement double counts; `run_detection` should default `roles=(None,None)` and require `pages`; matcher NaN / role-None handling; rename "Probability threshold" / `spotiflow_prob_raw` to "model score" wording; per-page annotator count; document reclassify semantics; `manifest.channel_mapping` top-level key is ambiguous (per-source list is correct).
7. The installer's job-status polling in `install/ci` waits on the wrong terminal status (the app returns `done`).

## 3. Open scientific results

* Spotiflow alone, 8 slides pooled, 10 px vs researcher marks (valid_unsaturated, ÷4): precision/recall 0.80/0.67 at model default 0.5, 0.73/0.77 at 0.2, 0.70/0.79 at 0.1. No single threshold fits every slide; AS-B stays at precision 0.26-0.37.
* Hybrid barely depends on threshold (pooled ~0.63 / 0.80); Big-FISH dominates the union.
* Density vs Excel: mean absolute deviation ~25%; sparse slides over-count, dense slides under-count.
* Agreement precision varies 0.63-0.93 by slide; disagreements hold 49-92% of the false positives.
* Researcher marks are not exhaustive; part of the "false positives" may be real polonies. Needs a reviewed sample.
* Fine-tuning: no evidence of benefit yet (8 slides only).
* Big-FISH default threshold must **not** be tuned on the 8-slide set.

## 4. Decisions waiting for people

* Researcher (Elena): yellow convention (exclusive vs channel-inclusive), meaning of T7A/T7B, concentration formula and dilution/efficiency inputs, grid-cell/field numbering, which cells are valid. Full list: [SCIENTIFIC_OPEN_QUESTIONS.md](SCIENTIFIC_OPEN_QUESTIONS.md).
* Project owner: RunPod/batch mode (not started); GPU access; how many slides are labelled; whether to enable GitHub Actions on the fork (needed to verify the Windows/macOS installers, never run); public fork vs research TIFFs and lab spreadsheets in `tests/test_data/lab_templates/` (currently tracked in a public repo - decide whether to remove them); Vercel port (paused, see [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md)).
* Security: rotate the Neon `polony_app` password; Neon tests have only run against local PG16 (run with a real connection string from your own shell).

## 5. Hackathon task list (suggested order)

**Before the event (about 1 hour)**
- [ ] Fresh clone, run the installer on the demo laptop (Windows or Mac), start the app, open one slide. Install `--with-ai` only if Spotiflow will be shown.
- [ ] Pre-download the Spotiflow `general` model (`polony-review prepare-model`) - no internet needed on stage.
- [ ] Choose the demo slides (suggest one dense and one sparse) and keep a saved project for fallback.
- [ ] Run the full test suites once (`make app-test`, `npm test` in `frontend/`).

**During the hackathon, by value**
- [ ] 1. Fix blockers 2.1 and 2.2 (registry wiring, locked-test enforcement) - makes the learning loop real.
- [ ] 2. Per-slide threshold calibration from ~30-40 reviewed points (no training required).
- [ ] 3. Reviewed sample of "false positives" to measure true precision against the researcher's marks.
- [ ] 4. Fine-tuning experiment: train on 6 slides, test on 2 unseen slides, promote only if it wins on the locked set.
- [ ] 5. Provenance gaps 2.5; smaller items 2.6.
- [ ] 6. Optional: RunPod/batch mode, if GPU access exists.

**Demo flow (5 minutes)**: upload slide, map channels, run hybrid, show agreement vs disagreement overlay, review a few disagreements (debris-like Big-FISH-only are queued first), undo/redo, approve, export lab-format file plus provenance, show the comparison to the researcher's marks.

**What to say honestly**: partial accuracy (see section 3), reviewer stays in the loop, uncertain objects are never counted, no training happens automatically.

## 6. Repository notes

* Research TIFFs are intentionally not in git (`resources/research_tiffs/`, local only).
* `frontend/src/lib` was previously hidden by a `.gitignore` rule (`lib/`) and is now tracked; `src/polony_app/webdist/` is the prebuilt UI (rebuild with `make web`).
* Legacy neural-network code (`src/polony`) is untouched and not used by the app ([LEGACY_AUDIT.md](LEGACY_AUDIT.md)).
