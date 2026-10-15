# All 8 raw slides: engine comparison against the researcher's embedded marks (preliminary)

> **Which columns are current.** The two Big-FISH columns (default, sensitive) are current. The two Spotiflow-based columns (Spotiflow ÷4 and agreement, both labelled "legacy auto normalisation") were obtained with the saturation-sensitive `auto` Spotiflow normalisation (and the zoom-based ÷4) and are superseded: the current Spotiflow numbers are the per-slide, per-threshold tables of [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md) (`normalizer = valid_unsaturated`, block-mean ÷4, 5 thresholds, 8 slides). The text below the table is kept as the comparison of the first run. In short, at ÷4 with `valid_unsaturated` Spotiflow alone gives (precision/recall at 10 px) AI-A 0.88/0.81, AI-B 0.83/0.79, AM-A 0.70/0.64, AM-B 0.60/0.51, AS-A 0.69/0.67, AS-B 0.29/0.88, FS-A 0.85/0.82, FS-B_D 0.61/0.84 at threshold 0.2; with the legacy normalisation the recall on AS-A / AS-B was only 0.16 / 0.07 at the model's default threshold. Not validated; the threshold was not tuned per slide.


Run through the real app (hybrid mode): Big-FISH default (engine=bigfish), Big-FISH sensitive (threshold_scale 0.5) + Spotiflow `general` (÷4, prob_thresh 0.1) as independent engines. TIFF page 0 only, only inside the researcher fields imported from the TIFF (mode "counted", plane 0). FS-B_D was counted on a grid shifted by about (−158,−34) relative to the other slides, and the overlay stored in that file is itself the shifted grid (see [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md) stage 4); the shift is a finding of the validation, not confirmed by the researcher. Reference = the researcher's embedded marks (one annotator). "marks" in the table counts the marks inside the imported fields, which can be lower than all marks embedded in the file (FS-B_D: 82 here, 151 embedded marks in REAL_DATA_VALIDATION.md: different scope, see REAL_DATA_VALIDATION.md). One-to-one matching, 10 px (and 6 px below). CPU. **Nothing here is validated accuracy; Spotiflow settings were chosen on AI-A; Big-FISH defaults unchanged.** "Precision/recall" = agreement with the researcher's marks, not truth (marks may be incomplete).

| slide | marks | Big-FISH default P/R | Big-FISH sensitive P/R | Spotiflow ÷4 P/R (legacy auto normalisation) | agreement P/R (legacy) |
|---|---|---|---|---|---|
| AI-A | 332 | 0.89 / 0.64 | 0.86 / 0.75 | 0.89 / 0.55 | 0.90 / 0.55 |
| AI-B | 400 | 0.83 / 0.69 | 0.81 / 0.77 | 0.86 / 0.66 | 0.86 / 0.64 |
| AM-A | 100 | 0.50 / 0.73 | 0.46 / 0.77 | 0.79 / 0.55 | 0.79 / 0.53 |
| AM-B | 103 | 0.46 / 0.46 | 0.44 / 0.54 | 0.59 / 0.40 | 0.61 / 0.40 |
| AS-A | 82 | 0.48 / 0.71 | 0.45 / 0.83 | 0.76 / 0.45 | 0.76 / 0.45 |
| AS-B | 60 | 0.25 / 0.50 | 0.27 / 0.70 | 0.62 / 0.25 | 0.62 / 0.25 |
| FS-A | 182 | 0.84 / 0.74 | 0.80 / 0.81 | 0.85 / 0.86 | 0.87 / 0.81 |
| FS-B_D | 82 | 0.60 / 0.83 | 0.56 / 0.84 | 0.57 / 0.84 | 0.58 / 0.82 |

Big-FISH-only points (sensitive vs Spotiflow) have low precision on the sparse slides (AM-A 0.24, AM-B 0.14, AS-A 0.30, AS-B 0.21 vs marks) and Spotiflow-only points are rare (0–13 per slide). At 6 px the same pattern holds with lower numbers (AI-A sensitive 0.68/0.60 …).

Field-level (final hybrid objects, red-positive vs the researcher's per-field counts; app `/fields/compare`): AI-A r 0.83 slope 0.81 MAE 3.5; AI-B r 0.82 slope 0.91 MAE 2.6; FS-A r 0.94 slope 0.97 MAE 1.6 (7 fields); **sparse slides poor**: AM-A r 0.26 slope 1.24; AM-B r 0.04 slope 0.89; AS-A r 0.41 slope 1.21; AS-B r 0.49 slope 1.29 (counts per field are small, 0–8, so r is noisy and the app over-counts 21–29 % on AM-A/AS-A/AS-B); FS-B_D r −0.86 over 5 fields (field numbering and the grid shift are unconfirmed — do not interpret).

Green page (nearly empty in most slides): Big-FISH proposes 22–79 valid "green" points on every slide (the researcher's plane-1 marks per slide: 0, 14, 0, 2, 0, 1, 0, 1, i.e. only 4 of the 8 slides carry any) — mostly debris; Spotiflow ÷4 proposes 0–8 on the non-FS slides but 68 / 55 on FS-A / FS-B_D (brighter scans; unknown whether real signal, bleed or debris).

## Reading
- Two regimes: dense slides (AI, FS) — all engines roughly agree with the marks (P 0.8–0.9, R 0.55–0.86); sparse slides (AM, AS) — Big-FISH over-detects (precision 0.25–0.50), Spotiflow ÷4 is more precise (0.59–0.79) but finds only 25–55 % of the marks.
- No single setting is best across slides; this is the case for per-slide settings, a review queue prioritising disagreements, and fine-tuning on confirmed data (leave-one-slide-out evaluation).
- Limits: 8 slides from one session, one annotator, incomplete-annotation bias, Spotiflow settings tuned on one slide, FS-B_D grid shift unconfirmed, green page not evaluable.
- `GET /raw-detections` without `run_id` returns the latest run per (engine, page); `all_runs=true` returns the full audit trail.
