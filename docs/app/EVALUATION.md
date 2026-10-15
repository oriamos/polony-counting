# Evaluation of detection configurations

`python -m polony_app.ml evaluate --predictions <analysis ids> --reference analysis|roi [--tolerance 10] [--out report.md]` writes a Markdown report and a JSON file next to it.
Model checkpoints are scored on the locked test crops with `eval-model` (see [FINE_TUNING.md](FINE_TUNING.md)).
Nothing here says what is true: the reference is either the **final human-reviewed points** (`analysis`, fully reviewed regions only) or the **researcher's reference marks**
embedded in the TIFF (`roi`, fields that have marks on that page; the file cannot distinguish "counted zero" from "not counted", so fields without marks are not scored).

## How points are compared
Per TIFF page, predictions and reference points are matched **one-to-one** (Hungarian assignment, maximum number of pairs first, then minimum total distance) when closer than the
tolerance, **once over the union of the page's evaluated fields**; each pair is then credited to the field of its reference point, so a pair that straddles a field border is not lost.
Page index is the TIFF page; colour is shown only when the channel mapping is confirmed.

**Tolerance.** The default is **10 px**, and **5, 10 and 15 px are always reported side by side**. A 5 px radius mostly measures the click scatter of the annotator
(two people marking the same polony differ by about that much), not whether the spot was detected; read the three columns together, and prefer the one that matches the physical spot size.

| metric | plain meaning |
|---|---|
| precision (= precision **vs this annotator's marks**) | of the predicted points, the share that has a reference partner. A real polony the annotator did not mark counts as a false positive, so this is a lower bound on the true precision |
| recall | of the reference points, the share that was found |
| F1 | harmonic mean of the two |
| mean matching distance | average pixel distance of matched pairs (localisation quality) |
| per-field count MAE | average of \|predicted − reference count\| over fields |
| count bias | average of (predicted − reference) over fields; > 0 over-counts |
| OLS slope and intercept | least-squares line predicted = intercept + slope × reference over fields (slope 1 and intercept 0 = no systematic error). A line through the origin is not used: it hides offsets (predicted = reference + 3 would show slope ≈ 1.1) |
| mean bias and limits of agreement | Bland–Altman, per slide: mean of (predicted − reference) and bias ± 1.96·SD |
| Pearson r (within slide) | correlation of per-field counts **inside one slide**; reported per slide with its median |
| pooled Pearson r | over several slides, labelled **between-slide (not accuracy)**: slides differ in density, so it is high even when the counts of fields of one slide are unrelated to the reference (synthetic example: within-slide r ≈ 0, pooled 0.96; real Spotiflow: pooled 0.95, within-slide mean 0.65) |
| false positives / false negatives per field | average unmatched predictions / unmatched reference points |
| % candidates needing review | candidates with a review reason or an engine disagreement |
| agreement / Big-FISH-only / Spotiflow-only acceptance | share of those candidates the reviewer kept |
| human-added rate | human-added points ÷ all final points (what the engines missed) |
| review time per field | estimated from action-log timestamps (gaps over 120 s count as breaks); empty when not measurable |

`undefined` values (e.g. precision with no predictions) are shown as `-`, never as 0.

## Configurations
`bigfish_default`, `bigfish_sensitive` (threshold_scale 0.5), `spotiflow_pretrained`, `spotiflow_finetuned` (model name is a registry id; fields below), `agreement_subset` (candidates both engines found), `union_before_review`
(all candidates before any human action), `final_human_reviewed` (against `roi` only; against `analysis` it would equal the reference). Engine-based configurations read the stored raw detections of the
analysis' engine runs; if the store has none, the config is reported as unavailable with the reason.

**Spotiflow runs are not one configuration.** Each distinct (downscale, probability threshold actually used, normaliser, model checksum, crop-to-valid / crop box) is
reported as its own configuration, e.g. `spotiflow_pretrained [downscale=4, thr=0.45, normaliser=valid_unsaturated, model=ab12cd34]`. All configurations found are listed
("Engine configurations found") and none is preferred or merged; only runs with an *identical* configuration on the same page collapse to the most recent one, with a warning. `--predictions id:run1,run2` restricts the runs considered.

## Reading the report
* Sections: **per TIFF page** (always), pooled row (only over pages with enough events, naming the pages it contains and the sparse ones it leaves out), then by slide, field, channel, spot radius and model version.
* Pages are never merged silently: a sparse (e.g. green) page is reported on its own and **not** added into the red totals.
* **preliminary**: set automatically when fewer than 30 fields, fewer than 30 reference events on the page, or only one annotator/review session (reviewer bias unmeasurable). Preliminary numbers are for
  orientation and for deciding where to look, not for claims. The reasons are listed under each table.
* Compare configurations on the *same* analyses and reference only. A better F1 against human-reviewed points can partly be agreement with the reviewer's habits (e.g. accepting what an engine proposed).
* **Count agreement** is reported per slide and page (OLS slope, intercept, bias, limits of agreement, within-slide r) with the median per-slide r; the pooled r is labelled between-slide.
* **Annotation completeness.** When raw detections are available the report adds, per slide, how many raw proposals inside the reviewed fields the annotator left unmarked and how many of those are
  *bright* (score at least the lower quartile of the confirmed proposals). Some are real polonies nobody marked, so precision is a lower bound; the same counts are in dataset manifests.
* **Model score.** The Spotiflow score is a heatmap value at the peak, **not a calibrated probability**: measured against reviewer decisions the expected calibration error is about 0.27–0.30 and the AUC about 0.55–0.64.
  It is therefore called "model score" everywhere and must not be read as a probability or used as a confidence statement.
* No numbers from earlier validation runs are built into the tool; every figure comes from the analyses you pass in.
