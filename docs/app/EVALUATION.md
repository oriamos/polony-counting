# Evaluation of detection configurations

`python -m polony_app.ml evaluate --predictions <analysis ids> --reference analysis|roi [--tolerance 5] [--out report.md]` writes a Markdown report and a JSON file next to it.
Model checkpoints are scored on the locked test crops with `eval-model` (see [FINE_TUNING.md](FINE_TUNING.md)).
Nothing here says what is true: the reference is either the **final human-reviewed points** (`analysis`, fully reviewed regions only) or the **researcher's reference marks**
embedded in the TIFF (`roi`, fields that have marks on that page; the file cannot distinguish "counted zero" from "not counted", so fields without marks are not scored).

## How points are compared
Per field and per TIFF page, predictions and reference points are matched **one-to-one** (Hungarian assignment, maximum number of pairs first, then minimum total distance) when closer than the
tolerance (default 5 px). Page index is the TIFF page; colour is shown only when the channel mapping is confirmed.

| metric | plain meaning |
|---|---|
| precision | of the predicted points, the share that has a reference partner |
| recall | of the reference points, the share that was found |
| F1 | harmonic mean of the two |
| mean matching distance | average pixel distance of matched pairs (localisation quality) |
| per-field count MAE | average of \|predicted − reference count\| over fields |
| count bias | average of (predicted − reference) over fields; > 0 over-counts |
| regression slope | line through the origin: predicted = slope × reference (1 = no systematic scale error) |
| Pearson r | correlation of per-field counts |
| false positives / false negatives per field | average unmatched predictions / unmatched reference points |
| % candidates needing review | candidates with a review reason or an engine disagreement |
| agreement / Big-FISH-only / Spotiflow-only acceptance | share of those candidates the reviewer kept |
| human-added rate | human-added points ÷ all final points (what the engines missed) |
| review time per field | estimated from action-log timestamps (gaps over 120 s count as breaks); empty when not measurable |

`undefined` values (e.g. precision with no predictions) are shown as `-`, never as 0.

## Configurations
`bigfish_default`, `bigfish_sensitive` (threshold_scale 0.5), `spotiflow_pretrained`, `spotiflow_finetuned` (model name is a registry id), `agreement_subset` (candidates both engines found), `union_before_review`
(all candidates before any human action), `final_human_reviewed` (against `roi` only; against `analysis` it would equal the reference). Engine-based configurations read the stored raw detections of the
analysis' engine runs (latest run per page); if the store has none, the config is reported as unavailable with the reason. `--predictions id:run1,run2` restricts the runs considered.

## Reading the report
* Sections: **per TIFF page** (always), pooled row (only over pages with enough events, naming the pages it contains and the sparse ones it leaves out), then by slide, field, channel, spot radius and model version.
* Pages are never merged silently: a sparse (e.g. green) page is reported on its own and **not** added into the red totals.
* **preliminary**: set automatically when fewer than 30 fields, fewer than 30 reference events on the page, or only one annotator/review session (reviewer bias unmeasurable). Preliminary numbers are for
  orientation and for deciding where to look, not for claims. The reasons are listed under each table.
* Compare configurations on the *same* analyses and reference only. A better F1 against human-reviewed points can partly be agreement with the reviewer's habits (e.g. accepting what an engine proposed).
* No numbers from earlier validation runs are built into the tool; every figure comes from the analyses you pass in.
