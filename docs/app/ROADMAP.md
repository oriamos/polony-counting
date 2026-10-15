# Roadmap and open items

What is done, what is known to be incomplete, and what the laboratory still has to decide. Nothing here is a claim of
accuracy; measured results are in [ALL_SLIDES_EVAL.md](ALL_SLIDES_EVAL.md) and [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md).

## Done

* Review of one slide: upload or open from disk, channel mapping, counting grid (click to select fields, Randomise with a
  recorded seed), Big-FISH / Spotiflow / hybrid detection with consensus layers, review queue, per-field approval,
  finalize, exports with provenance, Save next to the original scan in the lab's CSV layout, lab-format Excel export.
* Experiments: import many slides, one grid for all (or randomised per slide), unattended sequential run that skips a
  failed slide and continues, review page for the next morning, one CSV per experiment with provenance.
* Desktop window (pywebview) around the local server; one-command installers for Windows / macOS / Linux; optional AI
  extra with automatic MPS → CPU fallback on Apple Silicon.
* Dataset snapshots from reviewed slides, controlled fine-tuning with a locked test set, model registry; a fine-tuned
  model is shipped in `artifacts/`.

## Known gaps (software)

1. The experiment page shows a slide's grid seed only in the session where Randomise ran; the slide rows of
   `GET /experiments/{id}` do not carry the grid spec yet.
2. Per-slide detection overrides exist in the API but have no UI.
3. The left-rail experiment chip updates live only while that experiment is open.
4. "Pause after current slide" for an overnight run is not implemented (Cancel stops after the current slide).
5. Intel Macs cannot install the AI extra (no current PyTorch wheels); Big-FISH works.
6. The channel alignment estimate (`POST /analyses/{id}/alignment`) has no UI control.
7. Model registry and promotion: the locked test set is checked at evaluation time, but promotion does not yet refuse a
   model whose evaluation did not cover the locked set; see [FINE_TUNING.md](FINE_TUNING.md).

## Open scientific points

* No single Spotiflow threshold fits every slide; precision varies strongly between slides. The reviewer stays in the loop.
* The researcher's embedded marks are not exhaustive, so part of the measured "false positives" may be real polonies; a
  reviewed sample is needed to measure true precision.
* Fine-tuning has shown no evidence of beating the pretrained model on the 8 available slides.
* Decisions that only the laboratory can take (yellow convention, series meaning, concentration inputs, field
  numbering): [SCIENTIFIC_OPEN_QUESTIONS.md](SCIENTIFIC_OPEN_QUESTIONS.md).

## Verification still owed on real machines

* Apple Silicon: `polony-review status` prints `device mps` after `--with-ai`; one hybrid detection completes.
* Windows / macOS: the Desktop icon opens the native window; closing it stops the server when idle and asks when a
  detection runs ([INSTALL.md](INSTALL.md), "Window mode").
