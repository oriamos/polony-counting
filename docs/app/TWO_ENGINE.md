# Two-engine detection: Big-FISH + Spotiflow

Status of the design: see [TWO_ENGINE_CONTRACT.md](TWO_ENGINE_CONTRACT.md). This file documents the engine and
matching layers (`src/polony_app/engines/`, `consensus/`, `detection/pipeline.py`, `detection/two_engine.py`).
Other layers (review queue, export, dataset / fine-tuning) are described by their owners further down this file
or in their own documents.

## 1. Architecture

```
page (original dtype, never 8-bit)
   |-- Big-FISH  (background correction -> LoG -> threshold -> measure)  -> RawDetection[bigfish]
   |-- Spotiflow (percentile normalisation inside the model -> heatmap)  -> RawDetection[spotiflow]
                                   |
                  consensus/matching.py  (per page/channel, one-to-one, tolerance px)
                                   |
        agreement | bigfish_only | spotiflow_only   -> union candidates
                                   |
   existing cross-channel association (detection/matching.py) -> red / green / yellow / uncertain objects
                                   |
           existing review-reason code (+ bigfish_only / spotiflow_only)
```

* `DetectionRequest.engine`: `bigfish` (default, behaviour unchanged, Spotiflow never imported), `spotiflow`
  (Spotiflow only, no consensus status) or `hybrid` (both, independently, per channel page).
* Every engine run on a page produces one `EngineRunInfo` (model name / version / sha256, parameters,
  preprocessing, device, versions, duration, status) and `RawDetection` records in full-resolution pixel
  coordinates (x = column, y = row, pixel centre at the integer). Raw detections are never edited; detections
  outside the counting area / inside an exclusion are KEPT with `valid=false` and an `excluded_reason`. Validity
  uses the single region rule of `detection/masks.py` (half-open, pixel-centre, first counting region wins,
  exclusions win), plus the automatic saturated exclusion and the green alignment strip (no data there).
* Ids are deterministic per run (`bf-p0-000001`, `sf-p1-000007`, numbered in (y, x) order); the run id is a
  random uuid per execution (reruns append, they do not overwrite).
* Pages and colours: each channel is run on its own TIFF page index (`pages=(red_page, green_page)` argument of
  `run_detection`). The colour role recorded on a run comes only from the user-confirmed channel mapping
  (`roles=`); with an unconfirmed mapping the role is `None`. Page order is never used to infer colour.
* Scores: Big-FISH `score` = LoG response in noise units (`bigfish_log_response_noise_units`, the unit of the
  detection threshold); Spotiflow `score` = the model's heatmap value at the detected point
  (`spotiflow_prob_raw`, from `details.prob`). Neither is a calibrated probability or a "confidence %"; the UI,
  CSV and docs call them "LoG response (noise units)" and "model score". The two scales are not comparable.

### Hybrid details
* Big-FISH runs exactly as in the Big-FISH-only mode (same threshold cache, tiling, preview-window logic); its
  objects are not altered by the presence of Spotiflow.
* Only VALID detections of both engines are matched. Big-FISH spots outside the valid area flow through the
  pipeline unchanged (and are dropped by the existing region rule); invalid Spotiflow detections stay raw only.
* `agreement`: the object keeps the **Big-FISH** `RawSpot` (position, background-corrected peak, local
  background, SNR, shape extras exactly as in Big-FISH-only mode); the Spotiflow id / x / y / model score and the
  match distance are added to the channel evidence. No review reason is added for plain agreement.
* `bigfish_only`: keeps the Big-FISH spot; review reason `bigfish_only`.
* `spotiflow_only`: a `RawSpot` is created at the exact Spotiflow coordinates and its peak, local background,
  SNR, saturation, elongation and scale ratio are **measured** by the same code as every Big-FISH spot
  (`BigFishEngine._measure`) on the same background-corrected image. Caveats: the measurement interpolates the
  peak at its own LoG-refined centre next to the (rounded) Spotiflow pixel, the LoG response is a Big-FISH
  quantity, and the Big-FISH threshold is attached, so `near_threshold` / `weak_signal` will often (rightly)
  fire for such spots. Review reason `spotiflow_only`.
* Nothing is auto-rejected on either side. Association between red and green, `ambiguous_match`, `dense_cluster`,
  `near_edge`, ... then run unchanged on the union.
* Spotiflow unavailable / failing in `hybrid`: the result is the Big-FISH result, the failed runs are returned
  (`EngineRunInfo.status='failed'` with the error text) and a `WARNING` note is added; the detection job still
  succeeds. In `spotiflow` mode the failure raises `EngineError` with a clear message. A failed run never
  silently yields an empty result.
* Progress (hybrid): Big-FISH red 2-7 %, Spotiflow red 7-40 %, Big-FISH green 40-45 %, Spotiflow green 45-78 %,
  association from 88 %. Cancellation is checked between steps and, for tiled Spotiflow inference, between tiles
  (an untiled single-tile inference cannot be interrupted once started).
* Preview windows also work with `hybrid`; Spotiflow then sees only the padded window crop, so its automatic
  normalisation can differ slightly from the full run, and preview is much slower than with Big-FISH alone
  (CPU: seconds per megapixel).

## 2. Matching (`consensus/matching.py`)

Per (page, channel role) group: candidate pairs = all (Big-FISH, Spotiflow) pairs at distance <= tolerance
(default `consensus_tolerance_px = 4`); per connected component of the candidate graph a Hungarian assignment
(`scipy.optimize.linear_sum_assignment`) maximises the number of pairs and then minimises the summed distance.
Inputs are sorted by (y, x, id), so the result does not depend on input order. One-to-one: a detection is used
at most once. Each side keeps its own coordinates (nothing is averaged); the distance is stored.
`ambiguous_with` lists the other-engine detections inside the tolerance whose distance competes with the chosen
one (d2 - d1 < 1.5 px or d2 < 1.25 d1, the margins of the red/green association); it is informational and
appears in the object history, it does not change classes.

### What agreement and disagreement mean
* **Agreement** = both algorithms put a spot within a few pixels of each other on this channel page. It is
  evidence that the spot is a well-defined local intensity peak. It is **not ground truth**: both detectors can
  fire on the same debris, speck, or saturated rim, and they share the same image. Agreement objects are
  therefore not "confirmed" and are NOT reviewed unless sampled: a fixed-size random QC sample of them (default 40 per slide, `qc_sample`) can be sent to the reviewer. Measured on 7 research slides, 15 % of agreements (pooled; 90 % of the debris_like ones) were not matched to a researcher mark. Disagreements are 24 % to 85 % of the candidates depending on the slide, so they are not a fixed share.
* **bigfish_only / spotiflow_only** = the other engine found nothing within the tolerance. Reasons are many and
  mostly benign: a weak or crowded spot, different sensitivity at the chosen thresholds, a different centre
  convention, tiling/normalisation effects, a model trained on other imaging. These are the candidates a human
  should look at first; either side can be right.
* The tolerance, both engines' thresholds and the model choice change the three counts; the counts are
  descriptive statistics of engine behaviour on this image, not accuracy.

## 3. Spotiflow install, model cache and checksums

* Optional install: `pip install spotiflow` (pulls PyTorch; the CPU build is sufficient). `import polony_app.engines`
  never imports torch or spotiflow; they are imported on first use. `SpotiflowEngine.available()` says why it is
  not usable (package missing, torch missing) and whether the model is already cached.
* Model cache directory: environment `POLONY_MODEL_DIR`, else `<POLONY_HOME>/models/spotiflow`. The pretrained
  model (default name `general`, ~88 MB, from the `weigertlab/spotiflow-models` release) is downloaded **once**;
  when `<cache>/<name>/config.yaml` + weights exist nothing is downloaded or contacted. For offline machines copy
  the extracted folder (`config.yaml`, `best.pt`, ...) to `<cache>/general/`. A folder name inside the cache or a
  path to a model folder also works (fine-tuned models).
* Checksums: the sha256 of the weights artifacts (`*.pt`) is computed once and stored in
  `<model folder>/.polony_checksums.json` (keyed by size + mtime, so it is stable across runs and recomputed only
  if a file changes). The sha256 of `best.pt` (the checkpoint that is loaded) is recorded as `model_checksum` on
  every run, together with the model name, release (`spotiflow-models 0.6.0`), the used probability threshold,
  the spotiflow and torch versions, device, and the preprocessing.
* Opt-in real-model test: `POLONY_SPOTIFLOW_E2E=1 pytest tests/app/test_engines_spotiflow_real.py -s`.

## 4. CPU / GPU

`device='auto'` picks CUDA, then Apple MPS, then CPU. An explicitly requested device that does not exist falls
back to CPU with a note; a CUDA/MPS load or inference error is retried once on CPU and recorded in
`EngineRunInfo.notes` (the device that really ran is in `EngineRunInfo.device`). Measured on a CPU-only
container: a 1000 x 1000 uint16 crop takes about 8-10 s (127-135 points), a 300 x 300 image about 1-3 s once the
model is loaded. Large pages are tiled automatically by Spotiflow (`max_tile_size`, `n_tiles`).

## 5. Scientific limitations

* The Spotiflow score is the raw heatmap value at the detected point, not a calibrated probability. Do not
  threshold it as "x % confidence", do not compare it to the Big-FISH response, and do not average the two.
* **Normalisation (`normalizer`).** `auto` (default, legacy) is Spotiflow's own percentile normalisation (1 / 99.8 %)
  of the array it receives: the cropped (and, if downscaled, block-mean) valid bounding box. That box still contains
  saturated and rim pixels, so p99.8 was the dtype maximum (65535) on 5 of 7 research slides and the model's
  sensitivity became an uncontrolled per-slide variable (AS-B: 24 detections with `auto` vs 171 with the fixed
  normaliser). **`valid_unsaturated`** (opt-in; explicit, recorded) computes lo/hi once per page, BEFORE any
  crop/zoom, as the percentiles 1 / 99.8 of the pixels that are inside the valid (counting-area) mask AND not
  saturated (raw < dtype maximum; floats: no exclusion), and passes `(x - lo) / (hi - lo)` to the model as a
  callable normaliser, the same in the native and the downscaled path (the map is affine, so it commutes with the
  block mean). If fewer than 1000 pixels qualify it falls back to all valid pixels, then to the page (noted).
  `preprocessing.normalizer_stats` records lo, hi, the pixel counts (page, valid, valid-and-saturated, used) and the
  rule. `none` passes raw values and is only appropriate for models trained that way. The adapter always feeds the original dtype (never an 8-bit conversion) and crops to the
  bounding box of the counting area + 32 px (`crop_to_valid`, `preprocessing.crop_box_xyxy`; coordinates are mapped
  back exactly). Which default to ship is the maintainers' decision; numbers per slide and threshold are in
  [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md).
* The pretrained `general` model was trained on other microscopy data (FISH/spots); its behaviour on these
  polonies is not validated. It is a second opinion, not a reference.
* TIFF page -> colour is unconfirmed science: detection is per page index, colour roles only come from the
  confirmed mapping.
* Spotiflow has its own prior on spot size/shape and ignores the app's `spot_radius_px`; Big-FISH needs it.
  Dense clusters and saturated plateaus are handled very differently by the two.
* **Large blobs: `downscale` (÷1 default, 2, 4, 8).** The pretrained models expect small spots; polonies are ~25 px
  blobs. With `downscale = d > 1` the adapter downscales by a **block mean** (area average, anti-aliased) over
  `d x d` pixels, runs the model with `subpix=False`, and maps points back. The crop is anchored to an
  image-wide grid: its origin is a multiple of `16·d` full-resolution pixels (16 = the U-Net stride; extended
  outward, still inside the image), its far side the next multiple (capped at the last full block; the remainder
  of the page is `preprocessing.trimmed_px`), and the downscaled array is reflect-padded on the right/bottom to a
  multiple of 16 by the adapter (`reflect_pad_hw`; peaks in the padding are dropped) so Spotiflow's own centred
  padding is a no-op. Result: the block grid and the network's stride phase do not depend on where the counting
  area starts. Measured on real slides (AI-A, AS-A, FS-A), shifting the counting area by 1-3 px in x and/or y
  changed 0-1.5 % of the detections beyond 2 px (AS-A: up to 3 of 86); the previous zoom + free crop origin changed
  4-7 % on AI-A. Tiled inference on very large crops (Spotiflow `max_tile_size`) still depends on the tile grid.
  We downscale ourselves rather than pass `scale=` because Spotiflow's own `scale` zooms without anti-aliasing and
  returns points in the original frame on its single-tile path but in the downscaled frame on its tiled path.
  **Mapping:** a block `i` covers full-resolution pixels `i·d .. i·d+d-1`, so its centre is
  `x_full = x0 + i·d + (d-1)/2` (unit-tested with sub-pixel Gaussians at d=2,4,8: unbiased < 0.02 px). The older
  align-corners formula `x_s·(n-1)/(m-1)` belonged to the zoom resampler and no longer applies.
  **Localisation error (measured, real `general` model):** RMS 0.83 / 1.74 / 3.60 px and maximum 1.4 / 3.25 /
  5.57 px at d = 2 / 4 / 8 against the native-scale model (independent review). Against a model-free blob centre
  (mean-shift of the smoothed image, 3 real slides, unrefined integer peaks) RMS is 1.25 / 1.83 / 3.31 px; with the
  default **quadratic peak refinement** on `details.heatmap` (`refine_peaks`, a parabola through the 3 heatmap
  samples per axis, clipped to ±0.5 downscaled px; recorded) it is 1.04 / 1.21 / 2.13 px (median 0.33 / 0.60 /
  1.46 px). So ÷4 is good to about ±1 to 2 px: fine for 25 px blobs and 4-10 px matching tolerances, not for
  sub-pixel claims, and not "±d/2". `min_distance`/`exclude_border` act in the downscaled grid. Exploratory evidence
  in [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md); not validated; default stays ÷1. Unsupported values
  and `subpix=true` with `downscale>1` are errors.
* **Consensus tolerance and downscale.** `consensus_tolerance_px` keeps its schema default of 4. When the request
  does not set it explicitly and the Spotiflow downscale `d` exceeds it, the pipeline uses `d` (so ÷8 pairs within
  8 px); an explicit value is used as given. The effective value and the reason are in the run notes
  ("consensus (tolerance 8 px, raised from the 4 px default to the Spotiflow downscale)").
* Big-FISH default threshold is unchanged and not tuned to Spotiflow; the consensus tolerance is a starting
  point, not a validated cut-off.

## 6. Failure modes

| Situation | Behaviour |
|---|---|
| spotiflow / torch not installed | `available()` False with the reason; hybrid: Big-FISH result + failed run + WARNING; spotiflow mode: `EngineUnavailable` |
| model missing and no network | `EngineUnavailable` naming the cache dir and how to copy the model manually |
| corrupt weights / load error | `EngineError` ("failed to load Spotiflow model ...") |
| inference error (OOM, bad tile, ...) | `EngineError` with the original message; GPU errors are first retried on CPU |
| points and scores of different length | `EngineError` (nothing is guessed) |
| NaN/inf in the image | `EngineError` |
| empty counting area | run `status='skipped'` with a note, no detections |
| cancel | `DetectionCancelled` (between tiles for tiled inference) |
