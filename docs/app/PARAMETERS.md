# Detection parameters

All values below are read from `src/polony_app/schemas.py` (`DetectionParams`, `DetectionRequest`),
`detection/bigfish_engine.py`, `detection/pipeline.py`, `detection/matching.py` and
`detection/review_reasons.py`. Everything is in **image pixels**: the app never converts to um
because the pixel size is unknown (see [SCIENTIFIC_OPEN_QUESTIONS.md](SCIENTIFIC_OPEN_QUESTIONS.md)).
Red and green have separate parameter sets (`request.red`, `request.green`); the Detect tab has
"Copy red -> green" buttons.

> **None of these defaults is a validated setting.** They were chosen on one real red-only image (no longer distributed;
> the repository's `test_img.tif` is a synthetic stand-in) and later checked, not tuned, on 8 research slides
> ([ALL_SLIDES_EVAL.md](ALL_SLIDES_EVAL.md)). Always check the result with Preview and a full manual review. The count depends strongly on spot radius (the UI shows a real example: radius 5 ->
> 108 objects, 3 -> 207, 2 -> 331).

## Per-channel parameters (`DetectionParams`)

| Parameter (UI label) | Default | Meaning / exact semantics |
|---|---|---|
| `spot_radius_px` ("Spot radius") | 5.0 | Passed to Big-FISH as `log_kernel_size`: the **standard deviation (sigma) of the Laplacian-of-Gaussian in px**, not a radius or diameter. On `test_img.tif` the fitted polony sigma had median 5.3 px (IQR 4.6-6.1). The sigma of other slides may differ (screenshots suggested 8-12 px, not measured). Also sets the size of the measurement disc, the local-background annulus (2.0 to 3.5 x radius) and the near-edge distance (2 x radius). |
| `minimum_distance_px` ("Min. distance") | 5.0 | Big-FISH local-maximum window: `ndi.maximum_filter` with a **square** window of side `2*ceil(d)+1`. Two peaks closer than `ceil(d)` along both axes collapse into one. Changing it between 1x and 2x sigma changed counts by <= 4 % on test_img. Also used by the `dense_cluster` rule. |
| `threshold` (mode "Manual", "Manual value") | `None` = automatic | A threshold on the **LoG response** of the background-corrected float32 image, expressed in **noise units**: LoG response divided by its robust noise sigma (1.4826 x MAD of the signed LoG response over the valid pixels, computed after masking and background correction). It is therefore independent of image brightness / exposure, but still depends on the spot radius. UI label: "Threshold (noise units, robust sigma of the LoG response)". The absolute LoG threshold applied is `threshold x noise sigma` (stored as `threshold_abs` in the run info). Values below about 5 are within the noise. If set, `threshold_scale` is ignored (a note says so). |
| `threshold_scale` ("Auto scale") | 1.0 | Multiplies the automatic value (or the noise fallback below). 1 = unchanged. On test_img the automatic value was about 2-3 x stricter than the researcher's manual marks; lower scales find more spots but also more debris. |
| `background.method` ("Background") | `gaussian_subtract` | `none`, `gaussian_subtract` or `white_tophat`. See below. |
| `background.sigma_px` | 50.0 | Sigma of the Gaussian background estimate (`gaussian_subtract`), about 10 x the spot sigma. |
| `background.radius_px` | 25 | Disc radius for `white_tophat`. |
| `background.clip_negative` | true | Clip the corrected image at 0 for detection (SNR statistics always use the unclipped image). |
| `dense_decomposition` | **false** | Big-FISH `decompose_dense`. Keep OFF, see below. |
| `remove_duplicate_radius_px` | 3.0 | After measurement, of two spots closer than this the one with the weaker LoG response is dropped (saturated plateaus first). Not shown in the UI. |
| `tile_size` ("Tile size") | 0 | 0 or >= the image size = whole image. Otherwise the image is processed in tiles of this core size. |
| `tile_overlap_px` | 64 | Halo around each tile. Raised automatically to `ceil(4*radius) + ceil(min_distance) + 2` when smaller (a note is recorded). |

### Threshold: auto, manual, scale

1. **Automatic** (`threshold = None`): the Big-FISH `automated_threshold_setting` (elbow of the
   spot-count vs threshold curve) is applied to the **noise-normalised** LoG response (divided by its robust sigma, see above) **restricted to the valid mask** (counting regions
   minus exclusions), computed **once on the whole masked image per channel** and cached (together with the noise sigma, which a manual threshold also needs), then multiplied
   by `threshold_scale`. Why noise units: Big-FISH's candidate grid is absolute (integer steps up to the 99.9999th percentile), so on raw LoG units the same slide imaged at x0.5 / x0.25 brightness gave a different automatic threshold and different detections (R=5 on test_img: 103 -> 67 -> 64 red detections, recall vs the 77 marks 0.56 -> 0.44 -> 0.43). In noise units the detections are identical at x1, x0.5, x0.25 (tested, synthetic and test_img) and with a constant offset. Detections of a *clipped* image still change when the exposure change un-clips pixels (plateau merging). Preview windows and tiles reuse exactly this value, so a preview shows what the full run would
   find in the window.
2. **Fallback**: if Big-FISH cannot pick a value (too few maxima, monotone curve, `ValueError`), returns <= 0, or picks an elbow below 5 noise units (the noise regime: local maxima of pure LoG noise rarely exceed ~4 sigma, i.e. there is no separable population of spots, as in an empty or very sparse channel), the app uses
   5 noise units (`5 x 1.4826 x MAD` of the signed -LoG response over the valid pixels), and says so in the notes (the noise-regime case reads "WARNING automatic threshold (x noise units) is in the noise regime ... this channel may contain no (or very few) real spots") ("WARNING automatic
   threshold unusable ... Set a threshold manually if unsure"). The automatic method is known to fail on sparse slides (few true spots)
   and on images with bright structures inside the analysed area.
3. **Manual**: the given value is used as is, in noise units (the noise sigma is still estimated on the full counting area, so previews match the full run).

**Empty-channel warning.** If more than 50 % of a channel's spots (at least 3) are debris-like (compact relative to the configured radius), the run adds a WARNING note per channel suggesting the channel may be empty or the radius too large; classes are not changed.

A counting region is strongly recommended: without one, large saturated areas (>= 5000 px, dilated by about 3 radii) are
excluded automatically as a best effort and a WARNING note is added; otherwise the whole image is counted. On test_img an unmasked
run produced 461 spots of which 371 lay outside the chamber.

### Background correction

* `gaussian_subtract`: image minus a masked Gaussian (normalised convolution) of the valid pixels; pixels more than 3 robust sigmas above
  the first estimate (spots, bright structures) get weight 0 in a second pass; saturated raw pixels always get weight 0.
  For sigma >= 16 it is computed on a block-averaged grid and interpolated (fast, smooth by construction).
* `white_tophat`: image minus a morphological opening with a disc.
* `none`: raw values (a note is recorded).
* Inside the mask the choice barely matters on test_img (recall 0.55-0.57 for all methods); masking matters far more.
* The `processed_*` image views are computed without the counting mask (display only).

### Dense decomposition: off, and why

Big-FISH `decompose_dense` assumes identical, diffraction-limited Gaussian spots whose brightness adds. Polonies are colonies with a wide brightness
range (about 16x between p10 and p90) and are often elongated, so a polony brighter than the median reference spot is split into several
"spots". On test_img it turned 91 spots into 439 and reduced the matched manual marks from 43 to 13. It is therefore off by default,
adds a WARNING note when enabled, and marks the extra spots `dense_cluster`. Touching polonies should be separated with the manual **split** action.
(Implementation note: Big-FISH crashes on float32 input here, so the image is re-quantised to uint16 for this step only.)

### Tile size and overlap

Tiling exists for memory/speed. The threshold is computed once globally (never per tile, which would make results depend on the tile
grid) and each spot is reported by the tile whose core contains its peak pixel, so the result equals the untiled run up to float rounding.

### Saturated spots

Connected saturated components (raw value == dtype maximum, inside the mask) are merged into **one** spot at the plateau centroid, flagged
`saturated`. A plateau gets `suspicious_shape` if its equivalent diameter exceeds 6 x spot radius, if its elongation exceeds 1.3, or if more than one LoG maximum lay on it (touching saturated polonies; synthetic test: two sigma-5 polonies 10-18 px apart are flagged as one merged object, 20 px or more apart they are two objects). The merge is kept, not split: clipped data do not say how the true intensity divides. A plateau is a lower bound of the true spot. SNR is undefined for a plateau-merged spot (clipped peak, annulus possibly inside the plateau), so `weak_signal` and `near_threshold` are not evaluated for it.

## Cross-channel association (`DetectionRequest`)

| Parameter | Default | Meaning |
|---|---|---|
| `match_tolerance_px` ("Red/green match tolerance") | 6.0 | Maximum distance between a red and a green spot to be paired into one **yellow** object. About 1.2 x the default spot sigma; must exceed registration residual (about 1-3 px) plus localisation error and stay well below the minimum spacing of distinct polonies (14.6 px on test_img). It is not scaled automatically when the spot radius changes. |
| `preview_window` | none | `[x0,y0,x1,y1]` restricts the result to objects whose centre lies in the window; nothing is saved. |

Algorithm (`detection/matching.py`): candidate pairs within the tolerance; per connected component a Hungarian assignment minimises total distance, so each
spot is used at most once. Pair -> `yellow` at the mean position (`association_distance_px` recorded); unmatched -> `red` / `green`. Whether a pair is really one physical
object (or one object was split into two spots) is not decided here.

**Ambiguity rule.** For each matched pair with distance d1, look at every other opposite-channel spot within `tolerance + max(1.5 px, 25 % of tolerance)`
of either member (distance d2). The pair is ambiguous if `d2 - d1 < 1.5 px` or `d2 < 1.25 x d1`. All objects tied by such alternatives become `uncertain` with reason
`ambiguous_match` and `ambiguous_with` = the other ids. Nothing is resolved automatically.
(Constants `AMBIGUITY_ABS_PX = 1.5`, `AMBIGUITY_REL = 0.25`; keyword-only in `associate`, not user-settable in the UI.)

**Alignment.** A green->red integer shift (explicit user step) leaves an uncovered zero strip in green; it is excluded from green detection.

## Measured evidence per spot

* `peak`: background-corrected image interpolated at the sub-pixel refined centre.
* local background = median of an annulus 2.0-3.5 x radius (valid, finite pixels; at least 12); noise = 1.4826 x MAD.
* `snr = (peak - background) / noise` (single-pixel value against per-pixel noise; not a matched-filter statistic). On synthetic noise-only data the median is about 0.1.
* `saturated`: any raw pixel at dtype max within the radius disc.
* elongation: `sqrt(l1/l2)` of the second-moment eigenvalues of the half-maximum blob around the peak (needs >= 6 pixels).
* scale ratio (debris filter): `max over sigma in {0.3r, 0.4r} of sigma^2 * (-LoG)` divided by the same over `sigma in {1.0r, 1.4r}`, maximum in a 5x5 window around the spot.

## Review-reason cut-offs (`ReviewConfig`)

Reasons are computed from measured quantities only; there is no confidence score. They only decide **what is queued for review and flagged**; they never delete
an object. **These cut-offs are unvalidated starting points** (the module docstring says the same); they have not been tuned on green/yellow polonies.

| Reason | Rule (defaults) |
|---|---|
| `weak_signal` | SNR < `weak_snr` = 5.0 in any detected channel |
| `near_threshold` | LoG response of the spot < `near_threshold_factor` = 1.25 x the threshold actually used (after `threshold_scale`) |
| `saturated` | raw pixel at dtype max within the spot disc, or a merged saturated plateau |
| `dense_cluster` | another same-channel spot within `dense_distance_factor` = 2.0 x `minimum_distance_px`; or a spot added by dense decomposition |
| `near_edge` | distance to the valid-mask boundary or the image border < `edge_radius_factor` = 2.0 x `spot_radius_px` |
| `suspicious_shape` | elongation > `elongation_max` = 1.8; or an oversized saturated plateau (> 6 x radius) |
| `debris_like` | scale ratio >= `debris_ratio` = 1.0 in any detected channel. The criterion is "compact relative to the CONFIGURED radius", not an absolute size: with `spot_radius_px` = R, blobs with sigma below about 0.6 R become debris-like (measured: 144/144 small dual-channel Gaussian polonies with sigma 1.5-3 px were made `uncertain` at R = 5). Set the radius from the visible blob size. |
| `ambiguous_match` | set by the matching step (above) |

Special rule: a matched red+green pair that is debris-like in **both** channels is not auto-classed yellow; it becomes `uncertain` (with a history note). On test_img this changed 22 yellow
to 0 yellow / 21 uncertain; those pairs were compact, bright in both channels, and none was a marked polony, i.e. probably broadband-fluorescent debris. With debris in one channel only the object stays
yellow and is queued. Single-channel objects keep their class (flag only). Compact debris is **not** removed or hidden.

Review-queue order: see [REVIEW_POLICY.md](REVIEW_POLICY.md) (the single definition); within the same priority objects are ordered by y, then x.

## Spotiflow `downscale` and `prob_thresh`

`downscale` (1, 2, 4, 8; default **1** = native scale, unchanged behaviour) divides the image (block mean over `d x d` pixels, on an image-anchored grid) before the Spotiflow model sees it, for ~25 px blobs; the Detect tab has a "Large blobs (÷4)" preset. Localisation: RMS about 1.0 / 1.2 / 2.1 px at ÷2 / ÷4 / ÷8 after the default quadratic peak refinement (1.25 / 1.8 / 3.3 px unrefined). `prob_thresh` empty = the model's own optimised threshold (0.5 for `general`, recorded as `prob_thresh_used`); on 8 research slides lower thresholds (0.05 to 0.3) trade precision for recall smoothly and no single value is best (curves in [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md)).

`normalizer`: `auto` (default, legacy; Spotiflow's percentile normalisation of the cropped area, **saturation-sensitive**: p99.8 was 65535 on most slides), `valid_unsaturated` (p1 / p99.8 of the valid, unsaturated page pixels, computed once per page before crop/downscale, recorded in the run), `none`. Use `valid_unsaturated` with `downscale > 1`. `refine_peaks` (default true, adapter-level) refines ÷d peaks with a quadratic fit on the heatmap.

`consensus_tolerance_px` (default 4): if you do not set it, it is raised to the Spotiflow downscale when that is larger (÷8 pairs within 8 px); a value you set is used as given. Details, coordinate mapping and the evidence: [TWO_ENGINE.md](TWO_ENGINE.md), [SPOTIFLOW_EXPLORATION.md](SPOTIFLOW_EXPLORATION.md).

## Re-run merge radius

When detection is re-run, a new automatic object within **3 px** (`RERUN_MERGE_RADIUS_PX`) of a preserved (manual / edited / reviewed) object or of an object you deleted is dropped; see [RERUN_POLICY.md](RERUN_POLICY.md).

## Where the values are recorded

Every run stores the request actually used, package versions (big-fish, numpy, scipy, scikit-image, tifffile, python), thresholds and notes in `provenance.json`. If
parameters, regions, channel mapping or alignment differ from that record, the detection is **stale** and finalizing is blocked.
