# Spotiflow exploration on real data (NOT validated; one annotator)

## Update: fixed normalisation, 8 slides, threshold curves (supersedes every number further down)

**All numbers in the sections below this one were obtained with the saturation-sensitive `auto` normalisation** (Spotiflow's p1/p99.8 of the cropped, zoomed valid bounding box, which still contains saturated and rim pixels; p99.8 was the dtype maximum on 5 of 7 slides). That made Spotiflow's sensitivity an uncontrolled per-slide variable (an independent review found AS-B: 24 detections vs 171 once the saturated pixels were excluded). They are kept only for comparison and are labelled "legacy auto".

New default behaviour is opt-in (`normalizer = "valid_unsaturated"`, the request default stays `auto`): lo/hi = percentiles 1 / 99.8 of the page pixels that are inside the valid (counting-area) mask AND not saturated (raw < dtype maximum), computed once per page before the crop/downscale and applied identically to the downscaled image (an affine map commutes with the block mean). lo/hi, pixel counts and the number of excluded saturated pixels are recorded in `EngineRunInfo.preprocessing.normalizer_stats`. The ÷d downscale is now a block mean on a block grid anchored to the image (crop origin a multiple of 16·d), coordinates `x = x0 + i·d + (d−1)/2`, peaks refined by a quadratic fit on the heatmap (see TWO_ENGINE.md).

Setup: the real app (hybrid / spotiflow engines through the HTTP API), TIFF page 0, inside the researcher fields (FS-B_D with the unconfirmed shift (−158,−34)), reference = the researcher's embedded marks, one-to-one matching at 10 px. Big-FISH sensitive (threshold_scale 0.5); Spotiflow `general` ÷4. Model default threshold = 0.5 (`prob_thresh_best`). Thresholds above 0.02 are obtained by filtering one run at 0.02 by score (identical peaks). Cell format: precision/recall (number of objects). **No threshold was tuned on all slides; the curves are reported and one pooled value is not a recommendation.**

Big-FISH sensitive alone P/R (n): AI-A 0.86/0.75 (292) | AI-B 0.81/0.77 (380) | AM-A 0.46/0.77 (169) | AM-B 0.44/0.54 (126) | AS-A 0.45/0.83 (151) | AS-B 0.27/0.70 (155) | FS-A 0.80/0.81 (186) | FS-B_D 0.56/0.84 (123)

### Spotiflow ÷4 alone, `valid_unsaturated`
| thr | AI-A | AI-B | AM-A | AM-B | AS-A | AS-B | FS-A | FS-B_D | pooled P/R |
|---|---|---|---|---|---|---|---|---|---|
| model default | 0.92/0.71 (256) | 0.87/0.70 (319) | 0.78/0.49 (63) | 0.69/0.39 (58) | 0.80/0.54 (55) | 0.37/0.77 (126) | 0.89/0.75 (152) | 0.67/0.80 (98) | 0.80/0.67 |
| 0.05 | 0.79/0.85 (357) | 0.80/0.82 (411) | 0.66/0.69 (105) | 0.53/0.53 (104) | 0.66/0.74 (93) | 0.26/0.88 (203) | 0.83/0.85 (185) | 0.58/0.88 (124) | 0.68/0.80 |
| 0.1 | 0.83/0.83 (331) | 0.81/0.81 (399) | 0.65/0.68 (104) | 0.57/0.53 (97) | 0.66/0.70 (86) | 0.28/0.88 (191) | 0.84/0.84 (180) | 0.60/0.87 (118) | 0.70/0.79 |
| 0.2 | 0.88/0.81 (305) | 0.83/0.79 (380) | 0.70/0.64 (91) | 0.60/0.51 (88) | 0.69/0.67 (80) | 0.29/0.88 (181) | 0.85/0.82 (177) | 0.61/0.84 (113) | 0.73/0.77 |
| 0.3 | 0.89/0.80 (296) | 0.83/0.78 (375) | 0.73/0.63 (86) | 0.61/0.49 (82) | 0.72/0.63 (72) | 0.29/0.82 (171) | 0.85/0.81 (174) | 0.62/0.83 (110) | 0.74/0.75 |

### Spotiflow ÷4 alone, legacy `auto` (saturation-sensitive; old numbers)
| thr | AI-A | AI-B | AM-A | AM-B | AS-A | AS-B | FS-A | FS-B_D | pooled P/R |
|---|---|---|---|---|---|---|---|---|---|
| model default | 0.94/0.45 (160) | 0.87/0.52 (237) | 0.80/0.35 (44) | 0.76/0.28 (38) | 0.93/0.16 (14) | 0.67/0.07 (6) | 0.89/0.75 (153) | 0.67/0.80 (98) | 0.85/0.48 |
| 0.05 | 0.93/0.60 (214) | 0.84/0.67 (320) | 0.71/0.55 (77) | 0.66/0.48 (74) | 0.79/0.51 (53) | 0.47/0.27 (34) | 0.83/0.85 (185) | 0.58/0.88 (125) | 0.79/0.64 |
| 0.1 | 0.93/0.57 (203) | 0.85/0.65 (305) | 0.74/0.53 (72) | 0.66/0.46 (71) | 0.80/0.48 (49) | 0.47/0.25 (32) | 0.84/0.84 (180) | 0.60/0.87 (119) | 0.80/0.62 |
| 0.2 | 0.94/0.55 (196) | 0.87/0.63 (290) | 0.74/0.49 (66) | 0.65/0.42 (66) | 0.79/0.45 (47) | 0.48/0.22 (27) | 0.85/0.82 (177) | 0.61/0.84 (113) | 0.81/0.59 |
| 0.3 | 0.94/0.54 (190) | 0.87/0.60 (278) | 0.75/0.49 (65) | 0.69/0.41 (61) | 0.79/0.41 (43) | 0.50/0.20 (24) | 0.85/0.81 (174) | 0.62/0.83 (110) | 0.82/0.58 |

### Hybrid = Big-FISH sensitive plus Spotiflow-only points (consensus tolerance 4 px), `valid_unsaturated`
| thr | AI-A | AI-B | AM-A | AM-B | AS-A | AS-B | FS-A | FS-B_D | pooled P/R |
|---|---|---|---|---|---|---|---|---|---|
| model default | 0.85/0.76 (297) | 0.81/0.78 (383) | 0.46/0.77 (169) | 0.44/0.54 (126) | 0.45/0.83 (151) | 0.28/0.78 (168) | 0.80/0.81 (186) | 0.56/0.84 (123) | 0.64/0.77 |
| 0.05 | 0.76/0.85 (370) | 0.79/0.83 (420) | 0.45/0.77 (173) | 0.45/0.58 (134) | 0.44/0.83 (153) | 0.25/0.88 (215) | 0.79/0.85 (194) | 0.54/0.87 (132) | 0.61/0.82 |
| 0.1 | 0.81/0.83 (344) | 0.80/0.82 (412) | 0.45/0.77 (173) | 0.45/0.58 (134) | 0.45/0.83 (152) | 0.26/0.88 (205) | 0.80/0.85 (192) | 0.55/0.87 (129) | 0.62/0.81 |
| 0.2 | 0.84/0.81 (320) | 0.80/0.80 (398) | 0.45/0.77 (171) | 0.45/0.57 (131) | 0.45/0.83 (151) | 0.27/0.88 (195) | 0.80/0.84 (190) | 0.55/0.84 (126) | 0.63/0.80 |
| 0.3 | 0.84/0.80 (316) | 0.80/0.80 (397) | 0.45/0.77 (170) | 0.44/0.54 (127) | 0.45/0.83 (151) | 0.26/0.82 (189) | 0.80/0.83 (189) | 0.55/0.84 (125) | 0.63/0.79 |

### Hybrid, legacy `auto` (old numbers)
| thr | AI-A | AI-B | AM-A | AM-B | AS-A | AS-B | FS-A | FS-B_D | pooled P/R |
|---|---|---|---|---|---|---|---|---|---|
| model default | 0.86/0.75 (292) | 0.81/0.77 (381) | 0.45/0.77 (170) | 0.44/0.54 (126) | 0.45/0.83 (151) | 0.27/0.70 (155) | 0.80/0.81 (186) | 0.56/0.84 (123) | 0.64/0.76 |
| 0.05 | 0.85/0.76 (296) | 0.81/0.78 (386) | 0.45/0.77 (172) | 0.44/0.54 (127) | 0.45/0.83 (152) | 0.27/0.70 (155) | 0.79/0.85 (194) | 0.54/0.87 (132) | 0.64/0.77 |
| 0.1 | 0.85/0.75 (295) | 0.81/0.78 (385) | 0.45/0.77 (170) | 0.44/0.54 (127) | 0.45/0.83 (151) | 0.27/0.70 (155) | 0.80/0.85 (192) | 0.55/0.87 (130) | 0.64/0.77 |
| 0.2 | 0.85/0.75 (294) | 0.81/0.78 (383) | 0.45/0.77 (170) | 0.44/0.54 (127) | 0.45/0.83 (151) | 0.27/0.70 (155) | 0.80/0.84 (190) | 0.55/0.84 (126) | 0.64/0.76 |
| 0.3 | 0.85/0.75 (294) | 0.81/0.77 (382) | 0.45/0.77 (170) | 0.44/0.54 (126) | 0.45/0.83 (151) | 0.27/0.70 (155) | 0.80/0.83 (189) | 0.55/0.84 (125) | 0.64/0.76 |

Reading (descriptive, not a validation):
* The normaliser matters a lot for Spotiflow alone: with legacy `auto` at the model's own threshold, recall on AS-A / AS-B was 0.16 / 0.07 (14 / 6 points); with `valid_unsaturated` it is 0.54 / 0.77 (55 / 126 points). The slides become comparable, at the cost of precision on the sparse slides (AS-B 0.37, AM-B 0.69 at the default threshold).
* Lowering the threshold trades precision for recall smoothly on every slide; there is no threshold at which all slides are good. Pooled over 8 slides: default 0.80/0.67, 0.2 → 0.73/0.77, 0.05 → 0.68/0.80.
* The hybrid (Big-FISH sensitive plus Spotiflow-only) hardly depends on the Spotiflow threshold or normaliser: Big-FISH's sensitive setting dominates the union; Spotiflow adds a few points (precision 0.62-0.64 pooled vs Big-FISH alone). The engines' value is as a disagreement signal for review, not as a better count.

### Leave-one-slide-out threshold choice (illustration only; criterion pooled F1 on the other 7 slides)
Spotiflow alone:
```
AI-A: chose 0.2 -> P/R 0.88/0.81 F1 0.84; (oracle best on this slide F1 0.84 at 0.2)
AI-B: chose 0.2 -> P/R 0.83/0.79 F1 0.81; (oracle best on this slide F1 0.81 at 0.1)
AM-A: chose 0.2 -> P/R 0.70/0.64 F1 0.67; (oracle best on this slide F1 0.68 at 0.3)
AM-B: chose 0.2 -> P/R 0.60/0.51 F1 0.55; (oracle best on this slide F1 0.55 at 0.2)
AS-A: chose 0.2 -> P/R 0.69/0.67 F1 0.68; (oracle best on this slide F1 0.70 at 0.05)
AS-B: chose 0.2 -> P/R 0.29/0.88 F1 0.44; (oracle best on this slide F1 0.49 at None)
FS-A: chose 0.2 -> P/R 0.85/0.82 F1 0.84; (oracle best on this slide F1 0.84 at 0.1)
FS-B_D: chose 0.2 -> P/R 0.61/0.84 F1 0.71; (oracle best on this slide F1 0.73 at None)

```
Hybrid:
```
AI-A: chose 0.1 -> P/R 0.81/0.83 F1 0.82; (oracle best on this slide F1 0.83 at 0.2)
AI-B: chose 0.2 -> P/R 0.80/0.80 F1 0.80; (oracle best on this slide F1 0.81 at 0.1)
AM-A: chose 0.1 -> P/R 0.45/0.77 F1 0.56; (oracle best on this slide F1 0.57 at None)
AM-B: chose 0.2 -> P/R 0.45/0.57 F1 0.50; (oracle best on this slide F1 0.51 at 0.05)
AS-A: chose 0.1 -> P/R 0.45/0.83 F1 0.58; (oracle best on this slide F1 0.58 at None)
AS-B: chose 0.1 -> P/R 0.26/0.88 F1 0.40; (oracle best on this slide F1 0.42 at 0.2)
FS-A: chose 0.2 -> P/R 0.80/0.84 F1 0.82; (oracle best on this slide F1 0.82 at 0.1)
FS-B_D: chose 0.2 -> P/R 0.55/0.84 F1 0.66; (oracle best on this slide F1 0.67 at None)

```
The leave-one-out choice is almost always 0.2 (Spotiflow alone) or 0.1-0.2 (hybrid) and lands within about 0.05 F1 of the per-slide oracle, but with 8 slides from one session and one annotator this is an illustration, not a tuned or validated setting. The default threshold stays the model's own.

On the legacy-normalisation hybrid run (7 slides; Big-FISH-only is defined against the Spotiflow set, so the counts shift with the normaliser), Big-FISH-only detections that are also `debris_like` are 97 % unmatched to a mark (333 of 345 over 7 slides) against 18 % for the other Big-FISH-only ones (43 of 237); the review queue now ranks them with saturated objects (see REVIEW_POLICY.md). Disagreement rates between the engines are 24 % to 85 % of candidates depending on the slide, not "about 40 %".

---
# (legacy, obtained with the saturation-sensitive `auto` normalisation) Spotiflow exploration, single slide

Slide: `29-09-2026_T7_AB_AI-A.tif`, TIFF page 0, inside the 18 imported researcher fields; reference = the researcher's 332 embedded marks (one annotator).
Matching: one-to-one Hungarian, tolerance 10 px (blob diameter ≈ 25 px; the marks are clicks). CPU, official pretrained model `general` (spotiflow-models 0.6.0, Spotiflow 0.6.5). Big-FISH = sensitive setting (threshold_scale 0.5).
**Exploratory: the Spotiflow settings below were chosen by looking at this same slide and reference — optimistic. Needs blind evaluation on other slides.**

| detector | points in fields | precision | recall |
|---|---|---|---|
| Big-FISH (sensitive) | 292 | 0.86 | 0.75 |
| Spotiflow `general`, native scale | 62 | 0.73 | 0.14 |
| Spotiflow `general`, ÷2, default threshold | 117 | 0.93 | 0.33 |
| Spotiflow `general`, ÷4, threshold 0.1 | 204 | 0.92 | 0.57 |
| Spotiflow `general`, ÷8, threshold 0.1 | 176 | 0.80 | 0.42 |

Two engines (Spotiflow ÷4, thr 0.1 vs Big-FISH sensitive, same-point tolerance 4 px): agreement 185 (precision 0.93), Big-FISH-only 107 (0.72), Spotiflow-only 19 (0.84; 16 of 19 coincide with a mark). Union adds no net recall in this test (0.75).
Reading: the pretrained model at native scale is not suitable for these ~25 px blobs; downscaling makes it a precise but conservative independent detector. Agreement is a useful *priority signal* (disagreements are 24 % to 85 % of candidates depending on the slide), not ground truth.
Mechanics learned (Spotiflow 0.6.5): `scale != 1` requires `subpix=False` and image sides divisible by `1/scale`; returned points are in the DOWNSCALED frame on the tiled path but in the ORIGINAL frame on the single-tile path (the app therefore downscales itself). The legacy path used the zoom (align-corners) mapping x_orig = x_s·(n−1)/(m−1); the app now block-means and maps with x_s·d + (d−1)/2 + crop origin (see TWO_ENGINE.md).

## Re-check through the app's own ÷4 implementation (same slide, same reference; still preliminary)
Run through the real app (`engine=hybrid`, Spotiflow `general`, `downscale=4`, `prob_thresh=0.1`, align-corners coordinate mapping; Big-FISH sensitive), page 0, 18 fields:

| detector | points in fields | precision @10 px | recall @10 px | precision @6 px | recall @6 px |
|---|---|---|---|---|---|
| Big-FISH (sensitive) | 292 | 0.86 | 0.75 | 0.68 | 0.60 |
| Spotiflow ÷4, thr 0.1 (app) | 204 | 0.89 | 0.55 | 0.71 | 0.44 |
| agreement (SF within 4 px of BF) | 203 | 0.90 | 0.55 | 0.71 | 0.44 |

Consensus on this slide: agreement 203, Big-FISH-only 89, Spotiflow-only 1 (page 0). The earlier exploratory numbers (0.92 / 0.57) used Spotiflow's own `scale=` path and are superseded by this table. Runtime with ÷4: ≈6.5 s per page on CPU (vs ≈21 s at native scale). Green page (nearly empty): Spotiflow ÷4 found 2 valid points while Big-FISH proposed 43 (most flagged debris-like).
Per-field correlation was not recomputed here (the first quick script mapped fields to grid cells incorrectly); use the app's `/fields/compare` endpoint.
