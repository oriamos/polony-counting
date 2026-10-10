# Spotiflow exploration on real data (preliminary, single slide — NOT validated)

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
Reading: the pretrained model at native scale is not suitable for these ~25 px blobs; downscaling makes it a precise but conservative independent detector. Agreement is a useful *priority signal* (disagreements ≈ 40 % of candidates), not ground truth.
Mechanics learned (Spotiflow 0.6.5): `scale != 1` requires `subpix=False` and image sides divisible by `1/scale`; returned points are in the DOWNSCALED frame on the tiled path but in the ORIGINAL frame on the single-tile path (the app therefore downscales itself). Correct mapping is align-corners, x_orig = x_s·(n−1)/(m−1), not the pixel-centre x_s·d + (d−1)/2 (see TWO_ENGINE.md).

## Re-check through the app's own ÷4 implementation (same slide, same reference; still preliminary)
Run through the real app (`engine=hybrid`, Spotiflow `general`, `downscale=4`, `prob_thresh=0.1`, align-corners coordinate mapping; Big-FISH sensitive), page 0, 18 fields:

| detector | points in fields | precision @10 px | recall @10 px | precision @6 px | recall @6 px |
|---|---|---|---|---|---|
| Big-FISH (sensitive) | 292 | 0.86 | 0.75 | 0.68 | 0.60 |
| Spotiflow ÷4, thr 0.1 (app) | 204 | 0.89 | 0.55 | 0.71 | 0.44 |
| agreement (SF within 4 px of BF) | 203 | 0.90 | 0.55 | 0.71 | 0.44 |

Consensus on this slide: agreement 203, Big-FISH-only 89, Spotiflow-only 1 (page 0). The earlier exploratory numbers (0.92 / 0.57) used Spotiflow's own `scale=` path and are superseded by this table. Runtime with ÷4: ≈6.5 s per page on CPU (vs ≈21 s at native scale). Green page (nearly empty): Spotiflow ÷4 found 2 valid points while Big-FISH proposed 43 (most flagged debris-like).
Per-field correlation was not recomputed here (the first quick script mapped fields to grid cells incorrectly); use the app's `/fields/compare` endpoint.
