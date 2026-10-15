# Polony counter – frontend

Vite + React + TypeScript (+ zustand) research UI for the polony-counting app. It talks to the
FastAPI backend at `/api` (see `docs/app/CONTRACTS.md`; types mirrored by hand in
`src/api/types.ts` from `src/polony_app/schemas.py`). In production the backend serves
`frontend/dist` at `/`.

## Commands

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173, proxies /api -> http://127.0.0.1:8000
API_TARGET=http://127.0.0.1:8787 npm run dev   # ... or against the mock API
npm run mock           # mock API on :8787 (PORT=... to change); also serves dist/ at '/'
npm run build          # tsc --noEmit && vite build -> dist/
npm test               # vitest: coordinate transforms, zoom about cursor, class glyph logic
npm run e2e            # real Chromium (playwright-core) against the mock; run `npm run build` first
```

`npm run e2e` uses `/opt/pw-browsers/chromium` (override with `CHROMIUM_PATH`). It writes screenshots
to `e2e-artifacts/` (git-ignored) and exits non-zero on any failed check.

## Layout

* Left rail: analyses (status chips), TIFF drag & drop upload, CSV import (blank cells shown distinct
  from 0; colour mapping must be chosen explicitly, never inferred from column names).
* Centre: viewer (wheel zoom about cursor, drag pan, Fit / 1:1, five views, display-only contrast
  sliders, coordinate readout, scale bar), toolbar for tools, class toggles, hide-overlays.
* Right: inspector for the selection + tabs Review (queue), Regions, Detect, Totals, Finish.
* `?` opens the keyboard-shortcut help.

## Coordinate model (important)

All coordinates are image pixels, pixel (0,0) has its centre at (0,0). The viewer is one CSS-transformed
"world" element: `screen = (tx,ty) + scale * image`. The full-resolution PNG is placed in it at natural
pixel size and shifted by `transform: translate(-0.5px,-0.5px)` (a CSS *transform*, not `left/top`:
Chrome pixel-snaps fractional layout offsets of images, which produced a 0.5 image-px misalignment that
the e2e screenshot test caught). The SVG overlay lives in the same world element, so overlay and image
share one transform. `src/lib/geometry.ts` holds the pure maths (unit-tested); `src/lib/glyph.ts` the
class glyphs.

Overlay glyphs: red-only = circle, green-only = diamond, yellow = square, uncertain = dashed circle;
each with a dark + light double outline and a thin class-coloured line; amber dot = unreviewed,
green tick = reviewed; cyan ring = selected.

## Mock API

`mock/server.mjs` is a dependency-free node http server implementing the whole contract in memory
(synthetic 512x384 two-channel scene with Gaussian spots at known integer pixel centres, undo/redo,
jobs with progress/cancel/failure, regions, approval rules, finalize/reopen, CSV, exports).
Test helpers: `POST /api/__mock/reset`, `POST /api/__mock/fail-next-action {"message": "..."}`.
Seeded analyses: `demo` (in review, with an embedded ROI suggestion), `unmapped` (needs channel
mapping), `composite` (composite-only).

## Beyond the contract

* Channel thumbnails in the mapping dialog request `/image/ch{index}` (per-channel, role-independent).
  If the backend lacks it the card shows "preview unavailable".
* "Reopen" calls `POST /api/analyses/{id}/reopen`; on 404/405 it falls back to `unapprove_region` per region.
* `window.__polony` exposes the zustand store for debugging/e2e.

## Two-engine UI (docs/app/TWO_ENGINE_CONTRACT.md)

* Detect tab: engine selector (Big-FISH default / Spotiflow / Both), model prepare job, consensus tolerance, engine-runs table.
* Viewer toolbar "Engine layers": consensus rings (solid = both, long dash = Big-FISH only, dots = Spotiflow only, double = human added),
  raw Big-FISH (hollow circle) / Spotiflow (hollow triangle) layers fetched lazily per page (cap 5000, valid only unless "show excluded"),
  "Show rejected" (deleted objects). Wording: "model score" / "LoG response (noise units)", never a percentage.
* Review: engine-status filter, summary card per page, QC sample of agreements (POST consensus/qc-sample), Approve/Reject = confirm/delete.
* Mock extras: `analysis hybrid`, `POST /api/__mock/spotiflow {available}`, model `smfish_3d` makes the Spotiflow run fail.
