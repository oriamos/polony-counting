// Mock of the polony-counting REST API (docs/app/CONTRACTS.md) for developing and
// testing the frontend without the real backend. Plain node http, no dependencies.
//
//   node mock/server.mjs            # http://127.0.0.1:8787  (PORT env to change)
//   API_TARGET=http://127.0.0.1:8787 npm run dev
//
// It also serves ../dist at '/' when built, like the production backend.
// Test helpers: POST /api/__mock/reset, POST /api/__mock/fail-next-action {message}
// Synthetic scene: see SPOTS below (integer pixel centres, Gaussian sigma 2.5).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, '../dist');
const PORT = Number(process.env.PORT || 8787);

// ------------------------------------------------------------------ synthetic image
export const W = 512;
export const H = 384;
export const BG = 2000;
export const AMP = 40000;
export const SIGMA = 2.5;
// red/green amplitudes; positions are integer pixel centres (x = column, y = row)
export const SPOTS = [
  { key: 1, x: 100, y: 100, r: 1, g: 0, cls: 'red', reasons: [] },
  { key: 2, x: 200, y: 150, r: 0, g: 1, cls: 'green', reasons: ['weak_signal'] },
  { key: 3, x: 300, y: 250, r: 1, g: 1, cls: 'yellow', reasons: [] },
  { key: 4, x: 400, y: 80, r: 1, g: 1, cls: 'uncertain', reasons: ['ambiguous_match'] },
  { key: 5, x: 60, y: 320, r: 1, g: 0, cls: 'red', reasons: ['near_threshold'] },
  { key: 6, x: 450, y: 330, r: 0, g: 1, cls: 'green', reasons: ['saturated'] },
];

function noise(x, y, s) {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  h ^= h >>> 16;
  return ((h >>> 0) % 1000) / 1000; // 0..1
}
function buildChannel(which) {
  const a = new Uint16Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let v = BG + noise(x, y, which) * 300;
      for (const s of SPOTS) {
        const amp = which === 0 ? s.r : s.g;
        if (!amp) continue;
        const dx = x - s.x,
          dy = y - s.y;
        if (dx * dx + dy * dy < 100) v += AMP * amp * Math.exp(-(dx * dx + dy * dy) / (2 * SIGMA * SIGMA));
      }
      a[y * W + x] = Math.min(65535, Math.round(v));
    }
  return a;
}
const CH = [buildChannel(0), buildChannel(1)];

// ------------------------------------------------------------------ PNG encoding
const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function encodePng(w, h, rgb) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    rgb.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 3 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const AUTO_LO = 1800;
const AUTO_HI = 32000;
function renderView(view, q) {
  const lo = q.has('lo') ? Number(q.get('lo')) : AUTO_LO;
  const hi = q.has('hi') ? Number(q.get('hi')) : AUTO_HI;
  const gamma = q.has('gamma') ? Number(q.get('gamma')) : 1;
  const x0 = q.has('x0') ? Number(q.get('x0')) : 0,
    y0 = q.has('y0') ? Number(q.get('y0')) : 0,
    x1 = q.has('x1') ? Number(q.get('x1')) : W,
    y1 = q.has('y1') ? Number(q.get('y1')) : H;
  const cw = x1 - x0,
    ch = y1 - y0;
  const maxDim = q.has('max_dim') ? Number(q.get('max_dim')) : 0;
  const k = maxDim && Math.max(cw, ch) > maxDim ? maxDim / Math.max(cw, ch) : 1;
  const ow = Math.max(1, Math.round(cw * k)),
    oh = Math.max(1, Math.round(ch * k));
  const out = Buffer.alloc(ow * oh * 3);
  const norm = (v) => Math.pow(Math.max(0, Math.min(1, (v - lo) / Math.max(1, hi - lo))), 1 / gamma);
  for (let y = 0; y < oh; y++)
    for (let x = 0; x < ow; x++) {
      const sx = Math.min(W - 1, Math.max(0, x0 + Math.floor(x / k))),
        sy = Math.min(H - 1, Math.max(0, y0 + Math.floor(y / k)));
      const i = sy * W + sx;
      let r = 0,
        g = 0,
        b = 0;
      const rv = view === 'processed_red' ? Math.max(0, CH[0][i] - BG) + AUTO_LO : CH[0][i];
      const gv = view === 'processed_green' ? Math.max(0, CH[1][i] - BG) + AUTO_LO : CH[1][i];
      if (view === 'red' || view === 'ch0' || view === 'processed_red') r = norm(rv) * 255;
      else if (view === 'green' || view === 'ch1' || view === 'processed_green') g = norm(gv) * 255;
      else {
        r = norm(rv) * 255;
        g = norm(gv) * 255;
      }
      const o = (y * ow + x) * 3;
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
    }
  return { png: encodePng(ow, oh, out), w: ow, h: oh, crop: [x0, y0, x1, y1] };
}

// ------------------------------------------------------------------ state
const now = () => new Date().toISOString();
const clone = (o) => JSON.parse(JSON.stringify(o));
const defParams = () => ({
  spot_radius_px: 4,
  minimum_distance_px: 6,
  threshold: null,
  threshold_scale: 1,
  background: { method: 'gaussian_subtract', sigma_px: 50, radius_px: 25, clip_negative: true },
  dense_decomposition: false,
  remove_duplicate_radius_px: 3,
  tile_size: 0,
  tile_overlap_px: 64,
});
const defReq = () => ({ red: defParams(), green: defParams(), preview_window: null, match_tolerance_px: 5 });
const rect = (a, b, c, d) => [
  [a, b],
  [c, b],
  [c, d],
  [a, d],
];

function pip(x, y, p) {
  let inside = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i],
      [xj, yj] = p[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function regionOf(a, x, y) {
  if (a.regions.some((r) => r.kind === 'exclusion' && pip(x, y, r.polygon))) return null;
  const c = a.regions.find((r) => r.kind === 'counting' && pip(x, y, r.polygon));
  return c ? c.id : null;
}
function area(p) {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i],
      [x2, y2] = p[(i + 1) % p.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

function evidence(s, which) {
  const on = which === 'r' ? s.r : s.g;
  if (!on) return { detected: false, x: null, y: null, peak_intensity: null, local_background: null, snr: null, saturated: false, spot_id: null };
  return {
    detected: true,
    x: s.x + (which === 'g' && s.cls === 'uncertain' ? 6 : 0),
    y: s.y,
    peak_intensity: AMP * 0.9,
    local_background: BG,
    snr: s.reasons.includes('weak_signal') ? 2.4 : 14.2 + s.key,
    saturated: s.reasons.includes('saturated'),
    spot_id: `${which}${s.key}`,
  };
}
function objFromSpot(s, prefix = 'o') {
  return {
    id: `${prefix}-${String(s.key).padStart(4, '0')}`,
    x: s.x,
    y: s.y,
    cls: s.cls,
    origin: 'auto',
    red: evidence(s, 'r'),
    green: evidence(s, 'g'),
    association_distance_px: s.r && s.g ? (s.cls === 'uncertain' ? 6 : 0.4) : null,
    review_reasons: [...s.reasons],
    ambiguous_with: s.cls === 'uncertain' ? ['o-0003'] : [],
    reviewed: false,
    deleted: false,
    region_id: null,
    history: ['detected by mock engine'],
  };
}
const noEv = () => ({ detected: false, x: null, y: null, peak_intensity: null, local_background: null, snr: null, saturated: false, spot_id: null });

const store = {
  engine: new Map(), // analysis id -> { runs, raws }
  sfAvailable: true,
  sfCached: new Set(['general']),
  analyses: new Map(),
  jobs: new Map(),
  csv: new Map(),
  undo: new Map(),
  redo: new Map(),
  seq: 100,
  failNext: null,
  rawRequests: [],
  experiments: new Map(),
  cancelBatch: new Set(),
};

function baseImage(kind, roles, extra = {}) {
  return {
    kind,
    height: H,
    width: W,
    dtype: 'uint16',
    bit_depth: 16,
    sha256: { 'ch0.tif': 'a'.repeat(64), 'ch1.tif': 'b'.repeat(64) },
    channels: [0, 1].map((i) => ({
      index: i,
      source_file: i === 0 ? 'slide_A.tif' : 'slide_B.tif',
      page: kind === 'raw_two_channel' ? i : 0,
      label_hint: i === 0 ? '635 [W1]' : '488 [W2]',
      role: roles ? roles[i] : null,
      role_confirmed: !!roles,
      min: 1700,
      max: 42000,
      saturated_fraction: i === 1 ? 0.00004 : 0,
    })),
    pixel_size_um: null,
    tiff_info_text: null,
    embedded_rois: [],
    warnings: [],
    composite_only: false,
    ...extra,
  };
}
function emptyAnalysis(id, name, image, status) {
  return {
    id,
    name,
    created_at: now(),
    updated_at: now(),
    status,
    image,
    regions: [],
    objects: [],
    reporting_mode: 'exclusive',
    detection_request: defReq(),
    provenance: {
      schema_version: 1,
      app_version: 'mock',
      engine: 'bigfish',
      package_versions: {},
      detection_request: null,
      detection_ran_at: null,
      reporting_mode: 'exclusive',
      stale_detection: false,
    },
    slide: { slide_id: null, sampling_date: null, polony_date: null, qc: null, field_area: null, grid_area: null, lab_series: [], efficiency_input: null, sample_volume_ul: null, concentration_factor: null, sampled_area_note: null, counting_area_px2: null, pixel_size_um: null, csv_color_mapping: null, skipped_fields: [], grid: null },
    finalized_at: null,
    undo_depth: 0,
    redo_depth: 0,
    experiment_id: null,
    order_in_experiment: 0,
  };
}

function seed() {
  store.analyses.clear();
  store.engine.clear();
  store.rawRequests = [];
  store.sfAvailable = true;
  store.sfCached = new Set(['general']);
  store.jobs.clear();
  store.csv.clear();
  store.undo.clear();
  store.redo.clear();
  store.experiments.clear();
  store.cancelBatch.clear();
  // 1: ready for review
  const demo = emptyAnalysis('demo', 'Demo slide (in review)', baseImage('raw_two_channel', ['red', 'green'], {
    embedded_rois: [{ source: 'ROI', roi_type: 'rect', name: 'embedded-1', left: 30, top: 30, right: 220, bottom: 200, n_points: null, c_position: 0, z_position: 0, t_position: 0 }],
  }), 'in_review');
  demo.regions = [
    { id: 'reg-1', kind: 'counting', name: 'Region 1', polygon: rect(20, 20, 480, 360), approved: false, approved_at: null, approval_fingerprint: null, inspected: false },
    { id: 'ex-1', kind: 'exclusion', name: 'Edge debris', polygon: rect(0, 0, 14, 14), approved: false, approved_at: null, approval_fingerprint: null, inspected: false },
  ];
  demo.objects = SPOTS.map((s) => {
    const o = objFromSpot(s);
    o.region_id = regionOf(demo, o.x, o.y);
    return o;
  });
  demo.provenance.detection_ran_at = now();
  store.analyses.set('demo', demo);
  // 1a: hybrid (two-engine) review
  const hy = clone(demo);
  hy.id = 'hybrid';
  hy.name = 'Hybrid slide (two engines)';
  hy.detection_request = { ...hy.detection_request, engine: 'hybrid', spotiflow: { model_name: 'general', prob_thresh: null, min_distance: 1, exclude_border: false, subpix: null, n_tiles: null, max_tile_size: null, downscale: 1, normalizer: 'auto', device: 'auto' }, consensus_tolerance_px: 4 };
  store.analyses.set('hybrid', hy);
  applyEngines(hy, hy.detection_request);
  // 1b: researcher reference (GenePix-like 635 stack: suggested roles, embedded points + grid)
  const rs = clone(demo);
  rs.id = 'refslide';
  rs.name = 'Reference slide (researcher marks)';
  rs.regions = [];
  rs.objects.forEach((o) => (o.region_id = null));
  rs.image.channels.forEach((c, i) => {
    c.suggested_role = i === 0 ? 'red' : 'green';
    c.role = null;
    c.role_confirmed = false;
  });
  rs.status = 'needs_channel_mapping';
  store.analyses.set('refslide', rs);
  // 2: needs mapping
  store.analyses.set('unmapped', emptyAnalysis('unmapped', 'Fresh upload (needs mapping)', baseImage('raw_separate_files', null), 'needs_channel_mapping'));
  // 3: composite only
  const comp = emptyAnalysis('composite', 'Composite-only image', baseImage('rgb_composite', ['red', 'green'], { composite_only: true, warnings: ['Image is an RGB composite; raw channels are not available.'] }), 'ready');
  comp.image.channels = comp.image.channels.map((c) => ({ ...c, label_hint: null }));
  store.analyses.set('composite', comp);
  // csv
  const csvText = 'slide,T7A,T7B,field1,field2,field3\nS1,12,,5,,0\nS2,3,4,7,8,\n';
  store.csv.set('csv-1', parseCsv('csv-1', 'manual_counts.csv', csvText));
}

function parseCsv(id, filename, text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  const header = lines[0].split(',').map((s) => s.trim());
  const isField = (h) => /^(field|f)\s*\d+$/i.test(h);
  const rows = lines.slice(1).map((l) => {
    const cells = l.split(',').map((s) => s.trim());
    const meta = {};
    const raw_fields = [];
    const fields = [];
    header.forEach((h, i) => {
      if (i === 0) return;
      if (isField(h)) {
        raw_fields.push(cells[i] ?? '');
        const v = (cells[i] ?? '') === '' ? null : Number.parseInt(cells[i], 10);
        fields.push(Number.isNaN(v) ? null : v);
      } else meta[h] = cells[i] ?? '';
    });
    return { slide_id: cells[0], meta, fields, raw_fields };
  });
  return { id, filename, sha256: 'c'.repeat(64), header, rows, color_mapping: null, notes: ['Blank cells are kept as blank (not zero).'] };
}

// ------------------------------------------------------------------ logic
class HttpError extends Error {
  constructor(status, code, message, detail) {
    super(message);
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}
const getA = (id) => {
  const a = store.analyses.get(id);
  if (!a) throw new HttpError(404, 'not_found', `analysis ${id} not found`);
  return a;
};
// ---- researcher reference (mock): 7 x 4 grid of 60 px cells at (20,20); fields numbered row-major
const REF_ORIGIN = [20, 20];
const REF_CELL = [60, 60];
const REF_COLS = 7;
const REF_ROWS = 4;
const REF_POINTS = [
  { x: 100, y: 100, counter: 1, plane: 0, field: 2 },
  { x: 105, y: 110, counter: 2, plane: 0, field: 2 },
  { x: 200, y: 150, counter: 1, plane: 1, field: 12 },
  { x: 300, y: 250, counter: 1, plane: 0, field: 19 },
  { x: 305, y: 255, counter: 1, plane: 1, field: 19 },
  { x: 60, y: 320, counter: 3, plane: 0, field: 22 },
];
function refCells() {
  const cells = [];
  for (let r = 0; r < REF_ROWS; r++)
    for (let c = 0; c < REF_COLS; c++) {
      const x0 = REF_ORIGIN[0] + c * REF_CELL[0],
        y0 = REF_ORIGIN[1] + r * REF_CELL[1];
      cells.push({ field: r * REF_COLS + c + 1, polygon: rect(x0, y0, x0 + REF_CELL[0], y0 + REF_CELL[1]) });
    }
  return cells;
}
function refInfo(a) {
  if (a.id !== 'refslide') return { available: false, source_file: null, points: [], grid: null, reference_counts: {}, planes: [], warnings: [] };
  const rc = {};
  for (const p of REF_POINTS) {
    rc[p.field] = rc[p.field] || { plane0: 0, plane1: 0 };
    rc[p.field]['plane' + p.plane]++;
  }
  return {
    available: true,
    source_file: 'slide_A.tif',
    points: REF_POINTS,
    grid: { source: 'overlay', origin: REF_ORIGIN, cell: REF_CELL, cols: REF_COLS, rows: REF_ROWS, cells: refCells() },
    reference_counts: rc,
    planes: [0, 1].map((page) => ({ page, point_count: REF_POINTS.filter((p) => p.plane === page).length })),
    warnings: ['Plane meaning is not recorded in the file.'],
  };
}
const live = (a) => a.objects.filter((o) => !o.deleted);

function refresh(a) {
  a.updated_at = now();
  a.undo_depth = (store.undo.get(a.id) || []).length;
  a.redo_depth = (store.redo.get(a.id) || []).length;
  a.slide.counting_area_px2 = a.regions.filter((r) => r.kind === 'counting').reduce((s, r) => s + area(r.polygon), 0) || null;
  a.provenance.reporting_mode = a.reporting_mode;
  if (a.status !== 'finalized' && a.status !== 'detecting' && a.status !== 'needs_channel_mapping') {
    a.status = a.objects.length || a.provenance.detection_ran_at ? 'in_review' : 'ready';
  }
  return a;
}
// ---- counting grid geometry: same rules as fields/grid.py / src/lib/labgrid.ts
function gridCells(g) {
  const rows = Math.max(0, Math.floor(g.rows)), cols = Math.max(0, Math.floor(g.cols));
  const rr = g.corner === 'bl' || g.corner === 'br', rc = g.corner === 'tr' || g.corner === 'br';
  const ro = Array.from({ length: rows }, (_, i) => (rr ? rows - 1 - i : i));
  const co = Array.from({ length: cols }, (_, i) => (rc ? cols - 1 - i : i));
  const seq = [];
  if (g.order === 'row-major') for (const r of ro) for (const c of co) seq.push([r, c]);
  else for (const c of co) for (const r of ro) seq.push([r, c]);
  const ox = g.origin_x + (g.shift_dx || 0), oy = g.origin_y + (g.shift_dy || 0);
  return seq.map(([r, c], n) => {
    const x0 = ox + c * g.cell_w, y0 = oy + r * g.cell_h, x1 = ox + (c + 1) * g.cell_w, y1 = oy + (r + 1) * g.cell_h;
    return { fi: g.first_index + n, polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] };
  });
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function axisShift(origin, n, cell, size, u) {
  const lo = -0.5 - origin, hi = size - 0.5 - n * cell - origin;
  if (hi < lo) return 0;
  const p = Math.ceil(lo), q = Math.floor(hi);
  if (p > q) return lo;
  return p + Math.min(q - p, Math.floor(u * (q - p + 1))) + 0;
}
function snapshot(a) {
  const st = store.undo.get(a.id) || [];
  st.push(clone({ objects: a.objects, regions: a.regions, stale: a.provenance.stale_detection, grid: a.slide.grid ?? null, skipped: a.slide.skipped_fields ?? [] }));
  store.undo.set(a.id, st);
  store.redo.set(a.id, []);
}
function unapproveRegion(a, rid) {
  const r = a.regions.find((x) => x.id === rid);
  if (r && r.approved) {
    r.approved = false;
    r.approved_at = null;
    r.approval_fingerprint = null;
  }
}
function countsFor(a, scopeId, regionIds, isFinal) {
  const objs = live(a).filter((o) => regionIds.includes(o.region_id));
  const n = (c) => objs.filter((o) => o.cls === c).length;
  const red_only = n('red'),
    green_only = n('green'),
    yellow = n('yellow'),
    uncertain = n('uncertain');
  return {
    scope: scopeId,
    red_only,
    green_only,
    yellow,
    uncertain,
    red_positive: red_only + yellow,
    green_positive: green_only + yellow,
    red_positive_excl_debris: objs.filter((o) => (o.cls === 'red' || o.cls === 'yellow') && !o.review_reasons.includes('debris_like')).length,
    green_positive_excl_debris: objs.filter((o) => (o.cls === 'green' || o.cls === 'yellow') && !o.review_reasons.includes('debris_like')).length,
    total_objects_exclusive: red_only + green_only + yellow,
    reporting_mode: a.reporting_mode,
    is_final: isFinal,
  };
}
function counts(a) {
  const cr = a.regions.filter((r) => r.kind === 'counting');
  const fin = a.status === 'finalized';
  return { regions: cr.map((r) => countsFor(a, r.id, [r.id], fin)), slide: countsFor(a, 'slide', cr.map((r) => r.id), fin) };
}
function queue(a) {
  return live(a)
    .filter((o) => !o.reviewed && (o.review_reasons.length || o.cls === 'uncertain'))
    .map((o) => ({ object_id: o.id, reasons: o.review_reasons, priority: (o.cls === 'uncertain' ? 10 : 0) + o.review_reasons.length + (o.review_reasons.some((r) => r === 'bigfish_only' || r === 'spotiflow_only') ? 50 : 0), x: o.x, y: o.y }))
    .sort((p, q) => q.priority - p.priority || p.object_id.localeCompare(q.object_id));
}


// ---- two-engine (hybrid) mock: statuses per SPOT key; shapes follow docs/app/TWO_ENGINE_CONTRACT.md
const ENGINE_STATUS = { 1: 'agreement', 2: 'spotiflow_only', 3: 'agreement', 4: 'bigfish_only', 5: 'agreement', 6: 'bigfish_only' };
const SF_MODELS = ['general', 'hybiss', 'smfish_3d'];
function chanPages(a) {
  return a.image.channels.map((c) => ({ page: c.page, role: c.role_confirmed ? c.role : null, which: c.index === 0 ? 'r' : 'g' }));
}
function runInfo(a, engine, ch, status, model, error) {
  const sf = engine === 'spotiflow';
  return {
    run_id: `run-${engine}-${ch.page}-${++store.seq}`,
    analysis_id: a.id,
    image_sha256: a.image.sha256,
    engine,
    engine_version: sf ? '0.3.1-mock' : '0.6.2-mock',
    model_name: sf ? model : null,
    model_version: sf ? '1.0' : null,
    model_checksum: sf ? 'c0ffee12'.repeat(8) : null,
    page: ch.page,
    channel_role: ch.role,
    params: {},
    preprocessing: {},
    started_at: now(),
    finished_at: now(),
    device: sf ? 'cpu' : 'cpu',
    duration_s: sf ? 31.4 : 1.2,
    status,
    error: error || null,
    n_raw: status === 'ok' ? 7 : 0,
    n_valid: status === 'ok' ? 6 : 0,
    notes: [],
  };
}
function applyEngines(a, req) {
  store.engine.delete(a.id);
  if (!a.image?.channels.every((c) => c.role_confirmed)) return;
  const pages = chanPages(a);
  const runs = [];
  const raws = [];
  const hybrid = req.engine === 'hybrid' || req.engine === 'spotiflow';
  const model = req.spotiflow?.model_name || 'general';
  const failSf = hybrid && model === 'smfish_3d';
  for (const ch of pages) {
    const bf = runInfo(a, 'bigfish', ch, 'ok');
    runs.push(bf);
    const sfr = hybrid ? (failSf ? runInfo(a, 'spotiflow', ch, 'failed', model, 'Spotiflow model "smfish_3d" failed to load: checksum mismatch (mock failure).') : runInfo(a, 'spotiflow', ch, 'ok', model)) : null;
    if (sfr) runs.push(sfr);
    for (const sp of SPOTS) {
      if (!(ch.which === 'r' ? sp.r : sp.g)) continue;
      const st = ENGINE_STATUS[sp.key];
      if (st !== 'spotiflow_only')
        raws.push({ id: `bf-${ch.page}-${sp.key}`, run_id: bf.run_id, analysis_id: a.id, engine: 'bigfish', page: ch.page, channel_role: ch.role, x: sp.x, y: sp.y, score: 20 + sp.key * 3.3, score_kind: 'bigfish_log_response_noise_units', valid: true, region_id: 'reg-1', excluded_reason: null });
      if (sfr && sfr.status === 'ok' && st !== 'bigfish_only')
        raws.push({ id: `sf-${ch.page}-${sp.key}`, run_id: sfr.run_id, analysis_id: a.id, engine: 'spotiflow', page: ch.page, channel_role: ch.role, x: sp.x + (st === 'agreement' ? 1.2 : 0), y: sp.y - 0.8, score: 0.5 + sp.key * 0.07, score_kind: 'spotiflow_prob_raw', valid: true, region_id: 'reg-1', excluded_reason: null });
    }
    // one raw point of each engine inside the exclusion mask (invalid)
    raws.push({ id: `bf-${ch.page}-ex`, run_id: bf.run_id, analysis_id: a.id, engine: 'bigfish', page: ch.page, channel_role: ch.role, x: 6, y: 6, score: 9.1, score_kind: 'bigfish_log_response_noise_units', valid: false, region_id: null, excluded_reason: 'exclusion region' });
    if (sfr && sfr.status === 'ok') raws.push({ id: `sf-${ch.page}-ex`, run_id: sfr.run_id, analysis_id: a.id, engine: 'spotiflow', page: ch.page, channel_role: ch.role, x: 8, y: 9, score: 0.55, score_kind: 'spotiflow_prob_raw', valid: false, region_id: null, excluded_reason: 'exclusion region' });
  }
  store.engine.set(a.id, { runs, raws });
  if (hybrid && !failSf) {
    for (const o of a.objects) {
      const m = /^o-0*(\d+)$/.exec(o.id);
      const key = m ? +m[1] : 0;
      const st = ENGINE_STATUS[key];
      if (!st || o.origin !== 'auto') continue;
      for (const w of ['red', 'green']) {
        const e = o[w];
        if (!e.detected) continue;
        e.consensus_status = st;
        const bf = st !== 'spotiflow_only';
        const sf = st !== 'bigfish_only';
        e.bigfish_x = bf ? e.x : null;
        e.bigfish_y = bf ? e.y : null;
        e.bigfish_score = bf ? 20 + key * 3.3 : null;
        e.spotiflow_x = sf ? e.x + (st === 'agreement' ? 1.2 : 0) : null;
        e.spotiflow_y = sf ? e.y - 0.8 : null;
        e.spotiflow_score = sf ? 0.5 + key * 0.07 : null;
        e.match_distance_px = st === 'agreement' ? 1.44 : null;
      }
      if (st !== 'agreement' && !o.review_reasons.includes(st)) o.review_reasons.push(st);
    }
  }
}
function consensusSummary(a) {
  const pages = chanPages(a).map((ch) => {
    const w = ch.which === 'r' ? 'red' : 'green';
    const c = { page: ch.page, channel_role: ch.role, agreement: 0, bigfish_only: 0, spotiflow_only: 0, human_added: 0, human_rejected: 0, confirmed: 0, rejected: 0, unresolved: 0 };
    for (const o of a.objects) {
      const st = o[w].consensus_status || (o.origin === 'manual' && o[w].detected ? 'human_added' : null);
      if (!st) continue;
      if (o.deleted) {
        c.rejected++;
        c.human_rejected++;
        continue;
      }
      c[st]++;
      if (o.reviewed) c.confirmed++;
      else c.unresolved++;
    }
    return c;
  });
  const total = pages.reduce((t, p) => {
    for (const k of Object.keys(t)) if (k !== 'page' && k !== 'channel_role') t[k] += p[k];
    return t;
  }, { page: null, channel_role: null, agreement: 0, bigfish_only: 0, spotiflow_only: 0, human_added: 0, human_rejected: 0, confirmed: 0, rejected: 0, unresolved: 0 });
  return { pages, total, qc: qcStatus(a) };
}

const QC_NOTE = 'agreement is not ground truth. Agreement objects are NOT reviewed unless sampled; the interval bounds the error rate of the UNSAMPLED agreements only if the sample is random.';
const nForMargin = (N, p) => {
  if (N <= 0) return 0;
  const n0 = (1.96 * 1.96 * p * (1 - p)) / (0.05 * 0.05);
  return Math.min(N, Math.ceil((N * n0) / (N + n0 - 1) - 1e-9));
};
/** exact Clopper-Pearson 95 % interval by bisection on the binomial tail */
function clopperPearson(k, n) {
  const logC = (m, j) => { let r = 0; for (let i = 1; i <= j; i++) r += Math.log((m - j + i) / i); return r; };
  const cdf = (p, kk) => { let t = 0; for (let j = 0; j <= kk; j++) t += Math.exp(logC(n, j) + (j ? j * Math.log(p) : 0) + (n - j ? (n - j) * Math.log(1 - p) : 0)); return t; };
  const solve = (f) => { let lo = 1e-12, hi = 1 - 1e-12; for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (f(mid)) lo = mid; else hi = mid; } return (lo + hi) / 2; };
  const lower = k <= 0 ? 0 : solve((p) => 1 - cdf(p, k - 1) < 0.025);
  const upper = k >= n ? 1 : solve((p) => cdf(p, k) > 0.025);
  return [lower, upper];
}
function qcStatus(a) {
  const sampled = a.objects.filter((o) => o.review_reasons.includes('qc_sample'));
  const rejected = sampled.filter((o) => o.deleted).length;
  const confirmed = sampled.filter((o) => !o.deleted && o.reviewed).length;
  const reviewed = rejected + confirmed;
  const pop = a.objects.filter((o) => [o.red, o.green].some((e) => e.detected) && [o.red, o.green].filter((e) => e.detected).every((e) => e.consensus_status === 'agreement')).length;
  return {
    sample_size: sampled.length, reviewed, pending: sampled.length - reviewed, confirmed, rejected, population_agreements: pop, seed: null,
    error_rate: reviewed ? rejected / reviewed : null, ci95: reviewed ? clopperPearson(rejected, reviewed) : null,
    interval_method: 'Clopper-Pearson exact, 95 % two-sided', n_for_margin_0_05: nForMargin(pop, 0.153), n_for_margin_0_05_conservative: nForMargin(pop, 0.5), planning_error_rate: 0.153, note: QC_NOTE,
  };
}

function applyAction(a, type, p) {
  if (store.failNext) {
    const m = store.failNext;
    store.failNext = null;
    throw new HttpError(409, 'mock_failure', m);
  }
  if (a.status === 'finalized') throw new HttpError(409, 'finalized', 'Analysis is finalized; reopen it to edit.');
  const byId = (id) => {
    const o = a.objects.find((x) => x.id === id && !x.deleted);
    if (!o) throw new HttpError(404, 'object_not_found', `object ${id} not found`);
    return o;
  };
  const touchRegions = (...ids) => ids.filter(Boolean).forEach((r) => unapproveRegion(a, r));
  const nextId = () => `o-m${++store.seq}`;
  const inImg = (x, y) => x >= -0.5 && y >= -0.5 && x <= W - 0.5 && y <= H - 0.5;
  const mutating = !['confirm', 'region_inspected', 'approve_region', 'unapprove_region', 'accept_region'].includes(type);
  const validate = () => {
    if (type === 'add' && !inImg(p.x, p.y)) throw new HttpError(422, 'out_of_bounds', 'Point is outside the image.');
    if (type === 'merge' && (!p.object_ids || p.object_ids.length < 2)) throw new HttpError(422, 'bad_payload', 'merge needs at least 2 objects');
  };
  validate();
  snapshot(a);
  const undoSt = store.undo.get(a.id);
  const fail = (e) => {
    undoSt.pop();
    throw e;
  };
  try {
    switch (type) {
      case 'add': {
        const o = { id: nextId(), x: p.x, y: p.y, cls: p.cls, origin: 'manual', red: p.cls === 'red' || p.cls === 'yellow' ? { ...noEv(), detected: true, x: p.x, y: p.y } : noEv(), green: p.cls === 'green' || p.cls === 'yellow' ? { ...noEv(), detected: true, x: p.x, y: p.y } : noEv(), association_distance_px: null, review_reasons: [], ambiguous_with: [], reviewed: true, deleted: false, region_id: regionOf(a, p.x, p.y), history: [`added manually as ${p.cls}`] };
        a.objects.push(o);
        touchRegions(o.region_id);
        break;
      }
      case 'delete':
        for (const id of p.object_ids) {
          const o = byId(id);
          o.deleted = true;
          o.history.push('deleted by user');
          touchRegions(o.region_id);
        }
        break;
      case 'move': {
        const o = byId(p.object_id);
        const old = o.region_id;
        o.x = p.x;
        o.y = p.y;
        if (o.origin === 'auto') o.origin = 'edited';
        o.region_id = regionOf(a, o.x, o.y);
        o.history.push(`moved to ${p.x.toFixed(1)},${p.y.toFixed(1)}`);
        touchRegions(old, o.region_id);
        break;
      }
      case 'reclassify':
        for (const id of p.object_ids) {
          const o = byId(id);
          o.cls = p.cls;
          if (o.origin === 'auto') o.origin = 'edited';
          o.reviewed = true;
          o.history.push(`reclassified as ${p.cls}`);
          touchRegions(o.region_id);
        }
        break;
      case 'merge': {
        const os = p.object_ids.map(byId);
        const x = os.reduce((s, o) => s + o.x, 0) / os.length,
          y = os.reduce((s, o) => s + o.y, 0) / os.length;
        const cls = os.every((o) => o.cls === os[0].cls) ? os[0].cls : 'uncertain';
        os.forEach((o) => {
          o.deleted = true;
          touchRegions(o.region_id);
        });
        const m = { ...clone(os[0]), id: nextId(), x, y, cls, origin: 'manual', reviewed: false, deleted: false, region_id: regionOf(a, x, y), history: [`merged ${os.map((o) => o.id).join('+')}`] };
        a.objects.push(m);
        touchRegions(m.region_id);
        break;
      }
      case 'split': {
        const o = byId(p.object_id);
        o.deleted = true;
        touchRegions(o.region_id);
        p.positions.forEach((pos, i) => {
          const n = { ...clone(o), id: nextId(), x: pos[0], y: pos[1], cls: p.classes[i], origin: 'manual', reviewed: false, deleted: false, region_id: regionOf(a, pos[0], pos[1]), history: [`split from ${o.id}`] };
          a.objects.push(n);
          touchRegions(n.region_id);
        });
        break;
      }
      case 'confirm':
        for (const id of p.object_ids) {
          const o = byId(id);
          o.reviewed = true;
          o.history.push('confirmed by user');
        }
        break;
      case 'region_inspected': {
        const r = a.regions.find((x) => x.id === p.region_id);
        if (!r) throw new HttpError(404, 'region_not_found', 'region not found');
        r.inspected = !!p.inspected;
        if (!r.inspected) unapproveRegion(a, r.id);
        break;
      }
      case 'accept_region': {
        const r = a.regions.find((x) => x.id === p.region_id);
        if (!r) throw new HttpError(404, 'region_not_found', 'region not found');
        if (!r.inspected) throw new HttpError(409, 'not_inspected', 'Region has not been marked as inspected for omissions.');
        live(a).filter((o) => o.region_id === r.id && o.cls !== 'uncertain').forEach((o) => (o.reviewed = true));
        break;
      }
      case 'approve_region': {
        const r = a.regions.find((x) => x.id === p.region_id);
        if (!r) throw new HttpError(404, 'region_not_found', 'region not found');
        const reasons = [];
        if (r.kind !== 'counting') reasons.push('only counting regions can be approved');
        if (!r.inspected) reasons.push('region has not been marked as inspected for omissions');
        const unresolved = live(a).filter((o) => o.region_id === r.id && o.cls === 'uncertain').length;
        if (unresolved) reasons.push(`${unresolved} unresolved uncertain object(s)`);
        if (reasons.length) throw new HttpError(409, 'approval_refused', `Cannot approve ${r.name}: ${reasons.join('; ')}.`, { reasons });
        r.approved = true;
        r.approved_at = now();
        r.approval_fingerprint = 'fp-' + store.seq++;
        break;
      }
      case 'unapprove_region':
        unapproveRegion(a, p.region_id);
        break;
      default:
        throw new HttpError(400, 'unknown_action', `unknown action ${type}`);
    }
  } catch (e) {
    fail(e);
  }
  if (!mutating) {
    // keep undo stack for review actions too (contract: all edits go through actions)
  }
  if (a.status === 'ready') a.status = 'in_review';
  return refresh(a);
}

function runJob(a, req, extra = {}, stepMs = 280) {
  const id = `job-${++store.seq}`;
  const job = { id, analysis_id: a.id, status: 'queued', progress: 0, message: 'queued', error: null, started_at: null, finished_at: null, kind: 'detect', parent_id: null, experiment_id: null, detail: null, ...extra };
  store.jobs.set(id, job);
  a.status = 'detecting';
  let step = 0;
  const total = 10;
  const t = setInterval(() => {
    if (job.status === 'cancelled') {
      clearInterval(t);
      a.status = a.objects.length ? 'in_review' : 'ready';
      refresh(a);
      return;
    }
    step++;
    job.status = 'running';
    job.started_at = job.started_at || now();
    job.progress = step / total;
    job.message = step < 5 ? 'detecting red spots' : step < 9 ? 'detecting green spots' : 'matching channels';
    if (req.red.spot_radius_px > 20 || req.green.spot_radius_px > 20) {
      clearInterval(t);
      job.status = 'failed';
      job.error = 'Spot radius too large for this image (mock failure).';
      job.finished_at = now();
      a.status = a.objects.length ? 'in_review' : 'ready';
      refresh(a);
      return;
    }
    if (step >= total) {
      clearInterval(t);
      // replace only untouched auto objects; preserve manual/edited/reviewed
      const kept = a.objects.filter((o) => o.origin !== 'auto' || o.reviewed || o.deleted);
      const keptIds = new Set(kept.map((o) => o.id));
      const fresh = SPOTS.filter((s) => !(req.red.threshold_scale > 1.5 && s.reasons.includes('weak_signal'))).map((s) => objFromSpot(s)).filter((o) => !keptIds.has(o.id));
      fresh.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
      const changed = new Set([...a.objects.filter((o) => !keptIds.has(o.id)).map((o) => o.region_id), ...fresh.map((o) => o.region_id)]);
      snapshot(a);
      a.objects = [...kept, ...fresh];
      applyEngines(a, req);
      changed.forEach((r) => unapproveRegion(a, r));
      a.provenance.detection_ran_at = now();
      a.provenance.stale_detection = false;
      a.provenance.detection_request = clone(req);
      a.detection_request = clone(req);
      a.status = 'in_review';
      job.status = 'done';
      job.message = 'done';
      job.finished_at = now();
      refresh(a);
    }
  }, stepMs);
  return job;
}

// Mirrors docs/app/LAB_EXPORT.md: missing items => 422 listing each; fields > 24 refused; drafts refused in batch.
function labProblems(a, series) {
  const m = a.slide;
  const p = [];
  const need = (ok, t) => ok || p.push(`${a.name}: missing ${t}`);
  need(m.slide_id, 'slide_id');
  need(m.sampling_date, 'sampling_date');
  need(m.polony_date, 'polony_date');
  need(m.qc === 0 || m.qc === 1, 'qc');
  need(m.sample_volume_ul > 0, 'sample_volume_ul');
  need(m.concentration_factor > 0, 'concentration_factor');
  need(m.field_area > 0, 'field_area');
  need(m.grid_area > 0, 'grid_area');
  need(m.concentration_formula_confirmed, 'confirmation of field geometry and series mapping');
  if (series && !(m.lab_series || []).some((q) => q.name === series)) p.push(`${a.name}: series "${series}" is not defined`);
  const f = a.regions.filter((r) => r.kind === 'counting' && r.field_index != null);
  if (!f.length) p.push(`${a.name}: no counting regions with a field_index`);
  if (f.some((r) => r.field_index > 24)) p.push(`${a.name}: field_index above 24 is not supported (template ends at field 24)`);
  return p;
}

function exportBody(a, kind) {
  const fin = a.status === 'finalized';
  const status = fin ? 'FINAL' : 'DRAFT';
  const base = `${a.name.replace(/\W+/g, '_')}_${kind}_${status}`;
  if (kind === 'provenance') return { type: 'application/json', name: `${base}.json`, body: JSON.stringify({ status, provenance: a.provenance }, null, 2) };
  if (kind === 'bundle') return { type: 'application/zip', name: `${base}.zip`, body: 'PK-mock' };
  const rows = kind === 'objects' ? ['id,x,y,cls,status', ...live(a).map((o) => `${o.id},${o.x},${o.y},${o.cls},${status}`)] : ['scope,status', `slide,${status}`];
  return { type: 'text/csv', name: `${base}.csv`, body: rows.join('\n') + '\n' };
}


// ------------------------------------------------------------------ experiments (docs/app/EXPERIMENTS_API.md)
const getE = (eid) => {
  const e = store.experiments.get(eid);
  if (!e) throw new HttpError(404, 'experiment_not_found', `experiment ${eid} not found`);
  return e;
};
const expSlides = (eid) => [...store.analyses.values()].filter((a) => a.experiment_id === eid).sort((p, q) => p.order_in_experiment - q.order_in_experiment);
const isActiveJob = (j) => !!j && (j.status === 'queued' || j.status === 'running');
const lastBatch = (eid) => [...store.jobs.values()].filter((j) => j.kind === 'batch' && j.experiment_id === eid).pop() ?? null;
const lastJobOf = (aid) => [...store.jobs.values()].filter((j) => j.analysis_id === aid).pop() ?? null;
const mappingOk = (a) => !!a.image?.channels.every((c) => c.role_confirmed);
function expDict(e) {
  return { ...clone(e), status: isActiveJob(lastBatch(e.id)) ? 'running' : 'open' };
}
function slideRow(a) {
  const regs = a.regions.filter((r) => r.kind === 'counting');
  const approved = regs.length > 0 && regs.every((r) => r.approved);
  return {
    id: a.id, name: a.name, order_in_experiment: a.order_in_experiment, status: a.status,
    export_status: a.status === 'finalized' ? 'finalized' : approved ? 'approved' : 'draft',
    updated_at: a.updated_at, finalized_at: a.finalized_at, mapping_confirmed: mappingOk(a),
    image: a.image ? { width: a.image.width, height: a.image.height, kind: a.image.kind, dtype: a.image.dtype, n_channels: a.image.channels.length } : null,
    has_grid: !!a.slide.grid, n_counting_regions: regs.length, n_fields: regs.filter((r) => r.field_index != null).length,
    n_approved: regs.filter((r) => r.approved).length, approved, review_queue: a.objects.length ? queue(a).length : 0,
    detection_ran_at: a.provenance.detection_ran_at, engine: a.provenance.detection_ran_at ? a.detection_request.engine || 'bigfish' : null,
    counts: counts(a).slide, last_job: lastJobOf(a.id),
  };
}
const expView = (eid) => ({ ...expDict(getE(eid)), slides: expSlides(eid).map(slideRow), last_batch: lastBatch(eid) });
function batchView(eid) {
  const b = lastBatch(eid);
  if (!b) return { batch: null, slides: [], summary: null };
  const latest = new Map();
  for (const j of store.jobs.values()) if (j.parent_id === b.id) latest.set(j.analysis_id, j);
  const active = isActiveJob(b);
  const slides = b.detail.slides.map((p) => {
    const c = latest.get(p.analysis_id);
    const status = p.skip ? 'skipped' : c ? c.status : active ? 'pending' : 'not_run';
    return { analysis_id: p.analysis_id, name: p.name, order: p.order, status, skip_reason: p.skip, override: !!p.override, job_id: c?.id ?? null, progress: c?.progress ?? 0, message: c?.message ?? '', error: c?.error ?? null, started_at: c?.started_at ?? null, finished_at: c?.finished_at ?? null };
  });
  const summary = { total: slides.filter((x) => x.status !== 'skipped').length };
  for (const k of ['done', 'failed', 'cancelled', 'running', 'queued', 'pending', 'not_run', 'skipped']) summary[k] = slides.filter((x) => x.status === k).length;
  return { batch: b, slides, summary };
}
const EXP_KEYS = ['name', 'notes', 'default_detection', 'default_grid'];
function checkKeys(body, allowed) {
  const unk = Object.keys(body || {}).filter((k) => !allowed.includes(k));
  if (unk.length) throw new HttpError(422, 'unknown_fields', `unknown field(s): ${unk.join(', ')}`, { fields: unk });
}
function checkDetection(d, where) {
  if (d && d.preview_window != null) throw new HttpError(422, 'invalid_request', `${where}.preview_window is not a run setting`);
}
const runReq = (d) => {
  const base = defReq();
  const x = clone(d || {});
  return { ...base, ...x, red: { ...base.red, ...(x.red || {}) }, green: { ...base.green, ...(x.green || {}) }, preview_window: null };
};
function failedChild(bid, eid, aid, error) {
  const j = { id: `job-${++store.seq}`, analysis_id: aid, status: 'failed', progress: 0, message: '', error, started_at: now(), finished_at: now(), kind: 'detect', parent_id: bid, experiment_id: eid, detail: null };
  store.jobs.set(j.id, j);
  return j;
}
/** Sequential "unattended" run: skip and continue; a slide whose name contains "fail" fails in the engine. */
async function runExperimentBatch(e, b, plan, reqs) {
  const run = plan.filter((p) => !p.skip);
  let ok = 0, failed = 0;
  b.status = 'running';
  b.started_at = now();
  for (let i = 0; i < run.length; i++) {
    if (store.cancelBatch.has(b.id)) {
      b.status = 'cancelled';
      b.message = `cancelled after ${i} of ${run.length} slides (${ok} done, ${failed} failed)`;
      b.finished_at = now();
      e.status = 'open';
      return;
    }
    const p = run[i];
    b.progress = i / run.length;
    b.message = `slide ${i + 1} / ${run.length}: ${p.name} (${ok} done, ${failed} failed)`;
    const a = store.analyses.get(p.analysis_id);
    let status = 'failed';
    if (!a) failedChild(b.id, e.id, p.analysis_id, 'not_found: analysis not found');
    else if (!mappingOk(a)) failedChild(b.id, e.id, a.id, 'channel_mapping_unconfirmed: confirm the red/green channel mapping before detecting');
    else if (!a.regions.some((r) => r.kind === 'counting')) failedChild(b.id, e.id, a.id, 'no_counting_region: no counting region: apply a grid or draw a counting region first');
    else if (/fail/i.test(a.name)) {
      const j = failedChild(b.id, e.id, a.id, null);
      j.status = 'running';
      j.finished_at = null;
      await delay(400);
      j.status = 'failed';
      j.error = `RuntimeError: mock engine failure on '${a.name}' (the file name contains "fail")`;
      j.finished_at = now();
    } else {
      const j = runJob(a, reqs[a.id], { parent_id: b.id, experiment_id: e.id }, 60);
      while (isActiveJob(j)) await delay(50);
      status = j.status;
    }
    if (status === 'done') ok++;
    else failed++;
    b.progress = (i + 1) / run.length;
    b.message = `${ok + failed} / ${run.length} slides (${ok} done, ${failed} failed)`;
  }
  const nSkip = plan.length - run.length;
  b.status = failed ? 'done_with_failures' : 'done';
  b.message = `${ok} / ${run.length} done, ${failed} failed${nSkip ? `, ${nSkip} skipped` : ''}`;
  b.finished_at = now();
  e.status = 'open';
}
function csvRows(e, sel) {
  const head = ['Slide name', 'What counted', 'NofPol', 'NofGrids', ...Array.from({ length: 24 }, (_, i) => `Field${i + 1}`), 'Experiment', 'Status', 'Engine'];
  const lines = [head.join(',')];
  const included = [], skipped = [];
  const LABEL = { red: 'red, not yellow', green: 'green, not yellow', yellow: 'yellow' };
  for (const a of expSlides(e.id)) {
    const fields = a.regions.filter((r) => r.kind === 'counting' && r.field_index != null);
    if (!a.provenance.detection_ran_at) { skipped.push({ analysis_id: a.id, name: a.name, reason: 'no_detection' }); continue; }
    if (!fields.length) { skipped.push({ analysis_id: a.id, name: a.name, reason: 'no_grid_fields' }); continue; }
    included.push({ analysis_id: a.id, name: a.name, slide_name: a.name, status: slideRow(a).export_status });
    for (const cls of sel) {
      const per = Array(24).fill('');
      let tot = 0;
      for (const r of fields) {
        const n = live(a).filter((o) => o.region_id === r.id && cls.includes(o.cls)).length;
        if (r.field_index <= 24) per[r.field_index - 1] = n;
        tot += n;
      }
      const q = (v) => (/[",\r\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
      lines.push([a.name, cls.map((c) => LABEL[c]).join(' and '), tot, fields.length, ...per, e.name, slideRow(a).export_status, a.detection_request.engine || 'bigfish'].map(q).join(','));
    }
  }
  return { text: lines.join('\r\n') + '\r\n', included, skipped };
}
async function handleExperiments(req, res, url, route, m) {
  let mt;
  const readJson = async () => JSON.parse((await readBody(req)).toString() || '{}');
  if (route === '/experiments' && m === 'GET') {
    const list = [...store.experiments.values()].sort((p, q) => q.updated_at.localeCompare(p.updated_at)).map((e) => {
      const sl = expSlides(e.id);
      return { ...expDict(e), n_slides: sl.length, n_detected: sl.filter((a) => a.status === 'in_review' || a.status === 'finalized').length, n_finalized: sl.filter((a) => a.status === 'finalized').length, last_batch: lastBatch(e.id) };
    });
    return send(res, 200, list);
  }
  if (route === '/experiments' && m === 'POST') {
    const b = await readJson();
    checkKeys(b, EXP_KEYS);
    if (!(b.name || '').trim()) throw new HttpError(422, 'invalid_name', 'the experiment needs a name');
    checkDetection(b.default_detection, 'default_detection');
    const id = (++store.seq).toString(16).padStart(12, '0');
    store.experiments.set(id, { id, name: b.name.trim().slice(0, 120), notes: b.notes || '', created_at: now(), updated_at: now(), default_grid: b.default_grid ?? null, default_detection: b.default_detection ?? defReq(), status: 'open' });
    return send(res, 200, expView(id));
  }
  if (!(mt = route.match(/^\/experiments\/([^/]+)(\/.*)?$/))) throw new HttpError(404, 'not_found', `${m} ${route}`);
  const e = getE(mt[1]);
  const sub = mt[2] || '';
  if (sub === '' && m === 'GET') return send(res, 200, expView(e.id));
  if (sub === '' && m === 'PUT') {
    const b = await readJson();
    checkKeys(b, EXP_KEYS);
    if ('name' in b) {
      if (!(b.name || '').trim()) throw new HttpError(422, 'invalid_name', 'the experiment needs a name');
      e.name = b.name.trim().slice(0, 120);
    }
    if ('notes' in b) e.notes = b.notes || '';
    if ('default_detection' in b) {
      checkDetection(b.default_detection, 'default_detection');
      e.default_detection = b.default_detection ?? defReq();
    }
    if ('default_grid' in b) e.default_grid = b.default_grid;
    e.updated_at = now();
    return send(res, 200, expView(e.id));
  }
  if (sub === '' && m === 'DELETE') {
    if (isActiveJob(lastBatch(e.id))) throw new HttpError(409, 'batch_in_progress', 'a batch run of this experiment is active; cancel it first');
    const sl = expSlides(e.id);
    const del = url.searchParams.get('delete_slides') === 'true';
    if (del) {
      const busy = sl.filter((a) => a.status === 'detecting').map((a) => a.id);
      if (busy.length) throw new HttpError(409, 'job_in_progress', 'a detection job is running on a slide of this experiment', { analysis_ids: busy });
      const fin = sl.filter((a) => a.status === 'finalized').map((a) => a.id);
      if (fin.length) throw new HttpError(409, 'analysis_finalized', 'finalized slides are not deleted; reopen them first, or delete the experiment without delete_slides (its slides are kept)', { analysis_ids: fin });
      sl.forEach((a) => store.analyses.delete(a.id));
    } else sl.forEach((a) => ((a.experiment_id = null), (a.order_in_experiment = 0)));
    for (const [k, j] of store.jobs) if (j.kind === 'batch' && j.experiment_id === e.id) store.jobs.delete(k);
    store.experiments.delete(e.id);
    return send(res, 200, { ok: true, deleted_slides: del ? sl.map((a) => a.id) : [], detached_slides: del ? [] : sl.map((a) => a.id) });
  }
  const nextOrder = () => Math.max(0, ...expSlides(e.id).map((a) => a.order_in_experiment)) + 1;
  if (sub === '/import' && m === 'POST') {
    await delay(150);
    const buf = (await readBody(req)).toString('latin1');
    const files = [...buf.matchAll(/name="files"; filename="([^"]+)"/g)].map((x) => x[1]);
    const tpl = expSlides(e.id).find(mappingOk);
    let order = nextOrder();
    const created = [], errors = [];
    for (const f of files) {
      if (!/\.tiff?$/i.test(f)) {
        errors.push({ file: f, code: 'unsupported_file', message: `'${f}': only .tif/.tiff files are accepted` });
        continue;
      }
      const id = (++store.seq).toString(16).padStart(12, '0');
      const a = emptyAnalysis(id, f.replace(/\.tiff?$/i, ''), baseImage('raw_two_channel', null), 'needs_channel_mapping');
      a.image.channels.forEach((c) => (c.source_file = f));
      a.experiment_id = e.id;
      a.order_in_experiment = order++;
      let copied = null;
      if (tpl && tpl.image.kind === a.image.kind) {
        a.image.channels.forEach((c, i) => ((c.role = tpl.image.channels[i].role), (c.role_confirmed = true)));
        a.status = 'ready';
        copied = tpl.id;
      }
      store.analyses.set(id, a);
      created.push({ id, name: a.name, order_in_experiment: a.order_in_experiment, status: a.status, mapping_copied_from: copied });
    }
    if (!created.length) throw new HttpError(422, 'import_failed', 'no file could be imported', { errors });
    return send(res, 200, { created, errors });
  }
  if (sub === '/slides' && m === 'POST') {
    const b = await readJson();
    const ids = [...new Set(b.analysis_ids || [])];
    const other = ids.map(getA).filter((a) => a.experiment_id && a.experiment_id !== e.id).map((a) => ({ analysis_id: a.id, experiment_id: a.experiment_id }));
    if (other.length) throw new HttpError(409, 'in_other_experiment', 'these analyses belong to another experiment; remove them from it first', { analyses: other });
    let order = nextOrder();
    for (const id of ids) {
      const a = getA(id);
      if (a.experiment_id === e.id) continue;
      a.experiment_id = e.id;
      a.order_in_experiment = order++;
    }
    return send(res, 200, expView(e.id));
  }
  if ((mt = sub.match(/^\/slides\/([^/]+)$/)) && m === 'DELETE') {
    const a = getA(mt[1]);
    if (a.experiment_id !== e.id) throw new HttpError(404, 'not_in_experiment', 'this analysis is not part of the experiment');
    a.experiment_id = null;
    a.order_in_experiment = 0;
    return send(res, 200, expView(e.id));
  }
  if (sub === '/order' && m === 'PUT') {
    const b = await readJson();
    const cur = expSlides(e.id).map((a) => a.id);
    if (JSON.stringify([...(b.analysis_ids || [])].sort()) !== JSON.stringify([...cur].sort())) throw new HttpError(422, 'invalid_order', 'analysis_ids must list every slide of the experiment exactly once', { analysis_ids: cur });
    b.analysis_ids.forEach((id, i) => (getA(id).order_in_experiment = i + 1));
    return send(res, 200, expView(e.id));
  }
  if (sub === '/run' && m === 'GET') return send(res, 200, batchView(e.id));
  if (sub === '/run' && m === 'POST') {
    const b = await readJson();
    checkKeys(b, ['only_missing', 'analysis_ids', 'overrides']);
    if (isActiveJob(lastBatch(e.id))) throw new HttpError(409, 'batch_in_progress', 'a batch run of this experiment is already active');
    let sl = expSlides(e.id);
    if (b.analysis_ids) {
      const extra = b.analysis_ids.filter((id) => !sl.some((a) => a.id === id));
      if (extra.length) throw new HttpError(422, 'not_in_experiment', 'analysis_ids must be slides of this experiment', { analysis_ids: extra });
      sl = sl.filter((a) => b.analysis_ids.includes(a.id));
    }
    const ov = b.overrides || {};
    for (const [k, d] of Object.entries(ov)) checkDetection(d, `overrides.${k}`);
    const plan = sl.map((a) => ({ analysis_id: a.id, name: a.name, order: a.order_in_experiment, skip: a.status === 'finalized' ? 'finalized' : b.only_missing && a.provenance.detection_ran_at ? 'already_detected' : null, override: a.id in ov }));
    if (!plan.some((p) => !p.skip)) throw new HttpError(422, 'nothing_to_run', `no slide to run (all finalized${b.only_missing ? ' or already detected)' : ')'}`, { slides: plan });
    const reqs = Object.fromEntries(sl.map((a) => [a.id, runReq(ov[a.id] || e.default_detection)]));
    const job = { id: `job-${++store.seq}`, analysis_id: `experiment:${e.id}`, status: 'queued', progress: 0, message: 'queued', error: null, started_at: null, finished_at: null, kind: 'batch', parent_id: null, experiment_id: e.id, detail: { slides: plan, only_missing: !!b.only_missing, default_detection: clone(e.default_detection), overrides: clone(ov) } };
    store.jobs.set(job.id, job);
    e.status = 'running';
    setTimeout(() => void runExperimentBatch(e, job, plan, reqs), 100);
    return send(res, 200, job);
  }
  const series = () => {
    const raw = url.searchParams.getAll('series');
    const sel = (raw.length ? raw : ['red', 'green', 'yellow']).map((x) => x.split(',').map((c) => c.trim()));
    if (sel.some((c) => c.some((x) => !['red', 'green', 'yellow'].includes(x)))) throw new HttpError(422, 'invalid_selection', 'unknown class');
    return sel;
  };
  const stem = e.name.replace(/[^\w.-]+/g, '_');
  if (sub === '/export.csv' && m === 'GET') {
    const r = csvRows(e, series());
    return send(res, 200, '﻿' + r.text, { 'Content-Type': 'text/csv', 'Content-Disposition': `attachment; filename="${stem}_experiment.csv"`, 'X-Slides-Included': String(r.included.length), 'X-Slides-Skipped': String(r.skipped.length) });
  }
  if (sub === '/provenance.json' && m === 'GET') {
    const r = csvRows(e, series());
    const lb = batchView(e.id);
    const body = { exported_at: now(), app_version: 'mock', schema_version: 1, experiment: clone(e), csv: { header: r.text.split('\r\n')[0].split(','), included_slides: r.included, skipped_slides: r.skipped }, last_batch: lb.batch ? { job_id: lb.batch.id, status: lb.batch.status, message: lb.batch.message, error: lb.batch.error, summary: lb.summary } : null, slides: expSlides(e.id).map((a) => ({ analysis_id: a.id, name: a.name, order_in_experiment: a.order_in_experiment, grid: a.slide.grid, skipped_fields: (a.slide.skipped_fields || []).map((x) => x.field_index) })) };
    return send(res, 200, JSON.stringify(body, null, 2), { 'Content-Type': 'application/json', 'Content-Disposition': `attachment; filename="${stem}_provenance.json"` });
  }
  throw new HttpError(404, 'not_found', `${m} /api${route}`);
}

// ------------------------------------------------------------------ http
function send(res, status, body, headers = {}) {
  const isBuf = Buffer.isBuffer(body) || typeof body === 'string';
  res.writeHead(status, { 'Content-Type': isBuf ? 'text/plain' : 'application/json', ...headers });
  res.end(isBuf ? body : JSON.stringify(body));
}
async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json' };

async function handle(req, res) {
  const url = new URL(req.url, 'http://x');
  const p = url.pathname;
  if (!p.startsWith('/api/')) {
    // static
    let f = path.join(DIST, p === '/' ? 'index.html' : p);
    if (!f.startsWith(DIST) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) f = path.join(DIST, 'index.html');
    if (!fs.existsSync(f)) return send(res, 404, 'frontend/dist not built (run npm run build)');
    return send(res, 200, fs.readFileSync(f), { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
  }
  const route = p.slice(4);
  const m = req.method;
  let mt;
  if (route === '/health') return send(res, 200, { ok: true, version: 'mock-0.1', bigfish_version: '0.6.2-mock' });
  if (route === '/engines' && m === 'GET')
    return send(res, 200, [
      { name: 'bigfish', available: true, reason: null, version: '0.6.2-mock', device: { cuda: false, mps: false, cpu: true }, models: [] },
      {
        name: 'spotiflow',
        available: store.sfAvailable,
        reason: store.sfAvailable ? null : 'the spotiflow package is not installed (pip install spotiflow)',
        version: store.sfAvailable ? '0.3.1-mock' : null,
        device: { cuda: false, mps: false, cpu: true },
        models: SF_MODELS.map((n) => ({ name: n, cached: store.sfCached.has(n), version: '1.0', checksum: store.sfCached.has(n) ? 'c0ffee12'.repeat(8) : null })),
      },
    ]);
  if (route === '/__mock/spotiflow' && m === 'POST') {
    store.sfAvailable = !!JSON.parse((await readBody(req)).toString() || '{}').available;
    return send(res, 200, { ok: true });
  }
  if (route === '/__mock/raw-requests' && m === 'GET') return send(res, 200, store.rawRequests);
  if (route === '/engines/spotiflow/prepare' && m === 'POST') {
    const body = JSON.parse((await readBody(req)).toString() || '{}');
    if (!store.sfAvailable) throw new HttpError(409, 'engine_unavailable', 'Spotiflow is not installed.');
    const id = `job-${++store.seq}`;
    const job = { id, analysis_id: 'engine-spotiflow', status: 'running', progress: 0, message: `downloading ${body.model_name}`, error: null, started_at: now(), finished_at: null, kind: 'prepare_model', parent_id: null, experiment_id: null, detail: null };
    store.jobs.set(id, job);
    const t = setInterval(() => {
      job.progress = Math.min(1, job.progress + 0.25);
      if (job.progress >= 1) {
        clearInterval(t);
        store.sfCached.add(body.model_name);
        job.status = 'done';
        job.message = 'ready';
        job.finished_at = now();
      }
    }, 250);
    return send(res, 200, job);
  }
  if (route === '/__mock/reset' && m === 'POST') {
    seed();
    return send(res, 200, { ok: true });
  }
  if (route === '/__mock/fail-next-action' && m === 'POST') {
    store.failNext = JSON.parse((await readBody(req)).toString() || '{}').message || 'injected failure';
    return send(res, 200, { ok: true });
  }
  if ((mt = route.match(/^\/__mock\/(force-final|confirm-lab)\/(.+)$/)) && m === 'POST') {
    const a = getA(mt[2]);
    if (mt[1] === 'force-final') a.status = 'finalized';
    else a.slide.concentration_formula_confirmed = true;
    return send(res, 200, { ok: true });
  }
  if (route === '/lab-export' && m === 'POST') {
    const b = JSON.parse((await readBody(req)).toString() || '{}');
    if ((b.experiment_id != null) === (b.analysis_ids != null)) throw new HttpError(422, 'invalid_request', 'give analysis_ids or experiment_id (not both)');
    if (b.experiment_id != null) {
      getE(b.experiment_id);
      b.analysis_ids = expSlides(b.experiment_id).filter((a) => a.status === 'finalized').map((a) => a.id);
      if (!b.analysis_ids.length) throw new HttpError(409, 'no_finalized_slides', 'the experiment has no finalized slide');
    }
    const problems = [];
    for (const id of b.analysis_ids || []) {
      const a = store.analyses.get(id);
      if (!a) problems.push(`${id}: not found`);
      else if (a.status !== 'finalized') problems.push(`${a.name}: not finalized (draft analyses cannot be lab-exported)`);
      else problems.push(...labProblems(a, b.style === 't7' ? b.series : null));
    }
    if (!(b.analysis_ids || []).length) problems.push('no analyses selected');
    if (problems.length) throw new HttpError(422, 'lab_export_blocked', 'Lab export blocked:\n' + problems.join('\n'), { blockers: problems });
    const ext = b.style === 'counts' ? 'xlsx' : b.analysis_ids.length > 1 ? 'zip' : b.format;
    const nm = b.style === 'counts' ? 'Counts_mock' : `${b.series}_mock`;
    return send(res, 200, 'mock-lab-export', { 'Content-Type': 'application/octet-stream', 'Content-Disposition': `attachment; filename="${nm}.${ext}"` });
  }
  if (route.startsWith('/experiments')) return handleExperiments(req, res, url, route, m);
  if (route === '/analyses' && m === 'GET')
    return send(res, 200, [...store.analyses.values()].map((a) => ({ id: a.id, name: a.name, status: a.status, updated_at: a.updated_at, finalized_at: a.finalized_at, counts: a.objects.length ? counts(a).slide : null, experiment_id: a.experiment_id ?? null, order_in_experiment: a.order_in_experiment ?? 0 })));
  if (route === '/analyses' && m === 'POST') {
    await delay(300);
    const buf = (await readBody(req)).toString('latin1');
    const files = [...buf.matchAll(/name="files"; filename="([^"]+)"/g)].map((x) => x[1]);
    const nm = buf.match(/name="name"\r\n\r\n([^\r]*)\r\n/);
    if (!files.length) throw new HttpError(422, 'no_files', 'Upload 1–2 TIFF files.');
    if (files.length > 2) throw new HttpError(422, 'too_many_files', 'At most 2 files.');
    if (files.some((f) => !/\.tiff?$/i.test(f))) throw new HttpError(422, 'not_tiff', 'Only TIFF files are accepted.');
    const id = `an-${++store.seq}`;
    const a = emptyAnalysis(id, (nm && nm[1]) || files[0].replace(/\.tiff?$/i, ''), baseImage(files.length === 2 ? 'raw_separate_files' : 'raw_two_channel', null), 'needs_channel_mapping');
    a.image.channels.forEach((c, i) => (c.source_file = files[Math.min(i, files.length - 1)]));
    store.analyses.set(id, a);
    return send(res, 200, a);
  }
  if ((mt = route.match(/^\/analyses\/([^/]+)$/))) {
    const a = getA(mt[1]);
    if (m === 'GET') return send(res, 200, a);
    if (m === 'DELETE') {
      store.analyses.delete(a.id);
      return send(res, 200, { ok: true });
    }
  }
  if ((mt = route.match(/^\/analyses\/([^/]+)\/(.+)$/))) {
    const a = getA(mt[1]);
    const sub = mt[2];
    const body = m === 'POST' || m === 'PUT' ? JSON.parse((await readBody(req)).toString() || '{}') : null;
    if (sub === 'channels' && m === 'PUT') {
      const had = a.objects.length > 0;
      for (const mp of body.mapping) {
        const c = a.image.channels.find((x) => x.index === mp.index);
        if (c) {
          c.role = mp.role;
          c.role_confirmed = true;
        }
      }
      if (a.status === 'needs_channel_mapping') a.status = 'ready';
      if (had) {
        a.provenance.stale_detection = true;
        a.regions.forEach((r) => unapproveRegion(a, r.id));
      }
      return send(res, 200, refresh(a));
    }
    if (sub === 'regions' && m === 'PUT') {
      snapshot(a);
      const old = new Map(a.regions.map((r) => [r.id, r]));
      let changed = false;
      a.regions = body.regions.map((r) => {
        const o = old.get(r.id);
        if (!o || JSON.stringify(o.polygon) !== JSON.stringify(r.polygon) || o.kind !== r.kind) {
          changed = true;
          return { ...r, approved: false, approved_at: null, approval_fingerprint: null };
        }
        return { ...r, approved: o.approved, approved_at: o.approved_at, approval_fingerprint: o.approval_fingerprint, inspected: o.inspected };
      });
      if (old.size !== a.regions.length) changed = true;
      a.objects.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
      if (changed) {
        if (a.objects.length) a.provenance.stale_detection = true;
        a.regions.filter((r) => r.kind === 'counting').forEach((r) => unapproveRegion(a, r.id));
      }
      return send(res, 200, refresh(a));
    }
    if (sub === 'grid-selection' && m === 'PUT') {
      if (a.status === 'finalized') throw new HttpError(409, 'analysis_finalized', 'Analysis is finalized; reopen it to edit.');
      const fi = body.field_index;
      const isF = (r) => r.kind === 'counting' && r.field_index != null;
      const cur = a.regions.find((r) => isF(r) && r.field_index === fi);
      const sk = (a.slide.skipped_fields || []).find((x) => x.field_index === fi);
      if (body.counted ? cur : !cur && sk) return send(res, 200, refresh(a));
      if (body.counted ? !sk : !cur) throw new HttpError(404, 'unknown_field', `field ${fi} is not part of the grid`);
      snapshot(a);
      if (body.counted) {
        const r = { id: `reg-g${++store.seq}`, kind: 'counting', name: sk.name, field_index: fi, polygon: sk.polygon, approved: false, approved_at: null, approval_fingerprint: null, inspected: false };
        a.regions = [...a.regions.filter(isF), r].sort((p, q) => p.field_index - q.field_index).concat(a.regions.filter((x) => !isF(x)));
        a.slide.skipped_fields = a.slide.skipped_fields.filter((x) => x.field_index !== fi);
      } else {
        a.regions = a.regions.filter((r) => r.id !== cur.id);
        a.slide.skipped_fields = [...(a.slide.skipped_fields || []).filter((x) => x.field_index !== fi), { field_index: fi, name: cur.name, polygon: cur.polygon }];
      }
      a.objects.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
      return send(res, 200, refresh(a));
    }
    if (sub === 'grid' && m === 'PUT') {
      // mirrors PUT /analyses/{aid}/grid (src/polony_app/api/main.py) and fields/grid.py
      if (a.status === 'finalized') throw new HttpError(409, 'analysis_finalized', 'Analysis is finalized; reopen it to edit.');
      const isF = (r) => r.kind === 'counting' && r.field_index != null;
      const fields = a.regions.filter(isF);
      const others = a.regions.filter((r) => !isF(r));
      const skipped = a.slide.skipped_fields || [];
      if (!body.grid && !body.randomise) {
        if (!a.slide.grid) return send(res, 200, refresh(a));
        const own = new Set(gridCells(a.slide.grid).map((c) => c.fi));
        const appr = fields.filter((r) => own.has(r.field_index) && r.approved).map((r) => r.field_index);
        if (appr.length && !body.force) throw new HttpError(409, 'grid_fields_approved', `approved grid field(s) ${appr.join(', ')}: unapprove them first`);
        snapshot(a);
        a.regions = [...fields.filter((r) => !own.has(r.field_index)), ...others];
        a.slide.skipped_fields = skipped.filter((x) => !own.has(x.field_index));
        a.slide.grid = null;
        a.objects.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
        return send(res, 200, refresh(a));
      }
      const spec = clone(body.grid || a.slide.grid);
      if (!spec) throw new HttpError(422, 'no_grid', 'randomise needs a grid');
      spec.seed = spec.seed ?? null;
      spec.shift_dx = spec.shift_dx ?? 0;
      spec.shift_dy = spec.shift_dy ?? 0;
      if (!(spec.rows >= 1 && spec.cols >= 1 && spec.cell_w > 0 && spec.cell_h > 0 && spec.first_index >= 1)) throw new HttpError(422, 'invalid_grid', 'bad grid');
      if (spec.first_index + spec.rows * spec.cols - 1 > 24) throw new HttpError(422, 'invalid_grid', 'field numbers exceed 24');
      if (body.randomise) {
        spec.seed = body.seed ?? Math.floor(Math.random() * 2 ** 31);
        const rng = mulberry32(spec.seed);
        const ux = rng(), uy = rng();
        spec.shift_dx = axisShift(spec.origin_x, spec.cols, spec.cell_w, W, ux);
        spec.shift_dy = axisShift(spec.origin_y, spec.rows, spec.cell_h, H, uy);
      }
      snapshot(a);
      const cells = gridCells(spec);
      const own = new Set(cells.map((c) => c.fi));
      const byIdx = new Map();
      for (const r of fields) if (!byIdx.has(r.field_index)) byIdx.set(r.field_index, r);
      const skIdx = new Map(skipped.map((x) => [x.field_index, x]));
      const made = [];
      const off = [];
      for (const c of cells) {
        const cur = byIdx.get(c.fi);
        if (cur) {
          const moved = JSON.stringify(cur.polygon) !== JSON.stringify(c.polygon);
          made.push(moved ? { ...cur, polygon: c.polygon, approved: false, approved_at: null, approval_fingerprint: null, inspected: false } : cur);
        } else if (skIdx.has(c.fi)) off.push({ ...skIdx.get(c.fi), polygon: c.polygon });
        else made.push({ id: `reg-g${++store.seq}`, kind: 'counting', name: `Field-${c.fi}`, field_index: c.fi, polygon: c.polygon, approved: false, approved_at: null, approval_fingerprint: null, inspected: false });
      }
      const kept = body.replace === false ? fields.filter((r) => !own.has(r.field_index)) : [];
      const keptOff = body.replace === false ? skipped.filter((x) => !own.has(x.field_index)) : [];
      a.regions = [...made, ...kept].sort((p, q) => p.field_index - q.field_index).concat(others);
      a.slide.skipped_fields = [...keptOff, ...off].sort((p, q) => p.field_index - q.field_index);
      a.slide.grid = spec;
      a.objects.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
      return send(res, 200, refresh(a));
    }
    if (sub === 'settings' && m === 'PUT') {
      if (body.reporting_mode) a.reporting_mode = body.reporting_mode;
      if (body.slide) a.slide = { ...a.slide, ...body.slide, counting_area_px2: a.slide.counting_area_px2, grid: a.slide.grid ?? null };
      if (body.detection_request) {
        a.detection_request = body.detection_request;
        if (a.objects.length) a.provenance.stale_detection = true;
      }
      return send(res, 200, refresh(a));
    }
    if (sub === 'detect' && m === 'POST') {
      if (!a.image.channels.every((c) => c.role_confirmed)) throw new HttpError(409, 'channels_unconfirmed', 'Confirm the channel mapping before detection.');
      return send(res, 200, runJob(a, body));
    }
    if (sub === 'detect/preview' && m === 'POST') {
      await delay(500);
      const w = body.preview_window;
      if (!w) throw new HttpError(422, 'no_window', 'preview_window required');
      const objs = SPOTS.filter((s) => s.x >= w[0] && s.x <= w[2] && s.y >= w[1] && s.y <= w[3]).map((s) => ({ ...objFromSpot(s, 'p'), region_id: null }));
      return send(res, 200, { objects: objs, notes: [`mock preview: ${objs.length} candidates in window`] });
    }
    if (sub === 'reference' && m === 'GET') return send(res, 200, refInfo(a));
    if (sub === 'fields/import' && m === 'POST') {
      const info = refInfo(a);
      if (!info.available) throw new HttpError(422, 'no_grid', 'This analysis has no researcher grid to import.');
      const existing = a.regions.filter((r) => /^Field-/.test(r.name));
      if (existing.length && !body.replace) throw new HttpError(409, 'fields_exist', 'Field-* regions already exist; set replace to overwrite them.');
      const [dx, dy] = body.shift || [0, 0];
      const plane = body.plane == null ? 0 : body.plane;
      const counted = new Set(Object.entries(info.reference_counts).filter(([, v]) => v['plane' + plane] > 0).map(([k]) => Number(k)));
      snapshot(a);
      a.regions = a.regions.filter((r) => !/^Field-/.test(r.name));
      for (const c of info.grid.cells) {
        if (body.mode === 'counted' && !counted.has(c.field)) continue;
        a.regions.push({ id: `reg-ref-${c.field}`, kind: 'counting', name: `Field-${c.field}`, polygon: c.polygon.map(([x, y]) => [x + dx, y + dy]), approved: false, approved_at: null, approval_fingerprint: null, inspected: false, field_index: c.field });
      }
      a.objects.forEach((o) => (o.region_id = regionOf(a, o.x, o.y)));
      if (a.objects.length) a.provenance.stale_detection = true;
      return send(res, 200, refresh(a));
    }
    if (sub === 'fields/compare' && m === 'GET') {
      const info = refInfo(a);
      const plane = url.searchParams.get('plane') == null ? 0 : Number(url.searchParams.get('plane'));
      const ct = url.searchParams.get('count_type') || 'red_positive';
      const cts = ['red_positive', 'green_positive', 'red_only', 'green_only', 'yellow'];
      if (!cts.includes(ct)) throw new HttpError(422, 'bad_count_type', `count_type must be one of ${cts.join(', ')}`);
      const rows = a.regions
        .filter((r) => r.kind === 'counting' && r.field_index != null)
        .map((r) => {
          const c = countsFor(a, r.id, [r.id], false)[ct];
          const rc = info.reference_counts[r.field_index];
          const ref = rc ? rc['plane' + plane] : 0;
          return { field: r.field_index, reference_count: ref, app_count: c, delta: c - ref };
        })
        .sort((p, q) => p.field - q.field);
      const n = rows.length;
      const xs = rows.map((r) => r.reference_count), ys = rows.map((r) => r.app_count);
      const mean = (v) => v.reduce((s, x) => s + x, 0) / (v.length || 1);
      const mx = mean(xs), my = mean(ys);
      const sxx = xs.reduce((s, x) => s + (x - mx) ** 2, 0), syy = ys.reduce((s, y) => s + (y - my) ** 2, 0);
      const sxy = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
      return send(res, 200, {
        rows,
        summary: {
          n_fields: n,
          pearson_r: n > 1 && sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null,
          slope: n > 1 && sxx > 0 ? sxy / sxx : null,
          mae: n ? mean(rows.map((r) => Math.abs(r.delta))) : null,
          total_reference: xs.reduce((s, x) => s + x, 0),
          total_app: ys.reduce((s, y) => s + y, 0),
        },
      });
    }
    if (sub === 'regions/auto-chamber' && m === 'POST') {
      const rim = Number(body.rim_px ?? 40);
      if (!(rim >= 0) || rim * 2 >= Math.min(W, H)) throw new HttpError(422, 'bad_rim', 'rim_px too large for this image.');
      const polygon = rect(rim, rim, W - rim, H - rim);
      return send(res, 200, { polygon, area_px2: area(polygon), rim_px: rim, method: 'mock-inset', warnings: ['mock: chamber found by inset, not by image content'], suggested_exclusions: [rect(W - rim - 50, rim, W - rim, rim + 40)] });
    }
    if (sub === 'counts' && m === 'GET') return send(res, 200, counts(a));
    if (sub === 'engine-runs' && m === 'GET') return send(res, 200, store.engine.get(a.id)?.runs ?? []);
    if (sub === 'raw-detections' && m === 'GET') {
      const q = url.searchParams;
      let r = store.engine.get(a.id)?.raws ?? [];
      if (q.get('engine')) r = r.filter((d) => d.engine === q.get('engine'));
      if (q.get('page') != null) r = r.filter((d) => d.page === +q.get('page'));
      if (q.get('valid') != null) r = r.filter((d) => String(d.valid) === q.get('valid'));
      store.rawRequests.push(req.url);
      return send(res, 200, r.slice(0, +(q.get('limit') || 1000)));
    }
    if (sub === 'consensus/summary' && m === 'GET') return send(res, 200, consensusSummary(a));
    if (sub === 'consensus/qc-sample' && m === 'POST') {
      if (body.fraction != null && !(body.fraction >= 0 && body.fraction <= 1)) throw new HttpError(422, 'invalid_fraction', 'fraction must be in 0..1');
      if (body.n != null && !(Number.isInteger(body.n) && body.n >= 0)) throw new HttpError(422, 'invalid_n', 'n must be a non-negative integer');
      const pool = live(a).filter((o) => !o.reviewed && [o.red.consensus_status, o.green.consensus_status].includes('agreement') && !o.review_reasons.includes('qc_sample')).sort((x, y) => x.id.localeCompare(y.id));
      const want = body.n != null ? body.n : body.fraction != null ? Math.max(40, Math.ceil(pool.length * body.fraction)) : 40;
      const n = pool.length ? Math.min(Math.max(want, 1), pool.length) : 0;
      const off = pool.length ? Math.abs(body.seed | 0) % pool.length : 0;
      const pick = pool.map((_, i) => pool[(i + off) % pool.length]).slice(0, n);
      pick.forEach((o) => o.review_reasons.push('qc_sample'));
      store.rawRequests.push('qc ' + JSON.stringify(body));
      const pop = a.objects.filter((o) => !o.deleted && [o.red, o.green].filter((e) => e.detected).every((e) => e.consensus_status === 'agreement') && (o.red.detected || o.green.detected)).length;
      return send(res, 200, { marked: pick.length, newly_marked: pick.length, eligible: pool.length, n, n_requested: want, fraction: body.fraction ?? null, seed: body.seed ?? 0, population_agreements: pop, n_for_margin_0_05: nForMargin(pop, 0.153), n_for_margin_0_05_conservative: nForMargin(pop, 0.5), qc: qcStatus(a), note: QC_NOTE });
    }
    if (sub === 'review-queue' && m === 'GET') return send(res, 200, queue(a));
    if (sub === 'actions' && m === 'POST') {
      await delay(40);
      return send(res, 200, applyAction(a, body.type, body.payload || {}));
    }
    if ((sub === 'undo' || sub === 'redo') && m === 'POST') {
      const from = store[sub].get(a.id) || [];
      const to = store[sub === 'undo' ? 'redo' : 'undo'].get(a.id) || [];
      const s = from.pop();
      if (!s) throw new HttpError(409, 'nothing_to_undo', `Nothing to ${sub}.`);
      to.push(clone({ objects: a.objects, regions: a.regions, stale: a.provenance.stale_detection, grid: a.slide.grid ?? null, skipped: a.slide.skipped_fields ?? [] }));
      store[sub].set(a.id, from);
      store[sub === 'undo' ? 'redo' : 'undo'].set(a.id, to);
      a.objects = s.objects;
      a.regions = s.regions;
      a.provenance.stale_detection = s.stale;
      if (s.grid !== undefined) {
        a.slide.grid = s.grid;
        a.slide.skipped_fields = s.skipped;
      }
      return send(res, 200, refresh(a));
    }
    if (sub === 'finalize' && m === 'POST') {
      const cr = a.regions.filter((r) => r.kind === 'counting');
      const bl = [];
      if (!cr.length) bl.push('no counting region');
      if (cr.some((r) => !r.approved)) bl.push('not all counting regions approved');
      if (!a.image.channels.every((c) => c.role_confirmed)) bl.push('channel mapping not confirmed');
      if (a.provenance.stale_detection) bl.push('detection is stale');
      if (bl.length) throw new HttpError(409, 'finalize_blocked', `Cannot finalize: ${bl.join('; ')}.`, { blockers: bl });
      a.status = 'finalized';
      a.finalized_at = now();
      return send(res, 200, refresh(a));
    }
    if (sub === 'reopen' && m === 'POST') {
      a.status = 'in_review';
      a.finalized_at = null;
      a.regions.forEach((r) => unapproveRegion(a, r.id));
      return send(res, 200, refresh(a));
    }
    if (sub === 'lab-export' && m === 'GET') {
      if (a.status !== 'finalized') throw new HttpError(409, 'not_final', 'Lab export needs a finalized analysis.');
      const sr = url.searchParams.get('series');
      const pr = labProblems(a, sr);
      if (pr.length) throw new HttpError(422, 'lab_export_blocked', 'Lab export blocked:\n' + pr.join('\n'), { blockers: pr });
      const fmt = url.searchParams.get('format') || 'xlsx';
      return send(res, 200, 'mock-lab-export', { 'Content-Type': 'application/octet-stream', 'Content-Disposition': `attachment; filename="${sr}_mock.${fmt}"` });
    }
    if (sub === 'export' && m === 'GET') {
      const e = exportBody(a, url.searchParams.get('kind') || 'objects');
      return send(res, 200, e.body, { 'Content-Type': e.type, 'Content-Disposition': `attachment; filename="${e.name}"` });
    }
    if ((mt = sub.match(/^image\/(.+)$/)) && m === 'GET') {
      const view = mt[1];
      if (!['red', 'green', 'composite', 'processed_red', 'processed_green', 'ch0', 'ch1'].includes(view)) throw new HttpError(404, 'bad_view', `unknown view ${view}`);
      if (a.status === 'needs_channel_mapping' && !view.startsWith('ch') && view !== 'composite') throw new HttpError(409, 'channels_unconfirmed', 'Confirm the channel mapping first.');
      await delay(60);
      const r = renderView(view, url.searchParams);
      return send(res, 200, r.png, {
        'Content-Type': 'image/png',
        'X-Image-Width': String(W),
        'X-Image-Height': String(H),
        'X-Crop': r.crop.join(','),
        'X-Auto-Lo': String(AUTO_LO),
        'X-Auto-Hi': String(AUTO_HI),
        'Access-Control-Expose-Headers': 'X-Image-Width,X-Image-Height,X-Crop,X-Auto-Lo,X-Auto-Hi',
      });
    }
    if ((mt = sub.match(/^thumb\/(.+)$/)) && m === 'GET') {
      const o = a.objects.find((x) => x.id === decodeURIComponent(mt[1]));
      if (!o) throw new HttpError(404, 'object_not_found', 'object not found');
      const q = new URLSearchParams({ x0: Math.round(o.x - 16), y0: Math.round(o.y - 16), x1: Math.round(o.x + 16), y1: Math.round(o.y + 16), max_dim: url.searchParams.get('size') || '64' });
      return send(res, 200, renderView('composite', q).png, { 'Content-Type': 'image/png' });
    }
    if ((mt = sub.match(/^histogram\/(red|green)$/)) && m === 'GET') return send(res, 200, { bins: [0, 1], counts: [1, 1], lo: AUTO_LO, hi: AUTO_HI, p995: AUTO_HI });
  }
  if ((mt = route.match(/^\/jobs\/([^/]+)(\/cancel)?$/))) {
    const j = store.jobs.get(mt[1]);
    if (!j) throw new HttpError(404, 'job_not_found', 'job not found');
    if (mt[2] && m === 'POST' && j.kind === 'batch') {
      // like the server: the slide being detected finishes, no further slide starts
      if (j.status === 'running' || j.status === 'queued') store.cancelBatch.add(j.id);
      return send(res, 200, j);
    }
    if (mt[2] && m === 'POST') {
      if (j.status === 'running' || j.status === 'queued') {
        j.status = 'cancelled';
        j.message = 'cancelled by user';
        j.finished_at = now();
        const a = store.analyses.get(j.analysis_id);
        if (a) {
          a.status = a.objects.length ? 'in_review' : 'ready';
          refresh(a);
        }
      }
    }
    return send(res, 200, j);
  }
  if (route === '/csv-imports' && m === 'GET') return send(res, 200, [...store.csv.values()]);
  if (route === '/csv-imports' && m === 'POST') {
    const buf = (await readBody(req)).toString('utf8');
    const fn = (buf.match(/filename="([^"]+)"/) || [])[1] || 'upload.csv';
    const text = buf.split(/\r?\n\r?\n/).slice(1).join('\n\n').replace(/\r?\n--[-\w]+--\s*$/, '');
    const id = `csv-${++store.seq}`;
    const c = parseCsv(id, fn, text);
    store.csv.set(id, c);
    return send(res, 200, c);
  }
  if ((mt = route.match(/^\/csv-imports\/([^/]+)$/))) {
    const c = store.csv.get(mt[1]);
    if (!c) throw new HttpError(404, 'csv_not_found', 'csv import not found');
    if (m === 'PUT') {
      const b = JSON.parse((await readBody(req)).toString() || '{}');
      if (b.color_mapping && !['red', 'green', 'yellow'].includes(b.color_mapping)) throw new HttpError(422, 'bad_colour', 'color_mapping must be red, green or yellow');
      c.color_mapping = b.color_mapping || null;
    }
    return send(res, 200, c);
  }
  throw new HttpError(404, 'not_found', `${m} ${p}`);
}

seed();
const server = http.createServer((req, res) => {
  handle(req, res).catch((e) => {
    if (e instanceof HttpError) return send(res, e.status, { code: e.code, message: e.message, detail: e.detail ?? null });
    console.error(e);
    send(res, 500, { code: 'internal', message: String(e?.message || e) });
  });
});
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  server.listen(PORT, '127.0.0.1', () => console.log(`mock API on http://127.0.0.1:${PORT}`));
}
export { server };
