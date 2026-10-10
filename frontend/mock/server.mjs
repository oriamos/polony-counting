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
  analyses: new Map(),
  jobs: new Map(),
  csv: new Map(),
  undo: new Map(),
  redo: new Map(),
  seq: 100,
  failNext: null,
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
    slide: { slide_id: null, sampling_date: null, polony_date: null, qc: null, field_area: null, grid_area: null, lab_series: [], efficiency_input: null, sample_volume_ul: null, concentration_factor: null, sampled_area_note: null, counting_area_px2: null, pixel_size_um: null, csv_color_mapping: null },
    finalized_at: null,
    undo_depth: 0,
    redo_depth: 0,
  };
}

function seed() {
  store.analyses.clear();
  store.jobs.clear();
  store.csv.clear();
  store.undo.clear();
  store.redo.clear();
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
function snapshot(a) {
  const st = store.undo.get(a.id) || [];
  st.push(clone({ objects: a.objects, regions: a.regions, stale: a.provenance.stale_detection }));
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
    .map((o) => ({ object_id: o.id, reasons: o.review_reasons, priority: (o.cls === 'uncertain' ? 10 : 0) + o.review_reasons.length, x: o.x, y: o.y }))
    .sort((p, q) => q.priority - p.priority || p.object_id.localeCompare(q.object_id));
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

function runJob(a, req) {
  const id = `job-${++store.seq}`;
  const job = { id, analysis_id: a.id, status: 'queued', progress: 0, message: 'queued', error: null, started_at: null, finished_at: null };
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
  }, 280);
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
  if (route === '/analyses' && m === 'GET')
    return send(res, 200, [...store.analyses.values()].map((a) => ({ id: a.id, name: a.name, status: a.status, updated_at: a.updated_at, finalized_at: a.finalized_at, counts: a.objects.length ? counts(a).slide : null })));
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
    if (sub === 'settings' && m === 'PUT') {
      if (body.reporting_mode) a.reporting_mode = body.reporting_mode;
      if (body.slide) a.slide = { ...a.slide, ...body.slide, counting_area_px2: a.slide.counting_area_px2 };
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
      to.push(clone({ objects: a.objects, regions: a.regions, stale: a.provenance.stale_detection }));
      store[sub].set(a.id, from);
      store[sub === 'undo' ? 'redo' : 'undo'].set(a.id, to);
      a.objects = s.objects;
      a.regions = s.regions;
      a.provenance.stale_detection = s.stale;
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
