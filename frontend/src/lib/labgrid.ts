import type { Analysis, GridSpec as ApiGridSpec, Region } from '../api/types';
import { pointInPolygon, rectPolygon, type Poly } from './geometry';

/** Template columns end at field 24; the server refuses larger numbers. */
export const MAX_FIELD = 24;

export type GridOrder = 'row-major' | 'col-major';
export type StartCorner = 'tl' | 'tr' | 'bl' | 'br';

export interface GridSpec {
  rows: number;
  cols: number;
  cellW: number;
  cellH: number;
  originX: number;
  originY: number;
  order: GridOrder;
  corner: StartCorner;
  firstIndex: number;
  /** seed of the random shift (null/absent = not randomised); the server picks it on Randomise */
  seed?: number | null;
  /** shift of the whole grid from (originX, originY), px; effective origin = origin + shift */
  shiftDx?: number;
  shiftDy?: number;
}

export interface GridCell {
  row: number;
  col: number;
  index: number;
  polygon: Poly;
}

/**
 * Numbering order is a CHOICE (not known from the image). row-major: walk along a row first, then step to the next
 * row; col-major: walk down a column first. The start corner sets the direction of both walks. No serpentine.
 */
export function generateGrid(g: GridSpec): GridCell[] {
  const ox = g.originX + (g.shiftDx ?? 0);
  const oy = g.originY + (g.shiftDy ?? 0);
  const rows = Math.max(0, Math.floor(g.rows));
  const cols = Math.max(0, Math.floor(g.cols));
  const rev = {
    r: g.corner === 'bl' || g.corner === 'br', // start at the bottom
    c: g.corner === 'tr' || g.corner === 'br', // start at the right
  };
  const rOrder = Array.from({ length: rows }, (_, i) => (rev.r ? rows - 1 - i : i));
  const cOrder = Array.from({ length: cols }, (_, i) => (rev.c ? cols - 1 - i : i));
  const seq: [number, number][] = [];
  if (g.order === 'row-major') for (const r of rOrder) for (const c of cOrder) seq.push([r, c]);
  else for (const c of cOrder) for (const r of rOrder) seq.push([r, c]);
  return seq.map(([row, col], n) => ({
    row,
    col,
    index: g.firstIndex + n,
    polygon: rectPolygon(ox + col * g.cellW, oy + row * g.cellH, ox + (col + 1) * g.cellW, oy + (row + 1) * g.cellH),
  }));
}

/** The grid cell containing image point (x, y) (cells are half-open [x0, x1) x [y0, y1)); null outside the grid. */
export function cellAt(g: GridSpec, x: number, y: number): GridCell | null {
  const rows = Math.max(0, Math.floor(g.rows));
  const cols = Math.max(0, Math.floor(g.cols));
  if (!(g.cellW > 0 && g.cellH > 0)) return null;
  const col = Math.floor((x - g.originX - (g.shiftDx ?? 0)) / g.cellW);
  const row = Math.floor((y - g.originY - (g.shiftDy ?? 0)) / g.cellH);
  if (row < 0 || col < 0 || row >= rows || col >= cols) return null;
  return generateGrid(g).find((c) => c.row === row && c.col === col) ?? null;
}

/** Small seeded PRNG (floats in [0, 1)); bit-identical to `mulberry32` in src/polony_app/fields/grid.py. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One axis of randomShift: whole pixels, uniform over the slack; 0 when the grid is larger than the image. */
function axisShift(origin: number, n: number, cell: number, size: number, u: number): number {
  const lo = -0.5 - origin;
  const hi = size - 0.5 - n * cell - origin;
  if (hi < lo) return 0;
  const a = Math.ceil(lo), b = Math.floor(hi);
  if (a > b) return lo; // less than one pixel of slack: put it at the edge
  return a + Math.min(b - a, Math.floor(u * (b - a + 1))) + 0; // + 0 turns -0 into 0
}

/**
 * Random position for the whole grid, relative to (originX, originY): cell size and count never change and the grid
 * stays inside the image ([-0.5, W-0.5] in the pixel-centre convention). Mirrors `random_shift` on the server, which
 * is authoritative (it picks the seed on Randomise); with `mulberry32(seed)` both give the same shift.
 */
export function randomShift(g: GridSpec, imageW: number, imageH: number, rng: () => number): { dx: number; dy: number } {
  const ux = rng();
  const uy = rng();
  return { dx: axisShift(g.originX, Math.floor(g.cols), g.cellW, imageW, ux), dy: axisShift(g.originY, Math.floor(g.rows), g.cellH, imageH, uy) };
}

/** Server spec -> form spec. */
export function fromApiGrid(s: ApiGridSpec): GridSpec {
  return {
    rows: s.rows,
    cols: s.cols,
    cellW: s.cell_w,
    cellH: s.cell_h,
    originX: s.origin_x,
    originY: s.origin_y,
    order: s.order,
    corner: s.corner,
    firstIndex: s.first_index,
    seed: s.seed,
    shiftDx: s.shift_dx,
    shiftDy: s.shift_dy,
  };
}

/** Form spec -> server spec. */
export function toApiGrid(g: GridSpec): ApiGridSpec {
  return {
    rows: g.rows,
    cols: g.cols,
    cell_w: g.cellW,
    cell_h: g.cellH,
    origin_x: g.originX,
    origin_y: g.originY,
    order: g.order,
    corner: g.corner,
    first_index: g.firstIndex,
    seed: g.seed ?? null,
    shift_dx: g.shiftDx ?? 0,
    shift_dy: g.shiftDy ?? 0,
  };
}

export type GridFieldState = 'on' | 'off' | 'missing';

/** On/off state of every cell of the stored grid: on = a field region exists, off = remembered in skipped_fields,
 * missing = neither (the region was deleted by hand; re-apply the grid to get it back). */
export function storedGridCells(a: Analysis): (GridCell & { state: GridFieldState })[] {
  const g = a.slide.grid;
  if (!g) return [];
  const on = new Set(a.regions.filter(isLabField).map((r) => r.field_index as number));
  const off = new Set((a.slide.skipped_fields ?? []).map((s) => s.field_index));
  return generateGrid(fromApiGrid(g)).map((c) => ({ ...c, state: on.has(c.index) ? 'on' : off.has(c.index) ? 'off' : 'missing' }));
}

/** Default grid for an image: 4 x 7 cells of ~316 px (the researcher's test image), centred, oriented by aspect. */
export function defaultGrid(w: number, h: number): GridSpec {
  const landscape = w >= h;
  const rows = landscape ? 4 : 7;
  const cols = landscape ? 7 : 4;
  let cell = 316;
  if (cols * cell > w || rows * cell > h) cell = Math.max(1, Math.floor(Math.min(w / cols, h / rows)));
  return {
    rows,
    cols,
    cellW: cell,
    cellH: cell,
    originX: Math.max(0, Math.round((w - cols * cell) / 2)),
    originY: Math.max(0, Math.round((h - rows * cell) / 2)),
    order: 'row-major',
    corner: 'tl',
    firstIndex: 1,
  };
}

/** Fraction of a polygon's bounding-box sample points that fall inside any exclusion polygon (sampled 12x12). */
export function exclusionCoverage(poly: Poly, exclusions: Poly[]): number {
  if (!exclusions.length) return 0;
  const xs = poly.map((p) => p[0]);
  const ys = poly.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const N = 12;
  let inside = 0, hit = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const x = x0 + ((i + 0.5) / N) * (x1 - x0);
      const y = y0 + ((j + 0.5) / N) * (y1 - y0);
      if (!pointInPolygon(x, y, poly)) continue;
      inside++;
      if (exclusions.some((e) => pointInPolygon(x, y, e))) hit++;
    }
  return inside ? hit / inside : 0;
}

export type CoverageState = 'none' | 'partial' | 'full';
export const coverageState = (frac: number): CoverageState => (frac <= 0 ? 'none' : frac >= 0.999 ? 'full' : 'partial');

export interface FieldIssues {
  tooHigh: number[];
  invalid: number[];
  duplicates: number[];
}

/** Check a list of field indices (null entries ignored). */
export function fieldIssues(indices: (number | null | undefined)[]): FieldIssues {
  const seen = new Map<number, number>();
  const out: FieldIssues = { tooHigh: [], invalid: [], duplicates: [] };
  for (const v of indices) {
    if (v == null) continue;
    if (!Number.isInteger(v) || v < 1) {
      out.invalid.push(v);
      continue;
    }
    if (v > MAX_FIELD) out.tooHigh.push(v);
    seen.set(v, (seen.get(v) ?? 0) + 1);
  }
  out.duplicates = [...seen].filter(([, n]) => n > 1).map(([k]) => k);
  return out;
}

export const hasIssues = (i: FieldIssues) => i.tooHigh.length + i.invalid.length + i.duplicates.length > 0;

export function describeIssues(i: FieldIssues): string[] {
  const m: string[] = [];
  if (i.invalid.length) m.push(`field number(s) ${i.invalid.join(', ')} are not whole numbers >= 1`);
  if (i.tooHigh.length) m.push(`field number(s) ${[...new Set(i.tooHigh)].join(', ')} exceed ${MAX_FIELD} (the lab template ends at field ${MAX_FIELD})`);
  if (i.duplicates.length) m.push(`duplicate field number(s): ${i.duplicates.join(', ')}`);
  return m;
}

export const isLabField = (r: Region) => r.kind === 'counting' && r.field_index != null;
