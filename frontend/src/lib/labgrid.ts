import type { Region } from '../api/types';
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
    polygon: rectPolygon(g.originX + col * g.cellW, g.originY + row * g.cellH, g.originX + (col + 1) * g.cellW, g.originY + (row + 1) * g.cellH),
  }));
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
