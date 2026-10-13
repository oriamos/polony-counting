import type { CompareRow, ImportMode, ReferenceGrid, ReferenceInfo } from '../api/types';

export type Poly = [number, number][];

/** Shift a polygon by (dx, dy) image pixels. */
export const shiftPolygon = (p: Poly, dx: number, dy: number): Poly => p.map(([x, y]) => [x + dx, y + dy]);

export interface PreviewCell {
  field: number;
  polygon: Poly;
  /** the researcher has at least one point in this cell (on the chosen plane) */
  counted: boolean;
  /** cell will be created by the import */
  included: boolean;
}

/** Number of reference points of `plane` per field, from reference_counts (plane null = any plane). */
export function countedFields(ref: ReferenceInfo, plane: number | null): Set<number> {
  const out = new Set<number>();
  for (const [k, v] of Object.entries(ref.reference_counts ?? {})) {
    const n = plane == null ? Math.max(v.plane0 ?? 0, v.plane1 ?? 0) : plane === 0 ? v.plane0 ?? 0 : plane === 1 ? v.plane1 ?? 0 : 0;
    if (n > 0) out.add(Number(k));
  }
  return out;
}

/** The grid cells as they would be imported, shifted by (dx, dy). Pure; used for the live preview. */
export function previewImport(grid: ReferenceGrid, ref: ReferenceInfo, mode: ImportMode, plane: number | null, dx: number, dy: number): PreviewCell[] {
  const counted = countedFields(ref, plane);
  return grid.cells.map((c) => ({
    field: c.field,
    polygon: shiftPolygon(c.polygon, dx, dy),
    counted: counted.has(c.field),
    included: mode === 'all' || counted.has(c.field),
  }));
}

/** Points of a plane (null = all planes). */
export const pointsOfPlane = (ref: ReferenceInfo, plane: number | null) => (plane == null ? ref.points : ref.points.filter((p) => p.plane === plane));

/** Neutral (non-judgemental) sign class for a delta: no good/bad colours. */
export const deltaClass = (d: number | null): 'dpos' | 'dneg' | 'dzero' | 'dnone' => (d == null ? 'dnone' : d > 0 ? 'dpos' : d < 0 ? 'dneg' : 'dzero');

export const fmtDelta = (d: number | null) => (d == null ? '–' : d > 0 ? `+${d}` : String(d));

/** Totals over the examined (eligible) rows only; unknown/conflict rows (reference null) never enter. null when none is eligible. */
export function totals(rows: CompareRow[]) {
  let r = 0,
    a = 0,
    n = 0;
  for (const x of rows) {
    if (x.reference_count == null || x.examined === false) continue;
    r += x.reference_count;
    a += x.app_count ?? 0;
    n++;
  }
  return n ? { reference: r, app: a, n } : null;
}

export const STATUS_LABEL: Record<CompareRow['status'], string> = {
  positive_marked: 'examined (marks)',
  verified_zero: 'examined (counted zero)',
  unknown: 'unknown - not examined on this plane, excluded',
  conflict: 'conflict - evidence contradicts the marks, excluded',
};

export const COUNT_TYPE_LABEL = {
  red_positive: 'red-positive',
  green_positive: 'green-positive',
  red_only: 'red-only',
  green_only: 'green-only',
  yellow: 'yellow',
} as const;
