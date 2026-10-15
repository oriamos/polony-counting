import { describe, expect, it } from 'vitest';
import type { CompareRow, ReferenceInfo } from '../api/types';
import { totals, countedFields, deltaClass, fmtDelta, pointsOfPlane, previewImport, shiftPolygon } from './reference';

const ref: ReferenceInfo = {
  available: true,
  points: [
    { x: 1, y: 1, counter: 1, plane: 0, field: 1 },
    { x: 2, y: 2, counter: 2, plane: 1, field: 2 },
  ],
  grid: {
    source: 'overlay',
    origin: [10, 20],
    cell: [100, 50],
    cols: 2,
    rows: 1,
    cells: [
      { field: 1, polygon: [[10, 20], [110, 20], [110, 70], [10, 70]] },
      { field: 2, polygon: [[110, 20], [210, 20], [210, 70], [110, 70]] },
    ],
  },
  reference_counts: { '1': { plane0: 3, plane1: 0 }, '2': { plane0: 0, plane1: 4 } },
  planes: [{ page: 0, point_count: 1 }, { page: 1, point_count: 1 }],
  warnings: [],
};

describe('reference grid shift', () => {
  it('shifts every vertex by dx, dy', () => {
    expect(shiftPolygon([[0, 0], [2, 3]], 5, -1)).toEqual([[5, -1], [7, 2]]);
  });
  it('previews shifted cells without mutating the source', () => {
    const p = previewImport(ref.grid!, ref, 'all', null, 7, -3);
    expect(p[0].polygon[0]).toEqual([17, 17]);
    expect(ref.grid!.cells[0].polygon[0]).toEqual([10, 20]);
    expect(p.every((c) => c.included)).toBe(true);
  });
  it('counted mode includes only fields with points on the chosen plane', () => {
    expect(previewImport(ref.grid!, ref, 'counted', 0, 0, 0).map((c) => c.included)).toEqual([true, false]);
    expect(previewImport(ref.grid!, ref, 'counted', 1, 0, 0).map((c) => c.included)).toEqual([false, true]);
    expect([...countedFields(ref, null)].sort()).toEqual([1, 2]);
  });
  it('filters points by plane', () => {
    expect(pointsOfPlane(ref, 1)).toHaveLength(1);
    expect(pointsOfPlane(ref, null)).toHaveLength(2);
  });
  it('delta is sign-only, no quality judgement', () => {
    expect(deltaClass(3)).toBe('dpos');
    expect(deltaClass(-1)).toBe('dneg');
    expect(deltaClass(0)).toBe('dzero');
    expect(deltaClass(null)).toBe('dnone');
    expect(fmtDelta(3)).toBe('+3');
    expect(fmtDelta(-2)).toBe('-2');
  });
});

describe('totals over compare rows', () => {
  const row = (field: number, ref: number | null, app: number, status: CompareRow['status']): CompareRow =>
    ({ field, reference_count: ref, app_count: app, delta: ref == null ? null : app - ref, status, examined: ref != null, examination_source: '' });
  it('ignores unknown rows (their app count never enters) and is null without eligible rows', () => {
    expect(totals([row(1, 3, 2, 'positive_marked'), row(2, 0, 1, 'verified_zero'), row(3, null, 50, 'unknown')])).toEqual({ reference: 3, app: 3, n: 2 });
    expect(totals([row(3, null, 50, 'unknown')])).toBeNull();
  });
});

describe('marks without a valid plane', () => {
  const r2: ReferenceInfo = {
    ...ref,
    points: [...ref.points, { x: 3, y: 3, counter: 1, plane: null, field: 2 }],
    reference_counts: { '1': { plane0: 3, plane1: 0 }, '2': { plane0: 0, plane1: 0, unplaned: 1 } },
  };
  it('are evidence for no plane', () => {
    expect(pointsOfPlane(r2, 0)).toHaveLength(1);
    expect(pointsOfPlane(r2, 1)).toHaveLength(1);
    expect([...countedFields(r2, null)]).toEqual([1]);
    expect([...countedFields(r2, 1)]).toEqual([]);
  });
});
