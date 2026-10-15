import { describe, expect, it } from 'vitest';
import {
  cellAt,
  coverageState,
  defaultGrid,
  describeIssues,
  exclusionCoverage,
  fieldIssues,
  fromApiGrid,
  generateGrid,
  mulberry32,
  randomShift,
  storedGridCells,
  toApiGrid,
  type GridSpec,
} from './labgrid';
import type { Analysis } from '../api/types';
import { rectPolygon } from './geometry';

const base: GridSpec = { rows: 2, cols: 3, cellW: 10, cellH: 20, originX: 5, originY: 7, order: 'row-major', corner: 'tl', firstIndex: 1 };
const at = (cells: ReturnType<typeof generateGrid>) => cells.map((c) => `${c.index}:${c.row},${c.col}`).join(' ');

describe('generateGrid', () => {
  it('row-major from top-left', () => {
    expect(at(generateGrid(base))).toBe('1:0,0 2:0,1 3:0,2 4:1,0 5:1,1 6:1,2');
  });
  it('column-major from top-left', () => {
    expect(at(generateGrid({ ...base, order: 'col-major' }))).toBe('1:0,0 2:1,0 3:0,1 4:1,1 5:0,2 6:1,2');
  });
  it('start corner bottom-right, row-major', () => {
    expect(at(generateGrid({ ...base, corner: 'br' }))).toBe('1:1,2 2:1,1 3:1,0 4:0,2 5:0,1 6:0,0');
  });
  it('first index offset and geometry', () => {
    const c = generateGrid({ ...base, firstIndex: 10 });
    expect(c[0].index).toBe(10);
    expect(c[4].polygon).toEqual(rectPolygon(15, 27, 25, 47));
  });
});

describe('defaultGrid', () => {
  it('4x7 of 316px centred in a large landscape image', () => {
    const g = defaultGrid(2400, 1600);
    expect([g.rows, g.cols, g.cellW]).toEqual([4, 7, 316]);
    expect(g.originX).toBe(Math.round((2400 - 7 * 316) / 2));
  });
  it('shrinks cells to fit small images', () => {
    const g = defaultGrid(700, 400);
    expect(g.cols * g.cellW).toBeLessThanOrEqual(700);
    expect(g.rows * g.cellH).toBeLessThanOrEqual(400);
  });
});

describe('exclusion coverage and issues', () => {
  const cell = rectPolygon(0, 0, 100, 100);
  it('classifies none / partial / full', () => {
    expect(coverageState(exclusionCoverage(cell, [rectPolygon(200, 200, 300, 300)]))).toBe('none');
    expect(coverageState(exclusionCoverage(cell, [rectPolygon(0, 0, 50, 100)]))).toBe('partial');
    expect(coverageState(exclusionCoverage(cell, [rectPolygon(-5, -5, 105, 105)]))).toBe('full');
  });
  it('flags >24, duplicates, invalid', () => {
    const i = fieldIssues([1, 2, 2, 25, null, 0]);
    expect(i.duplicates).toEqual([2]);
    expect(i.tooHigh).toEqual([25]);
    expect(i.invalid).toEqual([0]);
    expect(describeIssues(i).length).toBe(3);
  });
});

describe('cellAt (grid-pick hit test)', () => {
  it('finds the cell and its field number', () => {
    expect(cellAt(base, 5, 7)?.index).toBe(1); // top-left corner belongs to the first cell
    expect(cellAt(base, 24.9, 26.9)?.index).toBe(2);
    expect(cellAt(base, 25, 27)?.index).toBe(6); // [x0, x1) x [y0, y1): the far edge belongs to the next cell
    expect(cellAt({ ...base, corner: 'br' }, 6, 8)?.index).toBe(6);
  });
  it('null outside the grid', () => {
    expect(cellAt(base, 4.9, 10)).toBeNull();
    expect(cellAt(base, 35, 10)).toBeNull();
    expect(cellAt(base, 10, 47)).toBeNull();
  });
  it('follows the shift', () => {
    const g = { ...base, shiftDx: 100, shiftDy: 50 };
    expect(cellAt(g, 5, 7)).toBeNull();
    expect(cellAt(g, 106, 58)?.index).toBe(1);
    expect(generateGrid(g)[0].polygon).toEqual(rectPolygon(105, 57, 115, 77));
  });
});

describe('randomShift', () => {
  // same numbers as tests/app/test_grid.py (the server is authoritative; both must agree bit for bit)
  it('mulberry32 matches the Python implementation', () => {
    const r = mulberry32(42);
    expect([r(), r(), r()].map((v) => v.toFixed(10))).toEqual(['0.6011037519', '0.4482905590', '0.8524657935']);
  });
  const g: GridSpec = { rows: 2, cols: 3, cellW: 40, cellH: 40, originX: 10, originY: 10, order: 'row-major', corner: 'tl', firstIndex: 1 };
  it('keeps the grid inside the image, whole pixels, and size unchanged', () => {
    const seen = new Set<string>();
    for (let s = 0; s < 300; s++) {
      const { dx, dy } = randomShift(g, 160, 120, mulberry32(s));
      seen.add(`${dx},${dy}`);
      expect(Number.isInteger(dx) && Number.isInteger(dy)).toBe(true);
      const cells = generateGrid({ ...g, shiftDx: dx, shiftDy: dy });
      expect(cells.length).toBe(6);
      for (const c of cells) for (const [x, y] of c.polygon) {
        expect(x).toBeGreaterThanOrEqual(-0.5);
        expect(x).toBeLessThanOrEqual(159.5);
        expect(y).toBeGreaterThanOrEqual(-0.5);
        expect(y).toBeLessThanOrEqual(119.5);
      }
    }
    expect(seen.size).toBeGreaterThan(50);
  });
  it('same seed, same shift; no slack when the grid is larger than the image', () => {
    expect(randomShift(g, 160, 120, mulberry32(1234))).toEqual(randomShift(g, 160, 120, mulberry32(1234)));
    expect(randomShift(g, 160, 120, mulberry32(1234))).toEqual({ dx: -8, dy: 18 }); // = server (test_grid.py)
    for (let s = 0; s < 20; s++) expect(randomShift({ ...g, cols: 5 }, 160, 120, mulberry32(s)).dx).toBe(0);
  });
});

describe('stored grid', () => {
  it('api round trip and on/off/missing state', () => {
    const api = toApiGrid({ ...base, seed: 7, shiftDx: 3, shiftDy: -2 });
    expect(api).toMatchObject({ cell_w: 10, cell_h: 20, origin_x: 5, first_index: 1, seed: 7, shift_dx: 3, shift_dy: -2 });
    expect(fromApiGrid(api)).toEqual({ ...base, seed: 7, shiftDx: 3, shiftDy: -2 });
    const a = {
      regions: [{ id: 'r1', kind: 'counting', name: 'Field-1', field_index: 1, polygon: [] }],
      slide: { grid: api, skipped_fields: [{ field_index: 2, name: 'Field-2', polygon: [] }] },
    } as unknown as Analysis;
    expect(storedGridCells(a).map((c) => c.state)).toEqual(['on', 'off', 'missing', 'missing', 'missing', 'missing']);
  });
});
