import { describe, expect, it } from 'vitest';
import { coverageState, defaultGrid, describeIssues, exclusionCoverage, fieldIssues, generateGrid, type GridSpec } from './labgrid';
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
