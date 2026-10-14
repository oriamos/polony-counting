import { describe, expect, it } from 'vitest';
import type { Analysis, Region } from '../api/types';
import { defaultLabel, dirOf, gridFields, selectionProblems } from './saveslide';

const reg = (i: number): Region => ({
  id: `r${i}`,
  kind: 'counting',
  name: `Field-${i}`,
  field_index: i,
  polygon: [[0, 0], [1, 0], [1, 1], [0, 1]],
  approved: false,
  approved_at: null,
  approval_fingerprint: null,
  inspected: false,
});

describe('defaultLabel', () => {
  it('matches the server wording', () => {
    expect(defaultLabel(['red'])).toBe('red, not yellow');
    expect(defaultLabel(['yellow'])).toBe('yellow');
    expect(defaultLabel(['red', 'yellow'])).toBe('red and yellow');
    expect(defaultLabel(['green'])).toBe('green, not yellow');
    expect(defaultLabel(['green', 'red'])).toBe('red and green, not yellow');
    expect(defaultLabel(['yellow', 'green', 'red'])).toBe('red, green and yellow');
    expect(defaultLabel([])).toBe('');
  });
});

describe('gridFields', () => {
  it('merges counted and switched-off fields in field order', () => {
    const a = { regions: [reg(3), reg(1)], slide: { skipped_fields: [{ field_index: 2, name: 'Field-2', polygon: [[0, 0], [1, 0], [1, 1], [0, 1]] }] } } as unknown as Analysis;
    expect(gridFields(a).map((f) => [f.field, f.counted])).toEqual([[1, true], [2, false], [3, true]]);
  });
  it('ignores a stale skipped entry that has a region again', () => {
    const a = { regions: [reg(2)], slide: { skipped_fields: [{ field_index: 2, name: 'x', polygon: [] }] } } as unknown as Analysis;
    expect(gridFields(a)).toHaveLength(1);
  });
});

describe('selectionProblems', () => {
  it('flags empty, no class and duplicate labels', () => {
    expect(selectionProblems([])).toHaveLength(1);
    expect(selectionProblems([{ classes: [] }])).toHaveLength(1);
    expect(selectionProblems([{ classes: ['red'] }, { classes: ['red'] }])).toHaveLength(1);
    expect(selectionProblems([{ classes: ['red'] }, { classes: ['yellow'] }])).toEqual([]);
    expect(selectionProblems([{ classes: ['red'], label: 'A' }, { classes: ['red'], label: 'B' }])).toEqual([]);
  });
});

describe('dirOf', () => {
  it('handles both separators', () => {
    expect(dirOf('/a/b/c.tif')).toBe('/a/b');
    expect(dirOf('C:\\lab\\c.tif')).toBe('C:\\lab');
  });
});
