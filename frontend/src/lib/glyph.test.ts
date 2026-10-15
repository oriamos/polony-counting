import { describe, expect, it } from 'vitest';
import { CLASS_ORDER, glyphFor, glyphPath, glyphVertices, ringRadiusImage } from './glyph';

describe('class glyphs', () => {
  it('maps each class to a distinct shape', () => {
    expect(glyphFor('red').shape).toBe('circle');
    expect(glyphFor('green').shape).toBe('diamond');
    expect(glyphFor('yellow').shape).toBe('square');
    expect(glyphFor('uncertain').shape).toBe('dashed-circle');
    expect(new Set(CLASS_ORDER.map((c) => glyphFor(c).shape)).size).toBe(4);
  });
  it('only uncertain is dashed', () => {
    expect(CLASS_ORDER.filter((c) => glyphFor(c).dashed)).toEqual(['uncertain']);
  });
  it('polygonal glyphs are centred on the point', () => {
    for (const shape of ['diamond', 'square'] as const) {
      const v = glyphVertices(shape, 12.5, -3, 4);
      expect(v).toHaveLength(4);
      const cx = v.reduce((s, p) => s + p[0], 0) / 4;
      const cy = v.reduce((s, p) => s + p[1], 0) / 4;
      expect(cx).toBeCloseTo(12.5);
      expect(cy).toBeCloseTo(-3);
      expect(glyphPath(shape, 12.5, -3, 4)).toMatch(/^M.*Z$/);
    }
    expect(glyphPath('circle', 0, 0, 3)).toBeNull();
  });
  it('ring has bounded screen size', () => {
    for (const s of [0.05, 0.5, 1, 4, 32]) {
      const px = ringRadiusImage(s) * s;
      expect(px).toBeGreaterThanOrEqual(7 - 1e-9);
      expect(px).toBeLessThanOrEqual(14 + 1e-9);
    }
  });
});
