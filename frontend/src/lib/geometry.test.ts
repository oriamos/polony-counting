import { describe, expect, it } from 'vitest';
import {
  centerOn,
  fitBox,
  fitView,
  imageToScreen,
  panBy,
  pointInPolygon,
  polygonArea,
  rectPolygon,
  screenToImage,
  zoomAbout,
  MAX_SCALE,
} from './geometry';

const close = (a: number, b: number, eps = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe('screen <-> image transform', () => {
  it('round-trips arbitrary points', () => {
    const v = { scale: 3.7, tx: 123.4, ty: -56.7 };
    for (const [x, y] of [
      [0, 0],
      [100.5, 200.25],
      [-3, 9999],
    ]) {
      const s = imageToScreen(v, x, y);
      const p = screenToImage(v, s.x, s.y);
      close(p.x, x);
      close(p.y, y);
    }
  });

  it('pixel (0,0) centre maps to (tx,ty)', () => {
    const v = { scale: 8, tx: 40, ty: 30 };
    expect(imageToScreen(v, 0, 0)).toEqual({ x: 40, y: 30 });
    // pixel (0,0) spans image -0.5..0.5 => screen tx-4 .. tx+4
    expect(imageToScreen(v, -0.5, -0.5)).toEqual({ x: 36, y: 26 });
  });
});

describe('zoomAbout', () => {
  it('keeps the image point under the cursor fixed', () => {
    let v = { scale: 1, tx: 10, ty: 20 };
    const cursor = { x: 300, y: 210 };
    const before = screenToImage(v, cursor.x, cursor.y);
    for (const f of [1.25, 1.25, 0.5, 4, 0.9]) {
      v = zoomAbout(v, cursor.x, cursor.y, f);
      const after = screenToImage(v, cursor.x, cursor.y);
      close(after.x, before.x, 1e-7);
      close(after.y, before.y, 1e-7);
    }
  });
  it('clamps scale and still preserves the anchor', () => {
    const v0 = { scale: 40, tx: 5, ty: 5 };
    const v = zoomAbout(v0, 100, 100, 10);
    expect(v.scale).toBe(MAX_SCALE);
    const a = screenToImage(v0, 100, 100);
    const b = screenToImage(v, 100, 100);
    close(a.x, b.x, 1e-7);
    close(a.y, b.y, 1e-7);
  });
  it('pan then zoom composes', () => {
    const v = zoomAbout(panBy({ scale: 2, tx: 0, ty: 0 }, 50, -20), 0, 0, 2);
    expect(v.scale).toBe(4);
    expect(v.tx).toBe(100);
    expect(v.ty).toBe(-40);
  });
});

describe('fit / center', () => {
  it('fitView centres image extent in the viewport', () => {
    const v = fitView(1000, 500, 800, 600, 0);
    close(v.scale, 0.8);
    // left edge of image (x=-0.5) lands at (800-800)/2 = 0
    close(imageToScreen(v, -0.5, -0.5).x, 0);
    close(imageToScreen(v, 999.5, 499.5).x, 800);
    close(imageToScreen(v, -0.5, -0.5).y, 100);
  });
  it('centerOn puts the point at the viewport centre', () => {
    const v = centerOn(120, 80, 6, 900, 700);
    const s = imageToScreen(v, 120, 80);
    close(s.x, 450);
    close(s.y, 350);
  });
  it('fitBox contains the box', () => {
    const v = fitBox(100, 100, 200, 150, 800, 600, 20);
    const a = imageToScreen(v, 100, 100);
    const b = imageToScreen(v, 200, 150);
    expect(a.x).toBeGreaterThanOrEqual(20 - 1e-6);
    expect(b.x).toBeLessThanOrEqual(780 + 1e-6);
  });
});

describe('polygon helpers', () => {
  it('area and containment of a rectangle', () => {
    const r = rectPolygon(10, 20, 4, 8);
    expect(polygonArea(r)).toBe(6 * 12);
    expect(pointInPolygon(5, 10, r)).toBe(true);
    expect(pointInPolygon(11, 10, r)).toBe(false);
  });
});
