/**
 * View transform between image pixel space and screen (viewport) space.
 *
 * Image coordinates follow the schema: pixel (0,0) has its CENTRE at (0,0), so
 * the pixel occupies [-0.5, 0.5]. The "world" element is CSS-transformed with
 *   screen = (tx, ty) + scale * image
 * and the <img> is offset by (-0.5, -0.5) image px inside it. Overlays are drawn
 * in the same world element, so alignment is exact by construction.
 */
export interface View {
  scale: number; // screen px per image px
  tx: number; // screen x of image coordinate x=0 (pixel centre of column 0)
  ty: number;
}

export interface Pt {
  x: number;
  y: number;
}

export const MIN_SCALE = 0.02;
export const MAX_SCALE = 64;

export const clampScale = (s: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

export function screenToImage(v: View, sx: number, sy: number): Pt {
  return { x: (sx - v.tx) / v.scale, y: (sy - v.ty) / v.scale };
}

export function imageToScreen(v: View, x: number, y: number): Pt {
  return { x: v.tx + x * v.scale, y: v.ty + y * v.scale };
}

/** Zoom by `factor` keeping the image point under screen point (sx,sy) fixed. */
export function zoomAbout(v: View, sx: number, sy: number, factor: number): View {
  const scale = clampScale(v.scale * factor);
  const k = scale / v.scale;
  return { scale, tx: sx - (sx - v.tx) * k, ty: sy - (sy - v.ty) * k };
}

export function panBy(v: View, dx: number, dy: number): View {
  return { ...v, tx: v.tx + dx, ty: v.ty + dy };
}

/** Fit the whole image (extent -0.5..W-0.5) into a viewport with padding. */
export function fitView(imgW: number, imgH: number, vpW: number, vpH: number, pad = 16): View {
  const aw = Math.max(1, vpW - 2 * pad);
  const ah = Math.max(1, vpH - 2 * pad);
  const scale = clampScale(Math.min(aw / imgW, ah / imgH));
  return {
    scale,
    tx: (vpW - imgW * scale) / 2 + 0.5 * scale,
    ty: (vpH - imgH * scale) / 2 + 0.5 * scale,
  };
}

/** Centre the viewport on image point (x,y) at the given scale. */
export function centerOn(x: number, y: number, scale: number, vpW: number, vpH: number): View {
  const s = clampScale(scale);
  return { scale: s, tx: vpW / 2 - x * s, ty: vpH / 2 - y * s };
}

/** Fit a bounding box (image px) into the viewport. */
export function fitBox(x0: number, y0: number, x1: number, y1: number, vpW: number, vpH: number, pad = 48): View {
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const s = clampScale(Math.min((vpW - 2 * pad) / w, (vpH - 2 * pad) / h));
  return centerOn((x0 + x1) / 2, (y0 + y1) / 2, s, vpW, vpH);
}

/** Visible image-space rectangle for a viewport. */
export function visibleRect(v: View, vpW: number, vpH: number) {
  const a = screenToImage(v, 0, 0);
  const b = screenToImage(v, vpW, vpH);
  return { x0: a.x, y0: a.y, x1: b.x, y1: b.y };
}

// ---------------------------------------------------------------- polygons
export type Poly = [number, number][];

export function polygonArea(p: Poly): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i];
    const [x2, y2] = p[(i + 1) % p.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

export function polygonCentroid(p: Poly): Pt {
  if (!p.length) return { x: 0, y: 0 };
  let sx = 0,
    sy = 0;
  for (const [x, y] of p) {
    sx += x;
    sy += y;
  }
  return { x: sx / p.length, y: sy / p.length };
}

export function polygonBounds(p: Poly) {
  const xs = p.map((q) => q[0]);
  const ys = p.map((q) => q[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

export function pointInPolygon(x: number, y: number, p: Poly): boolean {
  let inside = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [xi, yi] = p[i];
    const [xj, yj] = p[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function rectPolygon(x0: number, y0: number, x1: number, y1: number): Poly {
  const a = Math.min(x0, x1),
    b = Math.min(y0, y1),
    c = Math.max(x0, x1),
    d = Math.max(y0, y1);
  return [
    [a, b],
    [c, b],
    [c, d],
    [a, d],
  ];
}

/** Distance from point to segment, and the projection parameter. */
export function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax,
    dy = by - ay;
  const l2 = dx * dx + dy * dy;
  const t = l2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
  const cx = ax + t * dx,
    cy = ay + t * dy;
  return { d: Math.hypot(px - cx, py - cy), t, x: cx, y: cy };
}

export const round2 = (n: number) => Math.round(n * 100) / 100;
