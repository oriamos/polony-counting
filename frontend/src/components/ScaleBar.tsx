import type { View } from '../lib/geometry';

const NICE = [1, 2, 5];
export function niceLength(target: number): number {
  const exp = Math.floor(Math.log10(target));
  const base = Math.pow(10, exp);
  let best = base;
  for (const m of NICE) if (m * base <= target) best = m * base;
  return best;
}

/** Scale bar in image px, or in µm when the pixel size is known. */
export function ScaleBar({ view, pixelSizeUm }: { view: View; pixelSizeUm: number | null }) {
  const targetPx = 110; // screen px
  const unitPerPx = pixelSizeUm ?? 1;
  const len = niceLength((targetPx / view.scale) * unitPerPx);
  const width = (len / unitPerPx) * view.scale;
  const label = pixelSizeUm ? (len >= 1000 ? `${len / 1000} mm` : `${len} µm`) : `${len} px`;
  return (
    <div className="scalebar" aria-label="scale bar" data-testid="scalebar">
      <div className="scalebar-label">{label}</div>
      <div className="scalebar-line" style={{ width }} />
    </div>
  );
}
