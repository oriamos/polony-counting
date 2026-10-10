import type { ObjectClass } from '../api/types';

/**
 * Class glyphs. Shape (not only colour) distinguishes classes so the overlay stays
 * readable on red / green / yellow signal:
 *   red-only: circle, green-only: diamond, yellow: square, uncertain: dashed circle.
 */
export type GlyphShape = 'circle' | 'diamond' | 'square' | 'dashed-circle';

export interface GlyphSpec {
  shape: GlyphShape;
  color: string; // thin class-coloured inner line
  dashed: boolean;
  label: string;
}

export const CLASS_ORDER: ObjectClass[] = ['red', 'green', 'yellow', 'uncertain'];

export const CLASS_LABEL: Record<ObjectClass, string> = {
  red: 'red-only',
  green: 'green-only',
  yellow: 'yellow (dual)',
  uncertain: 'uncertain',
};

export const CLASS_KEY: Record<ObjectClass, string> = { red: 'R', green: 'G', yellow: 'Y', uncertain: 'U' };

const SPECS: Record<ObjectClass, GlyphSpec> = {
  red: { shape: 'circle', color: '#ff3b47', dashed: false, label: CLASS_LABEL.red },
  green: { shape: 'diamond', color: '#22d36b', dashed: false, label: CLASS_LABEL.green },
  yellow: { shape: 'square', color: '#ffd400', dashed: false, label: CLASS_LABEL.yellow },
  uncertain: { shape: 'dashed-circle', color: '#c58bff', dashed: true, label: CLASS_LABEL.uncertain },
};

export function glyphFor(cls: ObjectClass): GlyphSpec {
  return SPECS[cls];
}

/** SVG path for polygonal glyphs centred at (cx,cy) with "radius" r (centre-to-edge for square, to vertex for diamond). */
export function glyphPath(shape: GlyphShape, cx: number, cy: number, r: number): string | null {
  switch (shape) {
    case 'diamond':
      return `M${cx} ${cy - r}L${cx + r} ${cy}L${cx} ${cy + r}L${cx - r} ${cy}Z`;
    case 'square':
      return `M${cx - r} ${cy - r}L${cx + r} ${cy - r}L${cx + r} ${cy + r}L${cx - r} ${cy + r}Z`;
    default:
      return null; // circles use <circle>
  }
}

/** Parse a path produced by glyphPath into its vertices (used by tests / hit-testing). */
export function glyphVertices(shape: GlyphShape, cx: number, cy: number, r: number): [number, number][] {
  if (shape === 'diamond')
    return [
      [cx, cy - r],
      [cx + r, cy],
      [cx, cy + r],
      [cx - r, cy],
    ];
  if (shape === 'square')
    return [
      [cx - r, cy - r],
      [cx + r, cy - r],
      [cx + r, cy + r],
      [cx - r, cy + r],
    ];
  return [];
}

/** Ring radius in IMAGE px so the glyph is a constant-ish screen size, floored by spot size. */
export function ringRadiusImage(scale: number, spotRadiusPx = 4, compact = false): number {
  const screenR = compact ? Math.min(8, Math.max(3.5, spotRadiusPx * scale * 1.3)) : Math.min(14, Math.max(7, spotRadiusPx * scale * 1.6));
  return screenR / scale;
}

export const REASON_TEXT: Record<string, string> = {
  weak_signal: 'Weak signal: peak is low relative to local background noise.',
  ambiguous_match: 'Ambiguous red/green association: more than one plausible partner within tolerance.',
  dense_cluster: 'Dense cluster: neighbouring spots are closer than expected; count may be unreliable.',
  saturated: 'Saturated pixels at the peak: intensity is clipped.',
  suspicious_shape: 'Unusual spot shape (not compact/round).',
  near_edge: 'Near the edge of a counting region or the image.',
  near_threshold: 'Peak is close to the detection threshold; a small parameter change could flip it.',
  bigfish_only: 'Engine disagreement: only Big-FISH proposed this candidate; Spotiflow did not.',
  spotiflow_only: 'Engine disagreement: only Spotiflow proposed this candidate; Big-FISH did not.',
  qc_sample: 'In the random QC sample of engine agreements. Agreement is not ground truth.',
  debris_like: 'Compact, very bright object resembling debris/dust — check carefully',
};

export const REASON_SHORT: Record<string, string> = {
  weak_signal: 'weak signal',
  ambiguous_match: 'ambiguous match',
  dense_cluster: 'dense cluster',
  saturated: 'saturated',
  suspicious_shape: 'odd shape',
  near_edge: 'near edge',
  near_threshold: 'near threshold',
  bigfish_only: 'Big-FISH only',
  spotiflow_only: 'Spotiflow only',
  qc_sample: 'QC sample',
  debris_like: 'debris-like',
};
