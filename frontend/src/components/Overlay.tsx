import { memo } from 'react';
import type { DetectedObject, EmbeddedRoi, Region } from '../api/types';
import { glyphFor, glyphPath, ringRadiusImage } from '../lib/glyph';
import { polygonBounds } from '../lib/geometry';

/** Region status shown in badges and styling. */
export type RegionStatus = 'unreviewed' | 'inspected' | 'approved' | 'exclusion';
export const regionStatus = (r: Region): RegionStatus =>
  r.kind === 'exclusion' ? 'exclusion' : r.approved ? 'approved' : r.inspected ? 'inspected' : 'unreviewed';

export const STATUS_COLOR: Record<RegionStatus, string> = {
  unreviewed: '#ffb020',
  inspected: '#4da3ff',
  approved: '#36d399',
  exclusion: '#ff4fb0',
};

const DARK = 'rgba(0,0,0,0.88)';
const LIGHT = '#ffffff';

export const shortId = (id: string) => (id.length > 7 ? id.slice(-6) : id);

interface GlyphProps {
  o: { x: number; y: number; cls: DetectedObject['cls'] };
  scale: number;
  selected?: boolean;
  reviewed?: boolean;
  showReview?: boolean;
  faded?: boolean;
  preview?: boolean;
}

/**
 * One object marker. All geometry is in IMAGE pixel units; widths are divided by
 * `scale` so strokes stay constant on screen. Double outline (dark + light) + thin
 * class-coloured line keeps it legible over red, green and yellow signal.
 */
export const Glyph = memo(function Glyph({ o, scale, selected, reviewed, showReview, faded, preview }: GlyphProps) {
  const g = glyphFor(o.cls);
  const r = ringRadiusImage(scale);
  const k = 1 / scale;
  const dash = g.dashed ? `${3.2 * k} ${2.4 * k}` : undefined;
  const path = glyphPath(g.shape, o.x, o.y, g.shape === 'square' ? r * 0.86 : r * 1.12);
  const shape = (stroke: string, w: number, dashArray?: string) =>
    path ? (
      <path d={path} fill="none" stroke={stroke} strokeWidth={w * k} strokeLinejoin="miter" strokeDasharray={dashArray} />
    ) : (
      <circle cx={o.x} cy={o.y} r={r} fill="none" stroke={stroke} strokeWidth={w * k} strokeDasharray={dashArray} />
    );
  return (
    <g opacity={faded ? 0.55 : 1} data-cls={o.cls}>
      {selected && (
        <>
          <circle cx={o.x} cy={o.y} r={r + 5 * k} fill="none" stroke={DARK} strokeWidth={5 * k} />
          <circle cx={o.x} cy={o.y} r={r + 5 * k} fill="none" stroke="#00e5ff" strokeWidth={2.4 * k} />
        </>
      )}
      {shape(DARK, 4.6, dash)}
      {shape(LIGHT, 2.6, dash)}
      {shape(g.color, 1.1, dash)}
      {preview && <circle cx={o.x} cy={o.y} r={r + 3 * k} fill="none" stroke="#ffb020" strokeWidth={1.2 * k} strokeDasharray={`${1.5 * k} ${2 * k}`} />}
      {showReview &&
        (reviewed ? (
          <path
            d={`M${o.x + r * 0.55} ${o.y - r * 1.05}l${2 * k} ${2.2 * k}l${4.2 * k} ${-4.4 * k}`}
            fill="none"
            stroke="#36d399"
            strokeWidth={2 * k}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ filter: 'drop-shadow(0 0 0.5px #000)' }}
          />
        ) : (
          <circle cx={o.x + r * 0.85} cy={o.y - r * 0.85} r={2.4 * k} fill="#ffb020" stroke={DARK} strokeWidth={1 * k} />
        ))}
    </g>
  );
});

export const HaloText = memo(function HaloText({
  x,
  y,
  scale,
  text,
  size = 11,
  fill = '#fff',
  anchor = 'start',
}: {
  x: number;
  y: number;
  scale: number;
  text: string;
  size?: number;
  fill?: string;
  anchor?: 'start' | 'middle' | 'end';
}) {
  const k = 1 / scale;
  return (
    <text
      x={x}
      y={y}
      fontSize={size * k}
      textAnchor={anchor}
      fill={fill}
      stroke="rgba(0,0,0,0.92)"
      strokeWidth={3.4 * k}
      strokeLinejoin="round"
      paintOrder="stroke"
      fontFamily="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
      fontWeight={600}
      style={{ userSelect: 'none' }}
    >
      {text}
    </text>
  );
});

interface ObjectsProps {
  objects: DetectedObject[];
  scale: number;
  selection: string[];
  showLabels: 'auto' | 'on' | 'off';
  showReview: boolean;
  preview?: boolean;
  dragOverride?: { id: string; x: number; y: number } | null;
}

export const ObjectsLayer = memo(function ObjectsLayer({ objects, scale, selection, showLabels, showReview, preview, dragOverride }: ObjectsProps) {
  const sel = new Set(selection);
  const labels = showLabels === 'on' || (showLabels === 'auto' && scale >= 3 && objects.length < 1500);
  const r = ringRadiusImage(scale);
  const k = 1 / scale;
  return (
    <g>
      {objects.map((o) => {
        const pos = dragOverride && dragOverride.id === o.id ? { ...o, x: dragOverride.x, y: dragOverride.y } : o;
        const selected = sel.has(o.id);
        return (
          <g key={o.id}>
            <Glyph o={pos} scale={scale} selected={selected} reviewed={o.reviewed} showReview={showReview && !preview} preview={preview} />
            {(labels || selected) && <HaloText x={pos.x + r + 4 * k} y={pos.y + r + 9 * k} scale={scale} text={shortId(o.id)} />}
          </g>
        );
      })}
    </g>
  );
});

interface RegionsProps {
  regions: Region[];
  scale: number;
  selectedId: string | null;
  draft?: Region | null;
}

export const RegionsLayer = memo(function RegionsLayer({ regions, scale, selectedId, draft }: RegionsProps) {
  const k = 1 / scale;
  return (
    <g>
      {regions.map((r0) => {
        const r = draft && draft.id === r0.id ? draft : r0;
        const st = regionStatus(r);
        const color = STATUS_COLOR[st];
        const pts = r.polygon.map((p) => p.join(',')).join(' ');
        const b = polygonBounds(r.polygon);
        const sel = r.id === selectedId;
        const dash = r.kind === 'exclusion' ? `${7 * k} ${4 * k}` : undefined;
        return (
          <g key={r.id} data-region={r.id}>
            <polygon points={pts} fill={r.kind === 'exclusion' ? 'rgba(255,79,176,0.16)' : 'rgba(77,163,255,0.05)'} stroke="none" />
            <polygon points={pts} fill="none" stroke={DARK} strokeWidth={(sel ? 5.4 : 4) * k} strokeLinejoin="round" strokeDasharray={dash} />
            <polygon points={pts} fill="none" stroke={color} strokeWidth={(sel ? 2.8 : 1.8) * k} strokeLinejoin="round" strokeDasharray={dash} />
            <HaloText x={b.x0 + 4 * k} y={b.y0 - 6 * k} scale={scale} size={12} fill={color} text={`${r.name} · ${st}`} />
            {sel &&
              r.polygon.map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r={5 * k} fill="#fff" stroke={DARK} strokeWidth={1.6 * k} />
              ))}
          </g>
        );
      })}
    </g>
  );
});

export const SuggestionsLayer = memo(function SuggestionsLayer({ rois, scale }: { rois: EmbeddedRoi[]; scale: number }) {
  const k = 1 / scale;
  return (
    <g>
      {rois.map((r, i) => (
        <g key={i}>
          <rect x={r.left} y={r.top} width={r.right - r.left} height={r.bottom - r.top} fill="none" stroke={DARK} strokeWidth={4 * k} />
          <rect
            x={r.left}
            y={r.top}
            width={r.right - r.left}
            height={r.bottom - r.top}
            fill="none"
            stroke="#d0d7e2"
            strokeWidth={1.8 * k}
            strokeDasharray={`${2 * k} ${4 * k}`}
          />
          <HaloText x={r.left + 4 * k} y={r.bottom + 14 * k} scale={scale} text={`suggested ROI${r.name ? ` "${r.name}"` : ` #${i + 1}`}`} />
        </g>
      ))}
    </g>
  );
});
