import { memo } from 'react';
import { useApp } from '../state/store';
import type { ConsensusStatus, DetectedObject, EmbeddedRoi, RawDetection, Region } from '../api/types';
import { objectStatus, triangleD } from '../lib/consensus';
import { glyphFor, glyphPath, ringRadiusImage } from '../lib/glyph';
import { polygonBounds } from '../lib/geometry';
import { storedGridCells } from '../lib/labgrid';

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
  /** thinner strokes and smaller rings (zoomed out / many spots) */
  compact?: boolean;
}

/**
 * One object marker. All geometry is in IMAGE pixel units; widths are divided by
 * `scale` so strokes stay constant on screen. Double outline (dark + light) + thin
 * class-coloured line keeps it legible over red, green and yellow signal.
 */
export const Glyph = memo(function Glyph({ o, scale, selected, reviewed, showReview, faded, preview, compact }: GlyphProps) {
  const g = glyphFor(o.cls);
  const r = ringRadiusImage(scale, 4, compact);
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
      {shape(DARK, compact ? 2.6 : 4.6, dash)}
      {shape(LIGHT, compact ? 1.3 : 2.6, dash)}
      {shape(g.color, compact ? 0.8 : 1.1, dash)}
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

/** Pattern per engine status (readable without colour): solid ring, long dashes, dots, double ring. */
export const STATUS_STYLE: Record<Exclude<ConsensusStatus, 'human_rejected'>, { tint: string; mark: string; name: string }> = {
  agreement: { tint: '#9ad1ff', mark: '=', name: 'both engines (solid ring)' },
  bigfish_only: { tint: '#5ec8ff', mark: 'B', name: 'Big-FISH only (long-dash ring)' },
  spotiflow_only: { tint: '#ffb86b', mark: 'S', name: 'Spotiflow only (dotted ring)' },
  human_added: { tint: '#e5a8ff', mark: '+', name: 'human added (double ring)' },
};

/** Outer ring + letter badge around an object glyph showing which engine(s) proposed the candidate. */
export const ConsensusMark = memo(function ConsensusMark({ x, y, status, scale, compact }: { x: number; y: number; status: ConsensusStatus; scale: number; compact?: boolean }) {
  if (status === 'human_rejected') return null;
  const st = STATUS_STYLE[status];
  const k = 1 / scale;
  const r = ringRadiusImage(scale, 4, compact) + 4.5 * k;
  const dash = status === 'bigfish_only' ? `${6 * k} ${3.4 * k}` : status === 'spotiflow_only' ? `${1.1 * k} ${2.6 * k}` : undefined;
  const ring = (rr: number) => (
    <>
      <circle cx={x} cy={y} r={rr} fill="none" stroke={DARK} strokeWidth={(compact ? 2.6 : 3.6) * k} strokeDasharray={dash} strokeLinecap={status === 'spotiflow_only' ? 'round' : undefined} />
      <circle cx={x} cy={y} r={rr} fill="none" stroke="#fff" strokeWidth={(compact ? 1 : 1.5) * k} strokeDasharray={dash} strokeLinecap={status === 'spotiflow_only' ? 'round' : undefined} />
    </>
  );
  return (
    <g data-consensus={status} pointerEvents="none">
      {ring(r)}
      {status === 'human_added' && ring(r + 3 * k)}
      {!compact && <HaloText x={x - r - 8 * k} y={y + r + 6 * k} scale={scale} size={9} fill={st.tint} text={st.mark} />}
    </g>
  );
});

/** Human-rejected candidates (deleted by a user): grey dashed circle with a cross. Only drawn on request. */
export const RejectedLayer = memo(function RejectedLayer({ objects, scale }: { objects: DetectedObject[]; scale: number }) {
  const k = 1 / scale;
  const r = 5 * k;
  return (
    <g data-testid="rejected-layer" data-count={objects.length} pointerEvents="none">
      {objects.map((o) => {
        const d = `M${o.x - r * 0.7} ${o.y - r * 0.7}L${o.x + r * 0.7} ${o.y + r * 0.7}M${o.x - r * 0.7} ${o.y + r * 0.7}L${o.x + r * 0.7} ${o.y - r * 0.7}`;
        return (
          <g key={o.id} data-rejected={o.id}>
            <circle cx={o.x} cy={o.y} r={r} fill="none" stroke={DARK} strokeWidth={3.6 * k} strokeDasharray={`${2 * k} ${2 * k}`} />
            <circle cx={o.x} cy={o.y} r={r} fill="none" stroke="#c9ced8" strokeWidth={1.3 * k} strokeDasharray={`${2 * k} ${2 * k}`} />
            <path d={d} stroke={DARK} strokeWidth={3.4 * k} fill="none" />
            <path d={d} stroke="#c9ced8" strokeWidth={1.3 * k} fill="none" />
          </g>
        );
      })}
    </g>
  );
});

export const RAW_COLOR = { bigfish: '#5ec8ff', spotiflow: '#ffb86b' } as const;

/** Raw engine output: Big-FISH = small hollow circle, Spotiflow = small hollow triangle. Excluded (invalid) points are faint. */
export const RawLayer = memo(function RawLayer({ items, engine, scale }: { items: RawDetection[]; engine: 'bigfish' | 'spotiflow'; scale: number }) {
  const k = 1 / scale;
  const r = 3.4 * k;
  const color = RAW_COLOR[engine];
  return (
    <g data-testid={`raw-${engine}`} data-count={items.length} pointerEvents="none">
      {items.map((d) => {
        const op = d.valid ? 1 : 0.5;
        if (engine === 'bigfish')
          return (
            <g key={d.id} opacity={op} data-valid={d.valid ? '1' : '0'}>
              <circle cx={d.x} cy={d.y} r={r} fill="none" stroke={DARK} strokeWidth={3 * k} />
              <circle cx={d.x} cy={d.y} r={r} fill="none" stroke={color} strokeWidth={1.3 * k} strokeDasharray={d.valid ? undefined : `${1.5 * k} ${1.5 * k}`} />
            </g>
          );
        const tri = triangleD(d.x, d.y, r * 1.25);
        return (
          <g key={d.id} opacity={op} data-valid={d.valid ? '1' : '0'}>
            <path d={tri} fill="none" stroke={DARK} strokeWidth={3 * k} strokeLinejoin="round" />
            <path d={tri} fill="none" stroke={color} strokeWidth={1.3 * k} strokeLinejoin="round" strokeDasharray={d.valid ? undefined : `${1.5 * k} ${1.5 * k}`} />
          </g>
        );
      })}
    </g>
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
  consensus?: boolean;
}

export const ObjectsLayer = memo(function ObjectsLayer({ objects, scale, selection, showLabels, showReview, preview, dragOverride, consensus }: ObjectsProps) {
  const sel = new Set(selection);
  const labels = showLabels === 'on' || (showLabels === 'auto' && scale >= 3 && objects.length < 1500);
  const compact = scale < 1 || objects.length > 500;
  const r = ringRadiusImage(scale, 4, compact);
  const k = 1 / scale;
  return (
    <g>
      {objects.map((o) => {
        const pos = dragOverride && dragOverride.id === o.id ? { ...o, x: dragOverride.x, y: dragOverride.y } : o;
        const selected = sel.has(o.id);
        return (
          <g key={o.id}>
            <Glyph o={pos} scale={scale} compact={compact && !selected} selected={selected} reviewed={o.reviewed} showReview={showReview && !preview} preview={preview} />
            {consensus && !preview && (() => {
              const st = objectStatus(o);
              return st ? <ConsensusMark x={pos.x} y={pos.y} status={st} scale={scale} compact={compact && !selected} /> : null;
            })()}
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
        const sel = r.id === selectedId;
        const dash = r.kind === 'exclusion' ? `${7 * k} ${4 * k}` : undefined;
        return (
          <g key={r.id} data-region={r.id}>
            <polygon points={pts} fill={r.kind === 'exclusion' ? 'rgba(255,79,176,0.16)' : 'rgba(77,163,255,0.05)'} stroke="none" />
            <polygon points={pts} fill="none" stroke={DARK} strokeWidth={(sel ? 5.4 : 4) * k} strokeLinejoin="round" strokeDasharray={dash} />
            <polygon points={pts} fill="none" stroke={color} strokeWidth={(sel ? 2.8 : 1.8) * k} strokeLinejoin="round" strokeDasharray={dash} />
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

/**
 * Region names, drawn in a separate layer ABOVE the object rings: placed outside the polygon at its
 * top-left corner (just above the bounding box) with halo text, so labels never sit on spot rings.
 */
export const RegionLabelsLayer = memo(function RegionLabelsLayer({ regions, scale, draft }: { regions: Region[]; scale: number; draft?: Region | null }) {
  const k = 1 / scale;
  return (
    <g pointerEvents="none">
      {regions.map((r0) => {
        const r = draft && draft.id === r0.id ? draft : r0;
        const st = regionStatus(r);
        const b = polygonBounds(r.polygon);
        const above = b.y0 - 8 * k > 0;
        // zoomed far out the labels collide: show only the field number (no status text)
        const low = scale < 0.4;
        const short = r.field_index != null ? String(r.field_index) : (/(\d+)\s*$/.exec(r.name)?.[1] ?? r.name);
        return (
          <HaloText
            key={r.id}
            x={b.x0}
            y={above ? b.y0 - 8 * k : b.y0 - 2 * k + 16 * k}
            scale={scale}
            size={11}
            fill={STATUS_COLOR[st]}
            text={low ? (r.kind === 'exclusion' ? 'x' : short) : r.kind === 'exclusion' ? `${r.name} (excluded)` : `${r.name} · ${st}`}
          />
        );
      })}
    </g>
  );
});

/**
 * Every cell of the APPLIED grid (slide.grid): counted = solid outline + field number, skipped = dotted grey,
 * hovered (grid-pick tool) = highlighted. The field regions themselves are drawn by RegionsLayer as usual.
 */
export function GridCellsLayer({ scale }: { scale: number }) {
  const a = useApp((s) => s.current);
  const hover = useApp((s) => s.gridHover);
  const picking = useApp((s) => s.tool === 'grid-pick');
  if (!a?.slide.grid) return null;
  const k = 1 / scale;
  const cells = storedGridCells(a);
  return (
    <g data-testid="grid-cells" data-count={cells.length} pointerEvents="none">
      {cells.map((c) => {
        const pts = c.polygon.map((p) => p.join(',')).join(' ');
        const xs = c.polygon.map((p) => p[0]);
        const ys = c.polygon.map((p) => p[1]);
        const on = c.state === 'on';
        const hot = picking && hover === c.index;
        const color = on ? LIGHT : '#8b93a1';
        return (
          <g key={c.index} data-field={c.index} data-on={on ? '1' : '0'} data-hover={hot ? '1' : '0'}>
            {hot && <polygon points={pts} fill={on ? 'rgba(255,255,255,0.14)' : 'rgba(139,147,161,0.18)'} stroke="none" />}
            {!on && <polygon points={pts} fill="none" stroke={DARK} strokeWidth={3 * k} strokeDasharray={`${2 * k} ${4 * k}`} />}
            <polygon
              points={pts}
              fill="none"
              stroke={hot ? '#00e5ff' : color}
              strokeWidth={(hot ? 2.4 : on ? 1 : 1.4) * k}
              strokeDasharray={on ? undefined : `${2 * k} ${4 * k}`}
            />
            {(on || picking) && (
              <HaloText
                x={(Math.min(...xs) + Math.max(...xs)) / 2}
                y={(Math.min(...ys) + Math.max(...ys)) / 2 + 6 * k}
                scale={scale}
                size={16}
                anchor="middle"
                fill={on ? LIGHT : '#8b93a1'}
                text={on ? String(c.index) : `${c.index} off`}
              />
            )}
          </g>
        );
      })}
    </g>
  );
}

/** Unapplied grid-field generator preview (dashed cyan; amber = flagged). */
export function GridPreviewLayer({ scale }: { scale: number }) {
  const cells = useApp((s) => s.gridPreview);
  if (!cells) return null;
  const k = 1 / scale;
  return (
    <g data-testid="grid-preview" pointerEvents="none">
      {cells.map((c) => {
        const pts = c.polygon.map((p) => p.join(',')).join(' ');
        const xs = c.polygon.map((p) => p[0]);
        const ys = c.polygon.map((p) => p[1]);
        const color = c.flag ? '#ffb020' : '#00e5ff';
        return (
          <g key={c.index}>
            <polygon points={pts} fill="rgba(0,229,255,0.06)" stroke={DARK} strokeWidth={4 * k} />
            <polygon points={pts} fill="none" stroke={color} strokeWidth={1.8 * k} strokeDasharray={`${6 * k} ${4 * k}`} />
            <HaloText x={(Math.min(...xs) + Math.max(...xs)) / 2} y={(Math.min(...ys) + Math.max(...ys)) / 2 + 5 * k} scale={scale} size={14} anchor="middle" fill={color} text={`Field-${c.index}`} />
          </g>
        );
      })}
    </g>
  );
}

/**
 * Researcher reference marks: a white cross with a dark halo (readable on red and green signal), deliberately unlike the
 * object rings. Hover (when zoomed in) shows the per-field counter. Plane is chosen in the Regions tab.
 */
export function ReferencePointsLayer({ scale }: { scale: number }) {
  const ref = useApp((s) => s.reference);
  const show = useApp((s) => s.showReference);
  const plane = useApp((s) => s.refPlane);
  if (!show || !ref?.available) return null;
  const k = 1 / scale;
  const r = 4.5 * k;
  const pts = plane == null ? ref.points : ref.points.filter((p) => p.plane === plane);
  const hover = scale >= 2;
  return (
    <g data-testid="reference-points" data-count={pts.length}>
      {pts.map((p, i) => {
        const d = `M${p.x - r} ${p.y}H${p.x + r}M${p.x} ${p.y - r}V${p.y + r}`;
        return (
          <g key={i} pointerEvents={hover ? 'all' : 'none'}>
            <title>{`reference: ${p.field != null ? `field ${p.field}, ` : ''}counter ${p.counter}, plane ${p.plane ?? "unspecified (evidence for no plane)"}`}</title>
            <path d={d} stroke="rgba(0,0,0,0.9)" strokeWidth={4 * k} strokeLinecap="round" fill="none" />
            <path d={d} stroke="#ffffff" strokeWidth={1.7 * k} strokeLinecap="round" fill="none" />
            {hover && <circle cx={p.x} cy={p.y} r={r * 1.4} fill="transparent" />}
          </g>
        );
      })}
    </g>
  );
}

/** Live preview of the shifted reference grid for the import dialog (dashed white/violet; greyed = not imported). */
export function ReferenceGridPreviewLayer({ scale }: { scale: number }) {
  const cells = useApp((s) => s.refGridPreview);
  if (!cells) return null;
  const k = 1 / scale;
  return (
    <g data-testid="ref-grid-preview" pointerEvents="none">
      {cells.map((c) => {
        const pts = c.polygon.map((p) => p.join(',')).join(' ');
        const xs = c.polygon.map((p) => p[0]);
        const ys = c.polygon.map((p) => p[1]);
        const color = c.included ? '#d9b3ff' : '#8b93a1';
        return (
          <g key={c.field} data-included={c.included ? '1' : '0'} opacity={c.included ? 1 : 0.7}>
            <polygon points={pts} fill={c.included ? 'rgba(217,179,255,0.07)' : 'none'} stroke={DARK} strokeWidth={4 * k} />
            <polygon points={pts} fill="none" stroke={color} strokeWidth={1.8 * k} strokeDasharray={`${6 * k} ${4 * k}`} />
            <HaloText x={(Math.min(...xs) + Math.max(...xs)) / 2} y={(Math.min(...ys) + Math.max(...ys)) / 2 + 5 * k} scale={scale} size={13} anchor="middle" fill={color} text={`${c.field}${c.counted ? '' : ' (no marks)'}`} />
          </g>
        );
      })}
    </g>
  );
}

/** Auto chamber region preview (solid teal) and suggested exclusions (dashed pink). Nothing is applied until confirmed. */
export function ChamberPreviewLayer({ scale }: { scale: number }) {
  const c = useApp((s) => s.chamberPreview);
  if (!c) return null;
  const k = 1 / scale;
  const pts = (p: [number, number][]) => p.map((q) => q.join(',')).join(' ');
  return (
    <g data-testid="chamber-preview" pointerEvents="none">
      <polygon points={pts(c.polygon)} fill="rgba(45,212,191,0.08)" stroke={DARK} strokeWidth={5 * k} />
      <polygon points={pts(c.polygon)} fill="none" stroke="#2dd4bf" strokeWidth={2.4 * k} />
      {c.suggested_exclusions.map((e, i) => (
        <g key={i}>
          <polygon points={pts(e)} fill="rgba(255,79,176,0.14)" stroke={DARK} strokeWidth={4 * k} />
          <polygon points={pts(e)} fill="none" stroke="#ff4fb0" strokeWidth={1.8 * k} strokeDasharray={`${5 * k} ${3 * k}`} />
        </g>
      ))}
    </g>
  );
}
