import type { ImageView, ObjectClass } from '../api/types';
import { CLASS_KEY, CLASS_LABEL, CLASS_ORDER, glyphFor, glyphPath } from '../lib/glyph';
import { liveObjects, useApp, type Tool } from '../state/store';
import { useCursor } from './Viewer';
import { EngineLayersMenu } from './EngineLayers';

const VIEWS: { id: ImageView; label: string; key: string }[] = [
  { id: 'red', label: 'Raw red', key: '1' },
  { id: 'green', label: 'Raw green', key: '2' },
  { id: 'composite', label: 'Composite', key: '3' },
  { id: 'processed_red', label: 'Processed red', key: '4' },
  { id: 'processed_green', label: 'Processed green', key: '5' },
];

export function GlyphIcon({ cls, size = 16 }: { cls: ObjectClass; size?: number }) {
  const g = glyphFor(cls);
  const c = size / 2;
  const r = size * 0.34;
  const path = glyphPath(g.shape, c, c, g.shape === 'square' ? r * 0.88 : r * 1.12);
  const common = { fill: 'none', strokeDasharray: g.dashed ? '2.4 1.8' : undefined } as const;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
      {path ? (
        <>
          <path d={path} {...common} stroke="#000" strokeWidth={3.4} />
          <path d={path} {...common} stroke="#fff" strokeWidth={2} />
          <path d={path} {...common} stroke={g.color} strokeWidth={0.9} />
        </>
      ) : (
        <>
          <circle cx={c} cy={c} r={r} {...common} stroke="#000" strokeWidth={3.4} />
          <circle cx={c} cy={c} r={r} {...common} stroke="#fff" strokeWidth={2} />
          <circle cx={c} cy={c} r={r} {...common} stroke={g.color} strokeWidth={0.9} />
        </>
      )}
    </svg>
  );
}

export function ViewerToolbar() {
  const a = useApp((s) => s.current)!;
  const s = useApp();
  const live = liveObjects(a);
  const counts: Record<ObjectClass, number> = { red: 0, green: 0, yellow: 0, uncertain: 0 };
  for (const o of live) counts[o.cls]++;
  const tools: { id: Tool; label: string; key?: string; title: string }[] = [
    { id: 'select', label: 'Select', key: 'V', title: 'Select / move / pan (V). Shift+click multi-select, Shift+drag box-select' },
    { id: 'add', label: 'Add spot', key: 'A', title: 'Click to add a spot of the chosen class (A)' },
    { id: 'split', label: 'Split', key: 'S', title: 'Split the selected spot: click two positions (S)' },
  ];
  const regionTools: { id: Tool; label: string; title: string }[] = [
    { id: 'draw-count-rect', label: '▭ Count', title: 'Draw rectangular counting region' },
    { id: 'draw-count-poly', label: '⬠ Count', title: 'Draw polygon counting region' },
    { id: 'draw-excl-rect', label: '▭ Exclude', title: 'Draw rectangular exclusion mask' },
    { id: 'draw-excl-poly', label: '⬠ Exclude', title: 'Draw polygon exclusion mask' },
  ];
  const canEdit = a.status !== 'finalized';
  return (
    <div className="vtool" role="toolbar" aria-label="Viewer toolbar">
      <div className="vrow">
        <div className="seg" role="group" aria-label="Image view">
          {VIEWS.map((v) => (
            <button key={v.id} className={s.imageView === v.id ? 'on' : ''} onClick={() => s.setImageView(v.id)} title={`${v.label} (${v.key})`} data-testid={`view-${v.id}`}>
              {v.label}
            </button>
          ))}
        </div>
        <div className="sep" />
        <button className="btn sm" onClick={() => s.fit()} title="Fit image to window (F)" data-testid="fit">
          Fit
        </button>
        <button className="btn sm" onClick={() => s.zoomToPoint(a.image!.width / 2, a.image!.height / 2, 1)} title="Zoom to 100%">
          1:1
        </button>
        <div className="sep" />
        <div className="class-toggles" role="group" aria-label="Show classes">
          {CLASS_ORDER.map((c) => (
            <label key={c} className={`ctoggle ${s.classVisible[c] ? 'on' : ''}`} title={`${CLASS_LABEL[c]} – toggle overlay`}>
              <input type="checkbox" checked={s.classVisible[c]} onChange={() => s.patch({ classVisible: { ...s.classVisible, [c]: !s.classVisible[c] } })} data-testid={`vis-${c}`} />
              <GlyphIcon cls={c} />
              <span className="mono">{counts[c]}</span>
            </label>
          ))}
        </div>
        <button
          className={`btn sm ${s.hideOverlays ? 'on' : ''}`}
          aria-pressed={s.hideOverlays}
          onClick={() => s.patch({ hideOverlays: !s.hideOverlays })}
          title="Hide / show all overlays to check for omissions (H)"
          data-testid="hide-overlays"
        >
          Hide overlays <kbd>H</kbd>
        </button>
        <EngineLayersMenu />
        <select
          className="input sm"
          value={s.showLabels}
          onChange={(e) => s.patch({ showLabels: e.target.value as 'auto' | 'on' | 'off' })}
          aria-label="ID labels"
          title="ID labels"
          data-testid="labels-mode"
        >
          <option value="auto">IDs: auto</option>
          <option value="on">IDs: on</option>
          <option value="off">IDs: off</option>
        </select>
        <label className="chk small">
          <input type="checkbox" checked={s.showRegions} onChange={(e) => s.patch({ showRegions: e.target.checked })} /> regions
        </label>
        {a.image!.embedded_rois.length > 0 && (
          <label className="chk small" title="Show ROIs embedded in the TIFF (suggestions only)">
            <input type="checkbox" checked={s.showSuggestions} onChange={(e) => s.patch({ showSuggestions: e.target.checked })} /> ROI suggestions
          </label>
        )}
      </div>
      <div className="vrow">
        <div className="seg" role="group" aria-label="Tool">
          {tools.map((t) => (
            <button key={t.id} className={s.tool === t.id ? 'on' : ''} onClick={() => s.setTool(t.id)} title={t.title} disabled={!canEdit && t.id !== 'select'} data-testid={`tool-${t.id}`}>
              {t.label} {t.key && <kbd>{t.key}</kbd>}
            </button>
          ))}
        </div>
        {s.tool === 'add' && (
          <div className="seg" role="group" aria-label="Class for new spots">
            {CLASS_ORDER.map((c) => (
              <button key={c} className={s.addClass === c ? 'on' : ''} onClick={() => s.patch({ addClass: c })} title={`Add as ${CLASS_LABEL[c]}`} data-testid={`addcls-${c}`}>
                <GlyphIcon cls={c} size={14} /> {CLASS_KEY[c]}
              </button>
            ))}
          </div>
        )}
        <button className="btn sm" disabled={s.selection.length < 2 || !canEdit} onClick={() => void s.act('merge', { object_ids: s.selection })} title="Merge selected spots (M)" data-testid="merge">
          Merge <kbd>M</kbd>
        </button>
        <button className="btn sm" disabled={!s.selection.length || !canEdit} onClick={() => void s.act('delete', { object_ids: s.selection })} title="Delete selected (Del)" data-testid="delete">
          Delete <kbd>Del</kbd>
        </button>
        <div className="sep" />
        <div className="seg" role="group" aria-label="Region tools">
          {regionTools.map((t) => (
            <button key={t.id} className={s.tool === t.id ? 'on' : ''} onClick={() => s.setTool(t.id)} title={t.title} disabled={!canEdit} data-testid={`tool-${t.id}`}>
              {t.label}
            </button>
          ))}
        </div>
        <button className={`btn sm ${s.tool === 'preview-window' ? 'on' : ''}`} onClick={() => s.setTool('preview-window')} title="Draw a window to preview detection" data-testid="tool-preview-window">
          ◳ Preview window
        </button>
        <div className="sep" />
        <button className="btn sm" disabled={a.undo_depth <= 0 || !canEdit} onClick={() => void s.undo()} title="Undo (Ctrl+Z)" data-testid="undo">
          ↶ Undo{a.undo_depth > 0 ? ` (${a.undo_depth})` : ''}
        </button>
        <button className="btn sm" disabled={a.redo_depth <= 0 || !canEdit} onClick={() => void s.redo()} title="Redo (Ctrl+Shift+Z)" data-testid="redo">
          ↷ Redo
        </button>
      </div>
      <DisplayControls />
    </div>
  );
}

const sliderToVal = (t: number, max: number) => Math.round(max * Math.pow(t / 1000, 2));
const valToSlider = (v: number, max: number) => Math.round(1000 * Math.sqrt(Math.max(0, Math.min(1, v / max))));

function DisplayControls() {
  const a = useApp((s) => s.current)!;
  const view = useApp((s) => s.imageView);
  const d = useApp((s) => s.display[s.imageView]);
  const auto = useApp((s) => s.autoRange);
  const setDisplay = useApp((s) => s.setDisplay);
  const reset = useApp((s) => s.resetDisplay);
  const img = a.image!;
  const dtypeMax = Math.pow(2, img.bit_depth || 8) - 1;
  const chMax = Math.max(0, ...img.channels.map((c) => c.max));
  const max = Math.max(view === 'composite' && img.composite_only ? 255 : dtypeMax > 65535 ? chMax || 65535 : dtypeMax, auto?.hi ?? 0, 1);
  const lo = d.lo ?? auto?.lo ?? 0;
  const hi = d.hi ?? auto?.hi ?? max;
  const isAuto = d.lo == null && d.hi == null && d.gamma === 1;
  const baseline = { lo, hi };
  return (
    <div className="vrow display-row" data-testid="display-controls">
      <span className="display-label" title="These sliders only change how the PNG is rendered for your eyes. Detection always uses the raw data.">
        Display only – does not affect detection
      </span>
      <label className="slider">
        lo
        <input
          type="range"
          min={0}
          max={1000}
          value={valToSlider(lo, max)}
          onChange={(e) => setDisplay({ ...baseline, lo: Math.min(sliderToVal(+e.target.value, max), hi - 1) })}
          aria-label="Display lower bound"
          data-testid="slider-lo"
        />
        <span className="mono num">{Math.round(lo)}</span>
      </label>
      <label className="slider">
        hi
        <input
          type="range"
          min={0}
          max={1000}
          value={valToSlider(hi, max)}
          onChange={(e) => setDisplay({ ...baseline, hi: Math.max(sliderToVal(+e.target.value, max), lo + 1) })}
          aria-label="Display upper bound"
          data-testid="slider-hi"
        />
        <span className="mono num">{Math.round(hi)}</span>
      </label>
      <label className="slider">
        γ
        <input type="range" min={0.2} max={3} step={0.05} value={d.gamma} onChange={(e) => setDisplay({ gamma: +e.target.value })} aria-label="Display gamma" data-testid="slider-gamma" />
        <span className="mono num">{d.gamma.toFixed(2)}</span>
      </label>
      <button className={`btn sm ${isAuto ? 'on' : ''}`} onClick={reset} title="Automatic contrast (percentile stretch computed by the server)" data-testid="auto-contrast">
        Auto
      </button>
    </div>
  );
}

export function StatusBar() {
  const a = useApp((s) => s.current)!;
  const view = useApp((s) => s.view);
  const pt = useCursor((s) => s.pt);
  const busy = useApp((s) => s.busy);
  const ps = a.image!.pixel_size_um;
  return (
    <div className="statusbar" role="status">
      <span className="mono" data-testid="coord-readout">
        {pt ? `x ${pt.x.toFixed(1)}  y ${pt.y.toFixed(1)} px` : 'x –  y –'}
      </span>
      <span className="muted">·</span>
      <span className="mono" data-testid="scale-readout">
        zoom {(view.scale * 100).toFixed(view.scale < 1 ? 1 : 0)}%
      </span>
      <span className="muted">·</span>
      <span className="mono">
        {a.image!.width}×{a.image!.height} px · {a.image!.dtype}
      </span>
      <span className="muted">·</span>
      {ps ? <span className="mono">{ps} µm/px</span> : <span className="warn-text" title="Pixel size unknown: all measures are in pixels and nothing is converted to µm">pixel size unknown – units are pixels</span>}
      {busy > 0 && (
        <span className="right">
          <span className="spinner" /> saving…
        </span>
      )}
    </div>
  );
}
