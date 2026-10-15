import { useMemo, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { CountClass, SaveSlideReport } from '../api/types';
import { COUNT_CLASSES, defaultLabel, dirOf, gridFields, selectionProblems } from '../lib/saveslide';
import { useApp } from '../state/store';

const CLASS_TEXT: Record<CountClass, string> = { red: 'red only', green: 'green only', yellow: 'yellow (red + green)' };

/** The researcher's Save: choose what to count, which grid fields count, then write the labelled scan + CSVs. */
export function SaveTab() {
  const a = useApp((s) => s.current)!;
  const sel = useApp((s) => s.saveSel);
  const patch = useApp((s) => s.patch);
  const toast = useApp((s) => s.toast);
  const [outDir, setOutDir] = useState('');
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<SaveSlideReport | null>(null);
  const [error, setError] = useState('');
  const src = a.image?.source_path ?? null;
  const problems = selectionProblems(sel);
  const fields = useMemo(() => gridFields(a), [a]);
  const counted = fields.filter((f) => f.counted).length;
  const needDir = !src && !outDir.trim();

  const setRow = (i: number, p: Partial<(typeof sel)[number]>) => patch({ saveSel: sel.map((r, k) => (k === i ? { ...r, ...p } : r)) });
  const toggleClass = (i: number, c: CountClass) => {
    const cur = sel[i].classes;
    setRow(i, { classes: cur.includes(c) ? cur.filter((x) => x !== c) : COUNT_CLASSES.filter((x) => x === c || cur.includes(x)) });
  };

  const save = async () => {
    setBusy(true);
    setError('');
    setReport(null);
    try {
      const r = await api.saveSlide(a.id, sel, src ? undefined : outDir.trim());
      setReport(r);
      toast('success', `Saved ${r.files.length} files to ${r.directory}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="savetab" data-testid="save-tab">
      <h3>Save</h3>
      <div className="muted small">
        Writes the scan with your labels and the counts as CSV files. Check and correct the labels first (Review tab); Save can be repeated and replaces the earlier result.
      </div>

      <section>
        <h4>1. What to count</h4>
        {sel.map((r, i) => (
          <div className="save-row" key={i} data-testid={`save-row-${i}`}>
            <div className="row wrap">
              {COUNT_CLASSES.map((c) => (
                <label className="chk small" key={c}>
                  <input type="checkbox" checked={r.classes.includes(c)} onChange={() => toggleClass(i, c)} data-testid={`save-${i}-${c}`} /> {CLASS_TEXT[c]}
                </label>
              ))}
              {sel.length > 1 && (
                <button className="btn sm ghost" onClick={() => patch({ saveSel: sel.filter((_, k) => k !== i) })} aria-label="Remove this count">
                  ✕
                </button>
              )}
            </div>
            <label className="small">
              Written in the CSV as
              <input
                className="input sm"
                value={r.label ?? ''}
                placeholder={defaultLabel(r.classes) || 'what counted'}
                onChange={(e) => setRow(i, { label: e.target.value })}
                data-testid={`save-label-${i}`}
              />
            </label>
          </div>
        ))}
        <button className="btn sm" onClick={() => patch({ saveSel: [...sel, { classes: ['yellow'] }] })} data-testid="save-add-row">
          + Count something else too
        </button>
        <div className="muted small">Each line becomes its own row in the experiment CSV (the same grid fields, a different thing counted).</div>
      </section>

      <section>
        <h4>
          2. Grid fields to count{' '}
          <span className="muted small" data-testid="save-grid-count">
            ({counted} of {fields.length})
          </span>
        </h4>
        {fields.length === 0 ? (
          <div className="empty">No grid yet. Open the Regions tab → “Generate grid…” to add the grid (same size as in ImageJ).</div>
        ) : (
          <>
            <GridMap />
            <div className="muted small">Click a field to switch it off (bad area) or on. A field that is off stays empty in the CSV; a counted field with no polonies is 0.</div>
          </>
        )}
      </section>

      <section>
        <h4>3. Where</h4>
        {src ? (
          <div className="small" data-testid="save-where">
            Next to the original: <span className="mono">{dirOf(src)}</span>
          </div>
        ) : (
          <label className="small">
            This scan was uploaded, so its folder is unknown. Folder to save in:
            <input className="input sm" value={outDir} onChange={(e) => setOutDir(e.target.value)} placeholder="/path/to/folder" data-testid="save-outdir" />
          </label>
        )}
      </section>

      {problems.length > 0 && (
        <div className="notice warn small" role="alert">
          {problems.join(' ')}
        </div>
      )}
      <button className="btn primary" disabled={busy || problems.length > 0 || counted === 0 || needDir} onClick={() => void save()} data-testid="save-btn">
        {busy ? 'Saving…' : 'Save'}
      </button>
      {error && (
        <div className="notice err small" role="alert" data-testid="save-error">
          {error}
        </div>
      )}
      {report && (
        <div className="notice ok small" data-testid="save-report">
          <strong>Saved</strong> in <span className="mono">{report.directory}</span>
          <ul>
            {report.files.map((f) => (
              <li key={f} className="mono">
                {f}
              </li>
            ))}
          </ul>
          {report.rows.map((r) => (
            <div key={r.what_counted}>
              {r.what_counted}: {r.nofpol} polonies in {r.nofgrids} fields
            </div>
          ))}
          {report.warnings.map((w) => (
            <div key={w} className="warn-text">
              {w}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Small map of the grid: each field is a clickable cell placed where it lies on the image. */
function GridMap() {
  const a = useApp((s) => s.current)!;
  const setAnalysis = useApp((s) => s.setAnalysis);
  const toast = useApp((s) => s.toast);
  const fields = useMemo(() => gridFields(a), [a]);
  const locked = a.status === 'finalized';
  const W = a.image?.width ?? 1;
  const H = a.image?.height ?? 1;
  const toggle = async (field: number, counted: boolean) => {
    try {
      setAnalysis(await api.setGridField(a.id, field, counted));
      void useApp.getState().refreshDerived();
    } catch (e) {
      toast('error', `grid field ${field}: ${e instanceof ApiError ? e.message : String(e)}`);
    }
  };
  return (
    <svg className="gridmap" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Grid fields" data-testid="grid-map">
      <rect x={0} y={0} width={W} height={H} className="gridmap-bg" />
      {fields.map((f) => {
        const xs = f.polygon.map((p) => p[0]);
        const ys = f.polygon.map((p) => p[1]);
        const x0 = Math.min(...xs), y0 = Math.min(...ys);
        const w = Math.max(...xs) - x0, h = Math.max(...ys) - y0;
        return (
          <g
            key={f.field}
            className={`gridmap-cell ${f.counted ? 'on' : 'off'}`}
            role="checkbox"
            aria-checked={f.counted}
            aria-label={`Field ${f.field}`}
            tabIndex={locked ? -1 : 0}
            data-testid={`grid-cell-${f.field}`}
            onClick={() => !locked && void toggle(f.field, !f.counted)}
            onKeyDown={(e) => (e.key === ' ' || e.key === 'Enter') && !locked && (e.preventDefault(), void toggle(f.field, !f.counted))}
          >
            <rect x={x0} y={y0} width={w} height={h} />
            <text x={x0 + w / 2} y={y0 + h / 2} textAnchor="middle" dominantBaseline="middle" style={{ fontSize: Math.max(12, Math.min(w, h) / 3) }}>
              {f.counted ? f.field : `${f.field} ✕`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
