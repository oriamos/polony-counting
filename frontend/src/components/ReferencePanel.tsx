import { useEffect, useMemo, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { CompareCountType, CompareResponse, ImportMode, Region } from '../api/types';
import { COUNT_TYPE_LABEL, deltaClass, fmtDelta, previewImport, STATUS_LABEL } from '../lib/reference';
import { useApp } from '../state/store';
import { NumInput } from './DetectTab';

const errText = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));

/** Plane choices with the channel role the backend merely SUGGESTS (never a colour label, never applied). */
function usePlanes() {
  const a = useApp((s) => s.current)!;
  const ref = useApp((s) => s.reference);
  return (ref?.planes ?? []).map((p) => {
    const ch = a.image?.channels.find((c) => c.page === p.page);
    return { ...p, suggested: ch?.suggested_role ?? null };
  });
}

export function ReferenceSection() {
  const a = useApp((s) => s.current)!;
  const ref = useApp((s) => s.reference);
  const show = useApp((s) => s.showReference);
  const plane = useApp((s) => s.refPlane);
  const patch = useApp((s) => s.patch);
  const planes = usePlanes();
  const [open, setOpen] = useState(false);
  if (!ref?.available) return null;
  return (
    <section className="gridgen refsec" data-testid="reference-section">
      <h4>Researcher reference</h4>
      <div className="muted small">
        Marks embedded in the TIFF{ref.source_file ? ` (${ref.source_file})` : ''}. Read-only: they are never used as detections and never change your counts.
      </div>
      <ul className="plane-list" data-testid="reference-planes">
        {planes.map((p) => (
          <li key={p.page}>
            <span className="mono">page {p.page}</span>: {p.point_count} point{p.point_count === 1 ? '' : 's'}
            {p.suggested && (
              <span className="tag draft" title="Suggested from validated GenePix stacks. Not applied.">
                suggested channel role: {p.suggested}
              </span>
            )}
          </li>
        ))}
      </ul>
      {ref.warnings.map((w, i) => (
        <div key={i} className="small warn-text">
          {w}
        </div>
      ))}
      <div className="row wrap">
        <label className="chk small">
          <input type="checkbox" checked={show} onChange={(e) => patch({ showReference: e.target.checked })} data-testid="ref-show" /> show reference points
        </label>
        <label className="small row">
          plane
          <select
            className="input sm"
            value={plane ?? ''}
            onChange={(e) => patch({ refPlane: e.target.value === '' ? null : Number(e.target.value) })}
            data-testid="ref-plane-view"
            aria-label="Reference plane to display"
          >
            <option value="">all</option>
            {planes.map((p) => (
              <option key={p.page} value={p.page}>
                page {p.page}
              </option>
            ))}
          </select>
        </label>
      </div>
      {ref.grid && a.status !== 'finalized' && (
        <button className="btn sm" onClick={() => setOpen(!open)} aria-expanded={open} data-testid="ref-import-toggle">
          {open ? 'Close import' : 'Import researcher fields…'}
        </button>
      )}
      {!ref.grid && <div className="small muted">No researcher grid found, so fields cannot be imported.</div>}
      {open && ref.grid && <ImportPanel onDone={() => setOpen(false)} />}
      {!open && null}
    </section>
  );
}

function ImportPanel({ onDone }: { onDone: () => void }) {
  const a = useApp((s) => s.current)!;
  const ref = useApp((s) => s.reference)!;
  const patch = useApp((s) => s.patch);
  const setAnalysis = useApp((s) => s.setAnalysis);
  const toast = useApp((s) => s.toast);
  const planes = usePlanes();
  const grid = ref.grid!;
  const [mode, setMode] = useState<ImportMode>('counted');
  const [plane, setPlane] = useState<number | null>(planes.length ? planes[0].page : null);
  const [dx, setDx] = useState(0);
  const [dy, setDy] = useState(0);
  const [replace, setReplace] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const existing = a.regions.filter((r) => r.name.startsWith('Field-') || (r.kind === 'counting' && r.field_index != null));

  const cells = useMemo(() => previewImport(grid, ref, mode, plane, dx, dy), [grid, ref, mode, plane, dx, dy]);
  const nIncluded = cells.filter((c) => c.included).length;
  const nUncounted = cells.filter((c) => !c.counted).length;
  useEffect(() => {
    patch({ refGridPreview: cells, showRegions: true });
  }, [cells, patch]);
  useEffect(() => () => patch({ refGridPreview: null }), [patch]);

  const apply = async () => {
    if (existing.length && replace && !confirm(`Replace ${existing.length} existing Field-* region(s) with ${nIncluded} researcher field(s)? Their approvals and inspections are lost.`)) return;
    setBusy(true);
    setErr('');
    try {
      const next = await api.importFields(a.id, { mode, shift: [dx, dy], plane, replace });
      setAnalysis(next);
      patch({ refGridPreview: null });
      toast('success', `Imported ${nIncluded} researcher field region(s)`);
      onDone();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="refimport" role="group" aria-label="Import researcher fields" data-testid="ref-import">
      <div className="notice small" data-testid="ref-ambiguity">
        Fields with no researcher points are ambiguous: not counted or counted as zero. Choose.
      </div>
      <div className="form-grid">
        <label>
          Fields to import
          <select className="input sm" value={mode} onChange={(e) => setMode(e.target.value as ImportMode)} data-testid="ref-mode">
            <option value="counted">counted fields only (with researcher points)</option>
            <option value="all">all {grid.cells.length} cells (uncounted fields become 0)</option>
          </select>
        </label>
        <label>
          Plane
          <select
            className="input sm"
            value={plane ?? ''}
            onChange={(e) => setPlane(e.target.value === '' ? null : Number(e.target.value))}
            data-testid="ref-plane"
          >
            <option value="">backend default</option>
            {planes.map((p) => (
              <option key={p.page} value={p.page}>
                page {p.page} ({p.point_count} points){p.suggested ? ` · suggested role ${p.suggested}` : ''}
              </option>
            ))}
          </select>
        </label>
        <label>
          Grid shift dx (px)
          <NumInput value={dx} step={1} onChange={(v) => setDx(v ?? 0)} testid="ref-dx" />
        </label>
        <label>
          Grid shift dy (px)
          <NumInput value={dy} step={1} onChange={(v) => setDy(v ?? 0)} testid="ref-dy" />
        </label>
      </div>
      <div className="row wrap small">
        nudge:
        {([['←', -5, 0], ['→', 5, 0], ['↑', 0, -5], ['↓', 0, 5]] as const).map(([l, x, y]) => (
          <button key={l} className="btn sm ghost" onClick={() => (setDx(dx + x), setDy(dy + y))} aria-label={`Nudge grid ${l} by 5 px`}>
            {l} 5
          </button>
        ))}
        <button className="btn sm ghost" onClick={() => (setDx(0), setDy(0))}>
          reset shift
        </button>
      </div>
      <div className="small mono muted" data-testid="ref-import-summary">
        {nIncluded} of {grid.cells.length} cells will be created · {nUncounted} with no researcher points · shift ({dx}, {dy}) px · preview shown on the image
      </div>
      {existing.length > 0 && (
        <label className="chk small">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} data-testid="ref-replace" /> Replace the {existing.length} existing Field-* region(s) (approvals lost)
        </label>
      )}
      {err && (
        <div className="notice err small" role="alert" data-testid="ref-import-error">
          {err}
        </div>
      )}
      <div className="row wrap">
        <button className="btn sm primary" disabled={busy || nIncluded === 0 || (existing.length > 0 && !replace)} onClick={() => void apply()} data-testid="ref-import-apply">
          {busy ? 'Importing…' : `Import ${nIncluded} field region(s)`}
        </button>
        <button className="btn sm ghost" onClick={onDone}>
          Cancel
        </button>
      </div>
    </div>
  );
}

export function ChamberSection() {
  const a = useApp((s) => s.current)!;
  const preview = useApp((s) => s.chamberPreview);
  const patch = useApp((s) => s.patch);
  const saveRegions = useApp((s) => s.saveRegions);
  const [rim, setRim] = useState(40);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [withExcl, setWithExcl] = useState(true);
  useEffect(() => () => patch({ chamberPreview: null }), [patch]);
  const locked = a.status === 'finalized';

  const run = async () => {
    setBusy(true);
    setErr('');
    try {
      patch({ chamberPreview: await api.autoChamber(a.id, rim) });
    } catch (e) {
      patch({ chamberPreview: null });
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };
  const apply = () => {
    if (!preview) return;
    const stamp = Date.now().toString(36);
    const base = { approved: false, approved_at: null, approval_fingerprint: null, inspected: false };
    const made: Region[] = [{ id: `reg-ch${stamp}`, kind: 'counting', name: 'Chamber', polygon: preview.polygon, ...base }];
    if (withExcl) preview.suggested_exclusions.forEach((p, i) => made.push({ id: `reg-chx${stamp}-${i}`, kind: 'exclusion', name: `Chamber exclusion ${i + 1}`, polygon: p, ...base }));
    void saveRegions([...a.regions, ...made]);
    patch({ chamberPreview: null });
  };

  return (
    <section className="gridgen" data-testid="chamber-section">
      <h4>Auto chamber region</h4>
      <div className="muted small">Proposes a counting region inside the chamber, leaving a rim at the edge. A preview only: nothing is added until you apply it.</div>
      <div className="row wrap">
        <label className="row small">
          Rim width (px)
          <NumInput value={rim} min={0} step={5} onChange={(v) => setRim(v ?? 40)} testid="chamber-rim" />
        </label>
        <button className="btn sm" disabled={busy || locked} onClick={() => void run()} data-testid="chamber-run">
          {busy ? 'Working…' : 'Auto chamber region'}
        </button>
        {preview && (
          <button className="btn sm ghost" onClick={() => patch({ chamberPreview: null })}>
            Discard preview
          </button>
        )}
      </div>
      {err && (
        <div className="notice err small" role="alert" data-testid="chamber-error">
          {err}
        </div>
      )}
      {preview && (
        <div data-testid="chamber-info" className="small">
          <div className="mono muted">
            {Math.round(preview.area_px2).toLocaleString()} px² · rim {preview.rim_px} px · method {preview.method} · {preview.suggested_exclusions.length} suggested exclusion(s)
          </div>
          {preview.warnings.map((w, i) => (
            <div key={i} className="warn-text">
              {w}
            </div>
          ))}
          {preview.suggested_exclusions.length > 0 && (
            <label className="chk">
              <input type="checkbox" checked={withExcl} onChange={(e) => setWithExcl(e.target.checked)} data-testid="chamber-with-excl" /> also add the suggested exclusion(s)
            </label>
          )}
          <button className="btn sm primary" disabled={locked} onClick={apply} data-testid="chamber-apply">
            Apply as counting region
          </button>
        </div>
      )}
    </section>
  );
}

const COUNT_TYPES = Object.keys(COUNT_TYPE_LABEL) as CompareCountType[];

export function CompareSection() {
  const a = useApp((s) => s.current)!;
  const ref = useApp((s) => s.reference);
  const planes = usePlanes();
  const [plane, setPlane] = useState<number | null>(null);
  const [ct, setCt] = useState<CompareCountType>('red_positive');
  const [data, setData] = useState<CompareResponse | null>(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    if (plane == null && planes.length) setPlane(planes[0].page);
  }, [planes.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const key = `${a.id}|${a.updated_at}|${plane}|${ct}|${a.objects.length}`;
  useEffect(() => {
    if (!ref?.available) return;
    let live = true;
    api
      .compareFields(a.id, plane, ct)
      .then((d) => live && (setData(d), setErr('')))
      .catch((e) => live && (setData(null), setErr(errText(e))));
    return () => {
      live = false;
    };
  }, [key, ref?.available]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!ref?.available) return null;
  const f = (n: number | null | undefined, d = 2) => (n == null || !Number.isFinite(n) ? '–' : n.toFixed(d));
  return (
    <section className="compare" data-testid="compare-section">
      <h4>Compare with reference</h4>
      <div className="row wrap small">
        <label className="row">
          Reference plane
          <select className="input sm" value={plane ?? ''} onChange={(e) => setPlane(e.target.value === '' ? null : Number(e.target.value))} data-testid="cmp-plane">
            {planes.map((p) => (
              <option key={p.page} value={p.page}>
                page {p.page}
              </option>
            ))}
          </select>
        </label>
        <label className="row">
          App count
          <select className="input sm" value={ct} onChange={(e) => setCt(e.target.value as CompareCountType)} data-testid="cmp-type">
            {COUNT_TYPES.map((t) => (
              <option key={t} value={t}>
                {COUNT_TYPE_LABEL[t]}
              </option>
            ))}
          </select>
        </label>
      </div>
      {err && (
        <div className="notice err small" role="alert" data-testid="cmp-error">
          {err}
        </div>
      )}
      {data && (
        <>
          <div className="tbl-wrap">
            <table className="tbl" data-testid="cmp-table">
              <thead>
                <tr>
                  <th>field</th>
                  <th>status</th>
                  <th className="num">reference</th>
                  <th className="num">app</th>
                  <th className="num">delta</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r) => (
                  <tr key={r.field} className={r.examined ? '' : 'muted'} data-testid={`cmp-row-${r.field}`}>
                    <th scope="row">{r.field}</th>
                    <td className="small" data-testid={`cmp-status-${r.field}`} title={r.reason ?? r.examination_source}>
                      {STATUS_LABEL[r.status] ?? r.status}
                    </td>
                    <td className="num mono">{r.reference_count ?? '–'}</td>
                    <td className="num mono">{r.app_count ?? '–'}</td>
                    <td className={`num mono ${deltaClass(r.delta)}`} data-testid={`cmp-delta-${r.field}`}>
                      {fmtDelta(r.delta)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(data.summary.unplaned_marks ?? 0) > 0 && (
            <div className="notice warn-text small" role="alert" data-testid="cmp-unplaned">
              <strong>{data.summary.unplaned_marks} reference mark(s) have no valid plane number.</strong> They count as evidence for no plane. Fields
              containing them are conflicts on every plane and excluded from all metrics
              {data.summary.unplaned_mark_fields?.length ? `: field ${data.summary.unplaned_mark_fields.join(', ')}` : ''}.
            </div>
          )}
          <dl className="kv" data-testid="cmp-summary">
            <dt>fields in the analysis</dt>
            <dd className="mono">{data.summary.n_fields}</dd>
            <dt>eligible (examined)</dt>
            <dd className="mono" data-testid="cmp-eligible">{data.summary.eligible_examined_fields}</dd>
            <dt>positive marked</dt>
            <dd className="mono">{data.summary.positive_marked_fields}</dd>
            <dt>verified zero</dt>
            <dd className="mono">{data.summary.verified_zero_fields}</dd>
            <dt>unknown, excluded</dt>
            <dd className="mono" data-testid="cmp-unknown">{data.summary.unknown_fields_excluded}</dd>
            {data.summary.conflict_fields_excluded > 0 && (
              <>
                <dt>conflict, excluded</dt>
                <dd className="mono">{data.summary.conflict_fields_excluded}</dd>
              </>
            )}
            <dt>evidence</dt>
            <dd className="mono">{data.summary.eligibility_source}</dd>
            <dt>Pearson r</dt>
            <dd className="mono">{f(data.summary.pearson_r)}</dd>
            <dt>slope</dt>
            <dd className="mono">{f(data.summary.slope)}</dd>
            <dt>MAE</dt>
            <dd className="mono">{f(data.summary.mae)}</dd>
            <dt>total reference</dt>
            <dd className="mono">{data.summary.total_reference ?? '–'}</dd>
            <dt>total app</dt>
            <dd className="mono">{data.summary.total_app ?? '–'}</dd>
          </dl>
          <div className="muted small">Totals, r, slope and MAE use the eligible fields only. A field without marks on this plane is unknown, not zero.</div>
          {(data.warnings ?? []).map((w, k) => (
            <div key={k} className="notice small" data-testid="cmp-warning">
              {w}
            </div>
          ))}
        </>
      )}
      <div className="muted small" data-testid="cmp-note">
        Reference = the researcher&apos;s embedded marks for this plane; agreement is not the same as accuracy.
      </div>
    </section>
  );
}
