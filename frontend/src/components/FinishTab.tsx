import { useEffect, useState } from 'react';
import { api, exportUrl } from '../api/client';
import type { SlideMeta } from '../api/types';
import { liveObjects, useApp } from '../state/store';
import { NumInput } from './DetectTab';
import { DraftBadge } from './TotalsTab';

export function FinishTab() {
  const a = useApp((s) => s.current)!;
  const finalize = useApp((s) => s.finalize);
  const reopen = useApp((s) => s.reopen);
  const patch = useApp((s) => s.patch);
  const final = a.status === 'finalized';
  const counting = a.regions.filter((r) => r.kind === 'counting');
  const live = liveObjects(a);
  const unresolved = live.filter((o) => o.cls === 'uncertain').length;
  const mapped = !!a.image && a.image.channels.length > 0 && a.image.channels.every((c) => c.role_confirmed);
  const items: { ok: boolean; text: string; fix?: () => void; fixLabel?: string }[] = [
    { ok: mapped, text: 'Channel mapping confirmed', fix: () => patch({ mappingOpen: true }), fixLabel: 'Open mapping' },
    { ok: counting.length > 0, text: 'At least one counting region', fix: () => patch({ rightTab: 'regions' }), fixLabel: 'Regions' },
    { ok: !!a.provenance.detection_ran_at || a.objects.length > 0, text: 'Detection has been run', fix: () => patch({ rightTab: 'detect' }), fixLabel: 'Detect' },
    { ok: !a.provenance.stale_detection, text: 'Detection is up to date (not stale)', fix: () => patch({ rightTab: 'detect' }), fixLabel: 'Re-run' },
    {
      ok: counting.length > 0 && counting.every((r) => r.inspected),
      text: `Every counting region inspected for omissions (${counting.filter((r) => r.inspected).length}/${counting.length})`,
      fix: () => patch({ rightTab: 'regions' }),
      fixLabel: 'Regions',
    },
    { ok: unresolved === 0, text: unresolved ? `${unresolved} uncertain object(s) still unresolved` : 'No unresolved uncertain objects', fix: () => patch({ rightTab: 'review' }), fixLabel: 'Review' },
    {
      ok: counting.length > 0 && counting.every((r) => r.approved),
      text: `Every counting region approved (${counting.filter((r) => r.approved).length}/${counting.length})`,
      fix: () => patch({ rightTab: 'regions' }),
      fixLabel: 'Regions',
    },
  ];
  const blockers = items.filter((i) => !i.ok);
  const label = final ? 'FINAL' : 'DRAFT';
  const kinds: { kind: string; label: string; hint: string }[] = [
    { kind: 'objects', label: 'Objects (CSV)', hint: 'one row per detected object with evidence' },
    { kind: 'regions', label: 'Regions (CSV)', hint: 'per-region counts and status' },
    { kind: 'slide', label: 'Slide (CSV)', hint: 'slide totals' },
    { kind: 'provenance', label: 'Provenance (JSON)', hint: 'parameters, versions, hashes' },
    { kind: 'actions', label: 'Action log (JSONL)', hint: 'every review action' },
    { kind: 'history', label: 'Object history (CSV)', hint: 'per-object history trail' },
    { kind: 'bundle', label: 'Bundle (ZIP)', hint: 'everything + README' },
  ];
  return (
    <div className="finish" data-testid="finish-tab">
      <div className="row between">
        <h3>Finalize</h3>
        <DraftBadge final={final} />
      </div>
      <ul className="checklist" data-testid="checklist">
        {items.map((i, n) => (
          <li key={n} className={i.ok ? 'ok' : 'bad'}>
            <span className="tick" aria-hidden>
              {i.ok ? '✓' : '✗'}
            </span>
            <span>{i.text}</span>
            {!i.ok && i.fix && !final && (
              <button className="btn sm ghost" onClick={i.fix}>
                {i.fixLabel}
              </button>
            )}
          </li>
        ))}
      </ul>
      {final ? (
        <div className="row">
          <div className="notice ok small grow">Finalized {a.finalized_at ? new Date(a.finalized_at).toLocaleString() : ''}. Editing is locked.</div>
          <button className="btn" onClick={() => void reopen()} data-testid="reopen">
            Reopen
          </button>
        </div>
      ) : (
        <>
          <button className="btn primary block" disabled={blockers.length > 0} onClick={() => void finalize()} data-testid="finalize">
            Finalize analysis
          </button>
          {blockers.length > 0 && (
            <div className="small muted" data-testid="blockers">
              {blockers.length} blocker{blockers.length > 1 ? 's' : ''} remaining.
            </div>
          )}
        </>
      )}

      <h3>Export</h3>
      <div className="muted small">
        Exports are labelled <strong>{label}</strong>
        {final ? '.' : ': filenames and a status column carry DRAFT until the analysis is finalized.'}
      </div>
      <div className="exports">
        {kinds.map((k) => (
          <a key={k.kind} className="btn" href={exportUrl(a.id, k.kind)} download title={k.hint} data-testid={`export-${k.kind}`}>
            {k.label} <span className={`tag ${final ? 'final' : 'draft'}`}>{label}</span>
          </a>
        ))}
      </div>
      <SlideForm />
    </div>
  );
}

function SlideForm() {
  const a = useApp((s) => s.current)!;
  const setAnalysis = useApp((s) => s.setAnalysis);
  const toast = useApp((s) => s.toast);
  const [m, setM] = useState<SlideMeta>(a.slide);
  const [busy, setBusy] = useState(false);
  useEffect(() => setM(a.slide), [a.id, a.slide]);
  const locked = a.status === 'finalized';
  const dirty = JSON.stringify(m) !== JSON.stringify(a.slide);
  const mapped = !!a.image && a.image.channels.length > 0 && a.image.channels.every((c) => c.role_confirmed);
  const hasArea = (a.slide.counting_area_px2 ?? 0) > 0;
  // Exactly the server's requirements (export/exporter.py concentration()); pixel size is NOT one of them.
  const required: { ok: boolean; text: string }[] = [
    { ok: !!m.concentration_formula_confirmed, text: 'formula confirmation' },
    { ok: (m.sample_volume_ul ?? 0) > 0, text: 'sample volume' },
    { ok: (m.concentration_factor ?? 0) > 0, text: 'concentration factor' },
    { ok: hasArea, text: 'counting area (from the counting regions)' },
    { ok: (m.sample_total_area_px2 ?? 0) > 0, text: 'total sample area' },
    { ok: mapped, text: 'confirmed channel mapping' },
  ];
  const missing = required.filter((r) => !r.ok).map((r) => r.text);
  const complete = missing.length === 0;
  const save = async () => {
    setBusy(true);
    try {
      setAnalysis(await api.putSettings(a.id, { slide: m }));
      toast('success', 'Slide metadata saved');
    } catch (e) {
      toast('error', `slide metadata: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="slideform" data-testid="slide-form">
      <h3>Slide metadata</h3>
      <div className="muted small">All fields are optional and empty by default. A concentration is computed only when every required input is provided; nothing is assumed.</div>
      <div className="form-grid">
        <label>
          Slide ID
          <input className="input sm" value={m.slide_id ?? ''} disabled={locked} onChange={(e) => setM({ ...m, slide_id: e.target.value || null })} data-testid="slide-id" />
        </label>
        <label>
          Sample volume <span className="unit">µL</span>
          <NumInput value={m.sample_volume_ul} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, sample_volume_ul: v })} testid="slide-volume" />
        </label>
        <label>
          Concentration factor <span className="unit">×</span>
          <NumInput value={m.concentration_factor} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, concentration_factor: v })} testid="slide-factor" />
        </label>
        <label>
          Pixel size <span className="unit">µm/px</span>
          <NumInput
            value={m.pixel_size_um}
            nullable
            min={0}
            disabled={locked}
            placeholder={a.image?.pixel_size_um ? String(a.image.pixel_size_um) : 'unknown'}
            onChange={(v) => setM({ ...m, pixel_size_um: v })}
          />
        </label>
        <label className="span2">
          Total sample area <span className="unit">px², same units as counting area</span>
          <NumInput value={m.sample_total_area_px2 ?? null} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, sample_total_area_px2: v })} testid="slide-total-area" />
        </label>
        <label className="span2">
          Sampled-area note
          <input className="input sm" value={m.sampled_area_note ?? ''} disabled={locked} onChange={(e) => setM({ ...m, sampled_area_note: e.target.value || null })} />
        </label>
      </div>
      <div className="notice warn small" data-testid="formula-note">
        <strong>PROPOSED / UNCONFIRMED formula:</strong>{' '}
        <span className="mono">concentration = count / counting area × total sample area ÷ sample volume × concentration factor</span>. Concentration is computed only by the server, and only after you confirm the
        formula and geometry below; this page never calculates it.
        <label className="chk">
          <input
            type="checkbox"
            checked={!!m.concentration_formula_confirmed}
            disabled={locked}
            onChange={(e) => setM({ ...m, concentration_formula_confirmed: e.target.checked })}
            data-testid="formula-confirmed"
          />{' '}
          I confirm the concentration formula and geometry
        </label>
      </div>
      <div className="small mono muted">counting area (from regions): {m.counting_area_px2 != null ? `${Math.round(m.counting_area_px2).toLocaleString()} px²` : '–'}</div>
      <div className={`notice small ${complete ? '' : 'warn'}`} data-testid="conc-inputs">
        {complete
          ? 'All required inputs are present; the server decides whether a concentration is exported.'
          : `Concentration will not be computed. Still missing: ${missing.join(', ')}.`}
        <div className="muted">
          Required: formula confirmation, sample volume, concentration factor, counting area (from regions), total sample area, confirmed channel mapping. Pixel size is not required (areas are in px²).
        </div>
      </div>
      <div className="small muted" data-testid="concentration-status">
        The exported slide CSV states in <span className="mono">concentration_status</span> whether the estimate was computed or which inputs were missing.
      </div>
      <button className="btn primary" disabled={!dirty || busy || locked} onClick={() => void save()} data-testid="slide-save">
        {busy ? 'Saving…' : 'Save slide metadata'}
      </button>
    </section>
  );
}
