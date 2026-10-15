import { useEffect, useState } from 'react';
import { api, exportUrl, labExportUrl } from '../api/client';
import type { LabCountType, SlideMeta } from '../api/types';
import { fieldIssues, describeIssues, MAX_FIELD } from '../lib/labgrid';
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
      {final && <LabExportButtons />}
      <SlideForm />
    </div>
  );
}

function LabExportButtons() {
  const a = useApp((s) => s.current)!;
  const series = a.slide.lab_series ?? [];
  return (
    <section data-testid="lab-export-panel">
      <h3>Lab export</h3>
      <div className="muted small">
        Lab layout (one table per series; blank = field not counted, 0 = counted zero). Requires the lab metadata, series mapping and field numbers below; the server lists anything missing.
      </div>
      {series.length === 0 && <div className="notice warn small">No count series defined yet: add one in the lab metadata form below and save.</div>}
      <div className="exports">
        {series.map((sr) => (
          <div key={sr.name} className="row wrap">
            <a className="btn grow" href={labExportUrl(a.id, sr.name, 'xlsx')} download data-testid={`lab-export-${sr.name}-xlsx`}>
              Lab export (T7-style XLSX) · {sr.name}
            </a>
            <a className="btn" href={labExportUrl(a.id, sr.name, 'csv')} download data-testid={`lab-export-${sr.name}-csv`}>
              CSV
            </a>
          </div>
        ))}
      </div>
    </section>
  );
}

const COUNT_TYPES: { v: LabCountType; label: string }[] = [
  { v: 'red_only', label: 'red-only' },
  { v: 'green_only', label: 'green-only' },
  { v: 'yellow', label: 'yellow (co-localized)' },
  { v: 'red_positive', label: 'red-positive (red-only + yellow)' },
  { v: 'green_positive', label: 'green-positive (green-only + yellow)' },
];

type SeriesDraft = { name: string; count_type: LabCountType | ''; sheet_title: string };
const DATE_RE = /^(0[1-9]|[12]\d|3[01])\/(0[1-9]|1[0-2])\/\d{4}$/;
const dateOk = (v: string | null | undefined) => !!v && DATE_RE.test(v);

function Chips({ chips, disabled }: { chips: { label: string; fill: () => void; testid?: string }[]; disabled: boolean }) {
  return (
    <span className="chips">
      {chips.map((c) => (
        <button key={c.label} type="button" className="btn sm ghost chipbtn" disabled={disabled} onClick={c.fill} data-testid={c.testid}>
          {c.label}
        </button>
      ))}
    </span>
  );
}

function SlideForm() {
  const a = useApp((s) => s.current)!;
  const setAnalysis = useApp((s) => s.setAnalysis);
  const toast = useApp((s) => s.toast);
  const [m, setM] = useState<SlideMeta>(a.slide);
  const toDraft = (x: SlideMeta): SeriesDraft[] => (x.lab_series ?? []).map((q) => ({ name: q.name, count_type: q.count_type, sheet_title: q.sheet_title ?? '' }));
  const [series, setSeries] = useState<SeriesDraft[]>(toDraft(a.slide));
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setM(a.slide);
    setSeries(toDraft(a.slide));
  }, [a.id, a.slide]);
  const locked = a.status === 'finalized';
  const seriesErr: string[] = [];
  const names = series.map((q) => q.name.trim());
  if (series.some((q) => !q.name.trim())) seriesErr.push('every series needs a name');
  if (new Set(names).size !== names.length) seriesErr.push('series names must be unique');
  if (series.some((q) => !q.count_type)) seriesErr.push('every series needs a count type (required, never inferred)');
  const merged: SlideMeta = {
    ...m,
    lab_series: series.filter((q) => q.count_type).map((q) => ({ name: q.name.trim(), count_type: q.count_type as LabCountType, sheet_title: q.sheet_title.trim() || null })),
  };
  const dirty = JSON.stringify(merged) !== JSON.stringify(a.slide) || series.length !== (a.slide.lab_series ?? []).length;
  const mapped = !!a.image && a.image.channels.length > 0 && a.image.channels.every((c) => c.role_confirmed);
  const fields = a.regions.filter((r) => r.kind === 'counting' && r.field_index != null);
  const fIssues = fieldIssues(fields.map((r) => r.field_index));
  const required: { ok: boolean; text: string }[] = [
    { ok: !!m.slide_id, text: 'slide ID' },
    { ok: dateOk(m.sampling_date), text: 'sampling date (dd/mm/yyyy)' },
    { ok: dateOk(m.polony_date), text: 'polony date (dd/mm/yyyy)' },
    { ok: m.qc === 0 || m.qc === 1, text: 'QC (0/1)' },
    { ok: (m.sample_volume_ul ?? 0) > 0, text: 'sample volume' },
    { ok: (m.concentration_factor ?? 0) > 0, text: 'concentration factor / dilution' },
    { ok: (m.field_area ?? 0) > 0, text: 'field area' },
    { ok: (m.grid_area ?? 0) > 0, text: 'grid area' },
    { ok: merged.lab_series!.length > 0 && seriesErr.length === 0, text: 'series mapping with count types' },
    { ok: fields.length > 0 && !fIssues.duplicates.length && !fIssues.tooHigh.length && !fIssues.invalid.length, text: `field regions with unique numbers 1-${MAX_FIELD}` },
    { ok: !!m.concentration_formula_confirmed, text: 'geometry / series confirmation' },
    { ok: mapped, text: 'confirmed channel mapping' },
  ];
  const missing = required.filter((r) => !r.ok).map((r) => r.text);
  const complete = missing.length === 0;
  const save = async () => {
    setBusy(true);
    try {
      setAnalysis(await api.putSettings(a.id, { slide: merged }));
      toast('success', 'Slide metadata saved');
    } catch (e) {
      toast('error', `slide metadata: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  const txt = (k: 'slide_id' | 'sampling_date' | 'polony_date', testid: string, placeholder?: string) => (
    <input
      className="input sm"
      value={m[k] ?? ''}
      placeholder={placeholder}
      disabled={locked}
      onChange={(e) => setM({ ...m, [k]: e.target.value || null })}
      data-testid={testid}
    />
  );
  const upd = (i: number, p: Partial<SeriesDraft>) => setSeries(series.map((q, n) => (n === i ? { ...q, ...p } : q)));
  return (
    <section className="slideform" data-testid="slide-form">
      <h3>Lab metadata</h3>
      <div className="muted small">Everything is empty by default and nothing is assumed. Chips only fill a field; they do not confirm it.</div>
      <div className="form-grid">
        <label>
          Slide ID <span className="unit">e.g. AI_A</span>
          {txt('slide_id', 'slide-id')}
        </label>
        <label>
          QC <span className="unit">0/1, meaning unconfirmed</span>
          <select
            className="input sm"
            value={m.qc ?? ''}
            disabled={locked}
            onChange={(e) => setM({ ...m, qc: e.target.value === '' ? null : Number(e.target.value) })}
            data-testid="slide-qc"
          >
            <option value="">— not set —</option>
            <option value="0">0</option>
            <option value="1">1</option>
          </select>
        </label>
        <label>
          Sampling date <span className="unit">dd/mm/yyyy</span>
          {txt('sampling_date', 'slide-sampling-date', 'dd/mm/yyyy')}
          {m.sampling_date && !dateOk(m.sampling_date) && <span className="warn-text">use dd/mm/yyyy</span>}
        </label>
        <label>
          Polony date <span className="unit">dd/mm/yyyy</span>
          {txt('polony_date', 'slide-polony-date', 'dd/mm/yyyy')}
          {m.polony_date && !dateOk(m.polony_date) && <span className="warn-text">use dd/mm/yyyy</span>}
        </label>
        <label className="span2">
          Sample volume <span className="unit">µL (no default)</span>
          <NumInput value={m.sample_volume_ul} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, sample_volume_ul: v })} testid="slide-volume" />
          <Chips
            disabled={locked}
            chips={[
              { label: 'use 2.5 (CSV)', fill: () => setM({ ...m, sample_volume_ul: 2.5 }), testid: 'chip-vol-2.5' },
              { label: 'use 2.5056 = 5.4×11.6/25 (Counts workbook)', fill: () => setM({ ...m, sample_volume_ul: 2.5056 }), testid: 'chip-vol-2.5056' },
            ]}
          />
        </label>
        <label className="span2">
          Concentration factor / dilution <span className="unit">× (applied only in the Counts-style sheet)</span>
          <NumInput value={m.concentration_factor} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, concentration_factor: v })} testid="slide-factor" />
        </label>
        <label>
          Field area <span className="unit">lab units</span>
          <NumInput value={m.field_area ?? null} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, field_area: v })} testid="slide-field-area" />
        </label>
        <label>
          Grid area <span className="unit">lab units</span>
          <NumInput value={m.grid_area ?? null} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, grid_area: v })} testid="slide-grid-area" />
        </label>
        <div className="span2">
          <Chips
            disabled={locked}
            chips={[{ label: 'template: 2.9 / 0.1', fill: () => setM({ ...m, field_area: 2.9, grid_area: 0.1 }), testid: 'chip-areas' }]}
          />
          <span className="small muted"> field/grid = {(m.field_area ?? 0) > 0 && (m.grid_area ?? 0) > 0 ? +((m.field_area as number) / (m.grid_area as number)).toPrecision(6) : '–'}</span>
        </div>
        <label className="span2">
          Efficiency denominator <span className="unit">optional (template 1500)</span>
          <NumInput value={m.efficiency_input ?? null} nullable min={0} disabled={locked} onChange={(v) => setM({ ...m, efficiency_input: v })} testid="slide-efficiency" />
          <Chips disabled={locked} chips={[{ label: 'template: 1500', fill: () => setM({ ...m, efficiency_input: 1500 }) }]} />
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
        <label>
          Sampled-area note
          <input className="input sm" value={m.sampled_area_note ?? ''} disabled={locked} onChange={(e) => setM({ ...m, sampled_area_note: e.target.value || null })} />
        </label>
      </div>

      <h4>Series mapping</h4>
      <div className="muted small" data-testid="series-note">
        One lab table per series (e.g. T7A, T7B). Each series needs a count type, <strong>never inferred from names or colours</strong>. Sheet title is used in the Counts-style workbook.
      </div>
      {series.map((q, i) => (
        <div key={i} className="seriesrow" data-testid={`series-${i}`}>
          <label>
            Series name <span className="unit">e.g. T7A</span>
            <input className="input sm" value={q.name} disabled={locked} onChange={(e) => upd(i, { name: e.target.value })} data-testid={`series-name-${i}`} />
          </label>
          <label>
            Sheet title <span className="unit">optional, e.g. Clade A</span>
            <input className="input sm" value={q.sheet_title} disabled={locked} onChange={(e) => upd(i, { sheet_title: e.target.value })} data-testid={`series-title-${i}`} />
          </label>
          <label className="span2">
            Count type <span className="unit">required</span>
            <select
              className={`input sm ${q.count_type ? '' : 'invalid'}`}
              value={q.count_type}
              disabled={locked}
              onChange={(e) => upd(i, { count_type: e.target.value as LabCountType | '' })}
              data-testid={`series-type-${i}`}
            >
              <option value="">— choose (required) —</option>
              {COUNT_TYPES.map((t) => (
                <option key={t.v} value={t.v}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <button className="btn sm ghost danger-hover" disabled={locked} onClick={() => setSeries(series.filter((_, n) => n !== i))}>
            Remove series
          </button>
        </div>
      ))}
      <button className="btn sm" disabled={locked} onClick={() => setSeries([...series, { name: '', count_type: '', sheet_title: '' }])} data-testid="series-add">
        + Add series
      </button>
      {seriesErr.length > 0 && (
        <div className="small warn-text" data-testid="series-errors">
          {seriesErr.join('; ')}.
        </div>
      )}

      <div className="notice warn small" data-testid="formula-note">
        <strong>Lab formula (T7 sheet):</strong>{' '}
        <span className="mono">abundance = mean(field counts) × (field area / grid area) ÷ sample volume × 1000</span>. Blank fields are ignored by the mean; zero counts are included. The concentration factor is applied only in the
        Counts-style sheet. The server computes everything; this page never calculates it.
        <label className="chk">
          <input
            type="checkbox"
            checked={!!m.concentration_formula_confirmed}
            disabled={locked}
            onChange={(e) => setM({ ...m, concentration_formula_confirmed: e.target.checked })}
            data-testid="formula-confirmed"
          />{' '}
          I confirm field geometry and series mapping match the lab convention
        </label>
      </div>
      <div className="small mono muted">
        lab fields: {fields.length} {fields.length ? `(${fields.map((r) => r.field_index).sort((x, y) => (x ?? 0) - (y ?? 0)).join(', ')})` : '– define them in the Regions tab'}
      </div>
      {describeIssues(fIssues).length > 0 && <div className="small warn-text">Field numbering: {describeIssues(fIssues).join('; ')}.</div>}
      <div className={`notice small ${complete ? '' : 'warn'}`} data-testid="conc-inputs">
        {complete ? 'All inputs for the lab export are present; the server decides and lists anything it still finds missing.' : `Lab export not yet possible. Still missing: ${missing.join(', ')}.`}
      </div>
      <button className="btn primary" disabled={!dirty || busy || locked || seriesErr.length > 0} onClick={() => void save()} data-testid="slide-save">
        {busy ? 'Saving…' : 'Save lab metadata'}
      </button>
    </section>
  );
}
