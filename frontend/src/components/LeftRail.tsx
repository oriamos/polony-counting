import { useEffect, useRef, useState } from 'react';
import { api, ApiError, batchLabExport, type LabFormat, type LabStyle } from '../api/client';
import type { AnalysisStatus, CsvImport } from '../api/types';
import { useApp } from '../state/store';
import { OpenFromDisk } from './OpenFromDisk';
import { ExperimentsList } from './ExperimentsList';

export const STATUS_LABEL: Record<AnalysisStatus, string> = {
  created: 'created',
  needs_channel_mapping: 'needs channel mapping',
  ready: 'ready',
  detecting: 'detecting',
  in_review: 'in review',
  finalized: 'finalized',
};

export function StatusChip({ status }: { status: AnalysisStatus }) {
  return <span className={`chip st-${status}`}>{STATUS_LABEL[status] ?? status}</span>;
}

export function LeftRail() {
  return (
    <aside className="left" aria-label="Analyses">
      <ExperimentsList />
      <AnalysesList />
      <LabBatchPanel />
      <CsvPanel />
    </aside>
  );
}

function AnalysesList() {
  const analyses = useApp((s) => s.analyses);
  const loading = useApp((s) => s.listLoading);
  const err = useApp((s) => s.listError);
  const current = useApp((s) => s.current);
  const open = useApp((s) => s.openAnalysis);
  const refresh = useApp((s) => s.refreshList);
  const del = useApp((s) => s.deleteAnalysis);
  const mainView = useApp((s) => s.mainView);
  const experiments = useApp((s) => s.experiments);
  const openExp = useApp((s) => s.openExperiment);
  const expName = new Map(experiments.map((e) => [e.id, e.name]));
  return (
    <section className="rail-section">
      <header className="rail-head">
        <h2>Analyses</h2>
        <button className="btn sm ghost" onClick={() => void refresh()} title="Refresh list">
          ↻
        </button>
      </header>
      <OpenFromDisk />
      <UploadBox />
      {loading && analyses.length === 0 && <div className="empty"><span className="spinner" /> Loading…</div>}
      {err && (
        <div className="notice err" role="alert">
          Could not load analyses: {err}
          <button className="btn sm" onClick={() => void refresh()}>Retry</button>
        </div>
      )}
      {!loading && !err && analyses.length === 0 && <div className="empty">No analyses yet. Drop 1–2 TIFF files above to start.</div>}
      <ul className="alist">
        {analyses.map((a) => (
          <li key={a.id} className={current?.id === a.id && mainView === 'analysis' ? 'active' : ''}>
            <button className="alist-main" onClick={() => void open(a.id)} data-testid={`analysis-${a.id}`}>
              <span className="alist-name">{a.name}</span>
              <span className="alist-meta">
                <StatusChip status={a.status} />
                {a.counts && (
                  <span className="muted mono" title="red-only / green-only / yellow / uncertain">
                    {a.counts.red_only}/{a.counts.green_only}/{a.counts.yellow}/{a.counts.uncertain}
                  </span>
                )}
              </span>
              {a.experiment_id && (
                <span
                  className="alist-exp small"
                  role="link"
                  tabIndex={-1}
                  title="Slide of this experiment (click to open the experiment)"
                  data-testid={`analysis-exp-${a.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    void openExp(a.experiment_id as string);
                  }}
                >
                  ▤ {expName.get(a.experiment_id) ?? 'experiment'}
                  {a.order_in_experiment ? ` · #${a.order_in_experiment}` : ''}
                </span>
              )}
            </button>
            <button
              className="btn sm ghost danger-hover"
              title="Delete analysis"
              aria-label={`Delete ${a.name}`}
              onClick={() => {
                if (confirm(`Delete analysis "${a.name}"? This cannot be undone.`)) void del(a.id);
              }}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function UploadBox() {
  const [over, setOver] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const open = useApp((s) => s.openAnalysis);
  const refresh = useApp((s) => s.refreshList);

  const accept = (list: FileList | File[]) => {
    const arr = Array.from(list);
    const tiffs = arr.filter((f) => /\.tiff?$/i.test(f.name));
    if (tiffs.length !== arr.length) setErr('Only .tif / .tiff files are accepted.');
    else if (tiffs.length > 2) setErr('Select at most 2 TIFF files (one per channel, or one multi-channel file).');
    else setErr('');
    if (tiffs.length && tiffs.length <= 2) setFiles(tiffs);
  };

  const submit = async () => {
    setBusy(true);
    setErr('');
    try {
      const a = await api.createAnalysis(files, name.trim() || undefined);
      setFiles([]);
      setName('');
      await refresh();
      await open(a.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="upload">
      <div
        className={`drop ${over ? 'over' : ''}`}
        data-testid="dropzone"
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          accept(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          accept=".tif,.tiff,image/tiff"
          multiple
          hidden
          data-testid="file-input"
          onChange={(e) => e.target.files && accept(e.target.files)}
        />
        <strong>Drop TIFF(s) here</strong>
        <span className="muted">1 multi-channel file, or 2 single-channel files</span>
      </div>
      {files.length > 0 && (
        <div className="upload-pending">
          <ul>
            {files.map((f) => (
              <li key={f.name} className="mono">
                {f.name} <span className="muted">({(f.size / 1e6).toFixed(1)} MB)</span>
              </li>
            ))}
          </ul>
          <input className="input" placeholder="Analysis name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => void submit()}>
              {busy ? 'Uploading…' : 'Create analysis'}
            </button>
            <button className="btn" disabled={busy} onClick={() => setFiles([])}>
              Clear
            </button>
          </div>
          <div className="muted small">Channel colours are never inferred from file names; you will assign them next.</div>
        </div>
      )}
      {err && (
        <div className="notice err" role="alert">
          {err}
        </div>
      )}
    </div>
  );
}

function CsvPanel() {
  const imports = useApp((s) => s.csvImports);
  const refresh = useApp((s) => s.refreshCsv);
  const toast = useApp((s) => s.toast);
  const [openId, setOpenId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => void refresh(), [refresh]);

  const upload = async (f: File) => {
    setBusy(true);
    try {
      const c = await api.uploadCsv(f);
      await refresh();
      setOpenId(c.id);
    } catch (e) {
      toast('error', `CSV import: ${e instanceof ApiError ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rail-section csv">
      <header className="rail-head">
        <h2>CSV import</h2>
        <button className="btn sm" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? '…' : 'Import CSV'}
        </button>
        <input ref={input} type="file" accept=".csv,text/csv" hidden data-testid="csv-input" onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
      </header>
      <div className="muted small">
        Manual counts for comparison. Blank cells are kept distinct from 0. Column names such as T7A/T7B are never assumed to be colours.
      </div>
      {imports.length === 0 && <div className="empty small">No CSV imported.</div>}
      {imports.map((c) => (
        <div key={c.id} className="csv-item">
          <button className="csv-title" onClick={() => setOpenId(openId === c.id ? null : c.id)} aria-expanded={openId === c.id}>
            <span>{openId === c.id ? '▾' : '▸'}</span> <span className="mono">{c.filename}</span>
            <span className="muted"> · {c.rows.length} rows</span>
            {!c.color_mapping && <span className="chip warn">colour not set</span>}
          </button>
          {openId === c.id && <CsvDetail csv={c} onChange={() => void refresh()} />}
        </div>
      ))}
    </section>
  );
}

function CsvDetail({ csv, onChange }: { csv: CsvImport; onChange: () => void }) {
  const [val, setVal] = useState(csv.color_mapping ?? '');
  const [busy, setBusy] = useState(false);
  const toast = useApp((s) => s.toast);
  const maxFields = Math.max(0, ...csv.rows.map((r) => r.fields.length));
  const save = async () => {
    setBusy(true);
    try {
      await api.setCsvMapping(csv.id, val || null);
      toast('success', val ? `CSV counts assigned to colour "${val}"` : 'CSV colour cleared');
      onChange();
    } catch (e) {
      toast('error', `CSV mapping: ${e instanceof ApiError ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };
  const metaKeys = Array.from(new Set(csv.rows.flatMap((r) => Object.keys(r.meta))));
  return (
    <div className="csv-detail">
      <div className="csv-map">
        <label className="field-label" htmlFor={`csvmap-${csv.id}`}>
          These counts are of which colour? <span className="muted">(you must choose)</span>
        </label>
        <div className="row">
          <select id={`csvmap-${csv.id}`} className="input" value={val} onChange={(e) => setVal(e.target.value)} data-testid="csv-colour">
            <option value="">— not chosen —</option>
            <option value="red">red</option>
            <option value="green">green</option>
            <option value="yellow">yellow</option>
          </select>
          <button className="btn primary" disabled={busy || val === (csv.color_mapping ?? '')} onClick={() => void save()}>
            Save
          </button>
        </div>
        {!csv.color_mapping && <div className="notice warn small">No colour mapping chosen: values are shown but not interpreted.</div>}
      </div>
      {csv.notes.map((n, i) => (
        <div key={i} className="notice small">
          {n}
        </div>
      ))}
      <div className="csv-scroll">
        <table className="tbl csv-tbl">
          <thead>
            <tr>
              <th>slide</th>
              {metaKeys.map((k) => (
                <th key={k}>{k}</th>
              ))}
              {Array.from({ length: maxFields }, (_, i) => (
                <th key={i} className="num">
                  f{i + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {csv.rows.map((r, ri) => (
              <tr key={ri}>
                <td className="mono">{r.slide_id}</td>
                {metaKeys.map((k) => (
                  <td key={k} className="mono">
                    {r.meta[k] ?? ''}
                  </td>
                ))}
                {Array.from({ length: maxFields }, (_, i) => {
                  const v = r.fields[i];
                  return v == null ? (
                    <td key={i} className="num cell-blank" title={`blank (not measured)${r.raw_fields[i] ? ` – raw "${r.raw_fields[i]}"` : ''}`}>
                      blank
                    </td>
                  ) : (
                    <td key={i} className="num cell-val" title="measured value">
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="muted small">
        <span className="cell-blank inline">blank</span> = not measured (excluded from totals) · <span className="cell-val inline">0</span> = measured zero
      </div>
    </div>
  );
}

function LabBatchPanel() {
  const analyses = useApp((s) => s.analyses);
  const current = useApp((s) => s.current);
  const toast = useApp((s) => s.toast);
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState<string[]>([]);
  const [style, setStyle] = useState<LabStyle>('t7');
  const [series, setSeries] = useState('');
  const [format, setFormat] = useState<LabFormat>('xlsx');
  const [busy, setBusy] = useState(false);
  const [blockers, setBlockers] = useState<string[]>([]);
  const finalIds = new Set(analyses.filter((a) => a.status === 'finalized').map((a) => a.id));
  const chosen = sel.filter((id) => finalIds.has(id));
  const suggestions = (current?.slide.lab_series ?? []).map((q) => q.name);
  const needSeries = style === 't7';
  const canGo = chosen.length > 0 && !busy && (!needSeries || series.trim() !== '');
  const toggle = (id: string) => setSel((x) => (x.includes(id) ? x.filter((y) => y !== id) : [...x, id]));
  const go = async () => {
    setBusy(true);
    setBlockers([]);
    try {
      const { blob, filename } = await batchLabExport({ analysis_ids: chosen, style, series: series.trim() || null, format: style === 'counts' ? 'xlsx' : format });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      toast('success', `Lab export downloaded: ${filename}`);
    } catch (e) {
      const err = e as ApiError;
      const lines = err.message.split('\n').filter(Boolean);
      const d = err.detail as Record<string, unknown> | null;
      for (const k of ['blockers', 'missing', 'errors']) {
        const v = d?.[k];
        if (Array.isArray(v)) for (const x of v) { const t = typeof x === 'string' ? x : JSON.stringify(x); if (!lines.includes(t)) lines.push(t); }
      }
      setBlockers(lines);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="rail-section labbatch" data-testid="lab-batch">
      <header className="rail-head">
        <h2>Lab export (batch)</h2>
        <button className="btn sm" onClick={() => setOpen(!open)} aria-expanded={open} data-testid="lab-batch-toggle">
          {open ? 'Hide' : 'Open'}
        </button>
      </header>
      {open && (
        <>
          <div className="muted small">Only finalized analyses can be exported. Drafts are disabled.</div>
          <ul className="batch-list">
            {analyses.map((a) => {
              const fin = a.status === 'finalized';
              return (
                <li key={a.id}>
                  <label className="chk small" title={fin ? '' : 'Draft: finalize this analysis first'}>
                    <input type="checkbox" disabled={!fin} checked={fin && sel.includes(a.id)} onChange={() => toggle(a.id)} data-testid={`lab-sel-${a.id}`} />
                    <span>{a.name}</span> <StatusChip status={a.status} />
                  </label>
                </li>
              );
            })}
          </ul>
          <label className="field-label">
            Style
            <select className="input sm" value={style} onChange={(e) => setStyle(e.target.value as LabStyle)} data-testid="lab-style">
              <option value="t7">T7-style per series</option>
              <option value="counts">Counts-style workbook</option>
            </select>
          </label>
          <label className="field-label">
            Series {needSeries ? '(required)' : '(blank = all series)'}
            <input className="input sm" list="lab-series-list" value={series} onChange={(e) => setSeries(e.target.value)} placeholder="e.g. T7A" data-testid="lab-series" />
            <datalist id="lab-series-list">
              {suggestions.map((n) => (
                <option key={n} value={n} />
              ))}
            </datalist>
          </label>
          <label className="field-label">
            Format
            <select className="input sm" value={style === 'counts' ? 'xlsx' : format} disabled={style === 'counts'} onChange={(e) => setFormat(e.target.value as LabFormat)} data-testid="lab-format">
              <option value="xlsx">XLSX</option>
              <option value="csv">CSV</option>
            </select>
          </label>
          <button className="btn primary" disabled={!canGo} onClick={() => void go()} data-testid="lab-download">
            {busy ? 'Exporting…' : `Download (${chosen.length})`}
          </button>
          {blockers.length > 0 && (
            <div className="notice err small" role="alert" data-testid="lab-blockers">
              Lab export refused:
              <ul className="blockers">
                {blockers.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}
