import { useEffect, useRef, useState } from 'react';
import { api, exportUrl } from '../api/client';
import type { EngineChoice, EngineInfo } from '../api/types';
import { useApp } from '../state/store';
import { comparisonCsv, isTiff, newRow, runBatch, summaryCsv, type BatchRow } from '../lib/batch';

const STAGE_LABEL: Record<BatchRow['stage'], string> = {
  queued: 'queued',
  uploading: 'uploading…',
  mapping: 'mapping channels…',
  region: 'finding chamber…',
  detecting: 'detecting…',
  counting: 'counting…',
  done: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
};

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Upload many multi-channel TIFFs at once, run detection on each and download one summary CSV. */
export function BatchPanel() {
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [engine, setEngine] = useState<EngineChoice>('bigfish');
  const [chamber, setChamber] = useState(true);
  const [rim, setRim] = useState(40);
  const [compare, setCompare] = useState(true);
  const [manifest, setManifest] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const stop = useRef(false);
  const input = useRef<HTMLInputElement>(null);
  const manifestInput = useRef<HTMLInputElement>(null);
  const refresh = useApp((s) => s.refreshList);
  const open_ = useApp((s) => s.openAnalysis);

  useEffect(() => {
    if (!open || engines.length) return;
    api.engines().then(setEngines).catch(() => undefined);
  }, [open, engines.length]);
  const spotiflow = engines.find((e) => e.name === 'spotiflow');
  const hybridOk = spotiflow?.available ?? false;

  const pick = (list: FileList | null) => {
    if (!list) return;
    const all = Array.from(list);
    const tiffs = all.filter((f) => isTiff(f.name));
    setErr(tiffs.length !== all.length ? 'Only .tif / .tiff files are accepted; others were ignored.' : '');
    setFiles(tiffs);
    setRows(tiffs.map((f) => newRow(f.name)));
  };

  const start = async () => {
    setBusy(true);
    stop.current = false;
    setErr('');
    const result = await runBatch(
      files,
      { engine, autoChamber: chamber, rimPx: rim, compare, manifest },
      api,
      (i, row) => setRows((r) => r.map((x, k) => (k === i ? row : x))),
      () => stop.current,
    );
    setRows(result);
    setBusy(false);
    void refresh();
  };

  const finished = rows.length > 0 && rows.every((r) => ['done', 'failed', 'cancelled'].includes(r.stage));
  const okCount = rows.filter((r) => r.stage === 'done').length;
  return (
    <section className="rail-section batchup" data-testid="batch-upload">
      <header className="rail-head">
        <h2>Batch upload</h2>
        <button className="btn sm" onClick={() => setOpen(!open)} aria-expanded={open} data-testid="batch-toggle">
          {open ? 'Hide' : 'Open'}
        </button>
      </header>
      {open && (
        <>
          <div className="muted small">
            Many multi-channel TIFFs at once: one analysis per file, detection runs on each in turn, then you download a summary CSV. Results stay <b>DRAFT</b>: nothing is reviewed or approved.
          </div>
          <input ref={input} type="file" accept=".tif,.tiff,image/tiff" multiple hidden data-testid="batch-input" onChange={(e) => pick(e.target.files)} />
          <button className="btn" disabled={busy} onClick={() => input.current?.click()} data-testid="batch-pick">
            Choose TIFF files…
          </button>
          {files.length > 0 && (
            <>
              <label className="field-label">
                Detector
                <select className="input sm" value={engine} disabled={busy} onChange={(e) => setEngine(e.target.value as EngineChoice)} data-testid="batch-engine">
                  <option value="bigfish">Big-FISH (fast, classical)</option>
                  <option value="hybrid" disabled={!hybridOk}>
                    Hybrid: Big-FISH + Spotiflow{hybridOk ? '' : ' (AI extra not installed)'}
                  </option>
                </select>
              </label>
              <label className="chk small">
                <input type="checkbox" checked={chamber} disabled={busy} onChange={(e) => setChamber(e.target.checked)} data-testid="batch-chamber" />
                Counting region = automatic chamber, rim
                <input className="input sm" style={{ width: 54, marginLeft: 4 }} type="number" min={0} max={500} value={rim} disabled={busy || !chamber} onChange={(e) => setRim(Math.max(0, Math.min(500, Number(e.target.value) || 0)))} />
                px
              </label>
              <label className="chk small">
                <input type="checkbox" checked={compare} disabled={busy} onChange={(e) => setCompare(e.target.checked)} data-testid="batch-compare" />
                Compare with the researcher's marks embedded in each TIFF (imports her fields as regions; a field counts as examined on a page only if it has marks on that page or the manifest says so)
              </label>
              {compare && (
                <div className="small">
                  <input ref={manifestInput} type="file" accept=".json,.csv" hidden data-testid="batch-manifest-input" onChange={(e) => setManifest(e.target.files?.[0] ?? null)} />
                  <button className="btn sm" disabled={busy} onClick={() => manifestInput.current?.click()} data-testid="batch-manifest-pick">
                    Examined-fields manifest (optional)…
                  </button>{' '}
                  {manifest ? <span className="mono">{manifest.name}</span> : <span className="muted">none: only fields with marks on a page count as examined on that page</span>}
                </div>
              )}
              <label className="chk small">
                <input type="checkbox" checked={confirmed} disabled={busy} onChange={(e) => setConfirmed(e.target.checked)} data-testid="batch-confirm" />
                I confirm for all {files.length} files: TIFF page 0 = <b>red</b>, page 1 = <b>green</b>
              </label>
              <div className="row">
                <button className="btn primary" disabled={busy || !confirmed} onClick={() => void start()} data-testid="batch-run">
                  {busy ? 'Running…' : `Run ${files.length} file${files.length === 1 ? '' : 's'}`}
                </button>
                {busy && (
                  <button className="btn" onClick={() => (stop.current = true)} data-testid="batch-cancel">
                    Stop after this file
                  </button>
                )}
              </div>
              <div className="muted small">
                Colours are never inferred; the one confirmation above applies to every file. Files whose layout differs will fail and are listed below. Big-FISH needs about 1–2 s per file, Spotiflow on CPU about 20–60 s.
              </div>
            </>
          )}
          {err && (
            <div className="notice err small" role="alert">
              {err}
            </div>
          )}
          {rows.length > 0 && (
            <ul className="batch-rows" data-testid="batch-rows">
              {rows.map((r, i) => (
                <li key={`${r.file}-${i}`} className={`batch-row st-${r.stage}`}>
                  <div className="batch-name mono" title={r.file}>
                    {r.analysisId ? (
                      <button className="linklike" onClick={() => void open_(r.analysisId as string)}>
                        {r.file}
                      </button>
                    ) : (
                      r.file
                    )}
                  </div>
                  <div className="small">
                    <span className={`chip ${r.stage === 'done' ? 'st-in_review' : ''}`}>{STAGE_LABEL[r.stage]}</span>
                    {r.stage === 'detecting' && <span className="muted"> {Math.round(r.progress * 100)}%</span>}
                    {r.counts && (
                      <span className="mono">
                        {' '}
                        red-only {r.counts.red_only} · green-only {r.counts.green_only} · yellow {r.counts.yellow} · uncertain {r.counts.uncertain}
                      </span>
                    )}
                    {r.analysisId && r.stage === 'done' && (
                      <a className="small" style={{ marginLeft: 10 }} href={exportUrl(r.analysisId, 'bundle')} download>
                        bundle (DRAFT)
                      </a>
                    )}
                  </div>
                  {r.compare.length > 0 && (
                    <ul className="small mono batch-cmp">
                      {r.compare.map((c) => (
                        <li key={`${c.plane}-${c.countType}`}>
                          page {c.plane} · {c.countType}:{' '}
                          {c.eligibleExaminedFields === 0
                            ? `no examined field on this page (${c.unknownFieldsExcluded} unknown, excluded); no metrics`
                            : `app ${c.totalApp} vs ref ${c.totalReference} (${c.eligibleExaminedFields} examined = ${c.positiveMarkedFields} marked + ${c.verifiedZeroFields} zero; ${c.unknownFieldsExcluded} unknown excluded${c.conflictFieldsExcluded ? `, ${c.conflictFieldsExcluded} conflict excluded` : ''}; r ${c.pearsonR == null ? '–' : c.pearsonR.toFixed(2)}, MAE ${c.mae == null ? '–' : c.mae.toFixed(1)})`}
                        </li>
                      ))}
                    </ul>
                  )}
                  {r.stage === 'done' && r.refStatus === 'no marks' && <div className="muted small">no researcher marks in this TIFF</div>}
                  {r.error && <div className="notice err small">{r.error}</div>}
                  {r.notes.length > 0 && (
                    <details className="small muted">
                      <summary>notes ({r.notes.length})</summary>
                      {r.notes.map((n, k) => (
                        <div key={k}>{n}</div>
                      ))}
                    </details>
                  )}
                </li>
              ))}
            </ul>
          )}
          {finished && (
            <>
              <button className="btn primary" onClick={() => download(`polony_batch_summary_DRAFT.csv`, summaryCsv(rows, engine))} data-testid="batch-download">
                Download summary CSV ({okCount}/{rows.length} done)
              </button>
              {rows.some((r) => r.compare.length > 0) && (
                <button className="btn" onClick={() => download(`polony_batch_vs_researcher_DRAFT.csv`, comparisonCsv(rows))} data-testid="batch-download-compare">
                  Download comparison with researcher marks CSV
                </button>
              )}
              {compare && (
                <div className="muted small">
                  The summary counts the whole chamber; the researcher counted only her fields. Compare per field (second file), not the slide totals. Fields without evidence of being examined on a page are excluded, never counted as zero. Agreement with her marks is not accuracy.
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
