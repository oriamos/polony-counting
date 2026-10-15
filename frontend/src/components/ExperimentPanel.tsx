import { useEffect, useRef, useState } from 'react';
import { api, ApiError, batchLabExport, experimentCsvUrl, experimentProvenanceUrl } from '../api/client';
import type { Analysis, EngineChoice, ExperimentImportResult, ExperimentSlide, ExperimentView, GridSpec, RunSlide } from '../api/types';
import {
  dropSlide,
  experimentGrid,
  fieldToggles,
  gridExceeds,
  finalizePlan,
  isJobActive,
  JOB_STATUS_LABEL,
  moveSlide,
  nextUnreviewed,
  overwriteWarning,
  planApplyAll,
  runProgress,
  specOf,
  unreviewedSlides,
} from '../lib/experiment';
import { storedGridCells } from '../lib/labgrid';
import { useApp } from '../state/store';
import { EngineSelect, runSettings } from './ExperimentsList';
import { StatusChip } from './LeftRail';
import { Modal } from './Modal';

const errText = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Per-slide outcome of a grid action on the experiment page (kept until the page is left). */
interface GridResult {
  ok: boolean;
  text: string;
  seed?: number | null;
  exceeds?: boolean;
}

/** The experiment page: slides, grid for all, unattended run, review the next morning, finalize, one CSV. */
export function ExperimentPanel() {
  const e = useApp((s) => s.experiment);
  const loading = useApp((s) => s.experimentLoading);
  const err = useApp((s) => s.experimentError);
  const refresh = useApp((s) => s.refreshExperiment);
  const [over, setOver] = useState(false);
  const [gridResults, setGridResults] = useState<Record<string, GridResult>>({});
  const importRef = useRef<(files: File[]) => void>(() => undefined);
  useEffect(() => setGridResults({}), [e?.id]);

  if (loading && !e)
    return (
      <main className="center expage-wrap">
        <div className="empty big">
          <span className="spinner" /> Loading experiment…
        </div>
      </main>
    );
  if (!e)
    return (
      <main className="center expage-wrap">
        <div className="empty big">
          <div className="notice err" role="alert">
            Could not open the experiment: {err ?? 'not found'}
            <button className="btn sm" onClick={() => void refresh()}>
              Retry
            </button>
          </div>
        </div>
      </main>
    );
  const hasFiles = (ev: React.DragEvent) => Array.from(ev.dataTransfer.types).includes('Files');
  return (
    <main
      className={`center expage-wrap ${over ? 'drop-over' : ''}`}
      data-testid="experiment-page"
      onDragOver={(ev) => {
        if (!hasFiles(ev)) return;
        ev.preventDefault();
        setOver(true);
      }}
      onDragLeave={(ev) => {
        if (ev.currentTarget === ev.target) setOver(false);
      }}
      onDrop={(ev) => {
        if (!hasFiles(ev)) return;
        ev.preventDefault();
        setOver(false);
        importRef.current(Array.from(ev.dataTransfer.files));
      }}
    >
      <div className="expage">
        <ExpHeader e={e} />
        {err && (
          <div className="notice err small" role="alert">
            {err}
          </div>
        )}
        <RunBox e={e} />
        <div className="exp-cols">
          <ImportBox e={e} registerImport={(f) => (importRef.current = f)} />
          <GridBox e={e} results={gridResults} setResults={setGridResults} />
        </div>
        <SlideTable e={e} gridResults={gridResults} />
        <ExportBox e={e} />
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------------------------------- header
function ExpHeader({ e }: { e: ExperimentView }) {
  const setExperiment = useApp((s) => s.setExperiment);
  const toast = useApp((s) => s.toast);
  const [name, setName] = useState(e.name);
  const [notes, setNotes] = useState(e.notes);
  const [deleting, setDeleting] = useState(false);
  useEffect(() => {
    setName(e.name);
    setNotes(e.notes);
  }, [e.id, e.name, e.notes]);
  const save = async (body: Parameters<typeof api.updateExperiment>[1]) => {
    try {
      setExperiment(await api.updateExperiment(e.id, body));
      void useApp.getState().refreshExperiments();
    } catch (err) {
      toast('error', `experiment: ${errText(err)}`);
      setName(e.name);
      setNotes(e.notes);
    }
  };
  const engine: EngineChoice = (e.default_detection.engine as EngineChoice | undefined) ?? 'bigfish';
  const sf = e.default_detection.spotiflow;
  const scale = e.default_detection.red?.threshold_scale;
  return (
    <section className="exp-head card">
      <div className="row between">
        <input
          className="input exp-title"
          value={name}
          aria-label="Experiment name"
          data-testid="exp-title"
          onChange={(ev) => setName(ev.target.value)}
          onBlur={() => name.trim() !== e.name && void save({ name: name.trim() })}
          onKeyDown={(ev) => ev.key === 'Enter' && (ev.target as HTMLInputElement).blur()}
        />
        <span className="row">
          {e.status === 'running' && <span className="chip st-detecting">running</span>}
          <button className="btn sm danger" onClick={() => setDeleting(true)} data-testid="exp-delete">
            Delete…
          </button>
        </span>
      </div>
      <textarea
        className="input small"
        rows={1}
        placeholder="Notes (sample, phage, date…)"
        value={notes}
        data-testid="exp-notes-edit"
        onChange={(ev) => setNotes(ev.target.value)}
        onBlur={() => notes !== e.notes && void save({ notes })}
      />
      <div className="row wrap small">
        <span className="muted">Run defaults:</span>
        <EngineSelect
          value={engine}
          testid="exp-default-engine"
          disabled={e.status === 'running'}
          onChange={(eng) => void save({ default_detection: { ...runSettings(e.default_detection, eng), preview_window: null } })}
        />
        <span className="muted">
          {sf && engine !== 'bigfish' ? `Spotiflow ${sf.model_name}, 1/${sf.downscale} scale, ${sf.normalizer} normalisation · ` : ''}
          Big-FISH threshold {scale == null || scale === 1 ? 'auto' : `auto × ${scale}`} · applies to the next run
        </span>
      </div>
      {deleting && <DeleteDialog e={e} onClose={() => setDeleting(false)} />}
    </section>
  );
}

function DeleteDialog({ e, onClose }: { e: ExperimentView; onClose: () => void }) {
  const openExp = useApp((s) => s.openExperiment);
  const refreshList = useApp((s) => s.refreshList);
  const refreshExps = useApp((s) => s.refreshExperiments);
  const toast = useApp((s) => s.toast);
  const [busy, setBusy] = useState(false);
  const [reasons, setReasons] = useState<string[]>([]);
  const names = new Map(e.slides.map((s) => [s.id, s.name]));
  const go = async (withSlides: boolean) => {
    setBusy(true);
    setReasons([]);
    try {
      const r = await api.deleteExperiment(e.id, withSlides);
      toast('success', withSlides ? `Experiment and ${r.deleted_slides.length} slide(s) deleted` : `Experiment deleted; ${r.detached_slides.length} slide(s) kept as loose analyses`);
      onClose();
      await openExp(null);
      await Promise.all([refreshExps(), refreshList()]);
    } catch (err) {
      const lines = [errText(err)];
      const ids = (err instanceof ApiError ? (err.detail?.analysis_ids as string[] | undefined) : undefined) ?? [];
      if (ids.length) lines.push(`Slides: ${ids.map((i) => names.get(i) ?? i).join(', ')}`);
      setReasons(lines);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title={`Delete experiment “${e.name}”?`} onClose={onClose}>
      <div className="col">
        <p className="small">
          <b>Keep slides</b>: the experiment is removed, its {e.slides.length} slide(s) stay as loose analyses (nothing is lost). <b>Delete slides too</b>: the slide analyses
          are deleted as well; refused while a slide is detecting or finalized.
        </p>
        {reasons.length > 0 && (
          <div className="notice err small" role="alert" data-testid="exp-delete-refused">
            Refused: {reasons.map((r, i) => <div key={i}>{r}</div>)}
          </div>
        )}
        <div className="row">
          <button className="btn danger" disabled={busy} onClick={() => void go(false)} data-testid="exp-delete-keep">
            Delete, keep slides
          </button>
          <button className="btn danger" disabled={busy} onClick={() => void go(true)} data-testid="exp-delete-all">
            Delete slides too
          </button>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------------------------------- run
function RunBox({ e }: { e: ExperimentView }) {
  const run = useApp((s) => s.experimentRun);
  const start = useApp((s) => s.runExperiment);
  const cancel = useApp((s) => s.cancelExperimentRun);
  const next = useApp((s) => s.openNextUnreviewed);
  const p = runProgress(run);
  const b = run?.batch ?? null;
  const active = !!b && isJobActive(b.status);
  const nUnrev = unreviewedSlides(e.slides).length;
  const noSlides = e.slides.length === 0;
  return (
    <section className="card runbox" data-testid="exp-run">
      <div className="row wrap">
        <button className="btn primary" disabled={active || noSlides} onClick={() => void start({})} data-testid="exp-run-all" title="Detect every slide that is not finalized, one after another">
          ▶ Run all slides
        </button>
        <button className="btn" disabled={active || noSlides} onClick={() => void start({ only_missing: true })} data-testid="exp-run-missing" title="Skip slides that already have a detection result">
          Run missing only
        </button>
        {active && (
          <button className="btn danger" onClick={() => void cancel()} data-testid="exp-cancel">
            Cancel run
          </button>
        )}
        <span className="spacer" />
        <button className="btn" disabled={nUnrev === 0} onClick={() => void next()} data-testid="exp-next-unreviewed" title="Open the first slide whose counting regions are not all approved (key ])">
          Next unreviewed <kbd>]</kbd> <span className="muted">({nUnrev})</span>
        </button>
      </div>
      {active && <div className="muted small">The run continues on the server when you close this tab or the laptop lid (as long as the app keeps running). Cancel stops after the slide being detected.</div>}
      {p && b && (
        <div className="job" data-testid="exp-progress" data-status={b.status}>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p.fraction * 100)}>
            <div className={`bar ${b.status}`} style={{ width: `${Math.round(p.fraction * 100)}%` }} />
          </div>
          <div className="small">
            <strong data-testid="exp-batch-status">{JOB_STATUS_LABEL[b.status] ?? b.status}</strong> · <span data-testid="exp-progress-label">{p.label}</span>
          </div>
          {b.message && <div className="muted small mono">{b.message}</div>}
          {b.error && (
            <div className="notice err small" role="alert">
              {b.error}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------------------------------- import
function ImportBox({ e, registerImport }: { e: ExperimentView; registerImport: (f: (files: File[]) => void) => void }) {
  const setExperiment = useApp((s) => s.setExperiment);
  const refreshList = useApp((s) => s.refreshList);
  const refreshExps = useApp((s) => s.refreshExperiments);
  const toast = useApp((s) => s.toast);
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ExperimentImportResult | null>(null);
  const [mapBusy, setMapBusy] = useState(false);
  const doImport = async (files: File[]) => {
    if (!files.length || busy) return;
    setBusy(true);
    setResult(null);
    try {
      const r = await api.importToExperiment(e.id, files);
      setResult(r);
      toast(r.errors.length ? 'info' : 'success', `${r.created.length} slide(s) imported${r.errors.length ? `, ${r.errors.length} file(s) refused` : ''}`);
    } catch (err) {
      const errors = (err instanceof ApiError ? (err.detail?.errors as ExperimentImportResult['errors'] | undefined) : undefined) ?? [];
      setResult({ created: [], errors: errors.length ? errors : [{ file: '', code: err instanceof ApiError ? err.code : 'error', message: errText(err) }] });
    } finally {
      setBusy(false);
      if (input.current) input.current.value = '';
      try {
        setExperiment(await api.getExperiment(e.id));
      } catch {
        /* shown on next refresh */
      }
      void refreshList();
      void refreshExps();
    }
  };
  registerImport((files) => void doImport(files));

  // channel mapping: the import copies a confirmed mapping only from slides that existed before; offer the same
  // copy for the rest after the user has confirmed one slide (colours are never inferred)
  const tpl = e.slides.find((s) => s.mapping_confirmed && s.image);
  const unmapped = e.slides.filter((s) => !s.mapping_confirmed && s.image);
  const same = tpl ? unmapped.filter((s) => s.image!.kind === tpl.image!.kind && s.image!.dtype === tpl.image!.dtype && s.image!.n_channels === tpl.image!.n_channels) : [];
  const copyMapping = async () => {
    if (!tpl) return;
    setMapBusy(true);
    try {
      const a = await api.getAnalysis(tpl.id);
      const mapping = (a.image?.channels ?? []).map((c) => ({ index: c.index, role: (c.role ?? 'ignore') as 'red' | 'green' | 'ignore' }));
      const desc = (a.image?.channels ?? []).filter((c) => c.role !== 'ignore').map((c) => `${c.role} = channel ${c.index} (page ${c.page}${c.label_hint ? `, ${c.label_hint}` : ''})`).join(', ');
      if (!confirm(`Apply the confirmed mapping of “${tpl.name}” (${desc}) to ${same.length} slide(s) with the same file layout?\n\nOnly do this if every file was scanned the same way.`)) return;
      const bad: string[] = [];
      for (const s of same) {
        try {
          await api.setChannels(s.id, mapping);
        } catch (err) {
          bad.push(`${s.name}: ${errText(err)}`);
        }
      }
      if (bad.length) toast('error', `mapping refused for ${bad.join('; ')}`);
      else toast('success', `Channel mapping copied to ${same.length} slide(s)`);
      setExperiment(await api.getExperiment(e.id));
      void refreshList();
    } catch (err) {
      toast('error', `mapping: ${errText(err)}`);
    } finally {
      setMapBusy(false);
    }
  };
  return (
    <section className="card" data-testid="exp-import">
      <h3>Import slides</h3>
      <div className="drop small" role="button" tabIndex={0} onClick={() => input.current?.click()} onKeyDown={(ev) => (ev.key === 'Enter' || ev.key === ' ') && input.current?.click()} data-testid="exp-dropzone">
        <strong>{busy ? 'Importing…' : 'Drop TIFFs anywhere on this page, or click to choose'}</strong>
        <span className="muted">One 2-channel TIFF (or RGB composite) per slide, appended in file order</span>
      </div>
      <input ref={input} type="file" accept=".tif,.tiff,image/tiff" multiple hidden data-testid="exp-file-input" onChange={(ev) => ev.target.files && void doImport(Array.from(ev.target.files))} />
      {result && (
        <div className="small" data-testid="exp-import-result">
          {result.created.length > 0 && (
            <div>
              Imported {result.created.length}: {result.created.map((c) => `${c.name}${c.mapping_copied_from ? ' (mapping copied)' : ''}`).join(', ')}
            </div>
          )}
          {result.errors.length > 0 && (
            <ul className="notice err small blockers" role="alert" data-testid="exp-import-errors">
              {result.errors.map((x, i) => (
                <li key={i}>
                  {x.file && <span className="mono">{x.file}: </span>}
                  {x.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {unmapped.length > 0 && (
        <div className="notice warn small" data-testid="exp-mapping-note">
          {unmapped.length} slide(s) need the red/green channel mapping.{' '}
          {tpl ? (
            same.length > 0 && (
              <button className="btn sm" disabled={mapBusy} onClick={() => void copyMapping()} data-testid="exp-copy-mapping">
                Copy the mapping of “{tpl.name}” to {same.length} slide(s)…
              </button>
            )
          ) : (
            <span>Open one slide and confirm its mapping; it can then be copied to the others.</span>
          )}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------------------------------- grid
function GridBox({ e, results, setResults }: { e: ExperimentView; results: Record<string, GridResult>; setResults: (f: (r: Record<string, GridResult>) => Record<string, GridResult>) => void }) {
  const setExperiment = useApp((s) => s.setExperiment);
  const toast = useApp((s) => s.toast);
  const withGrid = e.slides.filter((s) => s.has_grid);
  const dg = e.default_grid;
  const [tplId, setTplId] = useState<string>('');
  const [tpl, setTpl] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState('');
  useEffect(() => {
    // default template: the slide the stored grid came from, else the first slide with a grid; '' = stored grid
    if (tplId && withGrid.some((s) => s.id === tplId)) return;
    const fromDefault = dg?.template_id && withGrid.some((s) => s.id === dg.template_id) ? dg.template_id : null;
    setTplId(fromDefault ?? (dg ? '' : withGrid[0]?.id ?? ''));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [e.id, withGrid.map((s) => s.id).join(), dg?.template_id]);
  useEffect(() => {
    setTpl(null);
    if (!tplId) return;
    let alive = true;
    api
      .getAnalysis(tplId)
      .then((a) => alive && setTpl(a))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [tplId, withGrid.find((s) => s.id === tplId)?.updated_at]);

  // the spec to copy: the template slide's grid, or the experiment's stored grid
  let spec: GridSpec | null = null;
  let skipped: number[] = [];
  if (tplId && tpl?.slide.grid) {
    const g = experimentGrid(tpl.slide.grid, (tpl.slide.skipped_fields ?? []).map((x) => x.field_index), tpl.id);
    spec = specOf(g);
    skipped = g.skipped_fields ?? [];
  } else if (!tplId && dg) {
    spec = specOf(dg);
    skipped = dg.skipped_fields ?? [];
  }
  const running = e.status === 'running';

  const persist = async () => {
    if (!spec) return;
    try {
      setExperiment(await api.updateExperiment(e.id, { default_grid: { ...spec, skipped_fields: skipped, template_id: tplId || dg?.template_id || null } }));
    } catch (err) {
      toast('error', `could not store the experiment grid: ${errText(err)}`);
    }
  };

  /** spec -> slide, then make it skip exactly the template's skipped fields */
  const applyOne = async (id: string, body: Parameters<typeof api.putGrid>[1], skip: number[] | null) => {
    let a = await api.putGrid(id, body);
    if (skip) for (const t of fieldToggles(storedGridCells(a), skip)) a = await api.setGridField(id, t.field, t.counted);
    return a;
  };

  const apply = async (mode: 'missing' | 'all') => {
    if (!spec) return;
    const plan = planApplyAll(e.slides, spec, tplId || null, mode);
    if (!plan.targets.length) {
      toast('info', mode === 'missing' ? 'Every slide already has a grid (use “Apply to all slides” to overwrite).' : 'No other slide to change (finalized slides are never changed).');
      return;
    }
    if (mode === 'all') {
      const w = overwriteWarning(plan);
      if (w && !confirm(w)) return;
    }
    setBusy(mode);
    await persist();
    for (const t of plan.targets) {
      try {
        await applyOne(t.id, { grid: spec, replace: true }, skipped);
        const notes = [t.exceeds ? 'grid may exceed image' : t.sizeDiffers ? 'image size differs from the template' : ''].filter(Boolean);
        setResults((r) => ({ ...r, [t.id]: { ok: true, text: `grid applied${notes.length ? ` (${notes.join(', ')})` : ''}`, exceeds: t.exceeds || t.sizeDiffers } }));
      } catch (err) {
        setResults((r) => ({ ...r, [t.id]: { ok: false, text: `grid refused: ${errText(err)}` } }));
      }
    }
    if (plan.finalized.length) toast('info', `Not changed (finalized): ${plan.finalized.join(', ')}`);
    setBusy('');
    setExperiment(await api.getExperiment(e.id));
  };

  const randomiseAll = async () => {
    const targets = [...e.slides].sort((p, q) => p.order_in_experiment - q.order_in_experiment).filter((s) => s.status !== 'finalized' && (s.has_grid || spec));
    if (!targets.length) {
      toast('info', 'No slide with a grid to randomise.');
      return;
    }
    const appr = targets.filter((s) => s.n_approved > 0).length;
    const fresh = targets.filter((s) => !s.has_grid).length;
    const msg =
      `Randomise the grid position on ${targets.length} slide(s), each with its own random seed?` +
      (fresh ? `\n${fresh} slide(s) without a grid get the experiment grid first.` : '') +
      (appr ? `\nThe fields move: approvals on ${appr} slide(s) are lost.` : '');
    if (!confirm(msg)) return;
    setBusy('random');
    if (spec && fresh) await persist();
    for (const s of targets) {
      try {
        const a = s.has_grid ? await applyOne(s.id, { grid: null, randomise: true }, null) : await applyOne(s.id, { grid: spec, randomise: true }, skipped);
        const g = a.slide.grid;
        const exceeds = !!(g && a.image && gridExceeds(g, a.image.width, a.image.height));
        setResults((r) => ({ ...r, [s.id]: { ok: true, text: `randomised, seed ${g?.seed ?? '?'}${exceeds ? ' (grid larger than the image: not shifted on that axis)' : ''}`, seed: g?.seed ?? null, exceeds } }));
      } catch (err) {
        setResults((r) => ({ ...r, [s.id]: { ok: false, text: `randomise refused: ${errText(err)}` } }));
      }
    }
    setBusy('');
    setExperiment(await api.getExperiment(e.id));
  };

  const nMissing = e.slides.filter((s) => !s.has_grid && s.id !== tplId && s.status !== 'finalized').length;
  const nRes = Object.values(results);
  return (
    <section className="card" data-testid="exp-grid">
      <h3>Grid for all slides</h3>
      <label className="field-label row">
        Template
        <select className="input sm" value={tplId} onChange={(ev) => setTplId(ev.target.value)} data-testid="exp-grid-template">
          {dg && <option value="">stored experiment grid</option>}
          {withGrid.map((s) => (
            <option key={s.id} value={s.id}>
              slide {s.name}
            </option>
          ))}
          {!dg && !withGrid.length && <option value="">— no slide has a grid yet —</option>}
        </select>
      </label>
      {spec ? (
        <div className="small mono" data-testid="exp-grid-spec">
          {spec.rows} × {spec.cols} fields of {spec.cell_w} × {spec.cell_h} px at ({Math.round(spec.origin_x)}, {Math.round(spec.origin_y)}), numbered from {spec.first_index}
          {skipped.length ? ` · skipped: ${skipped.join(', ')}` : ' · all fields counted'}
        </div>
      ) : (
        <div className="muted small">
          {tplId ? 'Loading the template grid…' : 'Open a slide, generate its grid on the Regions tab (Grid generator), pick the fields to count, then come back here to copy it to every slide.'}
        </div>
      )}
      <div className="row wrap">
        <button className="btn sm primary" disabled={!spec || !!busy || running} onClick={() => void apply('missing')} data-testid="exp-grid-apply-missing" title="Copy the grid to every slide without one">
          {busy === 'missing' ? 'Applying…' : `Apply to slides without grid (${nMissing})`}
        </button>
        <button className="btn sm" disabled={!spec || !!busy || running} onClick={() => void apply('all')} data-testid="exp-grid-apply-all" title="Copy the grid to every slide, replacing existing grids">
          {busy === 'all' ? 'Applying…' : 'Apply to all slides…'}
        </button>
        <button className="btn sm" disabled={(!spec && !withGrid.length) || !!busy || running} onClick={() => void randomiseAll()} data-testid="exp-grid-randomise" title="Shift the grid of every slide to an independent random position (fresh seed per slide)">
          {busy === 'random' ? 'Randomising…' : 'Randomise on all…'}
        </button>
      </div>
      <div className="muted small">Same rows, columns, cell size, origin and skipped fields on every slide. Finalized slides are never changed.{running ? ' Not while a run is active.' : ''}</div>
      {nRes.length > 0 && (
        <div className="small" data-testid="exp-grid-results">
          {nRes.filter((r) => r.ok).length} ok, {nRes.filter((r) => !r.ok).length} refused (see the Grid column)
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------------------------------- slides
function runRowFor(runSlides: RunSlide[], s: ExperimentSlide): { status: string; error: string | null; message: string } | null {
  const r = runSlides.find((x) => x.analysis_id === s.id);
  if (r) return { status: r.status, error: r.error, message: r.status === 'skipped' ? `skipped: ${r.skip_reason ?? ''}` : r.message };
  const j = s.last_job;
  if (j) return { status: j.status, error: j.error, message: j.message };
  return null;
}

function SlideTable({ e, gridResults }: { e: ExperimentView; gridResults: Record<string, GridResult> }) {
  const run = useApp((s) => s.experimentRun);
  const start = useApp((s) => s.runExperiment);
  const openAnalysis = useApp((s) => s.openAnalysis);
  const setExperiment = useApp((s) => s.setExperiment);
  const refreshList = useApp((s) => s.refreshList);
  const toast = useApp((s) => s.toast);
  const [drag, setDrag] = useState<string | null>(null);
  const slides = [...e.slides].sort((p, q) => p.order_in_experiment - q.order_in_experiment);
  const ids = slides.map((s) => s.id);
  const active = !!run?.batch && isJobActive(run.batch.status);
  const runSlides = run?.slides ?? [];
  const tplSize = (() => {
    const id = e.default_grid?.template_id;
    const t = slides.find((s) => s.id === id)?.image;
    return t ? `${t.width}x${t.height}` : null;
  })();
  const nextId = nextUnreviewed(slides)?.id;

  const reorder = async (next: string[]) => {
    if (next.join() === ids.join()) return;
    try {
      setExperiment(await api.orderSlides(e.id, next));
    } catch (err) {
      toast('error', `reorder: ${errText(err)}`);
    }
  };
  const detach = async (s: ExperimentSlide) => {
    if (!confirm(`Remove “${s.name}” from this experiment? The analysis is kept as a loose analysis.`)) return;
    try {
      setExperiment(await api.detachSlide(e.id, s.id));
      void refreshList();
    } catch (err) {
      toast('error', `detach: ${errText(err)}`);
    }
  };

  if (!slides.length) return <section className="card empty">No slides yet: import the TIFFs of this experiment above.</section>;
  return (
    <section className="card">
      <h3>
        Slides <span className="muted small">({slides.length}; drag a row or use ▲▼ to reorder)</span>
      </h3>
      <div className="tbl-wrap">
        <table className="tbl exp-tbl" data-testid="exp-slides">
          <thead>
            <tr>
              <th>#</th>
              <th></th>
              <th>Slide</th>
              <th>Mapping</th>
              <th>Grid</th>
              <th>Run</th>
              <th className="num" title="red-only / green-only / yellow (exclusive)">
                R / G / Y
              </th>
              <th className="num" title="numbered grid fields counted">
                Fields
              </th>
              <th className="num" title="objects waiting in the review queue">
                Queue
              </th>
              <th>Review</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {slides.map((s, i) => {
              const rr = runRowFor(runSlides, s);
              const gr = gridResults[s.id];
              const failed = rr?.status === 'failed';
              const size = s.image ? `${s.image.width}x${s.image.height}` : null;
              const sizeDiff = !!(s.has_grid && tplSize && size && size !== tplSize);
              const c = s.counts;
              const detected = !!s.detection_ran_at;
              return (
                <tr
                  key={s.id}
                  className={`${failed ? 'row-failed' : ''} ${drag === s.id ? 'dragging' : ''} ${s.id === nextId ? 'row-next' : ''}`}
                  data-testid={`exp-row-${s.id}`}
                  data-name={s.name}
                  data-status={rr?.status ?? ''}
                  draggable
                  onDragStart={(ev) => {
                    setDrag(s.id);
                    ev.dataTransfer.setData('text/x-slide', s.id);
                    ev.dataTransfer.effectAllowed = 'move';
                  }}
                  onDragEnd={() => setDrag(null)}
                  onDragOver={(ev) => {
                    if (drag) ev.preventDefault();
                  }}
                  onDrop={(ev) => {
                    if (!drag) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    void reorder(dropSlide(ids, drag, s.id));
                    setDrag(null);
                  }}
                >
                  <td className="mono">
                    <span className="drag-handle" aria-hidden>
                      ⋮⋮
                    </span>
                    {i + 1}
                    <span className="updown">
                      <button className="btn sm ghost" disabled={i === 0} onClick={() => void reorder(moveSlide(ids, s.id, -1))} aria-label={`Move ${s.name} up`}>
                        ▲
                      </button>
                      <button className="btn sm ghost" disabled={i === slides.length - 1} onClick={() => void reorder(moveSlide(ids, s.id, 1))} aria-label={`Move ${s.name} down`} data-testid={`exp-down-${s.id}`}>
                        ▼
                      </button>
                    </span>
                  </td>
                  <td>
                    {s.mapping_confirmed && s.image ? (
                      <img className="exp-thumb" alt="" loading="lazy" src={`/api/analyses/${s.id}/image/composite?max_dim=56`} onError={(ev) => ((ev.target as HTMLImageElement).style.visibility = 'hidden')} />
                    ) : (
                      <span className="exp-thumb" />
                    )}
                  </td>
                  <td>
                    <button className="linklike" onClick={() => void openAnalysis(s.id)} title={size ? `${size} px` : ''}>
                      {s.name}
                    </button>
                    <div>
                      <StatusChip status={s.status} />
                    </div>
                  </td>
                  <td>{s.mapping_confirmed ? <span className="chip ok">confirmed</span> : <span className="chip warn">needed</span>}</td>
                  <td title={gr?.seed != null ? `seed ${gr.seed}` : gr?.text ?? ''} data-testid={`exp-grid-${s.id}`} data-seed={gr?.seed ?? ''}>
                    {s.has_grid ? <span className="chip ok">grid</span> : s.n_fields ? <span className="chip">{s.n_fields} fields</span> : <span className="chip warn">none</span>}
                    {gr && <div className={`small ${gr.ok ? 'muted' : 'err-text'}`}>{gr.text}</div>}
                    {!gr && sizeDiff && <div className="small warn-text">image size differs: grid may exceed image</div>}
                  </td>
                  <td>
                    {rr ? (
                      <>
                        <span className={`chip run-${rr.status}`} data-testid={`exp-run-${s.id}`}>
                          {JOB_STATUS_LABEL[rr.status as keyof typeof JOB_STATUS_LABEL] ?? rr.status.replace('_', ' ')}
                        </span>
                        {rr.error ? (
                          <div className="small err-text exp-err" data-testid={`exp-err-${s.id}`}>
                            {rr.error}
                          </div>
                        ) : rr.message && rr.message !== rr.status ? (
                          <div className="small muted">{rr.message}</div>
                        ) : null}
                      </>
                    ) : (
                      <span className="muted small">not run</span>
                    )}
                  </td>
                  <td className="num mono">{detected ? `${c.red_only} / ${c.green_only} / ${c.yellow}` : '–'}</td>
                  <td className="num mono">{s.n_fields || '–'}</td>
                  <td className="num mono">{detected ? s.review_queue : '–'}</td>
                  <td>
                    {s.status === 'finalized' ? (
                      <span className="chip st-finalized">finalized</span>
                    ) : s.approved ? (
                      <span className="chip ok">approved</span>
                    ) : (
                      <span className="small muted">
                        {s.n_approved} / {s.n_counting_regions} approved
                      </span>
                    )}
                  </td>
                  <td>
                    <div className="row nowrap">
                      <button className="btn sm" onClick={() => void openAnalysis(s.id)} data-testid={`exp-open-${s.id}`}>
                        Open
                      </button>
                      {(failed || rr?.status === 'cancelled' || rr?.status === 'not_run') && (
                        <button className="btn sm primary" disabled={active} onClick={() => void start({ analysis_ids: [s.id] })} data-testid={`exp-retry-${s.id}`}>
                          Retry
                        </button>
                      )}
                      <button className="btn sm ghost danger-hover" disabled={active} onClick={() => void detach(s)} title="Remove from the experiment (the analysis is kept)" data-testid={`exp-detach-${s.id}`}>
                        Detach
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------------------------------- finalize + export
function ExportBox({ e }: { e: ExperimentView }) {
  const setExperiment = useApp((s) => s.setExperiment);
  const openAnalysis = useApp((s) => s.openAnalysis);
  const refreshList = useApp((s) => s.refreshList);
  const refreshExps = useApp((s) => s.refreshExperiments);
  const toast = useApp((s) => s.toast);
  const [blocked, setBlocked] = useState<ExperimentSlide[]>([]);
  const [problems, setProblems] = useState<string[]>([]);
  const [busy, setBusy] = useState('');
  const nFinal = e.slides.filter((s) => s.status === 'finalized').length;
  const finalizeAll = async () => {
    setProblems([]);
    let v = e;
    try {
      v = await api.getExperiment(e.id);
      setExperiment(v);
    } catch {
      /* use what we have */
    }
    const plan = finalizePlan(v.slides);
    setBlocked(plan.blocked);
    if (plan.blocked.length) return;
    if (!plan.toFinalize.length) {
      toast('info', 'Every slide is already finalized');
      return;
    }
    if (!confirm(`Finalize ${plan.toFinalize.length} approved slide(s)? Finalized slides are locked (reopen one to edit it).`)) return;
    setBusy('final');
    const bad: string[] = [];
    for (const s of plan.toFinalize) {
      try {
        await api.finalize(s.id);
      } catch (err) {
        bad.push(`${s.name}: ${errText(err)}`);
      }
    }
    setProblems(bad);
    setBusy('');
    if (!bad.length) toast('success', `${plan.toFinalize.length} slide(s) finalized`);
    setExperiment(await api.getExperiment(e.id));
    void refreshList();
    void refreshExps();
  };
  const labExport = async () => {
    setBusy('lab');
    setProblems([]);
    try {
      const { blob, filename } = await batchLabExport({ experiment_id: e.id, style: 'counts', series: null, format: 'xlsx' });
      saveBlob(blob, filename);
      toast('success', `Lab export downloaded: ${filename}`);
    } catch (err) {
      const lines = errText(err).split('\n').filter(Boolean);
      const d = err instanceof ApiError ? err.detail : null;
      for (const k of ['blockers', 'missing', 'errors']) {
        const x = d?.[k];
        if (Array.isArray(x)) for (const y of x) { const t = typeof y === 'string' ? y : JSON.stringify(y); if (!lines.includes(t)) lines.push(t); }
      }
      setProblems(lines);
    } finally {
      setBusy('');
    }
  };
  return (
    <section className="card" data-testid="exp-export">
      <h3>Finalize and export</h3>
      <div className="row wrap">
        <button className="btn primary" disabled={!!busy || !e.slides.length} onClick={() => void finalizeAll()} data-testid="exp-finalize">
          {busy === 'final' ? 'Finalizing…' : 'Finalize experiment…'}
        </button>
        <a className="btn" href={experimentCsvUrl(e.id)} download data-testid="exp-csv">
          ⤓ Download CSV
        </a>
        <a className="btn" href={experimentProvenanceUrl(e.id)} download data-testid="exp-provenance">
          Provenance (JSON)
        </a>
        <button className="btn" disabled={!!busy || nFinal === 0} onClick={() => void labExport()} title={nFinal ? `Counts-style workbook of the ${nFinal} finalized slide(s)` : 'Finalize slides first'} data-testid="exp-lab">
          {busy === 'lab' ? 'Exporting…' : `Lab export (xlsx, ${nFinal} final)`}
        </button>
      </div>
      <div className="muted small">
        One CSV for the whole experiment: one row per slide and colour (red, green, yellow), the field columns of <span className="mono">experiment_counts.csv</span> plus experiment, status and engine. Slides
        without detection or without numbered grid fields are left out (listed in the provenance). Keep the provenance file next to the CSV.
      </div>
      {blocked.length > 0 && (
        <div className="notice warn small" role="alert" data-testid="exp-finalize-blocked">
          Not finalized: {blocked.length} slide(s) are not fully approved yet. Review them first:
          <ul className="blockers">
            {blocked.map((s) => (
              <li key={s.id}>
                <button className="linklike" onClick={() => void openAnalysis(s.id)}>
                  {s.name}
                </button>{' '}
                <span className="muted">
                  ({s.n_counting_regions ? `${s.n_approved} / ${s.n_counting_regions} regions approved` : 'no counting region'})
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {problems.length > 0 && (
        <div className="notice err small" role="alert" data-testid="exp-export-problems">
          <ul className="blockers">
            {problems.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
