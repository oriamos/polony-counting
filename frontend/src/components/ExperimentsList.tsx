import { useEffect, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { DetectionRequest, EngineChoice, EngineInfo } from '../api/types';
import { batchRequest } from '../lib/batch';
import { experimentChip } from '../lib/experiment';
import { useApp } from '../state/store';
import { Modal } from './Modal';

/** Left rail: experiments (an ordered set of slides that run unattended and export one CSV). */
export function ExperimentsList() {
  const experiments = useApp((s) => s.experiments);
  const err = useApp((s) => s.experimentsError);
  const refresh = useApp((s) => s.refreshExperiments);
  const open = useApp((s) => s.openExperiment);
  const sel = useApp((s) => (s.mainView === 'experiment' ? s.experiment?.id : null));
  const [creating, setCreating] = useState(false);
  useEffect(() => void refresh(), [refresh]);
  return (
    <section className="rail-section" data-testid="experiments">
      <header className="rail-head">
        <h2>Experiments</h2>
        <span className="row">
          <button className="btn sm primary" onClick={() => setCreating(true)} data-testid="exp-new">
            + New experiment
          </button>
          <button className="btn sm ghost" onClick={() => void refresh()} title="Refresh experiments">
            ↻
          </button>
        </span>
      </header>
      {err && (
        <div className="notice err small" role="alert">
          Could not load experiments: {err}
        </div>
      )}
      {!err && experiments.length === 0 && (
        <div className="empty small">No experiment yet. An experiment holds all slides of one run: import them, set one grid, run overnight, export one CSV.</div>
      )}
      <ul className="alist">
        {experiments.map((e) => {
          const chip = experimentChip(e);
          return (
            <li key={e.id} className={sel === e.id ? 'active' : ''}>
              <button className="alist-main" onClick={() => void open(e.id)} data-testid={`experiment-${e.id}`}>
                <span className="alist-name">{e.name}</span>
                <span className="alist-meta">
                  <span className={`chip ${chip.cls}`} data-testid={`exp-chip-${e.id}`}>
                    {chip.label}
                  </span>
                  <span className="muted small">
                    {e.n_slides} slide{e.n_slides === 1 ? '' : 's'}
                    {e.n_slides ? ` · ${e.n_detected} detected · ${e.n_finalized} final` : ''}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {creating && <NewExperimentDialog onClose={() => setCreating(false)} />}
    </section>
  );
}

/** Detection settings without the preview window (not a run setting). */
export function runSettings(req: DetectionRequest | Partial<DetectionRequest>, engine: EngineChoice): Partial<DetectionRequest> {
  const { preview_window: _p, ...rest } = req as DetectionRequest;
  const out: Partial<DetectionRequest> = { ...rest, engine };
  if (engine === 'bigfish') delete out.spotiflow;
  else if (!out.spotiflow) out.spotiflow = batchRequest(engine).spotiflow;
  return out;
}

export function EngineSelect({ value, onChange, testid, disabled }: { value: EngineChoice; onChange: (e: EngineChoice) => void; testid?: string; disabled?: boolean }) {
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  useEffect(() => {
    api.engines().then(setEngines).catch(() => undefined);
  }, []);
  const sf = engines.find((e) => e.name === 'spotiflow');
  const ok = sf?.available ?? false;
  return (
    <select className="input sm" value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as EngineChoice)} data-testid={testid}>
      <option value="bigfish">Big-FISH (fast, classical)</option>
      <option value="hybrid" disabled={!ok && value !== 'hybrid'}>
        Hybrid: Big-FISH + Spotiflow{ok ? '' : ' (AI extra not installed)'}
      </option>
      <option value="spotiflow" disabled={!ok && value !== 'spotiflow'}>
        Spotiflow only{ok ? '' : ' (AI extra not installed)'}
      </option>
    </select>
  );
}

function NewExperimentDialog({ onClose }: { onClose: () => void }) {
  const current = useApp((s) => s.current);
  const refresh = useApp((s) => s.refreshExperiments);
  const open = useApp((s) => s.openExperiment);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  const [engine, setEngine] = useState<EngineChoice>('bigfish');
  const [fromCurrent, setFromCurrent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const submit = async () => {
    setBusy(true);
    setErr('');
    try {
      const det = fromCurrent && current ? runSettings(current.detection_request, engine) : batchRequest(engine);
      const v = await api.createExperiment({ name: name.trim(), notes: notes.trim(), default_detection: { ...det, preview_window: null } });
      await refresh();
      onClose();
      await open(v.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title="New experiment" onClose={onClose}>
      <form
        className="col"
        data-testid="exp-dialog"
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) void submit();
        }}
      >
        <label className="field-label">
          Name
          <input className="input" autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Clade run 12" data-testid="exp-name" />
        </label>
        <label className="field-label">
          Notes <span className="muted">(optional)</span>
          <textarea className="input" rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} data-testid="exp-notes" />
        </label>
        <label className="field-label">
          Detector for the unattended run
          <EngineSelect value={engine} onChange={setEngine} testid="exp-engine" />
        </label>
        {current && (
          <label className="chk small">
            <input type="checkbox" checked={fromCurrent} onChange={(e) => setFromCurrent(e.target.checked)} data-testid="exp-from-current" />
            Use the detection parameters of the open analysis “{current.name}” (thresholds, background, Spotiflow settings)
          </label>
        )}
        <div className="muted small">
          {fromCurrent && current
            ? 'Every slide is detected with these parameters unless you change the experiment defaults later.'
            : engine === 'bigfish'
              ? 'Server defaults for Big-FISH. You can change the parameters later from an open slide.'
              : 'Spotiflow at 1/4 scale with saturation-robust normalisation (the batch defaults); Big-FISH with server defaults.'}
        </div>
        {err && (
          <div className="notice err small" role="alert">
            {err}
          </div>
        )}
        <div className="row">
          <button className="btn primary" type="submit" disabled={busy || !name.trim()} data-testid="exp-create">
            {busy ? 'Creating…' : 'Create experiment'}
          </button>
          <button className="btn" type="button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  );
}
