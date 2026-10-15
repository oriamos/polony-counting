import { useEffect } from 'react';
import type { DetectionRequest, EngineChoice, EngineInfo, SpotiflowParams } from '../api/types';
import { DEFAULT_SPOTIFLOW } from '../api/types';
import { runLabel, shortChecksum } from '../lib/consensus';
import { useApp } from '../state/store';
import { NumInput } from './DetectTab';
import { AgreementInfo } from './EngineLayers';
import { isJobActive, JOB_STATUS_LABEL } from '../lib/experiment';

const CHOICES: { id: EngineChoice; label: string }[] = [
  { id: 'bigfish', label: 'Big-FISH (default)' },
  { id: 'spotiflow', label: 'Spotiflow' },
  { id: 'hybrid', label: 'Both – compare' },
];

const FALLBACK_MODELS = ['general'];

/** 'Detection engine' selector + Spotiflow model preparation + consensus tolerance. */
export function EngineSection({ req, set, disabled }: { req: DetectionRequest; set: (r: DetectionRequest) => void; disabled: boolean }) {
  const engines = useApp((s) => s.engines);
  const enginesError = useApp((s) => s.enginesError);
  const loadEngines = useApp((s) => s.loadEngines);
  const prepareJob = useApp((s) => s.prepareJob);
  const prepare = useApp((s) => s.prepareModel);
  useEffect(() => {
    void loadEngines();
  }, [loadEngines]);
  const sf: EngineInfo | undefined = engines?.find((e) => e.name === 'spotiflow');
  const known = engines != null;
  const sfAvailable = !known || !!sf?.available; // unknown (older backend) => let the server decide
  const choice = req.engine ?? 'bigfish';
  const spot = req.spotiflow ?? DEFAULT_SPOTIFLOW;
  const needsSf = choice !== 'bigfish';
  const models = sf?.models?.length ? sf.models : FALLBACK_MODELS.map((name) => ({ name, cached: false, version: null, checksum: null }));
  const cur = models.find((m) => m.name === spot.model_name);
  const preparing = isJobActive(prepareJob?.status);
  const gpu = sf?.device?.cuda ? 'CUDA GPU' : sf?.device?.mps ? 'Apple GPU (MPS)' : null;
  return (
    <section className="runbox" data-testid="engine-section">
      <h4>Detection engine</h4>
      <div className="seg full" role="group" aria-label="Detection engine">
        {CHOICES.map((c) => (
          <button
            key={c.id}
            type="button"
            className={choice === c.id ? 'on' : ''}
            disabled={disabled || (c.id !== 'bigfish' && !sfAvailable)}
            onClick={() => set({ ...req, engine: c.id, spotiflow: spot })}
            data-testid={`engine-${c.id}`}
          >
            {c.label}
          </button>
        ))}
      </div>
      {known && !sfAvailable && (
        <div className="notice small" data-testid="spotiflow-unavailable">
          <strong>Spotiflow is not available:</strong> {sf?.reason || 'not installed'}. Spotiflow is optional; Big-FISH keeps working.
        </div>
      )}
      {enginesError && <div className="small muted">Engine list unavailable ({enginesError}); the server will report if Spotiflow cannot run.</div>}
      {choice === 'hybrid' && (
        <div className="small muted">
          Both engines run independently on the same pages; their points are matched into <em>candidates</em> for you to review. Nothing is merged silently, and the raw output of each engine is kept.
        </div>
      )}
      {needsSf && (
        <fieldset className="params" disabled={disabled}>
          <label className="match">
            Spotiflow model
            <select
              className="input sm"
              value={spot.model_name}
              onChange={(e) => set({ ...req, spotiflow: { ...spot, model_name: e.target.value } })}
              data-testid="spotiflow-model"
            >
              {!models.some((m) => m.name === spot.model_name) && <option value={spot.model_name}>{spot.model_name}</option>}
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {m.cached ? ` (ready${m.version ? `, v${m.version}` : ''})` : sf ? ' (not downloaded)' : ''}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="btn sm"
              disabled={!sfAvailable || preparing || disabled}
              onClick={() => void prepare(spot.model_name)}
              title="Download / load the model once; it is cached afterwards"
              data-testid="prepare-model"
            >
              {preparing ? 'Preparing…' : cur?.cached ? 'Re-check model' : 'Prepare model'}
            </button>
          </label>
          <div className="match" data-testid="spotiflow-downscale-box">
            <span>Image downscale</span>
            <select
              className="input sm"
              value={spot.downscale ?? 1}
              onChange={(e) => set({ ...req, spotiflow: { ...spot, downscale: Number(e.target.value) as SpotiflowParams['downscale'] } })}
              data-testid="spotiflow-downscale"
              aria-label="Spotiflow image downscale"
            >
              {[1, 2, 4, 8].map((d) => (
                <option key={d} value={d}>
                  {`÷${d}${d === 1 ? ' (native, default)' : ''}`}
                </option>
              ))}
            </select>
            <button type="button" className="btn sm" onClick={() => set({ ...req, spotiflow: { ...spot, downscale: 4 } })} data-testid="spotiflow-preset-large" title="Set the downscale to ÷4 (exploratory preset for ~25 px blobs)">
              Large blobs (÷4)
            </button>
            <span className="hint" data-testid="spotiflow-downscale-note">
              Polonies are ~25 px blobs; the pretrained models expect small spots. Over 8 research slides, ÷4 with the “valid unsaturated” normalisation matched the researcher marks with precision 0.73 and recall 0.77 at probability threshold 0.2 (10 px tolerance; exploratory, one threshold for all slides, not validated; see docs/app/SPOTIFLOW_EXPLORATION.md). Default stays ÷1. Positions are good to about 1 to 2 px at ÷4 (RMS 1.2 px after peak refinement).
            </span>
          </div>
          <label className="match">
            Intensity normalisation
            <select className="input sm" value={spot.normalizer ?? 'auto'} onChange={(e) => set({ ...req, spotiflow: { ...spot, normalizer: e.target.value as SpotiflowParams['normalizer'] } })} data-testid="spotiflow-normalizer" aria-label="Spotiflow intensity normalisation">
              <option value="auto">Spotiflow auto (legacy, saturation-sensitive)</option>
              <option value="valid_unsaturated">Valid unsaturated pixels (recommended for ÷2 to ÷8)</option>
              <option value="none">None (raw values)</option>
            </select>
            <span className="hint">“Valid unsaturated” takes the 1st and 99.8th percentile of the pixels inside the counting area that are not saturated, once per page, so a saturated rim or debris cannot change the detector’s sensitivity. “Auto” uses all pixels of the cropped area, saturated ones included.</span>
          </label>
          <label className="match">
            Probability threshold
            <NumInput
              value={spot.prob_thresh}
              min={0}
              step={0.05}
              nullable
              placeholder="model default"
              onChange={(v) => set({ ...req, spotiflow: { ...spot, prob_thresh: v } })}
              testid="spotiflow-prob-thresh"
              disabled={disabled}
            />
            <span className="hint">Empty = the model’s own optimised threshold (recorded with the run). Lower values keep more, weaker points.</span>
          </label>
          {cur?.checksum && <div className="hint">weights sha256 {shortChecksum(cur.checksum)}…</div>}
          {prepareJob && (
            <div className="job" data-testid="prepare-job" data-status={prepareJob.status}>
              <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(prepareJob.progress * 100)}>
                <div className={`bar ${prepareJob.status}`} style={{ width: `${Math.round(prepareJob.progress * 100)}%` }} />
              </div>
              <div className="small">
                <strong>{JOB_STATUS_LABEL[prepareJob.status] ?? prepareJob.status}</strong> {Math.round(prepareJob.progress * 100)}% {prepareJob.message && <span className="muted">· {prepareJob.message}</span>}
              </div>
              {prepareJob.error && (
                <div className="notice err" role="alert">
                  {prepareJob.error}
                </div>
              )}
            </div>
          )}
        </fieldset>
      )}
      {choice === 'hybrid' && (
        <label className="match">
          Engine consensus tolerance <span className="unit">px</span>
          <NumInput value={req.consensus_tolerance_px ?? 4} min={0} step={0.5} onChange={(v) => set({ ...req, consensus_tolerance_px: v ?? 4 })} testid="consensus-tol" disabled={disabled} />
          <span className="hint">
            A Big-FISH and a Spotiflow point on the same channel closer than this are paired one-to-one (“both engines”). If you never change it, the default 4 px is raised to the Spotiflow downscale (for example 8 px at ÷8); a value you set yourself is used as entered. Unpaired points become “Big-FISH only” or “Spotiflow only” candidates. Not the same as the red/green match tolerance.
          </span>
        </label>
      )}
      <ul className="engine-notes small muted" data-testid="engine-notes">
        <li>Spotiflow on CPU takes about 30 s per page on a typical laptop CPU (measured ≈ 8.6 s per 1000 × 1000 px); a GPU is used automatically when available{gpu ? ` (detected: ${gpu})` : ''}.{sf?.device_note ? ` ${sf.device_note}.` : ''}</li>
        <li>The Big-FISH default threshold is unchanged. Choosing another engine never alters it.</li>
        <li>Spotiflow results are model-score proposals, not counts.</li>
      </ul>
      {needsSf && <AgreementInfo />}
    </section>
  );
}

const fmtDur = (s: number | null) => (s == null ? '–' : s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`);

/** After a hybrid run: one row per engine execution (engine, model + version, checksum, page, device, duration, status). */
export function EngineRuns() {
  const runs = useApp((s) => s.engineRuns);
  const hasSf = runs.some((r) => r.engine === 'spotiflow');
  if (!runs.length) return null;
  const failedSf = runs.filter((r) => r.engine === 'spotiflow' && r.status === 'failed');
  return (
    <section className="runbox" data-testid="engine-runs">
      <h4>Engine runs</h4>
      {failedSf.map((r) => (
        <div key={r.run_id} className="notice err" role="alert" data-testid="engine-run-error">
          <strong>Spotiflow run failed (page {r.page}):</strong> {r.error || 'no error text reported'}
          <div>Big-FISH result is still valid.</div>
        </div>
      ))}
      <div className="tbl-wrap">
        <table className="tbl small">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Model</th>
              <th>Checksum</th>
              <th className="num">Page</th>
              <th>Device</th>
              <th className="num">Time</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id} data-testid={`run-${r.engine}-${r.page}`} data-status={r.status}>
                <td>{runLabel(r)}</td>
                <td>{r.model_name ? `${r.model_name}${r.model_version ? ` v${r.model_version}` : ''}` : r.engine_version ? `v${r.engine_version}` : '–'}</td>
                <td className="mono">{shortChecksum(r.model_checksum)}</td>
                <td className="num">{r.page}</td>
                <td>{r.device ?? '–'}</td>
                <td className="num">{fmtDur(r.duration_s)}</td>
                <td className={r.status === 'failed' ? 'warn-text' : ''}>
                  {r.status}
                  {r.status === 'ok' ? ` · ${r.n_valid}/${r.n_raw} in area` : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!hasSf && <div className="small muted">Big-FISH only run.</div>}
    </section>
  );
}
