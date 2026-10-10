import { useEffect, useState } from 'react';
import type { DetectionParams } from '../api/types';
import { useApp } from '../state/store';

export function NumInput({
  value,
  onChange,
  min,
  step,
  nullable,
  placeholder,
  id,
  disabled,
  testid,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  step?: number;
  nullable?: boolean;
  placeholder?: string;
  id?: string;
  disabled?: boolean;
  testid?: string;
}) {
  const [text, setText] = useState(value == null ? '' : String(value));
  useEffect(() => {
    setText((t) => (Number(t) === value || (value == null && t === '') ? t : value == null ? '' : String(value)));
  }, [value]);
  return (
    <input
      id={id}
      className="input sm num"
      type="number"
      inputMode="decimal"
      min={min}
      step={step ?? 'any'}
      value={text}
      placeholder={placeholder}
      disabled={disabled}
      data-testid={testid}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value === '') {
          if (nullable) onChange(null);
          return;
        }
        const n = Number(e.target.value);
        if (Number.isFinite(n) && (min == null || n >= min)) onChange(n);
      }}
    />
  );
}

function ParamForm({ ch, p, set, disabled }: { ch: 'red' | 'green'; p: DetectionParams; set: (p: DetectionParams) => void; disabled: boolean }) {
  const auto = p.threshold == null;
  const u = (patch: Partial<DetectionParams>) => set({ ...p, ...patch });
  return (
    <fieldset className={`params ch-${ch}`} disabled={disabled}>
      <legend>
        <span className={`dot ${ch}`} /> {ch} channel
      </legend>
      <label>
        Spot radius <span className="unit">px</span>
        <NumInput value={p.spot_radius_px} min={0.5} step={0.5} onChange={(v) => u({ spot_radius_px: v ?? p.spot_radius_px })} testid={`${ch}-radius`} />
      </label>
      <label>
        Min. distance <span className="unit">px</span>
        <NumInput value={p.minimum_distance_px} min={1} step={0.5} onChange={(v) => u({ minimum_distance_px: v ?? p.minimum_distance_px })} />
      </label>
      <div className="span2">
        <div className="seg sm" role="group" aria-label={`${ch} threshold mode`}>
          <button type="button" className={auto ? 'on' : ''} onClick={() => u({ threshold: null })}>
            Auto threshold
          </button>
          <button type="button" className={!auto ? 'on' : ''} onClick={() => u({ threshold: p.threshold ?? 100 })}>
            Manual
          </button>
        </div>
      </div>
      {auto ? (
        <label className="span2">
          Auto-threshold scale <span className="unit">×</span>
          <NumInput value={p.threshold_scale} min={0.05} step={0.05} onChange={(v) => u({ threshold_scale: v ?? 1 })} />
          <span className="hint">multiplies the Big-FISH automatic threshold (1.0 = unchanged)</span>
        </label>
      ) : (
        <label className="span2">
          Manual threshold <span className="unit">intensity</span>
          <NumInput value={p.threshold} min={0} onChange={(v) => u({ threshold: v ?? 0 })} />
        </label>
      )}
      <label>
        Background
        <select className="input sm" value={p.background.method} onChange={(e) => u({ background: { ...p.background, method: e.target.value as DetectionParams['background']['method'] } })}>
          <option value="none">none</option>
          <option value="gaussian_subtract">Gaussian subtract</option>
          <option value="white_tophat">white top-hat</option>
        </select>
      </label>
      {p.background.method === 'gaussian_subtract' && (
        <label>
          σ <span className="unit">px</span>
          <NumInput value={p.background.sigma_px} min={1} onChange={(v) => u({ background: { ...p.background, sigma_px: v ?? 50 } })} />
        </label>
      )}
      {p.background.method === 'white_tophat' && (
        <label>
          Radius <span className="unit">px</span>
          <NumInput value={p.background.radius_px} min={1} step={1} onChange={(v) => u({ background: { ...p.background, radius_px: Math.round(v ?? 25) } })} />
        </label>
      )}
      <label>
        Tile size <span className="unit">px (0 = whole)</span>
        <NumInput value={p.tile_size} min={0} step={64} onChange={(v) => u({ tile_size: Math.round(v ?? 0) })} />
      </label>
      <label className="chk span2">
        <input type="checkbox" checked={p.dense_decomposition} onChange={(e) => u({ dense_decomposition: e.target.checked })} /> Dense-region decomposition
      </label>
      {p.dense_decomposition && (
        <div className="notice warn small span2">
          Dense decomposition can split or merge clumps by modelling a reference spot. It changes counts in crowded areas, so review those regions carefully. Off by default.
        </div>
      )}
    </fieldset>
  );
}

export function DetectTab() {
  const a = useApp((s) => s.current)!;
  const req = useApp((s) => s.detReq);
  const setReq = useApp((s) => s.setDetReq);
  const job = useApp((s) => s.job);
  const run = useApp((s) => s.runDetection);
  const cancel = useApp((s) => s.cancelJob);
  const preview = useApp((s) => s.preview);
  const previewBusy = useApp((s) => s.previewBusy);
  const clearPreview = useApp((s) => s.clearPreview);
  const setTool = useApp((s) => s.setTool);
  const tool = useApp((s) => s.tool);
  const patch = useApp((s) => s.patch);
  if (!req) return null;
  const mapped = a.status !== 'needs_channel_mapping' && a.status !== 'created';
  const running = job?.status === 'queued' || job?.status === 'running' || a.status === 'detecting';
  const locked = a.status === 'finalized';
  const hasEdits = a.objects.some((o) => o.origin !== 'auto' || o.reviewed);
  const hasApprovals = a.regions.some((r) => r.approved);
  const unitsUnknown = !a.image?.pixel_size_um;
  return (
    <div className="detect" data-testid="detect-tab">
      <h3>Detection (Big-FISH)</h3>
      {unitsUnknown && <div className="notice small">Pixel size unknown: all distances and radii here are in image pixels. Nothing is converted to µm.</div>}
      {!mapped && (
        <div className="notice warn">
          Confirm the channel mapping first.{' '}
          <button className="btn sm" onClick={() => patch({ mappingOpen: true })}>
            Open mapping
          </button>
        </div>
      )}
      {a.provenance.stale_detection && a.objects.length > 0 && (
        <div className="notice warn small">Detection is stale: regions, mapping or parameters changed since the last run.</div>
      )}
      <div className="params-grid">
        <ParamForm ch="red" p={req.red} set={(p) => setReq({ ...req, red: p })} disabled={locked} />
        <ParamForm ch="green" p={req.green} set={(p) => setReq({ ...req, green: p })} disabled={locked} />
      </div>
      <div className="row wrap">
        <button className="btn sm" disabled={locked} onClick={() => setReq({ ...req, green: JSON.parse(JSON.stringify(req.red)) })} title="Copy red parameters to green">
          Copy red → green
        </button>
        <button className="btn sm" disabled={locked} onClick={() => setReq({ ...req, red: JSON.parse(JSON.stringify(req.green)) })} title="Copy green parameters to red">
          Copy green → red
        </button>
        <button className="btn sm ghost" onClick={() => setReq(a.detection_request)} title="Reset to the last saved request">
          Revert
        </button>
      </div>
      <label className="match">
        Red/green match tolerance <span className="unit">px</span>
        <NumInput value={req.match_tolerance_px} min={0} step={0.5} onChange={(v) => setReq({ ...req, match_tolerance_px: v ?? 5 })} testid="match-tol" />
        <span className="hint">spots of different colours closer than this are considered one yellow object (one-to-one); ambiguous cases become “uncertain”.</span>
      </label>

      <section className="runbox">
        <h4>Preview</h4>
        <div className="row wrap">
          <button className={`btn ${tool === 'preview-window' ? 'on' : ''}`} disabled={!mapped || previewBusy} onClick={() => setTool('preview-window')} data-testid="preview-btn">
            Preview on selected window
          </button>
          {preview && (
            <button className="btn ghost" onClick={clearPreview}>
              Clear preview
            </button>
          )}
        </div>
        {tool === 'preview-window' && <div className="small muted">Drag a window on the image.</div>}
        {previewBusy && (
          <div className="small">
            <span className="spinner" /> running…
          </div>
        )}
        {preview && (
          <div className="small" data-testid="preview-info">
            Preview window [{preview.window.join(', ')}]: <strong>{preview.objects.length}</strong> candidates (temporary overlay with dotted amber rings; nothing saved).
            {preview.notes.map((n, i) => (
              <div key={i} className="muted">
                {n}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="runbox">
        <h4>Run on all regions</h4>
        <div className="notice small">
          Re-running keeps your manual edits and objects you reviewed. Fresh automatic results replace only untouched automatic objects, and approvals of regions whose detections change are cleared
          {hasEdits || hasApprovals ? ` (this analysis has ${hasEdits ? 'edited/reviewed objects' : ''}${hasEdits && hasApprovals ? ' and ' : ''}${hasApprovals ? 'approved regions' : ''}).` : '.'}
        </div>
        <div className="row">
          <button className="btn primary" disabled={!mapped || running || locked} onClick={() => void run()} data-testid="run-detection">
            {running ? 'Running…' : a.objects.length ? 'Re-run detection' : 'Run detection'}
          </button>
          {running && job && job.id && (
            <button className="btn danger" onClick={() => void cancel()} data-testid="cancel-job">
              Cancel
            </button>
          )}
        </div>
        {a.regions.filter((r) => r.kind === 'counting').length === 0 && <div className="small muted">No counting regions: detection runs on the whole image minus exclusions.</div>}
        {job && (
          <div className="job" data-testid="job" data-status={job.status}>
            <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}>
              <div className={`bar ${job.status}`} style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </div>
            <div className="small">
              <strong>{job.status}</strong> {Math.round(job.progress * 100)}% {job.message && <span className="muted">· {job.message}</span>}
            </div>
            {job.error && (
              <div className="notice err" role="alert">
                {job.error}
              </div>
            )}
          </div>
        )}
        {running && !job && <div className="small">Detection running on the server…</div>}
      </section>
    </div>
  );
}
