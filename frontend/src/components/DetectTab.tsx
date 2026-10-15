import { useEffect, useState } from 'react';
import type { DetectionParams, DetectionRequest } from '../api/types';
import { useApp } from '../state/store';
import { EngineRuns, EngineSection } from './EngineSection';
import { isJobActive, JOB_STATUS_LABEL } from '../lib/experiment';

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

type Ch = 'red' | 'green';

/** Red and green parameters side by side: one row per parameter, one column per channel. */
function ParamTable({ req, set, disabled }: { req: DetectionRequest; set: (r: DetectionRequest) => void; disabled: boolean }) {
  const u = (ch: Ch, patch: Partial<DetectionParams>) => set({ ...req, [ch]: { ...req[ch], ...patch } });
  const chs: Ch[] = ['red', 'green'];
  const Row = ({ label, unit, hint, children }: { label: string; unit?: string; hint?: string; children: (ch: Ch, p: DetectionParams) => React.ReactNode }) => (
    <tr>
      <th scope="row">
        {label} {unit && <span className="unit">{unit}</span>}
        {hint && <div className="hint">{hint}</div>}
      </th>
      {chs.map((ch) => (
        <td key={ch}>{children(ch, req[ch])}</td>
      ))}
    </tr>
  );
  return (
    <fieldset className="params" disabled={disabled}>
      <table className="ptable">
        <thead>
          <tr>
            <th></th>
            <th>
              <span className="dot red" /> red
            </th>
            <th>
              <span className="dot green" /> green
            </th>
          </tr>
        </thead>
        <tbody>
          <Row label="Spot radius" unit="px">
            {(ch, p) => <NumInput value={p.spot_radius_px} min={0.5} step={0.5} onChange={(v) => u(ch, { spot_radius_px: v ?? p.spot_radius_px })} testid={`${ch}-radius`} />}
          </Row>
          <Row label="Min. distance" unit="px">
            {(ch, p) => <NumInput value={p.minimum_distance_px} min={1} step={0.5} onChange={(v) => u(ch, { minimum_distance_px: v ?? p.minimum_distance_px })} />}
          </Row>
          <Row label="Threshold (noise units, robust σ of the LoG response)">
            {(ch, p) => (
              <div className="seg sm" role="group" aria-label={`${ch} threshold mode`}>
                <button type="button" className={p.threshold == null ? 'on' : ''} onClick={() => u(ch, { threshold: null })}>
                  Auto
                </button>
                <button type="button" className={p.threshold != null ? 'on' : ''} onClick={() => u(ch, { threshold: p.threshold ?? 50 })}>
                  Manual
                </button>
              </div>
            )}
          </Row>
          <Row label="Auto scale" unit="×">
            {(ch, p) => <NumInput value={p.threshold_scale} min={0.05} step={0.05} disabled={p.threshold != null} onChange={(v) => u(ch, { threshold_scale: v ?? 1 })} />}
          </Row>
          <Row label="Manual value" unit="noise units (σ of LoG)">
            {(ch, p) => <NumInput value={p.threshold} min={0} disabled={p.threshold == null} onChange={(v) => u(ch, { threshold: v ?? 0 })} />}
          </Row>
          <Row label="Background">
            {(ch, p) => (
              <select
                className="input sm"
                value={p.background.method}
                aria-label={`${ch} background method`}
                onChange={(e) => u(ch, { background: { ...p.background, method: e.target.value as DetectionParams['background']['method'] } })}
              >
                <option value="none">none</option>
                <option value="gaussian_subtract">Gaussian subtract</option>
                <option value="white_tophat">white top-hat</option>
              </select>
            )}
          </Row>
          <Row label="Background σ / radius" unit="px">
            {(ch, p) =>
              p.background.method === 'gaussian_subtract' ? (
                <NumInput value={p.background.sigma_px} min={1} onChange={(v) => u(ch, { background: { ...p.background, sigma_px: v ?? 50 } })} />
              ) : p.background.method === 'white_tophat' ? (
                <NumInput value={p.background.radius_px} min={1} step={1} onChange={(v) => u(ch, { background: { ...p.background, radius_px: Math.round(v ?? 25) } })} />
              ) : (
                <span className="muted small">–</span>
              )
            }
          </Row>
          <Row label="Tile size" unit="px (0 = whole)">
            {(ch, p) => <NumInput value={p.tile_size} min={0} step={64} onChange={(v) => u(ch, { tile_size: Math.round(v ?? 0) })} />}
          </Row>
          <Row label="Dense decomposition">
            {(ch, p) => (
              <label className="chk">
                <input type="checkbox" checked={p.dense_decomposition} onChange={(e) => u(ch, { dense_decomposition: e.target.checked })} data-testid={`${ch}-dense`} /> on
              </label>
            )}
          </Row>
        </tbody>
      </table>
      <div className="hint">Thresholds are in noise units: the LoG response divided by its robust noise σ (1.4826 × MAD over the counting area), so they do not depend on image brightness. “Auto scale” multiplies the automatic threshold (1 = unchanged); “Manual value” replaces it (the automatic one is typically 40–60 on the test image; a value below ~5 is in the noise).</div>
      {(req.red.dense_decomposition || req.green.dense_decomposition) && (
        <div className="notice warn small">
          Dense decomposition can split or merge clumps by modelling a reference spot. It changes counts in crowded areas, so review those regions carefully. Off by default.
        </div>
      )}
    </fieldset>
  );
}

/** Sets the automatic-threshold multiplier on both channels (back to Auto mode). */
export const withScale = (req: DetectionRequest, scale: number): DetectionRequest => ({
  ...req,
  red: { ...req.red, threshold: null, threshold_scale: scale },
  green: { ...req.green, threshold: null, threshold_scale: scale },
});

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
  const running = isJobActive(job?.status) || a.status === 'detecting';
  const locked = a.status === 'finalized';
  const hasEdits = a.objects.some((o) => o.origin !== 'auto' || o.reviewed);
  const hasApprovals = a.regions.some((r) => r.approved);
  const unitsUnknown = !a.image?.pixel_size_um;
  return (
    <div className="detect" data-testid="detect-tab">
      <h3>Detection</h3>
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
      {mapped && (
        <div className="row small">
          <span className="muted">Channel mapping confirmed.</span>
          <button className="btn sm ghost" disabled={locked} onClick={() => patch({ mappingOpen: true })} data-testid="edit-mapping">
            Change channel mapping…
          </button>
        </div>
      )}
      <EngineSection req={req} set={setReq} disabled={locked} />
      <ParamTable req={req} set={setReq} disabled={locked} />
      <div className="presets" data-testid="sensitivity-presets">
        <div className="row wrap">
          <span className="small">Sensitivity preset:</span>
          <button className="btn sm" disabled={locked} onClick={() => setReq(withScale(req, 1))} data-testid="preset-default">
            Default sensitivity
          </button>
          <button className="btn sm" disabled={locked} onClick={() => setReq(withScale(req, 0.5))} data-testid="preset-sensitive">
            More sensitive (threshold ×0.5)
          </button>
        </div>
        <div className="muted small" data-testid="preset-note">
          On the 8 validated slides the default threshold counted 18–33% fewer polonies than the researcher; ×0.5 recovered most but adds false positives — check the review queue (debris-like flagged). Not yet validated beyond these slides.
        </div>
      </div>
      <div className="hint-strong" data-testid="radius-hint">
        <strong>Count depends strongly on spot radius</strong> (real example: radius 5 → 108 objects, 3 → 207, 2 → 331). Choose it from the visible blob size using Preview, don't rely on the default.
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
              <strong>{JOB_STATUS_LABEL[job.status] ?? job.status}</strong> {Math.round(job.progress * 100)}% {job.message && <span className="muted">· {job.message}</span>}
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
      <EngineRuns />
    </div>
  );
}
