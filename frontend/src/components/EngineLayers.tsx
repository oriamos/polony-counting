import type { ReactNode } from 'react';
import { AGREEMENT_NOT_TRUTH, hasEngineInfo } from '../lib/consensus';
import { triangleD } from '../lib/consensus';
import { useApp, type EngineLayers as Layers } from '../state/store';
import { RAW_COLOR, STATUS_STYLE } from './Overlay';

/** Short popover: why two engines agreeing is not evidence of truth. */
export function AgreementInfo({ label = 'Why agreement is not ground truth' }: { label?: string }) {
  return (
    <details className="infopop" data-testid="agreement-info">
      <summary>{label}</summary>
      <div className="infopop-body" role="note">
        {AGREEMENT_NOT_TRUTH}
      </div>
    </details>
  );
}

const Swatch = ({ children }: { children: ReactNode }) => (
  <svg width={22} height={22} viewBox="0 0 22 22" aria-hidden>
    {children}
  </svg>
);
const ring = (r: number, dash?: string, cap?: 'round') => (
  <>
    <circle cx={11} cy={11} r={r} fill="none" stroke="#000" strokeWidth={3.4} strokeDasharray={dash} strokeLinecap={cap} />
    <circle cx={11} cy={11} r={r} fill="none" stroke="#fff" strokeWidth={1.4} strokeDasharray={dash} strokeLinecap={cap} />
  </>
);
const LEGEND: { key: string; label: string; icon: ReactNode }[] = [
  { key: 'agreement', label: STATUS_STYLE.agreement.name, icon: <Swatch>{ring(8)}</Swatch> },
  { key: 'bigfish_only', label: STATUS_STYLE.bigfish_only.name, icon: <Swatch>{ring(8, '4.5 2.5')}</Swatch> },
  { key: 'spotiflow_only', label: STATUS_STYLE.spotiflow_only.name, icon: <Swatch>{ring(8, '0.9 2.4', 'round')}</Swatch> },
  {
    key: 'human_added',
    label: STATUS_STYLE.human_added.name,
    icon: (
      <Swatch>
        {ring(6)}
        {ring(9)}
      </Swatch>
    ),
  },
  {
    key: 'rejected',
    label: 'human rejected (grey dashed cross)',
    icon: (
      <Swatch>
        <circle cx={11} cy={11} r={7} fill="none" stroke="#c9ced8" strokeWidth={1.4} strokeDasharray="2 2" />
        <path d="M7.5 7.5L14.5 14.5M7.5 14.5L14.5 7.5" stroke="#c9ced8" strokeWidth={1.4} />
      </Swatch>
    ),
  },
];

export function EngineLayersMenu() {
  const layers = useApp((s) => s.layers);
  const setLayer = useApp((s) => s.setLayer);
  const a = useApp((s) => s.current)!;
  const runs = useApp((s) => s.engineRuns);
  const cache = useApp((s) => s.rawCache);
  const rawError = useApp((s) => s.rawError);
  const capped = Object.values(cache).some((c) => c.capped);
  const nRejected = a.objects.filter((o) => o.deleted).length;
  if (!runs.length && !hasEngineInfo(a.objects)) return null;
  const T = ({ k, children, testid }: { k: keyof Layers; children: ReactNode; testid: string }) => (
    <label className="chk">
      <input type="checkbox" checked={layers[k]} onChange={(e) => setLayer({ [k]: e.target.checked })} data-testid={testid} /> {children}
    </label>
  );
  return (
    <details className="layers" data-testid="engine-layers">
      <summary className="btn sm" title="Engine layers and legend">
        Engine layers ▾
      </summary>
      <div className="layers-body">
        <T k="consensus" testid="layer-consensus">
          Engine status on candidates
        </T>
        <T k="rawBigfish" testid="layer-raw-bigfish">
          <svg width={14} height={14} viewBox="0 0 14 14" aria-hidden>
            <circle cx={7} cy={7} r={4} fill="none" stroke={RAW_COLOR.bigfish} strokeWidth={1.6} />
          </svg>{' '}
          Raw Big-FISH detections (hollow circle)
        </T>
        <T k="rawSpotiflow" testid="layer-raw-spotiflow">
          <svg width={14} height={14} viewBox="0 0 14 14" aria-hidden>
            <path d={triangleD(7, 7, 5)} fill="none" stroke={RAW_COLOR.spotiflow} strokeWidth={1.6} />
          </svg>{' '}
          Raw Spotiflow detections (hollow triangle)
        </T>
        {(layers.rawBigfish || layers.rawSpotiflow) && (
          <label className="chk sub">
            <input type="checkbox" checked={layers.rawExcluded} onChange={(e) => setLayer({ rawExcluded: e.target.checked })} data-testid="layer-raw-excluded" /> show excluded (outside counting area, faint)
          </label>
        )}
        <T k="rejected" testid="layer-rejected">
          Show rejected ({nRejected})
        </T>
        {capped && <div className="small warn-text">Raw layer capped: only the first points per page are loaded.</div>}
        {rawError && <div className="small warn-text">Raw detections unavailable: {rawError}</div>}
        <div className="legend" data-testid="layer-legend">
          {LEGEND.map((l) => (
            <div key={l.key} className="row small">
              {l.icon} {l.label}
            </div>
          ))}
          <div className="muted small">Letters: = both, B Big-FISH, S Spotiflow, + human. Class shape/colour is unchanged; tick = reviewed, amber dot = unresolved.</div>
        </div>
        <AgreementInfo />
      </div>
    </details>
  );
}
