import { useState } from 'react';
import type { ConsensusSummary } from '../api/types';
import { clampQcN, ENGINE_FILTERS, QC_DEFAULT_N, qcBoundText, type EngineFilter } from '../lib/consensus';
import { useApp } from '../state/store';
import { AgreementInfo } from './EngineLayers';

const sum = (c: ConsensusSummary) => c.agreement + c.bigfish_only + c.spotiflow_only + c.human_added;

function Line({ c }: { c: ConsensusSummary }) {
  return (
    <span className="mono small">
      Both engines {c.agreement} · Big-FISH only {c.bigfish_only} · Spotiflow only {c.spotiflow_only} · Human added {c.human_added} · Confirmed {c.confirmed} · Rejected {c.rejected} · Unresolved {c.unresolved}
    </span>
  );
}

/** Review summary per page/channel (pages are never merged silently; the total is a separate line). */
export function ConsensusSummaryCard() {
  const sm = useApp((s) => s.consensusSummary);
  const a = useApp((s) => s.current)!;
  if (!sm || (sum(sm.total) === 0 && sm.pages.every((p) => sum(p) === 0))) return null;
  const roleText = (c: ConsensusSummary) => {
    if (c.channel_role) return `confirmed role: ${c.channel_role}`;
    const ch = a.image?.channels.find((x) => x.page === c.page);
    return ch?.suggested_role ? `suggested role: ${ch.suggested_role} (not confirmed)` : 'role not confirmed';
  };
  return (
    <section className="runbox" data-testid="consensus-summary">
      <h4>Engine review summary</h4>
      <div data-testid="summary-total">
        <strong className="small">All pages (sum): </strong>
        <Line c={sm.total} />
      </div>
      {sm.pages.length > 0 && (
        <ul className="sum-pages">
          {sm.pages.map((p, i) => (
            <li key={i} data-testid={`summary-page-${p.page ?? 'x'}`}>
              <strong className="small">Page {p.page ?? '?'}</strong> <span className="muted small">· {roleText(p)}</span>
              <div>
                <Line c={p} />
              </div>
            </li>
          ))}
        </ul>
      )}
      {sm.qc && sm.qc.sample_size > 0 && (
        <div className="small" data-testid="summary-qc">
          <strong>Agreement QC: </strong>
          {qcBoundText(sm.qc)}
        </div>
      )}
      <div className="hint">“Both engines” means both proposed a nearby point. It is not a verified polony.</div>
    </section>
  );
}

export function EngineFilterBar({ filter, onChange }: { filter: EngineFilter; onChange: (f: EngineFilter) => void }) {
  return (
    <div className="row wrap small" data-testid="engine-filter">
      <label>
        Engine status{' '}
        <select className="input sm" value={filter} onChange={(e) => onChange(e.target.value as EngineFilter)} data-testid="engine-filter-select">
          {ENGINE_FILTERS.map((f) => (
            <option key={f.id} value={f.id}>
              {f.label}
            </option>
          ))}
        </select>
      </label>
      <span className="muted">Queue order comes from the server: engine disagreements first.</span>
    </div>
  );
}

export function QcSample() {
  const qc = useApp((s) => s.qcSample);
  const locked = useApp((s) => s.current?.status === 'finalized');
  const status = useApp((s) => s.consensusSummary?.qc);
  const [n, setN] = useState(QC_DEFAULT_N);
  const [seed, setSeed] = useState(1);
  const pop = status?.population_agreements ?? null;
  const nEff = clampQcN(n, pop);
  return (
    <details className="infopop" data-testid="qc-sample">
      <summary>QC sample of agreements</summary>
      <div className="infopop-body">
        <p>
          <strong>Agreement is not ground truth</strong> (agreement ≠ ground truth): two engines can be wrong together, and agreement objects are not reviewed unless sampled. Pick a fixed number of unreviewed agreement candidates at random and check them by eye; the result bounds the error rate of the agreements you did not check, only if the sample is random (one seed per slide). The same seed gives the same sample.
        </p>
        <div className="row wrap">
          <label>
            Sample size{' '}
            <input className="input sm num" type="number" min={1} step={1} value={n} onChange={(e) => setN(Math.max(1, Math.round(Number(e.target.value) || QC_DEFAULT_N)))} data-testid="qc-n" />
          </label>
          <label>
            Seed <input className="input sm num" type="number" step={1} value={seed} onChange={(e) => setSeed(Math.round(Number(e.target.value)))} data-testid="qc-seed" />
          </label>
          <button className="btn sm" disabled={locked || !(n >= 1)} onClick={() => void qc(nEff, seed)} data-testid="qc-run">
            Mark {nEff} for review
          </button>
        </div>
        {status && status.population_agreements > 0 && (
          <div className="hint small" data-testid="qc-needed">
            {status.population_agreements} agreement candidates on this slide. Sample size for a ±0.05 margin on the error rate: {status.n_for_margin_0_05} (assuming the measured {Math.round(100 * status.planning_error_rate)} % error; {status.n_for_margin_0_05_conservative} if unknown).
          </div>
        )}
        <div className="small" data-testid="qc-bound">
          {qcBoundText(status)}
        </div>
        <AgreementInfo />
      </div>
    </details>
  );
}
