import type { ChannelEvidence, DetectedObject, ObjectClass } from '../api/types';
import { thumbUrl } from '../api/client';
import { CLASS_KEY, CLASS_LABEL, CLASS_ORDER, REASON_SHORT, REASON_TEXT } from '../lib/glyph';
import { liveObjects, useApp, type RightTab } from '../state/store';
import { shortId } from './Overlay';
import { GlyphIcon } from './ViewerToolbar';
import { RegionsTab } from './RegionsTab';
import { DetectTab } from './DetectTab';
import { FinishTab } from './FinishTab';
import { SaveTab } from './SaveTab';
import { TotalsTab } from './TotalsTab';
import { AgreementInfo } from './EngineLayers';
import { SCORE_LABEL, STATUS_LABEL, filteredQueue, hasEngineInfo, objectStatus } from '../lib/consensus';
import { ConsensusSummaryCard, EngineFilterBar, QcSample } from './EngineReview';

const TABS: { id: RightTab; label: string }[] = [
  { id: 'review', label: 'Review' },
  { id: 'regions', label: 'Regions' },
  { id: 'detect', label: 'Detect' },
  { id: 'totals', label: 'Totals' },
  { id: 'finish', label: 'Finish' },
  { id: 'save', label: 'Save' },
];

export function RightPanel() {
  const tab = useApp((s) => s.rightTab);
  const patch = useApp((s) => s.patch);
  const a = useApp((s) => s.current)!;
  const queueN = useApp((s) => s.queue.length);
  const nUncertain = liveObjects(a).filter((o) => o.cls === 'uncertain').length;
  return (
    <aside className="right" aria-label="Analysis panels">
      <Inspector />
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'on' : ''} onClick={() => patch({ rightTab: t.id })} data-testid={`tab-${t.id}`}>
            {t.label}
            {t.id === 'review' && queueN > 0 && <span className="badge">{queueN}</span>}
            {t.id === 'totals' && nUncertain > 0 && <span className="badge warn" title="uncertain objects">{nUncertain}</span>}
          </button>
        ))}
      </div>
      <div className="tabbody" role="tabpanel">
        {tab === 'review' && <ReviewTab />}
        {tab === 'regions' && <RegionsTab />}
        {tab === 'detect' && <DetectTab />}
        {tab === 'totals' && <TotalsTab />}
        {tab === 'finish' && <FinishTab />}
        {tab === 'save' && <SaveTab />}
      </div>
    </aside>
  );
}

const f = (n: number | null | undefined, d = 2) => (n == null || !Number.isFinite(n) ? '–' : n.toFixed(d));

function Evidence({ label, e }: { label: string; e: ChannelEvidence }) {
  if (!e.detected)
    return (
      <tr>
        <th>{label}</th>
        <td colSpan={4} className="muted">
          no spot detected in this channel
        </td>
      </tr>
    );
  return (
    <tr>
      <th>{label}</th>
      <td className="num mono" title="signal-to-noise ratio (measured)">{f(e.snr, 1)}</td>
      <td className="num mono" title="background-corrected peak intensity">{f(e.peak_intensity, 0)}</td>
      <td className="num mono" title="local background">{f(e.local_background, 0)}</td>
      <td className={`num ${e.saturated ? 'warn-text' : 'muted'}`}>{e.saturated ? 'saturated' : 'no'}</td>
    </tr>
  );
}

function EngineProvenance({ o }: { o: DetectedObject }) {
  const chans = (['red', 'green'] as const).filter((c) => o[c].consensus_status);
  if (!chans.length) return null;
  return (
    <div className="provenance" data-testid="engine-provenance">
      <div className="small">
        <strong>Engine provenance</strong> <span className="muted">(candidate proposal, not a verdict)</span>
      </div>
      <table className="tbl evid">
        <thead>
          <tr>
            <th>Channel</th>
            <th>Produced by</th>
            <th>Big-FISH x, y</th>
            <th className="num" title="raw Laplacian-of-Gaussian response divided by robust noise sigma">{SCORE_LABEL.bigfish}</th>
            <th>Spotiflow x, y</th>
            <th className="num" title="raw Spotiflow heatmap value; not calibrated">{SCORE_LABEL.spotiflow}</th>
          </tr>
        </thead>
        <tbody>
          {chans.map((c) => {
            const e = o[c];
            const st = e.consensus_status!;
            const hasBf = e.bigfish_x != null && e.bigfish_y != null;
            const hasSf = e.spotiflow_x != null && e.spotiflow_y != null;
            return (
              <tr key={c} data-testid={`prov-${c}`} data-status={st}>
                <th>{c}</th>
                <td>
                  {STATUS_LABEL[st]}
                  {st === 'agreement' && e.match_distance_px != null && <div className="muted">matched, {f(e.match_distance_px)} px apart</div>}
                </td>
                <td className="mono">{hasBf ? `${f(e.bigfish_x)}, ${f(e.bigfish_y)}` : '–'}</td>
                <td className="num mono">{f(e.bigfish_score, 1)}</td>
                <td className="mono">{hasSf ? `${f(e.spotiflow_x)}, ${f(e.spotiflow_y)}` : '–'}</td>
                <td className="num mono">{f(e.spotiflow_score, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="hint">Scores are each engine’s raw output on its own scale. They are not probabilities and cannot be compared between engines.</div>
      {chans.some((c) => o[c].consensus_status === 'agreement') && <AgreementInfo />}
    </div>
  );
}

export function ReasonList({ reasons }: { reasons: DetectedObject['review_reasons'] }) {
  if (!reasons.length) return <div className="muted small">No measurable review concerns.</div>;
  return (
    <ul className="reasons">
      {reasons.map((r) => (
        <li key={r}>{REASON_TEXT[r] ?? r}</li>
      ))}
    </ul>
  );
}

function Inspector() {
  const a = useApp((s) => s.current)!;
  const selection = useApp((s) => s.selection);
  const act = useApp((s) => s.act);
  const setTool = useApp((s) => s.setTool);
  const patch = useApp((s) => s.patch);
  const confirmSel = useApp((s) => s.confirmSelected);
  const objs = selection.map((id) => a.objects.find((o) => o.id === id)).filter((o): o is DetectedObject => !!o && !o.deleted);
  const locked = a.status === 'finalized';
  if (!objs.length) return null;
  const o = objs[0];
  const engineObj = objs.length === 1 && objectStatus(o) != null;
  const region = o.region_id ? a.regions.find((r) => r.id === o.region_id) : null;
  return (
    <section className="inspector" aria-label="Selected spot" data-testid="inspector">
      <header className="insp-head">
        {objs.length === 1 ? (
          <>
            <GlyphIcon cls={o.cls} size={18} /> <strong className="mono">{shortId(o.id)}</strong>
            <span className={`chip cls-${o.cls}`}>{CLASS_LABEL[o.cls]}</span>
            <span className="chip">{o.origin}</span>
            {objectStatus(o) && <span className="chip" data-testid="engine-chip">{STATUS_LABEL[objectStatus(o)!]}</span>}
            {o.reviewed ? <span className="chip ok">reviewed</span> : <span className="chip warn">unreviewed</span>}
          </>
        ) : (
          <strong>{objs.length} spots selected</strong>
        )}
        <button className="btn sm ghost right" onClick={() => patch({ selection: [] })} aria-label="Clear selection">
          ✕
        </button>
      </header>
      <div className="insp-actions">
        <span className="small muted">Class</span>
        {CLASS_ORDER.map((c: ObjectClass) => (
          <button
            key={c}
            className={`btn sm ${objs.length === 1 && o.cls === c ? 'on' : ''}`}
            disabled={locked}
            onClick={() => void act('reclassify', { object_ids: objs.map((x) => x.id), cls: c })}
            title={`Set class to ${CLASS_LABEL[c]} (${CLASS_KEY[c]})`}
            data-testid={`cls-${c}`}
          >
            <GlyphIcon cls={c} size={14} /> {CLASS_KEY[c]}
          </button>
        ))}
      </div>
      <div className="insp-actions">
        <button className="btn sm primary" disabled={locked} onClick={() => void confirmSel()} title={engineObj ? 'Approve this candidate (= confirm; Enter)' : 'Mark as reviewed (Enter)'} data-testid="confirm">
          {engineObj ? 'Approve' : 'Confirm'} <kbd>Enter</kbd>
        </button>
        <button className="btn sm" disabled={locked || objs.length < 2} onClick={() => void act('merge', { object_ids: objs.map((x) => x.id) })} title="Merge (M)">
          Merge <kbd>M</kbd>
        </button>
        <button className="btn sm" disabled={locked || objs.length !== 1} onClick={() => setTool('split')} title="Split (S)">
          Split <kbd>S</kbd>
        </button>
        <button className="btn sm danger" disabled={locked} onClick={() => void act('delete', { object_ids: objs.map((x) => x.id) })} title={engineObj ? 'Reject this candidate (= delete; kept in the audit history; Del)' : 'Delete (Del)'} data-testid="inspector-delete">
          {engineObj ? 'Reject' : 'Delete'} <kbd>Del</kbd>
        </button>
      </div>
      {objs.length === 1 && (
        <>
          <dl className="kv">
            <dt>Position</dt>
            <dd className="mono">
              x {f(o.x)} · y {f(o.y)} px
            </dd>
            <dt>Region</dt>
            <dd>{region ? region.name : <span className="muted">outside counting regions</span>}</dd>
            {o.association_distance_px != null && (
              <>
                <dt>Association</dt>
                <dd className="mono">{f(o.association_distance_px)} px red↔green</dd>
              </>
            )}
            {o.ambiguous_with.length > 0 && (
              <>
                <dt>Ambiguous with</dt>
                <dd className="mono">{o.ambiguous_with.map(shortId).join(', ')}</dd>
              </>
            )}
          </dl>
          <table className="tbl evid">
            <thead>
              <tr>
                <th>Evidence</th>
                <th className="num">SNR</th>
                <th className="num">peak</th>
                <th className="num">bkg</th>
                <th className="num">sat.</th>
              </tr>
            </thead>
            <tbody>
              <Evidence label="red" e={o.red} />
              <Evidence label="green" e={o.green} />
            </tbody>
          </table>
          <EngineProvenance o={o} />
          <ReasonList reasons={o.review_reasons} />
          {o.history.length > 0 && (
            <details className="small">
              <summary>History ({o.history.length})</summary>
              <ul className="hist">
                {o.history.map((h, i) => (
                  <li key={i}>{h}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </section>
  );
}

function ReviewTab() {
  const a = useApp((s) => s.current)!;
  const queue = useApp((s) => s.queue);
  const selection = useApp((s) => s.selection);
  const patch = useApp((s) => s.patch);
  const goto = useApp((s) => s.gotoQueue);
  const zoom = useApp((s) => s.zoomToPoint);
  const objects = new Map(a.objects.map((o) => [o.id, o]));
  const filter = useApp((s) => s.reviewFilter);
  const engineData = hasEngineInfo(a.objects);
  const listed = engineData ? filteredQueue(queue, a.objects, filter) : queue;
  const noDetection = a.objects.length === 0;
  const unreviewed = a.objects.filter((o) => !o.deleted && !o.reviewed);
  const stepUnreviewed = (dir: 1 | -1) => {
    if (!unreviewed.length) return;
    const cur = selection.length === 1 ? unreviewed.findIndex((o) => o.id === selection[0]) : -1;
    const n = unreviewed.length;
    const o = unreviewed[cur < 0 ? (dir === 1 ? 0 : n - 1) : (cur + dir + n) % n];
    patch({ selection: [o.id], tool: 'select' });
    zoom(o.x, o.y, 10);
  };
  return (
    <div className="review" data-testid="review-tab">
      <div className="row between">
        <h3>Review queue</h3>
        <div className="row">
          <button className="btn sm" onClick={() => goto(-1)} disabled={!queue.length} title="Previous (K)">
            ↑ Prev <kbd>K</kbd>
          </button>
          <button className="btn sm" onClick={() => goto(1)} disabled={!queue.length} title="Next (J)">
            ↓ Next <kbd>J</kbd>
          </button>
        </div>
      </div>
      <div className="notice small" data-testid="uncertain-note">
        Uncertain objects (including dual-channel debris-like specks) are never counted in colour totals until you resolve them.
      </div>
      {engineData && (
        <>
          <ConsensusSummaryCard />
          <EngineFilterBar filter={filter} onChange={(f) => patch({ reviewFilter: f })} />
          <QcSample />
        </>
      )}
      <div className="muted small">Ordered by measured review concerns (not confidence). Enter confirms the selected spot and moves on.</div>
      {noDetection && <div className="empty">No detections yet. Run detection from the Detect tab, or add spots manually.</div>}
      {!noDetection && engineData && filter !== 'all' && listed.length === 0 && <div className="empty" data-testid="filter-empty">No candidates match this engine-status filter.</div>}
      {!noDetection && filter === 'all' && queue.length === 0 && unreviewed.length === 0 && (
        <div className="empty ok">Nothing flagged for review. Still inspect each region for omissions before approving.</div>
      )}
      {!noDetection && filter === 'all' && queue.length === 0 && unreviewed.length > 0 && (
        <div className="empty" data-testid="unreviewed-note">
          {unreviewed.length} object{unreviewed.length > 1 ? 's' : ''} unreviewed (no automatic flags).
          <div className="row">
            <button className="btn sm" onClick={() => stepUnreviewed(-1)} data-testid="unreviewed-prev">
              ↑ Prev unreviewed
            </button>
            <button className="btn sm" onClick={() => stepUnreviewed(1)} data-testid="unreviewed-next">
              ↓ Next unreviewed
            </button>
          </div>
        </div>
      )}
      <ol className="queue">
        {listed.map((q, i) => {
          const o = objects.get(q.object_id);
          const es = o ? objectStatus(o) : null;
          return (
            <li key={q.object_id} className={selection.length === 1 && selection[0] === q.object_id ? 'active' : ''}>
              <button
                className="qitem"
                onClick={() => {
                  patch({ selection: [q.object_id], tool: 'select' });
                  zoom(q.x, q.y, 10);
                }}
                data-testid={`queue-${q.object_id}`}
              >
                <img className="qthumb" src={thumbUrl(a.id, q.object_id, 64)} alt="" loading="lazy" width={52} height={52} onError={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')} />
                <span className="qbody">
                  <span className="qtop">
                    <span className="muted mono">#{i + 1}</span> {o && <GlyphIcon cls={o.cls} size={14} />}
                    <strong className="mono">{shortId(q.object_id)}</strong>
                    {o && <span className="small">{CLASS_LABEL[o.cls]}</span>}
                    {es && <span className="chip" data-testid={`qstatus-${q.object_id}`}>{STATUS_LABEL[es]}</span>}
                  </span>
                  <span className="qreasons small">
                    {q.reasons.length ? q.reasons.map((r) => REASON_SHORT[r] ?? r).join(' · ') : es ? STATUS_LABEL[es] : 'flagged'}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
