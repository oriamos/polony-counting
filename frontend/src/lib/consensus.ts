import type { ConsensusStatus, DetectedObject, EngineRunInfo, QcStatus, ReviewQueueItem } from '../api/types';

/** Review filter by engine status. */
export type EngineFilter = 'all' | 'disagreements' | 'agreement' | 'bigfish_only' | 'spotiflow_only' | 'human_added' | 'unresolved';

export const ENGINE_FILTERS: { id: EngineFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'disagreements', label: 'Disagreements only' },
  { id: 'agreement', label: 'Agreement' },
  { id: 'bigfish_only', label: 'Big-FISH only' },
  { id: 'spotiflow_only', label: 'Spotiflow only' },
  { id: 'human_added', label: 'Human added' },
  { id: 'unresolved', label: 'Unresolved' },
];

export const STATUS_LABEL: Record<ConsensusStatus, string> = {
  agreement: 'both engines',
  bigfish_only: 'Big-FISH only',
  spotiflow_only: 'Spotiflow only',
  human_added: 'human added',
  human_rejected: 'human rejected',
};

const DISAGREE = new Set<ConsensusStatus>(['bigfish_only', 'spotiflow_only']);

/**
 * One status per object. A dual-channel object carries evidence on both channels; a disagreement on either
 * channel wins over an agreement (it needs a human), then human_added, then agreement. null = no engine info.
 */
export function objectStatus(o: Pick<DetectedObject, 'red' | 'green' | 'origin'>): ConsensusStatus | null {
  const ss = [o.red?.consensus_status, o.green?.consensus_status].filter((x): x is ConsensusStatus => !!x);
  if (!ss.length) return null;
  return ss.find((s) => s === 'bigfish_only') ?? ss.find((s) => s === 'spotiflow_only') ?? ss.find((s) => s === 'human_added') ?? ss.find((s) => s === 'human_rejected') ?? 'agreement';
}

export const hasEngineInfo = (objs: DetectedObject[]) => objs.some((o) => objectStatus(o) != null);

export function matchesFilter(o: DetectedObject, f: EngineFilter): boolean {
  if (f === 'all') return true;
  if (f === 'unresolved') return !o.deleted && !o.reviewed;
  const s = objectStatus(o);
  if (f === 'disagreements') return s != null && DISAGREE.has(s);
  return s === f;
}

/**
 * Items for the filtered review list. 'all' = the server queue unchanged. Other filters list matching live objects,
 * keeping the server's order for queued ones (disagreements first) and appending the rest by id.
 */
export function filteredQueue(queue: ReviewQueueItem[], objects: DetectedObject[], f: EngineFilter): ReviewQueueItem[] {
  if (f === 'all') return queue;
  const rank = new Map(queue.map((q, i) => [q.object_id, i]));
  const byId = new Map(queue.map((q) => [q.object_id, q]));
  return objects
    .filter((o) => !o.deleted && matchesFilter(o, f))
    .map((o) => byId.get(o.id) ?? { object_id: o.id, reasons: [], priority: 0, x: o.x, y: o.y })
    .sort((p, q) => (rank.get(p.object_id) ?? 1e9) - (rank.get(q.object_id) ?? 1e9) || p.object_id.localeCompare(q.object_id));
}

export const shortChecksum = (c: string | null | undefined) => (c ? c.slice(0, 8) : '–');

export function runLabel(r: EngineRunInfo): string {
  return r.engine === 'spotiflow' ? 'Spotiflow' : r.engine === 'bigfish' ? 'Big-FISH' : r.engine;
}

/** Wording guard: raw scores are shown as numbers with their own name, never as a percentage. */
export const SCORE_LABEL = { bigfish: 'LoG response (noise units)', spotiflow: 'model score' } as const;

export function triangleD(cx: number, cy: number, r: number): string {
  return `M${cx} ${cy - r}L${cx + r * 0.9} ${cy + r * 0.7}L${cx - r * 0.9} ${cy + r * 0.7}Z`;
}

export const AGREEMENT_NOT_TRUTH =
  'Agreement is not ground truth. Both engines can miss the same faint polony or both mark the same debris; agreement only means two detectors independently proposed a nearby point. Disagreements are reviewed first, and a random sample of agreements is checked so you can see how often they are wrong.';

/** Default fixed QC sample size per slide (a fraction cannot bound the agreement error). */
export const QC_DEFAULT_N = 40;

const pct = (x: number) => `${(100 * x).toFixed(x < 0.1 ? 1 : 0)} %`;

/** One-line statement of the agreement QC result. Never calls agreement “correct”. */
export function qcBoundText(qc: QcStatus | null | undefined): string {
  if (!qc || qc.sample_size === 0) return 'No QC sample drawn yet.';
  if (qc.reviewed === 0 || !qc.ci95 || qc.error_rate == null) return `QC sample of ${qc.sample_size} agreements: none reviewed yet (${qc.pending} pending).`;
  return `Of ${qc.reviewed} reviewed sampled agreements, ${qc.rejected} were rejected: error rate ${pct(qc.error_rate)}, exact 95 % interval ${pct(qc.ci95[0])} to ${pct(qc.ci95[1])} (sample of ${qc.sample_size}${qc.pending ? `, ${qc.pending} still pending` : ''}).`;
}

/** Clamp the requested sample size to [1, eligible] (eligible may be unknown = no upper clamp). */
export function clampQcN(n: number, eligible?: number | null): number {
  const k = Math.max(1, Math.round(Number.isFinite(n) ? n : QC_DEFAULT_N));
  return eligible != null && eligible > 0 ? Math.min(k, eligible) : k;
}
