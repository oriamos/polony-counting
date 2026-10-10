import { describe, expect, it } from 'vitest';
import type { DetectedObject, ReviewQueueItem } from '../api/types';
import type { QcStatus } from '../api/types';
import { clampQcN, filteredQueue, matchesFilter, objectStatus, qcBoundText, QC_DEFAULT_N, triangleD } from './consensus';

const ev = (s?: string) => ({ detected: true, x: 1, y: 1, peak_intensity: 1, local_background: 1, snr: 1, saturated: false, spot_id: null, consensus_status: s ?? null }) as DetectedObject['red'];
const none = () => ({ ...ev(), detected: false });
const obj = (id: string, red?: string, green?: string, reviewed = false): DetectedObject =>
  ({ id, x: 1, y: 2, cls: 'red', origin: 'auto', red: red ? ev(red) : none(), green: green ? ev(green) : none(), association_distance_px: null, review_reasons: [], ambiguous_with: [], reviewed, deleted: false, region_id: null, history: [] }) as DetectedObject;

describe('consensus helpers', () => {
  it('disagreement on either channel wins over agreement', () => {
    expect(objectStatus(obj('a', 'agreement', 'spotiflow_only'))).toBe('spotiflow_only');
    expect(objectStatus(obj('a', 'agreement'))).toBe('agreement');
    expect(objectStatus(obj('a'))).toBeNull();
  });
  it('filters', () => {
    expect(matchesFilter(obj('a', 'bigfish_only'), 'disagreements')).toBe(true);
    expect(matchesFilter(obj('a', 'agreement'), 'disagreements')).toBe(false);
    expect(matchesFilter(obj('a', 'agreement', undefined, true), 'unresolved')).toBe(false);
    expect(matchesFilter(obj('a', 'human_added'), 'human_added')).toBe(true);
  });
  it('keeps the server order, then appends unqueued matches', () => {
    const q: ReviewQueueItem[] = [
      { object_id: 'b', reasons: ['spotiflow_only'], priority: 9, x: 0, y: 0 },
      { object_id: 'a', reasons: ['bigfish_only'], priority: 8, x: 0, y: 0 },
    ];
    const objs = [obj('a', 'bigfish_only'), obj('b', 'spotiflow_only'), obj('c', 'agreement'), obj('d', 'agreement')];
    expect(filteredQueue(q, objs, 'all')).toBe(q);
    expect(filteredQueue(q, objs, 'disagreements').map((x) => x.object_id)).toEqual(['b', 'a']);
    expect(filteredQueue(q, objs, 'agreement').map((x) => x.object_id)).toEqual(['c', 'd']);
  });
  it('triangle path is closed', () => expect(triangleD(0, 0, 3)).toMatch(/Z$/));
});

describe('agreement QC wording', () => {
  const base: QcStatus = { sample_size: 40, reviewed: 0, pending: 40, confirmed: 0, rejected: 0, population_agreements: 203, seed: 1, error_rate: null, ci95: null, interval_method: 'Clopper-Pearson', n_for_margin_0_05: 101, n_for_margin_0_05_conservative: 134, planning_error_rate: 0.153, note: '' };
  it('defaults to a fixed n of 40 and clamps to the eligible agreements', () => {
    expect(QC_DEFAULT_N).toBe(40);
    expect(clampQcN(40, 24)).toBe(24);
    expect(clampQcN(0, 100)).toBe(1);
    expect(clampQcN(Number.NaN, 100)).toBe(40);
    expect(clampQcN(500)).toBe(500);
  });
  it('states no bound before review, then the interval with the sample size', () => {
    expect(qcBoundText(null)).toMatch(/No QC sample/);
    expect(qcBoundText(base)).toMatch(/none reviewed yet \(40 pending\)/);
    const t = qcBoundText({ ...base, reviewed: 40, pending: 0, confirmed: 34, rejected: 6, error_rate: 0.15, ci95: [0.057, 0.298] });
    expect(t).toMatch(/6 were rejected/);
    expect(t).toMatch(/15 %/);
    expect(t).toMatch(/exact 95 % interval 5\.7 % to 30 %/);
    expect(t).toMatch(/sample of 40/);
    expect(t).not.toMatch(/correct|verified|accuracy/i);
  });
});
