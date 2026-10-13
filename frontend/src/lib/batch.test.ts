import { describe, expect, it } from 'vitest';
import type { Analysis, Counts, Job } from '../api/types';
import { analysisName, batchRequest, isTiff, runBatch, summaryCsv, SUMMARY_HEADER, type BatchApi, type BatchRow } from './batch';

const counts = (n: number): Counts => ({
  scope: 'slide', red_only: n, green_only: 1, yellow: 2, uncertain: 3, red_positive: n + 2, green_positive: 3,
  total_objects_exclusive: n + 3, reporting_mode: 'exclusive', is_final: false,
});
const file = (name: string) => new File(['x'], name);
const job = (status: Job['status'], extra: Partial<Job> = {}): Job => ({
  id: 'j', analysis_id: 'a', status, progress: 0.5, message: '', error: null, started_at: null, finished_at: null, ...extra,
});

function fakeApi(over: Partial<BatchApi> = {}, log: string[] = []): BatchApi {
  let n = 0;
  return {
    createAnalysis: async (_f, name) => (log.push(`create ${name}`), { id: `a${++n}`, status: 'needs_channel_mapping' } as Analysis),
    setChannels: async (id, m) => (log.push(`map ${id} ${m.map((x) => x.role).join('/')}`), {} as Analysis),
    autoChamber: async () => ({ polygon: [[0, 0], [1, 0], [1, 1]] as [number, number][], warnings: [] }),
    putRegions: async (id) => (log.push(`regions ${id}`), {} as Analysis),
    detect: async (id, req) => (log.push(`detect ${id} ${req.engine}`), job('running')),
    job: async () => job('done', { message: '5 objects detected' }),
    getAnalysis: async () => ({ status: 'in_review' } as Analysis),
    counts: async () => ({ regions: [], slide: counts(10) }),
    ...over,
  };
}

describe('batch helpers', () => {
  it('recognises tiffs and names analyses after the file', () => {
    expect(isTiff('a.TIF')).toBe(true);
    expect(isTiff('a.png')).toBe(false);
    expect(analysisName('29-09_AI-A.tif')).toBe('29-09_AI-A');
  });
  it('sends Spotiflow settings only when it is used', () => {
    expect(batchRequest('bigfish').spotiflow).toBeUndefined();
    const h = batchRequest('hybrid');
    expect(h.engine).toBe('hybrid');
    expect(h.spotiflow?.downscale).toBe(4);
    expect(h.spotiflow?.normalizer).toBe('valid_unsaturated');
  });
});

describe('runBatch', () => {
  it('runs every file in order: create, fixed mapping, chamber, detect, counts', async () => {
    const log: string[] = [];
    const rows = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: true, rimPx: 40 }, fakeApi({}, log), () => {}, () => false, 0);
    expect(log).toEqual([
      'create a', 'map a1 red/green', 'regions a1', 'detect a1 bigfish',
      'create b', 'map a2 red/green', 'regions a2', 'detect a2 bigfish',
    ]);
    expect(rows.map((r) => r.stage)).toEqual(['done', 'done']);
    expect(rows[0].counts?.red_only).toBe(10);
    expect(rows[0].regionNote).toContain('Chamber');
  });

  it('a failing file does not stop the others', async () => {
    let calls = 0;
    const api = fakeApi({ createAnalysis: async (_f, name) => { if (calls++ === 0) throw new Error('boom'); return { id: `ok-${name}` } as Analysis; } });
    const rows = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40 }, api, () => {}, () => false, 0);
    expect(rows[0].stage).toBe('failed');
    expect(rows[0].error).toBe('boom');
    expect(rows[1].stage).toBe('done');
  });

  it('a missing chamber is a note, not a failure', async () => {
    const api = fakeApi({ autoChamber: async () => { throw new Error('chamber_not_found'); } });
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: true, rimPx: 40 }, api, () => {}, () => false, 0);
    expect(rows[0].stage).toBe('done');
    expect(rows[0].regionNote).toBe('none');
    expect(rows[0].notes.join()).toContain('chamber_not_found');
  });

  it('reports a failed detection job and supports cancel', async () => {
    const api = fakeApi({ job: async () => job('failed', { error: 'out of memory' }) });
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40 }, api, () => {}, () => false, 0);
    expect(rows[0]).toMatchObject({ stage: 'failed', error: 'out of memory' });
    const cancelled = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40 }, fakeApi(), () => {}, () => true, 0);
    expect(cancelled.map((r) => r.stage)).toEqual(['cancelled', 'cancelled']);
  });
});

describe('summaryCsv', () => {
  it('writes a header, one DRAFT row per file and quotes awkward text', () => {
    const rows: BatchRow[] = [
      { file: 'a,b.tif', stage: 'done', analysisId: 'x1', progress: 1, counts: counts(7), regionNote: 'Chamber', notes: ['n1', 'say "hi"'], error: '' },
      { file: 'c.tif', stage: 'failed', analysisId: null, progress: 0, counts: null, regionNote: '', notes: [], error: 'boom' },
    ];
    const lines = summaryCsv(rows, 'hybrid').trim().split('\r\n');
    expect(lines[0]).toBe(SUMMARY_HEADER.join(','));
    expect(lines).toHaveLength(3);
    expect(lines[1].startsWith('"a,b.tif",x1,done,DRAFT,hybrid,Chamber,7,1,2,3,9,3,10')).toBe(true);
    expect(lines[1]).toContain('"n1 | say ""hi"""');
    expect(lines[2]).toBe('c.tif,,failed,DRAFT,hybrid,,,,,,,,,,,,boom');
  });
});
