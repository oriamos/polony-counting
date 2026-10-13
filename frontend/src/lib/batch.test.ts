import { describe, expect, it } from 'vitest';
import type { Analysis, CompareResponse, Counts, Job, ReferenceInfo } from '../api/types';
import { analysisName, batchRequest, compareLine, comparisonCsv, COMPARE_HEADER, isTiff, runBatch, summaryCsv, SUMMARY_HEADER, type BatchApi, type BatchRow } from './batch';

const counts = (n: number): Counts => ({
  scope: 'slide', red_only: n, green_only: 1, yellow: 2, uncertain: 3, red_positive: n + 2, green_positive: 3,
  total_objects_exclusive: n + 3, reporting_mode: 'exclusive', is_final: false,
});
const file = (name: string) => new File(['x'], name);
const job = (status: Job['status'], extra: Partial<Job> = {}): Job => ({
  id: 'j', analysis_id: 'a', status, progress: 0.5, message: '', error: null, started_at: null, finished_at: null, ...extra,
});

const cmpResp = (kind: 'marks' | 'none'): CompareResponse => ({
  rows: [],
  summary: kind === 'marks'
    ? { n_fields: 8, eligible_examined_fields: 6, positive_marked_fields: 5, verified_zero_fields: 1, unknown_fields_excluded: 2, conflict_fields_excluded: 0,
        eligibility_source: 'embedded_imagej_marks+sidecar_csv', pearson_r: 0.89, slope: 0.96, mae: 2.38, total_reference: 209, total_app: 204 }
    : { n_fields: 8, eligible_examined_fields: 0, positive_marked_fields: 0, verified_zero_fields: 0, unknown_fields_excluded: 8, conflict_fields_excluded: 0,
        eligibility_source: 'none', pearson_r: null, slope: null, mae: null, total_reference: null, total_app: null },
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
    reference: async () => ({ available: true, points: [], grid: { cells: [] }, reference_counts: {}, planes: [], warnings: [] } as unknown as ReferenceInfo),
    importFields: async (id) => (log.push(`fields ${id}`), {} as Analysis),
    compareFields: async (id, plane, t) => (log.push(`compare ${id} ${plane} ${t}`), cmpResp(plane === 1 ? 'none' : 'marks')),
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
    const rows = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: true, rimPx: 40, compare: false }, fakeApi({}, log), () => {}, () => false, 0);
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
    const rows = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: false }, api, () => {}, () => false, 0);
    expect(rows[0].stage).toBe('failed');
    expect(rows[0].error).toBe('boom');
    expect(rows[1].stage).toBe('done');
  });

  it('a missing chamber is a note, not a failure', async () => {
    const api = fakeApi({ autoChamber: async () => { throw new Error('chamber_not_found'); } });
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: true, rimPx: 40, compare: false }, api, () => {}, () => false, 0);
    expect(rows[0].stage).toBe('done');
    expect(rows[0].regionNote).toBe('none');
    expect(rows[0].notes.join()).toContain('chamber_not_found');
  });

  it('reports a failed detection job and supports cancel', async () => {
    const api = fakeApi({ job: async () => job('failed', { error: 'out of memory' }) });
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: false }, api, () => {}, () => false, 0);
    expect(rows[0]).toMatchObject({ stage: 'failed', error: 'out of memory' });
    const cancelled = await runBatch([file('a.tif'), file('b.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: false }, fakeApi(), () => {}, () => true, 0);
    expect(cancelled.map((r) => r.stage)).toEqual(['cancelled', 'cancelled']);
  });
});

describe('comparison with the researcher marks', () => {
  it('imports fields before detection and compares planes 0 and 1', async () => {
    const log: string[] = [];
    const api = fakeApi({}, log);
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: true, rimPx: 40, compare: true }, api, () => {}, () => false, 0);
    expect(log.slice(0, 4)).toEqual(['create a', 'map a1 red/green', 'regions a1', 'fields a1']);
    expect(log.indexOf('fields a1')).toBeLessThan(log.indexOf('detect a1 bigfish'));
    expect(log.filter((l) => l.startsWith('compare'))).toEqual([
      'compare a1 0 red_positive', 'compare a1 0 yellow', 'compare a1 1 green_positive', 'compare a1 1 yellow',
    ]);
    expect(rows[0].refStatus).toBe('fields imported');
    expect(rows[0].compare[0]).toMatchObject({ plane: 0, countType: 'red_positive', totalReference: 209, totalApp: 204, eligibleExaminedFields: 6, verifiedZeroFields: 1, unknownFieldsExcluded: 2 });
    expect(rows[0].compare[2]).toMatchObject({ plane: 1, eligibleExaminedFields: 0, totalReference: null, pearsonR: null, mae: null, eligibilitySource: 'none' });
  });

  it('a TIFF without marks is noted, not an error', async () => {
    const api = fakeApi({ reference: async () => ({ available: false, points: [], grid: null, reference_counts: {}, planes: [], warnings: [] }) });
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: true }, api, () => {}, () => false, 0);
    expect(rows[0]).toMatchObject({ stage: 'done', refStatus: 'no marks', compare: [] });
  });

  it('writes one long-format row per file, plane and count type', async () => {
    const rows = await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: true }, fakeApi(), () => {}, () => false, 0);
    const lines = comparisonCsv(rows).trim().split('\r\n');
    expect(lines[0]).toBe(COMPARE_HEADER.join(','));
    expect(lines).toHaveLength(5);
    expect(COMPARE_HEADER.join(',')).toBe(
      'file,analysis_id,export_status,reference_plane,app_count_type,eligible_examined_fields,positive_marked_fields,verified_zero_fields,unknown_fields_excluded,conflict_fields_excluded,eligibility_source,total_reference,total_app,delta_total,pearson_r,slope,mae');
    expect(lines[1]).toBe('a.tif,a1,DRAFT,0,red_positive,6,5,1,2,0,embedded_imagej_marks+sidecar_csv,209,204,-5,0.890,0.960,2.38');
    // page 1: no field was examined -> no artificial zeros, blank (not 0 / NaN) metrics
    expect(lines[3]).toBe('a.tif,a1,DRAFT,1,green_positive,0,0,0,8,0,none,,,,,,');
    expect(lines[3].split(',')).toHaveLength(COMPARE_HEADER.length);
  });
});

describe('summaryCsv', () => {
  it('writes a header, one DRAFT row per file and quotes awkward text', () => {
    const rows: BatchRow[] = [
      { file: 'a,b.tif', stage: 'done', analysisId: 'x1', progress: 1, counts: counts(7), compare: [], refStatus: 'off', regionNote: 'Chamber', notes: ['n1', 'say "hi"'], error: '' },
      { file: 'c.tif', stage: 'failed', analysisId: null, progress: 0, counts: null, compare: [], refStatus: 'off', regionNote: '', notes: [], error: 'boom' },
    ];
    const lines = summaryCsv(rows, 'hybrid').trim().split('\r\n');
    expect(lines[0]).toBe(SUMMARY_HEADER.join(','));
    expect(lines).toHaveLength(3);
    expect(lines[1].startsWith('"a,b.tif",x1,done,DRAFT,hybrid,Chamber,7,1,2,3,9,3,10')).toBe(true);
    expect(lines[1]).toContain(',off,"n1 | say')
    expect(lines[1]).toContain('"n1 | say ""hi"""');
    expect(lines[2]).toBe('c.tif,,failed,DRAFT,hybrid,,,,,,,,,,,off,,boom');
  });
});

describe('compareLine', () => {
  it('maps the compare summary 1:1 and keeps nulls', () => {
    expect(compareLine(1, 'yellow', cmpResp('none'))).toEqual({
      plane: 1, countType: 'yellow', nFields: 8, eligibleExaminedFields: 0, positiveMarkedFields: 0, verifiedZeroFields: 0,
      unknownFieldsExcluded: 8, conflictFieldsExcluded: 0, eligibilitySource: 'none', pearsonR: null, slope: null, mae: null, totalReference: null, totalApp: null,
    });
    expect(compareLine(0, 'red_positive', cmpResp('marks'))).toMatchObject({ eligibleExaminedFields: 6, totalReference: 209, slope: 0.96 });
  });
});

describe('examined-fields manifest in the batch', () => {
  it('uploads it before importing all grid cells', async () => {
    const log: string[] = [];
    const api = fakeApi({
      uploadExamined: async (id, f) => (log.push(`manifest ${id} ${f.name}`), { matched_file_entries: 1, pages: { '0': 2 }, n_entries: 2, warnings: [] }),
      importFields: async (id, b) => (log.push(`fields ${id} ${b.mode}`), {} as Analysis),
    }, log);
    await runBatch([file('a.tif')], { engine: 'bigfish', autoChamber: false, rimPx: 40, compare: true, manifest: new File(['{}'], 'm.json') }, api, () => {}, () => false, 0);
    expect(log.indexOf('manifest a1 m.json')).toBeLessThan(log.indexOf('fields a1 all'));
  });
});
