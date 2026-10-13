import type { Analysis, Counts, CountsResponse, DetectionRequest, EngineChoice, Job, Region } from '../api/types';
import { DEFAULT_SPOTIFLOW } from '../api/types';

/** Batch upload: one analysis per TIFF (each file is a multi-channel scan), run in sequence. */

export type BatchStage = 'queued' | 'uploading' | 'mapping' | 'region' | 'detecting' | 'counting' | 'done' | 'failed' | 'cancelled';

export interface BatchRow {
  file: string;
  stage: BatchStage;
  analysisId: string | null;
  progress: number; // 0..1 of the detection job
  counts: Counts | null;
  regionNote: string;
  notes: string[];
  error: string;
}

export interface BatchOptions {
  engine: EngineChoice;
  autoChamber: boolean;
  rimPx: number;
}

/** The slice of the API client the batch needs (injected so the flow is testable). */
export interface BatchApi {
  createAnalysis: (files: File[], name?: string) => Promise<Analysis>;
  setChannels: (id: string, mapping: { index: number; role: 'red' | 'green' | 'ignore' }[]) => Promise<Analysis>;
  autoChamber: (id: string, rim: number) => Promise<{ polygon: [number, number][]; warnings: string[] }>;
  putRegions: (id: string, regions: Region[]) => Promise<Analysis>;
  detect: (id: string, req: DetectionRequest) => Promise<Job>;
  job: (id: string) => Promise<Job>;
  getAnalysis: (id: string) => Promise<Analysis>;
  counts: (id: string) => Promise<CountsResponse>;
}

export const isTiff = (name: string) => /\.tiff?$/i.test(name);

export const newRow = (file: string): BatchRow => ({
  file, stage: 'queued', analysisId: null, progress: 0, counts: null, regionNote: '', notes: [], error: '',
});

/** Strip the extension for the analysis name. */
export const analysisName = (file: string) => file.replace(/\.tiff?$/i, '');

/** Detection request sent for a batch. Everything not listed keeps the server default. */
export function batchRequest(engine: EngineChoice): DetectionRequest {
  const req: Partial<DetectionRequest> = { engine, preview_window: null };
  if (engine !== 'bigfish') req.spotiflow = { ...DEFAULT_SPOTIFLOW, downscale: 4, normalizer: 'valid_unsaturated' };
  return req as DetectionRequest;
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Process the files one after another. Channel mapping is fixed by the caller (the user confirmed
 * "page 0 = red, page 1 = green" once for the whole batch); it is never inferred. `onRow` is called
 * after every change. A failing file never stops the others.
 */
export async function runBatch(
  files: File[],
  opts: BatchOptions,
  api: BatchApi,
  onRow: (index: number, row: BatchRow) => void,
  cancelled: () => boolean,
  pollMs = 600,
): Promise<BatchRow[]> {
  const rows = files.map((f) => newRow(f.name));
  const set = (i: number, patch: Partial<BatchRow>) => {
    rows[i] = { ...rows[i], ...patch };
    onRow(i, rows[i]);
  };
  for (let i = 0; i < files.length; i++) {
    if (cancelled()) {
      set(i, { stage: 'cancelled' });
      continue;
    }
    try {
      set(i, { stage: 'uploading' });
      const a = await api.createAnalysis([files[i]], analysisName(files[i].name));
      const id = a.id;
      set(i, { analysisId: id, stage: 'mapping' });
      await api.setChannels(id, [
        { index: 0, role: 'red' },
        { index: 1, role: 'green' },
      ]);
      if (opts.autoChamber) {
        set(i, { stage: 'region' });
        try {
          const ch = await api.autoChamber(id, opts.rimPx);
          const region: Region = {
            id: 'chamber', kind: 'counting', name: `Chamber (rim ${opts.rimPx}px)`, polygon: ch.polygon,
            approved: false, approved_at: null, approval_fingerprint: null, inspected: false, field_index: null,
          };
          await api.putRegions(id, [region]);
          set(i, { regionNote: region.name, notes: [...rows[i].notes, ...ch.warnings] });
        } catch (e) {
          set(i, { regionNote: 'none', notes: [...rows[i].notes, `automatic chamber failed (${errText(e)}); no counting region set, count not restricted`] });
        }
      }
      set(i, { stage: 'detecting', progress: 0 });
      let job = await api.detect(id, batchRequest(opts.engine));
      while (job.status === 'queued' || job.status === 'running') {
        await sleep(pollMs);
        job = await api.job(job.id);
        set(i, { progress: job.progress });
      }
      if (job.status !== 'done') throw new Error(job.error || `detection ${job.status}`);
      set(i, { stage: 'counting', progress: 1, notes: job.message ? [...rows[i].notes, job.message] : rows[i].notes });
      let an = await api.getAnalysis(id);
      for (let k = 0; k < 8 && an.status === 'detecting'; k++) {
        await sleep(pollMs);
        an = await api.getAnalysis(id);
      }
      const c = await api.counts(id);
      set(i, { stage: 'done', counts: c.slide });
    } catch (e) {
      set(i, { stage: 'failed', error: errText(e) });
    }
  }
  return rows;
}

const csvCell = (v: unknown) => {
  const s = v == null ? '' : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export const SUMMARY_HEADER = [
  'file', 'analysis_id', 'state', 'export_status', 'engine', 'counting_region',
  'red_only', 'green_only', 'yellow', 'uncertain', 'red_positive', 'green_positive',
  'total_objects_exclusive', 'red_positive_excl_debris', 'green_positive_excl_debris', 'notes', 'error',
] as const;

/** One CSV row per file. Always DRAFT: nothing here has been reviewed or approved. */
export function summaryCsv(rows: BatchRow[], engine: EngineChoice): string {
  const lines = [SUMMARY_HEADER.join(',')];
  for (const r of rows) {
    const c = r.counts;
    lines.push(
      [
        r.file, r.analysisId ?? '', r.stage, 'DRAFT', engine, r.regionNote,
        c?.red_only, c?.green_only, c?.yellow, c?.uncertain, c?.red_positive, c?.green_positive,
        c?.total_objects_exclusive, c?.red_positive_excl_debris, c?.green_positive_excl_debris,
        r.notes.join(' | '), r.error,
      ].map(csvCell).join(','),
    );
  }
  return lines.join('\r\n') + '\r\n';
}
