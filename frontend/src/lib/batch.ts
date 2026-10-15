import type { Analysis, CompareCountType, CompareResponse, Counts, CountsResponse, DetectionRequest, EngineChoice, ExaminedFieldsResult, FieldImportRequest, Job, ReferenceInfo, Region } from '../api/types';
import { DEFAULT_SPOTIFLOW } from '../api/types';

/** Batch upload: one analysis per TIFF (each file is a multi-channel scan), run in sequence. */

export type BatchStage = 'queued' | 'uploading' | 'mapping' | 'region' | 'detecting' | 'counting' | 'done' | 'failed' | 'cancelled';

/**
 * Agreement of the app's per-field counts with the researcher's embedded marks for one plane / count type, over
 * the fields the researcher provably examined on THAT plane only. A field without marks on the plane is never read as zero.
 * Metrics are null when no field is eligible.
 */
export interface CompareLine {
  plane: number;
  countType: CompareCountType;
  /** all field regions (examined or not) */
  nFields: number;
  eligibleExaminedFields: number;
  positiveMarkedFields: number;
  verifiedZeroFields: number;
  unknownFieldsExcluded: number;
  conflictFieldsExcluded: number;
  eligibilitySource: string;
  pearsonR: number | null;
  slope: number | null;
  mae: number | null;
  totalReference: number | null;
  totalApp: number | null;
}

/** Map a compare response to one batch line (the only place the response shape is read). */
export function compareLine(plane: number, countType: CompareCountType, cmp: CompareResponse): CompareLine {
  const sm = cmp.summary;
  return {
    plane, countType, nFields: sm.n_fields,
    eligibleExaminedFields: sm.eligible_examined_fields ?? 0, positiveMarkedFields: sm.positive_marked_fields ?? 0,
    verifiedZeroFields: sm.verified_zero_fields ?? 0, unknownFieldsExcluded: sm.unknown_fields_excluded ?? 0,
    conflictFieldsExcluded: sm.conflict_fields_excluded ?? 0, eligibilitySource: sm.eligibility_source ?? 'none',
    pearsonR: sm.pearson_r ?? null, slope: sm.slope ?? null, mae: sm.mae ?? null,
    totalReference: sm.total_reference ?? null, totalApp: sm.total_app ?? null,
  };
}

export interface BatchRow {
  file: string;
  stage: BatchStage;
  analysisId: string | null;
  progress: number; // 0..1 of the detection job
  counts: Counts | null;
  compare: CompareLine[];
  /** 'off' | 'no marks' | 'fields imported' | error text */
  refStatus: string;
  regionNote: string;
  notes: string[];
  error: string;
}

export interface BatchOptions {
  engine: EngineChoice;
  autoChamber: boolean;
  rimPx: number;
  /** import the researcher's fields (when the TIFF carries them) and compare per field */
  compare: boolean;
  /** optional examined-fields manifest (JSON/CSV), uploaded to every analysis before the fields are imported */
  manifest?: File | null;
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
  reference: (id: string) => Promise<ReferenceInfo>;
  importFields: (id: string, body: FieldImportRequest) => Promise<Analysis>;
  compareFields: (id: string, plane: number | null, countType: CompareCountType) => Promise<CompareResponse>;
  uploadExamined?: (id: string, file: File) => Promise<ExaminedFieldsResult>;
}

export const isTiff = (name: string) => /\.tiff?$/i.test(name);

export const newRow = (file: string): BatchRow => ({
  file, stage: 'queued', analysisId: null, progress: 0, counts: null, compare: [], refStatus: 'off', regionNote: '', notes: [], error: '',
});

/** Strip the extension for the analysis name. */
export const analysisName = (file: string) => file.replace(/\.tiff?$/i, '');

/** Detection request sent for a batch. Everything not listed keeps the server default. */
export function batchRequest(engine: EngineChoice): DetectionRequest {
  const req: Partial<DetectionRequest> = { engine, preview_window: null };
  if (engine !== 'bigfish') req.spotiflow = { ...DEFAULT_SPOTIFLOW, downscale: 4, normalizer: 'valid_unsaturated' };
  return req as DetectionRequest;
}

/** Plane (TIFF page) -> app count types compared against the marks of that plane. */
const COMPARE_PLAN: [number, CompareCountType[]][] = [
  [0, ['red_positive', 'yellow']],
  [1, ['green_positive', 'yellow']],
];

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
      let hasFields = false;
      if (opts.compare) {
        try {
          const ref = await api.reference(id);
          if (ref.available && ref.grid) {
            const withManifest = !!(opts.manifest && api.uploadExamined);
            if (withManifest) {
              const m = await api.uploadExamined!(id, opts.manifest as File);
              set(i, { notes: [...rows[i].notes, `examined-fields manifest: ${m.n_entries} entr${m.n_entries === 1 ? 'y' : 'ies'} for this file`, ...m.warnings] });
            }
            // 'counted' = fields with marks on any page. Which plane a field was examined on is decided per plane at compare time.
            // With a manifest every grid cell is imported so that counted-zero fields have a region to be compared in.
            await api.importFields(id, { mode: withManifest ? 'all' : 'counted', shift: [0, 0], plane: null, replace: false });
            hasFields = true;
            set(i, { refStatus: 'fields imported', notes: [...rows[i].notes, ...ref.warnings] });
          } else {
            set(i, { refStatus: 'no marks' });
          }
        } catch (e) {
          set(i, { refStatus: `reference failed: ${errText(e)}` });
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
      const lines: CompareLine[] = [];
      if (hasFields) {
        // page 0 / page 1 are the red / green channels the user confirmed for the batch
        for (const [plane, types] of COMPARE_PLAN) {
          for (const t of types) {
            try {
              const cmp = await api.compareFields(id, plane, t);
              lines.push(compareLine(plane, t, cmp));
            } catch (e) {
              set(i, { notes: [...rows[i].notes, `compare plane ${plane} ${t}: ${errText(e)}`] });
            }
          }
        }
      }
      set(i, { stage: 'done', counts: c.slide, compare: lines });
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
  'total_objects_exclusive', 'red_positive_excl_debris', 'green_positive_excl_debris', 'researcher_marks', 'notes', 'error',
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
        r.refStatus, r.notes.join(' | '), r.error,
      ].map(csvCell).join(','),
    );
  }
  return lines.join('\r\n') + '\r\n';
}

/**
 * `fields_compared` was dropped: it counted every imported field. Use `eligible_examined_fields` (= positive_marked +
 * verified_zero); unknown/conflict fields are counted separately and enter no total or metric.
 */
export const COMPARE_HEADER = [
  'file', 'analysis_id', 'export_status', 'reference_plane', 'app_count_type',
  'eligible_examined_fields', 'positive_marked_fields', 'verified_zero_fields', 'unknown_fields_excluded', 'conflict_fields_excluded',
  'eligibility_source', 'total_reference', 'total_app', 'delta_total', 'pearson_r', 'slope', 'mae',
] as const;

/** Long format: one row per file x plane x count type. Agreement with the researcher's marks, not accuracy. Blank = not defined. */
export function comparisonCsv(rows: BatchRow[]): string {
  const lines = [COMPARE_HEADER.join(',')];
  for (const r of rows) {
    for (const c of r.compare) {
      lines.push(
        [
          r.file, r.analysisId ?? '', 'DRAFT', c.plane, c.countType,
          c.eligibleExaminedFields, c.positiveMarkedFields, c.verifiedZeroFields, c.unknownFieldsExcluded, c.conflictFieldsExcluded,
          c.eligibilitySource, c.totalReference, c.totalApp,
          c.totalReference == null || c.totalApp == null ? '' : c.totalApp - c.totalReference,
          c.pearsonR == null ? '' : c.pearsonR.toFixed(3), c.slope == null ? '' : c.slope.toFixed(3), c.mae == null ? '' : c.mae.toFixed(2),
        ].map(csvCell).join(','),
      );
    }
  }
  return lines.join('\r\n') + '\r\n';
}
