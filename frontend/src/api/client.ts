import type {
  ActionType,
  Analysis,
  AnalysisSummary,
  ApiErrorBody,
  AutoChamberResponse,
  CompareCountType,
  CompareResponse,
  ExaminedFieldsResult,
  FieldImportRequest,
  GridRequest,
  ReferenceInfo,
  CountsResponse,
  CsvImport,
  ConsensusSummaryResponse,
  QcSampleResponse,
  EngineInfo,
  EngineName,
  FsListing,
  SaveSelection,
  SaveSlideReport,
  EngineRunInfo,
  ExperimentDeleteResult,
  ExperimentGrid,
  ExperimentImportResult,
  ExperimentRun,
  ExperimentRunRequest,
  ExperimentSummary,
  ExperimentView,
  RawDetection,
  DetectionRequest,
  HealthResponse,
  Job,
  PreviewResponse,
  Region,
  ReportingMode,
  ReviewQueueItem,
  SlideMeta,
} from './types';

export class ApiError extends Error {
  code: string;
  status: number;
  detail: Record<string, unknown> | null;
  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.status = status;
    this.code = body.code;
    this.detail = body.detail ?? null;
  }
}

/** Normalise the contract's ApiError body and FastAPI's default {detail: ...} shapes. */
export function parseErrorBody(status: number, text: string): ApiErrorBody {
  try {
    const j = JSON.parse(text);
    if (j && typeof j.message === 'string') return { code: j.code ?? `http_${status}`, message: j.message, detail: j.detail };
    if (j && j.detail && typeof j.detail === 'object' && !Array.isArray(j.detail) && typeof j.detail.message === 'string') {
      return { code: j.detail.code ?? `http_${status}`, message: j.detail.message, detail: j.detail.detail };
    }
    if (j && typeof j.detail === 'string') return { code: `http_${status}`, message: j.detail };
    if (j && Array.isArray(j.detail)) {
      return {
        code: 'validation_error',
        message: j.detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).join('.')}: ${d.msg}`).join('; '),
      };
    }
  } catch {
    /* not JSON */
  }
  return { code: `http_${status}`, message: text.trim().slice(0, 300) || `HTTP ${status}` };
}

async function request<T>(method: string, path: string, body?: unknown, form?: FormData): Promise<T> {
  const init: RequestInit = { method };
  if (form) init.body = form;
  else if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch (e) {
    throw new ApiError(0, { code: 'network', message: 'Cannot reach the backend (network error).' });
  }
  const text = await res.text();
  if (!res.ok) throw new ApiError(res.status, parseErrorBody(res.status, text));
  return (text ? JSON.parse(text) : (undefined as unknown)) as T;
}

/** per page and engine cap for lazily fetched raw detections */
export const RAW_LIMIT = 5000;

export const api = {
  health: () => request<HealthResponse>('GET', '/health'),
  listAnalyses: () => request<AnalysisSummary[]>('GET', '/analyses'),
  getAnalysis: (id: string) => request<Analysis>('GET', `/analyses/${id}`),
  deleteAnalysis: (id: string) => request<void>('DELETE', `/analyses/${id}`),
  createAnalysis: (files: File[], name?: string) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f, f.name));
    if (name) fd.append('name', name);
    return request<Analysis>('POST', '/analyses', undefined, fd);
  },
  fsList: (path?: string) => request<FsListing>('GET', `/fs/list${path ? `?path=${encodeURIComponent(path)}` : ''}`),
  openPath: (path: string, name?: string) => request<Analysis>('POST', '/analyses/open-path', { path, name }),
  setGridField: (id: string, field_index: number, counted: boolean) =>
    request<Analysis>('PUT', `/analyses/${id}/grid-selection`, { field_index, counted }),
  putGrid: (id: string, body: GridRequest) => request<Analysis>('PUT', `/analyses/${id}/grid`, body),
  saveSlide: (id: string, selections: SaveSelection[], output_dir?: string) =>
    request<SaveSlideReport>('POST', `/analyses/${id}/save-slide`, { selections, output_dir: output_dir || null }),
  setChannels: (id: string, mapping: { index: number; role: 'red' | 'green' | 'ignore' }[]) =>
    request<Analysis>('PUT', `/analyses/${id}/channels`, { mapping }),
  putRegions: (id: string, regions: Region[]) => request<Analysis>('PUT', `/analyses/${id}/regions`, { regions }),
  putSettings: (id: string, s: { reporting_mode?: ReportingMode; slide?: SlideMeta; detection_request?: DetectionRequest }) =>
    request<Analysis>('PUT', `/analyses/${id}/settings`, s),
  detect: (id: string, req: DetectionRequest) => request<Job>('POST', `/analyses/${id}/detect`, req),
  preview: (id: string, req: DetectionRequest) => request<PreviewResponse>('POST', `/analyses/${id}/detect/preview`, req),
  job: (id: string) => request<Job>('GET', `/jobs/${id}`),
  cancelJob: (id: string) => request<Job | void>('POST', `/jobs/${id}/cancel`),
  counts: (id: string) => request<CountsResponse>('GET', `/analyses/${id}/counts`),
  reviewQueue: (id: string) => request<ReviewQueueItem[]>('GET', `/analyses/${id}/review-queue`),
  action: (id: string, type: ActionType, payload: Record<string, unknown>) =>
    request<Analysis>('POST', `/analyses/${id}/actions`, { type, payload }),
  undo: (id: string) => request<Analysis>('POST', `/analyses/${id}/undo`),
  redo: (id: string) => request<Analysis>('POST', `/analyses/${id}/redo`),
  finalize: (id: string) => request<Analysis>('POST', `/analyses/${id}/finalize`),
  reopen: (id: string) => request<Analysis>('POST', `/analyses/${id}/reopen`),
  reference: (id: string) => request<ReferenceInfo>('GET', `/analyses/${id}/reference`),
  importFields: (id: string, body: FieldImportRequest) => request<Analysis>('POST', `/analyses/${id}/fields/import`, body),
  compareFields: (id: string, plane: number | null, countType: CompareCountType) =>
    request<CompareResponse>('GET', `/analyses/${id}/fields/compare?${plane != null ? `plane=${plane}&` : ''}count_type=${countType}`),
  uploadExamined: (id: string, file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return request<ExaminedFieldsResult>('POST', `/analyses/${id}/examined-fields`, undefined, fd);
  },
  autoChamber: (id: string, rim_px: number) => request<AutoChamberResponse>('POST', `/analyses/${id}/regions/auto-chamber`, { rim_px }),
  engines: () => request<EngineInfo[]>('GET', '/engines'),
  prepareSpotiflow: (model_name: string) => request<Job>('POST', '/engines/spotiflow/prepare', { model_name }),
  engineRuns: (id: string) => request<EngineRunInfo[]>('GET', `/analyses/${id}/engine-runs`),
  rawDetections: (id: string, q: { engine: EngineName; page: number; valid?: boolean; limit?: number }) =>
    request<RawDetection[]>('GET', `/analyses/${id}/raw-detections?engine=${q.engine}&page=${q.page}${q.valid != null ? `&valid=${q.valid}` : ''}&limit=${q.limit ?? RAW_LIMIT}`),
  consensusSummary: (id: string) => request<ConsensusSummaryResponse>('GET', `/analyses/${id}/consensus/summary`),
  qcSample: (id: string, n: number, seed: number) =>
    request<QcSampleResponse>('POST', `/analyses/${id}/consensus/qc-sample`, { n, seed }),
  listCsv: () => request<CsvImport[]>('GET', '/csv-imports'),
  uploadCsv: (file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return request<CsvImport>('POST', '/csv-imports', undefined, fd);
  },
  setCsvMapping: (id: string, color_mapping: string | null) =>
    request<CsvImport>('PUT', `/csv-imports/${id}`, { color_mapping }),

  // ---- experiments (docs/app/EXPERIMENTS_API.md)
  listExperiments: () => request<ExperimentSummary[]>('GET', '/experiments'),
  createExperiment: (body: { name: string; notes?: string; default_detection?: Partial<DetectionRequest>; default_grid?: ExperimentGrid | null }) =>
    request<ExperimentView>('POST', '/experiments', body),
  getExperiment: (eid: string) => request<ExperimentView>('GET', `/experiments/${eid}`),
  /** only the keys present change; default_grid: null clears it, default_detection: null resets it */
  updateExperiment: (
    eid: string,
    body: Partial<{ name: string; notes: string; default_detection: Partial<DetectionRequest> | null; default_grid: ExperimentGrid | null }>,
  ) => request<ExperimentView>('PUT', `/experiments/${eid}`, body),
  deleteExperiment: (eid: string, deleteSlides = false) =>
    request<ExperimentDeleteResult>('DELETE', `/experiments/${eid}${deleteSlides ? '?delete_slides=true' : ''}`),
  importToExperiment: (eid: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f, f.name));
    return request<ExperimentImportResult>('POST', `/experiments/${eid}/import`, undefined, fd);
  },
  attachSlides: (eid: string, analysis_ids: string[]) => request<ExperimentView>('POST', `/experiments/${eid}/slides`, { analysis_ids }),
  detachSlide: (eid: string, aid: string) => request<ExperimentView>('DELETE', `/experiments/${eid}/slides/${aid}`),
  orderSlides: (eid: string, analysis_ids: string[]) => request<ExperimentView>('PUT', `/experiments/${eid}/order`, { analysis_ids }),
  runExperiment: (eid: string, body: ExperimentRunRequest = {}) => request<Job>('POST', `/experiments/${eid}/run`, body),
  experimentRun: (eid: string) => request<ExperimentRun>('GET', `/experiments/${eid}/run`),
};

/** series: each entry a comma-separated class set (red, green, yellow); omitted = red, green, yellow */
const seriesQs = (series?: string[]) => (series?.length ? '?' + series.map((x) => `series=${encodeURIComponent(x)}`).join('&') : '');
export const experimentCsvUrl = (eid: string, series?: string[]) => `/api/experiments/${eid}/export.csv${seriesQs(series)}`;
export const experimentProvenanceUrl = (eid: string, series?: string[]) => `/api/experiments/${eid}/provenance.json${seriesQs(series)}`;

export interface DisplayParams {
  lo: number | null;
  hi: number | null;
  gamma: number;
}

export function imageUrl(id: string, view: string, d: DisplayParams, extra = ''): string {
  const q = new URLSearchParams();
  if (d.lo != null) q.set('lo', String(d.lo));
  if (d.hi != null) q.set('hi', String(d.hi));
  if (d.gamma !== 1) q.set('gamma', String(d.gamma));
  const qs = q.toString();
  return `/api/analyses/${id}/image/${view}${qs || extra ? '?' : ''}${qs}${qs && extra ? '&' : ''}${extra}`;
}

export const thumbUrl = (id: string, objectId: string, size = 64, ver = '') =>
  `/api/analyses/${id}/thumb/${encodeURIComponent(objectId)}?size=${size}&view=composite${ver ? `&v=${ver}` : ''}`;

export type LabStyle = 't7' | 'counts';
export type LabFormat = 'xlsx' | 'csv';

export const labExportUrl = (id: string, series: string, format: LabFormat) =>
  `/api/analyses/${id}/lab-export?series=${encodeURIComponent(series)}&format=${format}`;

/** Batch lab export. Resolves with the file; a 422 (or other error) rejects with ApiError carrying the server's messages. */
export async function batchLabExport(
  body: { style: LabStyle; series: string | null; format: LabFormat } & ({ analysis_ids: string[] } | { experiment_id: string }),
): Promise<{ blob: Blob; filename: string }> {
  let res: Response;
  try {
    res = await fetch('/api/lab-export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch {
    throw new ApiError(0, { code: 'network', message: 'Cannot reach the backend (network error).' });
  }
  if (!res.ok) throw new ApiError(res.status, parseErrorBody(res.status, await res.text()));
  const cd = res.headers.get('Content-Disposition') ?? '';
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  return { blob: await res.blob(), filename: m ? decodeURIComponent(m[1]) : `lab-export.${body.style === 'counts' || body.format === 'xlsx' ? 'xlsx' : 'zip'}` };
}

export const exportUrl = (id: string, kind: string) => `/api/analyses/${id}/export?kind=${kind}`;
