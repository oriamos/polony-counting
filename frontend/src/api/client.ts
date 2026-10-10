import type {
  ActionType,
  Analysis,
  AnalysisSummary,
  ApiErrorBody,
  CountsResponse,
  CsvImport,
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
  listCsv: () => request<CsvImport[]>('GET', '/csv-imports'),
  uploadCsv: (file: File) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return request<CsvImport>('POST', '/csv-imports', undefined, fd);
  },
  setCsvMapping: (id: string, color_mapping: string | null) =>
    request<CsvImport>('PUT', `/csv-imports/${id}`, { color_mapping }),
};

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
export async function batchLabExport(body: {
  analysis_ids: string[];
  style: LabStyle;
  series: string | null;
  format: LabFormat;
}): Promise<{ blob: Blob; filename: string }> {
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
