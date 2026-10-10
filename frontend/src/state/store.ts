import { create } from 'zustand';
import { api, ApiError, RAW_LIMIT, type DisplayParams } from '../api/client';
import type { EngineFilter } from '../lib/consensus';
import type {
  ActionType,
  Analysis,
  AnalysisSummary,
  CountsResponse,
  CsvImport,
  DetectedObject,
  DetectionRequest,
  ConsensusSummaryResponse,
  EngineInfo,
  EngineName,
  EngineRunInfo,
  RawDetection,
  ImageView,
  Job,
  ObjectClass,
  Region,
  ReferenceInfo,
  AutoChamberResponse,
  ReviewQueueItem,
} from '../api/types';
import type { PreviewCell } from '../lib/reference';
import { centerOn, fitBox, fitView, type View } from '../lib/geometry';

export type Tool =
  | 'select'
  | 'add'
  | 'split'
  | 'draw-count-rect'
  | 'draw-count-poly'
  | 'draw-excl-rect'
  | 'draw-excl-poly'
  | 'preview-window';

export interface Toast {
  id: number;
  kind: 'error' | 'info' | 'success';
  text: string;
}

export interface EngineLayers {
  /** consensus styling of candidates by channel-evidence status */
  consensus: boolean;
  rawBigfish: boolean;
  rawSpotiflow: boolean;
  /** raw detections outside the valid counting area / excluded regions */
  rawExcluded: boolean;
  /** objects the user rejected (deleted) */
  rejected: boolean;
}
export interface RawSlot {
  items: RawDetection[];
  /** true when the server returned the cap, i.e. more may exist */
  capped: boolean;
}
export const rawKey = (engine: EngineName, page: number, excluded: boolean) => `${engine}|${page}|${excluded ? 'all' : 'valid'}`;

export type RightTab = 'review' | 'regions' | 'detect' | 'totals' | 'finish';

const NO_DISPLAY: DisplayParams = { lo: null, hi: null, gamma: 1 };
let toastSeq = 1;

export interface AppState {
  health: { ok: boolean; version: string; bigfish_version: string } | null;
  healthError: string | null;
  analyses: AnalysisSummary[];
  listLoading: boolean;
  listError: string | null;
  current: Analysis | null;
  loadingCurrent: boolean;
  loadError: string | null;
  counts: CountsResponse | null;
  queue: ReviewQueueItem[];
  busy: number;

  selection: string[];
  selectedRegionId: string | null;
  tool: Tool;
  addClass: ObjectClass;
  splitPts: [number, number][];
  hideOverlays: boolean;
  classVisible: Record<ObjectClass, boolean>;
  showLabels: 'auto' | 'on' | 'off';
  showRegions: boolean;
  showSuggestions: boolean;
  showReviewedStyle: boolean;

  imageView: ImageView;
  display: Record<ImageView, DisplayParams>;
  autoRange: { lo: number; hi: number } | null;
  view: View;
  viewport: { w: number; h: number };
  fitted: boolean;

  preview: { objects: DetectedObject[]; notes: string[]; window: [number, number, number, number] } | null;
  previewBusy: boolean;
  detReq: DetectionRequest | null;
  job: Job | null;
  engines: EngineInfo[] | null;
  enginesError: string | null;
  prepareJob: Job | null;
  engineRuns: EngineRunInfo[];
  consensusSummary: ConsensusSummaryResponse | null;
  layers: EngineLayers;
  rawCache: Record<string, RawSlot>;
  rawError: string | null;
  reviewFilter: EngineFilter;
  regionErrors: Record<string, string>;
  csvImports: CsvImport[];
  /** grid-field generator preview polygons (not yet applied) */
  gridPreview: { index: number; polygon: [number, number][]; flag?: boolean }[] | null;

  /** researcher reference marks (null until loaded / not available) */
  reference: ReferenceInfo | null;
  showReference: boolean;
  /** reference plane to display/compare (null = all planes) */
  refPlane: number | null;
  /** live preview of the shifted reference grid for the import dialog (null = hidden) */
  refGridPreview: PreviewCell[] | null;
  /** auto chamber region preview (not yet applied) */
  chamberPreview: AutoChamberResponse | null;

  rightTab: RightTab;
  mappingOpen: boolean;
  helpOpen: boolean;
  theme: 'auto' | 'light' | 'dark';
  toasts: Toast[];

  // actions
  toast: (kind: Toast['kind'], text: string) => void;
  dismissToast: (id: number) => void;
  refreshHealth: () => Promise<void>;
  refreshList: () => Promise<void>;
  openAnalysis: (id: string | null) => Promise<void>;
  setAnalysis: (a: Analysis) => void;
  refreshDerived: () => Promise<void>;
  refreshQueue: () => Promise<void>;
  act: (type: ActionType, payload: Record<string, unknown>) => Promise<Analysis | null>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  saveRegions: (regions: Region[]) => Promise<void>;
  setSelection: (ids: string[]) => void;
  toggleSelect: (id: string) => void;
  setTool: (t: Tool) => void;
  setImageView: (v: ImageView) => void;
  setDisplay: (patch: Partial<DisplayParams>) => void;
  resetDisplay: () => void;
  setView: (v: View) => void;
  setViewport: (w: number, h: number) => void;
  fit: () => void;
  zoomToPoint: (x: number, y: number, scale?: number) => void;
  zoomToBox: (x0: number, y0: number, x1: number, y1: number) => void;
  runPreview: (win: [number, number, number, number]) => Promise<void>;
  clearPreview: () => void;
  setDetReq: (r: DetectionRequest) => void;
  runDetection: () => Promise<void>;
  cancelJob: () => Promise<void>;
  gotoQueue: (delta: 1 | -1) => void;
  confirmSelected: () => Promise<void>;
  patch: (p: Partial<AppState>) => void;
  deleteAnalysis: (id: string) => Promise<void>;
  reopen: () => Promise<void>;
  finalize: () => Promise<void>;
  refreshCsv: () => Promise<void>;
  loadReference: (id: string) => Promise<void>;
  loadEngines: () => Promise<void>;
  prepareModel: (name: string) => Promise<void>;
  setLayer: (p: Partial<EngineLayers>) => void;
  loadRaw: () => Promise<void>;
  qcSample: (n: number, seed: number) => Promise<void>;
}

const errText = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));

let chain: Promise<unknown> = Promise.resolve();
/** Serialise mutating calls so rapid edits cannot interleave. */
function serial<T>(fn: () => Promise<T>): Promise<T> {
  const p = chain.then(fn, fn);
  chain = p.catch(() => undefined);
  return p;
}

export const useApp = create<AppState>((set, get) => ({
  health: null,
  healthError: null,
  analyses: [],
  listLoading: false,
  listError: null,
  current: null,
  loadingCurrent: false,
  loadError: null,
  counts: null,
  queue: [],
  busy: 0,

  selection: [],
  selectedRegionId: null,
  tool: 'select',
  addClass: 'red',
  splitPts: [],
  hideOverlays: false,
  classVisible: { red: true, green: true, yellow: true, uncertain: true },
  showLabels: 'auto',
  showRegions: true,
  showSuggestions: false,
  showReviewedStyle: true,

  imageView: 'composite',
  display: { red: NO_DISPLAY, green: NO_DISPLAY, composite: NO_DISPLAY, processed_red: NO_DISPLAY, processed_green: NO_DISPLAY },
  autoRange: null,
  view: { scale: 1, tx: 0, ty: 0 },
  viewport: { w: 800, h: 600 },
  fitted: false,

  preview: null,
  previewBusy: false,
  detReq: null,
  job: null,
  engines: null,
  enginesError: null,
  prepareJob: null,
  engineRuns: [],
  consensusSummary: null,
  layers: { consensus: true, rawBigfish: false, rawSpotiflow: false, rawExcluded: false, rejected: false },
  rawCache: {},
  rawError: null,
  reviewFilter: 'all',
  regionErrors: {},
  csvImports: [],
  gridPreview: null,

  reference: null,
  showReference: false,
  refPlane: null,
  refGridPreview: null,
  chamberPreview: null,

  rightTab: 'review',
  mappingOpen: false,
  helpOpen: false,
  theme: (() => {
    try {
      const t = localStorage.getItem('polony.theme');
      return t === 'light' || t === 'dark' ? t : 'auto';
    } catch {
      return 'auto';
    }
  })(),
  toasts: [],

  patch: (p) => set(p),

  toast: (kind, text) => {
    const id = toastSeq++;
    set((s) => ({ toasts: [...s.toasts, { id, kind, text }].slice(-5) }));
    setTimeout(() => get().dismissToast(id), kind === 'error' ? 9000 : 4000);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  refreshHealth: async () => {
    try {
      set({ health: await api.health(), healthError: null });
    } catch (e) {
      set({ health: null, healthError: errText(e) });
    }
  },

  refreshList: async () => {
    set({ listLoading: true });
    try {
      const analyses = await api.listAnalyses();
      set({ analyses, listError: null, listLoading: false });
    } catch (e) {
      set({ listError: errText(e), listLoading: false });
    }
  },

  loadReference: async (id) => {
    try {
      const r = await api.reference(id);
      if (get().current?.id === id) set({ reference: r });
    } catch {
      if (get().current?.id === id) set({ reference: null }); // optional feature: older backends have no endpoint
    }
  },

  refreshCsv: async () => {
    try {
      set({ csvImports: await api.listCsv() });
    } catch {
      /* optional panel; leave as is */
    }
  },

  openAnalysis: async (id) => {
    if (!id) {
      set({ current: null, counts: null, queue: [], selection: [], preview: null, job: null, loadError: null });
      return;
    }
    set({
      loadingCurrent: true,
      loadError: null,
      selection: [],
      selectedRegionId: null,
      preview: null,
      job: null,
      counts: null,
      queue: [],
      fitted: false,
      splitPts: [],
      tool: 'select',
      regionErrors: {},
      gridPreview: null,
      reference: null,
      showReference: false,
      refPlane: null,
      refGridPreview: null,
      chamberPreview: null,
      engineRuns: [],
      consensusSummary: null,
      rawCache: {},
      rawError: null,
      reviewFilter: 'all',
    });
    try {
      const a = await api.getAnalysis(id);
      set({ current: null });
      set({ detReq: a.detection_request, loadingCurrent: false, mappingOpen: a.status === 'needs_channel_mapping' });
      get().setAnalysis(a);
    } catch (e) {
      set({ loadingCurrent: false, loadError: errText(e), current: null });
    }
  },

  setAnalysis: (a) => {
    const s = get();
    const alive = new Set(a.objects.filter((o) => !o.deleted).map((o) => o.id));
    const first = !s.current || s.current.id !== a.id;
    set({
      current: a,
      selection: s.selection.filter((id) => alive.has(id)),
      selectedRegionId: s.selectedRegionId && a.regions.some((r) => r.id === s.selectedRegionId) ? s.selectedRegionId : null,
      detReq: first || !s.detReq ? a.detection_request : s.detReq,
      analyses: s.analyses.map((x) => (x.id === a.id ? { ...x, name: a.name, status: a.status, updated_at: a.updated_at, finalized_at: a.finalized_at } : x)),
    });
    if (first) void get().loadReference(a.id);
    if (first && a.image) {
      const { w, h } = get().viewport;
      set({ view: fitView(a.image.width, a.image.height, w, h), fitted: true });
    }
    void get().refreshDerived();
  },

  refreshDerived: async () => {
    const a = get().current;
    if (!a) return;
    const id = a.id;
    try {
      const [counts, queue] = await Promise.all([api.counts(id), api.reviewQueue(id)]);
      if (get().current?.id === id) set({ counts, queue });
    } catch {
      /* counts/queue are derived; failure shows as empty state */
    }
    // two-engine data is optional: an older backend simply has none of it
    try {
      const [runs, summary] = await Promise.all([api.engineRuns(id), api.consensusSummary(id)]);
      if (get().current?.id !== id) return;
      const sig = (r: EngineRunInfo[]) => r.map((x) => x.run_id).join(',');
      const changed = sig(runs) !== sig(get().engineRuns);
      set({ engineRuns: runs, consensusSummary: summary, ...(changed ? { rawCache: {} } : {}) });
      if (changed) void get().loadRaw();
    } catch {
      if (get().current?.id === id) set({ engineRuns: [], consensusSummary: null });
    }
  },

  refreshQueue: async () => {
    const a = get().current;
    if (!a) return;
    try {
      const queue = await api.reviewQueue(a.id);
      if (get().current?.id === a.id) set({ queue });
    } catch {
      /* ignore */
    }
  },

  act: (type, payload) =>
    serial(async () => {
      const a = get().current;
      if (!a) return null;
      set((s) => ({ busy: s.busy + 1 }));
      try {
        const next = await api.action(a.id, type, payload);
        set({ regionErrors: {} });
        get().setAnalysis(next);
        return next;
      } catch (e) {
        const msg = errText(e);
        const rid = typeof payload.region_id === 'string' ? payload.region_id : null;
        if (rid) set((s) => ({ regionErrors: { ...s.regionErrors, [rid]: msg } }));
        get().toast('error', `${type.replace(/_/g, ' ')}: ${msg}`);
        return null;
      } finally {
        set((s) => ({ busy: s.busy - 1 }));
      }
    }),

  undo: () =>
    serial(async () => {
      const a = get().current;
      if (!a || a.undo_depth <= 0) return;
      try {
        get().setAnalysis(await api.undo(a.id));
      } catch (e) {
        get().toast('error', `undo: ${errText(e)}`);
      }
    }),
  redo: () =>
    serial(async () => {
      const a = get().current;
      if (!a || a.redo_depth <= 0) return;
      try {
        get().setAnalysis(await api.redo(a.id));
      } catch (e) {
        get().toast('error', `redo: ${errText(e)}`);
      }
    }),

  saveRegions: (regions) =>
    serial(async () => {
      const a = get().current;
      if (!a) return;
      set((s) => ({ busy: s.busy + 1 }));
      try {
        get().setAnalysis(await api.putRegions(a.id, regions));
      } catch (e) {
        get().toast('error', `regions: ${errText(e)}`);
        void api.getAnalysis(a.id).then((x) => get().setAnalysis(x)).catch(() => undefined);
      } finally {
        set((s) => ({ busy: s.busy - 1 }));
      }
    }),

  setSelection: (ids) => set({ selection: ids }),
  toggleSelect: (id) =>
    set((s) => ({ selection: s.selection.includes(id) ? s.selection.filter((x) => x !== id) : [...s.selection, id] })),
  setTool: (t) => set({ tool: t, splitPts: [] }),

  setImageView: (v) => set({ imageView: v, autoRange: null }),
  setDisplay: (patch) => set((s) => ({ display: { ...s.display, [s.imageView]: { ...s.display[s.imageView], ...patch } } })),
  resetDisplay: () => set((s) => ({ display: { ...s.display, [s.imageView]: NO_DISPLAY } })),

  setView: (view) => set({ view }),
  setViewport: (w, h) => {
    const s = get();
    set({ viewport: { w, h } });
    if (s.current?.image && !s.fitted) set({ view: fitView(s.current.image.width, s.current.image.height, w, h), fitted: true });
  },
  fit: () => {
    const { current, viewport } = get();
    if (current?.image) set({ view: fitView(current.image.width, current.image.height, viewport.w, viewport.h) });
  },
  zoomToPoint: (x, y, scale = 8) => {
    const { viewport, view } = get();
    set({ view: centerOn(x, y, Math.max(scale, view.scale), viewport.w, viewport.h) });
  },
  zoomToBox: (x0, y0, x1, y1) => {
    const { viewport } = get();
    set({ view: fitBox(x0, y0, x1, y1, viewport.w, viewport.h) });
  },

  runPreview: async (win) => {
    const { current, detReq } = get();
    if (!current || !detReq) return;
    set({ previewBusy: true });
    try {
      const r = await api.preview(current.id, { ...detReq, preview_window: win });
      set({ preview: { objects: r.objects, notes: r.notes, window: win }, previewBusy: false });
      get().toast('info', `Preview: ${r.objects.length} candidate objects (temporary, not saved)`);
    } catch (e) {
      set({ previewBusy: false });
      get().toast('error', `preview: ${errText(e)}`);
    }
  },
  clearPreview: () => set({ preview: null }),
  setDetReq: (detReq) => set({ detReq }),

  runDetection: async () => {
    const { current, detReq } = get();
    if (!current || !detReq) return;
    try {
      let job = await api.detect(current.id, { ...detReq, preview_window: null });
      set({ job, preview: null });
      const id = current.id;
      while (job.status === 'queued' || job.status === 'running') {
        await new Promise((r) => setTimeout(r, 600));
        if (get().current?.id !== id) return;
        job = await api.job(job.id);
        set({ job });
      }
      let a = await api.getAnalysis(id);
      for (let i = 0; i < 8 && a.status === 'detecting'; i++) {
        // the server may flip the analysis status a moment after the job ends
        await new Promise((r) => setTimeout(r, 400));
        a = await api.getAnalysis(id);
      }
      if (get().current?.id === id) get().setAnalysis(a);
      if (job.status === 'done') get().toast('success', 'Detection finished');
      else if (job.status === 'failed') get().toast('error', `Detection failed: ${job.error ?? 'unknown error'}`);
    } catch (e) {
      set({ job: { id: '', analysis_id: current.id, status: 'failed', progress: 0, message: '', error: errText(e), started_at: null, finished_at: null } });
      get().toast('error', `detect: ${errText(e)}`);
    }
  },
  cancelJob: async () => {
    const j = get().job;
    if (!j || !j.id) return;
    try {
      await api.cancelJob(j.id);
    } catch (e) {
      get().toast('error', `cancel: ${errText(e)}`);
    }
  },

  loadEngines: async () => {
    try {
      set({ engines: await api.engines(), enginesError: null });
    } catch (e) {
      set({ engines: null, enginesError: errText(e) });
    }
  },

  prepareModel: async (name) => {
    try {
      let job = await api.prepareSpotiflow(name);
      set({ prepareJob: job });
      while (job.status === 'queued' || job.status === 'running') {
        await new Promise((r) => setTimeout(r, 500));
        job = await api.job(job.id);
        set({ prepareJob: job });
      }
      await get().loadEngines();
      if (job.status === 'done') get().toast('success', `Spotiflow model "${name}" is ready`);
      else get().toast('error', `Model preparation ${job.status}: ${job.error ?? ''}`);
    } catch (e) {
      set({ prepareJob: { id: '', analysis_id: '', status: 'failed', progress: 0, message: '', error: errText(e), started_at: null, finished_at: null } });
    }
  },

  setLayer: (p) => {
    set((s) => ({ layers: { ...s.layers, ...p } }));
    void get().loadRaw();
  },

  /** Lazily fetch raw detections, only for toggled engines, one capped request per engine and page. */
  loadRaw: async () => {
    const { current: a, layers, rawCache } = get();
    if (!a?.image) return;
    const pages = [...new Set(a.image.channels.filter((c) => c.role !== 'ignore').map((c) => c.page))];
    const want: EngineName[] = [];
    if (layers.rawBigfish) want.push('bigfish');
    if (layers.rawSpotiflow) want.push('spotiflow');
    const id = a.id;
    for (const eng of want)
      for (const pg of pages) {
        const k = rawKey(eng, pg, layers.rawExcluded);
        if (get().rawCache[k] || rawCache[k]) continue;
        try {
          const items = await api.rawDetections(id, { engine: eng, page: pg, valid: layers.rawExcluded ? undefined : true, limit: RAW_LIMIT });
          if (get().current?.id !== id) return;
          set((s) => ({ rawCache: { ...s.rawCache, [k]: { items, capped: items.length >= RAW_LIMIT } }, rawError: null }));
        } catch (e) {
          set({ rawError: errText(e) });
        }
      }
  },

  qcSample: async (n, seed) => {
    const a = get().current;
    if (!a) return;
    try {
      const r = await api.qcSample(a.id, n, seed);
      get().toast('info', `${r.marked} of ${r.eligible} unreviewed agreement candidate${r.eligible === 1 ? '' : 's'} marked for QC review`);
      await get().refreshDerived();
    } catch (e) {
      get().toast('error', `QC sample: ${errText(e)}`);
    }
  },

  gotoQueue: (delta) => {
    const { queue, selection } = get();
    if (!queue.length) {
      get().toast('info', 'Review queue is empty');
      return;
    }
    const cur = selection.length === 1 ? queue.findIndex((q) => q.object_id === selection[0]) : -1;
    const idx = cur < 0 ? (delta > 0 ? 0 : queue.length - 1) : (cur + delta + queue.length) % queue.length;
    const q = queue[idx];
    set({ selection: [q.object_id], tool: 'select' });
    get().zoomToPoint(q.x, q.y, 10);
  },

  confirmSelected: async () => {
    const { selection, queue } = get();
    if (!selection.length) return;
    const pos = selection.length === 1 ? queue.findIndex((q) => q.object_id === selection[0]) : -1;
    const r = await get().act('confirm', { object_ids: selection });
    if (!r) return;
    await get().refreshQueue();
    const q = get().queue;
    if (pos >= 0 && q.length) {
      const nxt = q[Math.min(pos, q.length - 1)];
      set({ selection: [nxt.object_id] });
      get().zoomToPoint(nxt.x, nxt.y, 10);
    }
  },

  deleteAnalysis: async (id) => {
    try {
      await api.deleteAnalysis(id);
      if (get().current?.id === id) await get().openAnalysis(null);
      await get().refreshList();
    } catch (e) {
      get().toast('error', `delete: ${errText(e)}`);
    }
  },

  finalize: () =>
    serial(async () => {
      const a = get().current;
      if (!a) return;
      try {
        get().setAnalysis(await api.finalize(a.id));
        get().toast('success', 'Analysis finalized');
      } catch (e) {
        get().toast('error', `finalize refused: ${errText(e)}`);
      }
    }),

  reopen: () =>
    serial(async () => {
      const a = get().current;
      if (!a) return;
      try {
        get().setAnalysis(await api.reopen(a.id));
        get().toast('info', 'Analysis reopened for review');
      } catch (e) {
        if (e instanceof ApiError && (e.status === 404 || e.status === 405)) {
          // Contract has no /reopen endpoint: fall back to un-approving every region.
          try {
            let cur = a;
            for (const r of a.regions.filter((r) => r.approved)) {
              cur = await api.action(a.id, 'unapprove_region', { region_id: r.id });
            }
            get().setAnalysis(cur);
            get().toast('info', 'Reopened by un-approving regions');
          } catch (e2) {
            get().toast('error', `reopen: ${errText(e2)}`);
          }
        } else get().toast('error', `reopen: ${errText(e)}`);
      }
    }),
}));

export const liveObjects = (a: Analysis | null) => (a ? a.objects.filter((o) => !o.deleted) : []);
