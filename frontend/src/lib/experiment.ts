import type {
  ExperimentGrid,
  ExperimentRun,
  ExperimentSlide,
  ExperimentSummary,
  GridSpec,
  Job,
  JobStatus,
} from '../api/types';

// ---- job statuses -------------------------------------------------------------------------------------------------
/** Still running: keep polling. Everything else (done, done_with_failures, failed, cancelled) is terminal. */
export const isJobActive = (s: JobStatus | null | undefined) => s === 'queued' || s === 'running';
export const isJobTerminal = (s: JobStatus | null | undefined) => !!s && !isJobActive(s);

export const JOB_STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'queued',
  running: 'running',
  done: 'done',
  done_with_failures: 'done with failures',
  failed: 'failed',
  cancelled: 'cancelled',
};

// ---- run progress -------------------------------------------------------------------------------------------------
export interface RunProgress {
  active: boolean;
  /** slides to run (skipped excluded) */
  total: number;
  done: number;
  failed: number;
  cancelled: number;
  /** pending + queued + running */
  pending: number;
  running: number;
  notRun: number;
  skipped: number;
  /** done + failed + cancelled */
  finished: number;
  /** 0..1 */
  fraction: number;
  label: string;
}

/**
 * Progress of the current / last batch. Uses the per-slide summary of GET /experiments/{eid}/run; without it (older
 * response) falls back to the batch job's plan (`detail.slides`) and its `progress`.
 */
export function runProgress(run: ExperimentRun | null | undefined): RunProgress | null {
  const b = run?.batch;
  if (!b) return null;
  const active = isJobActive(b.status);
  const s = run?.summary;
  let total: number, done: number, failed: number, cancelled: number, pending: number, running: number, notRun: number, skipped: number;
  if (s) {
    ({ total, done, failed, cancelled, running, skipped } = s);
    notRun = s.not_run;
    pending = s.pending + s.queued + s.running;
  } else {
    const plan = Array.isArray((b.detail as { slides?: unknown[] } | null)?.slides) ? ((b.detail as { slides: { skip: string | null }[] }).slides) : [];
    skipped = plan.filter((p) => p.skip).length;
    total = plan.length - skipped;
    const fin = Math.round(b.progress * total);
    done = active ? fin : b.status === 'done' ? total : fin;
    failed = 0;
    cancelled = 0;
    running = active ? 1 : 0;
    pending = active ? total - fin : 0;
    notRun = active ? 0 : total - fin;
  }
  const finished = done + failed + cancelled;
  const fraction = total > 0 ? Math.min(1, finished / total) : b.progress ?? 0;
  const parts = [`${finished} / ${total} finished`, `${done} done`, `${failed} failed`];
  if (pending) parts.push(`${pending} pending`);
  if (notRun) parts.push(`${notRun} not run`);
  if (skipped) parts.push(`${skipped} skipped`);
  return { active, total, done, failed, cancelled, pending, running, notRun, skipped, finished, fraction, label: parts.join(' · ') };
}

/** Status chip of an experiment in the left rail. */
export function experimentChip(e: Pick<ExperimentSummary, 'status' | 'last_batch'>): { label: string; cls: string } {
  const b: Job | null = e.last_batch ?? null;
  if (e.status === 'running' || (b && isJobActive(b.status))) {
    const pct = b ? Math.round((b.progress ?? 0) * 100) : 0;
    return { label: `running ${pct}%`, cls: 'st-detecting' };
  }
  if (b?.status === 'done_with_failures') return { label: 'done with failures', cls: 'warn' };
  if (b?.status === 'done') return { label: 'done', cls: 'ok' };
  if (b?.status === 'failed') return { label: 'run interrupted', cls: 'warn' };
  if (b?.status === 'cancelled') return { label: 'cancelled', cls: '' };
  return { label: 'open', cls: '' };
}

// ---- review -------------------------------------------------------------------------------------------------------
const byOrder = (slides: ExperimentSlide[]) => [...slides].sort((p, q) => p.order_in_experiment - q.order_in_experiment);

/** Not finalized and not every counting region approved (a slide without counting regions counts as unreviewed). */
export const isUnreviewed = (s: ExperimentSlide) => s.status !== 'finalized' && !s.finalized_at && !s.approved;

export const unreviewedSlides = (slides: ExperimentSlide[]) => byOrder(slides).filter(isUnreviewed);

/** The next unreviewed slide in experiment order after `afterId` (wrapping); the first one when afterId is null/unknown. */
export function nextUnreviewed(slides: ExperimentSlide[], afterId: string | null = null): ExperimentSlide | null {
  const s = byOrder(slides);
  const i = afterId ? s.findIndex((x) => x.id === afterId) : -1;
  for (let k = 1; k <= s.length; k++) {
    const c = s[(i + k + s.length) % s.length];
    if (i >= 0 && c.id === afterId) continue;
    if (isUnreviewed(c)) return c;
  }
  return null;
}

/** Finalize experiment: refused while any slide is unreviewed; otherwise the approved, not yet finalized slides. */
export function finalizePlan(slides: ExperimentSlide[]): { blocked: ExperimentSlide[]; toFinalize: ExperimentSlide[] } {
  const blocked = unreviewedSlides(slides);
  return { blocked, toFinalize: blocked.length ? [] : byOrder(slides).filter((s) => s.status !== 'finalized' && s.approved) };
}

// ---- grid for all slides ------------------------------------------------------------------------------------------
/** Spec copied to other slides: the template's effective position becomes the origin (shift 0, not randomised). */
export function copyableGrid(g: GridSpec): GridSpec {
  return { ...g, origin_x: g.origin_x + (g.shift_dx ?? 0), origin_y: g.origin_y + (g.shift_dy ?? 0), seed: null, shift_dx: 0, shift_dy: 0 };
}

/** What `default_grid` stores: the copyable spec plus the template's skipped field numbers. */
export function experimentGrid(g: GridSpec, skipped: number[], templateId: string | null): ExperimentGrid {
  return { ...copyableGrid(g), skipped_fields: [...new Set(skipped)].sort((p, q) => p - q), template_id: templateId };
}

/** The plain GridSpec of a stored default grid (drops the experiment-page extras). */
export function specOf(g: ExperimentGrid): GridSpec {
  const { skipped_fields: _s, template_id: _t, ...spec } = g;
  // default_grid is opaque on the server: fill what an older / hand-written value may lack
  return {
    ...spec,
    order: spec.order ?? 'row-major',
    corner: spec.corner ?? 'tl',
    first_index: spec.first_index ?? 1,
    seed: spec.seed ?? null,
    shift_dx: spec.shift_dx ?? 0,
    shift_dy: spec.shift_dy ?? 0,
  };
}

/** The grid (with its shift) reaches outside [-0.5, W-0.5] x [-0.5, H-0.5] (pixel-centre convention). */
export function gridExceeds(g: GridSpec, w: number, h: number): boolean {
  const ox = g.origin_x + (g.shift_dx ?? 0);
  const oy = g.origin_y + (g.shift_dy ?? 0);
  const e = 1e-6;
  return ox < -0.5 - e || oy < -0.5 - e || ox + g.cols * g.cell_w > w - 0.5 + e || oy + g.rows * g.cell_h > h - 0.5 + e;
}

export interface ApplyTarget {
  id: string;
  name: string;
  /** the slide already has a grid (it is replaced) */
  hasGrid: boolean;
  /** approved counting regions on the slide (approvals of moved fields are lost) */
  approved: number;
  /** image size differs from the template's */
  sizeDiffers: boolean;
  /** the copied grid reaches outside this slide's image */
  exceeds: boolean;
}

export interface ApplyPlan {
  targets: ApplyTarget[];
  /** finalized slides: never changed (reopen them first) */
  finalized: string[];
  /** among all non-finalized slides other than the template */
  withGrid: number;
  withApproved: number;
}

/**
 * Plan "Apply to all slides". mode 'missing': only slides without a grid; 'all': every slide (after a confirmation
 * naming withGrid / withApproved). The template slide itself and finalized slides are never targets.
 */
export function planApplyAll(
  slides: ExperimentSlide[],
  spec: GridSpec,
  templateId: string | null,
  mode: 'missing' | 'all',
): ApplyPlan {
  const tpl = slides.find((s) => s.id === templateId)?.image ?? null;
  const others = byOrder(slides).filter((s) => s.id !== templateId);
  const editable = others.filter((s) => s.status !== 'finalized');
  const targets = editable
    .filter((s) => mode === 'all' || !s.has_grid)
    .map((s) => ({
      id: s.id,
      name: s.name,
      hasGrid: s.has_grid,
      approved: s.n_approved,
      sizeDiffers: !!(tpl && s.image && (tpl.width !== s.image.width || tpl.height !== s.image.height)),
      exceeds: !!(s.image && gridExceeds(spec, s.image.width, s.image.height)),
    }));
  return {
    targets,
    finalized: others.filter((s) => s.status === 'finalized').map((s) => s.name),
    withGrid: editable.filter((s) => s.has_grid).length,
    withApproved: editable.filter((s) => s.has_grid && s.n_approved > 0).length,
  };
}

/** Confirmation text for overwriting existing grids, or null when nothing is overwritten. */
export function overwriteWarning(plan: ApplyPlan): string | null {
  const over = plan.targets.filter((t) => t.hasGrid);
  if (!over.length) return null;
  const appr = over.filter((t) => t.approved > 0).length;
  return (
    `${over.length} slide${over.length === 1 ? ' already has' : 's already have'} a grid` +
    (appr ? ` and ${appr} of them ${appr === 1 ? 'has' : 'have'} approved fields` : '') +
    `; overwrite? The grid fields are replaced` +
    (appr ? '; a field keeps its approval only if it did not move, so approvals may be lost.' : '.')
  );
}

/**
 * PUT /grid-selection calls that make a slide's stored grid skip exactly `skipped` (by field number).
 * `cells` = storedGridCells(analysis); cells in state 'missing' are left alone.
 */
export function fieldToggles(cells: { index: number; state: 'on' | 'off' | 'missing' }[], skipped: number[]): { field: number; counted: boolean }[] {
  const off = new Set(skipped);
  const out: { field: number; counted: boolean }[] = [];
  for (const c of cells) {
    if (c.state === 'missing') continue;
    const want = !off.has(c.index);
    if (want !== (c.state === 'on')) out.push({ field: c.index, counted: want });
  }
  return out;
}

/** Move one slide up (-1) or down (+1); returns the new id order (unchanged at the ends). */
export function moveSlide(ids: string[], id: string, delta: -1 | 1): string[] {
  const i = ids.indexOf(id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= ids.length) return ids;
  const out = [...ids];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

/** Drag-and-drop reorder: put `id` at the position of `targetId`. */
export function dropSlide(ids: string[], id: string, targetId: string): string[] {
  if (id === targetId) return ids;
  const out = ids.filter((x) => x !== id);
  const j = out.indexOf(targetId);
  if (j < 0) return ids;
  const from = ids.indexOf(id), to = ids.indexOf(targetId);
  out.splice(from < to ? j + 1 : j, 0, id);
  return out;
}
