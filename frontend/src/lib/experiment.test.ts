import { describe, expect, it } from 'vitest';
import type { Counts, ExperimentRun, ExperimentSlide, GridSpec, Job } from '../api/types';
import {
  copyableGrid,
  dropSlide,
  experimentChip,
  experimentGrid,
  fieldToggles,
  finalizePlan,
  gridExceeds,
  isJobActive,
  isJobTerminal,
  moveSlide,
  nextUnreviewed,
  overwriteWarning,
  planApplyAll,
  runProgress,
  specOf,
} from './experiment';

const counts: Counts = { scope: 'slide', red_only: 0, green_only: 0, yellow: 0, uncertain: 0, red_positive: 0, green_positive: 0, total_objects_exclusive: 0, reporting_mode: 'exclusive', is_final: false };
const slide = (id: string, order: number, p: Partial<ExperimentSlide> = {}): ExperimentSlide => ({
  id, name: `s${order}`, order_in_experiment: order, status: 'in_review', export_status: 'draft', updated_at: '', finalized_at: null,
  mapping_confirmed: true, image: { width: 512, height: 384, kind: 'raw_two_channel', dtype: 'uint16', n_channels: 2 },
  has_grid: false, n_counting_regions: 1, n_fields: 0, n_approved: 0, approved: false, review_queue: 0,
  detection_ran_at: null, engine: null, counts, last_job: null, ...p,
});
const job = (status: Job['status'], p: Partial<Job> = {}): Job => ({ id: 'b', analysis_id: 'experiment:e', status, progress: 0, message: '', error: null, started_at: null, finished_at: null, kind: 'batch', ...p });
const spec: GridSpec = { rows: 2, cols: 3, cell_w: 100, cell_h: 100, origin_x: 20, origin_y: 20, order: 'row-major', corner: 'tl', first_index: 1, seed: null, shift_dx: 0, shift_dy: 0 };

describe('job statuses', () => {
  it('done_with_failures is terminal', () => {
    expect(isJobActive('running')).toBe(true);
    expect(isJobActive('queued')).toBe(true);
    for (const s of ['done', 'done_with_failures', 'failed', 'cancelled'] as const) {
      expect(isJobActive(s)).toBe(false);
      expect(isJobTerminal(s)).toBe(true);
    }
    expect(isJobTerminal(undefined)).toBe(false);
  });
});

describe('runProgress', () => {
  it('aggregates the per-slide summary', () => {
    const run: ExperimentRun = {
      batch: job('running', { progress: 0.33 }),
      slides: [],
      summary: { total: 3, done: 1, failed: 1, cancelled: 0, running: 1, queued: 0, pending: 0, not_run: 0, skipped: 1 },
    };
    const p = runProgress(run)!;
    expect(p.active).toBe(true);
    expect(p.finished).toBe(2);
    expect(p.pending).toBe(1);
    expect(p.fraction).toBeCloseTo(2 / 3);
    expect(p.label).toBe('2 / 3 finished · 1 done · 1 failed · 1 pending · 1 skipped');
  });
  it('falls back to the batch plan in detail', () => {
    const b = job('running', { progress: 0.5, detail: { only_missing: false, slides: [1, 2, 3, 4, 5].map((i) => ({ analysis_id: `a${i}`, name: `s${i}`, order: i, skip: i === 5 ? 'finalized' : null, override: false })) } });
    const p = runProgress({ batch: b, slides: [], summary: null })!;
    expect(p.total).toBe(4);
    expect(p.skipped).toBe(1);
    expect(p.pending).toBe(2);
  });
  it('null without a batch', () => {
    expect(runProgress({ batch: null, slides: [], summary: null })).toBeNull();
    expect(runProgress(null)).toBeNull();
  });
});

describe('experimentChip', () => {
  it('shows running with progress, then the terminal state', () => {
    expect(experimentChip({ status: 'running', last_batch: job('running', { progress: 0.5 }) }).label).toBe('running 50%');
    expect(experimentChip({ status: 'open', last_batch: job('done_with_failures') }).label).toBe('done with failures');
    expect(experimentChip({ status: 'open', last_batch: job('done') }).label).toBe('done');
    expect(experimentChip({ status: 'open', last_batch: null }).label).toBe('open');
  });
});

describe('next unreviewed / finalize', () => {
  const slides = [
    slide('c', 3),
    slide('a', 1, { approved: true, n_approved: 1 }),
    slide('b', 2),
    slide('d', 4, { status: 'finalized', finalized_at: 'x' }),
  ];
  it('walks the experiment in order, wrapping, skipping approved and finalized slides', () => {
    expect(nextUnreviewed(slides)?.id).toBe('b');
    expect(nextUnreviewed(slides, 'b')?.id).toBe('c');
    expect(nextUnreviewed(slides, 'c')?.id).toBe('b');
    expect(nextUnreviewed(slides, 'a')?.id).toBe('b');
    expect(nextUnreviewed([slide('x', 1, { approved: true })])).toBeNull();
    // the only unreviewed slide is the open one: nothing else to go to
    expect(nextUnreviewed([slide('x', 1)], 'x')).toBeNull();
  });
  it('refuses to finalize while slides are unreviewed, else finalizes the approved ones', () => {
    expect(finalizePlan(slides).blocked.map((s) => s.id)).toEqual(['b', 'c']);
    expect(finalizePlan(slides).toFinalize).toEqual([]);
    const ok = finalizePlan([slides[1], slides[3]]);
    expect(ok.blocked).toEqual([]);
    expect(ok.toFinalize.map((s) => s.id)).toEqual(['a']);
  });
});

describe('grid for all slides', () => {
  it('copies the effective position, not the randomisation', () => {
    const g = copyableGrid({ ...spec, seed: 7, shift_dx: 5, shift_dy: -3 });
    expect([g.origin_x, g.origin_y, g.shift_dx, g.shift_dy, g.seed]).toEqual([25, 17, 0, 0, null]);
    const d = experimentGrid({ ...spec, seed: 7, shift_dx: 5 }, [4, 2, 2], 'tpl');
    expect(d.skipped_fields).toEqual([2, 4]);
    expect(d.template_id).toBe('tpl');
    expect(specOf(d)).toEqual({ ...spec, origin_x: 25 });
  });
  it('flags a grid outside the image', () => {
    expect(gridExceeds(spec, 512, 384)).toBe(false);
    expect(gridExceeds(spec, 300, 384)).toBe(true);
    expect(gridExceeds({ ...spec, origin_x: -1 }, 512, 384)).toBe(true);
  });
  it('plans slides without a grid or all slides, never the template or finalized ones', () => {
    const small = { width: 200, height: 200, kind: 'raw_two_channel' as const, dtype: 'uint16', n_channels: 2 };
    const slides = [
      slide('t', 1, { has_grid: true }),
      slide('g', 2, { has_grid: true, n_approved: 2 }),
      slide('n', 3),
      slide('s', 4, { image: small }),
      slide('f', 5, { status: 'finalized', has_grid: true }),
    ];
    const miss = planApplyAll(slides, spec, 't', 'missing');
    expect(miss.targets.map((t) => t.id)).toEqual(['n', 's']);
    expect(miss.targets[1]).toMatchObject({ sizeDiffers: true, exceeds: true });
    expect(miss.finalized).toEqual(['s5']);
    expect(overwriteWarning(miss)).toBeNull();
    const all = planApplyAll(slides, spec, 't', 'all');
    expect(all.targets.map((t) => t.id)).toEqual(['g', 'n', 's']);
    expect([all.withGrid, all.withApproved]).toEqual([1, 1]);
    expect(overwriteWarning(all)).toMatch(/^1 slide already has a grid and 1 of them has approved fields; overwrite\?.*approvals may be lost/);
  });
  it('toggles exactly the fields whose skip state differs', () => {
    const cells = [
      { index: 1, state: 'on' as const },
      { index: 2, state: 'on' as const },
      { index: 3, state: 'off' as const },
      { index: 4, state: 'missing' as const },
    ];
    expect(fieldToggles(cells, [2, 4])).toEqual([
      { field: 2, counted: false },
      { field: 3, counted: true },
    ]);
    expect(fieldToggles(cells, [3])).toEqual([]);
  });
});

describe('reorder', () => {
  it('moves up / down and drops onto a row', () => {
    expect(moveSlide(['a', 'b', 'c'], 'b', -1)).toEqual(['b', 'a', 'c']);
    expect(moveSlide(['a', 'b', 'c'], 'a', -1)).toEqual(['a', 'b', 'c']);
    expect(dropSlide(['a', 'b', 'c', 'd'], 'a', 'c')).toEqual(['b', 'c', 'a', 'd']);
    expect(dropSlide(['a', 'b', 'c', 'd'], 'd', 'b')).toEqual(['a', 'd', 'b', 'c']);
  });
});
