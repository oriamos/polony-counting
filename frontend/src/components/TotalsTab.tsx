import type { Counts, ReportingMode } from '../api/types';
import { api } from '../api/client';
import { useApp } from '../state/store';
import { CompareSection } from './ReferencePanel';
import { GlyphIcon } from './ViewerToolbar';

export function DraftBadge({ final }: { final: boolean }) {
  return (
    <span className={`badge-status ${final ? 'final' : 'draft'}`} data-testid="draft-badge">
      {final ? 'FINAL' : 'DRAFT'}
    </span>
  );
}

export const MODE_TEXT: Record<ReportingMode, { name: string; blurb: string }> = {
  exclusive: {
    name: 'Exclusive',
    blurb: 'Each object counted once: red-only, green-only or yellow. Columns add up to the object total.',
  },
  channel_inclusive: {
    name: 'Channel-inclusive',
    blurb: 'Red-positive and green-positive both include yellow objects; yellow is the overlap. Do not add these columns together.',
  },
};

export function TotalsTab() {
  const a = useApp((s) => s.current)!;
  const counts = useApp((s) => s.counts);
  const setAnalysis = useApp((s) => s.setAnalysis);
  const toast = useApp((s) => s.toast);
  const patch = useApp((s) => s.patch);
  const mode = a.reporting_mode;
  const final = a.status === 'finalized';
  const nUnresolved = counts ? counts.slide.uncertain : a.objects.filter((o) => !o.deleted && o.cls === 'uncertain').length;

  const setMode = async (m: ReportingMode) => {
    if (m === mode) return;
    try {
      setAnalysis(await api.putSettings(a.id, { reporting_mode: m }));
    } catch (e) {
      toast('error', `reporting mode: ${(e as Error).message}`);
    }
  };

  return (
    <div className="totals" data-testid="totals-tab">
      <div className="row between">
        <h3>Totals</h3>
        <DraftBadge final={final} />
      </div>
      <div className="seg full" role="group" aria-label="Reporting mode">
        {(Object.keys(MODE_TEXT) as ReportingMode[]).map((m) => (
          <button key={m} className={mode === m ? 'on' : ''} onClick={() => void setMode(m)} disabled={final} data-testid={`mode-${m}`}>
            {MODE_TEXT[m].name}
          </button>
        ))}
      </div>
      <div className="muted small" data-testid="mode-blurb">
        <strong>{MODE_TEXT[mode].name}:</strong> {MODE_TEXT[mode].blurb} Uncertain objects are never included in colour totals.
      </div>
      {!counts && <div className="empty"><span className="spinner" /> Loading totals…</div>}
      {counts && (
        <>
          <CountsTable title="Slide" rows={[{ label: 'All counting regions', c: counts.slide }]} mode={mode} />
          {counts.regions.length > 0 ? (
            <CountsTable
              title="Per region"
              rows={counts.regions.map((c) => ({ label: a.regions.find((r) => r.id === c.scope)?.name ?? c.scope, c }))}
              mode={mode}
            />
          ) : (
            <div className="empty small">No per-region counts yet (draw counting regions).</div>
          )}
          <ConvenienceCounts counts={counts} regions={a.regions} />
        </>
      )}
      <CompareSection />
      {nUnresolved > 0 && !final && (
        <div className="notice warn small" role="status" data-testid="uncertain-block">
          <strong>{nUnresolved} uncertain object{nUnresolved > 1 ? 's' : ''} unresolved.</strong> They are not included in any colour total, and while any remain the affected regions cannot be
          approved (nor the analysis finalized). Resolve each one (reclassify, confirm or delete) in the Review tab.{' '}
          <button className="btn sm" onClick={() => patch({ rightTab: 'review' })}>
            Go to Review
          </button>
        </div>
      )}
      {final ? (
        <div className="notice ok small">These are final, approved counts.</div>
      ) : (
        <div className="notice warn small">DRAFT: counts change while you edit and are not final until every counting region is approved and the analysis is finalized.</div>
      )}
      <ObjectEvidenceNote />
    </div>
  );
}

function ObjectEvidenceNote() {
  return <div className="muted small">The app reports measured evidence (SNR, peak, background, association distance, saturation) and review reasons, not confidence percentages.</div>;
}

function CountsTable({ title, rows, mode }: { title: string; rows: { label: string; c: Counts }[]; mode: ReportingMode }) {
  const cols =
    mode === 'exclusive'
      ? [
          { k: 'red_only', label: 'red-only', cls: 'red' as const },
          { k: 'green_only', label: 'green-only', cls: 'green' as const },
          { k: 'yellow', label: 'yellow', cls: 'yellow' as const },
          { k: 'uncertain', label: 'uncertain', cls: 'uncertain' as const },
          { k: 'total_objects_exclusive', label: 'total', cls: null },
        ]
      : [
          { k: 'red_positive', label: 'red+', cls: 'red' as const },
          { k: 'green_positive', label: 'green+', cls: 'green' as const },
          { k: 'yellow', label: 'yellow overlap', cls: 'yellow' as const },
          { k: 'uncertain', label: 'uncertain', cls: 'uncertain' as const },
        ];
  return (
    <div className="tbl-wrap">
      <table className="tbl totals-tbl" data-testid={`counts-${title.replace(/\s/g, '-').toLowerCase()}`}>
        <caption>
          {title} <span className="muted">· {MODE_TEXT[mode].name}</span>
        </caption>
        <thead>
          <tr>
            <th></th>
            {cols.map((c) => (
              <th key={c.k} className="num">
                {c.cls && <GlyphIcon cls={c.cls} size={12} />} {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.c.scope}>
              <th scope="row">
                {r.label} {r.c.scope !== 'slide' && <DraftBadge final={r.c.is_final} />}
              </th>
              {cols.map((c) => (
                <td key={c.k} className={`num mono ${c.k === 'uncertain' && (r.c as unknown as Record<string, number>)[c.k] > 0 ? 'warn-text' : ''}`}>
                  {(r.c as unknown as Record<string, number>)[c.k]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Secondary convenience counts: flagged (debris-like) objects are NOT removed from the main counts above. */
function ConvenienceCounts({ counts, regions }: { counts: { slide: Counts; regions: Counts[] }; regions: { id: string; name: string }[] }) {
  if (counts.slide.red_positive_excl_debris == null && counts.slide.green_positive_excl_debris == null) return null;
  const rows = [{ label: 'All counting regions', c: counts.slide }, ...counts.regions.map((c) => ({ label: regions.find((r) => r.id === c.scope)?.name ?? c.scope, c }))];
  return (
    <div className="tbl-wrap secondary" data-testid="convenience-counts">
      <table className="tbl totals-tbl">
        <caption>
          Convenience counts <span className="muted">· secondary</span>
        </caption>
        <thead>
          <tr>
            <th></th>
            <th className="num">red-positive excl. debris-like</th>
            <th className="num">green-positive excl. debris-like</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.c.scope}>
              <th scope="row">{r.label}</th>
              <td className="num mono">{r.c.red_positive_excl_debris ?? '–'}</td>
              <td className="num mono">{r.c.green_positive_excl_debris ?? '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted small">convenience count – flagged objects are not removed from the counts above.</div>
    </div>
  );
}
