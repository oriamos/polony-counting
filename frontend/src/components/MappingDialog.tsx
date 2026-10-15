import { useEffect, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { ChannelRole } from '../api/types';
import { useApp } from '../state/store';
import { Modal } from './Modal';

export function MappingDialog() {
  const a = useApp((s) => s.current);
  const open = useApp((s) => s.mappingOpen);
  const patch = useApp((s) => s.patch);
  const setAnalysis = useApp((s) => s.setAnalysis);
  const [choice, setChoice] = useState<Record<number, ChannelRole | undefined>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (!a?.image) return;
    const init: Record<number, ChannelRole | undefined> = {};
    for (const c of a.image.channels) init[c.index] = c.role_confirmed && c.role ? c.role : undefined;
    setChoice(init);
    setErr('');
  }, [a?.id, a?.image, open]);

  if (!open || !a?.image) return null;
  const img = a.image;
  const chans = img.channels;
  const roles = chans.map((c) => choice[c.index]);
  const nRed = roles.filter((r) => r === 'red').length;
  const nGreen = roles.filter((r) => r === 'green').length;
  const allChosen = roles.every((r) => r !== undefined);
  const problems: string[] = [];
  if (!allChosen) problems.push('Assign every channel (red, green or ignore).');
  if (nRed > 1) problems.push('More than one channel is set to red.');
  if (nGreen > 1) problems.push('More than one channel is set to green.');
  // same rule as the server: exactly one red and exactly one green
  if (allChosen && (nRed !== 1 || nGreen !== 1)) problems.push('Exactly one channel must be red and exactly one green.');
  const hasDetection = a.objects.length > 0;
  const prevRed = chans.find((c) => c.role_confirmed && c.role === 'red')?.index;
  const prevGreen = chans.find((c) => c.role_confirmed && c.role === 'green')?.index;
  const isSwap = prevRed !== undefined && prevGreen !== undefined && choice[prevRed] === 'green' && choice[prevGreen] === 'red';
  const hasReviewWork =
    a.objects.some((o) => !o.deleted) || a.regions.some((r) => r.approved || r.inspected) || a.undo_depth > 0;

  const confirm = async () => {
    setBusy(true);
    setErr('');
    try {
      const next = await api.setChannels(
        a.id,
        chans.map((c) => ({ index: c.index, role: choice[c.index]! })),
      );
      setAnalysis(next);
      patch({ mappingOpen: false });
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Channel mapping" onClose={a.status === 'needs_channel_mapping' ? undefined : () => patch({ mappingOpen: false })} wide>
      <div data-testid="mapping-dialog">
        {img.composite_only && (
          <div className="notice warn" role="alert">
            <strong>Raw channels unavailable; counts are approximate.</strong> This is a colour composite, so per-channel intensities are
            derived from the rendered colours rather than measured raw signal.
          </div>
        )}
        <p>
          Tell the app which physical channel is shown in red and which in green. Metadata labels are only a <em>hint</em>; nothing is assumed
          until you confirm.
        </p>
        <div className="map-grid">
          {chans.map((c) => (
            <fieldset key={c.index} className="map-card">
              <legend>Channel {c.index}</legend>
              <div className="thumb-box">
                <img
                  src={`/api/analyses/${a.id}/image/${img.composite_only ? 'composite' : `ch${c.index}`}?max_dim=360`}
                  alt={`Preview of channel ${c.index}`}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = 'none';
                    (e.currentTarget.nextSibling as HTMLElement | null)?.removeAttribute('hidden');
                  }}
                />
                <div className="thumb-missing" hidden>
                  preview unavailable
                </div>
              </div>
              <dl className="kv">
                <dt>Metadata hint</dt>
                <dd className="mono">
                  {c.label_hint ?? <span className="muted">none</span>}
                  {c.suggested_role && (
                    <span className="tag draft" title="A hint only; nothing is applied until you confirm" data-testid={`suggested-${c.index}`}>
                      suggested: {c.suggested_role}
                    </span>
                  )}
                </dd>
                <dt>Source</dt>
                <dd className="mono small">
                  {c.source_file}
                  {img.kind !== 'raw_separate_files' ? ` · page ${c.page}` : ''}
                </dd>
                <dt>Range</dt>
                <dd className="mono">
                  {c.min} – {c.max}
                </dd>
                {c.saturated_fraction > 0 && (
                  <>
                    <dt>Saturated</dt>
                    <dd className="warn-text">{(c.saturated_fraction * 100).toFixed(3)}% of pixels at max</dd>
                  </>
                )}
              </dl>
              <div className="radio-row" role="radiogroup" aria-label={`Role of channel ${c.index}`}>
                {(['red', 'green', 'ignore'] as ChannelRole[]).map((r) => (
                  <label key={r} className={`radio ${choice[c.index] === r ? 'on' : ''} role-${r} ${c.suggested_role === r && choice[c.index] !== r ? 'suggested' : ''}`} data-suggested={c.suggested_role === r ? '1' : undefined}>
                    <input
                      type="radio"
                      name={`ch-${c.index}`}
                      value={r}
                      checked={choice[c.index] === r}
                      onChange={() => setChoice((p) => ({ ...p, [c.index]: r }))}
                      data-testid={`role-${c.index}-${r}`}
                    />
                    {r}
                  </label>
                ))}
              </div>
              {c.suggested_role && (
                <div className="small suggest-note" data-testid={`suggest-note-${c.index}`}>
                  Suggested from validated GenePix stacks – confirm ({c.suggested_role}). Nothing is selected for you.
                </div>
              )}
            </fieldset>
          ))}
          {chans.length === 0 && <div className="empty">No channels were found in this image.</div>}
        </div>
        {img.warnings.length > 0 && (
          <div className="notice warn small">
            {img.warnings.map((w, i) => (
              <div key={i}>{w}</div>
            ))}
          </div>
        )}
        {hasDetection && (
          <div className="notice warn small">Changing the mapping after detection marks detection as stale and clears region approvals.</div>
        )}
        {isSwap && hasReviewWork && (
          <div className="notice warn small" role="alert" data-testid="swap-warning">
            <strong>Swapping red and green on an analysis that is already being reviewed</strong> mirrors the red/green classes of all
            objects, swaps their evidence, negates the alignment shift, marks every object as <em>not reviewed</em>, clears all region
            approvals and inspection marks, and discards the undo history. Detection becomes stale.
          </div>
        )}
        {problems.length > 0 && allChosen && (
          <div className="notice err small" role="alert">
            {problems.join(' ')}
          </div>
        )}
        {err && (
          <div className="notice err" role="alert">
            {err}
          </div>
        )}
        <div className="modal-actions">
          {a.status !== 'needs_channel_mapping' && (
            <button className="btn" onClick={() => patch({ mappingOpen: false })}>
              Cancel
            </button>
          )}
          <button className="btn primary" disabled={busy || problems.length > 0} onClick={() => void confirm()} data-testid="confirm-mapping">
            {busy ? 'Saving…' : 'Confirm mapping'}
          </button>
        </div>
      </div>
    </Modal>
  );
}
