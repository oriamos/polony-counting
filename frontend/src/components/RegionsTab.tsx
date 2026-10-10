import { useState } from 'react';
import type { EmbeddedRoi, Region } from '../api/types';
import { polygonArea, polygonBounds, rectPolygon } from '../lib/geometry';
import { liveObjects, useApp } from '../state/store';
import { regionStatus, STATUS_COLOR } from './Overlay';

export function StatusBadge({ region }: { region: Region }) {
  const st = regionStatus(region);
  return (
    <span className="chip" style={{ borderColor: STATUS_COLOR[st], color: STATUS_COLOR[st] }} data-testid={`region-status-${region.id}`}>
      {st}
    </span>
  );
}

export function RegionsTab() {
  const a = useApp((s) => s.current)!;
  const selectedId = useApp((s) => s.selectedRegionId);
  const setTool = useApp((s) => s.setTool);
  const showSug = useApp((s) => s.showSuggestions);
  const patch = useApp((s) => s.patch);
  const saveRegions = useApp((s) => s.saveRegions);
  const locked = a.status === 'finalized';
  const counting = a.regions.filter((r) => r.kind === 'counting');
  const excl = a.regions.filter((r) => r.kind === 'exclusion');
  const rois = a.image?.embedded_rois ?? [];

  const adopt = (roi: EmbeddedRoi, i: number, kind: 'counting' | 'exclusion') => {
    const region: Region = {
      id: 'reg-' + Date.now().toString(36) + i,
      kind,
      name: roi.name || `${kind === 'counting' ? 'Region' : 'Exclusion'} from ROI ${i + 1}`,
      polygon: rectPolygon(roi.left, roi.top, roi.right, roi.bottom),
      approved: false,
      approved_at: null,
      approval_fingerprint: null,
      inspected: false,
    };
    void saveRegions([...a.regions, region]);
    patch({ selectedRegionId: region.id });
  };

  return (
    <div className="regions" data-testid="regions-tab">
      <div className="row between">
        <h3>Regions</h3>
        <div className="row">
          <button className="btn sm" disabled={locked} onClick={() => setTool('draw-count-rect')}>
            + Counting
          </button>
          <button className="btn sm" disabled={locked} onClick={() => setTool('draw-excl-rect')}>
            + Exclusion
          </button>
        </div>
      </div>
      <div className="muted small">
        Counts are only reported for counting regions. Exclusion masks remove areas (edges, debris) from counting. Select a region to edit vertices: drag a
        vertex, double-click an edge to add one, Alt+click a vertex to remove it.
      </div>
      {a.regions.length === 0 && (
        <div className="empty">
          No regions yet. With no counting region the whole image is detected, but at least one counting region is required to finalize. Use the toolbar to draw one.
        </div>
      )}
      {counting.map((r) => (
        <RegionCard key={r.id} region={r} selected={selectedId === r.id} locked={locked} />
      ))}
      {excl.length > 0 && <h4>Exclusion masks</h4>}
      {excl.map((r) => (
        <RegionCard key={r.id} region={r} selected={selectedId === r.id} locked={locked} />
      ))}
      {rois.length > 0 && (
        <section className="suggestions" data-testid="roi-suggestions">
          <div className="row between">
            <h4>Embedded ROIs (suggestions)</h4>
            <label className="chk small">
              <input type="checkbox" checked={showSug} onChange={(e) => patch({ showSuggestions: e.target.checked })} /> show
            </label>
          </div>
          <div className="muted small">Found inside the TIFF. Their meaning is not assumed and they are never applied automatically. Adopt one explicitly if it is right.</div>
          <ul className="sug-list">
            {rois.map((r, i) => (
              <li key={i}>
                <div>
                  <strong>{r.name || `ROI ${i + 1}`}</strong> <span className="muted">({r.source}, {r.roi_type})</span>
                  <div className="mono small muted">
                    bbox {r.left},{r.top} → {r.right},{r.bottom}
                  </div>
                  {r.roi_type.toLowerCase() !== 'rect' && r.roi_type.toLowerCase() !== 'rectangle' && (
                    <div className="small warn-text">Only the bounding box is available; adopting creates a rectangle.</div>
                  )}
                </div>
                <div className="row">
                  <button className="btn sm" disabled={locked} onClick={() => adopt(r, i, 'counting')}>
                    Adopt as counting
                  </button>
                  <button className="btn sm" disabled={locked} onClick={() => adopt(r, i, 'exclusion')}>
                    as exclusion
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function RegionCard({ region: r, selected, locked }: { region: Region; selected: boolean; locked: boolean }) {
  const a = useApp((s) => s.current)!;
  const counts = useApp((s) => s.counts);
  const err = useApp((s) => s.regionErrors[r.id]);
  const act = useApp((s) => s.act);
  const patch = useApp((s) => s.patch);
  const saveRegions = useApp((s) => s.saveRegions);
  const zoomBox = useApp((s) => s.zoomToBox);
  const [name, setName] = useState(r.name);
  const [editing, setEditing] = useState(false);
  const live = liveObjects(a).filter((o) => o.region_id === r.id);
  const unresolved = live.filter((o) => o.cls === 'uncertain').length;
  const unreviewed = live.filter((o) => !o.reviewed).length;
  const c = counts?.regions.find((x) => x.scope === r.id);
  const area = polygonArea(r.polygon);
  const isCounting = r.kind === 'counting';

  const hints: string[] = [];
  if (isCounting && !r.approved) {
    if (!r.inspected) hints.push('not yet marked as inspected for omissions');
    if (unresolved > 0) hints.push(`${unresolved} unresolved uncertain object${unresolved > 1 ? 's' : ''}`);
  }

  const rename = () => {
    setEditing(false);
    const n = name.trim() || r.name;
    setName(n);
    if (n !== r.name) void saveRegions(a.regions.map((x) => (x.id === r.id ? { ...x, name: n } : x)));
  };
  const remove = () => {
    if (!confirm(`Delete ${r.kind} region "${r.name}"?${r.approved ? ' Its approval will be lost.' : ''}`)) return;
    void saveRegions(a.regions.filter((x) => x.id !== r.id));
  };

  return (
    <article className={`rcard ${selected ? 'sel' : ''}`} data-testid={`region-${r.id}`} aria-label={`Region ${r.name}`}>
      <header className="rhead" onClick={() => patch({ selectedRegionId: selected ? null : r.id })}>
        {editing ? (
          <input
            className="input sm"
            autoFocus
            value={name}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => setName(e.target.value)}
            onBlur={rename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') rename();
              if (e.key === 'Escape') {
                setName(r.name);
                setEditing(false);
              }
            }}
            data-testid="region-name-input"
          />
        ) : (
          <strong
            className="rname"
            onDoubleClick={(e) => {
              e.stopPropagation();
              if (!locked) setEditing(true);
            }}
            title="Double-click to rename"
          >
            {r.name}
          </strong>
        )}
        <span className="chip">{r.kind}</span>
        {isCounting && <StatusBadge region={r} />}
      </header>
      <div className="rmeta small muted mono">
        {Math.round(area).toLocaleString()} px² · {live.length} object{live.length === 1 ? '' : 's'}
        {isCounting && unreviewed > 0 ? ` · ${unreviewed} unreviewed` : ''}
        {c ? ` · R${c.red_only} G${c.green_only} Y${c.yellow} ?${c.uncertain}` : ''}
      </div>
      {isCounting && (
        <>
          <label className="chk inspected">
            <input
              type="checkbox"
              checked={r.inspected}
              disabled={locked}
              onChange={(e) => void act('region_inspected', { region_id: r.id, inspected: e.target.checked })}
              data-testid={`inspected-${r.id}`}
            />
            I inspected this region for omissions
          </label>
          <div className="muted small">
            Toggle “hide overlays” (H) and look for real spots that were not detected; add them manually before approving.
          </div>
          {hints.length > 0 && <div className="small warn-text">Approval will be refused: {hints.join('; ')}.</div>}
          <div className="row wrap">
            {r.approved ? (
              <button className="btn sm" disabled={locked} onClick={() => void act('unapprove_region', { region_id: r.id })} data-testid={`unapprove-${r.id}`}>
                Unapprove
              </button>
            ) : (
              <button className="btn sm primary" disabled={locked} onClick={() => void act('approve_region', { region_id: r.id })} data-testid={`approve-${r.id}`}>
                Approve region
              </button>
            )}
            <button
              className="btn sm"
              disabled={locked || !r.inspected}
              title={r.inspected ? 'Mark all non-deleted, non-uncertain objects in this region as reviewed' : 'Tick “I inspected this region” first'}
              onClick={() => void act('accept_region', { region_id: r.id })}
              data-testid={`accept-${r.id}`}
            >
              Accept all proposals
            </button>
          </div>
        </>
      )}
      {err && (
        <div className="notice err small" role="alert" data-testid={`region-error-${r.id}`}>
          Refused: {err}
        </div>
      )}
      <div className="row wrap">
        <button
          className="btn sm ghost"
          onClick={() => {
            const b = polygonBounds(r.polygon);
            patch({ selectedRegionId: r.id });
            zoomBox(b.x0, b.y0, b.x1, b.y1);
          }}
        >
          Zoom to
        </button>
        <button className="btn sm ghost" disabled={locked} onClick={() => setEditing(true)}>
          Rename
        </button>
        <button className="btn sm ghost danger-hover" disabled={locked} onClick={remove} data-testid={`delete-region-${r.id}`}>
          Delete
        </button>
      </div>
    </article>
  );
}
