import { useEffect, useMemo, useRef, useState } from 'react';
import { imageUrl } from '../api/client';
import type { DetectedObject, Region } from '../api/types';
import { create } from 'zustand';
import {
  distToSegment,
  imageToScreen,
  panBy,
  rectPolygon,
  round2,
  screenToImage,
  visibleRect,
  zoomAbout,
  type Poly,
} from '../lib/geometry';
import { ringRadiusImage } from '../lib/glyph';
import { cellAt, coverageState, exclusionCoverage, fromApiGrid, storedGridCells } from '../lib/labgrid';
import { liveObjects, useApp } from '../state/store';
import { ChamberPreviewLayer, ReferenceGridPreviewLayer, ReferencePointsLayer, GridCellsLayer, GridPreviewLayer, HaloText, ObjectsLayer, RawLayer, RejectedLayer, RegionLabelsLayer, RegionsLayer, SuggestionsLayer, Glyph } from './Overlay';
import { rawKey } from '../state/store';
import { ScaleBar } from './ScaleBar';

export const useCursor = create<{ pt: { x: number; y: number } | null; set: (p: { x: number; y: number } | null) => void }>((set) => ({
  pt: null,
  set: (pt) => set({ pt }),
}));

type Drag =
  | { kind: 'pan'; sx: number; sy: number }
  | { kind: 'obj'; id: string; sx: number; sy: number; moved: boolean; shift: boolean; wasSelected: boolean }
  | { kind: 'vertex'; regionId: string; index: number; sx: number; sy: number; moved: boolean }
  | { kind: 'rect'; a: [number, number]; b: [number, number]; tool: string; sx: number; sy: number }
  | { kind: 'band'; a: [number, number]; b: [number, number]; sx: number; sy: number }
  | { kind: 'click'; sx: number; sy: number; moved: boolean; shift: boolean };

const uid = () => 'reg-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

export function Viewer() {
  const a = useApp((s) => s.current);
  const view = useApp((s) => s.view);
  const imageView = useApp((s) => s.imageView);
  const display = useApp((s) => s.display[s.imageView]);
  const tool = useApp((s) => s.tool);
  const selection = useApp((s) => s.selection);
  const selectedRegionId = useApp((s) => s.selectedRegionId);
  const hide = useApp((s) => s.hideOverlays);
  const classVisible = useApp((s) => s.classVisible);
  const showLabels = useApp((s) => s.showLabels);
  const showRegions = useApp((s) => s.showRegions);
  const showSuggestions = useApp((s) => s.showSuggestions);
  const preview = useApp((s) => s.preview);
  const previewBusy = useApp((s) => s.previewBusy);
  const splitPts = useApp((s) => s.splitPts);
  const layers = useApp((s) => s.layers);
  const rawCache = useApp((s) => s.rawCache);
  const st = useApp;

  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const spaceDown = useRef(false);
  const [rect, setRect] = useState<{ a: [number, number]; b: [number, number]; tool: string } | null>(null);
  const [band, setBand] = useState<{ a: [number, number]; b: [number, number] } | null>(null);
  const [poly, setPoly] = useState<[number, number][]>([]);
  const [polyCursor, setPolyCursor] = useState<[number, number] | null>(null);
  const [objDrag, setObjDrag] = useState<{ id: string; x: number; y: number } | null>(null);
  const [regionDraft, setRegionDraft] = useState<Region | null>(null);

  // ---------------------------------------------------------- image fetching
  const [img, setImg] = useState<{ url: string; key: string } | null>(null);
  const [imgState, setImgState] = useState<'idle' | 'loading' | 'error'>('idle');
  const [imgError, setImgError] = useState('');
  const aid = a?.id;
  // Before the channel mapping is confirmed the backend refuses role-based views (409); show raw channel 0.
  const unmapped = !!a?.image && !a.image.composite_only && a.image.channels.length > 0 && !a.image.channels.every((c) => c.role_confirmed);
  const fetchView = unmapped ? `ch${a!.image!.channels[0].index}` : imageView;
  const imgVer = useMemo(() => {
    if (!a?.image) return '';
    const roles = a.image.channels.map((c) => `${c.index}:${c.role}`).join(',');
    const bg = imageView.startsWith('processed') ? JSON.stringify([a.detection_request.red.background, a.detection_request.green.background]) : '';
    return `${roles}|${bg}|${a.image.sha256 ? Object.keys(a.image.sha256).length : 0}`;
  }, [a?.image, a?.detection_request, imageView]);

  useEffect(() => {
    if (!aid) return;
    const ctl = new AbortController();
    const key = `${aid}|${fetchView}|${display.lo}|${display.hi}|${display.gamma}|${imgVer}`;
    setImgState('loading');
    const t = setTimeout(async () => {
      try {
        const res = await fetch(imageUrl(aid, fetchView, display), { signal: ctl.signal });
        if (!res.ok) {
          let msg = `HTTP ${res.status}`;
          try {
            const j = await res.json();
            msg = j.message ?? j.detail?.message ?? (typeof j.detail === 'string' ? j.detail : msg);
          } catch {
            /* ignore */
          }
          throw new Error(msg);
        }
        const lo = parseFloat(res.headers.get('X-Auto-Lo') ?? '');
        const hi = parseFloat(res.headers.get('X-Auto-Hi') ?? '');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setImg((old) => {
          if (old) setTimeout(() => URL.revokeObjectURL(old.url), 2000);
          return { url, key };
        });
        if (Number.isFinite(lo) && Number.isFinite(hi)) st.getState().patch({ autoRange: { lo, hi } });
        setImgState('idle');
      } catch (e) {
        if ((e as Error).name === 'AbortError') return;
        setImgError((e as Error).message);
        setImgState('error');
      }
    }, 180);
    return () => {
      clearTimeout(t);
      ctl.abort();
    };
  }, [aid, fetchView, display.lo, display.hi, display.gamma, imgVer, st]);

  // reset image when analysis changes
  useEffect(() => {
    setImg(null);
    setPoly([]);
    setRect(null);
  }, [aid]);

  // ---------------------------------------------------------- viewport size
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      st.getState().setViewport(r.width, r.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [st, aid]);

  // ---------------------------------------------------------- wheel zoom (non-passive)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      const f = Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0016));
      st.getState().setView(zoomAbout(st.getState().view, e.clientX - r.left, e.clientY - r.top, f));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [st, aid]);

  // ---------------------------------------------------------- local key handling (drawing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
      if (e.code === 'Space') spaceDown.current = e.type === 'keydown';
      if (e.type !== 'keydown') return;
      if (e.key === 'Escape') {
        setPoly([]);
        setRect(null);
        drag.current = null;
        st.getState().setTool('select');
      }
      if (e.key === 'Enter' && poly.length >= 3) {
        e.preventDefault();
        e.stopPropagation();
        commitPoly(poly);
      }
      if (e.key === 'Backspace' && poly.length) setPoly((p) => p.slice(0, -1));
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('keyup', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('keyup', onKey, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [poly, tool]);

  useEffect(() => {
    if (!tool.startsWith('draw-') || !tool.endsWith('poly')) setPoly([]);
  }, [tool]);

  if (!a || !a.image) return null;
  const W = a.image.width;
  const H = a.image.height;
  const scale = view.scale;
  const live = liveObjects(a);
  const shown = live.filter((o) => classVisible[o.cls]);

  // ---------------------------------------------------------- helpers
  const local = (e: { clientX: number; clientY: number }) => {
    const r = ref.current!.getBoundingClientRect();
    return { sx: e.clientX - r.left, sy: e.clientY - r.top };
  };
  const toImg = (sx: number, sy: number): [number, number] => {
    const p = screenToImage(st.getState().view, sx, sy);
    return [round2(p.x), round2(p.y)];
  };
  const inImage = (x: number, y: number) => x >= -0.5 && y >= -0.5 && x <= W - 0.5 && y <= H - 0.5;

  function hitObject(sx: number, sy: number): DetectedObject | null {
    const v = st.getState().view;
    const rad = Math.max(10, ringRadiusImage(v.scale) * v.scale + 3);
    let best: DetectedObject | null = null;
    let bd = rad;
    for (const o of shown) {
      const p = imageToScreen(v, o.x, o.y);
      const d = Math.hypot(p.x - sx, p.y - sy);
      if (d <= bd) {
        bd = d;
        best = o;
      }
    }
    return best;
  }

  function hitVertex(sx: number, sy: number): { regionId: string; index: number } | null {
    const v = st.getState().view;
    const rid = st.getState().selectedRegionId;
    const reg = st.getState().current?.regions.find((r) => r.id === rid);
    if (!reg) return null;
    for (let i = 0; i < reg.polygon.length; i++) {
      const p = imageToScreen(v, reg.polygon[i][0], reg.polygon[i][1]);
      if (Math.hypot(p.x - sx, p.y - sy) <= 9) return { regionId: reg.id, index: i };
    }
    return null;
  }

  function hitRegionEdge(sx: number, sy: number, tol = 7): { region: Region; seg: number; x: number; y: number } | null {
    const v = st.getState().view;
    let best: { region: Region; seg: number; x: number; y: number } | null = null;
    let bd = tol;
    for (const reg of st.getState().current?.regions ?? []) {
      const pts = reg.polygon;
      for (let i = 0; i < pts.length; i++) {
        const p = imageToScreen(v, pts[i][0], pts[i][1]);
        const q = imageToScreen(v, pts[(i + 1) % pts.length][0], pts[(i + 1) % pts.length][1]);
        const d = distToSegment(sx, sy, p.x, p.y, q.x, q.y);
        if (d.d <= bd) {
          bd = d.d;
          const im = screenToImage(v, d.x, d.y);
          best = { region: reg, seg: i, x: im.x, y: im.y };
        }
      }
    }
    return best;
  }

  function commitRegion(kind: 'counting' | 'exclusion', polygon: Poly) {
    const regs = st.getState().current!.regions;
    const n = regs.filter((r) => r.kind === kind).length + 1;
    const region: Region = {
      id: uid(),
      kind,
      name: kind === 'counting' ? `Region ${n}` : `Exclusion ${n}`,
      polygon,
      approved: false,
      approved_at: null,
      approval_fingerprint: null,
      inspected: false,
    };
    void st.getState().saveRegions([...regs, region]);
    st.getState().patch({ selectedRegionId: region.id, rightTab: 'regions', tool: 'select', selection: [] });
  }

  /** grid-pick: switch the stored-grid cell under (x, y) on/off. Cells outside the image or fully inside an exclusion
   * mask cannot be selected. */
  function toggleGridAt(x: number, y: number) {
    const S = st.getState();
    const a = S.current;
    const g = a?.slide.grid;
    if (!a || !g) {
      S.toast('error', 'No grid yet: generate and apply one in the Regions tab');
      return;
    }
    const hit = inImage(x, y) ? cellAt(fromApiGrid(g), x, y) : null; // a cell wholly off the image cannot be hit
    if (!hit) return;
    const cell = storedGridCells(a).find((c) => c.index === hit.index)!;
    if (cell.state === 'missing') {
      S.toast('error', `Field ${cell.index} was deleted; apply the grid again to get it back`);
      return;
    }
    if (cell.state === 'off') {
      const excl = a.regions.filter((r) => r.kind === 'exclusion').map((r) => r.polygon as Poly);
      if (coverageState(exclusionCoverage(cell.polygon, excl)) === 'full') {
        S.toast('error', `Field ${cell.index} is fully inside an exclusion mask`);
        return;
      }
    }
    void S.setGridField(cell.index, cell.state === 'off');
  }

  function commitPoly(pts: [number, number][]) {
    if (pts.length < 3) return;
    const kind = st.getState().tool === 'draw-excl-poly' ? 'exclusion' : 'counting';
    commitRegion(kind, pts);
    setPoly([]);
    setPolyCursor(null);
  }

  // ---------------------------------------------------------- pointer handlers
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 1) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const { sx, sy } = local(e);
    const S = st.getState();
    const [x, y] = toImg(sx, sy);
    if (e.button === 1 || spaceDown.current) {
      drag.current = { kind: 'pan', sx, sy };
      return;
    }
    if (S.tool === 'select') {
      const hv = hitVertex(sx, sy);
      if (hv) {
        if (e.altKey) {
          const reg = S.current!.regions.find((r) => r.id === hv.regionId)!;
          if (reg.polygon.length > 3) void S.saveRegions(S.current!.regions.map((r) => (r.id === reg.id ? { ...r, polygon: r.polygon.filter((_, i) => i !== hv.index) } : r)));
          else S.toast('error', 'A region needs at least 3 vertices');
          drag.current = null;
          return;
        }
        drag.current = { kind: 'vertex', regionId: hv.regionId, index: hv.index, sx, sy, moved: false };
        return;
      }
      const o = hitObject(sx, sy);
      if (o) {
        const wasSelected = S.selection.includes(o.id);
        if (!e.shiftKey && !wasSelected) S.setSelection([o.id]);
        else if (e.shiftKey && !wasSelected) S.setSelection([...S.selection, o.id]);
        S.patch({ selectedRegionId: null });
        drag.current = { kind: 'obj', id: o.id, sx, sy, moved: false, shift: e.shiftKey, wasSelected };
        return;
      }
      if (e.shiftKey) {
        drag.current = { kind: 'band', a: [x, y], b: [x, y], sx, sy };
        return;
      }
      drag.current = { kind: 'click', sx, sy, moved: false, shift: false };
      return;
    }
    if (S.tool === 'draw-count-rect' || S.tool === 'draw-excl-rect' || S.tool === 'preview-window') {
      drag.current = { kind: 'rect', a: [x, y], b: [x, y], tool: S.tool, sx, sy };
      setRect({ a: [x, y], b: [x, y], tool: S.tool });
      return;
    }
    // add / split / polygon tools: a click (without drag) acts, a drag pans
    drag.current = { kind: 'click', sx, sy, moved: false, shift: false };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { sx, sy } = local(e);
    const S = st.getState();
    const [x, y] = toImg(sx, sy);
    useCursor.getState().set(inImage(x, y) ? { x, y } : null);
    if (S.tool === 'grid-pick') {
      const g = S.current?.slide.grid;
      const hv = g ? cellAt(fromApiGrid(g), x, y)?.index ?? null : null;
      if (hv !== S.gridHover) S.patch({ gridHover: hv });
    }
    if (S.tool.endsWith('poly') && poly.length) setPolyCursor([x, y]);
    const d = drag.current;
    if (!d) return;
    if (d.kind === 'pan' || (d.kind === 'click' && Math.hypot(sx - d.sx, sy - d.sy) > 3)) {
      S.setView(panBy(S.view, sx - d.sx, sy - d.sy));
      drag.current = { kind: 'pan', sx, sy };
      return;
    }
    if (d.kind === 'obj') {
      if (!d.moved && Math.hypot(sx - d.sx, sy - d.sy) > 4) d.moved = true;
      if (d.moved) setObjDrag({ id: d.id, x, y });
    } else if (d.kind === 'vertex') {
      if (!d.moved && Math.hypot(sx - d.sx, sy - d.sy) > 2) d.moved = true;
      if (d.moved) {
        const reg = S.current!.regions.find((r) => r.id === d.regionId)!;
        setRegionDraft({ ...reg, polygon: reg.polygon.map((p, i) => (i === d.index ? [x, y] : p)) as Poly });
      }
    } else if (d.kind === 'rect') {
      d.b = [x, y];
      setRect({ a: d.a, b: d.b, tool: d.tool });
    } else if (d.kind === 'band') {
      d.b = [x, y];
      setBand({ a: d.a, b: d.b });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    if (!d) return;
    const { sx, sy } = local(e);
    const S = st.getState();
    const [x, y] = toImg(sx, sy);
    if (d.kind === 'obj') {
      setObjDrag(null);
      if (d.moved) {
        void S.act('move', { object_id: d.id, x, y });
      } else if (d.wasSelected && d.shift) {
        S.setSelection(S.selection.filter((i) => i !== d.id));
      } else if (d.wasSelected && !d.shift && S.selection.length > 1) {
        S.setSelection([d.id]);
      }
      return;
    }
    if (d.kind === 'vertex') {
      const draft = regionDraft;
      setRegionDraft(null);
      if (d.moved && draft) void S.saveRegions(S.current!.regions.map((r) => (r.id === draft.id ? { ...r, polygon: draft.polygon } : r)));
      return;
    }
    if (d.kind === 'band') {
      setBand(null);
      const x0 = Math.min(d.a[0], d.b[0]),
        x1 = Math.max(d.a[0], d.b[0]),
        y0 = Math.min(d.a[1], d.b[1]),
        y1 = Math.max(d.a[1], d.b[1]);
      const ids = shown.filter((o) => o.x >= x0 && o.x <= x1 && o.y >= y0 && o.y <= y1).map((o) => o.id);
      S.setSelection(Array.from(new Set([...S.selection, ...ids])));
      return;
    }
    if (d.kind === 'rect') {
      setRect(null);
      const w = Math.abs(d.b[0] - d.a[0]) * S.view.scale;
      const h = Math.abs(d.b[1] - d.a[1]) * S.view.scale;
      if (w < 4 || h < 4) return;
      const p = rectPolygon(d.a[0], d.a[1], d.b[0], d.b[1]);
      if (d.tool === 'preview-window') {
        const win: [number, number, number, number] = [
          Math.max(0, Math.round(p[0][0])),
          Math.max(0, Math.round(p[0][1])),
          Math.min(W, Math.round(p[2][0])),
          Math.min(H, Math.round(p[2][1])),
        ];
        S.setTool('select');
        S.patch({ rightTab: 'detect' });
        void S.runPreview(win);
      } else commitRegion(d.tool === 'draw-excl-rect' ? 'exclusion' : 'counting', p);
      return;
    }
    if (d.kind === 'pan') return;
    // plain click
    if (S.tool === 'select') {
      if (!inImage(x, y)) {
        S.setSelection([]);
        return;
      }
      const he = hitRegionEdge(sx, sy);
      if (he) {
        S.patch({ selectedRegionId: he.region.id, selection: [], rightTab: 'regions' });
      } else {
        S.patch({ selection: [], selectedRegionId: null });
      }
    } else if (S.tool === 'add') {
      if (!inImage(x, y)) {
        S.toast('error', 'Click inside the image to add a spot');
        return;
      }
      void S.act('add', { x, y, cls: S.addClass });
    } else if (S.tool === 'split') {
      const sel = S.selection;
      if (sel.length !== 1) {
        S.toast('error', 'Select exactly one spot, then press S and click two positions');
        return;
      }
      const pts = [...S.splitPts, [x, y] as [number, number]];
      if (pts.length < 2) S.patch({ splitPts: pts });
      else {
        const o = S.current!.objects.find((q) => q.id === sel[0]);
        S.patch({ splitPts: [], tool: 'select' });
        if (o) void S.act('split', { object_id: o.id, positions: pts, classes: [o.cls, o.cls] });
      }
    } else if (S.tool === 'grid-pick') {
      toggleGridAt(x, y);
    } else if (S.tool === 'draw-count-poly' || S.tool === 'draw-excl-poly') {
      if (poly.length >= 3) {
        const f = imageToScreen(S.view, poly[0][0], poly[0][1]);
        if (Math.hypot(f.x - sx, f.y - sy) <= 9) {
          commitPoly(poly);
          return;
        }
      }
      setPoly((p) => [...p, [x, y]]);
    }
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    const S = st.getState();
    const { sx, sy } = local(e);
    if (S.tool === 'draw-count-poly' || S.tool === 'draw-excl-poly') {
      // the two clicks of the double-click each added a point at the same place
      const pts = poly.length >= 2 ? poly.slice(0, -1) : poly;
      commitPoly(pts);
      return;
    }
    if (S.tool === 'select') {
      const he = hitRegionEdge(sx, sy);
      if (he && he.region.id === S.selectedRegionId) {
        const polyPts = he.region.polygon.slice();
        polyPts.splice(he.seg + 1, 0, [round2(he.x), round2(he.y)]);
        void S.saveRegions(S.current!.regions.map((r) => (r.id === he.region.id ? { ...r, polygon: polyPts } : r)));
      }
    }
  };

  const cursorStyle =
    tool === 'select' ? 'default' : tool === 'add' ? 'copy' : tool === 'grid-pick' ? 'pointer' : 'crosshair';

  // ---------------------------------------------------------- culling
  const vr = visibleRect(view, st.getState().viewport.w, st.getState().viewport.h);
  const m = 40 / scale;
  const inView = (o: { x: number; y: number }) => o.x >= vr.x0 - m && o.x <= vr.x1 + m && o.y >= vr.y0 - m && o.y <= vr.y1 + m;
  const visObjs = shown.filter((o) => inView(o) || selection.includes(o.id));
  const visPrev = preview ? preview.objects.filter((o) => !o.deleted && classVisible[o.cls] && inView(o)) : [];
  const regions = a.regions;
  const k = 1 / scale;
  const RAW_DRAW_CAP = 4000;
  const rawVisible = (eng: 'bigfish' | 'spotiflow') =>
    a.image!.channels
      .filter((c) => c.role !== 'ignore')
      .map((c) => c.page)
      .filter((pg, i, arr) => arr.indexOf(pg) === i)
      .flatMap((pg) => rawCache[rawKey(eng, pg, layers.rawExcluded)]?.items ?? [])
      .filter((d) => (layers.rawExcluded || d.valid) && inView(d))
      .slice(0, RAW_DRAW_CAP);
  const rejectedVis = layers.rejected ? a.objects.filter((o) => o.deleted && inView(o)) : [];
  const sel1 = selection.length === 1 ? live.find((o) => o.id === selection[0]) : null;

  return (
    <div
      ref={ref}
      className="viewer-canvas"
      data-testid="viewer"
      data-scale={scale}
      data-tx={view.tx}
      data-ty={view.ty}
      data-hidden={hide ? '1' : '0'}
      data-img-ready={img && imgState === 'idle' ? '1' : '0'}
      style={{ cursor: cursorStyle }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => {
        useCursor.getState().set(null);
        if (st.getState().gridHover != null) st.getState().patch({ gridHover: null });
      }}
      onDoubleClick={onDoubleClick}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div
        className="world"
        style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${scale})`, width: 0, height: 0 }}
      >
        {img && (
          <img
            src={img.url}
            alt=""
            draggable={false}
            data-testid="viewer-image"
            style={{
              position: 'absolute',
              // NB: the half-pixel shift must be a transform, not left/top: Chrome pixel-snaps
              // fractional layout offsets of replaced elements, which misaligns by 0.5 image px.
              left: 0,
              top: 0,
              transform: 'translate(-0.5px, -0.5px)',
              width: W,
              height: H,
              maxWidth: 'none',
              imageRendering: scale >= 2.5 ? 'pixelated' : 'auto',
              userSelect: 'none',
            }}
          />
        )}
        <svg
          data-testid="overlay-svg"
          style={{ position: 'absolute', left: 0, top: 0, width: 1, height: 1, overflow: 'visible', pointerEvents: 'none' }}
        >
          {showSuggestions && a.image.embedded_rois.length > 0 && <SuggestionsLayer rois={a.image.embedded_rois} scale={scale} />}
          {showRegions && <RegionsLayer regions={regions} scale={scale} selectedId={selectedRegionId} draft={regionDraft} />}
          {showRegions && <GridCellsLayer scale={scale} />}
          <GridPreviewLayer scale={scale} />
          <ReferenceGridPreviewLayer scale={scale} />
          <ChamberPreviewLayer scale={scale} />
          <ReferencePointsLayer scale={scale} />
          {!hide && layers.rawBigfish && <RawLayer items={rawVisible('bigfish')} engine="bigfish" scale={scale} />}
          {!hide && layers.rawSpotiflow && <RawLayer items={rawVisible('spotiflow')} engine="spotiflow" scale={scale} />}
          {!hide && layers.rejected && <RejectedLayer objects={rejectedVis} scale={scale} />}
          {!hide && (
            <g data-testid="objects-layer">
              <ObjectsLayer
                objects={visObjs}
                scale={scale}
                selection={selection}
                showLabels={showLabels}
                showReview={true}
                dragOverride={objDrag}
                consensus={layers.consensus}
              />
              {visPrev.length > 0 && (
                <g data-testid="preview-layer">
                  {visPrev.map((o) => (
                    <Glyph key={o.id} o={o} scale={scale} preview />
                  ))}
                </g>
              )}
            </g>
          )}
          {showRegions && <RegionLabelsLayer regions={regions} scale={scale} draft={regionDraft} />}
          {preview && (
            <g>
              <rect
                x={preview.window[0]}
                y={preview.window[1]}
                width={preview.window[2] - preview.window[0]}
                height={preview.window[3] - preview.window[1]}
                fill="none"
                stroke="rgba(0,0,0,.9)"
                strokeWidth={4 * k}
              />
              <rect
                x={preview.window[0]}
                y={preview.window[1]}
                width={preview.window[2] - preview.window[0]}
                height={preview.window[3] - preview.window[1]}
                fill="none"
                stroke="#ffb020"
                strokeWidth={1.8 * k}
                strokeDasharray={`${6 * k} ${4 * k}`}
              />
              <HaloText x={preview.window[0] + 4 * k} y={preview.window[1] + 14 * k} scale={scale} text="PREVIEW – temporary, not saved" fill="#ffb020" />
            </g>
          )}
          {rect && (
            <g>
              <rect
                x={Math.min(rect.a[0], rect.b[0])}
                y={Math.min(rect.a[1], rect.b[1])}
                width={Math.abs(rect.b[0] - rect.a[0])}
                height={Math.abs(rect.b[1] - rect.a[1])}
                fill="rgba(255,255,255,0.08)"
                stroke="#000"
                strokeWidth={4 * k}
              />
              <rect
                x={Math.min(rect.a[0], rect.b[0])}
                y={Math.min(rect.a[1], rect.b[1])}
                width={Math.abs(rect.b[0] - rect.a[0])}
                height={Math.abs(rect.b[1] - rect.a[1])}
                fill="none"
                stroke={rect.tool === 'draw-excl-rect' ? '#ff4fb0' : rect.tool === 'preview-window' ? '#ffb020' : '#4da3ff'}
                strokeWidth={1.8 * k}
                strokeDasharray={`${5 * k} ${3 * k}`}
              />
            </g>
          )}
          {band && (
            <rect
              x={Math.min(band.a[0], band.b[0])}
              y={Math.min(band.a[1], band.b[1])}
              width={Math.abs(band.b[0] - band.a[0])}
              height={Math.abs(band.b[1] - band.a[1])}
              fill="rgba(0,229,255,0.1)"
              stroke="#00e5ff"
              strokeWidth={1.4 * k}
              strokeDasharray={`${4 * k} ${3 * k}`}
            />
          )}
          {poly.length > 0 && (
            <g>
              <polyline points={[...poly, ...(polyCursor ? [polyCursor] : [])].map((p) => p.join(',')).join(' ')} fill="none" stroke="#000" strokeWidth={4 * k} />
              <polyline
                points={[...poly, ...(polyCursor ? [polyCursor] : [])].map((p) => p.join(',')).join(' ')}
                fill="none"
                stroke={tool === 'draw-excl-poly' ? '#ff4fb0' : '#4da3ff'}
                strokeWidth={1.8 * k}
              />
              {poly.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 6 : 4) * k} fill={i === 0 ? '#36d399' : '#fff'} stroke="#000" strokeWidth={1.4 * k} />
              ))}
            </g>
          )}
          {tool === 'split' && sel1 && (
            <g>
              <HaloText x={sel1.x + 10 * k} y={sel1.y - 10 * k} scale={scale} text={`split: click position ${splitPts.length + 1} of 2`} fill="#00e5ff" />
              {splitPts.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r={4 * k} fill="#00e5ff" stroke="#000" strokeWidth={1.2 * k} />
              ))}
            </g>
          )}
        </svg>
      </div>

      {imgState === 'loading' && (
        <div className="viewer-chip" style={{ top: 10, right: 10 }} role="status">
          <span className="spinner" /> loading image…
        </div>
      )}
      {imgState === 'error' && (
        <div className="viewer-error" role="alert">
          <strong>Image could not be rendered</strong>
          <div>{imgError}</div>
          <div className="muted">Check that the channel mapping is confirmed and the backend is running.</div>
        </div>
      )}
      {!img && imgState === 'loading' && <div className="viewer-empty">Loading image…</div>}
      {unmapped && <div className="viewer-chip warn" style={{ bottom: 44, left: 10 }} data-testid="unmapped-chip">Channel mapping not confirmed: showing raw channel {a.image.channels[0].index} (grey)</div>}
      {hide && <div className="viewer-chip warn" style={{ top: 10, left: 10 }}>All overlays hidden (H) – check the raw signal for omissions</div>}
      {previewBusy && (
        <div className="viewer-chip" style={{ top: 10, left: '50%', transform: 'translateX(-50%)' }}>
          <span className="spinner" /> running preview…
        </div>
      )}
      {tool !== 'select' && <ToolHint tool={tool} polyN={poly.length} />}
      <ScaleBar view={view} pixelSizeUm={a.image.pixel_size_um} />
    </div>
  );
}

function ToolHint({ tool, polyN }: { tool: string; polyN: number }) {
  const t: Record<string, string> = {
    add: 'Add mode: click to place a spot (class chosen in toolbar). Esc to leave.',
    split: 'Split: click two positions for the two resulting spots. Esc to cancel.',
    'draw-count-rect': 'Drag a rectangle for a COUNTING region.',
    'draw-excl-rect': 'Drag a rectangle for an EXCLUSION mask.',
    'draw-count-poly': `Click vertices for a COUNTING polygon (${polyN} placed). Double-click / Enter / click first point to finish. Backspace undoes last vertex.`,
    'draw-excl-poly': `Click vertices for an EXCLUSION polygon (${polyN} placed). Double-click / Enter / click first point to finish.`,
    'preview-window': 'Drag a window to run a detection preview (temporary, not saved).',
    'grid-pick': 'Click fields to count them; click again to skip. A all · N none · I invert · Esc to leave.',
  };
  return <div className="viewer-chip tool" style={{ bottom: 44, left: '50%', transform: 'translateX(-50%)' }}>{t[tool] ?? ''}</div>;
}
