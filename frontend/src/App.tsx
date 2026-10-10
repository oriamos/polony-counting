import { useEffect } from 'react';
import { liveObjects, useApp } from './state/store';
import { LeftRail, StatusChip } from './components/LeftRail';
import { Viewer } from './components/Viewer';
import { StatusBar, ViewerToolbar } from './components/ViewerToolbar';
import { RightPanel } from './components/RightPanel';
import { MappingDialog } from './components/MappingDialog';
import { HelpDialog } from './components/HelpDialog';
import { Hotkeys } from './components/Hotkeys';
import { DraftBadge } from './components/TotalsTab';

export function App() {
  const theme = useApp((s) => s.theme);
  const refreshHealth = useApp((s) => s.refreshHealth);
  const refreshList = useApp((s) => s.refreshList);

  useEffect(() => {
    void refreshHealth();
    void refreshList();
  }, [refreshHealth, refreshList]);

  useEffect(() => {
    const el = document.documentElement;
    if (theme === 'auto') el.removeAttribute('data-theme');
    else el.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('polony.theme', theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  const status = useApp((s) => s.current?.status);
  const cid = useApp((s) => s.current?.id);
  const job = useApp((s) => s.job);
  useEffect(() => {
    // detection started elsewhere (e.g. before a page reload): poll until it settles
    if (status !== 'detecting' || job || !cid) return;
    const t = setInterval(() => {
      void import('./api/client').then(({ api }) =>
        api
          .getAnalysis(cid)
          .then((a) => useApp.getState().setAnalysis(a))
          .catch(() => undefined),
      );
    }, 1500);
    return () => clearInterval(t);
  }, [status, cid, job]);

  return (
    <div className="app">
      <Header />
      <LeftRail />
      <Main />
      <Toasts />
      <MappingDialog />
      <HelpDialog />
      <Hotkeys />
    </div>
  );
}

function Header() {
  const a = useApp((s) => s.current);
  const health = useApp((s) => s.health);
  const healthError = useApp((s) => s.healthError);
  const theme = useApp((s) => s.theme);
  const patch = useApp((s) => s.patch);
  const next = theme === 'auto' ? 'light' : theme === 'light' ? 'dark' : 'auto';
  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo" aria-hidden>
          <i className="r" />
          <i className="g" />
        </span>
        <strong>Polony Counter</strong>
      </div>
      {a && (
        <div className="crumb">
          <span className="aname" title={a.name}>{a.name}</span>
          <StatusChip status={a.status} />
          <DraftBadge final={a.status === 'finalized'} />
          {a.provenance.stale_detection && a.objects.length > 0 && <span className="chip warn">detection stale</span>}
        </div>
      )}
      <div className="spacer" />
      <span className={`health ${health ? 'ok' : 'bad'}`} title={health ? `API ${health.version} · Big-FISH ${health.bigfish_version}` : (healthError ?? 'checking…')} data-testid="health">
        <i /> {health ? `API ${health.version}` : healthError ? 'backend unreachable' : 'connecting…'}
      </span>
      <button className="btn sm ghost" onClick={() => patch({ theme: next })} title={`Theme: ${theme} (click for ${next})`}>
        {theme === 'dark' ? '☾' : theme === 'light' ? '☀' : '◐'} {theme}
      </button>
      <button className="btn sm ghost" onClick={() => patch({ helpOpen: true })} title="Keyboard shortcuts (?)" data-testid="help-btn">
        ? Help
      </button>
    </header>
  );
}

function Main() {
  const a = useApp((s) => s.current);
  const loading = useApp((s) => s.loadingCurrent);
  const err = useApp((s) => s.loadError);
  if (loading)
    return (
      <main className="center">
        <div className="empty big"><span className="spinner" /> Loading analysis…</div>
      </main>
    );
  if (err)
    return (
      <main className="center">
        <div className="empty big">
          <div className="notice err" role="alert">Could not open analysis: {err}</div>
        </div>
      </main>
    );
  if (!a)
    return (
      <main className="center">
        <div className="empty big">
          <h2>Welcome</h2>
          <p>Select an analysis on the left, or drop one or two TIFF files to start a new one.</p>
          <p className="muted">Workflow: upload → confirm channel mapping → draw counting regions → run detection → review → approve regions → finalize &amp; export.</p>
        </div>
      </main>
    );
  return (
    <>
      <main className="center">
        <Banners />
        <ViewerToolbar />
        <div className="viewer-wrap">
          {a.image ? <Viewer /> : <div className="empty big">This analysis has no image data.</div>}
        </div>
        <StatusBar />
      </main>
      <RightPanel />
    </>
  );
}

function Banners() {
  const a = useApp((s) => s.current)!;
  const patch = useApp((s) => s.patch);
  const job = useApp((s) => s.job);
  const img = a.image;
  const stale = a.provenance.stale_detection && liveObjects(a).length > 0;
  return (
    <div className="banners">
      {a.status === 'needs_channel_mapping' && (
        <div className="banner warn" role="alert" data-testid="banner-mapping">
          <strong>Channel mapping required.</strong> Assign red / green before processed views, detection or counts are available.
          <button className="btn sm primary" onClick={() => patch({ mappingOpen: true })} data-testid="open-mapping">
            Map channels…
          </button>
        </div>
      )}
      {img?.composite_only && (
        <div className="banner warn" role="status" data-testid="banner-composite">
          <strong>Raw channels unavailable; counts are approximate.</strong> This image is a colour composite, so per-channel signal is derived from rendered colours.
        </div>
      )}
      {a.status === 'finalized' && (
        <div className="banner ok" role="status">
          <strong>Finalized.</strong> Editing is locked; use Finish → Reopen to make changes.
        </div>
      )}
      {a.status === 'detecting' && !job && (
        <div className="banner" role="status">
          <span className="spinner" /> Detection is running…
        </div>
      )}
      {stale && (
        <div className="banner warn" role="status">
          Detection is stale (regions, mapping or parameters changed). Re-run detection from the Detect tab.
        </div>
      )}
      {img?.warnings.map((w, i) => (
        <div key={i} className="banner" role="status">
          {w}
        </div>
      ))}
    </div>
  );
}

function Toasts() {
  const toasts = useApp((s) => s.toasts);
  const dismiss = useApp((s) => s.dismissToast);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role={t.kind === 'error' ? 'alert' : 'status'} data-testid="toast">
          <span>{t.text}</span>
          <button className="btn sm ghost" onClick={() => dismiss(t.id)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
