import { useEffect, useRef, useState } from 'react';
import { useApp } from '../state/store';

type Storage = {
  backend: 'sqlite' | 'postgres';
  host: string | null;
  database: string | null;
  schema: string | null;
  connected: boolean;
  latency_ms: number | null;
  last_error: string | null;
};

/** Where analyses live: shared database (Neon) or this computer only. */
export function StorageBadge() {
  const [s, setS] = useState<Storage | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch('/api/storage')
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => alive && setS(j))
        .catch(() => alive && setS(null));
    load();
    const t = setInterval(load, 30000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  if (!s) return null;
  const shared = s.backend === 'postgres';
  const label = !shared ? 'Local only' : s.connected ? 'Shared database (Neon): connected' : 'Shared database (Neon): NOT reachable';
  const title = shared
    ? `${s.host ?? ''}/${s.database ?? ''} · schema ${s.schema ?? ''}${s.latency_ms != null ? ` · ${s.latency_ms} ms` : ''}${s.last_error ? ` · ${s.last_error}` : ''}`
    : 'Analyses are stored on this computer only';
  return (
    <span className={`health ${!shared || s.connected ? 'ok' : 'bad'}`} title={title} data-testid="storage-badge">
      <i /> {label}
    </span>
  );
}

type Missing = { file: string; sha256: string }[];

/** Shown when this analysis was made on another computer: its original TIFF(s) are not here. */
export function OriginalsBanner({ id }: { id: string }) {
  const [missing, setMissing] = useState<Missing>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const toast = useApp((s) => s.toast);

  useEffect(() => {
    let alive = true;
    setMissing([]);
    fetch(`/api/analyses/${id}/originals`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => alive && j && setMissing(j.missing ?? []))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [id]);

  if (missing.length === 0) return null;

  const attach = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setErr(null);
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append('files', f, f.name));
    try {
      const r = await fetch(`/api/analyses/${id}/originals`, { method: 'POST', body: fd });
      const j = await r.json().catch(() => null);
      if (!r.ok) {
        setErr(j?.message ?? `HTTP ${r.status}`);
      } else if ((j?.missing ?? []).length === 0) {
        toast?.('success', 'Original image attached.');
        window.location.reload();
      } else {
        setMissing(j.missing);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="banner warn" role="alert" data-testid="banner-original-missing">
      <div style={{ flex: 1 }}>
        <strong>This analysis was created on another computer.</strong> Attach the original TIFF(s) to view the image:
        <ul style={{ margin: '2px 0 0 16px', padding: 0 }}>
          {missing.map((m) => (
            <li key={m.file}>
              <code>{m.file}</code> <span className="muted">SHA-256 {m.sha256.slice(0, 12)}…</span>
            </li>
          ))}
        </ul>
        {err && <div className="notice err" role="alert">{err}</div>}
      </div>
      <input ref={input} type="file" multiple accept=".tif,.tiff" hidden onChange={(e) => void attach(e.target.files)} />
      <button className="btn sm primary" disabled={busy} onClick={() => input.current?.click()} data-testid="attach-originals">
        {busy ? 'Checking…' : 'Attach original TIFF(s)…'}
      </button>
    </div>
  );
}
