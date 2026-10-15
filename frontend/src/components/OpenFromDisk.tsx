import { useEffect, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { FsListing } from '../api/types';
import { nativePickerAvailable, pickNativeFiles } from '../lib/nativePicker';
import { useApp } from '../state/store';
import { Modal } from './Modal';

const join = (dir: string, name: string) => `${dir.replace(/[\\/]+$/, '')}${dir.includes('\\') && !dir.includes('/') ? '\\' : '/'}${name}`;

/**
 * Open a scan from a folder on this computer (the app runs locally). The scan stays where it is, so Save can write the labelled
 * scan and the CSV files next to it.
 */
export function OpenFromDisk() {
  const [show, setShow] = useState(false);
  return (
    <div className="upload">
      <button className="btn" onClick={() => setShow(true)} data-testid="open-disk-btn">
        Open from disk…
      </button>
      <div className="muted small">Opens the file in place, so Save puts the labelled scan and CSV next to it.</div>
      {show && <OpenDialog onClose={() => setShow(false)} />}
    </div>
  );
}

function OpenDialog({ onClose }: { onClose: () => void }) {
  const open = useApp((s) => s.openAnalysis);
  const refresh = useApp((s) => s.refreshList);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [typed, setTyped] = useState('');
  const [picked, setPicked] = useState<string | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const go = async (path?: string) => {
    setErr('');
    try {
      const l = await api.fsList(path);
      setListing(l);
      setTyped(l.path);
      setPicked(null);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  };
  useEffect(() => {
    void go();
  }, []);

  const openFile = async (path: string) => {
    setBusy(true);
    setErr('');
    try {
      const a = await api.openPath(path);
      await refresh();
      onClose();
      await open(a.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
      setBusy(false);
    }
  };

  const browse = async () => {
    setErr('');
    try {
      const paths = await pickNativeFiles();
      if (paths.length) await openFile(paths[0]);
    } catch (e) {
      setErr(String(e));
    }
  };

  const target = picked && listing ? join(listing.path, picked) : null;
  return (
    <Modal title="Open scan from disk" onClose={onClose}>
      <div className="row">
        <input className="input" value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && void go(typed)} aria-label="Folder" data-testid="disk-path" />
        <button className="btn sm" onClick={() => void go(typed)}>
          Go
        </button>
        {nativePickerAvailable() && (
          <button className="btn sm" onClick={() => void browse()} disabled={busy} data-testid="disk-browse">
            Browse…
          </button>
        )}
      </div>
      {err && (
        <div className="notice err small" role="alert" data-testid="disk-error">
          {err}
        </div>
      )}
      {listing && (
        <ul className="opendisk-list" data-testid="disk-list">
          {listing.parent && (
            <li onClick={() => void go(listing.parent!)} data-testid="disk-up">
              ⬆ ..
            </li>
          )}
          {listing.dirs.map((d) => (
            <li key={'d' + d} onClick={() => void go(join(listing.path, d))}>
              📁 {d}
            </li>
          ))}
          {listing.files.map((f) => (
            <li key={'f' + f} className={picked === f ? 'sel' : ''} onClick={() => setPicked(f)} onDoubleClick={() => void openFile(join(listing.path, f))} data-testid={`disk-file-${f}`}>
              🖼 {f}
            </li>
          ))}
          {listing.dirs.length + listing.files.length === 0 && <li className="muted">No folders or TIFF files here.</li>}
        </ul>
      )}
      <div className="row between">
        <span className="small muted">{target ?? 'Pick a .tif file (double-click opens it).'}</span>
        <button className="btn primary" disabled={!target || busy} onClick={() => target && void openFile(target)} data-testid="disk-open">
          {busy ? 'Opening…' : 'Open'}
        </button>
      </div>
    </Modal>
  );
}
