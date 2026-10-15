import { useEffect } from 'react';
import type { ImageView, ObjectClass } from '../api/types';
import { useApp } from '../state/store';

const VIEW_KEYS: Record<string, ImageView> = {
  '1': 'red',
  '2': 'green',
  '3': 'composite',
  '4': 'processed_red',
  '5': 'processed_green',
};
const CLS_KEYS: Record<string, ObjectClass> = { r: 'red', g: 'green', y: 'yellow', u: 'uncertain' };

export function Hotkeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) {
        if (!(e.key === 'Escape')) return;
      }
      const S = useApp.getState();
      if (document.querySelector('.modal-backdrop') && e.key !== '?') return;
      const a = S.current;
      const k = e.key;
      const lower = k.toLowerCase();
      if (k === '?') {
        e.preventDefault();
        S.patch({ helpOpen: !S.helpOpen });
        return;
      }
      // ] = next unreviewed slide of the experiment (on its page, or while one of its slides is open)
      if (k === ']' && !e.ctrlKey && !e.metaKey && !e.altKey && (S.mainView === 'experiment' ? !!S.experiment : !!a?.experiment_id)) {
        e.preventDefault();
        void S.openNextUnreviewed();
        return;
      }
      if (S.mainView === 'experiment') return;
      if (!a) return;
      if ((e.ctrlKey || e.metaKey) && lower === 'z') {
        e.preventDefault();
        void (e.shiftKey ? S.redo() : S.undo());
        return;
      }
      if ((e.ctrlKey || e.metaKey) && lower === 'y') {
        e.preventDefault();
        void S.redo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const locked = a.status === 'finalized';
      // grid-pick tool: A / N / I switch every field of the applied grid (A otherwise = add mode)
      if (S.tool === 'grid-pick' && !locked && (lower === 'a' || lower === 'n' || lower === 'i')) {
        e.preventDefault();
        void S.setGridFields(lower === 'a' ? 'all' : lower === 'n' ? 'none' : 'invert');
        return;
      }
      if (lower === 'h') {
        S.patch({ hideOverlays: !S.hideOverlays });
      } else if (lower === 'f') {
        S.fit();
      } else if (VIEW_KEYS[k]) {
        S.setImageView(VIEW_KEYS[k]);
      } else if (lower === 'v') {
        S.setTool('select');
      } else if (lower === 'a' && !locked) {
        S.setTool(S.tool === 'add' ? 'select' : 'add');
      } else if (lower === 's' && !locked) {
        if (S.selection.length !== 1) S.toast('error', 'Select exactly one spot first, then press S and click two positions');
        else S.setTool('split');
      } else if (CLS_KEYS[lower] && !locked) {
        if (S.selection.length) void S.act('reclassify', { object_ids: S.selection, cls: CLS_KEYS[lower] });
        else if (S.tool === 'add') S.patch({ addClass: CLS_KEYS[lower] });
      } else if (k === 'Delete' && !locked) {
        if (S.selection.length) void S.act('delete', { object_ids: S.selection });
      } else if (lower === 'm' && !locked) {
        if (S.selection.length >= 2) void S.act('merge', { object_ids: S.selection });
        else S.toast('error', 'Select two or more spots (Shift+click) to merge');
      } else if (k === 'Enter' && !locked) {
        if (S.selection.length) {
          e.preventDefault();
          void S.confirmSelected();
        }
      } else if (lower === 'j') {
        S.gotoQueue(1);
      } else if (lower === 'k') {
        S.gotoQueue(-1);
      } else if (k === 'Escape') {
        if (S.tool === 'select') S.setSelection([]);
      } else return;
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return null;
}
