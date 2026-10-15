/** Native file picker exposed by the desktop window (pywebview js_api, see polony_app/launcher.py). */
export interface PywebviewApi {
  pick_files: () => Promise<unknown>;
}

declare global {
  interface Window {
    pywebview?: { api?: Partial<PywebviewApi> };
  }
}

type MaybeWindow = { pywebview?: { api?: Partial<PywebviewApi> } } | undefined;

/** True only inside the native window, where `pick_files` exists. */
export function nativePickerAvailable(w: MaybeWindow = typeof window === 'undefined' ? undefined : window): boolean {
  return typeof w?.pywebview?.api?.pick_files === 'function';
}

/** Keep only non-empty string paths from whatever the Python side returned. */
export function cleanPickedPaths(raw: unknown): string[] {
  return Array.isArray(raw) ? raw.filter((p): p is string => typeof p === 'string' && p.length > 0) : [];
}

export async function pickNativeFiles(): Promise<string[]> {
  const api = window.pywebview?.api;
  return api && typeof api.pick_files === 'function' ? cleanPickedPaths(await api.pick_files()) : [];
}
