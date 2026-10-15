import { describe, expect, it } from 'vitest';
import { cleanPickedPaths, nativePickerAvailable } from './nativePicker';

describe('nativePicker', () => {
  it('is available only when the js_api exposes pick_files', () => {
    expect(nativePickerAvailable(undefined)).toBe(false);
    expect(nativePickerAvailable({})).toBe(false);
    expect(nativePickerAvailable({ pywebview: { api: {} } })).toBe(false);
    expect(nativePickerAvailable({ pywebview: { api: { pick_files: async () => [] } } })).toBe(true);
  });
  it('keeps only non-empty string paths', () => {
    expect(cleanPickedPaths(['C:\\a.tif', '', 3, null, '/b.tif'])).toEqual(['C:\\a.tif', '/b.tif']);
    expect(cleanPickedPaths(null)).toEqual([]);
    expect(cleanPickedPaths('x')).toEqual([]);
  });
});
