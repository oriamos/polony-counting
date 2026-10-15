import { describe, expect, it } from 'vitest';
import { batchRequest } from './batch';

describe('batchRequest', () => {
  it('sends Spotiflow settings only when it is used', () => {
    expect(batchRequest('bigfish').spotiflow).toBeUndefined();
    expect(batchRequest('bigfish').preview_window).toBeNull();
    const h = batchRequest('hybrid');
    expect(h.engine).toBe('hybrid');
    expect(h.spotiflow?.downscale).toBe(4);
    expect(h.spotiflow?.normalizer).toBe('valid_unsaturated');
  });
});
