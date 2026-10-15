import type { DetectionRequest, EngineChoice } from '../api/types';
import { DEFAULT_SPOTIFLOW } from '../api/types';

/**
 * Default detection settings of an unattended experiment run (New experiment dialog). Everything not listed keeps the
 * server default; Spotiflow (hybrid / spotiflow) runs at 1/4 scale with the saturation-robust normalisation, which
 * suits the ~25 px blobs of the lab scans.
 */
export function batchRequest(engine: EngineChoice): DetectionRequest {
  const req: Partial<DetectionRequest> = { engine, preview_window: null };
  if (engine !== 'bigfish') req.spotiflow = { ...DEFAULT_SPOTIFLOW, downscale: 4, normalizer: 'valid_unsaturated' };
  return req as DetectionRequest;
}
