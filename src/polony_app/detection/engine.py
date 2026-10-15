"""Engine-neutral spot types. Detection engines sit behind ``SpotEngine``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from ..schemas import DetectionParams


@dataclass
class RawSpot:
    """One detected spot in one channel.

    Contract fields first (x=column, y=row, sub-pixel). ``peak``/``background``
    are in background-corrected intensity units; ``snr = (peak - background) /
    robust_std(annulus)`` (None when not measurable), where ``peak`` is the measured
    (unclipped, background-corrected) image interpolated AT the refined spot centre --
    not the maximum of a pixel block, which is biased upward by noise -- and
    ``background`` / the std come from a surrounding annulus (valid pixels only).
    Extra fields are optional measured quantities used for review reasons.
    """
    x: float
    y: float
    peak: float
    background: float
    snr: Optional[float]
    saturated: bool
    # extras (all measured, all optional)
    log_response: Optional[float] = None     # LoG response at the peak (Big-FISH units)
    threshold: Optional[float] = None        # threshold applied to log_response
    elongation: Optional[float] = None       # sqrt(l1/l2) of half-max blob second moments
    scale_ratio: Optional[float] = None      # small-/large-scale normalised LoG ratio; >=1 => compact (debris-like)
    simulated: bool = False                  # added by dense decomposition, not a direct detection
    plateau_area_px: Optional[int] = None    # merged saturated plateau: its area in px (centre = plateau centroid)
    plateau_oversize: bool = False           # merged plateau much larger than a spot, elongated (>1.3) or with >1 LoG maximum => suspicious_shape


@runtime_checkable
class SpotEngine(Protocol):
    name: str

    def version(self) -> str: ...

    def detect(self, img: np.ndarray, params: DetectionParams, valid_mask: Optional[np.ndarray],
               raw: Optional[np.ndarray] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None,
               measure: Optional[np.ndarray] = None,
               threshold_info: Optional[dict] = None) -> list[RawSpot]:
        """img: background-corrected float image. raw: original-dtype image used
        only for the saturation flag. measure: optional unclipped corrected image
        used for peak/background/SNR statistics (defaults to img). Extra args are optional extensions of the
        contract signature. threshold_info: optional result of the engine's
        ``estimate_threshold`` on the FULL image (keeps preview windows identical to the
        full run); engines without this extension may ignore it."""
        ...
