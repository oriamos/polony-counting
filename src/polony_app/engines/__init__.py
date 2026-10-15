"""Detection engines (Big-FISH, Spotiflow) behind one contract. Importing this package never imports torch."""
from .base import (DetectionEngine, EngineError, EngineResult, EngineUnavailable, build_detections, new_run_id,
                   raster_point_validity, raw_id)
from .bigfish_adapter import BigFishAdapter
from .spotiflow_adapter import ModelInfo, SpotiflowEngine, device_info, ensure_model, list_models, model_cache_dir

__all__ = ["DetectionEngine", "EngineError", "EngineResult", "EngineUnavailable", "BigFishAdapter",
           "SpotiflowEngine", "ModelInfo", "ensure_model", "device_info", "list_models", "model_cache_dir",
           "build_detections", "new_run_id", "raster_point_validity", "raw_id"]
