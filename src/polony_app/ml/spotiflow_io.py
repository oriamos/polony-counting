"""The only place of the ML package that imports spotiflow / torch (always lazily)."""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

from .common import MLError
from .registry import ModelRef, resolve_model

# used only when spotiflow cannot be imported (dry runs on machines without it)
_KNOWN_PRETRAINED = ("general", "hybiss", "synth_complex", "synth_3d", "smfish_3d", "tissuenet", "fluo_live")


def model_cache_dir() -> Path:
    env = os.environ.get("POLONY_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    from polony_app.config import home_dir
    return home_dir() / "models" / "spotiflow"


def import_spotiflow():
    try:
        from spotiflow.model import Spotiflow
        return Spotiflow
    except Exception as exc:
        raise MLError("Spotiflow / PyTorch cannot be imported in this environment "
                      f"({type(exc).__name__}: {exc}). Install the optional ML dependencies (spotiflow, torch) first.")


def known_pretrained() -> tuple[tuple, bool]:
    """(names, from_installed_spotiflow)"""
    try:
        from spotiflow.model.pretrained import list_registered
        return tuple(list_registered()), True
    except Exception:
        return _KNOWN_PRETRAINED, False


def describe_base(base: str, data_dir: str | Path) -> dict:
    """Resolve a base-model argument WITHOUT loading weights. Returns {kind, name, path, registry_id, ...}."""
    p = Path(base).expanduser()
    if p.is_dir():
        if not (p / "config.yaml").is_file():
            raise MLError(f"base model folder {p} has no config.yaml")
        return {"kind": "folder", "name": p.name, "path": str(p.resolve()), "registry_id": None}
    ref = resolve_model(base, data_dir)
    if ref.kind == "finetuned":
        return {"kind": "registry", "name": ref.name, "path": str(ref.path), "registry_id": ref.registry_id,
                "weights_sha256": ref.weights_sha256}
    names, installed = known_pretrained()
    if ref.name not in names:
        raise MLError(f"unknown base model '{base}': not a registry id, not a folder and not an official pretrained name "
                      f"({', '.join(names)})")
    return {"kind": "pretrained", "name": ref.name, "path": None, "registry_id": None}


def load_model(base: str, data_dir: str | Path, *, inference_mode: bool, device: str = "auto"):
    """Spotiflow instance. ``inference_mode=False`` keeps the model trainable (fine-tuning continues from its weights)."""
    Spotiflow = import_spotiflow()
    d = describe_base(base, data_dir)
    if d["kind"] in ("folder", "registry"):
        return Spotiflow.from_folder(d["path"], inference_mode=inference_mode, map_location=device)
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return Spotiflow.from_pretrained(d["name"], inference_mode=inference_mode, map_location=device, cache_dir=cache)


def hardware_info(device: str = "auto") -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "processor": platform.processor() or platform.machine()}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
                     "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())})
        if info["cuda_available"]:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import spotiflow
        info["spotiflow"] = getattr(spotiflow, "__version__", None)
    except Exception:
        info["spotiflow"] = None
    info["requested_device"] = device
    return info


def pick_device(requested: str = "cpu") -> str:
    """CPU by default; 'auto' uses CUDA/MPS when present."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
