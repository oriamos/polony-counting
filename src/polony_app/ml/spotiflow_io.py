"""The only place of the ML package that imports spotiflow / torch (always lazily)."""
from __future__ import annotations

import os
import platform
import sys
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


DEVICE_CHOICES = ("auto", "cpu", "cuda", "rocm", "mps")


def _torch_module(torch_mod=None):
    if torch_mod is not None:
        return torch_mod
    try:
        import torch
        return torch
    except Exception:
        return None


def _gpu_available(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available(torch) -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _linux_access_notes() -> list[str]:
    """Facts about /dev/kfd, the render / video groups and *_VISIBLE_DEVICES, as things to CHECK (never as a fix)."""
    notes: list[str] = []
    if not sys.platform.startswith("linux"):
        return notes
    kfd = "/dev/kfd"
    if not os.path.exists(kfd):
        notes.append(f"{kfd} does not exist (no amdgpu / KFD driver loaded, or a container started without --device={kfd})")
    elif not os.access(kfd, os.R_OK | os.W_OK):
        notes.append(f"{kfd} exists but this user cannot read+write it (check membership of the 'render' / 'video' groups; "
                     "a new login is needed after changing groups)")
    try:
        import glob
        import grp
        groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
        if not groups & {"render", "video"}:
            notes.append("this user is in neither the 'render' nor the 'video' group")
        nodes = glob.glob("/dev/dri/renderD*")
        if not nodes:
            notes.append("there is no /dev/dri/renderD* node")
        elif not any(os.access(n, os.R_OK | os.W_OK) for n in nodes):
            notes.append("no /dev/dri/renderD* node is readable+writable by this user")
    except Exception:
        pass
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if var in os.environ:
            notes.append(f"{var}={os.environ[var]!r} is set and may hide the GPU")
    return notes


def gpu_diagnosis(torch_mod=None) -> dict:
    """Why a GPU is (not) usable. ``kind``: 'ok' | 'no_torch' | 'cpu_only_build' | 'hip_no_device' | 'cuda_no_device'.
    The hints are things to CHECK; none of them is claimed to fix anything."""
    torch = _torch_module(torch_mod)
    if torch is None:
        return {"kind": "no_torch", "message": "PyTorch cannot be imported in this environment", "hints": []}
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    avail = _gpu_available(torch)
    if avail:
        return {"kind": "ok", "message": "a GPU is visible to PyTorch", "hints": [], "hip": hip, "cuda": cuda}
    if hip:
        hints = _linux_access_notes() + [
            "check the output of `rocminfo` / `rocm-smi` in the same environment (does the OS see the GPU at all?)",
            "check that this PyTorch ROCm wheel supports the GPU's gfx target (`gcnArchName`); some setups use "
            "HSA_OVERRIDE_GFX_VERSION as a workaround - that is only something to look into, nothing here claims it works"]
        return {"kind": "hip_no_device", "hip": hip, "cuda": cuda, "hints": hints,
                "message": f"this PyTorch is a ROCm/HIP build (torch.version.hip={hip}) but torch.cuda.is_available() is False: "
                           "no AMD GPU is visible to it"}
    if cuda:
        return {"kind": "cuda_no_device", "hip": hip, "cuda": cuda, "hints": [
            "check `nvidia-smi` (driver installed and the GPU listed?)", "check that the driver supports this CUDA runtime version"] + [
            n for n in _linux_access_notes() if "VISIBLE_DEVICES" in n],
                "message": f"this PyTorch is a CUDA build (torch.version.cuda={cuda}) but torch.cuda.is_available() is False: "
                           "no NVIDIA GPU is visible to it"}
    return {"kind": "cpu_only_build", "hip": None, "cuda": None, "hints": [
        "install a ROCm (AMD) or CUDA (NVIDIA) build of PyTorch to use a GPU; `pip show torch` / torch.__version__ shows the build"],
            "message": "this PyTorch is a CPU-only build (torch.version.hip and torch.version.cuda are both None)"}


def format_gpu_failure(requested: str, diag: dict) -> str:
    extra = ""
    if requested == "rocm":
        extra = " ('rocm' is an alias of 'cuda': PyTorch's ROCm builds expose AMD GPUs through the torch.cuda API)"
    hints = "".join(f"\n  - check: {h}" for h in diag.get("hints", []))
    return (f"--device {requested}{extra} was requested but no GPU is available: {diag['message']}. Nothing was started "
            f"(there is no silent fallback to the CPU; use --device cpu, or --device auto to allow it).{hints}")


def pick_device(requested: str = "cpu", torch_mod=None, say=None) -> str:
    """Resolve a --device request to what Spotiflow / Lightning accept: 'cpu', 'cuda' or 'mps'.

    * ``rocm`` is an alias of ``cuda`` (PyTorch ROCm builds expose AMD GPUs through ``torch.cuda``; ``torch.version.hip`` is set).
    * an EXPLICIT gpu request (cuda / rocm / mps) that cannot be honoured is an :class:`MLError` - never a silent CPU run.
    * ``auto`` = cuda (ROCm included) if ``torch.cuda.is_available()``, else mps, else cpu; falling back to the CPU is announced
      through ``say`` when this PyTorch is a GPU build that sees no device."""
    req = str(requested or "cpu").lower()
    if req not in DEVICE_CHOICES:
        raise MLError(f"unknown --device '{requested}' (choose one of {', '.join(DEVICE_CHOICES)})")
    if req == "cpu":
        return "cpu"
    torch = _torch_module(torch_mod)
    if req in ("cuda", "rocm"):
        if torch is not None and _gpu_available(torch):
            return "cuda"
        raise MLError(format_gpu_failure(req, gpu_diagnosis(torch_mod)))
    if req == "mps":
        if torch is not None and _mps_available(torch):
            return "mps"
        raise MLError("--device mps was requested but torch.backends.mps is not available here. Nothing was trained "
                      "(no silent fallback to the CPU).")
    # auto
    if torch is not None:
        if _gpu_available(torch):
            return "cuda"
        if _mps_available(torch):
            return "mps"
        if say is not None:
            diag = gpu_diagnosis(torch_mod)
            if diag["kind"] in ("hip_no_device", "cuda_no_device"):
                say(f"NOTICE: --device auto uses the CPU: {diag['message']}")
                for h in diag["hints"]:
                    say(f"  check: {h}")
            else:
                say(f"NOTICE: --device auto uses the CPU: {diag['message']}")
    elif say is not None:
        say("NOTICE: --device auto uses the CPU: PyTorch cannot be imported here")
    return "cpu"


def _selected_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def hardware_info(device: str = "auto", requested: Optional[str] = None, torch_mod=None) -> dict:
    """Hardware / software facts recorded in the registry entry. ``device`` = the RESOLVED device ('cpu' | 'cuda' | 'mps');
    ``requested`` = what the user asked for (--device value)."""
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "processor": platform.processor() or platform.machine()}
    torch = _torch_module(torch_mod)
    if torch is None:
        info.update({"torch": None, "torch_hip": None, "torch_cuda": None, "cuda_available": False})
    else:
        try:
            info.update({"torch": torch.__version__, "torch_hip": getattr(torch.version, "hip", None),
                         "torch_cuda": getattr(torch.version, "cuda", None), "cuda_available": _gpu_available(torch),
                         "mps_available": _mps_available(torch)})
            if info["cuda_available"]:
                n = int(torch.cuda.device_count())
                info["device_count"] = n
                info["device_names"] = [str(torch.cuda.get_device_name(i)) for i in range(n)]
                info["cuda_device"] = info["device_names"][0] if info["device_names"] else None
                if device.startswith("cuda"):
                    idx = _selected_index(device)
                    props = torch.cuda.get_device_properties(idx)
                    info["selected_device_index"] = idx
                    info["selected_device_name"] = str(torch.cuda.get_device_name(idx))
                    info["device_total_memory_bytes"] = int(getattr(props, "total_memory", 0)) or None
                    arch = getattr(props, "gcnArchName", None)
                    if arch:
                        info["gcn_arch"] = str(arch)
        except Exception as exc:        # never let bookkeeping break a run
            info["gpu_query_error"] = f"{type(exc).__name__}: {exc}"
    for key, mods in (("spotiflow", ("spotiflow",)), ("lightning", ("lightning", "pytorch_lightning"))):
        info[key] = None
        for m in mods:
            try:
                mod = __import__(m)
                info[key] = getattr(mod, "__version__", None)
                break
            except Exception:
                continue
    info["requested_device"] = requested if requested is not None else device
    info["resolved_device"] = device
    return info
