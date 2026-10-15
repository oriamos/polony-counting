"""``python -m polony_app.ml gpu-smoke`` - is a GPU (NVIDIA CUDA or AMD ROCm) usable by PyTorch, before a long training?

    python -m polony_app.ml gpu-smoke [--device auto|cuda|rocm] [--size 512] [--steps 3]

Prints the hardware / software facts that are recorded in a registry entry, then runs on the resolved device: a float32 matmul
with a backward pass and a tiny conv2d forward / backward (the kind of kernels the Spotiflow U-Net uses), checks the results are
finite and (on a GPU) equal to the CPU result within a tolerance. Exit codes: 0 = OK (on a GPU-less machine with ``--device auto``:
"CPU only", the CPU path is OK), 2 = refused / failed, with the precise reason. The timings printed are those of this smoke
test only; they are not a benchmark. ``rocm`` is an alias of ``cuda`` (PyTorch ROCm builds expose AMD GPUs through torch.cuda).
Nothing is written to the data directory.
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from .common import MLError

COMMANDS = ("gpu-smoke",)

# The same check without this package (for the docs / a bare shell):
ONE_LINER = r'''python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, getattr(p, "gcnArchName", ""), p.total_memory)
    a = torch.randn(512, 512, device="cuda", requires_grad=True)
    (a @ a).sum().backward()
    torch.cuda.synchronize()
    print("backward OK, grad finite:", bool(torch.isfinite(a.grad).all()))
PY'''


def register(sub) -> None:
    p = sub.add_parser("gpu-smoke", help="check that PyTorch can use the GPU (CUDA or ROCm): hardware info + matmul/conv backward")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "rocm"],
                   help="auto (default): the GPU if visible, else report 'CPU only' and test the CPU path. cuda / rocm (alias): "
                        "the GPU is required; if it is not visible the precise reason is printed and the exit code is 2")
    p.add_argument("--size", type=int, default=512, help="matrix side of the matmul (default 512)")
    p.add_argument("--steps", type=int, default=3, help="repetitions of each forward + backward (default 3)")


def _p(*a) -> None:
    print(*a, flush=True)


def smoke(device: str, size: int = 512, steps: int = 3, torch_mod=None) -> dict:
    """Run the checks on ``device`` ('cpu' | 'cuda'). Returns {ok, device, stages: [{name, ok, seconds, detail}], reason}.
    Never raises: a failure is reported with the stage and the exception."""
    res = {"ok": False, "device": device, "stages": [], "reason": None}
    try:
        torch = torch_mod
        if torch is None:
            import torch
    except Exception as exc:
        res["reason"] = f"PyTorch cannot be imported ({type(exc).__name__}: {exc})"
        return res

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def stage(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            sync()
            res["stages"].append({"name": name, "ok": True, "seconds": round(time.time() - t0, 4), "detail": detail})
            return True
        except Exception as exc:
            res["stages"].append({"name": name, "ok": False, "seconds": round(time.time() - t0, 4),
                                  "detail": f"{type(exc).__name__}: {exc}"})
            res["reason"] = res["reason"] or f"stage '{name}' failed on {device}: {type(exc).__name__}: {exc}"
            return False

    dev = torch.device(device)
    n = max(8, int(size))
    steps = max(1, int(steps))

    def matmul():
        g = torch.Generator().manual_seed(0)
        a0, b0 = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
        ref = a0.clone().requires_grad_()
        (ref @ b0).relu().sum().backward()
        grad = None
        for _ in range(steps):
            a = a0.clone().to(dev).requires_grad_()
            (a @ b0.to(dev)).relu().sum().backward()
            grad = a.grad
        sync()
        grad = grad.cpu()
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError("non-finite gradient")
        diff = float((grad - ref.grad).abs().max())
        scale = max(1.0, float(ref.grad.abs().max()))
        if diff > 1e-2 * scale:
            raise RuntimeError(f"gradient differs from the CPU result by {diff:.3g} (scale {scale:.3g})")
        return f"{n}x{n} matmul + backward x{steps}; max |grad - cpu grad| = {diff:.2g}"

    def conv():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 1, 3, padding=1))
        side = min(n, 128)
        x0 = torch.randn(2, 1, side, side)
        ref_m = m
        ref_m(x0).square().mean().backward()
        ref_g = [p.grad.clone() for p in ref_m.parameters()]
        m2 = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 1, 3, padding=1))
        m2.load_state_dict(ref_m.state_dict())
        m2 = m2.to(dev)
        for _ in range(steps):
            m2.zero_grad()
            m2(x0.to(dev)).square().mean().backward()
        sync()
        got = [p.grad.cpu() for p in m2.parameters()]
        if not all(bool(torch.isfinite(g).all()) for g in got):
            raise RuntimeError("non-finite conv gradient")
        diff = max(float((a - b).abs().max()) for a, b in zip(got, ref_g))
        scale = max(1.0, max(float(b.abs().max()) for b in ref_g))
        if diff > 1e-2 * scale:
            raise RuntimeError(f"conv gradients differ from the CPU result by {diff:.3g}")
        return f"conv2d(1->8->1, 3x3) on 2x1x{side}x{side} forward + backward x{steps}; max |grad - cpu grad| = {diff:.2g}"

    ok = stage("matmul+backward", matmul)
    ok = stage("conv2d forward+backward", conv) and ok
    res["ok"] = bool(ok)
    return res


def run(args, data_dir: Optional[str] = None) -> int:
    from . import spotiflow_io as sio
    requested = args.device
    try:
        device = sio.pick_device(requested)
    except MLError as exc:
        _p("gpu-smoke: FAILED - the requested GPU is not usable")
        _p(str(exc))
        info = sio.hardware_info("cpu", requested=requested)
        _print_info(info)
        return 2
    info = sio.hardware_info(device, requested=requested)
    _print_info(info)
    if device == "cpu":
        diag = sio.gpu_diagnosis()
        _p(f"CPU only: {diag['message']}")
        for h in diag.get("hints", []):
            _p(f"  check: {h}")
        _p("(running the same checks on the CPU path; this says nothing about a GPU)")
    elif device == "mps":      # pragma: no cover - auto can pick mps on a Mac; the smoke test targets cuda/ROCm
        _p("mps selected: the smoke test below runs on the CPU path only; mps is not exercised")
        device = "cpu"
    r = smoke(device, args.size, args.steps)
    for st in r["stages"]:
        _p(f"  [{'ok' if st['ok'] else 'FAIL'}] {st['name']}: {st['detail']} ({st['seconds']} s, smoke timing - not a benchmark)")
    if r["ok"]:
        if device == "cpu":
            _p("gpu-smoke: OK (CPU only - no GPU was used)")
        else:
            _p(f"gpu-smoke: OK on {info.get('selected_device_name') or 'cuda'}"
               f"{' (' + info['gcn_arch'] + ')' if info.get('gcn_arch') else ''}: matmul and conv2d forward/backward ran and match the CPU")
        return 0
    _p(f"gpu-smoke: FAILED - {r['reason']}")
    return 2


def _print_info(info: dict) -> None:
    _p("hardware:")
    keys = ("platform", "python", "cpu_count", "torch", "torch_hip", "torch_cuda", "cuda_available", "device_count", "device_names",
            "selected_device_name", "gcn_arch", "device_total_memory_bytes", "mps_available", "spotiflow", "lightning",
            "requested_device", "resolved_device")
    for k in keys:
        if k in info:
            _p(f"  {k}: {info[k]}")
    if info.get("gpu_query_error"):
        _p(f"  gpu_query_error: {info['gpu_query_error']}")
