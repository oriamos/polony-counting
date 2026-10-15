"""Optional AI detector (Spotiflow + PyTorch) for the installers. Standard library only (config flag excepted).

    ai_extra.py describe [--cuda yes|no|auto]   print what would be installed and how big it is
    ai_extra.py install  [--cuda yes|no|auto]   pip install matching torch+torchvision, then spotiflow; then check
    ai_extra.py check                           import torch, torchvision, spotiflow (+ a torchvision op) and report
    ai_extra.py flag on|off                     record "ai_enabled" in config.json (via polony_app.config)

Run it with the venv's python so `pip` targets the venv. Exit codes: 0 ok, 1 failed (Big-FISH still works),
4 platform without a supported PyTorch build.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TORCH = "2.14.1"
TORCHVISION = "0.29.1"          # the release that matches torch 2.14.1 (0.29.0 conflicts with it)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
CHECK_CODE = (
    "import torch, torchvision, spotiflow; from spotiflow.model import Spotiflow; "
    "from torchvision.ops import nms; nms(torch.zeros((0, 4)), torch.zeros(0), 0.5); "
    "d = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and "
    "torch.backends.mps.is_available() else 'cpu'); "
    "print('spotiflow %s, torch %s, torchvision %s, device %s' % (spotiflow.__version__, torch.__version__, "
    "torchvision.__version__, d))"
)


def detect_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def plan(system: str, machine: str, nvidia: bool, cuda: str = "auto") -> dict:
    """Which torch build to install. cuda: 'yes' | 'no' | 'auto' (auto = CUDA only when an NVIDIA GPU exists on Windows/Linux x86_64)."""
    m = machine.lower()
    pins = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    if system == "Darwin":
        if m in ("arm64", "aarch64"):
            return dict(supported=True, index_url=None, pins=pins, device="mps", size_gb="0.8",
                        label="PyTorch from PyPI (uses the Apple GPU via MPS when available)")
        return dict(supported=False, index_url=None, pins=[], device="cpu", size_gb="0",
                    label="Intel Macs: current PyTorch releases are not published for this platform")
    if system in ("Linux", "Windows"):
        if m not in ("x86_64", "amd64"):
            if system == "Linux":  # e.g. aarch64: PyPI wheels are CPU builds
                return dict(supported=True, index_url=None, pins=pins, device="cpu", size_gb="0.8",
                            label="PyTorch from PyPI (CPU)")
            return dict(supported=False, index_url=None, pins=[], device="cpu", size_gb="0",
                        label="Windows on ARM: no PyTorch build available")
        if nvidia and cuda != "no" and cuda in ("yes", "auto"):
            return dict(supported=True, index_url=CUDA_INDEX, pins=pins, device="cuda", size_gb="2.5",
                        label="PyTorch CUDA 12.6 build (NVIDIA GPU found)")
        return dict(supported=True, index_url=CPU_INDEX, pins=pins, device="cpu", size_gb="0.8",
                    label="PyTorch CPU build (no GPU needed)")
    return dict(supported=False, index_url=None, pins=[], device="cpu", size_gb="0", label=f"unsupported system {system}")


def pip_commands(p: dict) -> list[list[str]]:
    """The pip argument lists (without the interpreter) to run, in order."""
    first = ["install", "--disable-pip-version-check", *p["pins"]]
    if p["index_url"]:
        first += ["--index-url", p["index_url"]]
    second = ["install", "--disable-pip-version-check", "-r", str(HERE / "extra-ai-requirements.txt")]
    return [first, second]


def _current_plan(cuda: str) -> dict:
    return plan(platform.system(), platform.machine(), detect_nvidia(), cuda)


def cmd_describe(cuda: str) -> int:
    p = _current_plan(cuda)
    if not p["supported"]:
        print(f"AI detector not available here: {p['label']}. Big-FISH detection works without it.")
        return 4
    print(f"AI detector: {p['label']}; Spotiflow {_spotiflow_pin()}; download about {p['size_gb']} GB")
    return 0


def _spotiflow_pin() -> str:
    for line in (HERE / "extra-ai-requirements.txt").read_text().splitlines():
        if line.strip().startswith("spotiflow"):
            return line.strip().split("==")[-1]
    return "?"


def mps_hint(system: str, machine: str, check_output: str) -> str:
    """Hint for Apple Silicon when torch imports but the check reports device cpu (no MPS); '' otherwise."""
    if system == "Darwin" and machine == "arm64" and check_output.strip().endswith("device cpu"):
        return ("Apple GPU (MPS) is not available: it needs macOS 12.3 or newer and the arm64 build of Python. "
                "The AI detector will run on the CPU (slower, same results).")
    return ""


def cmd_check() -> int:
    r = subprocess.run([sys.executable, "-c", CHECK_CODE], capture_output=True, text=True)
    if r.returncode == 0:
        print(r.stdout.strip())
        hint = mps_hint(platform.system(), platform.machine(), r.stdout)
        if hint:
            print(hint)
        return 0
    err = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
    print(f"AI detector check FAILED: {err}")
    print("Big-FISH detection still works. You can retry with the installer or ignore the AI detector.")
    return 1


def cmd_flag(on: bool) -> int:
    from polony_app.config import load_config_file, save_config_file

    cfg = load_config_file()
    cfg["ai_enabled"] = bool(on)
    save_config_file(cfg)
    return 0


def cmd_install(cuda: str, dry: bool = False) -> int:
    p = _current_plan(cuda)
    if not p["supported"]:
        print(f"AI detector not available here: {p['label']}. Big-FISH detection works without it.")
        return 4
    print(f"Installing the AI detector: {p['label']} (download about {p['size_gb']} GB) ...", flush=True)
    for args in pip_commands(p):
        cmd = [sys.executable, "-m", "pip", *args]
        if dry:
            print("  [dry-run] " + " ".join(cmd))
            continue
        if subprocess.run(cmd).returncode != 0:
            print("Installing the AI detector failed (see messages above). Big-FISH detection still works.")
            return 1
    if dry:
        return 0
    rc = cmd_check()
    if rc == 0:
        cmd_flag(True)
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["describe", "install", "check", "flag"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--cuda", default="auto", choices=["yes", "no", "auto"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "describe":
        return cmd_describe(a.cuda)
    if a.cmd == "install":
        return cmd_install(a.cuda, a.dry_run)
    if a.cmd == "check":
        return cmd_check()
    return cmd_flag(a.arg == "on")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
