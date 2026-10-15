"""Optional AI detector (Spotiflow + PyTorch) in the installers: platform/index selection, pins, config flag,
dry-run output, uninstall keeping downloaded models."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "install"
BASH = shutil.which("bash")

spec = importlib.util.spec_from_file_location("ai_extra", INSTALL / "ai_extra.py")
ai = importlib.util.module_from_spec(spec)
sys.modules["ai_extra"] = ai
spec.loader.exec_module(ai)


def test_linux_windows_cpu_uses_cpu_index_with_matching_pair():
    for system in ("Linux", "Windows"):
        p = ai.plan(system, "x86_64" if system == "Linux" else "AMD64", nvidia=False)
        assert p["supported"] and p["index_url"] == ai.CPU_INDEX and p["device"] == "cpu"
        assert p["pins"] == [f"torch=={ai.TORCH}", f"torchvision=={ai.TORCHVISION}"]
        first, second = ai.pip_commands(p)
        assert first[-2:] == ["--index-url", ai.CPU_INDEX] and "--index-url" not in second  # spotiflow from PyPI
        assert second[-1].endswith("extra-ai-requirements.txt")


def test_cuda_only_with_gpu_and_not_when_declined():
    assert ai.plan("Linux", "x86_64", True, "auto")["index_url"] == ai.CUDA_INDEX
    assert ai.plan("Windows", "AMD64", True, "yes")["device"] == "cuda"
    assert ai.plan("Linux", "x86_64", True, "no")["index_url"] == ai.CPU_INDEX
    assert ai.plan("Linux", "x86_64", False, "yes")["index_url"] == ai.CPU_INDEX  # no GPU: CPU build whatever is asked


def test_macos_pypi_for_apple_silicon_and_unsupported_intel():
    p = ai.plan("Darwin", "arm64", False)
    assert p["supported"] and p["index_url"] is None and p["device"] == "mps"
    assert not ai.plan("Darwin", "x86_64", False)["supported"]
    assert not ai.plan("Windows", "ARM64", False)["supported"]
    assert ai.plan("Linux", "aarch64", False)["index_url"] is None


def test_pins_are_a_matching_pair_and_spotiflow_pinned():
    # torch 2.N pairs with torchvision 0.(N+15); 0.29.0 conflicts with torch 2.14.1 (the old broken venv)
    t, v = (int(x) for x in ai.TORCH.split(".")[:2]), (int(x) for x in ai.TORCHVISION.split(".")[:2])
    tmaj, tmin = t
    vmaj, vmin = v
    assert (tmaj, vmaj) == (2, 0) and vmin == tmin + 15
    req = (INSTALL / "extra-ai-requirements.txt").read_text()
    assert re.search(r"^spotiflow==\d+\.\d+\.\d+$", req, re.M)
    assert "torch" not in "".join(l for l in req.splitlines() if not l.startswith("#"))  # torch comes from the index step


def test_check_code_exercises_a_torchvision_op():
    # the 'torchvision::nms does not exist' failure only shows when the op is called
    assert "nms" in ai.CHECK_CODE and "from spotiflow.model import Spotiflow" in ai.CHECK_CODE


def test_ai_enabled_flag_in_config_keeps_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path))
    monkeypatch.delenv("POLONY_CONFIG", raising=False)
    sys.path.insert(0, str(ROOT / "src"))
    (tmp_path / "config.json").write_text(json.dumps({"port": 8765, "database_url": "postgresql://u:p@h/db"}))
    assert ai.cmd_flag(True) == 0
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["ai_enabled"] is True and cfg["port"] == 8765 and cfg["database_url"].startswith("postgresql://")
    if os.name != "nt":
        assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600
    ai.cmd_flag(False)
    assert json.loads((tmp_path / "config.json").read_text())["ai_enabled"] is False


def test_install_dry_run_prints_planned_pip_commands(capsys):
    assert ai.cmd_install("no", dry=True) in (0, 4)
    out = capsys.readouterr().out
    if "not available" not in out:
        assert "[dry-run]" in out and f"torch=={ai.TORCH}" in out and "extra-ai-requirements.txt" in out


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="bash not available")
def test_installer_dry_run_lists_ai_step_and_flag(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "POLONY_CI": "1"}
    for k in ("POLONY_HOME", "POLONY_SOURCE", "POLONY_WITH_AI"):
        env.pop(k, None)
    r = subprocess.run([BASH, str(INSTALL / "install.sh"), "--dry-run"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "AI detector (optional)" in r.stdout and "default no" in r.stdout
    r = subprocess.run([BASH, str(INSTALL / "install.sh"), "--dry-run", "--with-ai"], env=env, capture_output=True, text=True)
    assert "install the AI detector: matching torch+torchvision" in r.stdout
    assert list(tmp_path.iterdir()) == []
    ps = (INSTALL / "install.ps1").read_text()
    assert "WithAi" in ps and "POLONY_WITH_AI" in ps and "download.pytorch.org" in ps + (INSTALL / "ai_extra.py").read_text()


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="bash not available")
def test_uninstall_keeps_models_unless_purged(tmp_path):
    def fresh():
        ph = tmp_path / "PolonyReview"
        shutil.rmtree(ph, ignore_errors=True)
        (ph / "app").mkdir(parents=True)
        (ph / "venv").mkdir()
        (ph / "models" / "spotiflow" / "general").mkdir(parents=True)
        (ph / "models" / "spotiflow" / "general" / "best.pt").write_bytes(b"x")
        return ph

    env = {**os.environ, "HOME": str(tmp_path), "POLONY_CI": "1"}
    env.pop("POLONY_HOME", None)
    ph = fresh()
    subprocess.run([BASH, str(INSTALL / "uninstall.sh")], env=env, check=True, capture_output=True)
    assert sorted(p.name for p in ph.iterdir()) == ["models"] and (ph / "models/spotiflow/general/best.pt").exists()
    ph = fresh()
    subprocess.run([BASH, str(INSTALL / "uninstall.sh"), "--purge-data"], env=env, check=True, capture_output=True)
    assert not ph.exists()


def test_launcher_has_prepare_model_and_ai_status_line():
    from polony_app import launcher

    line = launcher._ai_line()
    assert line.startswith("AI detector: ")
    assert "prepare-model" in launcher.main.__code__.co_consts or hasattr(launcher, "cmd_prepare_model")


def test_mps_hint_only_for_apple_silicon_on_cpu():
    out = "spotiflow 0.6.5, torch 2.14.1, torchvision 0.29.1, device "
    assert "macOS 12.3" in ai.mps_hint("Darwin", "arm64", out + "cpu")
    assert ai.mps_hint("Darwin", "arm64", out + "mps") == ""
    assert ai.mps_hint("Linux", "x86_64", out + "cpu") == ""
    assert ai.mps_hint("Darwin", "x86_64", out + "cpu") == ""


def test_install_sh_clears_quarantine_on_venv_and_runs_check():
    sh = (INSTALL / "install.sh").read_text()
    assert 'xattr -dr com.apple.quarantine "$VENV"' in sh and "command -v xattr" in sh
    assert 'ai_extra.py" check' in sh
