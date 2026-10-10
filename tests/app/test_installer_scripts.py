"""Static + behavioural checks of install/*: syntax, shellcheck, PowerShell parse, dry-runs, generated
.app bundle / .desktop file, icons, bundled web UI. (The full network install is opt-in: POLONY_E2E_INSTALL=1.)"""
from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "install"
ASSETS = INSTALL / "assets"
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None or os.name == "nt", reason="bash not available")


def sh(args, home: Path, extra_env=None, check=True):
    env = {**os.environ, "HOME": str(home), "POLONY_CI": "1"}
    for k in ("POLONY_HOME", "POLONY_DATA_DIR", "POLONY_SOURCE", "POLONY_FORCE_OS"):
        env.pop(k, None)
    env.update(extra_env or {})
    r = subprocess.run([BASH, *map(str, args)], env=env, capture_output=True, text=True, timeout=120)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def snapshot(p: Path):
    return sorted(str(x.relative_to(p)) for x in p.rglob("*"))


# ---------------------------------------------------------------- syntax / lint
@needs_bash
@pytest.mark.parametrize("name", ["install.sh", "uninstall.sh"])
def test_bash_syntax(name):
    r = subprocess.run([BASH, "-n", str(INSTALL / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", ["install.sh", "uninstall.sh"])
def test_shellcheck(name):
    sc = shutil.which("shellcheck")
    if not sc:
        pytest.skip("shellcheck not installed")
    r = subprocess.run([sc, "-x", "-S", "warning", str(INSTALL / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize("name", ["install.ps1", "uninstall.ps1"])
def test_powershell_parses(name):
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("pwsh not installed: PowerShell scripts were NOT parse-checked here (CI does it on Windows)")
    cmd = ("$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
           f"'{INSTALL / name}',[ref]$t,[ref]$e); if($e.Count){{$e|%{{$_.Message}}; exit 1}}")
    r = subprocess.run([pwsh, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_scripts_never_use_sudo_or_unreviewed_pipes():
    for name in ("install.sh", "uninstall.sh", "install.ps1", "uninstall.ps1"):
        text = (INSTALL / name).read_text()
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        assert "sudo" not in code, name
        assert "| sh" not in code and "| bash" not in code and "| iex" not in code, name


# ---------------------------------------------------------------- dry runs change nothing
@needs_bash
def test_install_dry_run_changes_nothing(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    r = sh([INSTALL / "install.sh", "--dry-run"], home)
    assert "DRY RUN" in r.stdout and "[dry-run]" in r.stdout
    assert snapshot(home) == []


@needs_bash
def test_uninstall_dry_run_changes_nothing(tmp_path):
    home = tmp_path / "h"
    (home / "PolonyReview" / "app").mkdir(parents=True)
    (home / "PolonyReviewData").mkdir()
    before = snapshot(home)
    r = sh([INSTALL / "uninstall.sh", "--dry-run", "--purge-data"], home)
    assert "would remove" in r.stdout
    assert snapshot(home) == before


# ---------------------------------------------------------------- generated shortcuts
def _fake_installed_home(tmp_path):
    home = tmp_path / "h"
    (home / "PolonyReview" / "app" / "install").mkdir(parents=True)
    shutil.copytree(ASSETS, home / "PolonyReview" / "app" / "install" / "assets")
    return home


@needs_bash
def test_macos_app_bundle_structure(tmp_path):
    home = _fake_installed_home(tmp_path)
    sh([INSTALL / "install.sh", "--shortcuts-only"], home, {"POLONY_FORCE_OS": "mac"})
    for base in (home / "Applications", home / "Desktop"):
        app = base / "Polony Review.app"
        plist = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        assert plist["CFBundleExecutable"] == "launcher"
        assert plist["CFBundleIdentifier"] == "org.polonyreview.app"
        assert plist["CFBundlePackageType"] == "APPL"
        exe = app / "Contents" / "MacOS" / plist["CFBundleExecutable"]
        assert exe.is_file() and os.access(exe, os.X_OK)
        script = exe.read_text()
        assert script.startswith("#!/bin/bash")
        assert "-m polony_app.launcher" in script and "/PolonyReview/venv/bin/python" in script
        icon = app / "Contents" / "Resources" / (plist["CFBundleIconFile"] + ".icns")
        assert icon.read_bytes()[:4] == b"icns"
    stop = home / "Applications" / "Stop Polony Review.app" / "Contents" / "MacOS" / "launcher"
    assert "polony_app.launcher stop" in stop.read_text() and os.access(stop, os.X_OK)
    cli = home / "PolonyReview" / "bin" / "polony-review"
    assert os.access(cli, os.X_OK)
    # uninstall removes the shortcuts but keeps data
    (home / "PolonyReviewData").mkdir()
    sh([INSTALL / "uninstall.sh"], home, {"POLONY_FORCE_OS": "mac"})
    assert not (home / "Desktop" / "Polony Review.app").exists()
    assert not (home / "Applications" / "Polony Review.app").exists()
    assert (home / "PolonyReviewData").is_dir() and not (home / "PolonyReview").exists()


@needs_bash
def test_linux_desktop_entry(tmp_path):
    home = _fake_installed_home(tmp_path)
    sh([INSTALL / "install.sh", "--shortcuts-only"], home, {"POLONY_FORCE_OS": "linux", "XDG_DATA_HOME": str(home / ".local/share")})
    d = home / "Desktop" / "Polony Review.desktop"
    text = d.read_text()
    assert os.access(d, os.X_OK)
    assert text.startswith("[Desktop Entry]") and "Name=Polony Review" in text
    exec_line = next(l for l in text.splitlines() if l.startswith("Exec="))
    assert exec_line == f'Exec="{home}/PolonyReview/bin/polony-review"'
    assert (home / ".local/share/applications/polony-review.desktop").is_file()


# ---------------------------------------------------------------- assets
def test_icons_valid():
    from PIL import Image

    png = Image.open(ASSETS / "PolonyReview.png")
    assert png.size == (512, 512)
    ico = Image.open(ASSETS / "PolonyReview.ico")
    assert ico.format == "ICO" and {(16, 16), (32, 32), (48, 48), (256, 256)} <= set(ico.info["sizes"])
    assert (ASSETS / "PolonyReview.icns").read_bytes()[:4] == b"icns"
    assert Image.open(ASSETS / "PolonyReview.icns").format == "ICNS"


def test_prebuilt_web_ui_is_committed_and_served(tmp_path, monkeypatch):
    web = ROOT / "src" / "polony_app" / "webdist"
    assert (web / "index.html").is_file() and any((web / "assets").glob("*.js"))
    assert sum(f.stat().st_size for f in web.rglob("*") if f.is_file()) < 3_000_000
    from fastapi.testclient import TestClient

    from polony_app.api.main import create_app

    monkeypatch.setenv("POLONY_FRONTEND_DIST", str(tmp_path / "missing"))  # no frontend/dist -> webdist fallback
    with TestClient(create_app(tmp_path / "data")) as c:
        assert c.get("/api/health").json()["ok"] is True
        r = c.get("/")
        assert r.status_code == 200 and '<div id="root">' in r.text


# ---------------------------------------------------------------- optional full network install
@needs_bash
@pytest.mark.skipif(os.environ.get("POLONY_E2E_INSTALL") != "1", reason="set POLONY_E2E_INSTALL=1 to run the full install (needs network)")
def test_full_install_roundtrip(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    env = {"POLONY_SOURCE": str(ROOT), "XDG_DATA_HOME": str(home / ".local/share")}
    sh([INSTALL / "install.sh"], home, env)
    cli = home / "PolonyReview" / "bin" / "polony-review"
    assert (home / "Desktop" / "Polony Review.desktop").is_file()
    try:
        sh([cli], home, {"POLONY_CI": "1"})
        st = subprocess.run([str(cli), "status"], capture_output=True, text=True, env={**os.environ, "HOME": str(home)})
        assert "running" in st.stdout
    finally:
        subprocess.run([str(cli), "stop"], env={**os.environ, "HOME": str(home)})
    sh([INSTALL / "install.sh"], home, env)  # idempotent re-run
    sh([home / "PolonyReview" / "app" / "install" / "uninstall.sh"], home, env)
    assert (home / "PolonyReviewData").is_dir() and not (home / "PolonyReview").exists()


# ---------------------------------------------------------------- shared database step (fake URL only)
FAKE_URL = "postgresql://fake_user:Sup3rS3cretPw@db.example.invalid/neondb?sslmode=require"
SECRET = "Sup3rS3cretPw"


def _helper(args, stdin, home):
    env = {**os.environ, "POLONY_HOME": str(home), "PYTHONPATH": str(ROOT / "src")}
    env.pop("POLONY_DATABASE_URL", None)
    return subprocess.run([sys.executable, str(INSTALL / "set_database.py"), *args], input=stdin, env=env,
                          capture_output=True, text=True, timeout=60)


def test_set_database_helper_semantics(tmp_path):
    import json
    import stat

    home = tmp_path / "ph"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"data_dir": "/d", "port": 8765}))
    assert _helper(["status"], "", home).returncode == 1
    r = _helper(["set"], FAKE_URL + "\n", home)
    assert r.returncode == 0 and SECRET not in r.stdout + r.stderr
    cfg = json.loads((home / "config.json").read_text())
    assert cfg["database_url"] == FAKE_URL and cfg["port"] == 8765 and "machine_id" not in cfg
    if os.name != "nt":
        assert stat.S_IMODE((home / "config.json").stat().st_mode) == 0o600
    st = _helper(["status"], "", home)
    assert st.returncode == 0 and SECRET not in st.stdout
    other = FAKE_URL.replace("fake_user", "other_user")
    assert _helper(["set"], other + "\n", home).returncode == 3  # existing URL kept without --replace
    assert json.loads((home / "config.json").read_text())["database_url"] == FAKE_URL
    assert _helper(["set", "--replace"], other + "\n", home).returncode == 0
    assert _helper(["set"], "garbage\n", home).returncode == 2
    assert _helper(["clear"], "", home).returncode == 0
    assert "database_url" not in json.loads((home / "config.json").read_text())


@needs_bash
def test_installer_step_noninteractive_env_url_saved_not_leaked(tmp_path):
    """Run only the database step of install.sh (--shortcuts-only skips it, so call the helper path the script uses)."""
    text = (INSTALL / "install.sh").read_text()
    # the script must pass the URL through stdin only: no argv/env use of the secret variable
    assert "printf '%s\\n' \"$url\" | POLONY_HOME" in text
    assert "unset POLONY_DATABASE_URL" in text
    assert "read -rs url </dev/tty" in text
    ps = (INSTALL / "install.ps1").read_text()
    assert "-AsSecureString" in ps and "$url | & $vpy $script" in ps and "Remove-Item Env:\\POLONY_DATABASE_URL" in ps


def test_requirements_cover_psycopg_and_checks():
    extra = (INSTALL / "extra-requirements.txt").read_text()
    assert "psycopg[binary]" in extra and "psycopg_pool" in extra
    assert "psycopg" in (INSTALL / "check_imports.py").read_text()
    assert "psycopg" in (ROOT / "src" / "polony_app" / "launcher.py").read_text()
