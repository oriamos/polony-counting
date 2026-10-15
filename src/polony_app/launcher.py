"""Desktop launcher for the Polony Review app (standard library only).

    python -m polony_app.launcher [launch] [--no-browser] [--browser]
                                                             start the server (if needed) + open a native window
                                                             (pywebview) or, as fallback / with --browser, the browser
    python -m polony_app.launcher stop                       stop the background server
    python -m polony_app.launcher status                     print whether it runs, where, which data dir, shared database
    python -m polony_app.launcher set-database|clear-database  (hidden input) shared Neon/PostgreSQL database in config.json
    python -m polony_app.launcher --self-test [--window]     start a throw-away server, GET /api/health and '/', stop
                                                             (--window: also open and close a native window; skipped without a display)

Files (all under the install home, default ~/PolonyReview, or %LOCALAPPDATA%\\PolonyReview on Windows;
override with POLONY_HOME):
    config.json          {"data_dir": ..., "port": ..., "window": {"width": ..., "height": ...}}
    run/server.json      {"pid": ..., "port": ...}        (pid file of the detached server)
    logs/launcher.log    what the launcher did / why it failed
    logs/server.log      output of the server

Environment: POLONY_HOME, POLONY_DATA_DIR (beats config.json), POLONY_PORT (preferred port),
POLONY_CI=1 (never open a window/browser or show dialogs; used by the CI tests),
POLONY_WINDOW=0 (do not use the native window, open the browser instead; same as --browser).
The server always binds to 127.0.0.1 only and stays a detached background process; the window is just a client.
Closing the window stops the server only when no detection is running; with a running detection the user is asked
(keep it running / cancel the detection and stop). Otherwise the launcher never stops a server by itself; use `stop`.
"""
from __future__ import annotations

import argparse
import getpass
import json
import platform
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Optional

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
START_TIMEOUT_S = 60.0
IS_WIN = os.name == "nt"


# ------------------------------------------------------------------ paths / config
def home_dir() -> Path:
    env = os.environ.get("POLONY_HOME")
    if env:
        return Path(env).expanduser()
    # started from <home>/venv (shortcuts, wrappers): <home> is the folder above the venv
    prefix = Path(sys.prefix).resolve()
    if prefix.name == "venv" and (prefix.parent / "app" / "src" / "polony_app").is_dir():
        return prefix.parent
    if IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "PolonyReview"
    return Path.home() / "PolonyReview"


def default_data_dir() -> Path:
    if IS_WIN:
        docs = Path.home() / "Documents"
        return (docs if docs.exists() else Path.home()) / "PolonyReviewData"
    return Path.home() / "PolonyReviewData"


def config_path() -> Path:
    return home_dir() / "config.json"


def pid_path() -> Path:
    return home_dir() / "run" / "server.json"


def log_path() -> Path:
    return home_dir() / "logs" / "launcher.log"


def server_log_path() -> Path:
    return home_dir() / "logs" / "server.log"


def load_config() -> dict:
    try:
        d = json.loads(config_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    """Atomic write, mode 0600 (the file may hold the shared-database password)."""
    from polony_app.config import save_config_file

    save_config_file(cfg, config_path())
    _restrict_acl(config_path())


def _restrict_acl(p: Path) -> None:
    """Windows: only the current user may read the config file (no-op elsewhere)."""
    if not IS_WIN:
        return
    user = os.environ.get("USERNAME")
    if not user:
        return
    try:
        subprocess.run(["icacls", str(p), "/inheritance:r", "/grant:r", f"{user}:(R,W)"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def data_dir(cfg: Optional[dict] = None) -> Path:
    cfg = load_config() if cfg is None else cfg
    return Path(os.environ.get("POLONY_DATA_DIR") or cfg.get("data_dir") or default_data_dir()).expanduser()


def ci_mode() -> bool:
    return os.environ.get("POLONY_CI", "") not in ("", "0")


_log_ok = True


def log(msg: str) -> None:
    global _log_ok
    try:  # the database URL (password!) must never reach a log
        from polony_app.config import redact

        msg = redact(msg, configured_database_url())
    except Exception:
        pass
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        print(line, flush=True)
    except Exception:  # pythonw: no stdout
        pass
    if not _log_ok:
        return
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        _log_ok = False


# ------------------------------------------------------------------ health / ports
def health(port: int, timeout: float = 1.5) -> Optional[dict]:
    """Return the /api/health JSON if a Polony server answers on `port`, else None."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d if isinstance(d, dict) and d.get("ok") is True and "version" in d else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if not IS_WIN:  # like uvicorn: a port in TIME_WAIT (server just stopped) counts as free
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def choose_port(preferred: int) -> int:
    """`preferred` if it is ours (server already running) or free; otherwise any free port."""
    if health(preferred) is not None or port_is_free(preferred):
        return preferred
    return free_port()


def wait_for_health(port: int, timeout: float = START_TIMEOUT_S, alive: Optional[Callable[[], bool]] = None,
                    interval: float = 0.4) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health(port) is not None:
            return True
        if alive is not None and not alive():
            return False
        time.sleep(interval)
    return health(port) is not None


# ------------------------------------------------------------------ pid file
def write_pid(pid: int, port: int) -> None:
    p = pid_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": pid, "port": port}), encoding="utf-8")


def read_pid() -> Optional[dict]:
    try:
        d = json.loads(pid_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and isinstance(d.get("pid"), int) else None
    except (OSError, ValueError):
        return None


def clear_pid() -> None:
    try:
        pid_path().unlink()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if IS_WIN:
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=15).stdout
            return str(pid) in out
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------ starting the server
def _src_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def server_command(port: int, data: Path) -> list[str]:
    exe = sys.executable
    if IS_WIN and exe.lower().endswith("pythonw.exe"):  # the server wants real std handles (redirected to a log)
        cand = Path(exe).with_name("python.exe")
        if cand.exists():
            exe = str(cand)
    return [exe, "-m", "polony_app", "--data-dir", str(data), "--port", str(port)]


def spawn_server(port: int, data: Path, logfile: Path, local_only: bool = False,
                 config: Optional[Path] = None) -> subprocess.Popen:
    """local_only: never use a shared database (self-test): ignore POLONY_DATABASE_URL and use `config`."""
    data.mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["POLONY_HOME"] = str(home_dir())
    env["POLONY_CONFIG"] = str(config or config_path())  # a path, not a secret; the URL itself stays in the file
    if local_only:
        env.pop("POLONY_DATABASE_URL", None)
    env["PYTHONPATH"] = str(_src_dir()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["POLONY_DATA_DIR"] = str(data)
    kw: dict = {}
    if IS_WIN:
        kw["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                               | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200))
    else:
        kw["start_new_session"] = True
    out = open(logfile, "ab")
    try:
        return subprocess.Popen(server_command(port, data), stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                env=env, close_fds=True, **kw)
    finally:
        out.close()


def stop_pid(pid: int, grace: float = 15.0) -> bool:
    """Ask the process to terminate; True when it is gone."""
    try:
        os.kill(pid, signal.SIGTERM)  # on Windows this is TerminateProcess
    except (ProcessLookupError, OSError):
        return not pid_alive(pid)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    if not IS_WIN:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.5)
    return not pid_alive(pid)


# ------------------------------------------------------------------ user-facing helpers
def open_path(p: Path) -> None:
    try:
        if IS_WIN:
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def show_error(message: str) -> None:
    """Friendly native message box (best effort), never raises; always logged."""
    log("ERROR: " + message)
    if ci_mode():
        return
    text = message + f"\n\nDetails are in the log file:\n{log_path()}"
    try:
        if sys.platform == "darwin":
            esc = text.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(["osascript", "-e", f'display dialog "{esc}" with title "Polony Review" buttons {{"OK"}} '
                            f'default button "OK" with icon caution'], timeout=300)
        elif IS_WIN:
            esc = text.replace("'", "''")
            subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
                            "Add-Type -AssemblyName PresentationFramework; "
                            f"[void][System.Windows.MessageBox]::Show('{esc}','Polony Review')"],
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=300)
        else:
            import tkinter
            from tkinter import messagebox

            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror("Polony Review", text)
            root.destroy()
    except Exception:
        pass
    open_path(log_path())


def open_browser(url: str) -> None:
    if ci_mode():
        log(f"CI mode: not opening a browser ({url})")
        return
    try:
        webbrowser.open(url)
    except Exception as e:  # pragma: no cover
        log(f"could not open a browser: {e}; open {url} manually")


# ------------------------------------------------------------------ native window (pywebview, imported lazily)
WINDOW_TITLE = "Polony Review"
DEFAULT_WINDOW = (1360, 880)
MIN_WINDOW = (960, 640)
KEEP_RUNNING_QUESTION = "A detection is running. Keep Polony Review running in the background?"


def use_window(force_browser: bool = False, no_browser: bool = False) -> bool:
    """Should `launch` try the native window? (Import availability is checked when opening it.)"""
    if force_browser or no_browser or ci_mode():
        return False
    return os.environ.get("POLONY_WINDOW", "").strip().lower() not in ("0", "false", "no", "off")


def display_available() -> bool:
    return IS_WIN or sys.platform == "darwin" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def window_size(cfg: dict) -> tuple[int, int]:
    w = cfg.get("window")
    try:
        width, height = int(w["width"]), int(w["height"])
        if width >= MIN_WINDOW[0] and height >= MIN_WINDOW[1] and width < 20000 and height < 20000:
            return width, height
    except (KeyError, TypeError, ValueError):
        pass
    return DEFAULT_WINDOW


def active_jobs(port: int, timeout: float = 5.0) -> Optional[list]:
    """Running/queued jobs of the server on `port`; None when that cannot be determined (old server, error)."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/jobs/active", timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d if isinstance(d, list) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def cancel_jobs(port: int, jobs: list) -> None:
    for j in jobs:
        try:
            req = urllib.request.Request(f"http://{HOST}:{port}/api/jobs/{j['id']}/cancel", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=10).close()
        except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as e:
            log(f"could not cancel job: {e}")


def close_decision(jobs: Optional[list], confirm_keep: Callable[[], bool]) -> str:
    """What closing the window means: 'stop' (idle), 'keep' (leave the server running), 'cancel' (cancel, then stop).
    Unknown job state counts as running: the server is left alone."""
    if jobs is None:
        return "keep"
    if not jobs:
        return "stop"
    return "keep" if confirm_keep() else "cancel"


def finish_close(decision: str, port: int, wait_s: float = 20.0) -> None:
    """Carry out the close decision after the window is gone. Never stops a server that still runs a detection."""
    if decision == "keep":
        log("window closed; server left running")
        return
    if decision == "cancel":
        cancel_jobs(port, active_jobs(port) or [])
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and active_jobs(port):
            time.sleep(0.5)
    if active_jobs(port):  # still busy (or just started elsewhere): leave it
        log("window closed; a detection is still running, server left running")
        return
    cmd_stop()


class WindowApi:
    """Exposed to the page as window.pywebview.api (js_api)."""

    def __init__(self, webview_module) -> None:
        self._wv = webview_module

    def pick_files(self) -> list:
        wv = self._wv
        if not wv.windows:
            return []
        kind = getattr(getattr(wv, "FileDialog", None), "OPEN", None)
        if kind is None:
            kind = wv.OPEN_DIALOG
        res = wv.windows[0].create_file_dialog(kind, allow_multiple=True, file_types=("TIFF images (*.tif;*.tiff)",))
        return [str(p) for p in (res or [])]


def open_window(url: str, port: int, cfg: dict) -> bool:
    """Show the app in a native window until it is closed. False (reason logged) when no window could be made."""
    try:
        import webview  # pywebview; optional, may lack GTK WebKit on Linux
    except Exception as e:
        log(f"native window unavailable ({type(e).__name__}: {e}); using the browser")
        return False
    state = {"decision": "stop"}
    try:
        width, height = window_size(cfg)
        win = webview.create_window(WINDOW_TITLE, url, width=width, height=height, min_size=MIN_WINDOW,
                                    js_api=WindowApi(webview))

        def on_closing():
            try:
                w, h = int(win.width), int(win.height)
                if (w, h) != (width, height):
                    c = load_config()
                    c["window"] = {"width": w, "height": h}
                    save_config(c)
            except Exception as e:  # remembering the size is a nicety
                log(f"could not remember the window size: {e}")
            jobs = active_jobs(port)
            state["decision"] = close_decision(jobs, lambda: bool(win.create_confirmation_dialog(WINDOW_TITLE, KEEP_RUNNING_QUESTION)))
            return True

        win.events.closing += on_closing
        webview.start()
    except Exception as e:
        log(f"native window failed ({type(e).__name__}: {e}); using the browser")
        return False
    finish_close(state["decision"], port)
    return True


# ------------------------------------------------------------------ commands
def cmd_launch(no_browser: bool = False, force_browser: bool = False) -> int:
    cfg = load_config()
    preferred = int(os.environ.get("POLONY_PORT") or cfg.get("port") or DEFAULT_PORT)
    # an already-running server (ours) is reused, wherever the pid file says it listens
    info = read_pid()
    if info and info.get("port") and health(int(info["port"])) is not None:
        port = int(info["port"])
        log(f"server already running on port {port}")
    elif health(preferred) is not None:
        port = preferred
        log(f"server already running on port {port} (no pid file)")
    else:
        port = choose_port(preferred)
        if port != preferred:
            log(f"port {preferred} is used by another program; using {port}")
        data = data_dir(cfg)
        log(f"starting server on {HOST}:{port}, data dir {data}")
        log(_database_line(cfg, None))
        try:
            proc = spawn_server(port, data, server_log_path())
        except Exception as e:
            show_error(f"Polony Review could not be started ({type(e).__name__}: {e}).")
            return 1
        write_pid(proc.pid, port)
        if not wait_for_health(port, START_TIMEOUT_S, alive=lambda: proc.poll() is None):
            code = proc.poll()
            reason = f"the server exited with code {code}" if code is not None else f"no answer within {int(START_TIMEOUT_S)} s"
            if code is None:
                stop_pid(proc.pid, grace=3)
            clear_pid()
            show_error(f"Polony Review did not start ({reason}). See {server_log_path()} for the server output.")
            return 1
        log("server is up")
    if cfg.get("port") != port:
        cfg["port"] = port
        try:
            save_config(cfg)
        except OSError as e:
            log(f"could not remember the port: {e}")
    url = f"http://{HOST}:{port}/"
    if no_browser:
        log(f"ready: {url}")
    elif use_window(force_browser) and display_available():
        if not open_window(url, port, cfg):
            print("Polony Review: native window not available, opening the browser instead (details in logs/launcher.log).")
            open_browser(url)
    else:
        open_browser(url)
    return 0


def cmd_stop() -> int:
    info = read_pid()
    cfg = load_config()
    if not info:
        port = int(cfg.get("port") or DEFAULT_PORT)
        if health(port) is not None:
            print(f"A Polony Review server answers on port {port} but was not started by this launcher "
                  "(no pid file); not touching it.")
            return 1
        print("Polony Review is not running.")
        return 0
    pid, port = info["pid"], int(info.get("port") or DEFAULT_PORT)
    if not pid_alive(pid):
        clear_pid()
        print("Polony Review is not running (removed a stale pid file).")
        return 0
    if health(port) is None:
        # pid alive but not our server any more (pid reuse) or still booting: refuse to kill blindly
        print(f"Process {pid} does not answer as Polony Review on port {port}; not stopping it.")
        return 1
    ok = stop_pid(pid)
    if ok:
        clear_pid()
        log(f"stopped server (pid {pid})")
        print("Polony Review stopped.")
        return 0
    print(f"Could not stop process {pid}.")
    return 1


def cmd_status() -> int:
    cfg = load_config()
    info = read_pid()
    port = int((info or {}).get("port") or cfg.get("port") or DEFAULT_PORT)
    h = health(port)
    if h is not None:
        print(f"running: http://{HOST}:{port}/  (pid {(info or {}).get('pid', '?')}, version {h.get('version')})")
        print(f"data dir: {data_dir(cfg)}")
        print(_database_line(cfg, port))
        print(_ai_line())
        return 0
    print("not running")
    print(f"data dir: {data_dir(cfg)}")
    print(_database_line(cfg, None))
    print(_ai_line())
    return 3


def _ai_line() -> str:
    """'AI detector: installed (spotiflow x, torch y, device d) / not installed' (imports torch: a few seconds)."""
    import importlib.util
    from importlib import metadata

    try:
        have = importlib.util.find_spec("spotiflow") is not None and importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        have = False
    if not have:
        return "AI detector: not installed (Big-FISH only; re-run the installer with --with-ai / -WithAi to add it)"
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        dev = "cuda" if torch.cuda.is_available() else ("mps" if mps and mps.is_available() else "cpu")
        line = f"AI detector: installed (spotiflow {metadata.version('spotiflow')}, torch {torch.__version__}, device {dev})"
        try:
            from polony_app.engines.spotiflow_adapter import device_note

            note = device_note()      # only set inside a running server; empty for a fresh `status` process
            if note:
                line += f"; {note}"
        except Exception:  # noqa: BLE001  informational only
            pass
        if platform.system() == "Darwin" and platform.machine() == "arm64" and dev == "cpu":
            line += "; MPS unavailable (needs macOS 12.3 or newer), CPU will be used"
        return line
    except Exception as e:  # broken install: say so, Big-FISH still works
        return f"AI detector: installed but cannot be loaded ({type(e).__name__}: {str(e)[:150]}); Big-FISH still works"


def cmd_prepare_model(model: str) -> int:
    """Download (once) and verify the Spotiflow model into <POLONY_HOME>/models/spotiflow; print its sha256."""
    os.environ.setdefault("POLONY_HOME", str(home_dir()))
    try:
        from polony_app.engines.spotiflow_adapter import ensure_model, model_cache_dir
    except Exception as e:
        print(f"The AI detector is not installed ({e}). Re-run the installer with --with-ai / -WithAi.")
        return 1
    print(f"Model folder: {model_cache_dir()}")

    def progress(frac: float, msg: str) -> None:
        print(f"  [{int(frac * 100):3d}%] {msg}", flush=True)

    try:
        info = ensure_model(model, progress=progress)
    except Exception as e:
        print(f"Could not prepare model '{model}': {e}")
        return 1
    print(f"Model '{info.name}' ready ({'already on disk' if info.cached else 'downloaded now'}): {info.path}")
    print(f"sha256: {info.checksum_sha256}")
    return 0


def configured_database_url(cfg: Optional[dict] = None) -> Optional[str]:
    cfg = load_config() if cfg is None else cfg
    u = (os.environ.get("POLONY_DATABASE_URL") or "").strip() or cfg.get("database_url")
    return u.strip() if isinstance(u, str) and u.strip() else None


def _database_line(cfg: dict, port: Optional[int]) -> str:
    url = configured_database_url(cfg)
    if not url:
        return "Shared database: Local only"
    from polony_app.config import mask_url

    shown = mask_url(url)
    if port is not None:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/api/storage", timeout=5) as r:
                st = json.loads(r.read().decode("utf-8"))
            state = "connected" if st.get("connected") else "NOT reachable"
            return f"Shared database: {state} ({shown})"
        except (OSError, ValueError):
            pass
    return f"Shared database: configured ({shown})"


def _read_secret(prompt: str) -> str:
    """Hidden input from the terminal; when stdin is not a terminal, one line from stdin."""
    if sys.stdin is not None and sys.stdin.isatty():
        return getpass.getpass(prompt).strip()
    return (sys.stdin.readline() if sys.stdin is not None else "").strip()


def valid_database_url(url: str) -> bool:
    return url.startswith(("postgresql://", "postgres://")) and "@" in url and len(url) < 2000


def cmd_set_database() -> int:
    url = _read_secret("Paste the shared-database connection string (input is hidden), then Enter: ")
    if not valid_database_url(url):
        print("That does not look like a PostgreSQL connection string (postgresql://user:password@host/db). Nothing changed.")
        return 1
    cfg = load_config()
    cfg["database_url"] = url
    save_config(cfg)
    log("shared database set")
    print("Saved. Restart Polony Review (stop, then start) to use it.")
    return 0


def cmd_clear_database() -> int:
    cfg = load_config()
    if "database_url" in cfg:
        cfg.pop("database_url")
        save_config(cfg)
    log("shared database cleared")
    print("Shared database removed: Polony Review uses local storage only. Restart it to apply.")
    return 0


def window_smoke(port: int) -> bool:
    """Open the UI in a native window, check that the page rendered, close it. Skipped (True) without display/pywebview."""
    if not display_available():
        print("window smoke: skipped (no display)")
        return True
    try:
        import webview
    except Exception as e:
        print(f"window smoke: skipped (pywebview not usable: {type(e).__name__}: {e})")
        return True
    result: dict = {}
    win = webview.create_window(WINDOW_TITLE, f"http://{HOST}:{port}/", width=800, height=600)

    def probe():
        try:
            for _ in range(60):
                time.sleep(0.5)
                if win.evaluate_js('!!document.getElementById("root") && document.getElementById("root").children.length > 0'):
                    result["ok"] = True
                    break
        except Exception as e:
            result["err"] = str(e)
        finally:
            win.destroy()

    try:
        webview.start(probe)
    except Exception as e:
        print(f"window smoke: skipped (no web view available: {type(e).__name__}: {e})")
        return True
    print(f"window smoke: {'ok' if result.get('ok') else 'FAILED ' + result.get('err', '')}")
    return bool(result.get("ok"))


def self_test(window: bool = False) -> int:
    """Start a throw-away server on a free port, check /api/health and the UI at '/', stop it."""
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="polony-selftest-") as td:
        td_p = Path(td)
        proc = spawn_server(port, td_p / "data", td_p / "server.log", local_only=True, config=td_p / "config.json")
        try:
            if not wait_for_health(port, START_TIMEOUT_S, alive=lambda: proc.poll() is None):
                print("SELF-TEST FAILED: server did not become healthy")
                try:
                    print((td_p / "server.log").read_text(errors="replace")[-3000:])
                except OSError:
                    pass
                return 1
            h = health(port) or {}
            print(f"health ok: version={h.get('version')} big-fish={h.get('bigfish_version')}")
            try:
                import psycopg  # noqa: F401
                import psycopg_pool  # noqa: F401

                print(f"psycopg import ok ({psycopg.__version__})")
            except ImportError as e:
                print(f"SELF-TEST FAILED: {e}")
                return 1
            try:
                with urllib.request.urlopen(f"http://{HOST}:{port}/", timeout=10) as r:
                    body = r.read().decode("utf-8", "replace")
                ui = "<div id=\"root\">" in body
            except OSError:
                ui = False
            print(f"web UI served at '/': {'yes' if ui else 'NO'}")
            if not ui:
                return 1
            if window and not window_smoke(port):
                print("SELF-TEST FAILED: native window")
                return 1
            print("SELF-TEST OK")
            return 0
        finally:
            stop_pid(proc.pid, grace=10)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="polony_app.launcher", description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default="launch", choices=["launch", "stop", "status", "set-database", "clear-database", "prepare-model"])
    ap.add_argument("--model", default="general", help="prepare-model: Spotiflow model name (default: general)")
    ap.add_argument("--no-browser", action="store_true", help="start the server but do not open a browser")
    ap.add_argument("--browser", action="store_true", help="open the browser instead of the native window")
    ap.add_argument("--window", action="store_true", help="--self-test: also open and close a native window (skipped without a display)")
    ap.add_argument("--self-test", action="store_true", help="start a temporary server, check it, stop it")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test(window=a.window)
        if a.command == "stop":
            return cmd_stop()
        if a.command == "status":
            return cmd_status()
        if a.command == "prepare-model":
            return cmd_prepare_model(a.model)
        if a.command == "set-database":
            return cmd_set_database()
        if a.command == "clear-database":
            return cmd_clear_database()
        return cmd_launch(no_browser=a.no_browser, force_browser=a.browser)
    except Exception as e:  # last resort: the person double-clicked an icon, they must see something
        import traceback

        log("unexpected error:\n" + traceback.format_exc())
        show_error(f"Polony Review hit an unexpected problem ({type(e).__name__}: {e}).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
