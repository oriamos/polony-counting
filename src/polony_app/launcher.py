"""Desktop launcher for the Polony Review app (standard library only).

    python -m polony_app.launcher [launch] [--no-browser]   start the server (if needed) + open the browser
    python -m polony_app.launcher stop                       stop the background server
    python -m polony_app.launcher status                     print whether it runs, where, and which data dir
    python -m polony_app.launcher --self-test                start a throw-away server, GET /api/health and '/', stop

Files (all under the install home, default ~/PolonyReview, or %LOCALAPPDATA%\\PolonyReview on Windows;
override with POLONY_HOME):
    config.json          {"data_dir": ..., "port": ...}   (written by the installer / remembered port)
    run/server.json      {"pid": ..., "port": ...}        (pid file of the detached server)
    logs/launcher.log    what the launcher did / why it failed
    logs/server.log      output of the server

Environment: POLONY_HOME, POLONY_DATA_DIR (beats config.json), POLONY_PORT (preferred port),
POLONY_CI=1 (never open a browser or show dialogs; used by the CI tests).
The server always binds to 127.0.0.1 only. The launcher never stops a server by itself (no idle
shutdown: a running detection must not be killed); use `stop`.
"""
from __future__ import annotations

import argparse
import json
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
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def data_dir(cfg: Optional[dict] = None) -> Path:
    cfg = load_config() if cfg is None else cfg
    return Path(os.environ.get("POLONY_DATA_DIR") or cfg.get("data_dir") or default_data_dir()).expanduser()


def ci_mode() -> bool:
    return os.environ.get("POLONY_CI", "") not in ("", "0")


_log_ok = True


def log(msg: str) -> None:
    global _log_ok
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


def spawn_server(port: int, data: Path, logfile: Path) -> subprocess.Popen:
    data.mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
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


# ------------------------------------------------------------------ commands
def cmd_launch(no_browser: bool = False) -> int:
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
        return 0
    print("not running")
    print(f"data dir: {data_dir(cfg)}")
    return 3


def self_test() -> int:
    """Start a throw-away server on a free port, check /api/health and the UI at '/', stop it."""
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="polony-selftest-") as td:
        td_p = Path(td)
        proc = spawn_server(port, td_p / "data", td_p / "server.log")
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
                with urllib.request.urlopen(f"http://{HOST}:{port}/", timeout=10) as r:
                    body = r.read().decode("utf-8", "replace")
                ui = "<div id=\"root\">" in body
            except OSError:
                ui = False
            print(f"web UI served at '/': {'yes' if ui else 'NO'}")
            if not ui:
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
    ap.add_argument("command", nargs="?", default="launch", choices=["launch", "stop", "status"])
    ap.add_argument("--no-browser", action="store_true", help="start the server but do not open a browser")
    ap.add_argument("--self-test", action="store_true", help="start a temporary server, check it, stop it")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.command == "stop":
            return cmd_stop()
        if a.command == "status":
            return cmd_status()
        return cmd_launch(no_browser=a.no_browser)
    except Exception as e:  # last resort: the person double-clicked an icon, they must see something
        import traceback

        log("unexpected error:\n" + traceback.format_exc())
        show_error(f"Polony Review hit an unexpected problem ({type(e).__name__}: {e}).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
