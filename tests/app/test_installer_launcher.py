"""Unit tests of polony_app.launcher (port selection, pid file, health wait, stop) with fake servers."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from polony_app import launcher


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POLONY_CI", "1")
    monkeypatch.delenv("POLONY_DATA_DIR", raising=False)
    monkeypatch.delenv("POLONY_PORT", raising=False)
    return tmp_path / "home"


class _Fake:
    """A tiny HTTP server answering /api/health like the real app (or like something else)."""

    def __init__(self, body: bytes, port: int = 0):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", port), H)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


OURS = json.dumps({"ok": True, "version": "0.1.0", "bigfish_version": "0.6.2"}).encode()


def test_health_recognises_only_polony(_home):
    with _Fake(OURS) as f:
        assert launcher.health(f.port)["version"] == "0.1.0"
    with _Fake(b'{"hello": "world"}') as f:
        assert launcher.health(f.port) is None
    with _Fake(b"<html>nope</html>") as f:
        assert launcher.health(f.port) is None
    assert launcher.health(launcher.free_port()) is None  # nothing listens


def test_choose_port_free_taken_and_ours():
    p = launcher.free_port()
    assert launcher.choose_port(p) == p  # free -> keep
    with socket.socket() as blocker:  # taken by a foreign program -> another free port
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        taken = blocker.getsockname()[1]
        other = launcher.choose_port(taken)
        assert other != taken and launcher.port_is_free(other)
    with _Fake(OURS) as f:  # already our server -> keep (reuse)
        assert launcher.choose_port(f.port) == f.port


def test_wait_for_health_succeeds_when_server_comes_up_late():
    port = launcher.free_port()
    holder = {}

    def later():
        time.sleep(0.8)
        holder["f"] = _Fake(OURS, port).__enter__()

    t = threading.Thread(target=later)
    t.start()
    try:
        assert launcher.wait_for_health(port, timeout=15, interval=0.1)
    finally:
        t.join()
        holder["f"].__exit__()


def test_wait_for_health_gives_up_and_notices_dead_process():
    port = launcher.free_port()
    t0 = time.monotonic()
    assert not launcher.wait_for_health(port, timeout=0.7, interval=0.1)
    assert time.monotonic() - t0 < 5
    t0 = time.monotonic()
    assert not launcher.wait_for_health(port, timeout=30, alive=lambda: False, interval=0.1)  # process died
    assert time.monotonic() - t0 < 5


def test_pid_file_roundtrip_and_stale_handling(_home):
    assert launcher.read_pid() is None
    launcher.write_pid(os.getpid(), 4321)
    assert launcher.read_pid() == {"pid": os.getpid(), "port": 4321}
    assert launcher.pid_alive(os.getpid())
    launcher.clear_pid()
    assert launcher.read_pid() is None
    launcher.clear_pid()  # idempotent
    launcher.pid_path().parent.mkdir(parents=True, exist_ok=True)
    launcher.pid_path().write_text("garbage")
    assert launcher.read_pid() is None


def test_stop_with_stale_pid_file_cleans_up(_home, capsys):
    launcher.write_pid(2**22 + 12345, 8765)  # no such process
    assert launcher.cmd_stop() == 0
    assert launcher.read_pid() is None
    assert "not running" in capsys.readouterr().out


def test_stop_refuses_to_kill_a_process_that_is_not_polony(_home, capsys):
    launcher.write_pid(os.getpid(), launcher.free_port())  # alive pid, but nothing answers on the port
    assert launcher.cmd_stop() == 1
    assert "not stopping" in capsys.readouterr().out
    assert launcher.read_pid() is not None


def test_status_when_not_running(_home, capsys):
    assert launcher.cmd_status() == 3
    assert "not running" in capsys.readouterr().out


def test_config_and_data_dir_resolution(_home, tmp_path, monkeypatch):
    assert launcher.home_dir() == _home
    assert launcher.data_dir() == launcher.default_data_dir()
    launcher.save_config({"data_dir": str(tmp_path / "mydata"), "port": 9001})
    assert launcher.load_config()["port"] == 9001
    assert launcher.data_dir() == tmp_path / "mydata"
    monkeypatch.setenv("POLONY_DATA_DIR", str(tmp_path / "envdata"))
    assert launcher.data_dir() == tmp_path / "envdata"  # env beats config


def test_launch_reuses_running_server_without_spawning(_home, monkeypatch):
    with _Fake(OURS) as f:
        launcher.save_config({"port": f.port})
        monkeypatch.setattr(launcher, "spawn_server", lambda *a, **k: pytest.fail("must not spawn"))
        assert launcher.cmd_launch(no_browser=True) == 0


def test_launch_reports_failure_when_server_dies(_home, monkeypatch):
    class Dead:
        pid = 2**22 + 777

        def poll(self):
            return 3

    monkeypatch.setattr(launcher, "spawn_server", lambda *a, **k: Dead())
    launcher.save_config({"port": launcher.free_port()})
    assert launcher.cmd_launch(no_browser=True) == 1
    assert launcher.read_pid() is None  # no stale pid file left behind
    assert "did not start" in launcher.log_path().read_text()


def test_server_command_binds_localhost_only(tmp_path):
    cmd = launcher.server_command(8765, tmp_path)
    assert cmd[1:3] == ["-m", "polony_app"] and "--host" not in cmd  # __main__ default host is 127.0.0.1
    assert launcher.HOST == "127.0.0.1"


# ---------------------------------------------------------------- shared database settings (fake URL only)
FAKE_URL = "postgresql://fake_user:Sup3rS3cretPw@db.example.invalid/neondb?sslmode=require"


def _no_secret_in(root, secret="Sup3rS3cretPw"):
    for f in root.rglob("*"):
        if f.is_file() and f.name != "config.json":
            assert secret.encode() not in f.read_bytes(), f


def test_set_database_hidden_stdin_mode_0600_keeps_other_keys_and_masks(_home, monkeypatch, capsys):
    import io
    import stat

    launcher.save_config({"data_dir": "/x", "port": 9000})
    monkeypatch.setattr("sys.stdin", io.StringIO(FAKE_URL + "\n"))
    assert launcher.cmd_set_database() == 0
    cfg = launcher.load_config()
    assert cfg["database_url"] == FAKE_URL and cfg["port"] == 9000 and cfg["data_dir"] == "/x"
    assert "machine_id" not in cfg  # never invented/copied by the launcher
    if os.name != "nt":
        assert stat.S_IMODE(launcher.config_path().stat().st_mode) == 0o600
    launcher.log(f"connecting with {FAKE_URL}")  # even a careless log call is redacted
    _no_secret_in(_home)
    out = capsys.readouterr().out
    assert "Sup3rS3cretPw" not in out
    line = launcher._database_line(launcher.load_config(), None)
    assert "Sup3rS3cretPw" not in line and "fake_user:***@db.example.invalid" in line
    assert launcher.cmd_clear_database() == 0
    assert "database_url" not in launcher.load_config()
    assert launcher._database_line(launcher.load_config(), None) == "Shared database: Local only"


def test_set_database_rejects_garbage(_home, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("not a url\n"))
    assert launcher.cmd_set_database() == 1
    assert "database_url" not in launcher.load_config()


def test_server_env_has_no_url_and_selftest_is_local_only(_home, monkeypatch, tmp_path):
    seen = {}

    class P:
        pid = 1

    def fake_popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return P()

    monkeypatch.setenv("POLONY_DATABASE_URL", FAKE_URL)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    launcher.spawn_server(1234, tmp_path / "d", tmp_path / "s.log", local_only=True, config=tmp_path / "c.json")
    assert "POLONY_DATABASE_URL" not in seen["env"] and seen["env"]["POLONY_CONFIG"] == str(tmp_path / "c.json")
    assert not any("Sup3rS3cretPw" in a for a in seen["cmd"])
    launcher.spawn_server(1234, tmp_path / "d", tmp_path / "s.log")  # normal start: file/env only, never argv
    assert not any("Sup3rS3cretPw" in a for a in seen["cmd"])
