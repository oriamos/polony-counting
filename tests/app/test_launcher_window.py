"""Window mode of polony_app.launcher: window-vs-browser decision, close decision, pywebview faked."""
from __future__ import annotations

import sys
import types

import pytest

from polony_app import launcher


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POLONY_CI", raising=False)
    monkeypatch.delenv("POLONY_WINDOW", raising=False)
    return tmp_path


# ---------------------------------------------------------------- window vs browser
def test_use_window_default_and_overrides(monkeypatch):
    assert launcher.use_window() is True
    assert launcher.use_window(force_browser=True) is False
    assert launcher.use_window(no_browser=True) is False
    monkeypatch.setenv("POLONY_WINDOW", "0")
    assert launcher.use_window() is False
    monkeypatch.setenv("POLONY_WINDOW", "1")
    assert launcher.use_window() is True
    monkeypatch.setenv("POLONY_CI", "1")
    assert launcher.use_window() is False


def _fake_webview(monkeypatch, start_raises=None):
    created = {}
    handlers = []

    class Events:
        def __init__(self):
            self.closing = self

        def __iadd__(self, fn):
            handlers.append(fn)
            return self

    class Win:
        width, height = 1000, 700
        events = Events()
        confirms = []

        def create_confirmation_dialog(self, title, msg):
            self.confirms.append(msg)
            return Win.answer

    Win.answer = True
    win = Win()
    wv = types.ModuleType("webview")
    wv.windows = [win]

    def create_window(title, url, **kw):
        created.update(title=title, url=url, **kw)
        return win

    def start(*a, **k):
        if start_raises:
            raise start_raises
        for h in handlers:
            created["close_result"] = h()

    wv.create_window, wv.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", wv)
    return wv, win, created


@pytest.fixture
def launch_env(monkeypatch):
    """cmd_launch against an 'already running' server; records which UI was chosen."""
    calls = {"browser": [], "window": []}
    monkeypatch.setattr(launcher, "health", lambda port, timeout=1.5: {"ok": True, "version": "x"})
    monkeypatch.setattr(launcher, "display_available", lambda: True)
    monkeypatch.setattr(launcher, "open_browser", lambda url: calls["browser"].append(url))
    monkeypatch.setattr(launcher, "open_window", lambda url, port, cfg: calls["window"].append(url) or True)
    return calls


def test_launch_prefers_window(launch_env):
    assert launcher.cmd_launch() == 0
    assert len(launch_env["window"]) == 1 and launch_env["browser"] == []


@pytest.mark.parametrize("kw,env", [({"force_browser": True}, {}), ({}, {"POLONY_WINDOW": "0"}), ({}, {"POLONY_CI": "1"}),
                                    ({"no_browser": True}, {})])
def test_launch_browser_or_nothing(launch_env, monkeypatch, kw, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert launcher.cmd_launch(**kw) == 0
    assert launch_env["window"] == []
    assert len(launch_env["browser"]) == (0 if kw.get("no_browser") else 1)


def test_launch_without_display_uses_browser(launch_env, monkeypatch):
    monkeypatch.setattr(launcher, "display_available", lambda: False)
    launcher.cmd_launch()
    assert launch_env["window"] == [] and len(launch_env["browser"]) == 1


def test_launch_falls_back_when_window_fails(launch_env, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "open_window", lambda url, port, cfg: False)
    launcher.cmd_launch()
    assert len(launch_env["browser"]) == 1
    assert "opening the browser instead" in capsys.readouterr().out


def test_open_window_missing_pywebview_logs_and_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)  # import raises ImportError
    assert launcher.open_window("http://127.0.0.1:1/", 1, {}) is False
    assert "native window unavailable" in launcher.log_path().read_text()


def test_open_window_start_failure_returns_false(monkeypatch):
    _fake_webview(monkeypatch, start_raises=RuntimeError("no GTK"))
    assert launcher.open_window("http://127.0.0.1:1/", 1, {}) is False
    assert "no GTK" in launcher.log_path().read_text()


def test_open_window_creates_window_remembers_size_and_stops_when_idle(monkeypatch):
    _, win, created = _fake_webview(monkeypatch)
    monkeypatch.setattr(launcher, "active_jobs", lambda port, timeout=5.0: [])
    stopped = []
    monkeypatch.setattr(launcher, "cmd_stop", lambda: stopped.append(1) or 0)
    launcher.save_config({"window": {"width": 1200, "height": 800}})
    assert launcher.open_window("http://127.0.0.1:9/", 9, launcher.load_config()) is True
    assert created["title"] == "Polony Review" and created["url"] == "http://127.0.0.1:9/"
    assert (created["width"], created["height"]) == (1200, 800) and created["min_size"] == launcher.MIN_WINDOW
    assert callable(created["js_api"].pick_files)
    assert created["close_result"] is True and stopped == [1] and win.confirms == []
    assert launcher.load_config()["window"] == {"width": 1000, "height": 700}


def test_window_size_defaults_and_bounds():
    assert launcher.window_size({}) == launcher.DEFAULT_WINDOW
    assert launcher.window_size({"window": {"width": "x"}}) == launcher.DEFAULT_WINDOW
    assert launcher.window_size({"window": {"width": 10, "height": 10}}) == launcher.DEFAULT_WINDOW
    assert launcher.window_size({"window": {"width": 1500, "height": 900}}) == (1500, 900)


def test_pick_files_returns_strings(monkeypatch):
    wv, win, _ = _fake_webview(monkeypatch)
    wv.OPEN_DIALOG = 10
    seen = {}
    win.create_file_dialog = lambda kind, **kw: seen.update(kind=kind, **kw) or ("/a/x.tif", "/a/y.tif")
    assert launcher.WindowApi(wv).pick_files() == ["/a/x.tif", "/a/y.tif"]
    assert seen["allow_multiple"] is True and "tif" in seen["file_types"][0]
    win.create_file_dialog = lambda kind, **kw: None  # dialog cancelled
    assert launcher.WindowApi(wv).pick_files() == []


# ---------------------------------------------------------------- close decision
def test_close_decision():
    never = lambda: pytest.fail("must not ask")  # noqa: E731
    assert launcher.close_decision([], never) == "stop"
    assert launcher.close_decision(None, never) == "keep"  # unknown: never risk a detection
    assert launcher.close_decision([{"id": "j"}], lambda: True) == "keep"
    assert launcher.close_decision([{"id": "j"}], lambda: False) == "cancel"


def test_finish_close_keep_never_stops(monkeypatch):
    monkeypatch.setattr(launcher, "cmd_stop", lambda: pytest.fail("stopped"))
    launcher.finish_close("keep", 1)


def test_finish_close_stop_refuses_when_job_appeared(monkeypatch):
    monkeypatch.setattr(launcher, "active_jobs", lambda port, timeout=5.0: [{"id": "j"}])
    monkeypatch.setattr(launcher, "cmd_stop", lambda: pytest.fail("stopped"))
    launcher.finish_close("stop", 1)


def test_finish_close_cancel_then_stop(monkeypatch):
    state = {"jobs": [{"id": "j1"}], "cancelled": [], "stopped": 0}
    monkeypatch.setattr(launcher, "active_jobs", lambda port, timeout=5.0: list(state["jobs"]))

    def cancel(port, jobs):
        state["cancelled"] += [j["id"] for j in jobs]
        state["jobs"] = []

    monkeypatch.setattr(launcher, "cancel_jobs", cancel)
    monkeypatch.setattr(launcher, "cmd_stop", lambda: state.update(stopped=state["stopped"] + 1) or 0)
    launcher.finish_close("cancel", 1, wait_s=2)
    assert state["cancelled"] == ["j1"] and state["stopped"] == 1


def test_finish_close_cancel_that_does_not_finish_keeps_server(monkeypatch):
    monkeypatch.setattr(launcher, "active_jobs", lambda port, timeout=5.0: [{"id": "j"}])
    monkeypatch.setattr(launcher, "cancel_jobs", lambda port, jobs: None)
    monkeypatch.setattr(launcher, "cmd_stop", lambda: pytest.fail("stopped"))
    launcher.finish_close("cancel", 1, wait_s=0.2)


def test_on_closing_asks_when_job_running(monkeypatch):
    _, win, _ = _fake_webview(monkeypatch)
    win.confirms.clear()
    type(win).answer = False
    monkeypatch.setattr(launcher, "active_jobs", lambda port, timeout=5.0: [{"id": "j"}])
    got = []
    monkeypatch.setattr(launcher, "finish_close", lambda decision, port, wait_s=20.0: got.append(decision))
    assert launcher.open_window("http://127.0.0.1:9/", 9, {}) is True
    assert win.confirms == [launcher.KEEP_RUNNING_QUESTION] and got == ["cancel"]


def test_active_jobs_endpoint_lists_running_job(tmp_path):
    """The real API: /api/jobs/active (not shadowed by /jobs/{jid}) lists a running job until it ends."""
    import threading

    from fastapi.testclient import TestClient

    from polony_app.api.main import create_app

    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        assert c.get("/api/jobs/active").json() == []
        release = threading.Event()
        started = threading.Event()

        def work(progress, cancelled):
            started.set()
            release.wait(10)

        # a job needs an existing analysis row only for the shared DB; the local store accepts any id
        job = app.state.runner.submit("a-test", work)
        assert started.wait(10)
        listed = c.get("/api/jobs/active").json()
        assert [j["id"] for j in listed] == [job.id] and listed[0]["status"] in ("running", "queued")
        release.set()
        for _ in range(100):
            if not c.get("/api/jobs/active").json():
                break
            threading.Event().wait(0.1)
        assert c.get("/api/jobs/active").json() == []


def test_active_jobs_client_unreachable_is_none():
    assert launcher.active_jobs(launcher.free_port(), timeout=0.5) is None


def test_window_smoke_skips_without_display(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "display_available", lambda: False)
    assert launcher.window_smoke(1) is True
    assert "skipped" in capsys.readouterr().out


def _run_self_test(*extra):
    import os
    import subprocess

    env = {**os.environ, "POLONY_CI": "1", "PYTHONPATH": str(launcher._src_dir())}
    return subprocess.run([sys.executable, "-m", "polony_app.launcher", "--self-test", *extra], env=env,
                          capture_output=True, text=True, timeout=180)


def test_self_test_headless_still_passes():
    r = _run_self_test()
    assert r.returncode == 0 and "SELF-TEST OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not launcher.display_available() or sys.platform.startswith("linux"),
                    reason="needs a display and a web view (Windows/macOS runners)")
def test_self_test_window_smoke():
    r = _run_self_test("--window")
    assert r.returncode == 0 and "window smoke: ok" in r.stdout, r.stdout + r.stderr
