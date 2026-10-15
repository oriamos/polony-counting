"""Configuration + password masking + unreachable shared database (no real server needed)."""
import json
import os
import stat

import pytest
from fastapi.testclient import TestClient

from polony_app import config
from polony_app.api.main import create_app
from polony_app.config import Settings, load_settings, mask_url, redact
from polony_app.store.pg import DatabaseUnreachable, PgStore

SECRET = "s3cr3t-pa55"
BAD = f"postgresql://polony_app:{SECRET}@127.0.0.1:1/neondb?sslmode=require"


def test_mask_url_hides_password_and_query():
    m = mask_url("postgresql://u:" + SECRET + "@ep-x-pooler.c-7.eu-central-1.aws.neon.tech/neondb?sslmode=require")
    assert SECRET not in m and m == "postgresql://u:***@ep-x-pooler.c-7.eu-central-1.aws.neon.tech/neondb"


def test_redact_removes_url_and_password_anywhere():
    msg = f'connection to "{BAD}" failed: password {SECRET} postgresql://a:{SECRET}@h/db'
    out = redact(msg, BAD)
    assert SECRET not in out and BAD not in out


def test_settings_repr_never_contains_password():
    assert SECRET not in repr(Settings(database_url=BAD))


def test_env_beats_config_and_machine_id_persisted_with_0600(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"database_url": "postgresql://c:c@cfg/db", "data_dir": "/x"}))
    monkeypatch.setenv("POLONY_HOME", str(home))
    s = load_settings()
    assert s.source == "config" and s.database_url.endswith("@cfg/db")
    monkeypatch.setenv("POLONY_DATABASE_URL", "postgresql://e:e@env/db")
    s2 = load_settings()
    assert s2.source == "env" and s2.database_url.endswith("@env/db")
    cfg = json.loads((home / "config.json").read_text())
    assert cfg["machine_id"] == s2.machine_id == s.machine_id and cfg["data_dir"] == "/x"   # other keys kept
    if os.name != "nt":
        assert stat.S_IMODE((home / "config.json").stat().st_mode) == 0o600
    assert load_settings().machine_id == s.machine_id   # stable


def test_no_url_means_local_sqlite_and_nothing_written(monkeypatch, tmp_path):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("POLONY_DATABASE_URL", raising=False)
    s = load_settings()
    assert not s.shared and not (tmp_path / "h" / "config.json").exists()


def test_invalid_schema_name_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path))
    monkeypatch.setenv("POLONY_DATABASE_SCHEMA", 'x"; drop table y; --')
    with pytest.raises(ValueError):
        load_settings()


def test_unreachable_database_is_503_not_silent_local_fallback(tmp_path):
    st = Settings(database_url=BAD, database_schema="polony", machine_id="a" * 32)
    store = PgStore(tmp_path / "d", st)     # constructor must not raise
    try:
        with pytest.raises(DatabaseUnreachable) as ei:
            store.list_analyses()
        assert ei.value.status == 503 and ei.value.code == "database_unreachable"
        assert SECRET not in ei.value.message
        info = store.info()
        assert info["backend"] == "postgres" and info["connected"] is False
        assert SECRET not in json.dumps(info) and info["last_error"]
    finally:
        store.close()


def test_api_starts_and_reports_503_when_database_is_down(tmp_path):
    st = Settings(database_url=BAD, database_schema="polony", machine_id="b" * 32)
    with TestClient(create_app(tmp_path / "d", st)) as c:
        assert c.get("/api/health").status_code == 200          # the app still starts
        s = c.get("/api/storage").json()
        assert s["backend"] == "postgres" and s["connected"] is False and s["host"] == "127.0.0.1"
        assert SECRET not in json.dumps(s)
        r = c.get("/api/analyses")
        assert r.status_code == 503 and r.json()["code"] == "database_unreachable"
        assert SECRET not in r.text


def test_sqlite_is_default_and_storage_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("POLONY_DATABASE_URL", raising=False)
    with TestClient(create_app(tmp_path / "d")) as c:
        s = c.get("/api/storage").json()
        assert s["backend"] == "sqlite" and s["connected"] is True and s["host"] is None
