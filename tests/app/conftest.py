"""Backend parametrisation: every test that uses the ``client`` / ``data_dir`` fixtures runs on
SQLite and, when POLONY_TEST_DATABASE_URL is set, also on PostgreSQL in a throw-away schema
``test_<uuid>`` that the test code creates and drops itself.  The URL is a secret: it is only read
from the environment and never printed."""
import os
import uuid

import pytest

PG_URL = os.environ.get("POLONY_TEST_DATABASE_URL", "")
SKIP_REASON = "set POLONY_TEST_DATABASE_URL (a Postgres/Neon dev-branch URL) to run the Postgres backend"


def pytest_generate_tests(metafunc):
    names = set(metafunc.fixturenames)
    if "backend" in names and metafunc.definition.get_closest_marker("sqlite_only") is None \
            and ("client" in names or "data_dir" in names or "pg_too" in names):
        metafunc.parametrize("backend", ["sqlite", "postgres"], indirect=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "sqlite_only: test inspects the SQLite file directly")


@pytest.fixture(autouse=True)
def backend(request, monkeypatch, tmp_path):
    which = getattr(request, "param", "sqlite")
    monkeypatch.setenv("POLONY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POLONY_CONFIG", raising=False)
    monkeypatch.delenv("POLONY_DATABASE_URL", raising=False)
    monkeypatch.delenv("POLONY_DATABASE_SCHEMA", raising=False)
    if which == "sqlite":
        yield "sqlite"
        return
    if not PG_URL:
        pytest.skip(SKIP_REASON)
    schema = "test_" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("POLONY_DATABASE_URL", PG_URL)
    monkeypatch.setenv("POLONY_DATABASE_SCHEMA", schema)
    try:
        yield "postgres"
    finally:
        import psycopg

        with psycopg.connect(PG_URL, autocommit=True) as c:
            c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def pg_schema():
    """A fresh unique schema name for tests that build PgStore instances themselves."""
    if not PG_URL:
        pytest.skip(SKIP_REASON)
    schema = "test_" + uuid.uuid4().hex[:12]
    yield schema
    import psycopg

    with psycopg.connect(PG_URL, autocommit=True) as c:
        c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
