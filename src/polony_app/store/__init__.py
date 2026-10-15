"""Persistence layer: SQLite (default) or shared PostgreSQL + local files, job runner, CSV import."""
from __future__ import annotations

from pathlib import Path

from .db import NotFound, Store  # noqa: F401


def open_store(data_dir: str | Path, settings=None):
    """SQLite unless a database URL is configured; then PostgreSQL (never a silent fallback:
    if it is unreachable the returned store raises 503 ``database_unreachable``)."""
    if settings is not None and getattr(settings, "database_url", None):
        from .pg import PgStore

        return PgStore(data_dir, settings)
    return Store(data_dir)
