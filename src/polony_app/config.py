"""Runtime settings: optional shared (Postgres/Neon) database + stable machine id.

Precedence for the database URL (first one that is set wins):
  1. environment  POLONY_DATABASE_URL
  2. <home>/config.json  key "database_url"   (file should be chmod 600)
No URL configured -> the app uses local SQLite exactly as before.

Other keys: "database_schema" / env POLONY_DATABASE_SCHEMA (default "polony"),
"machine_id" (random UUID, generated once and persisted in config.json).

The URL contains a password: it is never logged; use ``mask_url`` / ``redact`` for anything
that is shown to a person.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

DEFAULT_SCHEMA = "polony"
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


def home_dir() -> Path:
    env = os.environ.get("POLONY_HOME")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "PolonyReview"
    return Path.home() / "PolonyReview"


def config_path() -> Path:
    env = os.environ.get("POLONY_CONFIG")
    return Path(env).expanduser() if env else home_dir() / "config.json"


def load_config_file(path: Optional[Path] = None) -> dict:
    try:
        d = json.loads((path or config_path()).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config_file(cfg: dict, path: Optional[Path] = None) -> None:
    """Atomic write, owner read/write only (the file may hold the database password)."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2))
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover (e.g. Windows)
        pass


# ---------------------------------------------------------------- masking
def mask_url(url: str) -> str:
    """postgresql://user:secret@host/db?x=y -> postgresql://user:***@host/db (query dropped)."""
    try:
        u = urlsplit(url)
        host = u.hostname or ""
        if u.port:
            host += f":{u.port}"
        user = u.username or ""
        auth = f"{user}:***@" if u.password is not None else (f"{user}@" if user else "")
        return f"{u.scheme}://{auth}{host}{u.path}"
    except Exception:  # noqa: BLE001
        return "postgresql://***"


def _secrets_of(url: str) -> list[str]:
    out = [url]
    try:
        u = urlsplit(url)
        if u.password:
            out.append(u.password)
        if u.query:
            out.append(u.query)
    except Exception:  # noqa: BLE001
        pass
    return sorted({s for s in out if s}, key=len, reverse=True)


def redact(text: str, url: Optional[str]) -> str:
    """Remove the URL / password from any message and mask stray user:pass@ patterns."""
    s = str(text)
    if url:
        for secret in _secrets_of(url):
            s = s.replace(secret, "***")
    return re.sub(r"(postgres(?:ql)?://[^:/@\s]+:)[^@\s]+@", r"\1***@", s)


@dataclass
class Settings:
    database_url: Optional[str] = None
    database_schema: str = DEFAULT_SCHEMA
    machine_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "none"          # env | config | argument | none

    def __repr__(self) -> str:  # never leak the password through repr/logging
        shown = mask_url(self.database_url) if self.database_url else None
        return (f"Settings(database_url={shown!r}, database_schema={self.database_schema!r}, "
                f"machine_id={self.machine_id!r}, source={self.source!r})")

    @property
    def shared(self) -> bool:
        return bool(self.database_url)


def _valid_schema(s: str) -> str:
    if not _SCHEMA_RE.match(s):
        raise ValueError("database_schema must match [a-z_][a-z0-9_]{0,40}")
    return s


def load_settings(*, persist_machine_id: bool = True) -> Settings:
    cfg = load_config_file()
    url = (os.environ.get("POLONY_DATABASE_URL") or "").strip() or None
    source = "env" if url else "none"
    if not url:
        v = cfg.get("database_url")
        if isinstance(v, str) and v.strip():
            url, source = v.strip(), "config"
    schema = _valid_schema(
        os.environ.get("POLONY_DATABASE_SCHEMA") or str(cfg.get("database_schema") or DEFAULT_SCHEMA)
    )
    mid = cfg.get("machine_id")
    if not (isinstance(mid, str) and re.fullmatch(r"[0-9a-f]{32}", mid)):
        mid = uuid.uuid4().hex
        if persist_machine_id and url:   # local-only installs need no machine id on disk
            try:
                cfg["machine_id"] = mid
                save_config_file(cfg)
            except OSError:
                pass
    return Settings(database_url=url, database_schema=schema, machine_id=mid, source=source)
