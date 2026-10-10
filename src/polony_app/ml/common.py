"""Shared helpers of the ML package: hashing, atomic writes, file lock, paths, geometry.

Nothing here imports torch / spotiflow."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence


class MLError(RuntimeError):
    """A refusal with a message meant for the end user (CLI prints it and exits non-zero)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def app_version() -> str:
    try:
        from polony_app import __version__
        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


# ------------------------------------------------------------------ hashing / json
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text (sorted keys, no whitespace variation)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def fnum(v: Optional[float]) -> str:
    """Round-trip exact float text for CSV ('' for None)."""
    return "" if v is None else repr(float(v))


# ------------------------------------------------------------------ files
def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


@contextlib.contextmanager
def file_lock(path: str | Path, timeout: float = 30.0) -> Iterator[None]:
    """Cross-process exclusive lock on ``path`` (a small side file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.time()
    try:
        try:
            import fcntl  # POSIX

            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - t0 > timeout:
                        raise MLError(f"could not lock {p} within {timeout:.0f}s (another process is writing it)")
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except ImportError:  # Windows
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() - t0 > timeout:
                        raise MLError(f"could not lock {p} within {timeout:.0f}s (another process is writing it)")
                    time.sleep(0.05)
            try:
                yield
            finally:
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
    finally:
        os.close(fd)


# ------------------------------------------------------------------ paths
def datasets_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "datasets"


def models_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "models"


def test_lock_path(data_dir: str | Path) -> Path:
    return datasets_dir(data_dir) / "test_lock.json"


def registry_path(data_dir: str | Path) -> Path:
    return models_dir(data_dir) / "registry.json"


# ------------------------------------------------------------------ geometry
Poly = Sequence[Sequence[float]]


def rect_of(polygon: Poly, tol: float = 1e-6) -> Optional[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) when the polygon is an axis-aligned rectangle (4 corners, or 5 with the first repeated)."""
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) == 5 and pts[0] == pts[-1]:
        pts = pts[:4]
    if len(pts) != 4:
        return None
    xs = sorted({round(p[0], 6) for p in pts})
    ys = sorted({round(p[1], 6) for p in pts})
    if len(xs) != 2 or len(ys) != 2:
        return None
    corners = {(round(x, 6), round(y, 6)) for x in xs for y in ys}
    if {(round(p[0], 6), round(p[1], 6)) for p in pts} != corners:
        return None
    if xs[1] - xs[0] <= tol or ys[1] - ys[0] <= tol:
        return None
    return xs[0], ys[0], xs[1], ys[1]


def pixel_box(rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Integer pixel box (x0, y0, x1, y1), exclusive upper bound, covering exactly the pixel centres the app's
    region rule counts as inside the rectangle ([x0, x1) x [y0, y1), centres at integer coordinates)."""
    return (math.ceil(rect[0]), math.ceil(rect[1]), math.ceil(rect[2]), math.ceil(rect[3]))
