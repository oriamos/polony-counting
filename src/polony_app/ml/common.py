"""Shared helpers of the ML package: hashing, atomic writes, file lock, paths, geometry.

Nothing here imports torch / spotiflow."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np


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


# ------------------------------------------------------------------ strict JSON (hand-written mapping / manifest files)
MAX_JSON_FILE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32                    # the mapping files nest 5-6 levels; anything deeper is not one
PAGE_KEY_RE = re.compile(r"[0-9]{1,4}")
_JSON_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]', re.S)


class DuplicateKeyError(ValueError):
    """A JSON object that repeats a key (the standard parser would silently keep the last value)."""

    def __init__(self, key):
        super().__init__(f"duplicate key {key!r}")
        self.key = key


def reject_duplicate_pairs(pairs):
    """``json.loads(..., object_pairs_hook=reject_duplicate_pairs)``: a repeated key is an error, never 'last one wins'."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise DuplicateKeyError(k)
        out[k] = v
    return out


def check_json_depth(text: str, what: str, limit: int = MAX_JSON_DEPTH) -> None:
    """Refuse absurdly nested JSON before the (recursive) parser sees it: bracket depth is scanned outside strings."""
    depth = 0
    for m in _JSON_TOKEN.finditer(text):
        t = m.group()
        if t == "[" or t == "{":
            depth += 1
            if depth > limit:
                raise MLError(f"{what}: JSON nested deeper than {limit} levels - this is not a valid file of that kind")
        elif t == "]" or t == "}":
            depth -= 1


def loads_strict(text: str, what: str) -> Any:
    """Parse JSON text: bounded depth, duplicate keys refused. Every failure is an :class:`MLError` naming ``what``."""
    check_json_depth(text, what)
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except DuplicateKeyError as exc:
        raise MLError(f"{what}: the key {exc.key!r} appears more than once in one JSON object; refusing to guess which statement "
                      "counts (a standard parser would silently keep the last one). Remove the duplicate.") from exc
    except (RecursionError, MemoryError) as exc:
        raise MLError(f"{what}: too deeply nested or too large to read") from exc
    except ValueError as exc:
        raise MLError(f"{what}: cannot be read as JSON ({exc})") from exc


def read_json_strict(path: str | Path, what: str, *, max_bytes: int = MAX_JSON_FILE_BYTES) -> tuple[Any, bytes]:
    """(parsed JSON, raw bytes) of a hand-written file: size-capped, UTF-8 (BOM allowed), depth-capped, duplicate keys refused.
    Every failure is an :class:`MLError` that names the file."""
    p = Path(path)
    label = f"{what} {p}"
    try:
        size = p.stat().st_size
        if size > max_bytes:
            raise MLError(f"{label}: larger than {max_bytes // (1024 * 1024)} MB - not a mapping file")
        raw = p.read_bytes()
    except OSError as exc:
        raise MLError(f"{label}: cannot be read ({exc.strerror or exc})") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MLError(f"{label}: is not valid UTF-8 ({exc})") from exc
    return loads_strict(text, label), raw


# ------------------------------------------------------------------ files
def utf8_name_problem(name: str) -> Optional[str]:
    """None when ``name`` (a file / path text) survives a strict UTF-8 round trip, else why not (lone surrogates from
    undecodable file-system bytes). Provenance names are written to UTF-8 files and must never be mangled or escaped."""
    try:
        name.encode("utf-8")
        return None
    except UnicodeEncodeError:
        return "the name contains bytes that are not valid UTF-8 / not decodable in the file-system encoding"


def printable_name(name: str) -> str:
    """The name for messages: undecodable bytes are shown as \\x.. escapes instead of crashing the output."""
    return name.encode("utf-8", "backslashreplace").decode("utf-8")


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    try:
        data = text.encode("utf-8")           # strict: a lone surrogate would otherwise surface after the file was opened
    except UnicodeEncodeError as exc:
        raise MLError(f"cannot write {printable_name(str(p))}: the text contains characters that are not valid UTF-8 "
                      f"({exc.reason}); file names of the sources must be valid UTF-8") from exc
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


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


@contextlib.contextmanager
def try_file_lock(path: str | Path) -> Iterator[bool]:
    """Non-blocking variant of :func:`file_lock`: yields True when this process now holds the exclusive lock on ``path`` (kept
    until the block ends), False when another live process holds it. A crashed holder never keeps the lock (the OS drops it)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    got = False
    unlock = None
    try:
        try:
            import fcntl  # POSIX
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                unlock = lambda: fcntl.flock(fd, fcntl.LOCK_UN)       # noqa: E731
            except OSError:
                pass
        except ImportError:  # Windows
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                got = True
                unlock = lambda: msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # noqa: E731
            except OSError:
                pass
        yield got
    finally:
        if got and unlock is not None:
            try:
                unlock()
            except OSError:
                pass
        os.close(fd)


# ------------------------------------------------------------------ paths
def staging_lock_path(data_dir: str | Path, stage_name: str) -> Path:
    """Lock file of a staging directory (``datasets/.staging-<x>``): kept OUTSIDE ``datasets/`` so that a leftover lock file is never
    mistaken for (or counted as) a staging directory or a snapshot."""
    return Path(data_dir) / ".locks" / f"{stage_name.lstrip('.')}.lock"


def datasets_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "datasets"


def models_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "models"


def lock_file_path(data_dir: str | Path) -> Path:
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


# ------------------------------------------------------------------ training / inference view of an image
# One definition of "what the network sees", shared by dataset build (page statistics), training, eval-model and the
# registry (a model remembers the configuration it was trained at). Pure numpy: nothing here imports torch / spotiflow
# and nothing depends on the engine adapter, so the numbers are reproducible without the optional ML stack.
SUPPORTED_DOWNSCALE = (1, 2, 4, 8)
NORMALISER_MODES = ("valid_unsaturated", "none")
DEFAULT_NORMALISER = "valid_unsaturated"
DEFAULT_PERCENTILES = (1.0, 99.8)          # Spotiflow's own 'auto' normaliser
MIN_NORM_PIXELS = 1000                     # fewer valid unsaturated pixels than this: fall back (same rule as the app)
LO_HI_POLICY = ("lo/hi = percentiles (1 / 99.8) of the pixels of the WHOLE PAGE that are inside the valid mask and not "
                "saturated (value < dtype maximum), computed once per page before any crop or downscale; the same "
                "lo/hi is applied to every crop of that page; x_norm = (x - lo) / (hi - lo), not clipped")


def check_downscale(d) -> int:
    try:
        di = int(d)
    except (TypeError, ValueError):
        raise MLError(f"downscale must be one of {list(SUPPORTED_DOWNSCALE)} (got {d!r})")
    if di != d or di not in SUPPORTED_DOWNSCALE:
        raise MLError(f"downscale must be one of {list(SUPPORTED_DOWNSCALE)} (got {d!r})")
    return di


def check_normaliser(mode: str) -> str:
    if mode not in NORMALISER_MODES:
        raise MLError(f"normaliser must be one of {list(NORMALISER_MODES)} (got {mode!r})")
    return mode


def saturation_level(dtype) -> Optional[float]:
    """Integer dtype maximum (clipped pixels); None for floats (no saturation information)."""
    dt = np.dtype(dtype)
    return float(np.iinfo(dt).max) if dt.kind in "ui" else None


def page_bounds(plane, valid=None, percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> dict:
    """lo/hi of a PAGE from its valid, unsaturated pixels (see ``LO_HI_POLICY``). Returns a JSON-able dict:
    lo, hi, p_lo, p_hi, n_used, n_valid, n_saturated, saturation_level, fallback (None | why lo/hi are not the plain rule)."""
    a = np.asarray(plane)
    if a.ndim != 2:
        raise MLError(f"a page must be 2-D, got shape {a.shape}")
    v = np.ones(a.shape, bool) if valid is None else np.asarray(valid, bool)
    if v.shape != a.shape:
        raise MLError("valid mask shape differs from the page shape")
    sat = saturation_level(a.dtype)
    sat_mask = (a >= sat) if sat is not None else np.zeros(a.shape, bool)
    use = v & ~sat_mask
    fallback = None
    if use.sum() < MIN_NORM_PIXELS:
        n_unsat = int(use.sum())
        if v.sum() >= MIN_NORM_PIXELS:
            use, what = v, "all valid pixels (saturated ones included)"
        else:
            use, what = np.ones(a.shape, bool), "the whole page"
        fallback = f"only {n_unsat} valid unsaturated pixel(s) (< {MIN_NORM_PIXELS}): percentiles taken over {what}"
    p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
    lo, hi = (float(x) for x in np.percentile(a[use].astype(np.float64), [p_lo, p_hi]))
    if not hi > lo:
        hi = lo + 1.0
        fallback = (fallback + "; " if fallback else "") + "degenerate percentiles (hi <= lo): hi set to lo + 1"
    return {"lo": lo, "hi": hi, "p_lo": p_lo, "p_hi": p_hi, "n_used": int(use.sum()), "n_valid": int(v.sum()),
            "n_saturated": int((v & sat_mask).sum()), "saturation_level": sat, "fallback": fallback}


def normalise_image(img, lo: Optional[float], hi: Optional[float], mode: str = DEFAULT_NORMALISER):
    """float32 network input. ``none`` = raw values; otherwise (x - lo) / (hi - lo) with the PAGE's lo/hi (not clipped)."""
    x = np.asarray(img).astype(np.float32)
    check_normaliser(mode)
    if mode == "none":
        return x
    if lo is None or hi is None or not float(hi) > float(lo):
        raise MLError("this crop has no usable page lo/hi (it was exported by an older version that normalised per crop); "
                      "rebuild the snapshot (and re-lock the test set) so the page statistics are recorded")
    return ((x - np.float32(lo)) / np.float32(float(hi) - float(lo))).astype(np.float32)


def block_mean_downscale(img, d: int):
    """Block mean over d x d blocks (float32). The bottom / right remainder (side % d pixels) is trimmed, never padded.
    Output pixel (i, j) has its centre at ``d*j + (d-1)/2`` in the input frame (see ``to_downscaled``)."""
    d = check_downscale(d)
    x = np.asarray(img, np.float32)
    if d == 1:
        return x
    h, w = x.shape[0] // d * d, x.shape[1] // d * d
    if h == 0 or w == 0:
        raise MLError(f"image {x.shape[1]}x{x.shape[0]} px is smaller than the downscale factor {d}")
    return x[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3), dtype=np.float32)


def to_downscaled(yx, d: int):
    """Native (y, x) pixel-centre coordinates -> block-mean frame: x_s = (x - (d-1)/2) / d."""
    d = check_downscale(d)
    return (np.asarray(yx, float).reshape(-1, 2) - (d - 1) / 2.0) / d


def from_downscaled(yx_s, d: int):
    """Inverse of :func:`to_downscaled`: x = x_s * d + (d-1)/2."""
    d = check_downscale(d)
    return np.asarray(yx_s, float).reshape(-1, 2) * d + (d - 1) / 2.0


def labels_in_frame(yx_s, shape_hw) -> tuple:
    """(kept labels, n_dropped): labels whose pixel footprint lies inside an image of ``shape_hw``."""
    p = np.asarray(yx_s, float).reshape(-1, 2)
    h, w = shape_hw
    ok = (p[:, 0] >= -0.5) & (p[:, 0] < h - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] < w - 0.5)
    return p[ok], int((~ok).sum())


def grid_offset(origin_xy, d: int) -> tuple:
    """(rows, cols) to trim at the top / left of a crop so that its origin sits on the page-wide block grid
    (page pixel columns / rows that are multiples of d), the grid the app's block mean uses."""
    d = check_downscale(d)
    return ((-int(origin_xy[1])) % d, (-int(origin_xy[0])) % d)


def prepare_crop(img, yx, *, lo, hi, downscale: int = 1, normaliser: str = DEFAULT_NORMALISER, origin_xy=(0, 0)):
    """(network input image, labels in its frame, n_labels_dropped, offset_yx) for one native-resolution crop whose
    top-left pixel is page pixel ``origin_xy`` = (x0, y0). ``offset_yx`` = native pixels trimmed at the top / left to
    reach the block grid: a point (y_s, x_s) of the input maps back to native crop pixels as
    ``from_downscaled(.) + offset_yx``."""
    oy, ox = grid_offset(origin_xy, downscale)
    x = block_mean_downscale(normalise_image(img, lo, hi, normaliser)[oy:, ox:], downscale)
    lab, dropped = labels_in_frame(to_downscaled(np.asarray(yx, float).reshape(-1, 2) - [oy, ox], downscale), x.shape)
    return x, lab.astype("float32"), dropped, (oy, ox)
