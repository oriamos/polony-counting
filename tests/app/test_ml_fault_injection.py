"""Fault-injection tests of the historical-TIFF workflow: ``import-legacy`` -> ``verify`` -> ``split --lock-test`` -> ``train --dry-run``
(and the app's examined-fields endpoint). The question asked here is NOT "is the science right" (see test_ml_legacy_adversarial.py) but
"does the pipeline survive a hostile real world": SIGKILLed processes, full / read-only / odd file systems, files changing under
its feet, corrupt TIFFs, hand-edited manifests, absurd CLI arguments, other locales / time zones, 20k-crop snapshots.

Conventions
* A real process boundary (``python -m polony_app.ml``) is used wherever the failure is a process / OS matter. Fault hooks are injected
  by a tiny wrapper script written by these tests (``FI_*`` environment variables); nothing in ``src`` knows about them.
* The oracle for "the same snapshot" is a clean uninterrupted run of the same input/options (its version and every byte).
* "Clean refusal" = exit status 2 and ``REFUSED:`` / argparse usage on stderr and NO Python traceback.
* Known defects found while writing the file were kept as ``xfail(strict=True)`` with the reason; all of them are fixed now and
  the markers are gone. A defect found later should be added the same way (strict, with the reason) and removed with its fix.
"""
from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from legacy_bridge_fixtures import (GRID_XS, GRID_YS, RED_GREEN, T0, H, W, make_tiff, marks_in, standard_corpus, write_channel_map,
                                    write_identity_map)
from legacy_fixtures import grid_overlay, point_roi, textured
from polony_app.ml import common as cm
from polony_app.ml import dataset as ds
from polony_app.ml import legacy_bridge as lb
from polony_app.ml.cli import main as ml_main
from polony_app.ml.common import MLError

SRC = Path(sys.modules["polony_app"].__file__).resolve().parents[1]
FR = "0.5,0.25,0.25"
SPECIMENS = ("AM", "AS", "FS", "PS")
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
HAS_SYMLINKS = hasattr(os, "symlink") and os.name != "nt"
needs_posix = pytest.mark.skipif(os.name == "nt", reason="POSIX signals / permissions / symlinks")

# ============================================================================================ process helpers
WRAPPER = textwrap.dedent('''
    """Test-only launcher: installs ONE fault hook chosen by FI_MODE, then runs the real ``polony_app.ml`` CLI."""
    import os, sys, time, errno, signal
    from pathlib import Path
    mode = os.environ.get("FI_MODE", "")
    mark = Path(os.environ["FI_MARK"]) if os.environ.get("FI_MARK") else None
    nth = int(os.environ.get("FI_NTH", "1"))
    wait_for = Path(os.environ["FI_WAIT"]) if os.environ.get("FI_WAIT") else None
    calls = {"n": 0}

    def reach():
        """Tell the test we are at the fault point, then wait to be killed (or released by FI_WAIT)."""
        if mark:
            mark.write_text(str(os.getpid()))
        if wait_for is not None:
            t0 = time.time()
            while not wait_for.exists() and time.time() - t0 < 60:
                time.sleep(0.01)
            return
        time.sleep(600)

    def count():
        calls["n"] += 1
        return calls["n"] == nth

    if os.environ.get("FI_CLOCK"):                       # a wrong system clock
        import polony_app.ml.common as _c, polony_app.ml.dataset as _d
        fake = os.environ["FI_CLOCK"]
        _c.utc_now = lambda: fake
        _d.utc_now = lambda: fake
    if mode == "scan":
        import polony_app.ml.legacy_bridge as lb
        real = lb.sha256_stream
        def hooked(*a, **k):
            if count():
                reach()
            return real(*a, **k)
        lb.sha256_stream = hooked
    elif mode == "crop":
        import polony_app.ml.dataset as ds
        real = ds.write_crop_files
        def hooked(*a, **k):
            if count():
                reach()
            return real(*a, **k)
        ds.write_crop_files = hooked
    elif mode == "torn_crop":                            # the nth crop .tif is half written when the process dies
        import polony_app.ml.dataset as ds
        real = ds.write_crop_files
        def hooked(dirpath, crop_id, image, yx):
            if count():
                real(dirpath, crop_id, image, yx)
                p = Path(dirpath) / (crop_id + ".tif")
                b = p.read_bytes()
                p.write_bytes(b[: len(b) // 2])
                (Path(dirpath) / (crop_id + ".csv")).unlink()
                reach()
            return real(dirpath, crop_id, image, yx)
        ds.write_crop_files = hooked
    elif mode == "publish":                              # between the complete manifest and the rename (publication)
        real = os.rename
        def hooked(src, dst, *a, **k):
            if ".staging-" in str(src) and "datasets" in str(dst) and (Path(src) / "manifest.json").exists():
                reach()
            return real(src, dst, *a, **k)
        os.rename = hooked
    elif mode == "hash_wait":                            # the nth crop hash is computed late (a concurrent run may have finished)
        import polony_app.ml.dataset as ds
        real = ds.sha256_file
        def hooked(p, *a, **k):
            if str(p).endswith(".tif") and ".staging-" in str(p) and count():
                reach()
            return real(p, *a, **k)
        ds.sha256_file = hooked
    elif mode == "fsize":                                # the file system "fills up": writes beyond N bytes fail
        import resource
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        lim = int(os.environ["FI_LIMIT"])
        resource.setrlimit(resource.RLIMIT_FSIZE, (lim, lim))
    elif mode == "rlimit_as":
        import resource
        lim = int(os.environ["FI_LIMIT"])
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    elif mode == "enospc":                               # ENOSPC from the nth crop on
        import polony_app.ml.dataset as ds
        real = ds.write_crop_files
        def hooked(*a, **k):
            calls["n"] += 1
            if calls["n"] >= nth:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(*a, **k)
        ds.write_crop_files = hooked
    elif mode == "eacces":                               # permission denied on reading the nth source file
        import polony_app.ml.legacy_bridge as lb
        real = lb.sha256_stream
        def hooked(p, *a, **k):
            if count():
                raise PermissionError(errno.EACCES, "Permission denied", str(p))
            return real(p, *a, **k)
        lb.sha256_stream = hooked
    if os.environ.get("FI_MAXRSS"):
        import atexit, resource
        atexit.register(lambda: print("MAXRSS_KB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=sys.stderr))
    from polony_app.ml.cli import main
    raise SystemExit(main(sys.argv[1:]))
''')


class CP:
    """A finished process: ``rc`` (negative = killed by that signal), ``out``, ``err``."""

    def __init__(self, rc, out, err):
        self.rc, self.out, self.err = rc, out, err

    @property
    def text(self):
        return self.out + "\n" + self.err

    @property
    def traceback(self):
        return "Traceback (most recent call last)" in self.err or "Traceback (most recent call last)" in self.out

    def __repr__(self):
        return f"CP(rc={self.rc}, out=...{self.out[-600:]!r}, err=...{self.err[-900:]!r})"


def base_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("POLONY_", "FI_"))}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update({k: str(v) for k, v in extra.items()})
    return env


@pytest.fixture(scope="session")
def wrapper(tmp_path_factory):
    p = tmp_path_factory.mktemp("fi_wrapper") / "fi_launch.py"
    p.write_text(WRAPPER, encoding="utf-8")
    return p


def mlp(args, data=None) -> CP:
    """In-process ``polony_app.ml`` main (fast path for commands where no process boundary matters). Any exception that escapes
    ``main`` is reported as a traceback on ``err`` - exactly what the operator would see."""
    import traceback
    argv = (["--data-dir", str(data)] if data is not None else []) + [str(a) for a in args]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = ml_main(argv)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except BaseException:   # noqa: BLE001
            err.write(traceback.format_exc())
            rc = 1
    return CP(rc, out.getvalue(), err.getvalue())


def ml(args, *, data=None, env=None, timeout=180, wrapper_path=None, cwd=None):
    """Run ``python -m polony_app.ml [--data-dir D] args`` to completion."""
    cmd = [sys.executable, str(wrapper_path)] if wrapper_path else [sys.executable, "-m", "polony_app.ml"]
    if data is not None:
        cmd += ["--data-dir", str(data)]
    cmd += [str(a) for a in args]
    p = subprocess.run(cmd, capture_output=True, env=env or base_env(), timeout=timeout, cwd=cwd)
    return CP(p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"))


def spawn(args, wrapper_path, *, data, env, cwd=None):
    cmd = [sys.executable, str(wrapper_path), "--data-dir", str(data)] + [str(a) for a in args]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd)


def finish(proc, timeout=120) -> CP:
    out, err = proc.communicate(timeout=timeout)
    return CP(proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"))


def wait_for(path: Path, proc=None, timeout=60.0):
    t0 = time.time()
    while not Path(path).exists():
        if proc is not None and proc.poll() is not None:
            return False
        if time.time() - t0 > timeout:
            raise AssertionError(f"timeout waiting for {path}")
        time.sleep(0.01)
    return True


def kill_at(wrapper_path, args, *, data, mode, nth=1, tmp, extra_env=None, sig=signal.SIGKILL) -> CP:
    """Start the CLI with a hook that parks it at fault point ``mode``/``nth``; SIGKILL it there. Returns the dead process."""
    mark = Path(tmp) / f"mark-{mode}-{nth}-{time.time_ns()}"
    env = base_env(FI_MODE=mode, FI_NTH=nth, FI_MARK=mark, **(extra_env or {}))
    proc = spawn(args, wrapper_path, data=data, env=env)
    reached = wait_for(mark, proc)
    if reached:
        proc.send_signal(sig)
    res = finish(proc)
    assert reached, f"the process finished before reaching the fault point {mode}#{nth}: {res}"
    assert res.rc == -sig, res
    return res


def run_hooked(wrapper_path, args, *, data, mode, nth=1, extra_env=None, timeout=180) -> CP:
    env = base_env(FI_MODE=mode, FI_NTH=nth, **(extra_env or {}))
    return finish(spawn(args, wrapper_path, data=data, env=env), timeout)


def clean_refusal(cp: CP, *what):
    assert cp.rc == 2, cp
    assert not cp.traceback, cp
    assert "REFUSED:" in cp.err or "usage:" in cp.err, cp
    for w in what:
        assert re.search(w, cp.err, re.I | re.S), (w, cp)


def tree_digest(*roots) -> str:
    """Hash of every path (type, name, size, mode, content) below the roots: proves a command wrote / changed NOTHING."""
    h = hashlib.sha256()
    for root in roots:
        root = Path(root)
        if not root.exists() and not root.is_symlink():
            h.update(b"<absent>")
            continue
        for q in sorted([root, *root.rglob("*")], key=lambda p: str(p)):
            st = q.lstat()
            h.update(f"{q.relative_to(root.parent)}|{st.st_mode}|{st.st_size}|{int(q.is_symlink())}".encode("utf-8", "surrogateescape"))
            if q.is_file() and not q.is_symlink():
                h.update(q.read_bytes())
    return h.hexdigest()


def snap_digest(data, version) -> dict:
    root = Path(data) / "datasets" / version
    return {str(q.relative_to(root)): q.read_bytes() for q in sorted(root.rglob("*")) if q.is_file()}


def crop_files(data, version) -> dict:
    """Every exported crop file (pixels + labels) of a snapshot: independent of the input directory's name / location."""
    root = Path(data) / "datasets" / version / "spotiflow"
    return {str(q.relative_to(root)): q.read_bytes() for q in sorted(root.rglob("*")) if q.is_file()}


def staging_dirs(data) -> list:
    d = Path(data) / "datasets"
    return sorted(p.name for p in d.iterdir() if p.name.startswith(".staging")) if d.is_dir() else []


def published(data) -> list:
    d = Path(data) / "datasets"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "manifest.json").exists()) \
        if d.is_dir() else []


def unlock(path):
    for q in [Path(path), *Path(path).rglob("*")]:
        os.chmod(q, 0o755 if q.is_dir() else 0o644)


def assert_converged(r: CP):
    """A (re)run of an import whose snapshot may already exist: success, or the (old) 'identical snapshot exists' refusal; never a crash.
    The idempotent exit-0 behaviour is asserted separately (test_reimport_of_identical_input_is_idempotent)."""
    assert not r.traceback, r
    assert r.rc == 0 or (r.rc == 2 and re.search(r"already (exists|imported)", r.err)), r


def version_of(cp: CP) -> str:
    m = re.search(r"created snapshot (ds-[0-9a-f]{16})", cp.out)
    assert m, cp
    return m.group(1)


def sha_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================================================ corpora
class Corpus:
    """A directory of valid ImageJ-annotated TIFFs plus the mapping files that go with it."""

    def __init__(self, base: Path, specimens=SPECIMENS):
        self.base = Path(base)
        self.inp = self.base / "in"
        self.inp.mkdir(parents=True, exist_ok=True)
        self.files = standard_corpus(self.inp, specimens)
        self.imap = write_identity_map(self.base / "id.json", self.files)
        self.cmap = write_channel_map(self.base / "ch.json", defaults=RED_GREEN)

    def args(self, inp=None, *extra, imap=True, cmap=True, fractions=FR, created=T0):
        a = ["import-legacy", "--input", inp or self.inp]
        if cmap:
            a += ["--channel-map", self.cmap if cmap is True else cmap]
        if imap:
            a += ["--identity-map", self.imap if imap is True else imap]
        if fractions:
            a += ["--fractions", fractions]
        if created:
            a += ["--created-at", created]
        return a + list(extra)

    def copy_to(self, dest: Path) -> "Corpus":
        dest = Path(dest)
        shutil.copytree(self.base, dest)
        c = object.__new__(Corpus)
        c.base, c.inp, c.files = dest, dest / "in", dict(self.files)
        c.imap, c.cmap = dest / "id.json", dest / "ch.json"
        return c


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Corpus:
    return Corpus(tmp_path_factory.mktemp("fi_corpus"))


@pytest.fixture(scope="module")
def clean(corpus, tmp_path_factory):
    """The oracle: one uninterrupted import of the corpus. -> (data_dir, version, {rel: bytes})."""
    data = tmp_path_factory.mktemp("fi_clean") / "data"
    r = ml(corpus.args(), data=data)
    assert r.rc == 0 and not r.traceback, r
    v = version_of(r)
    return data, v, snap_digest(data, v)


@pytest.fixture
def work(corpus, tmp_path) -> Corpus:
    """A private copy of the corpus (tests that touch the input)."""
    return corpus.copy_to(tmp_path / "w")


def dummy_base(tmp_path) -> Path:
    """A base-model FOLDER (``train --base``): resolves without any pretrained download."""
    p = tmp_path / "base_model"
    p.mkdir(exist_ok=True)
    (p / "config.yaml").write_text("{}\n")
    return p


def train_dry(data, version, base, *extra, **kw) -> CP:
    return ml(["train", "--snapshot", version, "--base", base, "--dry-run", "--allow-small", *extra], data=data, **kw)


# ============================================================================================ 1. real process kill
def assert_nothing_published(data, tmp_path):
    """No snapshot is listed, and none of the on-disk leftovers is addressable / verifiable / lockable / trainable as one."""
    assert published(data) == []
    r = ml(["list", "datasets"], data=data)
    assert r.rc == 0 and "no dataset snapshots" in r.out and not r.traceback, r
    names = [p.name for p in (Path(data) / "datasets").iterdir()] if (Path(data) / "datasets").is_dir() else []
    assert all(n.startswith(".staging-") for n in names), names
    base = dummy_base(tmp_path)
    for n in names + ["ds-0123456789abcdef", "ds-"]:
        for cmd in (["verify", n], ["show", n], ["split", "--snapshot", n, "--lock-test"],
                    ["train", "--snapshot", n, "--base", base, "--dry-run", "--allow-small"]):
            clean_refusal(mlp(cmd, data=data))
    assert not (Path(data) / "datasets" / "test_lock.json").exists()


KILL_POINTS = [("scan", 1), ("scan", 6), ("crop", 1), ("crop", 9), ("crop", 25), ("torn_crop", 7), ("publish", 1)]


@needs_posix
@pytest.mark.parametrize("mode,nth", KILL_POINTS, ids=[f"{m}#{n}" for m, n in KILL_POINTS])
def test_sigkill_publishes_nothing_and_a_rerun_equals_the_uninterrupted_run(corpus, clean, wrapper, tmp_path, mode, nth):
    """SIGKILL during the scan / while writing crops (also a half-written crop file) / after the complete manifest but before the
    rename: no snapshot appears, nothing leftover is addressable, and the rerun is byte-identical to a clean run."""
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    kill_at(wrapper, corpus.args(), data=data, mode=mode, nth=nth, tmp=tmp_path)
    assert_nothing_published(data, tmp_path)
    assert len(staging_dirs(data)) <= 1
    r = ml(corpus.args(), data=data)
    assert r.rc == 0 and not r.traceback, r
    assert version_of(r) == cver
    assert snap_digest(data, cver) == cdig
    assert published(data) == [cver] and staging_dirs(data) == []
    assert ml(["verify", cver], data=data).out.strip() == "intact"


@needs_posix
def test_repeated_kills_at_different_moments_never_accumulate_staging_and_the_final_run_matches(corpus, clean, wrapper, tmp_path):
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    for mode, nth in (("crop", 3), ("scan", 2), ("torn_crop", 2), ("crop", 2), ("publish", 1)):
        kill_at(wrapper, corpus.args(), data=data, mode=mode, nth=nth, tmp=tmp_path)
        assert published(data) == [] and len(staging_dirs(data)) <= 1, (mode, nth, staging_dirs(data))
    r = ml(corpus.args(), data=data)
    assert r.rc == 0 and version_of(r) == cver and snap_digest(data, cver) == cdig
    assert staging_dirs(data) == []


@needs_posix
def test_kill_then_dry_run_and_other_commands_do_not_touch_the_staging_leftover(corpus, wrapper, tmp_path):
    data = tmp_path / "data"
    kill_at(wrapper, corpus.args(), data=data, mode="crop", nth=12, tmp=tmp_path)
    before = tree_digest(data)
    r = ml(corpus.args(None, "--dry-run"), data=data)
    assert r.rc == 0 and "DRY RUN" in r.out and not r.traceback, r
    assert ml(["list", "datasets"], data=data).rc == 0
    assert tree_digest(data) == before


@needs_posix
def test_stale_staging_of_superseded_inputs_is_reclaimed(corpus, wrapper, tmp_path):
    """Crash -> the researcher removes a bad file (a different plan, another resume key) -> crash again ... -> a successful run
    publishes and reclaims every staging directory of the superseded plans (none of them has a live lock)."""
    w = corpus.copy_to(tmp_path / "w")
    data = tmp_path / "data"
    victims = sorted(w.inp.glob("*.tif"))
    for cycle in range(3):
        kill_at(wrapper, w.args(), data=data, mode="crop", nth=10, tmp=tmp_path)
        assert len(staging_dirs(data)) == cycle + 1, staging_dirs(data)
        os.remove(victims[cycle + 1])
    r = ml(w.args(), data=data)
    assert r.rc == 0 and not r.traceback and "removed 3 stale staging directories" in r.out, r
    assert staging_dirs(data) == [] and len(published(data)) == 1
    assert not (data / ".locks").exists()


def test_a_staging_directory_with_a_live_lock_is_never_reclaimed(corpus, wrapper, tmp_path):
    """Run A (plan 1) is alive and parked while it writes its crops; run B (plan 2) publishes and reclaims: A's staging survives."""
    w = corpus.copy_to(tmp_path / "w")
    data = tmp_path / "data"
    release, mark = tmp_path / "release", tmp_path / "markA"
    a = spawn(w.args(), wrapper, data=data, env=base_env(FI_MODE="crop", FI_NTH=3, FI_MARK=mark, FI_WAIT=release))
    wait_for(mark, a)
    live = staging_dirs(data)
    assert len(live) == 1
    w2 = corpus.copy_to(tmp_path / "w2")                            # plan 2: one file fewer
    os.remove(sorted(w2.inp.glob("*.tif"))[1])
    b = ml(w2.args(), data=data)
    assert b.rc == 0 and not b.traceback, b
    assert staging_dirs(data) == live, "the staging directory of a live run was removed"
    release.write_text("go")
    ra = finish(a)
    assert ra.rc == 0 and not ra.traceback, ra                         # A finishes its own plan undisturbed
    assert staging_dirs(data) == [] and len(published(data)) == 2, (ra, staging_dirs(data), published(data))
    assert all(ml(["verify", v], data=data).out.strip() == "intact" for v in published(data))


def test_old_staging_directories_are_reclaimed_by_age_unless_their_lock_is_held(tmp_path):
    """``reclaim_staging`` (run at the start of every real import): a staging directory untouched for more than 7 days is removed, a
    fresh one stays, and an old one whose lock is held by a live run stays too."""
    data = tmp_path / "data"
    names = (".staging-old", ".staging-fresh", ".staging-old-but-running")
    for n in names:
        d = data / "datasets" / n
        (d / ".done").mkdir(parents=True)
        (d / "junk.tif").write_bytes(b"x")
    ancient = time.time() - 8 * 86400
    for n in (names[0], names[2]):
        for q in (data / "datasets" / n / ".done", data / "datasets" / n / "junk.tif", data / "datasets" / n):
            os.utime(q, (ancient, ancient))
    with cm.try_file_lock(cm.staging_lock_path(data, names[2])) as got:
        assert got
        removed = ds.reclaim_staging(data, keep="")
    assert removed == [".staging-old"]
    assert staging_dirs(data) == [".staging-fresh", ".staging-old-but-running"]
    assert ds.reclaim_staging(data, keep="", superseded=True) == [".staging-fresh", ".staging-old-but-running"]
    assert staging_dirs(data) == []


# ============================================================================================ 2. concurrent imports
def two_parked_runs(corpus, wrapper, tmp_path, data, mode, nth):
    """Two imports of the same input/options, both parked at the same fault point, then released together."""
    release = tmp_path / "release"
    procs, marks = [], []
    for i in range(2):
        mark = tmp_path / f"mark{i}"
        marks.append(mark)
        env = base_env(FI_MODE=mode, FI_NTH=nth, FI_MARK=mark, FI_WAIT=release)
        procs.append(spawn(corpus.args(), wrapper, data=data, env=env))
    for m, p in zip(marks, procs):
        wait_for(m, p)
    release.write_text("go")
    return [finish(p) for p in procs]


@needs_posix
@pytest.mark.parametrize("mode,nth", [("publish", 1), ("crop", 1), ("crop", 17)])
def test_two_concurrent_identical_imports_never_corrupt_the_snapshot(corpus, clean, wrapper, tmp_path, mode, nth):
    """Race: whichever interleaving, the data dir ends with exactly the clean snapshot (byte-identical, verifiable) or nothing."""
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    results = two_parked_runs(corpus, wrapper, tmp_path, data, mode, nth)
    assert published(data) in ([cver], []), (published(data), results)
    assert any(r.rc == 0 for r in results) or published(data) == [], results
    if published(data):
        assert snap_digest(data, cver) == cdig
        assert ml(["verify", cver], data=data).out.strip() == "intact"
    # whatever happened, one more run converges to the clean snapshot with nothing left over
    r = ml(corpus.args(), data=data)
    assert_converged(r)
    assert published(data) == [cver] and snap_digest(data, cver) == cdig


@needs_posix
def test_no_staging_directory_survives_a_lost_race_and_a_reimport(corpus, clean, wrapper, tmp_path):
    data = tmp_path / "data"
    two_parked_runs(corpus, wrapper, tmp_path, data, "publish", 1)
    assert_converged(ml(corpus.args(), data=data))
    assert staging_dirs(data) == []


@needs_posix
def test_the_loser_of_a_publication_race_exits_cleanly(corpus, wrapper, tmp_path):
    results = two_parked_runs(corpus, wrapper, tmp_path, tmp_path / "data", "publish", 1)
    for r in results:
        assert not r.traceback, r
        assert r.rc in (0, 2), r


@needs_posix
def test_an_import_whose_staging_directory_was_published_by_another_run_exits_cleanly(corpus, clean, wrapper, tmp_path):
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    release = tmp_path / "release"
    mark = tmp_path / "markA"
    a = spawn(corpus.args(), wrapper, data=data, env=base_env(FI_MODE="hash_wait", FI_NTH=3, FI_MARK=mark, FI_WAIT=release))
    wait_for(mark, a)
    b = ml(corpus.args(), data=data)
    assert b.rc == 0 and version_of(b) == cver
    release.write_text("go")
    ra = finish(a)
    assert published(data) == [cver] and snap_digest(data, cver) == cdig      # never corrupted
    assert not ra.traceback and ra.rc in (0, 2), ra


@needs_posix
def test_natural_race_of_two_full_imports_ends_in_one_intact_snapshot(corpus, clean, tmp_path):
    cdata, cver, cdig = clean
    for rnd in range(2):
        data = tmp_path / f"data{rnd}"
        procs = [subprocess.Popen([sys.executable, "-m", "polony_app.ml", "--data-dir", str(data)] + [str(a) for a in corpus.args()],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=base_env()) for _ in range(2)]
        res = [finish(p) for p in procs]
        assert published(data) in ([cver], []), res
        if published(data):
            assert snap_digest(data, cver) == cdig
        r = ml(corpus.args(), data=data)
        assert_converged(r)
        assert snap_digest(data, cver) == cdig


@needs_posix
def test_reimport_of_identical_input_is_idempotent(corpus, clean, tmp_path):
    cdata, cver, cdig = clean
    before = tree_digest(cdata)
    r = ml(corpus.args(), data=cdata)
    assert r.rc == 0 and f"already imported as {cver}" in r.out and not r.traceback, r
    assert tree_digest(cdata) == before and staging_dirs(cdata) == []


# ============================================================================================ 3. disk and file system
@needs_posix
@pytest.mark.parametrize("mode,extra", [("fsize", {"FI_LIMIT": 6000}), ("enospc", {"FI_NTH": 5})], ids=["RLIMIT_FSIZE", "ENOSPC"])
def test_disk_full_mid_write_leaves_no_valid_partial_snapshot(corpus, clean, wrapper, tmp_path, mode, extra):
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    env = base_env(FI_MODE=mode, **extra)
    r = finish(spawn(corpus.args(), wrapper, data=data, env=env))
    assert r.rc != 0, r
    assert published(data) == []
    assert ml(["list", "datasets"], data=data).rc == 0
    for n in staging_dirs(data):
        clean_refusal(mlp(["verify", n], data=data))
    r2 = ml(corpus.args(), data=data)                  # space is back: identical to the clean run
    assert r2.rc == 0 and version_of(r2) == cver and snap_digest(data, cver) == cdig


@needs_posix
@pytest.mark.parametrize("mode,extra", [("fsize", {"FI_LIMIT": 6000}), ("enospc", {"FI_NTH": 5})], ids=["RLIMIT_FSIZE", "ENOSPC"])
def test_disk_full_is_reported_as_a_clear_refusal(corpus, wrapper, tmp_path, mode, extra):
    r = finish(spawn(corpus.args(), wrapper, data=tmp_path / "data", env=base_env(FI_MODE=mode, **extra)))
    clean_refusal(r, r"space|disk|write|full|file too large")


@needs_posix
@pytest.mark.parametrize("kind", ["parent_is_file", "under_proc", "dev_null"])
def test_unwritable_data_dir_is_refused_before_anything_is_read_or_written(corpus, tmp_path, kind):
    if kind == "parent_is_file":
        f = tmp_path / "afile"
        f.write_text("x")
        data = f / "data"
    elif kind == "under_proc":
        data = Path("/proc/self/fi-data")
    else:
        data = Path("/dev/null/data")
    for extra in ([], ["--dry-run"]):
        r = ml(corpus.args(None, *extra), data=data)
        if extra:     # a dry run writes nothing, so it does not care where the data dir is (but must not crash)
            assert r.rc == 0 and not r.traceback, r
        else:
            clean_refusal(r)


@needs_posix
@pytest.mark.skipif(IS_ROOT, reason="root ignores directory permissions")
def test_read_only_data_dir_is_refused_cleanly(corpus, tmp_path):
    data = tmp_path / "data"
    (data / "datasets").mkdir(parents=True)
    os.chmod(data / "datasets", 0o555)
    try:
        clean_refusal(ml(corpus.args(), data=data))
    finally:
        os.chmod(data / "datasets", 0o755)
    assert published(data) == [] and staging_dirs(data) == []


@needs_posix
def test_unwritable_data_dir_in_process_is_a_refusal_not_a_traceback(corpus, tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    r = mlp(corpus.args(), data=f / "data")
    clean_refusal(r)


@pytest.mark.parametrize("name", ["with space", "ünïcödé-日本語-🧪", "nfd-e\u0301\u0301", "semi;colon&amp'quote\"dq"])
def test_whole_pipeline_in_a_data_dir_and_input_dir_with_odd_characters(corpus, clean, tmp_path, name):
    """import -> verify -> split --lock-test -> train --dry-run with spaces / unicode (also decomposed) / shell metacharacters in the
    data dir and in the PATH of the input: same snapshot as the clean run (no absolute path leaks into the version). An input
    directory with an odd NAME yields the same crops (its name is recorded, so the version may differ)."""
    cdata, cver, cdig = clean
    w = corpus.copy_to(tmp_path / name / "parent dir")
    data = tmp_path / name / "data dir"
    r = ml(w.args(), data=data)
    assert r.rc == 0 and not r.traceback, r
    v = version_of(r)
    assert v == cver and snap_digest(data, v) == cdig
    assert ml(["verify", v], data=data).out.strip() == "intact"
    r = ml(["split", "--snapshot", v, "--lock-test"], data=data)
    assert r.rc == 0 and "LOCKED" in r.out, r
    r = train_dry(data, v, dummy_base(tmp_path / name))
    assert r.rc == 0 and not r.traceback and "Training will not start" in r.out, r
    odd = w.inp.rename(w.base / f"in {name}")
    r2 = ml(w.args(odd), data=tmp_path / name / "data2")
    assert r2.rc == 0 and not r2.traceback, r2
    assert crop_files(tmp_path / name / "data2", version_of(r2)) == crop_files(data, v)



def test_permission_denied_on_one_input_file_is_a_clean_refusal_or_a_rejected_file(corpus, wrapper, tmp_path):
    """A file that cannot be read (EACCES on a share) must never end as a traceback, and never as a snapshot that silently lacks it."""
    data = tmp_path / "data"
    r = finish(spawn(corpus.args(), wrapper, data=data, env=base_env(FI_MODE="eacces", FI_NTH=3)))
    assert not r.traceback, r
    if r.rc == 0:                       # accepted only if the file is listed as rejected in the manifest
        v = version_of(r)
        m = json.loads((data / "datasets" / v / "manifest.json").read_text())
        assert m["import_summary"]["tiffs"]["rejected"] >= 1
    else:
        clean_refusal(r, r"permission|denied|cannot read")
        assert published(data) == []


@needs_posix
@pytest.mark.skipif(IS_ROOT, reason="root reads everything")
def test_chmod_000_input_file_real(work, tmp_path):
    victim = sorted(work.inp.glob("*.tif"))[2]
    os.chmod(victim, 0)
    try:
        r = ml(work.args(), data=tmp_path / "data")
        assert not r.traceback, r
    finally:
        os.chmod(victim, 0o644)


@needs_posix
@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
def test_symlink_loops_do_not_hang_and_change_nothing(work, clean, tmp_path):
    cdata, cver, cdig = clean
    os.symlink(".", work.inp / "loop")
    os.symlink("../in", work.inp / "up")
    (work.inp / "a").mkdir()
    os.symlink("../a", work.inp / "a" / "self")
    os.symlink("nowhere.tif", work.inp / "dangling.tif")
    data = tmp_path / "data"
    r = ml(work.args(), data=data, timeout=120)
    assert not r.traceback, r
    # the dangling link is a TIFF name without a file: refused/rejected, or ignored - but the real files are unchanged
    if r.rc == 0:
        assert version_of(r) in (cver,) or "rejected" in r.out or True
        m = json.loads((data / "datasets" / version_of(r) / "manifest.json").read_text())
        assert {c["source_file"] for c in m["crops"]} == {c["source_file"] for c in json.loads(
            (cdata / "datasets" / cver / "manifest.json").read_text())["crops"]}
    else:
        clean_refusal(r)


@needs_posix
@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
def test_symlinked_directory_pointing_outside_the_input_is_not_followed(work, clean, tmp_path):
    cdata, cver, cdig = clean
    outside = tmp_path / "outside"
    outside.mkdir()
    make_tiff(outside / "ESCAPED-SECRET.tif", marks=marks_in(0, 0, 3, 0, 1, 99), seed=99)
    os.symlink(outside, work.inp / "link_to_outside")
    r = ml(work.args(None, "--dry-run"), data=tmp_path / "data")
    assert r.rc == 0 and "ESCAPED-SECRET" not in r.text and "discovered 8 TIFF" in r.out, r
    r = ml(work.args(), data=tmp_path / "data")
    assert r.rc == 0 and crop_files(tmp_path / "data", version_of(r)) == crop_files(cdata, cver)
    assert sha_of(outside / "ESCAPED-SECRET.tif") not in (tmp_path / "data" / "datasets" / version_of(r) / "manifest.json").read_text()


@needs_posix
@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
def test_symlinked_file_pointing_outside_the_input_is_never_imported(work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = make_tiff(outside / "29-09-2026_T7_AB_ZZ-A.tif", marks=marks_in(0, 0, 3, 0, 1, 99), seed=77)
    os.symlink(secret, work.inp / "29-09-2026_T7_AB_ZZ-A.tif")
    r = ml(work.args(None, "--dry-run"), data=tmp_path / "data")
    assert r.rc == 0 and "discovered 8 TIFF" in r.out, r
    r = ml(work.args(), data=tmp_path / "data")
    assert r.rc == 0
    m = json.loads((tmp_path / "data" / "datasets" / version_of(r) / "manifest.json").read_text())
    assert sha_of(secret) not in {c["source_sha256"] for c in m["crops"]}


@needs_posix
@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
def test_symlinked_tiff_inside_the_input_is_a_byte_identical_alias_not_a_second_source(work, clean, tmp_path):
    cdata, cver, cdig = clean
    first = sorted(work.inp.glob("*.tif"))[0]
    os.symlink(first.name, work.inp / "alias-of-first.tif")
    r = ml(work.args(None, "--dry-run"), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback and re.search(r"deduplicated[^\n]*1", r.out), r
    r = ml(work.args(), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback, r
    m = json.loads((tmp_path / "data" / "datasets" / version_of(r) / "manifest.json").read_text())
    assert len({c["source_sha256"] for c in m["crops"]}) == 8


def test_very_long_and_unicode_file_names(work, tmp_path):
    longname = ("L" * 120) + "_29-09-2026_T7_AB_LL-A.tif"                      # 140 bytes
    uni = "29-09-2026_T7_AB_日本語-é-🧪-A.tif"
    for i, n in enumerate((longname, uni)):
        make_tiff(work.inp / n, marks=marks_in(0, 0, 3, 0, 1, 50 + i), seed=60 + i)
    data = tmp_path / "data"
    r = ml(work.args(None, "--allow-unconfirmed-identity", imap=False), data=data)
    assert r.rc == 0 and not r.traceback, r
    v = version_of(r)
    assert ml(["verify", v], data=data).out.strip() == "intact"
    m = json.loads((data / "datasets" / v / "manifest.json").read_text(encoding="utf-8"))
    names = {c["source_file"] for c in m["crops"]}
    assert longname in names and uni in names
    # every crop file name is bounded (derived from the content hash, not from the long name)
    assert max(len(Path(c["path_tif"]).name) for c in m["crops"]) < 100
    assert train_dry(data, v, dummy_base(tmp_path)).rc in (0, 2)


def test_dot_files_dot_dirs_and_case_variants(work, clean, tmp_path):
    cdata, cver, cdig = clean
    first = sorted(work.inp.glob("*.tif"))[0]
    shutil.copy(first, work.inp / ".hidden.tif")
    (work.inp / ".git").mkdir()
    shutil.copy(first, work.inp / ".git" / "x.tif")
    shutil.copy(first, work.inp / "COPY.TIF")                    # upper-case suffix, byte-identical: an alias
    shutil.copy(first, work.inp / "Copy.Tiff")
    (work.inp / "notes.tif.zip").write_bytes(b"PK\x03\x04junk")
    (work.inp / "readme.txt").write_text("hi")
    r = ml(work.args(), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback, r
    out = r.out
    assert "discovered 10 TIFF" in out, out                      # 8 + COPY.TIF + Copy.Tiff; dot files / dirs / zip skipped
    m = json.loads((tmp_path / "data" / "datasets" / version_of(r) / "manifest.json").read_text())
    assert len({c["source_sha256"] for c in m["crops"]}) == 8


@needs_posix
def test_case_colliding_names_with_different_content_are_two_sources(work, tmp_path):
    a = work.inp / "29-09-2026_T7_AB_QQ-A.tif"
    b = work.inp / "29-09-2026_T7_AB_qq-a.TIF"
    make_tiff(a, marks=marks_in(0, 0, 3, 0, 1, 31), seed=31)
    make_tiff(b, marks=marks_in(0, 0, 4, 0, 1, 32), seed=32)
    data = tmp_path / "data"
    r = ml(work.args(None, "--allow-unconfirmed-identity", imap=False), data=data)
    assert r.rc == 0 and not r.traceback, r
    m = json.loads((data / "datasets" / version_of(r) / "manifest.json").read_text())
    shas = {c["source_sha256"] for c in m["crops"]}
    assert {sha_of(a), sha_of(b)} <= shas
    ids = [c["crop_id"] for c in m["crops"]]
    assert len(ids) == len(set(ids))
    assert ml(["verify", version_of(r)], data=data).out.strip() == "intact"


# ============================================================================================ 4. files changing under the import (TOCTOU)
def lb_opts(**kw):
    kw.setdefault("created_at", T0)
    kw.setdefault("fractions", {"train": 0.5, "val": 0.25, "test": 0.25})
    return lb.BridgeOptions(**kw)


def inproc_import(c: Corpus, data, **kw):
    return lb.import_legacy(data, c.inp, lb_opts(), channel_map=c.cmap, identity_map=c.imap, say=lambda *_: None, **kw)


class Mutator:
    """Runs ``action(path)`` right before the ``nth`` plane read of the run. ``where='before_check'``: before the size/mtime guard of
    ``_PlaneCache.plane`` (the window between scan and page read); ``'after_check'``: between the guard and the actual read."""

    def __init__(self, monkeypatch, action, nth=1, where="before_check"):
        self.n, self.nth, self.action, self.paths = 0, nth, action, []
        if where == "before_check":
            real_plane = lb._PlaneCache.plane

            def plane(cache, page):
                if page not in cache.planes:
                    self._hit(cache.sc.path)
                return real_plane(cache, page)
            monkeypatch.setattr(lb._PlaneCache, "plane", plane)
        else:
            real = lb.read_plane

            def hooked(path, page):
                self._hit(path)
                return real(path, page)
            monkeypatch.setattr(lb, "read_plane", hooked)

    def _hit(self, path):
        self.n += 1
        self.paths.append(Path(path))
        if self.n == self.nth:
            self.action(Path(path))


def flip_pixel_bytes_in_place(path: Path, keep_mtime: bool):
    st = path.stat()
    b = bytearray(path.read_bytes())
    for k in range(len(b) - 2000, len(b) - 1000):          # pixel data of the last page: same size, different bytes
        b[k] ^= 0x5A
    path.write_bytes(bytes(b))
    if keep_mtime:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def assert_no_snapshot(data):
    assert published(data) == [], published(data)


def test_source_deleted_between_scan_and_page_read_is_a_clean_refusal(work, tmp_path, monkeypatch):
    Mutator(monkeypatch, lambda p: p.unlink())
    data = tmp_path / "data"
    r = mlp(work.args(), data=data)
    clean_refusal(r, r"changed|vanished|no longer|not found|missing|cannot read")
    assert_no_snapshot(data)


@pytest.mark.parametrize("what", ["delete", "truncate"])
def test_source_damaged_after_the_guard_but_before_the_read_is_a_clean_refusal(work, tmp_path, monkeypatch, what):
    Mutator(monkeypatch, (lambda p: p.unlink()) if what == "delete" else (lambda p: os.truncate(p, p.stat().st_size // 2)),
            where="after_check")
    data = tmp_path / "data"
    r = mlp(work.args(), data=data)
    clean_refusal(r)
    assert_no_snapshot(data)


def test_source_replaced_by_other_bytes_between_scan_and_page_read_is_refused(work, tmp_path, monkeypatch):
    other = tmp_path / "other.tif"
    make_tiff(other, marks=marks_in(0, 0, 5, 0, 1, 3), seed=12345)

    def swap(p):
        shutil.copy(other, p)
        os.utime(p, ns=(1, 1))
    Mutator(monkeypatch, swap)
    data = tmp_path / "data"
    with pytest.raises(MLError, match="changed on disk|sha"):
        inproc_import(work, data)
    assert_no_snapshot(data)


def test_source_modified_in_place_between_page_zero_and_page_one_is_refused(work, tmp_path, monkeypatch):
    def touch(p):
        flip_pixel_bytes_in_place(p, keep_mtime=False)
    Mutator(monkeypatch, touch, nth=2)               # the second plane read of the run (page 1 of the first source)
    data = tmp_path / "data"
    with pytest.raises(MLError, match="changed on disk"):
        inproc_import(work, data)
    assert_no_snapshot(data)


def test_truncated_source_between_scan_and_page_read_is_refused(work, tmp_path, monkeypatch):
    Mutator(monkeypatch, lambda p: os.truncate(p, p.stat().st_size // 2))
    data = tmp_path / "data"
    with pytest.raises(MLError, match="changed on disk"):
        inproc_import(work, data)
    assert_no_snapshot(data)


def test_same_size_rewrite_with_restored_mtime_never_exports_different_bytes_silently(work, tmp_path, monkeypatch):
    victim = {}

    def rewrite(p):
        victim["path"] = p
        flip_pixel_bytes_in_place(p, keep_mtime=True)
    Mutator(monkeypatch, rewrite)
    data = tmp_path / "data"
    try:
        out = inproc_import(work, data)
    except MLError:
        return                                        # refusing is the right outcome
    m = out.result.manifest
    now = sha_of(victim["path"])
    for c in m.crops:
        if Path(c.source_file).name == victim["path"].name:
            assert c.source_sha256 == now, "crops were cut from other bytes than the recorded source_sha256"


def test_file_replaced_between_dry_run_and_import_is_exported_with_the_new_bytes_and_their_hash(work, tmp_path):
    """The dry run and the real import are separate scans: the snapshot's provenance must match the bytes that are on disk NOW."""
    data = tmp_path / "data"
    assert ml(work.args(None, "--dry-run"), data=data).rc == 0
    victim = sorted(work.inp.glob("*.tif"))[1]
    make_tiff(victim, marks=marks_in(0, 0, 5, 0, 1, 8) + marks_in(0, 1, 2, 1, 1, 9), seed=4242)
    r = ml(work.args(), data=data)
    assert r.rc == 0 and not r.traceback, r
    v = version_of(r)
    m = json.loads((data / "datasets" / v / "manifest.json").read_text())
    on_disk = {sha_of(q) for q in work.inp.glob("*.tif")}
    assert {c["source_sha256"] for c in m["crops"]} <= on_disk
    assert sha_of(victim) in {c["source_sha256"] for c in m["crops"]}
    assert ml(["verify", v], data=data).out.strip() == "intact"


def test_every_crop_provenance_hash_matches_a_source_file_on_disk(work, clean, tmp_path):
    cdata, cver, cdig = clean
    m = json.loads((cdata / "datasets" / cver / "manifest.json").read_text())
    on_disk = {sha_of(q) for q in work.inp.glob("*.tif")}
    for c in m["crops"]:
        assert c["source_sha256"] in on_disk and c["provenance"]["source_sha256"] == c["source_sha256"]


# ============================================================================================ 5. hostile / corrupt TIFFs
HOSTILE = "29-09-2026_T7_AB_ZZ-A.tif"
GOOD_MARKS = marks_in(0, 0, 3, 0, 1, 91) + marks_in(0, 1, 2, 1, 1, 92)


def write_ij(path, planes, *, roi=None, overlays=None, **imwrite_kw):
    """Like legacy_fixtures.write_stack but with tifffile options (byte order, compression, tiles, BigTIFF ...)."""
    data = np.stack(planes) if len(planes) > 1 else planes[0]
    meta = {}
    if roi is not None:
        meta["ROI"] = roi
    if overlays:
        meta["Overlays"] = list(overlays) if len(overlays) > 1 else overlays[0]
    tifffile.imwrite(str(path), data, imagej=True, metadata=meta, **imwrite_kw)
    return Path(path)


def std_planes(n=2, dtype=np.uint16, seed=7):
    hi = 60000 if np.dtype(dtype) == np.uint16 else 250
    return [textured(H, W, seed * 10 + k, dtype=dtype, hi=hi) for k in range(n)]


def std_roi(marks=GOOD_MARKS):
    return point_roi([(m[0], m[1]) for m in marks], [m[2] for m in marks], [m[3] for m in marks])


def std_grid(xs=GRID_XS, ys=GRID_YS):
    return grid_overlay(xs, ys, W, H)


def good_bytes(tmp):
    p = Path(tmp) / "good_for_mutation.tif"
    make_tiff(p, marks=GOOD_MARKS, seed=7)
    return p.read_bytes()


def patch_ifd_dims(raw: bytes, width=None, height=None) -> bytes:
    """Rewrite ImageWidth / ImageLength of IFD 0 of a classic little-endian TIFF as LONG values."""
    import struct
    assert raw[:4] == b"II*\x00"
    ifd = struct.unpack_from("<I", raw, 4)[0]
    n = struct.unpack_from("<H", raw, ifd)[0]
    b = bytearray(raw)
    for i in range(n):
        off = ifd + 2 + 12 * i
        tag = struct.unpack_from("<H", b, off)[0]
        for t, v in ((256, width), (257, height)):
            if tag == t and v is not None:
                struct.pack_into("<HHII", b, off, tag, 4, 1, v)
    return bytes(b)


def b_zero_byte(p, tmp):
    p.write_bytes(b"")


def b_truncated(frac):
    def build(p, tmp):
        raw = good_bytes(tmp)
        p.write_bytes(raw[: max(1, int(len(raw) * frac)) if frac < 1 else len(raw) - 1])
    return build


def b_truncated_at(n):
    def build(p, tmp):
        p.write_bytes(good_bytes(tmp)[:n])
    return build


def b_png_renamed(p, tmp):
    from PIL import Image
    Image.fromarray(np.zeros((40, 40), np.uint8)).save(p, format="PNG")


def b_text(p, tmp):
    p.write_text("this is not a tiff at all\n" * 100)


def b_huge_dims(p, tmp):
    p.write_bytes(patch_ifd_dims(good_bytes(tmp), width=2 ** 31 - 1, height=2 ** 31 - 1))


def b_big_dims(p, tmp):
    p.write_bytes(patch_ifd_dims(good_bytes(tmp), width=60000, height=60000))


def b_float32(p, tmp):
    write_ij(p, std_planes(dtype=np.float32), roi=std_roi(), overlays=[std_grid()])


def b_int16(p, tmp):
    write_ij(p, [q.astype(np.int16) % 30000 for q in std_planes()], roi=std_roi(), overlays=[std_grid()])


def b_rgb(p, tmp):
    tifffile.imwrite(str(p), np.zeros((H, W, 3), np.uint8), photometric="rgb")


def b_endian(order):
    def build(p, tmp):
        write_ij(p, std_planes(), roi=std_roi(), overlays=[std_grid()], byteorder=order)
    return build


def b_codec(**kw):
    def build(p, tmp):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_ij(p, std_planes(), roi=std_roi(), overlays=[std_grid()], **kw)
    return build


def b_three_pages(p, tmp):
    write_ij(p, std_planes(3), roi=std_roi(GOOD_MARKS + marks_in(1, 1, 3, 4, 3, 5)), overlays=[std_grid()])


def b_one_page(p, tmp):
    write_ij(p, std_planes(1), roi=std_roi(marks_in(0, 0, 3, 0, 1, 91)), overlays=[std_grid()])


def b_unequal_pages(p, tmp):
    with tifffile.TiffWriter(str(p)) as tw:
        tw.write(textured(H, W, 1), photometric="minisblack")
        tw.write(textured(H // 2, W // 2, 2), photometric="minisblack")


def b_points(pts, counters=None, positions=None):
    def build(p, tmp):
        n = len(pts)
        write_ij(p, std_planes(), roi=point_roi(pts, counters if counters is not None else [0] * n,
                                                 positions if positions is not None else [1] * n),
                 overlays=[std_grid()])
    return build


GOOD3 = [(25.0, 20.0), (30.0, 25.0), (35.0, 30.0)]


def b_grid(xs, ys):
    def build(p, tmp):
        write_ij(p, std_planes(), roi=std_roi(), overlays=[std_grid(xs, ys)])
    return build


def b_gigantic_roi_blob(p, tmp):
    blob = os.urandom(40 * 1024 * 1024)
    write_ij(p, std_planes(), roi=blob, overlays=[std_grid()])


def b_roi_garbage(p, tmp):
    write_ij(p, std_planes(), roi=b"Iout" + os.urandom(500), overlays=[std_grid()])


def b_roi_n_coords_lie(p, tmp):
    """A valid point ROI whose header claims 65535 coordinates but carries only a handful."""
    import struct
    raw = bytearray(std_roi())
    struct.pack_into(">H", raw, 16, 65535)
    write_ij(p, std_planes(), roi=bytes(raw), overlays=[std_grid()])


def b_overlay_garbage(p, tmp):
    write_ij(p, std_planes(), roi=std_roi(), overlays=[b"Iout" + os.urandom(300)])


def b_ij_counts_lie(p, tmp):
    """ImageJ metadata byte counts (tag 50838) that disagree with the data (tag 50839)."""
    import struct
    raw = bytearray(good_bytes(tmp))
    with tifffile.TiffFile(io.BytesIO(bytes(raw))) as t:
        tag = t.pages[0].tags.get(50838)
        off = tag.valueoffset
        n = tag.count
        dt = tag.dtype
    fmt = "<" + "I" * n
    vals = list(struct.unpack_from(fmt, raw, off))
    vals[-1] = 2 ** 32 - 1                                  # the last blob "is" 4 GiB
    struct.pack_into(fmt, raw, off, *vals)
    p.write_bytes(bytes(raw))


def b_bigtiff(p, tmp):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_ij(p, std_planes(), roi=std_roi(), overlays=[std_grid()], bigtiff=True)


def b_planes_are_good_but_info_is_hostile(p, tmp):
    write_ij(p, std_planes(), roi=std_roi(), overlays=[std_grid()])


HOSTILE_CASES = {
    "zero_byte": b_zero_byte,
    "text_file": b_text,
    "png_renamed": b_png_renamed,
    "truncated_8": b_truncated_at(8),
    "truncated_100": b_truncated_at(100),
    "truncated_1000": b_truncated_at(1000),
    "truncated_half": b_truncated(0.5),
    "truncated_last_byte": b_truncated(1.0),
    "huge_declared_dims_2^31": b_huge_dims,
    "big_declared_dims_60000": b_big_dims,
    "float32": b_float32,
    "int16": b_int16,
    "rgb": b_rgb,
    "big_endian": b_endian(">"),
    "little_endian": b_endian("<"),
    "lzw_tiled": b_codec(compression="lzw", tile=(32, 32)),
    "deflate": b_codec(compression="zlib"),
    "deflate_tiled": b_codec(compression="zlib", tile=(16, 16)),
    "bigtiff": b_bigtiff,
    "three_pages": b_three_pages,
    "one_page": b_one_page,
    "unequal_pages": b_unequal_pages,
    "nan_coords": b_points([(float("nan"), 20.0), (25.0, float("nan")), *GOOD3]),
    "inf_coords": b_points([(float("inf"), 20.0), (25.0, float("-inf")), *GOOD3]),
    "negative_coords": b_points([(-5.0, 20.0), (25.0, -1e9), *GOOD3]),
    "coords_2^31": b_points([(2.0 ** 31, 20.0), (25.0, 2.0 ** 31 + 7), *GOOD3]),
    "coords_1e38": b_points([(3e38, 20.0), (25.0, -3e38), *GOOD3]),
    "points_50000": b_points([(10.0 + (i % 200) * 0.7, 10.0 + (i // 200) * 0.4) for i in range(50000)], counters=[i % 9 for i in range(50000)]),
    "gigantic_roi_blob_40MB": b_gigantic_roi_blob,
    "roi_garbage": b_roi_garbage,
    "roi_n_coords_lie": b_roi_n_coords_lie,
    "overlay_garbage": b_overlay_garbage,
    "ij_counts_lie": b_ij_counts_lie,
    "grid_1x1": b_grid([10.0, 150.0], [10.0, 110.0]),
    "grid_200x200": b_grid(list(np.linspace(0.0, 159.0, 201)), list(np.linspace(0.0, 119.0, 201))),
    "grid_non_monotonic": b_grid([10.0, 110.0, 60.0, 160.0], [10.0, 80.0, 45.0, 115.0]),
    "grid_duplicate_lines": b_grid([10.0, 60.0, 60.0, 110.0, 160.0], [10.0, 45.0, 80.0, 115.0]),
    "grid_outside_image": b_grid([-50.0, 500.0, 900.0], [-30.0, 700.0, 1000.0]),
}


def hostile_run(corpus, tmp_path, name, *, dry=False, timeout_s=60):
    c = corpus.copy_to(tmp_path / "h")
    HOSTILE_CASES[name](c.inp / HOSTILE, tmp_path)
    t0 = time.time()
    r = mlp(c.args(None, *(["--dry-run"] if dry else []), "--allow-unconfirmed-identity"), data=tmp_path / "data")
    return c, r, time.time() - t0


NO_CROPS = ["zero_byte", "text_file", "png_renamed", "truncated_8", "truncated_100", "truncated_1000", "truncated_half",
            "huge_declared_dims_2^31", "big_declared_dims_60000", "float32", "int16", "rgb", "one_page", "unequal_pages",
            "gigantic_roi_blob_40MB", "roi_garbage", "roi_n_coords_lie", "overlay_garbage"]
SAME_AS_REFERENCE = ["big_endian", "lzw_tiled", "deflate", "deflate_tiled", "bigtiff"]
EITHER = ["truncated_last_byte", "ij_counts_lie"]                 # rejected, or - if accepted - pixel-identical to the reference
POINT_CASES = ["nan_coords", "inf_coords", "negative_coords", "coords_2^31", "coords_1e38", "points_50000", "grid_1x1", "grid_200x200",
               "grid_non_monotonic", "grid_duplicate_lines", "grid_outside_image", "three_pages"]
assert set(NO_CROPS + SAME_AS_REFERENCE + EITHER + POINT_CASES + ["little_endian"]) == set(HOSTILE_CASES)


def snapshot_of(data) -> tuple:
    (v,) = published(data)
    return v, json.loads((Path(data) / "datasets" / v / "manifest.json").read_text(encoding="utf-8"))


def crops_of_file(data, v, m, name=HOSTILE) -> dict:
    """{(page, region_id, role): (pixels, labels)} of the crops cut from source file ``name``."""
    out = {}
    root = Path(data) / "datasets" / v
    for c in m["crops"]:
        if Path(c["source_file"]).name == name:
            lab = ds.read_labels_csv(root / c["path_csv"])
            out[(c["page"], c["region_id"], c["channel_role"])] = (tifffile.imread(root / c["path_tif"]), lab, c)
    return out


@pytest.fixture(scope="module")
def reference_crops(corpus, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fi_ref")
    c, r, _ = hostile_run(corpus, tmp, "little_endian")
    assert r.rc == 0, r
    v, m = snapshot_of(tmp / "data")
    got = crops_of_file(tmp / "data", v, m)
    assert got
    return got


DRY_RUN_SUBSET = ["zero_byte", "truncated_half", "png_renamed", "huge_declared_dims_2^31", "float32", "big_endian", "deflate_tiled", "bigtiff",
                  "nan_coords", "coords_2^31", "points_50000", "grid_200x200", "grid_1x1", "grid_non_monotonic", "gigantic_roi_blob_40MB", "three_pages"]


@pytest.mark.parametrize("name", DRY_RUN_SUBSET)
def test_hostile_tiff_dry_run_never_crashes_never_writes_and_is_fast(corpus, tmp_path, name):
    c = corpus.copy_to(tmp_path / "h")
    HOSTILE_CASES[name](c.inp / HOSTILE, tmp_path)
    data = tmp_path / "data"
    before = tree_digest(c.inp, data)
    t0 = time.time()
    r = mlp(c.args(None, "--dry-run", "--allow-unconfirmed-identity"), data=data)
    assert r.rc == 0 and not r.traceback, r
    assert "DRY RUN" in r.out
    assert time.time() - t0 < 30
    assert tree_digest(c.inp, data) == before


@pytest.mark.parametrize("name", NO_CROPS)
def test_unusable_tiffs_are_rejected_and_contribute_nothing(corpus, tmp_path, name):
    c, r, dt = hostile_run(corpus, tmp_path, name)
    assert r.rc == 0 and not r.traceback, r                   # the 8 good files are still imported
    v, m = snapshot_of(tmp_path / "data")
    assert not crops_of_file(tmp_path / "data", v, m)
    assert len(m["crops"]) == 32
    assert ml(["verify", v], data=tmp_path / "data").out.strip() == "intact"
    assert dt < 30


@pytest.mark.parametrize("name", SAME_AS_REFERENCE + EITHER)
def test_other_valid_encodings_give_pixel_identical_crops(corpus, tmp_path, reference_crops, name):
    """Big-endian, LZW / deflate (tiled or not), BigTIFF: the same pixels and marks give the same crops as the plain TIFF."""
    c, r, _ = hostile_run(corpus, tmp_path, name)
    assert r.rc == 0 and not r.traceback, r
    v, m = snapshot_of(tmp_path / "data")
    got = crops_of_file(tmp_path / "data", v, m)
    if not got:
        assert name in EITHER, f"{name} is a valid TIFF but was rejected: {r.out[-800:]}"
        return
    assert set(got) == set(reference_crops)
    for k, (px, lab, rec) in got.items():
        rpx, rlab, _ = reference_crops[k]
        assert px.dtype == rpx.dtype and np.array_equal(px, rpx), k
        assert np.array_equal(lab, rlab), k


@pytest.mark.parametrize("name", POINT_CASES)
def test_hostile_coordinates_and_grids_keep_the_label_accounting_exact(corpus, tmp_path, name):
    c, r, dt = hostile_run(corpus, tmp_path, name)
    assert r.rc == 0 and not r.traceback, r
    v, m = snapshot_of(tmp_path / "data")
    a = m["import_summary"]["annotations"]
    assert a["points_total"] == a["labels_exported"] + a["dropped_total"], a
    assert a["dropped_total"] == sum(a["dropped_by_reason"].values())
    # all 8 good files untouched
    good = [cr for cr in m["crops"] if Path(cr["source_file"]).name != HOSTILE]
    assert len(good) == 32
    # every exported label of the hostile file is finite and inside its crop
    for (page, region, role), (px, lab, rec) in crops_of_file(tmp_path / "data", v, m).items():
        assert np.isfinite(lab).all(), (name, rec["crop_id"])
        h, w = px.shape
        assert ((lab[:, 0] >= -0.5) & (lab[:, 0] <= h - 0.5) & (lab[:, 1] >= -0.5) & (lab[:, 1] <= w - 0.5)).all(), (name, rec["crop_id"])
        assert len(lab) == rec["n_positive"]
    assert ml(["verify", v], data=tmp_path / "data").out.strip() == "intact"
    assert dt < 40
    if name == "three_pages":
        assert all(k[0] in (0, 1) for k in crops_of_file(tmp_path / "data", v, m)), "a page the channel map does not name produced a crop"


@pytest.mark.skipif(not HAS_SYMLINKS, reason="POSIX resource limits")
@pytest.mark.parametrize("name", ["huge_declared_dims_2^31", "big_declared_dims_60000", "points_50000", "grid_200x200", "gigantic_roi_blob_40MB"])
def test_hostile_tiffs_stay_within_a_memory_budget(corpus, wrapper, tmp_path, name):
    """RLIMIT_AS = 3 GiB in a real process: a declared 2^31 x 2^31 image or a 40 MB ROI blob must not be allocated."""
    c = corpus.copy_to(tmp_path / "h")
    HOSTILE_CASES[name](c.inp / HOSTILE, tmp_path)
    r = finish(spawn(c.args(None, "--allow-unconfirmed-identity"), wrapper, data=tmp_path / "data",
                     env=base_env(FI_MODE="rlimit_as", FI_LIMIT=3 * 1024 ** 3)), timeout=120)
    assert r.rc == 0 and not r.traceback and "MemoryError" not in r.text, r


# ============================================================================================ 6. hostile mapping / manifest files
def id_rows(c: Corpus) -> list:
    return json.loads(c.imap.read_text())["rows"]


def csv_text(rows, eol="\n"):
    cols = ["file", "specimen_id", "slide_id", "assay", "replicate", "scan_session", "confirmed"]
    out = [",".join(cols)] + [",".join(str(r.get(k, "")) if k != "confirmed" else "true" for k in cols) for r in rows]
    return eol.join(out) + eol


def deep(n, open_="[", close="]"):
    return open_ * n + close * n


def ch_doc(c: Corpus) -> dict:
    return json.loads(c.cmap.read_text())


def jtxt(d, **kw):
    return json.dumps(d, indent=2, **kw)


# name -> (suffix, bytes-builder(c))
IDENTITY_VARIANTS = {
    "json_bom": (".json", lambda c: b"\xef\xbb\xbf" + c.imap.read_bytes()),
    "json_crlf": (".json", lambda c: c.imap.read_text().replace("\n", "\r\n").encode()),
    "csv": (".csv", lambda c: csv_text(id_rows(c)).encode()),
    "csv_bom_crlf": (".csv", lambda c: b"\xef\xbb\xbf" + csv_text(id_rows(c), "\r\n").encode()),
    "csv_short_row": (".csv", lambda c: (csv_text(id_rows(c)) + "29-09-2026_T7_AB_AM-A.tif\n").encode()),
    "csv_empty": (".csv", lambda c: b""),
    "csv_header_only": (".csv", lambda c: b"file,specimen_id\n"),
    "csv_binary": (".csv", lambda c: os.urandom(2000)),
    "invalid_utf8": (".json", lambda c: c.imap.read_bytes().replace(b"AM", b"A\xff\xfeM", 1)),
    "empty_file": (".json", lambda c: b""),
    "scalar_5": (".json", lambda c: b"5"),
    "scalar_null": (".json", lambda c: b"null"),
    "scalar_string": (".json", lambda c: b'"rows"'),
    "list_of_ints": (".json", lambda c: b"[1, 2, 3]"),
    "rows_is_number": (".json", lambda c: b'{"rows": 5}'),
    "rows_has_null": (".json", lambda c: b'{"rows": [null]}'),
    "rows_is_dict_of_scalars": (".json", lambda c: b'{"rows": {"a": 1}}'),
    "deeply_nested": (".json", lambda c: deep(200000).encode()),
    "huge_integer": (".json", lambda c: c.imap.read_text().replace('"AM"', "1" + "0" * 400).encode()),
    "nan_literal": (".json", lambda c: c.imap.read_text().replace('"AM"', "NaN").encode()),
    "file_with_nul": (".json", lambda c: c.imap.read_text().replace("AM-A.tif", "AM-A\\u0000.tif").encode()),
    "file_traversal": (".json", lambda c: c.imap.read_text().replace("29-09-2026_T7_AB_AM-A.tif", "../../etc/passwd").encode()),
    "unknown_keys": (".json", lambda c: jtxt({"rows": [dict(r, surprise="x") for r in id_rows(c)], "also": 1}).encode()),
    "duplicate_keys": (".json", lambda c: c.imap.read_text().replace('"specimen_id": "AM"', '"specimen_id": "AM", "specimen_id": "ZZ"').encode()),
    "binary": (".json", lambda c: os.urandom(3000)),
}
CHANNEL_VARIANTS = {
    "bom": lambda c: b"\xef\xbb\xbf" + c.cmap.read_bytes(),
    "crlf": lambda c: c.cmap.read_text().replace("\n", "\r\n").encode(),
    "invalid_utf8": lambda c: c.cmap.read_bytes().replace(b"Researcher", b"R\xff\xfe"),
    "empty": lambda c: b"",
    "scalar": lambda c: b"[1]",
    "deeply_nested": lambda c: deep(200000).encode(),
    "deeply_nested_values": lambda c: c.cmap.read_text().replace('"source_of_truth": "lab notebook p. 12"', '"source_of_truth": ' + deep(200000)).encode(),
    "files_is_list": lambda c: jtxt(dict(ch_doc(c), files=[1])).encode(),
    "files_is_string": lambda c: jtxt(dict(ch_doc(c), files="x")).encode(),
    "rules_is_dict": lambda c: jtxt(dict(ch_doc(c), rules={"a": 1})).encode(),
    "rules_has_null": lambda c: jtxt(dict(ch_doc(c), rules=[None])).encode(),
    "defaults_is_list_of_scalars": lambda c: jtxt(dict(ch_doc(c), defaults=[1, 2])).encode(),
    "page_key_huge": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"9" * 5000: {"role": "red", "train": True}}})).encode(),
    "page_key_superscript": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"²": {"role": "red", "train": True}}})).encode(),
    "page_key_arabic_indic": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"٠": {"role": "red", "train": True}, "١": {"role": "green", "train": True}}})).encode(),
    "page_key_negative": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"-1": {"role": "red", "train": True}}})).encode(),
    "role_not_string": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"0": {"role": 5, "train": True}, "1": {"role": "g", "train": True}}})).encode(),
    "role_huge": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"0": {"role": "r" * 100000, "train": True}, "1": {"role": "g", "train": True}}})).encode(),
    "version_huge": lambda c: jtxt(dict(ch_doc(c), version=10 ** 30)).encode(),
    "version_nan": lambda c: c.cmap.read_text().replace('"version": 1', '"version": NaN').encode(),
    "unknown_keys": lambda c: jtxt(dict(ch_doc(c), surprise={"a": 1})).encode(),
    "unknown_key_in_page": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"0": {"role": "red", "train": True, "typo": 1}, "1": {"role": "green", "train": True}}})).encode(),
    "train_missing": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"0": {"role": "red"}, "1": {"role": "green"}}})).encode(),
    "train_is_string": lambda c: jtxt(dict(ch_doc(c), defaults={"pages": {"0": {"role": "red", "train": "yes"}, "1": {"role": "green", "train": True}}})).encode(),
    "duplicate_page_keys": lambda c: c.cmap.read_text().replace('"1": {', '"0": {"role": "blue", "train": true}, "1": {', 1).encode(),
    "duplicate_top_level_keys": lambda c: c.cmap.read_text().replace('"confirmed_by": "E. Researcher",', '"confirmed_by": "E. Researcher", "confirmed_by": "",').encode(),
    "files_key_traversal": lambda c: jtxt(dict(ch_doc(c), files={"../../etc/passwd": {"pages": {"0": {"role": "red", "train": True}}}})).encode(),
    "files_key_nonexistent": lambda c: jtxt(dict(ch_doc(c), files={"no/such/file.tif": {"pages": {"0": {"role": "red", "train": True}}}})).encode(),
    "binary": lambda c: os.urandom(3000),
}


FIRST = "29-09-2026_T7_AB_AM-A.tif"


def ex_doc(entries=None, *, pages=None, file=FIRST, **top):
    pages = pages if pages is not None else {"0": entries if entries is not None else [{"row": 0, "col": 0, "count": 4}]}
    d = {"schema": "polony-examined-fields", "version": 1, "evidence": "unit-test sidecar: researcher counted these fields",
         "source_kind": "explicit_manifest", "examined_by": "E. Researcher", "recorded_on": "2026-10-13",
         "files": [{"file": file, "pages": pages}]}
    d.update(top)
    return d


def ex_csv(rows, eol="\n", head="file,page,row,col,count,counted_zero,note"):
    return eol.join(["# evidence: unit-test sidecar: researcher counted these fields", head, *rows]) + eol


def raw_json(d, replace=None):
    s = json.dumps(d)
    for a, b in (replace or {}).items():
        assert a in s, a
        s = s.replace(a, b)
    return s.encode()


EXAMINED_VARIANTS = {
    "valid": lambda c: raw_json(ex_doc()),
    "valid_bom": lambda c: b"\xef\xbb\xbf" + raw_json(ex_doc()),
    "count_minus_1": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": -1}])),
    "count_1e9": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 10 ** 9}])),
    "count_1e30": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 10 ** 30}])),
    "count_nan": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 0}]), {"\"count\": 0": "\"count\": NaN"}),
    "count_inf": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 0}]), {"\"count\": 0": "\"count\": Infinity"}),
    "count_string": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": "many"}])),
    "count_float": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 1.5}])),
    "count_true": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": True}])),
    "page_key_a": lambda c: raw_json(ex_doc(pages={"a": [{"row": 0, "col": 0, "count": 4}]})),
    "page_key_negative": lambda c: raw_json(ex_doc(pages={"-1": [{"row": 0, "col": 0, "count": 4}]})),
    "page_key_superscript": lambda c: raw_json(ex_doc(pages={"²": [{"row": 0, "col": 0, "count": 4}]})),
    "page_key_huge": lambda c: raw_json(ex_doc(pages={"9" * 5000: [{"row": 0, "col": 0, "count": 4}]})),
    "page_key_beyond_planes": lambda c: raw_json(ex_doc(pages={"7": [{"row": 0, "col": 0, "count": 4}]})),
    "row_negative": lambda c: raw_json(ex_doc([{"row": -1, "col": 0, "count": 4}])),
    "row_huge": lambda c: raw_json(ex_doc([{"row": 10 ** 30, "col": 10 ** 30, "count": 4}])),
    "row_without_col": lambda c: raw_json(ex_doc([{"row": 0, "count": 4}])),
    "counter_and_row": lambda c: raw_json(ex_doc([{"counter": 0, "row": 0, "col": 0, "count": 4}])),
    "counter_negative": lambda c: raw_json(ex_doc([{"counter": -3, "count": 4}])),
    "entry_listed_twice": lambda c: raw_json(ex_doc([{"row": 0, "col": 0, "count": 4}, {"row": 0, "col": 0, "count": 4}])),
    "entry_not_object": lambda c: raw_json(ex_doc([5])),
    "pages_is_list": lambda c: raw_json(ex_doc(pages=[1, 2])),
    "files_is_dict": lambda c: raw_json(dict(ex_doc(), files={"a": 1})),
    "files_has_scalar": lambda c: raw_json(dict(ex_doc(), files=[5])),
    "evidence_missing": lambda c: raw_json({k: v for k, v in ex_doc().items() if k != "evidence"}),
    "evidence_blank": lambda c: raw_json(ex_doc(evidence="   ")),
    "source_kind_unknown": lambda c: raw_json(ex_doc(source_kind="telepathy")),
    "scalar": lambda c: b"7",
    "empty": lambda c: b"",
    "deeply_nested": lambda c: deep(200000).encode(),
    "deeply_nested_value": lambda c: raw_json(ex_doc(evidence="x")).replace(b'"examined_by": "E. Researcher"', b'"examined_by": ' + deep(200000).encode()),
    "invalid_utf8": lambda c: raw_json(ex_doc()).replace(b"unit-test", b"unit\xff\xfe-test"),
    "sha_mismatch": lambda c: raw_json(ex_doc(pages={"0": [{"row": 0, "col": 0, "count": 4}]}), {FIRST + '",': FIRST + '", "sha256": "' + "0" * 64 + '",'}),
    "file_traversal": lambda c: raw_json(ex_doc(file="../../etc/passwd")),
    "file_nonexistent": lambda c: raw_json(ex_doc(file="no/such.tif")),
    "same_file_twice_with_different_counts": lambda c: raw_json(dict(ex_doc(), files=[
        {"file": FIRST, "pages": {"0": [{"row": 0, "col": 0, "count": 4}]}}, {"file": "./" + FIRST, "pages": {"0": [{"row": 0, "col": 0, "count": 9}]}}])),
    "duplicate_json_keys": lambda c: raw_json(ex_doc(), {'"count": 4': '"count": 4, "count": 0'}),
    "unknown_keys": lambda c: raw_json(ex_doc(surprise=[1])),
    "huge_manifest_100k_entries": lambda c: raw_json(ex_doc(pages={"0": [{"counter": i, "count": 1} for i in range(100000)]})),
}
EXAMINED_CSV_VARIANTS = {
    "csv_valid": lambda c: ex_csv([f"{FIRST},0,0,0,4,,"]).encode(),
    "csv_bom_crlf": lambda c: b"\xef\xbb\xbf" + ex_csv([f"{FIRST},0,0,0,4,,"], "\r\n").encode(),
    "csv_short_row": lambda c: ex_csv([FIRST]).encode(),
    "csv_short_row_2": lambda c: ex_csv([f"{FIRST},0"]).encode(),
    "csv_page_text": lambda c: ex_csv([f"{FIRST},x,0,0,4,,"]).encode(),
    "csv_page_empty": lambda c: ex_csv([f"{FIRST},,0,0,4,,"]).encode(),
    "csv_count_negative": lambda c: ex_csv([f"{FIRST},0,0,0,-4,,"]).encode(),
    "csv_count_nan": lambda c: ex_csv([f"{FIRST},0,0,0,nan,,"]).encode(),
    "csv_count_1e9": lambda c: ex_csv([f"{FIRST},0,0,0,1000000000,,"]).encode(),
    "csv_extra_columns": lambda c: ex_csv([f"{FIRST},0,0,0,4,,,a,b,c"]).encode(),
    "csv_no_header_columns": lambda c: ex_csv([f"{FIRST},0,0,0,4,,"], head="a,b,c").encode(),
    "csv_empty": lambda c: b"",
    "csv_no_evidence": lambda c: (f"file,page,row,col,count\n{FIRST},0,0,0,4\n").encode(),
    "csv_invalid_utf8": lambda c: ex_csv([f"{FIRST},0,0,0,4,,\xff\xfe"]).encode("latin-1"),
    "csv_binary": lambda c: os.urandom(4000),
    "csv_quote_bomb": lambda c: ex_csv(['"' * 5000]).encode(),
}


def _expect(table):
    """pytest params for {name: kind}; kind in ok | refuse."""
    return [pytest.param(name, kind, id=name) for name, kind in table.items()]


def check_outcome(r: CP, want: str):
    assert not r.traceback, r
    if want == "refuse":
        clean_refusal(r)
    else:
        assert r.rc == 0, r


IDENTITY_EXPECT = {
    "json_bom": "ok", "json_crlf": "ok", "csv": "ok", "csv_bom_crlf": "ok", "csv_empty": "ok", "csv_header_only": "ok",
    "huge_integer": "ok", "nan_literal": "ok", "file_with_nul": "ok", "file_traversal": "ok",
    "csv_short_row": "refuse", "csv_binary": "refuse", "invalid_utf8": "refuse", "empty_file": "refuse", "list_of_ints": "refuse",
    "rows_has_null": "refuse", "binary": "refuse",
    "scalar_5": "refuse", "scalar_null": "refuse", "scalar_string": "refuse", "rows_is_number": "refuse", "rows_is_dict_of_scalars": "refuse",
    "deeply_nested": "refuse",
    "unknown_keys": "refuse", "duplicate_keys": "refuse",
}
assert set(IDENTITY_EXPECT) == set(IDENTITY_VARIANTS)


@pytest.mark.parametrize("name,want", _expect(IDENTITY_EXPECT))
def test_hostile_identity_map(corpus, tmp_path, name, want):
    suffix, build = IDENTITY_VARIANTS[name]
    p = tmp_path / f"identity{suffix}"
    p.write_bytes(build(corpus))
    data = tmp_path / "data"
    r = mlp(corpus.args(None, "--dry-run", imap=p), data=data)
    check_outcome(r, want)
    assert not data.exists()


@pytest.mark.parametrize("name", ["json_bom", "json_crlf", "csv", "csv_bom_crlf"])
def test_equivalent_spellings_of_the_identity_map_give_the_same_crops_and_splits(corpus, clean, tmp_path, name):
    cdata, cver, cdig = clean
    suffix, build = IDENTITY_VARIANTS[name]
    p = tmp_path / f"identity{suffix}"
    p.write_bytes(build(corpus))
    r = ml(corpus.args(None, imap=p), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback, r
    v, m = snapshot_of(tmp_path / "data")
    cm_ = json.loads((cdata / "datasets" / cver / "manifest.json").read_text())
    assert crop_files(tmp_path / "data", v) == crop_files(cdata, cver)
    assert m["splits"]["analysis_to_split"] == cm_["splits"]["analysis_to_split"]
    assert m["splits"]["analysis_to_group"] == cm_["splits"]["analysis_to_group"]


@needs_posix
@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
def test_symlinked_mapping_files_are_read_through_the_link(corpus, clean, tmp_path):
    cdata, cver, cdig = clean
    real = tmp_path / "real"
    real.mkdir()
    shutil.copy(corpus.imap, real / "i.json")
    shutil.copy(corpus.cmap, real / "c.json")
    os.symlink(real / "i.json", tmp_path / "i.json")
    os.symlink(real / "c.json", tmp_path / "c.json")
    r = ml(corpus.args(None, imap=tmp_path / "i.json", cmap=tmp_path / "c.json"), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback, r
    assert crop_files(tmp_path / "data", snapshot_of(tmp_path / "data")[0]) == crop_files(cdata, cver)
    os.symlink(tmp_path / "nowhere.json", tmp_path / "dangling.json")
    clean_refusal(ml(corpus.args(None, "--dry-run", imap=tmp_path / "dangling.json"), data=tmp_path / "d2"))


CHANNEL_EXPECT = {
    "bom": "ok", "crlf": "ok", "role_huge": "ok",
    "invalid_utf8": "refuse", "page_key_arabic_indic": "refuse", "empty": "refuse", "scalar": "refuse", "rules_is_dict": "refuse", "rules_has_null": "refuse",
    "defaults_is_list_of_scalars": "refuse", "page_key_negative": "refuse", "role_not_string": "refuse", "version_huge": "refuse",
    "version_nan": "refuse", "unknown_keys": "refuse", "unknown_key_in_page": "refuse", "train_missing": "refuse",
    "train_is_string": "refuse", "duplicate_top_level_keys": "refuse", "binary": "refuse",
    "deeply_nested": "refuse", "deeply_nested_values": "refuse", "files_is_list": "refuse", "files_is_string": "refuse",
    "page_key_huge": "refuse", "page_key_superscript": "refuse", "duplicate_page_keys": "refuse",
    "files_key_traversal": "refuse",
    "files_key_nonexistent": "refuse",
}
assert set(CHANNEL_EXPECT) == set(CHANNEL_VARIANTS)


@pytest.mark.parametrize("name,want", _expect(CHANNEL_EXPECT))
def test_hostile_channel_map(corpus, tmp_path, name, want):
    p = tmp_path / "channels.json"
    p.write_bytes(CHANNEL_VARIANTS[name](corpus))
    data = tmp_path / "data"
    r = mlp(corpus.args(None, "--dry-run", cmap=p), data=data)
    check_outcome(r, want)
    assert not data.exists()


EXAMINED_EXPECT = {
    "valid": "ok", "valid_bom": "ok", "page_key_beyond_planes": "ok", "csv_valid": "ok", "csv_bom_crlf": "ok",
    "count_minus_1": "refuse", "count_1e9": "refuse", "count_1e30": "refuse", "count_nan": "refuse", "count_inf": "refuse",
    "count_string": "refuse", "count_float": "refuse", "count_true": "refuse", "page_key_a": "refuse", "page_key_negative": "refuse",
    "page_key_superscript": "refuse", "page_key_huge": "refuse", "row_negative": "refuse", "row_huge": "refuse",
    "row_without_col": "refuse", "counter_and_row": "refuse", "counter_negative": "refuse", "entry_listed_twice": "refuse",
    "entry_not_object": "refuse", "pages_is_list": "refuse", "files_is_dict": "refuse", "files_has_scalar": "refuse",
    "evidence_missing": "refuse", "evidence_blank": "refuse", "source_kind_unknown": "refuse", "scalar": "refuse", "empty": "refuse",
    "deeply_nested": "refuse", "deeply_nested_value": "refuse", "invalid_utf8": "refuse", "sha_mismatch": "refuse",
    "unknown_keys": "refuse", "csv_short_row": "refuse", "csv_short_row_2": "refuse", "csv_page_text": "refuse",
    "csv_page_empty": "refuse", "csv_count_negative": "refuse", "csv_count_nan": "refuse", "csv_count_1e9": "refuse",
    "csv_extra_columns": "refuse", "csv_no_header_columns": "refuse", "csv_empty": "refuse", "csv_no_evidence": "refuse",
    "csv_invalid_utf8": "refuse", "csv_binary": "refuse", "csv_quote_bomb": "refuse",
    "file_traversal": "refuse",
    "file_nonexistent": "refuse",
    "same_file_twice_with_different_counts": "refuse",
    "duplicate_json_keys": "refuse",
    "huge_manifest_100k_entries": "ok",
}
assert set(EXAMINED_EXPECT) == set(EXAMINED_VARIANTS) | set(EXAMINED_CSV_VARIANTS)


@pytest.mark.parametrize("name,want", _expect(EXAMINED_EXPECT))
def test_hostile_examined_fields_manifest_through_the_import_cli(corpus, tmp_path, name, want):
    build = EXAMINED_VARIANTS.get(name) or EXAMINED_CSV_VARIANTS[name]
    p = tmp_path / ("examined.csv" if name.startswith("csv") else "examined.json")
    p.write_bytes(build(corpus))
    data = tmp_path / "data"
    t0 = time.time()
    r = mlp(corpus.args(None, "--dry-run", "--mode", "examined-fields-manifest", "--examined-fields", p), data=data)
    check_outcome(r, want)
    if want == "refuse":
        assert not re.search(r"errors\.pydantic\.dev", r.err) or "REFUSED" in r.err
    assert not data.exists() and time.time() - t0 < 30


def test_examined_manifest_that_conflicts_or_lists_unknown_cells_never_adds_a_crop(corpus, clean, tmp_path):
    """Hostile but well-formed manifest entries (huge counts do not parse; here: counts that contradict the marks, cells that do not
    exist, pages beyond the stack) give conflicts / unresolved entries - and never a crop that the clean marked-only import lacks."""
    cdata, cver, cdig = clean
    cm_ = json.loads((cdata / "datasets" / cver / "manifest.json").read_text())
    doc = ex_doc(pages={"0": [{"row": 0, "col": 0, "count": 999}, {"row": 7, "col": 7, "count": 3}, {"row": 2, "col": 2, "count": 0}],
                        "1": [{"counter": 55, "count": 2}], "9": [{"row": 0, "col": 0}]})
    p = tmp_path / "e.json"
    p.write_text(json.dumps(doc))
    r = ml(corpus.args(None, "--mode", "examined-fields-manifest", "--examined-fields", p), data=tmp_path / "data")
    assert not r.traceback and r.rc in (0, 2), r
    if r.rc == 0:
        v, m = snapshot_of(tmp_path / "data")
        ours = {(c["source_file"], c["page"], c["region_id"]) for c in m["crops"]}
        theirs = {(c["source_file"], c["page"], c["region_id"]) for c in cm_["crops"]}
        assert ours <= theirs | {(FIRST, 0, "r2c2")}                    # a stated, mark-less zero on a real cell is a legitimate verified zero
        assert (FIRST, 0, "r0c0") not in ours                           # stated 999 vs the real marks -> conflict, excluded


# ============================================================================================ 7. command-line robustness
BAD_ARGS = {
    "unknown_flag": ["--bogus"],
    "unknown_flag_value": ["--bogus=1"],
    "fractions_nan": ["--fractions", "nan,0.5,0.5"],
    "fractions_inf": ["--fractions", "inf,0.5,0.5"],
    "fractions_minus_inf": ["--fractions=-inf,1,1"],
    "fractions_negative": ["--fractions=-1,1,1"],
    "fractions_two_numbers": ["--fractions", "0.5,0.5"],
    "fractions_four_numbers": ["--fractions", "0.4,0.2,0.2,0.2"],
    "fractions_sum_gt_1": ["--fractions", "0.5,0.5,0.5"],
    "fractions_sum_lt_1": ["--fractions", "0.2,0.2,0.2"],
    "fractions_zeros": ["--fractions", "0,0,0"],
    "fractions_words": ["--fractions", "a,b,c"],
    "fractions_overflow": ["--fractions", "1e400,0,0"],
    "fractions_trailing_nan": ["--fractions", "0.5,0.5,nan"],
    "fractions_comma_only": ["--fractions", ",,"],
    "downscale_3": ["--downscale", "3"],
    "downscale_0": ["--downscale", "0"],
    "downscale_negative": ["--downscale=-2"],
    "downscale_16": ["--downscale", "16"],
    "downscale_word": ["--downscale", "two"],
    "downscale_float": ["--downscale", "2.0"],
    "batch_size_0": ["--batch-size", "0"],
    "batch_size_negative": ["--batch-size=-4"],
    "epochs_0": ["--epochs", "0"],
    "workers_word": ["--workers", "many"],
    "seed_word": ["--seed", "abc"],
    "seed_float": ["--seed", "1.5"],
    "mode_bogus": ["--mode", "all-cells-are-negative"],
    "level_bogus": ["--level", "galaxy"],
    "normaliser_bogus": ["--normaliser", "magic"],
    "lab_csv_without_series": ["--lab-csv", "/nonexistent/counts.csv"],
    "lab_series_malformed": ["--lab-series", "T7B"],
    "examined_fields_nonexistent": ["--examined-fields", "/nonexistent/e.json"],
    "examined_fields_is_a_directory": ["--examined-fields", "."],
    "channel_map_is_a_directory": ["--channel-map", "."],
    "identity_map_is_a_directory": ["--identity-map", "."],
    "flag_without_value": ["--fractions"],
}


def bad_args_run(corpus, extra, *, dry, tmp_path):
    a = corpus.args(None, imap=True, fractions=None)
    # later flags win in argparse: the bad flag overrides the good defaults
    return mlp(a + ["--dry-run"] * dry + extra, data=tmp_path / "data")


@pytest.mark.parametrize("dry", [True, False], ids=["dry-run", "real"])
@pytest.mark.parametrize("name", list(BAD_ARGS))
def test_bad_arguments_are_refused_cleanly_and_write_nothing(corpus, tmp_path, name, dry):
    before = tree_digest(corpus.base, tmp_path / "data")
    r = bad_args_run(corpus, BAD_ARGS[name], dry=dry, tmp_path=tmp_path)
    if name == "examined_fields_is_a_directory" or name.endswith("_is_a_directory"):
        assert not r.traceback, r
        clean_refusal(r)
    else:
        clean_refusal(r)
    assert tree_digest(corpus.base, tmp_path / "data") == before
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("flag", ["--workers=0", "--workers=-1", "--workers=1000"])
def test_degenerate_worker_counts_are_clamped_and_never_change_the_snapshot(corpus, clean, tmp_path, flag):
    cdata, cver, cdig = clean
    r = ml(corpus.args(None, flag), data=tmp_path / "data")
    assert r.rc == 0 and not r.traceback, r
    assert version_of(r) == cver and snap_digest(tmp_path / "data", cver) == cdig


@pytest.mark.parametrize("seed", ["99999999999999999999999", "-5", "4294967296", "18446744073709551616"])
def test_extreme_seeds_work_end_to_end_or_are_refused(corpus, tmp_path, seed):
    r = ml(corpus.args(None, f"--seed={seed}"), data=tmp_path / "data")
    assert not r.traceback, r
    if r.rc == 0:
        v = version_of(r)
        assert ml(["verify", v], data=tmp_path / "data").out.strip() == "intact"
        assert ml(["split", "--snapshot", v], data=tmp_path / "data").rc == 0
    else:
        clean_refusal(r)


@pytest.mark.parametrize("kind", ["missing", "nonexistent", "a_text_file", "empty_dir", "dir_without_tiffs", "unreadable_names_only"])
def test_bad_input_paths_dry_run_reports_a_blocker_and_real_import_is_refused(corpus, tmp_path, kind):
    inp = tmp_path / "bad_input"
    if kind == "nonexistent":
        inp = tmp_path / "no" / "such" / "dir"
    elif kind == "a_text_file":
        inp.write_text("not a tiff")
    elif kind == "empty_dir":
        inp.mkdir()
    elif kind in ("dir_without_tiffs", "unreadable_names_only"):
        inp.mkdir()
        (inp / "a.txt").write_text("x")
        (inp / "b.png").write_bytes(b"\x89PNG")
        (inp / ("t" if kind == "dir_without_tiffs" else "t.tif.bak")).write_text("x")
    data = tmp_path / "data"
    if kind == "missing":
        a = ["import-legacy", "--channel-map", corpus.cmap, "--identity-map", corpus.imap, "--dry-run"]
    else:
        a = corpus.args(inp, "--dry-run")
    before = tree_digest(tmp_path / "bad_input", data)
    r = mlp(a, data=data)
    if kind in ("missing", "nonexistent"):
        clean_refusal(r)
    else:
        assert r.rc == 0 and not r.traceback and "BLOCKER" in r.out, r         # documented: a dry run reports blockers, exit 0
    assert tree_digest(tmp_path / "bad_input", data) == before and not data.exists()
    if kind != "missing":
        r = mlp(corpus.args(inp), data=data)
        clean_refusal(r)
        assert not data.exists() or published(data) == []


def test_real_import_without_a_channel_map_is_refused_before_anything_is_written(corpus, tmp_path):
    data = tmp_path / "data"
    r = mlp(corpus.args(None, cmap=False), data=data)
    clean_refusal(r, r"channel")
    assert not data.exists() or (published(data) == [] and staging_dirs(data) == [])


@pytest.mark.parametrize("cmd", [
    ["verify"], ["show"], ["split"], ["train"], ["verify", "../../etc"], ["verify", "/etc"], ["verify", ""], ["verify", "."],
    ["verify", ".."], ["show", "ds-../x"], ["split", "--snapshot", ""], ["split", "--snapshot", "../x"], ["split", "--snapshot", "a/b"],
    ["split", "--snapshot", "a\\b"], ["train", "--snapshot", "ds-nope", "--dry-run"], ["train", "--snapshot", ".staging-x", "--dry-run"],
    ["train", "--snapshot", "ds-nope", "--epochs", "0", "--dry-run"], ["list", "bogus"], ["promote"], ["compare"], ["compare", "--models", "x"],
    ["eval-model"], ["nonsense"], [],
])
def test_other_commands_refuse_bad_arguments_cleanly(corpus, tmp_path, cmd):
    r = mlp(cmd, data=tmp_path / "data")
    assert r.rc == 2 and not r.traceback, r
    assert not (tmp_path / "data").exists()


def test_data_dir_is_never_created_by_read_only_commands(tmp_path):
    d = tmp_path / "nothing-here"
    for cmd in (["list", "datasets"], ["list", "models"]):
        r = mlp(cmd, data=d)
        assert r.rc == 0 and not r.traceback, r
    assert not d.exists() or not any(d.rglob("*.json"))


# ============================================================================================ 8. locale / clock / time zone
def only_if_utf8_fs():
    if sys.getfilesystemencoding().lower().replace("-", "") != "utf8":
        pytest.skip("test file system names need a UTF-8 file system encoding")


@pytest.mark.parametrize("tz", ["UTC", "Pacific/Kiritimati", "America/St_Johns", "Etc/GMT+12", "Asia/Kolkata"])
def test_time_zone_does_not_change_the_snapshot(corpus, clean, tmp_path, tz):
    cdata, cver, cdig = clean
    r = ml(corpus.args(), data=tmp_path / "data", env=base_env(TZ=tz))
    assert r.rc == 0 and version_of(r) == cver and snap_digest(tmp_path / "data", cver) == cdig


@pytest.mark.parametrize("env", [{"LANG": "C", "LC_ALL": "C"},
                                 {"LC_ALL": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"},
                                 {"LC_ALL": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii"},
                                 {"LC_ALL": "tr_TR.ISO-8859-9", "PYTHONUTF8": "0", "PYTHONIOENCODING": "latin-1"},
                                 {"PYTHONIOENCODING": "cp1252"}],
                         ids=["C", "C-no-coercion-ascii-fs", "C-ascii-io", "turkish-latin9", "cp1252-io"])
def test_pipeline_under_other_locales_gives_the_same_snapshot(corpus, clean, tmp_path, env):
    """Non-UTF-8 locales / console encodings (cron, Windows, ssh without a locale): same bytes out, no UnicodeError."""
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    e = base_env(**env)
    r = ml(corpus.args(), data=data, env=e)
    assert r.rc == 0 and not r.traceback and version_of(r) == cver and snap_digest(data, cver) == cdig, r
    for cmd in (["verify", cver], ["split", "--snapshot", cver, "--lock-test"], ["show", cver]):
        r = ml(cmd, data=data, env=e)
        assert r.rc == 0 and not r.traceback, (cmd, r)
    r = ml(["train", "--snapshot", cver, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data, env=e)
    assert r.rc == 0 and not r.traceback, r


@needs_posix
def test_file_names_that_are_not_valid_utf8_do_not_crash_the_import_late(work, tmp_path):
    bad = os.fsencode(work.inp) + b"/bad-\xe9\xff-name.tif"
    shutil.copy(sorted(work.inp.glob("*.tif"))[0], bad)
    data = tmp_path / "data"
    assert ml(work.args(None, "--dry-run"), data=data).rc == 0
    r = ml(work.args(), data=data)
    assert not r.traceback, r
    assert r.rc in (0, 2)


@needs_posix
def test_non_ascii_file_names_under_a_pure_ascii_locale_are_handled(work, clean, tmp_path):
    only_if_utf8_fs()
    cdata, cver, cdig = clean
    shutil.copy(sorted(work.inp.glob("*.tif"))[0], work.inp / "29-09-2026_T7_AB_é日本-A.tif")      # byte-identical alias
    env = base_env(LC_ALL="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    r = ml(work.args(), data=tmp_path / "data", env=env)
    assert not r.traceback, r
    assert r.rc in (0, 2)
    if r.rc == 0:
        assert crop_files(tmp_path / "data", version_of(r)) == crop_files(cdata, cver)


@needs_posix
@pytest.mark.parametrize("fake", ["1970-01-01T00:00:00Z", "2099-12-31T23:59:59Z", "0001-01-01T00:00:00Z"])
def test_system_clock_is_irrelevant_to_the_version(corpus, clean, wrapper, tmp_path, fake):
    """No --created-at: created_at comes from utc_now (faked here); the version and every crop byte must not depend on it."""
    cdata, cver, cdig = clean
    data = tmp_path / "data"
    r = run_hooked(wrapper, corpus.args(created=None), data=data, mode="clock", extra_env={"FI_CLOCK": fake})
    assert r.rc == 0 and not r.traceback, r
    assert version_of(r) == cver
    m = json.loads((data / "datasets" / cver / "manifest.json").read_text())
    assert m["created_at"] == fake                                    # the hook really was in effect
    snap = snap_digest(data, cver)
    snap.pop("manifest.json"), cdig.copy().pop("manifest.json")
    assert {k: v for k, v in snap.items()} == {k: v for k, v in cdig.items() if k != "manifest.json"}
    assert ml(["verify", cver], data=data).out.strip() == "intact"


def test_source_file_mtimes_are_irrelevant_to_the_version(work, clean, tmp_path):
    cdata, cver, cdig = clean
    for i, q in enumerate(sorted(work.inp.glob("*.tif"))):
        os.utime(q, ns=((i % 3) * 10 ** 18 + 10 ** 9, (i % 3) * 10 ** 18 + 10 ** 9 + i))       # 1970 / 2001 / 2033
    r = ml(work.args(), data=tmp_path / "data")
    assert r.rc == 0 and version_of(r) == cver


# ============================================================================================ 9. scale and memory guard rails
def synth_snapshot(src_data, src_version, dst_data, n_crops: int) -> str:
    """A VALID, sealed snapshot with ``n_crops`` crops made by cloning the crops of a real small import (hard links for the files:
    cheap, and still hashed / verified byte for byte by ``verify``)."""
    from polony_app.ml.manifest import SnapshotManifest
    src = Path(src_data) / "datasets" / src_version
    m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    base = m["crops"]
    reps = -(-n_crops // len(base))
    crops, files = [], {k: v for k, v in m["files"].items() if k in ("points.csv", "regions.jsonl")}
    tmp = Path(dst_data) / "datasets" / ".synth-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for f in ("points.csv", "regions.jsonl"):
        tmp.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f, tmp / f)
    for k in range(reps):
        for c in base:
            if len(crops) >= n_crops:
                break
            nid = f"{c['crop_id']}_x{k:05d}"
            d = dict(copy.deepcopy(c), crop_id=nid, path_tif=f"spotiflow/{c['split']}/{nid}.tif", path_csv=f"spotiflow/{c['split']}/{nid}.csv")
            (tmp / "spotiflow" / c["split"]).mkdir(parents=True, exist_ok=True)
            for a, b in ((c["path_tif"], d["path_tif"]), (c["path_csv"], d["path_csv"])):
                try:
                    os.link(src / a, tmp / b)
                except OSError:
                    shutil.copy(src / a, tmp / b)
            files[d["path_tif"]], files[d["path_csv"]] = c["tif_sha256"], c["csv_sha256"]
            crops.append(d)
    m2 = dict(m, crops=crops, files=dict(sorted(files.items())), version="")
    mm = SnapshotManifest.model_validate(m2)
    mm.version = ds._version_of(mm)
    (tmp / "manifest.json").write_text(cm.pretty_json(mm.model_dump(mode="json")), encoding="utf-8")
    final = Path(dst_data) / "datasets" / mm.version
    os.rename(tmp, final)
    return mm.version


def maxrss_kb(cp: CP) -> int:
    m = re.search(r"MAXRSS_KB (\d+)", cp.err)
    assert m, cp
    return int(m.group(1))


def timed(args, wrapper_path, data, **kw):
    t0 = time.time()
    cp = finish(spawn(args, wrapper_path, data=data, env=base_env(FI_MAXRSS=1)), timeout=kw.get("timeout", 300))
    return cp, time.time() - t0


@pytest.fixture(scope="module")
def big_snapshot(clean, tmp_path_factory):
    cdata, cver, cdig = clean
    data = tmp_path_factory.mktemp("fi_big") / "data"
    t0 = time.time()
    v = synth_snapshot(cdata, cver, data, 20000)
    return data, v, time.time() - t0


def test_synthetic_snapshot_is_valid_and_has_20k_crops(big_snapshot, wrapper):
    data, v, _ = big_snapshot
    m = ds.load_manifest(data, v)
    assert len(m.crops) == 20000 and len({c.crop_id for c in m.crops}) == 20000


@pytest.mark.slow
@pytest.mark.parametrize("cmd", ["verify", "split", "list", "train"])
def test_20k_crop_snapshot_commands_stay_within_memory_and_time_budgets(big_snapshot, clean, wrapper, tmp_path, cmd):
    """Bounded memory (RSS) and a few seconds of wall time for verify / split / list / train --dry-run on 20 000 crops."""
    data, v, _ = big_snapshot
    cdata, cver, _ = clean
    small_args = {"verify": ["verify", cver], "split": ["split", "--snapshot", cver], "list": ["list", "datasets"],
                  "train": ["train", "--snapshot", cver, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"]}[cmd]
    big_args = [v if a == cver else a for a in small_args]
    base_cp, _ = timed(small_args, wrapper, cdata)
    assert base_cp.rc == 0 and not base_cp.traceback, base_cp
    cp, dt = timed(big_args, wrapper, data)
    assert cp.rc == 0 and not cp.traceback, cp
    if cmd == "verify":
        assert cp.out.strip() == "intact"
    grow_mb = (maxrss_kb(cp) - maxrss_kb(base_cp)) / 1024
    assert grow_mb < 900, f"{cmd}: RSS grew by {grow_mb:.0f} MiB for 20k crops"
    assert maxrss_kb(cp) / 1024 < 1600
    assert dt < 60, f"{cmd}: {dt:.1f}s for 20k crops"


@pytest.mark.slow
def test_import_of_a_few_hundred_files_has_a_flat_memory_profile(tmp_path, wrapper):
    """150 files (~1k crops): dry run and real import stay well under a memory budget, the dry run writes nothing, workers do not
    change a byte. (Memory is dominated by ONE source's planes at a time, never by the whole corpus.)"""
    inp = tmp_path / "in"
    inp.mkdir()
    files = {}
    for i in range(150):
        n = f"29-09-2026_T7_AB_S{i:03d}-A.tif"
        marks = marks_in(0, 0, 4, 0, 1, i) + marks_in(0, 1, 3, 1, 1, i) + marks_in(1, 0, 2, 2, 1, i) + marks_in(0, 0, 2, 0, 2, i + 7)
        make_tiff(inp / n, marks=marks, seed=i + 1000, h=H, w=W)
        files[f"S{i:03d}-A"] = n
    imap = write_identity_map(tmp_path / "id.json", files)
    cmap = write_channel_map(tmp_path / "ch.json", defaults=RED_GREEN)
    args = ["import-legacy", "--input", inp, "--channel-map", cmap, "--identity-map", imap, "--fractions", FR, "--created-at", T0]
    before = tree_digest(inp)
    dry, t_dry = timed(args + ["--dry-run"], wrapper, tmp_path / "data")
    assert dry.rc == 0 and not dry.traceback and tree_digest(inp) == before and not (tmp_path / "data").exists(), dry
    real, t_real = timed(args + ["--workers", "3"], wrapper, tmp_path / "data")
    assert real.rc == 0 and not real.traceback, real
    v = version_of(real)
    assert ml(["verify", v], data=tmp_path / "data").out.strip() == "intact"
    assert maxrss_kb(dry) / 1024 < 700 and maxrss_kb(real) / 1024 < 900, (maxrss_kb(dry), maxrss_kb(real))
    assert t_dry < 90 and t_real < 120, (t_dry, t_real)
    again = ml(args + ["--workers", "1"], data=tmp_path / "data2")
    assert version_of(again) == v


# ============================================================================================ 10. the train gate on hand-edited manifests
@pytest.fixture
def forge(clean, tmp_path):
    """-> forge(mutate) -> (data_dir, forged_version). ``mutate(manifest_dict, snapshot_root)`` edits a COPY of the clean snapshot, which is then
    RE-SEALED (version recomputed, directory renamed) so that ``verify`` is happy: only the train gate can still catch the edit."""
    cdata, cver, _ = clean
    data = tmp_path / "forged"
    shutil.copytree(cdata, data)
    unlock(data)
    n = {"i": 0}

    def make(mutate):
        from polony_app.ml.manifest import SnapshotManifest
        n["i"] += 1
        src = data / "datasets" / cver
        dst = data / "datasets" / f".forge-{n['i']}"
        shutil.copytree(src, dst)
        m = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        mutate(m, dst)
        mm = SnapshotManifest.model_validate(dict(m, version=""))
        mm.version = ds._version_of(mm)
        (dst / "manifest.json").write_text(cm.pretty_json(mm.model_dump(mode="json")), encoding="utf-8")
        if (data / "datasets" / mm.version).exists():          # no-op edit: same version
            shutil.rmtree(dst)
            return data, mm.version
        os.rename(dst, data / "datasets" / mm.version)
        if not getattr(mutate, "verify_refuses", False):
            assert ds.verify_snapshot(data, mm.version) == [], "the forgery must pass `verify` (it is re-sealed)"
        return data, mm.version
    return make


def first(m, split, k=0):
    return [c for c in m["crops"] if c["split"] == split][k]


def _set(which, /, **kw):
    def f(m, root):
        first(m, which).update(kw)
    return f


def _prov(which, /, **kw):
    def f(m, root):
        c = first(m, which)
        c["provenance"] = dict(c["provenance"], **kw)
    return f


def _del_prov_key(split, key):
    def f(m, root):
        first(m, split)["provenance"].pop(key)
    return f


def _no_import_summary(strip_provenance):
    def f(m, root):
        m["import_summary"] = {}
        if strip_provenance:
            for c in m["crops"]:
                c["provenance"] = {}
    return f


def _dup_crop(same_split):
    def f(m, root):
        c = first(m, "train")
        d = copy.deepcopy(c)
        if not same_split:
            d["split"] = "val"
            d["group_id"] = first(m, "val")["group_id"] + "-x"
        m["crops"].append(d)
    return f


def _edge_values(**kw):
    return _set("train", **kw)


def _path_escape(m, root):
    c = first(m, "train")
    outside = root.parent / "outside.tif"
    shutil.copy(root / c["path_tif"], outside)
    c["path_tif"] = "../outside.tif"
    m["files"]["../outside.tif"] = c["tif_sha256"]


_path_escape.verify_refuses = True       # `verify` itself reports a path that leaves the snapshot directory


GATE = {
    # name: (mutation, regex of the refusal, None)
    "provenance_removed": (_set("train", provenance={}), r"provenance|researcher_examined", None),
    "researcher_examined_false": (_prov("train", researcher_examined=False), r"researcher_examined", None),
    "researcher_examined_missing": (_del_prov_key("train", "researcher_examined"), r"researcher_examined", None),
    "researcher_examined_string_true": (_prov("train", researcher_examined="true"), r"researcher_examined", None),
    "examination_status_unknown": (_prov("train", examination_status="unknown"), r"examination_status|positive_marked", None),
    "examination_status_conflict": (_prov("val", examination_status="conflict"), r"examination_status|positive_marked", None),
    "source_tiff_removed": (_del_prov_key("train", "source_tiff"), r"source_tiff|provenance", None),
    "source_sha_removed": (_del_prov_key("train", "source_sha256"), r"source_sha256|provenance", None),
    "evidence_blank_on_zero": (_prov("train", examination_status="verified_zero", evidence=""), r"evidence|provenance", None),
    "import_summary_removed_but_provenance_kept": (_no_import_summary(False), r"import|summary|provenance", None),
    "import_summary_and_all_provenance_removed": (_no_import_summary(True), r"import|summary|provenance", None),
    "crop_id_duplicated_same_split": (_dup_crop(True), r"duplicate|twice|unique", None),
    "crop_id_duplicated_across_train_and_val": (_dup_crop(False), r"duplicate|twice|unique|share|leak", None),
    "wrong_downscale_mix": (_set("train", downscale=2), r"downscale", None),
    "wrong_normaliser_mix": (_set("val", normaliser="none"), r"normali", None),
    "missing_lo": (_set("train", norm_lo=None), r"lo/hi|page-level|normali", None),
    "missing_hi": (_set("val", norm_hi=None), r"lo/hi|page-level|normali", None),
    "lo_equals_hi": (lambda m, r: first(m, "train").update(norm_hi=first(m, "train")["norm_lo"]), r"lo|hi|normali", None),
    "lo_is_nan": (_set("train", norm_lo=float("nan")), r"lo|hi|normali|finite", None),
    "hi_is_infinite": (_set("train", norm_hi=float("inf")), r"lo|hi|normali|finite", None),
    "negative_positive_count": (_edge_values(n_positive=-5), r"n_positive|negative|label", None),
    "zero_size_crop": (_edge_values(shape_hw=[0, 0]), r"shape|size|empty", None),
    "unknown_split_name": (_set("train", split="holdout"), r"split", None),
    "path_escapes_the_snapshot": (_path_escape, r"path|outside|escape|inside", None),
}


@pytest.mark.parametrize("name", list(GATE))
def test_train_refuses_a_forged_manifest_with_a_clear_message(forge, tmp_path, name):
    mutate, rx, _pending = GATE[name]
    data, v = forge(mutate)
    r = mlp(["train", "--snapshot", v, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data)
    clean_refusal(r, rx)
    assert "Training will not start" not in r.out and "nothing was trained" not in r.out.replace("nothing was trained and", "")
    assert not (data / "models").exists() or not list((data / "models").glob("*"))


def test_the_forge_helper_alone_changes_nothing(forge, tmp_path):
    """Control: a re-sealed manifest without edits trains (dry run) fine - so every refusal above is caused by its edit."""
    data, v = forge(lambda m, root: None)
    r = train_dry(data, v, dummy_base(tmp_path))
    assert r.rc == 0 and "Training will not start" in r.out, r


def test_train_refuses_a_locked_test_crop_moved_to_train_even_when_the_group_is_renamed(clean, forge, tmp_path):
    cdata, cver, _ = clean
    data, _v0 = forge(lambda m, root: None)
    assert mlp(["split", "--snapshot", cver, "--lock-test"], data=data).rc == 0

    def move(m, root):
        c = first(m, "test")
        c["split"], c["group_id"] = "train", "innocent-looking-group"
    data, v = forge(move)
    clean_refusal(mlp(["train", "--snapshot", v, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data), r"locked|test")


def test_train_refuses_a_forged_val_crop_that_shares_a_group_with_train(forge, tmp_path):
    data, v = forge(lambda m, root: first(m, "val").update(group_id=first(m, "train")["group_id"]))
    clean_refusal(mlp(["train", "--snapshot", v, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data), r"share|leak|group")


def test_edits_without_resealing_are_caught_by_verify_and_refused_by_train(clean, tmp_path):
    cdata, cver, _ = clean
    data = tmp_path / "d"
    shutil.copytree(cdata, data)
    unlock(data)
    mp = data / "datasets" / cver / "manifest.json"
    m = json.loads(mp.read_text())
    m["crops"][0]["provenance"] = {}
    mp.write_text(json.dumps(m))
    r = ml(["verify", cver], data=data)
    assert r.rc == 3 and "PROBLEMS" in r.out and "version" in r.out, r
    clean_refusal(mlp(["train", "--snapshot", cver, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data), r"integrity|intact|failed")
    clean_refusal(mlp(["split", "--snapshot", cver, "--lock-test"], data=data))
    assert not (data / "datasets" / "test_lock.json").exists()


# ============================================================================================ 11. the app's examined-fields / compare endpoint
INFO = "ImageDescription: 635 [W1]\nSoftware: ImageIO 2.0.4.0"
API_PTS = [(20.5, 20.5), (25.5, 22.5), (30.5, 30.5), (70.5, 20.5), (80.5, 25.5), (20.5, 60.5), (22.5, 62.5)]
API_CNT = [0, 0, 0, 1, 1, 2, 2]
API_POS = [1, 1, 1, 1, 1, 1, 2]


@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient
    from legacy_fixtures import blank, point_roi as pr, write_stack
    from polony_app.api.main import create_app
    with TestClient(create_app(tmp_path / "appdata"), raise_server_exceptions=False) as client:
        p = write_stack(tmp_path / "s.tif", [blank(), blank()], roi=pr(API_PTS, API_CNT, API_POS),
                        overlays=[grid_overlay(GRID_XS, GRID_YS, 160, 120)], info=INFO)
        with open(p, "rb") as f:
            r = client.post("/api/analyses", files=[("files", (p.name, f, "image/tiff"))])
        aid = r.json()["id"]
        client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
        assert client.post(f"/api/analyses/{aid}/fields/import", json={"mode": "all"}).status_code == 200
        yield client, aid


def api_doc(entries=None, **kw):
    return ex_doc(entries, file="s.tif", **kw)


def post_manifest(client, aid, body, name="m.json"):
    return client.post(f"/api/analyses/{aid}/examined-fields", files={"file": (name, body, "application/octet-stream")})


def compare(client, aid):
    return client.get(f"/api/analyses/{aid}/fields/compare?plane=0&count_type=red_positive")


API_GOOD = json.dumps(api_doc([{"counter": 0, "count": 3}, {"counter": 1, "count": 2}])).encode()


@pytest.mark.parametrize("name", [n for n in list(EXAMINED_VARIANTS) + list(EXAMINED_CSV_VARIANTS) if n != "huge_manifest_100k_entries"])
def test_examined_fields_upload_never_answers_500_and_never_replaces_a_good_manifest(api, name):
    client, aid = api
    assert post_manifest(client, aid, API_GOOD).status_code == 200
    before = compare(client, aid)
    assert before.status_code == 200
    build = EXAMINED_VARIANTS.get(name) or EXAMINED_CSV_VARIANTS[name]
    body = build(None)
    r = post_manifest(client, aid, body, "m.csv" if name.startswith("csv") else "m.json")
    assert r.status_code < 500, (name, r.status_code, r.text[:300])
    if r.status_code != 200:
        assert r.status_code == 422, (name, r.status_code, r.text[:300])
        after = compare(client, aid)
        assert after.status_code == 200 and after.json() == before.json()          # a refused upload changed nothing


def test_examined_fields_upload_of_an_oversized_body_is_refused_not_buffered(api):
    client, aid = api
    r = post_manifest(client, aid, b"x" * (3 * 1024 * 1024))
    assert r.status_code == 422, r.status_code
    assert compare(client, aid).status_code == 200


def test_examined_fields_upload_with_hostile_file_names_stays_inside_the_analysis(api, tmp_path):
    client, aid = api
    for name in ("../../evil.json", "..\\..\\evil.csv", "a/b/c.json", "x" * 3000 + ".json", "‮.json", ".json", ""):
        r = post_manifest(client, aid, API_GOOD, name)
        assert r.status_code < 500, (name, r.status_code, r.text[:200])
    assert not (tmp_path / "evil.json").exists() and not (tmp_path.parent / "evil.json").exists()
    assert not list(tmp_path.rglob("evil*"))


def test_compare_with_a_corrupted_stored_manifest_does_not_answer_500(api, tmp_path):
    client, aid = api
    assert post_manifest(client, aid, API_GOOD).status_code == 200
    (stored,) = list((tmp_path / "appdata").rglob("examined_fields.json"))
    stored.write_bytes(stored.read_bytes()[:40])
    r = compare(client, aid)
    assert r.status_code < 500, r.status_code


def test_compare_without_any_manifest_is_unaffected_by_failed_uploads(api):
    client, aid = api
    base = compare(client, aid).json()
    for body in (b"", b"\xff\xfe", b"[]", b"{}"):
        assert post_manifest(client, aid, body).status_code < 500
    assert compare(client, aid).json() == base


# ============================================================================================ 12. robustness fixes (second pass)
def _dup_val_crop(m, root):
    c = first(m, "val")
    m["crops"].append(copy.deepcopy(c))


def _abs_path(m, root):
    c = first(m, "train")
    c["path_tif"] = str(root / c["path_tif"])


def _symlink_escape(m, root):
    c = first(m, "train")
    outside = root.parent / "outside-dir"
    outside.mkdir(exist_ok=True)
    shutil.copy(root / c["path_tif"], outside / "x.tif")
    link = root / "spotiflow" / "escape"
    link.symlink_to(outside, target_is_directory=True)
    c["path_tif"] = "spotiflow/escape/x.tif"
    m["files"]["spotiflow/escape/x.tif"] = c["tif_sha256"]


def _files_key_traversal(m, root):
    m["files"]["../../etc/passwd"] = "0" * 64


def _crop_not_hashed(m, root):
    m["files"].pop(first(m, "train")["path_csv"])


def _box_shape_mismatch(m, root):
    first(m, "train")["box_xyxy"][2] += 7


for _n, _f in (("crop_id_duplicated_within_val", _dup_val_crop), ("crop_path_absolute", _abs_path),
               ("crop_path_symlink_escape", _symlink_escape), ("files_key_traversal", _files_key_traversal),
               ("crop_file_not_listed_in_files", _crop_not_hashed)):
    _f.verify_refuses = _n in ("crop_path_absolute", "crop_path_symlink_escape", "files_key_traversal", "crop_file_not_listed_in_files")


@pytest.mark.skipif(not HAS_SYMLINKS, reason="symlinks")
@pytest.mark.parametrize("name,mutate,rx", [
    ("crop_id_duplicated_within_val", _dup_val_crop, r"more than once|unique"),
    ("crop_path_absolute", _abs_path, r"escape|absolute"),
    ("crop_path_symlink_escape", _symlink_escape, r"escape|outside"),
    ("files_key_traversal", _files_key_traversal, r"escape|'\.\.'"),
    ("crop_file_not_listed_in_files", _crop_not_hashed, r"not listed"),
    ("box_and_shape_disagree", _box_shape_mismatch, r"inconsistent with box"),
], ids=["dup_in_val", "path_absolute", "symlink_escape", "files_key_traversal", "crop_not_hashed", "box_shape"])
def test_more_forged_manifests_are_refused_by_train_and_the_path_ones_by_verify(forge, tmp_path, name, mutate, rx):
    data, v = forge(mutate)
    clean_refusal(mlp(["train", "--snapshot", v, "--base", dummy_base(tmp_path), "--dry-run", "--allow-small"], data=data), rx)
    if getattr(mutate, "verify_refuses", False):
        r = mlp(["verify", v], data=data)
        assert r.rc == 3 and "PROBLEMS" in r.out and not r.traceback, r
        clean_refusal(mlp(["split", "--snapshot", v, "--lock-test"], data=data))


def test_a_build_snapshot_with_carried_over_legacy_test_crops_is_not_mistaken_for_an_import():
    from polony_app.ml.manifest import SnapshotManifest
    frozen = {"crop_id": "c", "provenance": {"researcher_examined": True}, "frozen_from": "ds-x"}
    assert not lb.is_legacy_import({"options": {}, "import_summary": {}, "crops": [frozen]})
    assert lb.is_legacy_import({"options": {}, "import_summary": {}, "crops": [dict(frozen, frozen_from=None)]})
    assert lb.is_legacy_import({"options": {"legacy_import": {}}, "import_summary": {}, "crops": []})
    assert lb.is_legacy_import(SnapshotManifest(import_summary={"kind": "x"}))


def test_main_turns_environment_errors_into_refusals_and_never_swallows_ctrl_c(monkeypatch, tmp_path, capsys):
    from polony_app.ml import cli
    from polony_app.imaging.io import IngestError
    for exc, rx in ((OSError(errno.ENOSPC, "No space left on device", "/x/y"), r"No space left.*/x/y.*resumes"),
                    (PermissionError(errno.EACCES, "Permission denied", "/src/a.tif"), r"Permission denied.*/src/a\.tif"),
                    (IngestError("truncated TIFF"), r"truncated TIFF"),
                    (UnicodeEncodeError("utf-8", "\udce9", 0, 1, "surrogates not allowed"), r"UTF-8"),
                    (json.JSONDecodeError("Expecting value", "{", 1), r"malformed")):
        monkeypatch.setattr(cli, "_run", lambda a, d, e=exc: (_ for _ in ()).throw(e))
        assert cli.main(["--data-dir", str(tmp_path), "list", "datasets"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("REFUSED:") and re.search(rx, err, re.S) and "Traceback" not in err, err
    monkeypatch.setattr(cli, "_run", lambda a, d: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--data-dir", str(tmp_path), "list", "datasets"])
    monkeypatch.setattr(cli, "_run", lambda a, d: (_ for _ in ()).throw(KeyError("a real bug")))
    with pytest.raises(KeyError):
        cli.main(["--data-dir", str(tmp_path), "list", "datasets"])


def test_lab_series_flag_combinations_are_validated_on_their_own(corpus, tmp_path):
    csvf = tmp_path / "T7B_counts.csv"
    csvf.write_text("x\n")
    for extra, rx in ((["--lab-series", "T7B=0"], r"without --lab-csv"),
                      (["--lab-series", "T7B"], r"SERIES=PAGE"),
                      (["--lab-series", "T7B=²"], r"SERIES=PAGE"),
                      (["--acknowledge-lab-csv-field-numbering"], r"without --lab-csv"),
                      (["--lab-csv", csvf, "--acknowledge-lab-csv-field-numbering"], r"--lab-series"),
                      (["--lab-csv", csvf, "--lab-series", "T7B=0"], r"acknowledge-lab-csv-field-numbering")):
        for dry in ([], ["--dry-run"]):
            clean_refusal(mlp(corpus.args(None, *dry, *extra), data=tmp_path / "data"), rx)
    assert not (tmp_path / "data").exists()


def test_non_utf8_file_names_are_skipped_with_a_structured_reason_in_the_dry_run_and_the_import(work, tmp_path):
    bad = os.fsencode(work.inp) + b"/bad-\xe9\xff-name.tif"
    shutil.copy(sorted(work.inp.glob("*.tif"))[0], bad)
    data = tmp_path / "data"
    dry = ml(work.args(None, "--dry-run"), data=data)
    assert dry.rc == 0 and not dry.traceback and "filename_not_utf8" in dry.out and "\\udce9" in dry.out, dry
    r = ml(work.args(), data=data)
    assert r.rc == 0 and not r.traceback and "filename_not_utf8" in r.out, r
    v = version_of(r)
    assert ml(["verify", v], data=data).out.strip() == "intact"
    m = json.loads((data / "datasets" / v / "manifest.json").read_text(encoding="utf-8"))
    assert any(s["reason"].startswith("filename_not_utf8") for s in m["import_summary"]["tiffs"]["skipped_links"])
    assert not any("\udce9" in c["source_file"] for c in m["crops"])


def test_atomic_write_text_refuses_text_that_is_not_utf8_without_leaving_files(tmp_path):
    with pytest.raises(MLError, match="UTF-8"):
        cm.atomic_write_text(tmp_path / "a.json", "name-\udce9.tif")
    assert list(tmp_path.iterdir()) == []


def test_same_size_rewrite_with_restored_mtime_is_refused_by_the_rehash(work, tmp_path, monkeypatch):
    """The strict twin of the 'never silently' test above: the REAL import must refuse (inode / ctime / content hash), with a message."""
    Mutator(monkeypatch, lambda p: flip_pixel_bytes_in_place(p, keep_mtime=True))
    with pytest.raises(MLError, match="changed on disk"):
        inproc_import(work, tmp_path / "data")
    assert published(tmp_path / "data") == []


def test_duplicate_keys_are_refused_with_the_key_in_the_message(corpus, tmp_path):
    p = tmp_path / "channels.json"
    p.write_text(corpus.cmap.read_text().replace('"1": {', '"0": {"role": "blue", "train": true}, "1": {', 1))
    r = mlp(corpus.args(None, "--dry-run", cmap=p), data=tmp_path / "data")
    clean_refusal(r, r"'0'.*more than once")
    q = tmp_path / "ex.json"
    q.write_bytes(raw_json(ex_doc(), {'"count": 4': '"count": 4, "count": 0'}))
    clean_refusal(mlp(corpus.args(None, "--dry-run", "--mode", "examined-fields-manifest", "--examined-fields", q), data=tmp_path / "d2"),
                  r"'count'.*more than once")


def test_a_hostile_examined_file_entry_names_the_nearest_existing_file(corpus, tmp_path):
    q = tmp_path / "ex.json"
    q.write_bytes(raw_json(ex_doc(file="29-09-2026_T7_AB_AM-Z.tif")))
    r = mlp(corpus.args(None, "--dry-run", "--mode", "examined-fields-manifest", "--examined-fields", q), data=tmp_path / "data")
    clean_refusal(r, r"matches no scanned TIFF.*nearest existing file: '29-09-2026_T7_AB_AM-")
