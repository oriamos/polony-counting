#!/usr/bin/env python3
"""End-to-end REHEARSAL of the historical-TIFF fine-tuning workflow on SYNTHETIC data with an INDEPENDENT ORACLE.

Drives the REAL CLI (``python -m polony_app.ml ...``, one subprocess per command) through

    write-templates -> refusal paths -> dry-run -> import (workers 1 and 4) -> verify (+ tamper) -> split -> lock-test ->
    import of extra data after the lock -> kill -9 / resume -> disk-full -> train --dry-run -> tiny CPU fine-tune ->
    eval-model -> compare

and checks every outcome against the oracle written by ``make_synthetic.py`` (computed from the generator's own parameters only).
Every check is PASS / FAIL / KNOWN_FAIL / SKIP with evidence in ``REHEARSAL.md`` and ``rehearsal.json`` (in ``--out``). The process exit
code is 1 when any check FAILs. Nothing is written into the repository.

    PYTHONPATH=src python3 scripts/preflight/rehearsal.py --n 1500 --cell-px 64 --workers 4 --out DIR --keep
    PYTHONPATH=src python3 scripts/preflight/rehearsal.py --n 24 --out DIR --fast --no-model --calibration 0     # smoke
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_synthetic as ms                                           # noqa: E402

T0 = "2026-01-01T00:00:00Z"
SRC = str(REPO / "src")
sys.path.insert(0, SRC)


# ====================================================================================== report
class Report:
    def __init__(self):
        self.checks: list = []
        self.runs: list = []
        self.perf: dict = {}
        self.notes: list = []

    def add(self, name, ok, evidence="", known=None, phase=""):
        st = "PASS" if ok else ("KNOWN_FAIL" if known else "FAIL")
        self.checks.append(dict(name=name, status=st, evidence=str(evidence)[:1500], phase=phase, known=known if not ok else None))
        print(f"  [{st:10}] {name}" + ("" if ok else f"  -- {str(evidence)[:300]}"), flush=True)
        return ok

    def skip(self, name, why, phase=""):
        self.checks.append(dict(name=name, status="SKIP", evidence=why, phase=phase, known=None))
        print(f"  [SKIP      ] {name}: {why}", flush=True)

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    def write(self, out: Path, meta: dict):
        (out / "rehearsal.json").write_text(json.dumps(dict(meta=meta, checks=self.checks, runs=self.runs, perf=self.perf, notes=self.notes),
                                                         indent=1, default=str), encoding="utf-8")
        cnt = Counter(c["status"] for c in self.checks)
        L = [f"# Rehearsal report", "", f"`{meta.get('command', '')}`", "",
             f"Checks: {dict(cnt)}  ({'ALL PASS' if not cnt.get('FAIL') else 'FAILURES PRESENT'})", ""]
        L += ["## Checks", "", "| status | phase | check | evidence |", "|---|---|---|---|"]
        for c in self.checks:
            L.append(f"| {c['status']} | {c['phase']} | {c['name']} | {c['evidence'][:300].replace('|', '/').replace(chr(10), ' ')} |")
        L += ["", "## Commands (wall time, peak RSS of the child, exit code)", "", "| label | wall s | peak RSS MB | exit |", "|---|---|---|---|"]
        for r in self.runs:
            L.append(f"| {r['label']} | {r['wall']:.1f} | {r['rss_mb']:.0f} | {r['exit']} |")
        L += ["", "## Measurements", "", "```json", json.dumps(self.perf, indent=1, default=str), "```", ""]
        for n in self.notes:
            L.append(f"* {n}")
        (out / "REHEARSAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ====================================================================================== subprocess driver
WRAPPER = r'''
import os, sys, time, shutil
sys.path.insert(0, os.environ["REHEARSAL_SRC"])
free = os.environ.get("REHEARSAL_DISK_FREE")
if free:
    _orig = shutil.disk_usage
    shutil.disk_usage = lambda p: _orig(p)._replace(free=int(free))
from polony_app.ml import dataset as _ds
slow = os.environ.get("REHEARSAL_SLOW_WRITE_MS")
after = os.environ.get("REHEARSAL_ENOSPC_AFTER")
if slow or after:
    import threading, errno
    _w = _ds.write_crop_files
    _n = [0]; _lk = threading.Lock()
    def write_crop_files(*a, **k):
        with _lk:
            _n[0] += 1
            n = _n[0]
        if after and n > int(after):
            raise OSError(errno.ENOSPC, "No space left on device (injected by the rehearsal)")
        if slow:
            time.sleep(int(slow) / 1000.0)
        return _w(*a, **k)
    _ds.write_crop_files = write_crop_files
gen = os.environ.get("REHEARSAL_GENERAL_STAND_IN")
if gen:
    from polony_app.ml import spotiflow_io as sio
    _real = sio.load_model
    def load_model(b, data_dir, **kw):
        return _real(gen if b == "general" else b, data_dir, **kw)
    sio.load_model = load_model
from polony_app.ml.cli import main
sys.exit(main(sys.argv[1:]))
'''


class Ctx:
    def __init__(self, out: Path, rep: Report):
        self.out = out
        self.rep = rep
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.wrapper = out / "_cli_wrapper.py"
        self.wrapper.write_text(WRAPPER)
        self.n = 0

    def run(self, label, args, data_dir=None, env=None, wrapper=False, popen=False):
        """Run one CLI command in a subprocess. -> dict(exit, out, err, wall, rss_mb)."""
        self.n += 1
        cmd = [sys.executable] + ([str(self.wrapper)] if wrapper else ["-m", "polony_app.ml"])
        if data_dir is not None:
            cmd += ["--data-dir", str(data_dir)]
        cmd += [str(a) for a in args]
        e = dict(os.environ, PYTHONPATH=SRC, REHEARSAL_SRC=SRC, PYTHONUNBUFFERED="1", **{k: str(v) for k, v in (env or {}).items()})
        e.pop("POLONY_DATA_DIR", None)
        so, se = self.logs / f"{self.n:03d}.out", self.logs / f"{self.n:03d}.err"
        fo, fe = open(so, "wb"), open(se, "wb")
        t = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=e, cwd=str(REPO))
        if popen:
            return dict(proc=p, files=(fo, fe), t=t, label=label, so=so, se=se, cmd=cmd)
        return self._finish(label, p, fo, fe, t, so, se, cmd)

    def _finish(self, label, p, fo, fe, t, so, se, cmd):
        hwm = 0.0
        while True:                                   # ru_maxrss of a forked child is inflated by the parent's pre-exec RSS: sample VmHWM instead
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmHWM:"):
                        hwm = max(hwm, int(line.split()[1]) / 1024.0)
            except OSError:
                pass
            pid_, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid_:
                break
            time.sleep(0.05)
        wall = time.perf_counter() - t
        p.returncode = os.waitstatus_to_exitcode(status)
        fo.close(); fe.close()
        r = dict(exit=p.returncode, out=so.read_text(errors="replace"), err=se.read_text(errors="replace"), wall=wall,
                 rss_mb=hwm or ru.ru_maxrss / 1024.0, label=label, cmd=" ".join(cmd[2:] if cmd[1] != str(self.wrapper) else cmd[2:]))
        self.rep.runs.append(dict(label=label, wall=wall, rss_mb=r["rss_mb"], exit=r["exit"]))
        return r


def tree_stat_hash(root: Path) -> str:
    h = hashlib.sha256()
    for dp, dn, fn in os.walk(root, followlinks=False):
        dn.sort()
        for f in sorted(fn):
            p = Path(dp) / f
            st = os.lstat(p)
            h.update(f"{p.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}|{st.st_mode}\n".encode())
        for d in dn:
            h.update(f"D{Path(dp, d).relative_to(root)}\n".encode())
    return h.hexdigest()


def tree_content_hashes(root: Path) -> dict:
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = Path(dp) / f
            out[str(p.relative_to(root))] = ms.file_sha256(p)
    return out


def dir_bytes(root: Path) -> int:
    return sum(os.path.getsize(Path(dp) / f) for dp, _d, fn in os.walk(root) for f in fn)


def kv(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, _, v = part.strip().partition("=")
        if k and v.strip().lstrip("-").isdigit():
            out[k.strip()] = int(v)
    return out


def snapshots(data_dir: Path) -> list:
    d = Path(data_dir) / "datasets"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "manifest.json").is_file()) if d.is_dir() else []


def load_snapshot(data_dir: Path, version: str) -> dict:
    root = Path(data_dir) / "datasets" / version
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    regions = [json.loads(l) for l in (root / "regions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(root / "points.csv", newline="", encoding="utf-8") as f:
        points = list(csv.DictReader(f))
    return dict(root=root, man=man, regions=regions, points=points)


REASON_CLASS = [("unspecified", "unspecified_plane"), ("plane", "unspecified_plane"), ("straddling", "straddle"), ("no researcher marks", "no_marks"),
                ("stated count", "stated_mismatch"), ("not listed", "not_listed"), ("no count is stated", "no_count"),
                ("two manifest entries", "manifest_clash")]


def reason_class(text):
    t = (text or "").lower()
    for k, c in REASON_CLASS:
        if k in t:
            return c
    return "" if not t else "other:" + t[:40]


# ====================================================================================== independent normalisation
def expected_norm(pages: list, page: int, grid: ms.Grid, eligible_boxes: list, mode: str, rim_px: int = 40):
    """lo/hi of one PAGE, recomputed from the TIFF pixels with the documented rule: percentiles 1 / 99.8 of the valid, unsaturated
    pixels. ``mode``: 'chamber' (app chamber mask, fallback = all grid cells), 'grid-cells', 'eligible-fields'."""
    a = pages[page]
    h, w = a.shape
    note = None

    def union(boxes):
        v = np.zeros(a.shape, bool)
        for x0, y0, x1, y1 in boxes:
            v[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        return v
    cells = [grid.box(c) for c in range(grid.rows * grid.cols)]
    if mode == "chamber":
        try:
            from polony_app import schemas
            from polony_app.detection.masks import build_valid_mask
            from polony_app.imaging.chamber import ChamberError, auto_chamber
            ch = auto_chamber(pages, rim_px=rim_px)
            reg = schemas.Region(id="chamber", kind=schemas.RegionKind.counting, name="chamber", polygon=[tuple(p) for p in ch["polygon"]])
            valid = build_valid_mask(a.shape, [reg])
            note = "chamber"
        except Exception as exc:                      # noqa: BLE001 - ChamberError and polygon problems alike -> documented fallback
            valid = union(cells)
            note = f"fallback_grid_cells ({type(exc).__name__})"
    elif mode == "grid-cells":
        valid = union(cells)
    else:
        valid = union(eligible_boxes)
    sat = a >= np.iinfo(a.dtype).max
    use = valid & ~sat
    if use.sum() < 1000:
        use = valid if valid.sum() >= 1000 else np.ones(a.shape, bool)
    lo, hi = (float(x) for x in np.percentile(a[use].astype(np.float64), [1.0, 99.8]))
    if not hi > lo:
        hi = lo + 1.0
    return lo, hi, note


# ====================================================================================== the big oracle comparison
def check_snapshot(rep: Report, tag: str, data_dir: Path, version: str, files: dict, root: Path, ch: dict, *, mode="marked-only",
                   mdoc=None, norm_mode="eligible-fields", pixel_files_limit=None, downscale=1):
    ph = f"oracle:{tag}"
    S = load_snapshot(data_dir, version)
    man, summ = S["man"], S["man"]["import_summary"]
    kept, alias = ms.dedupe_groups(files)
    exp = ms.expected_status(files, ch, mode)
    rel_of_id = {s["analysis_id"]: s["name"] for s in man["sources"]}
    # ---- files: acceptance, reasons, dedupe
    t = summ["tiffs"]
    exp_rej = {r: v["reason"] for r, v in exp.items() if v["status"] == "rejected"}
    got_rej = {r: c[0] for r, c in t["rejected_files"].items()}
    rep.add(f"[{tag}] discovered/unique/accepted/rejected/deduplicated counts equal the oracle",
            (t["discovered"], t["unique"], t["accepted"], t["rejected"], t["deduplicated"]) ==
            (len(files), len(kept), len(kept) - len(exp_rej), len(exp_rej), sum(len(v) for v in alias.values())),
            f"product {t['discovered']},{t['unique']},{t['accepted']},{t['rejected']},{t['deduplicated']} oracle "
            f"{len(files)},{len(kept)},{len(kept) - len(exp_rej)},{len(exp_rej)},{sum(len(v) for v in alias.values())}", phase=ph)
    diff = {r: (exp_rej.get(r), got_rej.get(r)) for r in set(exp_rej) | set(got_rej) if exp_rej.get(r) != got_rej.get(r)}
    rep.add(f"[{tag}] every rejected file and its reason class equals the oracle ({len(exp_rej)} rejected)", not diff, dict(list(diff.items())[:5]), phase=ph)
    rep.add(f"[{tag}] reason histogram equals", t["rejected_by_reason"] == dict(sorted(Counter(exp_rej.values()).items())),
            f"{t['rejected_by_reason']} vs {dict(Counter(exp_rej.values()))}", phase=ph)
    rep.add(f"[{tag}] dedupe: kept path (first in sorted order) and alias list equal the oracle",
            t["deduplicated_paths"] == {k: sorted(v) for k, v in sorted(alias.items())}, f"{t['deduplicated_paths']} vs {alias}", phase=ph)
    # ---- per (file, field, page) tuples
    accepted = {r: files[r] for r, v in exp.items() if v["status"] == "accepted"}
    D = {}
    for rel, rec in accepted.items():
        pm = {int(k): v for k, v in rec["mapping_pages"].items()}
        D[rel] = ms.derive(rec, pm, mode, (mdoc or {}).get(rel))
    bad, n_tuple = [], 0
    got_tuples = {}
    for reg in S["regions"]:
        rel = rel_of_id.get(reg["analysis_id"])
        if rel not in accepted:
            continue
        ncols = len(accepted[rel]["grid"]["xs"]) - 1
        cell = int(reg["grid_row"]) * ncols + int(reg["grid_col"])
        for pg, e in reg["examination"].items():
            got_tuples[(rel, int(pg), cell)] = e
    exp_tuples = {(rel, p, c): t_ for rel, d in D.items() for (p, c), t_ in d["tuples"].items()}
    rep.add(f"[{tag}] the set of (file, field, page) tuples equals the oracle ({len(exp_tuples)})", set(got_tuples) == set(exp_tuples),
            f"missing {len(set(exp_tuples) - set(got_tuples))} extra {len(set(got_tuples) - set(exp_tuples))}", phase=ph)
    for k in sorted(set(got_tuples) & set(exp_tuples)):
        g, e = got_tuples[k], exp_tuples[k]
        n_tuple += 1
        ok = g["status"] == e["status"] and reason_class(g.get("reason")) == e["reason"]
        if ok and e["status"] in ("positive_marked", "verified_zero"):
            ok = int(g.get("n_marks", -1)) == e["n_marks"]
        if not ok:
            bad.append((k[0][-30:], k[1], k[2], g["status"], g.get("reason", "")[:40], g.get("n_marks"), "| oracle", e["status"], e["reason"], e["n_marks"]))
    rep.add(f"[{tag}] per-(file,field,page) status / reason class / mark count equal the oracle", not bad,
            f"{len(bad)} of {n_tuple} differ, e.g. {bad[:3]}", phase=ph)
    tally = Counter(e["status"] for e in exp_tuples.values())
    f = summ["fields"]
    got_t = (f["positive_marked_fields"], f["verified_zero_fields"], f["unknown_fields_excluded"], f["conflict_fields_excluded"])
    exp_t = (tally["positive_marked"], tally["verified_zero"], tally["unknown"], tally["conflict"])
    rep.add(f"[{tag}] exact excluded-unproven count = unknown + conflict ({exp_t[2] + exp_t[3]})",
            got_t == exp_t and f["excluded_unproven"] == exp_t[2] + exp_t[3] and f["tuples_total"] == sum(exp_t), f"product {got_t} oracle {exp_t}", phase=ph)
    # ---- annotations
    a = summ["annotations"]
    exp_drop = Counter(r for d in D.values() for r in d["dropped"].values())
    exp_pts = sum(d["points_total"] for d in D.values())
    exp_lab = sum(d["labels_total"] for d in D.values())
    rep.add(f"[{tag}] annotations: points / labels / dropped-by-reason equal the oracle",
            (a["points_total"], a["labels_exported"]) == (exp_pts, exp_lab) and a["dropped_by_reason"] == dict(sorted(exp_drop.items())),
            f"product {a['points_total']},{a['labels_exported']},{a['dropped_by_reason']} oracle {exp_pts},{exp_lab},{dict(exp_drop)}", phase=ph)
    if mdoc is not None:
        un = sum(d["unresolved"] for d in D.values())
        rep.add(f"[{tag}] unresolved manifest entries are listed, not guessed ({un})", len(summ["unresolved_manifest_entries"]) >= un,
                f"product {len(summ['unresolved_manifest_entries'])} oracle>={un}", phase=ph)
    # ---- crops
    crops = man["crops"]
    live = [c for c in crops if not c.get("frozen_from")]
    got_crops = {}
    for c in live:
        rel = rel_of_id.get(c["analysis_id"])
        r_, c_ = c["provenance"]["grid_row"], c["provenance"]["grid_col"]
        ncols = len(files[rel]["grid"]["xs"]) - 1
        got_crops[(rel, c["page"], r_ * ncols + c_)] = c
    exp_crops = {(rel, p, c): v for rel, d in D.items() for (p, c), v in d["crops"].items()}
    rep.add(f"[{tag}] exported crops are EXACTLY the examined tuples ({len(exp_crops)}); none for unknown / conflict", set(got_crops) == set(exp_crops),
            f"missing {list(set(exp_crops) - set(got_crops))[:3]} extra {list(set(got_crops) - set(exp_crops))[:3]}", phase=ph)
    unk_crops = [k for k in got_crops if exp_tuples.get(k, {}).get("status") in ("unknown", "conflict")]
    rep.add(f"[{tag}] no crop for any unknown / conflict tuple", not unk_crops, unk_crops[:3], phase=ph)
    # per-crop content
    snap = S["root"]
    by_src = defaultdict(list)
    for k in sorted(set(got_crops) & set(exp_crops)):
        by_src[k[0]].append(k)
    n_pix = n_lab = n_prov = n_zero = n_zero_ok = 0
    bad_pix, bad_lab, bad_prov, bad_norm = [], [], [], []
    norm_done = {}
    norm_notes = Counter()
    for si, (rel, ks) in enumerate(sorted(by_src.items())):
        if pixel_files_limit and si >= pixel_files_limit:
            break
        import tifffile
        with tifffile.TiffFile(str(root / rel)) as tf:
            pages = [pg.asarray() for pg in tf.pages]
        rec = files[rel]
        grid = ms.Grid(**rec["grid"])
        for k in ks:
            c, e = got_crops[k], exp_crops[k]
            x0, y0, x1, y1 = e["box"]
            img = tifffile.imread(str(snap / c["path_tif"]))
            n_pix += 1
            if not (list(c["box_xyxy"]) == e["box"] and img.dtype == pages[k[1]].dtype and np.array_equal(img, pages[k[1]][y0:y1, x0:x1])):
                bad_pix.append((rel[-25:], k[1], k[2]))
            with open(snap / c["path_csv"], newline="", encoding="utf-8") as fh:
                rd = csv.reader(fh)
                head = next(rd)
                lab = np.array([[float(a_), float(b_)] for a_, b_ in rd], float).reshape(-1, 2)
            n_lab += 1
            elab = np.array(e["labels"], float).reshape(-1, 2)
            if head != ["y", "x"] or lab.shape != elab.shape or (lab.size and np.abs(lab[np.lexsort((lab[:, 1], lab[:, 0]))] -
                                                                                    elab[np.lexsort((elab[:, 1], elab[:, 0]))]).max() > 1e-9) \
                    or c["n_positive"] != e["n_positive"]:
                bad_lab.append((rel[-25:], k[1], k[2], lab.shape, elab.shape))
            if e["status"] == "verified_zero":
                n_zero += 1
                n_zero_ok += int(lab.shape[0] == 0)
            pv = c["provenance"]
            need = dict(researcher_examined=True, examination_status=e["status"], source_sha256=rec["sha256"], page=k[1], imagej_position=k[1] + 1,
                        channel_role=e["role"], channel_mapping_origin=rec["mapping_origin"], source_tiff=rel)
            n_prov += 1
            wrong = {a_: (pv.get(a_), b_) for a_, b_ in need.items() if pv.get(a_) != b_}
            if sorted(pv.get("field_counters", [])) != D[rel]["counters"].get(k[2], []):
                wrong["field_counters"] = (pv.get("field_counters"), D[rel]["counters"].get(k[2], []))
            if e["status"] == "positive_marked" and pv.get("n_researcher_marks") != e["n_marks"]:
                wrong["n_researcher_marks"] = (pv.get("n_researcher_marks"), e["n_marks"])
            if wrong:
                bad_prov.append((rel[-25:], k[1], k[2], wrong))
            if c["split"] not in ("train", "val", "test"):
                bad_prov.append(("split", c["split"]))
            key = (rel, k[1])
            if key not in norm_done:
                elig = [grid.box(cc) for (rr, pp, cc) in exp_crops if rr == rel and pp == k[1]]
                lo, hi, note = expected_norm(pages, k[1], grid, elig, norm_mode)
                norm_done[key] = (lo, hi)
                norm_notes[note or norm_mode] += 1
            lo, hi = norm_done[key]
            if c["norm_lo"] is None or abs(c["norm_lo"] - lo) > 1e-6 * max(1, abs(lo)) or abs(c["norm_hi"] - hi) > 1e-6 * max(1, abs(hi)):
                bad_norm.append((rel[-25:], k[1], c["norm_lo"], lo, c["norm_hi"], hi))
    lim = " (first %d files)" % pixel_files_limit if pixel_files_limit else ""
    rep.add(f"[{tag}] every exported crop's pixels equal the source crop bit-exact, dtype preserved, box equal ({n_pix}){lim}", not bad_pix, bad_pix[:3], phase=ph)
    rep.add(f"[{tag}] every crop's label CSV equals the oracle labels within 1e-9 (crop frame, native) ({n_lab}){lim}", not bad_lab, bad_lab[:3], phase=ph)
    rep.add(f"[{tag}] provenance on every crop: researcher_examined, status, sha, page, position, role, mapping origin, counters ({n_prov}){lim}",
            not bad_prov, bad_prov[:2], phase=ph)
    if n_zero or mode != "marked-only":
        rep.add(f"[{tag}] verified-zero crops have EMPTY label files ({n_zero_ok}/{n_zero})", n_zero == n_zero_ok and (n_zero > 0 or mode == "marked-only"),
                f"{n_zero_ok}/{n_zero}", phase=ph)
    rep.add(f"[{tag}] page-level lo/hi recomputed independently ({norm_mode}; {dict(norm_notes)}) equal the manifest ({len(norm_done)} pages){lim}",
            not bad_norm, bad_norm[:3], phase=ph)
    # recorded in manifest sources too
    # ---- downscale label mapping (independent formula) on a sample of crops
    rep_downscale_roundtrip(rep, tag, snap, [got_crops[k] for k in sorted(set(got_crops) & set(exp_crops))][:: max(1, len(got_crops) // 40)], exp_crops, got_crops, ph)
    # ---- leakage audit (independent of the product's split code)
    leak_audit(rep, tag, man, files, ph)
    return dict(man=man, summ=summ, D=D, exp_crops=exp_crops, got_crops=got_crops, n_crops=len(exp_crops), exp=exp)


def rep_downscale_roundtrip(rep, tag, snap, crops, exp_crops, got_crops, ph):
    from polony_app.ml import common as cm
    import tifffile
    inv = {id(v): k for k, v in got_crops.items()}
    worst, n = 0.0, 0
    bad = []
    for c in crops:
        d = int(c["downscale"])
        e = exp_crops[inv[id(c)]]
        img = tifffile.imread(str(snap / c["path_tif"]))
        lab = np.array(e["labels"], float).reshape(-1, 2)
        x0, y0 = c["box_xyxy"][:2]
        oy, ox = (-y0) % d, (-x0) % d
        x, labs, dropped, off = cm.prepare_crop(img, lab, lo=c["norm_lo"], hi=c["norm_hi"], downscale=d, normaliser=c["normaliser"], origin_xy=(x0, y0))
        h, w = img.shape
        keep = [(y - oy, xx - ox) for y, xx in lab if -0.5 <= (y - oy - (d - 1) / 2) / d < (h - oy) // d - 0.5 and -0.5 <= (xx - ox - (d - 1) / 2) / d < (w - ox) // d - 0.5]
        exp_s = np.array([((y - (d - 1) / 2) / d, (xx - (d - 1) / 2) / d) for y, xx in keep], float).reshape(-1, 2)
        n += 1
        if labs.shape != exp_s.shape or (labs.size and np.abs(labs - exp_s).max() > 1e-4):
            bad.append((c["crop_id"], labs.shape, exp_s.shape))
            continue
        # round trip: back to native pixel-centre coordinates
        back = exp_s * d + (d - 1) / 2.0
        if keep:
            worst = max(worst, float(np.abs(back - np.array(keep)).max()))
        # independent block mean of the normalised crop
        ref = img[oy:oy + (h - oy) // d * d, ox:ox + (w - ox) // d * d].astype(np.float64)
        ref = (ref - c["norm_lo"]) / (c["norm_hi"] - c["norm_lo"])
        ref = ref.reshape((h - oy) // d, d, (w - ox) // d, d).mean(axis=(1, 3)) if d > 1 else ref
        if x.shape != ref.shape or np.abs(x - ref).max() > 1e-4:
            bad.append((c["crop_id"], "block mean differs", float(np.abs(x - ref).max()) if x.shape == ref.shape else (x.shape, ref.shape)))
    rep.add(f"[{tag}] downscale d: label mapping x_s=(x-(d-1)/2)/d (after grid trim), exact round trip (worst {worst:.1e}) and block mean equal the "
            f"independent formula ({n} crops)", not bad and worst < 1e-9, bad[:3], phase=ph)


def leak_audit(rep, tag, man, files, ph):
    """Groups / specimens / slides / sha256 never in two splits; crops inherit the split of their source. Uses ONLY the manifest JSON and
    the oracle identities (no product code)."""
    src = {s["analysis_id"]: s for s in man["sources"]}
    by = {"specimen": defaultdict(set), "slide": defaultdict(set), "sha": defaultdict(set), "group": defaultdict(set)}
    bad_inherit = []
    for c in man["crops"]:
        s = src.get(c["analysis_id"])
        if s is None:
            bad_inherit.append(c["crop_id"])
            continue
        if c.get("frozen_from"):
            continue
        rec = files.get(s["name"])
        if rec is None:
            continue
        sp = c["split"]
        if s["split"] != sp:
            bad_inherit.append(c["crop_id"])
        i = rec["identity"]
        by["specimen"][(i["assay"], i["specimen_id"])].add(sp)
        by["slide"][i["slide_id"]].add(sp)
        by["sha"][rec["sha256"]].add(sp)
        by["group"][c["group_id"]].add(sp)
    multi = {k: {a: sorted(b) for a, b in v.items() if len(b) > 1} for k, v in by.items()}
    rep.add(f"[{tag}] leakage audit: no specimen / slide / sha256 / group in two splits; crops inherit their source's split",
            not any(multi.values()) and not bad_inherit, {k: list(v.items())[:2] for k, v in multi.items() if v} or bad_inherit[:3], phase=ph)
    splits = Counter(c["split"] for c in man["crops"])
    rep.add(f"[{tag}] train, val and test are all populated ({dict(splits)})", all(splits.get(k) for k in ("train", "val", "test")), dict(splits), phase=ph)


# ====================================================================================== phases
def phase_templates(ctx, rep, out, files, inp, fast=False):
    tpl = out / "tpl"
    r = ctx.run("write-templates", ["import-legacy", "--input", inp, "--write-templates", tpl], data_dir=out / "data_tpl")
    ok = r["exit"] == 0 and not (out / "data_tpl").exists()
    rep.add("write-templates: exit 0, writes only the three templates (no data dir created)", ok, r["err"][-200:], phase="templates")
    ident = json.loads((tpl / "identities.template.json").read_text())["rows"]
    rep.add("identity template lists exactly the visible TIFFs; all confirmed:false", {x["file"] for x in ident} == set(files) and not any(x["confirmed"] for x in ident),
            f"{len(ident)} rows vs {len(files)} files", phase="templates")
    wrong = [x["file"] for x in ident if x["specimen_id"] and x["specimen_id"] != files[x["file"]]["identity"]["specimen_id"]]
    rep.add("inferred (unconfirmed) specimen ids that the file name yields equal the generator's", not wrong, wrong[:3], phase="templates")
    if fast:
        return
    r = ctx.run("template channel map used as-is", ["import-legacy", "--input", inp, "--channel-map", tpl / "channels.template.json", "--dry-run"], data_dir=out / "data_tpl")
    rep.add("unconfirmed template channel map is refused (exit 2, says who must confirm)", r["exit"] == 2 and "confirmed_by" in r["err"], r["err"][-200:], phase="templates")


def phase_refusals(ctx, rep, out, variants, inp, M, fast):
    names = sorted(variants)
    if fast:
        pick = ["missing_confirmed_by", "id_unconfirmed_rows"]
        names = [n for n in names if n in pick]
    bad = []
    for n in names:
        v = variants[n]
        base = ["import-legacy", "--input", inp]
        ch, idn = str(M / "channels.json"), str(M / "identities.json")
        extra = []
        if v["kind"] == "channels":
            ch = v["path"]
        elif v["kind"] == "identity":
            idn = v["path"]
        else:
            extra = ["--mode", "examined-fields-manifest", "--examined-fields", v["path"]]
        a = base + ["--channel-map", ch, "--identity-map", idn] + extra
        dd = out / "data_refuse"
        if dd.exists():
            shutil.rmtree(dd)
        rd = ctx.run(f"refusal dry-run {n}", a + ["--dry-run"], data_dir=dd)
        rr = ctx.run(f"refusal real {n}", a, data_dir=dd)
        dry_ok = v["dry_exit"] if isinstance(v["dry_exit"], list) else [v["dry_exit"]]
        txt = (rd["err"] + rd["out"] + rr["err"] + rr["out"]).lower()
        real_ok = v["real_exit"] if isinstance(v["real_exit"], list) else [v["real_exit"]]
        ok = rd["exit"] in dry_ok and rr["exit"] in real_ok and any(t.lower() in txt for t in v["any_text"]) and (not snapshots(dd) or rr["exit"] == 0)
        if rr["exit"] == 2 and "Traceback" in rr["err"]:
            ok = False
        if not ok:
            bad.append((n, rd["exit"], rr["exit"], (rr["err"] or rd["out"])[-160:].replace("\n", " ")))
        else:
            rep.add(f"refusal '{n}': dry-run exit {rd['exit']}, import exit {rr['exit']}, actionable text, no snapshot written", True, "", phase="refusals")
    for b in bad:
        rep.add(f"refusal '{b[0]}': exits {b[1]}/{b[2]}, actionable text, nothing written", False, b[3], phase="refusals")
    if fast:
        return
    r = ctx.run("import without channel map", ["import-legacy", "--input", inp, "--identity-map", M / "identities.json"], data_dir=out / "data_refuse")
    rep.add("real import without a channel map is refused (exit 2)", r["exit"] == 2 and not snapshots(out / "data_refuse"), r["err"][-200:], phase="refusals")
    r = ctx.run("bad fractions", ["import-legacy", "--input", inp, "--channel-map", M / "channels.json", "--identity-map", M / "identities.json",
                                  "--fractions", "0.7,abc", "--dry-run"], data_dir=out / "data_refuse")
    rep.add("malformed --fractions exits 2 with the expected format", r["exit"] == 2 and "--fractions" in r["err"], r["err"][-200:], phase="refusals")


COMMON = ["--seed", 1, "--fractions", "0.7,0.15,0.15", "--created-at", T0, "--batch-size", 4, "--epochs", 10]


def import_args(inp, M, downscale, workers, extra=()):
    return ["import-legacy", "--input", inp, "--channel-map", M / "channels.json", "--identity-map", M / "identities.json", "--downscale", downscale,
            "--workers", workers, *COMMON, *extra]


def phase_dry_run(ctx, rep, out, inp, M, downscale, files, ch, norm_mode):
    guard = [inp, M]
    before = [tree_stat_hash(p) for p in guard]
    dd = out / "data_dry"
    r = ctx.run("import-legacy --dry-run", import_args(inp, M, downscale, 4, ["--dry-run"]), data_dir=dd)
    after = [tree_stat_hash(p) for p in guard]
    rep.add("dry-run wrote nothing: input and mapping trees unchanged (path/size/mtime hash) and no data dir created", r["exit"] == 0 and before == after and not dd.exists(),
            r["err"][-300:], phase="dry-run")
    txt = r["out"]
    m = re.search(r"SOURCES: discovered (\d+), unique (\d+), accepted (\d+), rejected (\d+), deduplicated[^\d]*(\d+)", txt)
    kept, alias = ms.dedupe_groups(files)
    exp = ms.expected_status(files, ch)
    n_rej = sum(1 for v in exp.values() if v["status"] == "rejected")
    expv = (len(files), len(kept), len(kept) - n_rej, n_rej, sum(len(v) for v in alias.values()))
    rep.add("dry-run SOURCES line equals the oracle", bool(m) and tuple(map(int, m.groups())) == expv, f"{m.groups() if m else txt[:200]} vs {expv}", phase="dry-run")
    mr = re.search(r"rejected by reason: (.*)", txt)
    rep.add("dry-run rejected-by-reason equals the oracle", bool(mr) and kv(mr.group(1)) == dict(Counter(v["reason"] for v in exp.values() if v["status"] == "rejected")),
            mr.group(1) if mr else "none", phase="dry-run")
    D = {rel: ms.derive(files[rel], {int(k): v for k, v in files[rel]["mapping_pages"].items()}) for rel, v in exp.items() if v["status"] == "accepted"}
    tally = Counter(e["status"] for d in D.values() for e in d["tuples"].values())
    m2 = re.search(r"EXCLUDED because the researcher's examination could not be proven: (\d+) \(unknown (\d+), conflict (\d+)\)", txt)
    rep.add(f"dry-run exact excluded-unproven count ({tally['unknown'] + tally['conflict']} = unknown {tally['unknown']} + conflict {tally['conflict']})",
            bool(m2) and tuple(map(int, m2.groups())) == (tally["unknown"] + tally["conflict"], tally["unknown"], tally["conflict"]), m2.groups() if m2 else txt[:200], phase="dry-run")
    m3 = re.search(r"examined and eligible (\d+) \(positive_marked (\d+), verified_zero (\d+)\)", txt)
    rep.add("dry-run examined / positive_marked / verified_zero counts equal the oracle",
            bool(m3) and tuple(map(int, m3.groups())) == (tally["positive_marked"] + tally["verified_zero"], tally["positive_marked"], tally["verified_zero"]),
            m3.groups() if m3 else "", phase="dry-run")
    m4 = re.search(r"ANNOTATIONS: (\d+) ImageJ point\(s\); (\d+) exported as labels; (\d+) NOT labels", txt)
    rep.add("dry-run ANNOTATIONS (points / labels / not labels) equal the oracle",
            bool(m4) and tuple(map(int, m4.groups())) == (sum(d["points_total"] for d in D.values()), sum(d["labels_total"] for d in D.values()),
                                                           sum(d["points_total"] - d["labels_total"] for d in D.values())), m4.groups() if m4 else "", phase="dry-run")
    return r


def phase_import(ctx, rep, out, inp, M, downscale, workers, files, ch, norm_mode, kill_n, fast=False):
    res = {}
    for w in (1, workers):
        dd = out / f"data_w{w}"
        if dd.exists():
            shutil.rmtree(dd)
        r = ctx.run(f"import-legacy workers={w}", import_args(inp, M, downscale, w), data_dir=dd)
        ok = r["exit"] == 0 and len(snapshots(dd)) == 1
        rep.add(f"real import workers={w}: exit 0, exactly one published snapshot, no staging left", ok and not list((dd / "datasets").glob(".staging*")),
                r["err"][-300:], phase="import")
        res[w] = (dd, snapshots(dd)[0] if snapshots(dd) else None, r)
    (d1, v1, r1), (d4, v4, r4) = res[1], res[workers]
    rep.add("workers 1 vs %d: identical snapshot version and byte-identical snapshot tree" % workers,
            v1 == v4 and tree_content_hashes(d1 / "datasets" / v1) == tree_content_hashes(d4 / "datasets" / v4), f"{v1} vs {v4}", phase="determinism")
    # determinism: a THIRD import (different workers again, fresh dir)
    if not fast:
        d3 = out / "data_w2"
        if d3.exists():
            shutil.rmtree(d3)
        ctx.run("import-legacy workers=2 (determinism)", import_args(inp, M, downscale, 2), data_dir=d3)
        v3 = snapshots(d3)[0] if snapshots(d3) else None
        rep.add("rerun in a fresh data dir with another worker count gives the same version", v3 == v1, f"{v3} vs {v1}", phase="determinism")
    # idempotent re-import (architect notice 6): same input again in the same data dir
    rr = ctx.run("import-legacy again (idempotent)", import_args(inp, M, downscale, workers), data_dir=d4)
    rep.add("re-importing identical input is idempotent: exit 0, says 'already imported as <version>', no new snapshot",
            rr["exit"] == 0 and v4 in (rr["out"] + rr["err"]) and "already" in (rr["out"] + rr["err"]).lower() and snapshots(d4) == [v4],
            f"exit {rr['exit']}: {(rr['err'] or rr['out'])[-200:]}", known="architect change (6) 'already imported' may not have landed yet" if rr["exit"] != 0 else None, phase="determinism")
    # verify + tamper
    rv = ctx.run("verify", ["verify", v1], data_dir=d1)
    rep.add("verify: exit 0 'intact'", rv["exit"] == 0 and "intact" in rv["out"], rv["out"][-100:], phase="verify")
    rep.perf["verify_s"] = rv["wall"]
    tamper(ctx, rep, out, d1, v1, fast)
    return dict(d1=d1, v1=v1, d4=d4, v4=v4, r1=r1, r4=r4)


def tamper(ctx, rep, out, d1, v1, fast=False):
    src = d1 / "datasets" / v1
    man = json.loads((src / "manifest.json").read_text())
    crop_rel = man["crops"][len(man["crops"]) // 2]["path_tif"]
    csv_rel = man["crops"][len(man["crops"]) // 3]["path_csv"]
    for name, mut in (("one flipped byte in a crop TIFF", lambda td: _flip(td / "datasets" / v1 / crop_rel)),
                      ("a label CSV edited", lambda td: _append(td / "datasets" / v1 / csv_rel)),
                      ("a crop file deleted", lambda td: _rm(td / "datasets" / v1 / crop_rel)),
                      ("manifest.json edited", lambda td: _manifest(td / "datasets" / v1 / "manifest.json"))):
        if fast and name in ("a label CSV edited", "a crop file deleted", "manifest.json edited"):
            continue
        td = out / "data_tamper"
        if td.exists():
            subprocess.run(["chmod", "-R", "u+w", str(td)])
            shutil.rmtree(td)
        (td / "datasets").mkdir(parents=True)
        subprocess.run(["cp", "-al", str(src), str(td / "datasets" / v1)], check=True)         # hard links: cheap, un-linked before editing
        mut(td)
        r = ctx.run(f"verify after tamper: {name}", ["verify", v1], data_dir=td)
        rep.add(f"tamper ({name}) -> verify fails (exit 3, problem named)", r["exit"] == 3 and "PROBLEMS" in r["out"], r["out"][-160:], phase="verify")
    subprocess.run(["chmod", "-R", "u+w", str(out / "data_tamper")])
    shutil.rmtree(out / "data_tamper")
    if fast:
        return
    # the pristine snapshot was not touched through the hard links
    r = ctx.run("verify pristine again", ["verify", v1], data_dir=d1)
    rep.add("hard-link tampering never touched the real snapshot (verify still intact)", r["exit"] == 0, r["out"][-100:], phase="verify")


def _unlink_writable(p: Path):
    os.chmod(p.parent, 0o755)
    data = p.read_bytes()
    os.unlink(p)
    return data


def _flip(p):
    d = bytearray(_unlink_writable(p))
    d[len(d) // 2] ^= 0x01
    p.write_bytes(bytes(d))


def _append(p):
    d = _unlink_writable(p)
    p.write_bytes(d + b"0.0,0.0\n")


def _rm(p):
    _unlink_writable(p)


def _manifest(p):
    d = json.loads(_unlink_writable(p))
    d["warnings"] = d.get("warnings", []) + ["tampered"]
    p.write_text(json.dumps(d))


def phase_lock(ctx, rep, out, info, files, inp, M, ch, downscale, workers, spec, norm_mode, fast=False):
    d1, v1 = info["d1"], info["v1"]
    r = ctx.run("split (inspect)", ["split", "--snapshot", v1], data_dir=d1) if not fast else dict(exit=0, out="", err="")
    rep.add("split (no lock) prints the group table and writes no lock", r["exit"] == 0 and not (d1 / "datasets" / "test_lock.json").exists(), r["err"][-200:], phase="lock")
    man = json.loads((d1 / "datasets" / v1 / "manifest.json").read_text())
    groups = {l.split()[1] for l in r["out"].splitlines() if re.match(r"\s+(train|val|test)\s", l)}
    if not fast:
      rep.add("split table lists every group of the manifest", groups == {s["group_id"] for s in man["sources"] if s.get("group_id")} or groups == {c["group_id"] for c in man["crops"]},
            f"{len(groups)} groups", phase="lock")
    t_pre = {c["crop_id"]: (c["tif_sha256"], c["csv_sha256"], ms.file_sha256(d1 / "datasets" / v1 / c["path_tif"])) for c in man["crops"] if c["split"] == "test"}
    r = ctx.run("split --lock-test", ["split", "--snapshot", v1, "--lock-test"], data_dir=d1)
    lock = json.loads((d1 / "datasets" / "test_lock.json").read_text()) if (d1 / "datasets" / "test_lock.json").exists() else {}
    rep.add("lock-test: exit 0, lock file written for the snapshot's test groups", r["exit"] == 0 and lock.get("snapshot") == v1 and
            set(lock.get("groups", [])) == {c["group_id"] for c in man["crops"] if c["split"] == "test"}, r["err"][-200:], phase="lock")
    r2 = ctx.run("split --lock-test (second)", ["split", "--snapshot", v1, "--lock-test"], data_dir=d1)
    rep.add("a second lock is refused (permanent, exit 2)", r2["exit"] == 2 and "permanent" in r2["err"], r2["err"][-200:], phase="lock")
    # test crops are exactly the lock's groups; test untouched byte for byte after the lock
    t_post = {c["crop_id"]: (c["tif_sha256"], c["csv_sha256"], ms.file_sha256(d1 / "datasets" / v1 / c["path_tif"])) for c in man["crops"] if c["split"] == "test"}
    rep.add("locked test crops untouched by the lock (hashes before == after)", t_pre == t_post, f"{len(t_post)} crops", phase="lock")
    # ---- extras after the lock
    kept, _ = ms.dedupe_groups(files)
    test_rels = sorted({s["name"] for s in man["sources"] if s["split"] == "test" and s["n_crops"]})
    extra_dir = out / "input_extra"
    if extra_dir.exists():
        shutil.rmtree(extra_dir)
    ex = ms.make_extras(spec, extra_dir, test_rels, Path(out / "input"), files, n_new=max(3, spec.n // 100))
    exM = out / "mapping_extra"
    exM.mkdir(exist_ok=True)
    ech = json.loads(json.dumps(ch))                       # a rule / file entry that matches no scanned file is a blocker: prune
    hit = {ms.resolve_pages(ch, rec)[0] for rec in ex.values()}
    ech["rules"] = [r for r in ech.get("rules", []) if f"rule:{r['id']}" in hit]
    ech["files"] = {k: v for k, v in ech.get("files", {}).items() if k in ex}
    ms.write_json(exM / "channels.json", ech)
    ms.write_json(exM / "identities.json", {"rows": ms.identity_rows(ex)})
    for rel, rec in ex.items():
        rec["mapping_origin"], pm = ms.resolve_pages(ech, rec)
        rec["mapping_pages"] = {str(k): v for k, v in pm.items()}
    r = ctx.run("import-legacy after the lock (extra data)", import_args(extra_dir, exM, downscale, workers), data_dir=d1)
    vs = [v for v in snapshots(d1) if v != v1]
    rep.add("import of extra data after the lock succeeds (new snapshot, v1 untouched)", r["exit"] == 0 and len(vs) == 1, r["err"][-300:], phase="lock")
    if len(vs) == 1:
        v2 = vs[0]
        m2 = json.loads((d1 / "datasets" / v2 / "manifest.json").read_text())
        frozen = {c["crop_id"]: c for c in m2["crops"] if c["split"] == "test"}
        same = set(frozen) == set(t_pre) and all(frozen[k]["tif_sha256"] == t_pre[k][0] and frozen[k]["csv_sha256"] == t_pre[k][1] and
                                                  ms.file_sha256(d1 / "datasets" / v2 / frozen[k]["path_tif"]) == t_pre[k][2] and frozen[k].get("frozen_from") == v1 for k in frozen)
        rep.add("v2 test crops == v1 test crops byte for byte, all marked frozen_from v1; no new test crops", same, f"{len(frozen)} vs {len(t_pre)}", phase="lock")
        src2 = {s["name"]: s for s in m2["sources"]}
        locked_exp = [rel for rel, r_ in ex.items() if r_["expect"] in ("locked_specimen", "locked_bytes")]
        notlocked = [rel for rel in locked_exp if src2.get(rel, {}).get("n_crops") or src2.get(rel, {}).get("split") != "test"]
        rep.add(f"extra files of a locked specimen / byte-identical to locked test files are test-only and NOT exported ({len(locked_exp)})", not notlocked and bool(locked_exp),
                notlocked[:3], phase="lock")
        newc = [c for c in m2["crops"] if not c.get("frozen_from")]
        rep.add("new crops after the lock are train/val only (no new test group)", all(c["split"] in ("train", "val") for c in newc) and len(newc) > 0, Counter(c["split"] for c in newc), phase="lock")
        # independent: no train/val source shares specimen / slide / sha with a locked test source
        lock_spec = {(files[s]["identity"]["assay"], files[s]["identity"]["specimen_id"]) for s in test_rels}
        lock_sha = {files[s]["sha256"] for s in test_rels}
        leaked = []
        for c in newc:
            n = next(s["name"] for s in m2["sources"] if s["analysis_id"] == c["analysis_id"])
            i = ex[n]["identity"]
            if (i["assay"], i["specimen_id"]) in lock_spec or ex[n]["sha256"] in lock_sha:
                leaked.append(n)
        rep.add("train/val after the lock never include locked specimens or locked bytes", not leaked, leaked[:3], phase="lock")
        if not fast:
            rv = ctx.run("verify v2", ["verify", v2], data_dir=d1)
            rep.add("verify v2 intact", rv["exit"] == 0, rv["out"][-100:], phase="lock")
    rep.add("v1 still verifies and its lock unchanged after the second import", (fast or ctx.run("verify v1", ["verify", v1], data_dir=d1)["exit"] == 0) and
            json.loads((d1 / "datasets" / "test_lock.json").read_text()).get("lock_sha256") == lock.get("lock_sha256"), "", phase="lock")
    return dict(lock=lock, t_pre=t_pre)


_FRESH = {}


def fresh_reference(ctx, out, inp, M, downscale, ref):
    """If the product code changed while the rehearsal ran, a snapshot built earlier is no longer the right reference: build a fresh one."""
    key = (str(inp), downscale)
    if key not in _FRESH:
        dd = out / "data_fresh_ref"
        if dd.exists():
            shutil.rmtree(dd)
        ctx.run("fresh uninterrupted reference import (code changed during the run?)", import_args(inp, M, downscale, 2), data_dir=dd)
        v = snapshots(dd)[0] if snapshots(dd) else None
        _FRESH[key] = (v, tree_content_hashes(dd / "datasets" / v) if v else {})
        ref_note = "the product changed during the rehearsal (version of the first import != a fresh import): resume/ENOSPC compared with a fresh reference"
        ctx.rep.notes.append(ref_note)
    return _FRESH[key]


def phase_kill_resume(ctx, rep, out, inp, M, downscale, ref, n_expected_crops, n_sources, fast=False):
    dd = out / "data_kill"
    if dd.exists():
        shutil.rmtree(dd)
    slow = int(min(100, max(2, (3000 if fast else 8000) / max(1, n_expected_crops))))
    p = ctx.run("import (to be killed -9)", import_args(inp, M, downscale, 2), data_dir=dd, wrapper=True, env={"REHEARSAL_SLOW_WRITE_MS": str(slow)}, popen=True)
    target = max(2, n_sources // 4)
    done = 0
    t_end = time.time() + 600
    while time.time() < t_end and p["proc"].poll() is None:
        done = len(list((dd / "datasets").glob(".staging-*/.done/*.json")))
        if done >= target:
            break
        time.sleep(0.02)
    alive = p["proc"].poll() is None
    if alive:
        p["proc"].send_signal(signal.SIGKILL)
    r = ctx._finish("import killed -9", p["proc"], *p["files"], p["t"], p["so"], p["se"], p["cmd"])
    killed = alive and r["exit"] == -signal.SIGKILL
    rep.add(f"real kill -9 of the import subprocess mid-way (after {done}/{n_sources} source markers)", killed, f"alive={alive} exit={r['exit']} markers={done}", phase="resume")
    if not killed:
        return
    stg = list((dd / "datasets").glob(".staging-*"))
    rep.add("after the kill: no published snapshot, a staging dir with verified source markers remains", not snapshots(dd) and bool(stg) and done >= 1, f"snapshots={snapshots(dd)} staging={[s.name for s in stg]}", phase="resume")
    r2 = ctx.run("import resumed after kill", import_args(inp, M, downscale, 2), data_dir=dd)
    vk = snapshots(dd)
    same = len(vk) == 1 and vk[0] == ref["v1"] and tree_content_hashes(dd / "datasets" / vk[0]) == tree_content_hashes(ref["d1"] / "datasets" / ref["v1"])
    if not same and len(vk) == 1:
        fv, fh = fresh_reference(ctx, out, inp, M, downscale, ref)
        same = vk[0] == fv and tree_content_hashes(dd / "datasets" / vk[0]) == fh
    rep.add("rerun after kill -9 resumes and publishes the IDENTICAL version / bytes as an uninterrupted import; no staging left",
            r2["exit"] == 0 and same and not list((dd / "datasets").glob(".staging*")), f"exit {r2['exit']} {vk} vs {ref['v1']}; {r2['err'][-200:]}", phase="resume")
    rep.perf["resume_wall_s"] = r2["wall"]


def phase_disk(ctx, rep, out, inp, M, downscale, ref, n_expected_crops, fast=False):
    dd = out / "data_disk"
    if dd.exists():
        shutil.rmtree(dd)
    r = ctx.run("import with disk full (shutil.disk_usage patched to 2 MB free)", import_args(inp, M, downscale, 2), data_dir=dd, wrapper=True, env={"REHEARSAL_DISK_FREE": 2_000_000})
    left = [p.name for p in (dd / "datasets").iterdir()] if (dd / "datasets").exists() else []
    rep.add("disk-full: import refused (exit 2, 'not enough free disk space'), nothing written (no staging, no snapshot)", r["exit"] == 2 and "free disk space" in r["err"] and not left,
            f"exit {r['exit']} left {left} {r['err'][-160:]}", phase="disk")
    if fast:
        rd = dict(exit=0, out="BLOCKER: not enough free disk")
    else:
        rd = ctx.run("dry-run with disk full", import_args(inp, M, downscale, 2, ["--dry-run"]), data_dir=dd, wrapper=True, env={"REHEARSAL_DISK_FREE": 2_000_000})
    if not fast:
        rep.add("disk-full: dry-run exit 0 and prints a BLOCKER for the disk", rd["exit"] == 0 and "BLOCKER: not enough free disk" in rd["out"], rd["out"][-200:], phase="disk")
    if fast:
        return
    # ENOSPC in the middle of writing crops
    dd2 = out / "data_enospc"
    if dd2.exists():
        shutil.rmtree(dd2)
    after = max(5, n_expected_crops // 3)
    r = ctx.run("import with ENOSPC injected mid-write", import_args(inp, M, downscale, 1), data_dir=dd2, wrapper=True, env={"REHEARSAL_ENOSPC_AFTER": after})
    rep.add("ENOSPC mid-write: non-zero exit, nothing published", r["exit"] != 0 and not snapshots(dd2), f"exit {r['exit']}", phase="disk")
    rep.add("ENOSPC mid-write is reported as a refusal (exit 2, no traceback)", r["exit"] == 2 and "Traceback" not in r["err"], f"exit {r['exit']}: {r['err'][-200:]}",
            known="OSError is not an MLError: the CLI prints a traceback and exits 1 (src/polony_app/ml/cli.py main() catches MLError only)" if r["exit"] != 2 else None, phase="disk")
    r = ctx.run("import after the disk problem is fixed", import_args(inp, M, downscale, 1), data_dir=dd2)
    same = snapshots(dd2) == [ref["v1"]]
    if not same and len(snapshots(dd2)) == 1:
        same = snapshots(dd2)[0] == fresh_reference(ctx, out, inp, M, downscale, ref)[0]
    rep.add("after ENOSPC the same command resumes and publishes the identical version", r["exit"] == 0 and same, f"{snapshots(dd2)} {r['err'][-200:]}", phase="disk")


def make_base_models(ctx, out):
    base, general = out / "tiny_base", out / "general_stand_in"
    code = ("import sys,torch\nfrom spotiflow.model import Spotiflow, SpotiflowModelConfig\n"
            "for p,s in ((sys.argv[1],0),(sys.argv[2],123)):\n    torch.manual_seed(s)\n"
            "    Spotiflow(SpotiflowModelConfig(initial_fmaps=8, levels=2, sigma=1.0)).save(p, which='best')\n")
    subprocess.run([sys.executable, "-c", code, str(base), str(general)], check=True, env=dict(os.environ, PYTHONPATH=SRC))
    return base, general


def fmt_bytes(n):                                   # same display rule as the CLI (independent re-implementation)
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def phase_train_dry(ctx, rep, out, d1, v1, base, files, tag="full", fast=False):
    man = json.loads((d1 / "datasets" / v1 / "manifest.json").read_text())
    for label, extra, bs, ep, nts, mv in (("default", [], 4, 10, None, None), ("bounded", ["--num-train-samples", 40, "--max-val-crops", 10], 4, 3, 40, 10)):
        if fast and label == "bounded":
            continue
        args = ["train", "--snapshot", v1, "--base", base, "--epochs", ep, "--batch-size", bs, "--crop-size", 32, "--device", "cpu", "--dry-run", *extra]
        before = tree_stat_hash(d1)
        r = ctx.run(f"train --dry-run ({label})", args, data_dir=d1)
        rep.add(f"[{tag}] train --dry-run ({label}) exits 0 and writes nothing (no models dir, tree unchanged)", r["exit"] == 0 and before == tree_stat_hash(d1) and not (d1 / "models").exists(),
                r["err"][-300:], phase="train-dry")
        txt = r["out"]
        crops = man["crops"]
        tr = [c for c in crops if c["split"] == "train"]
        va = sorted([c for c in crops if c["split"] == "val"], key=lambda c: hashlib.sha256(c["crop_id"].encode()).hexdigest()) if mv else [c for c in crops if c["split"] == "val"]
        if mv and mv < len(va):
            va = va[:mv]
        d = int(tr[0]["downscale"])

        def prep(c):
            h, w = c["shape_hw"]
            oy, ox = (-c["box_xyxy"][1]) % d, (-c["box_xyxy"][0]) % d
            return ((h - oy) // d) * ((w - ox) // d) * 4
        sp = nts if nts else len(tr)
        spe = math.ceil(sp / bs)
        pt, pv = sum(prep(c) for c in tr), sum(prep(c) for c in va)
        exp_lines = {
            "eligible": f"eligible crops: {len(crops)}  (train {len(tr)}, val {sum(1 for c in crops if c['split'] == 'val')}, test {sum(1 for c in crops if c['split'] == 'test')}",
            "steps": f"steps per epoch: {spe}   estimated optimizer steps: {spe * ep}",
            "val": f"validation: {len(va)} batch(es) per pass",
            "valtot": f"= {len(va) * ep} batches, + {min(2, len(va))} sanity-check batch(es) before epoch 1, + {2 * len(va)} for the final threshold optimisation",
            "prep": f"prepared in-memory size (float32 network inputs): {fmt_bytes(pt + pv)} (train {fmt_bytes(pt)}, val {fmt_bytes(pv)})"}
        miss = {k: v for k, v in exp_lines.items() if v not in txt}
        rep.add(f"[{tag}] train --dry-run ({label}): printed crops/steps/validation batches/prepared bytes equal independent arithmetic", not miss,
                f"missing {miss} in: {[l for l in txt.splitlines() if 'steps per' in l or 'validation:' in l or 'prepared' in l or 'eligible' in l]}", phase="train-dry")


def phase_model(ctx, rep, out, spec_n, seed, workers, base, general, cell_px):
    """Tiny CPU fine-tune on a SMALL separate snapshot: import -> lock -> train -> eval-model x2 -> compare, with oracle checks."""
    mini = out / "mini"
    sp = ms.Spec(n=spec_n, seed=seed + 11, cell_px=cell_px, workers=1)
    orc = ms.generate(sp, mini)
    files, ch = orc["files"], json.loads((mini / "mapping" / "channels.json").read_text())
    inp, M = mini / "input", mini / "mapping"
    dd = mini / "data"
    args = ["import-legacy", "--input", inp, "--channel-map", M / "channels.json", "--identity-map", M / "identities.json", "--downscale", 2, "--workers", 2,
            "--seed", 1, "--fractions", "0.5,0.25,0.25", "--created-at", T0]
    r = ctx.run("mini import", args, data_dir=dd)
    vs = snapshots(dd)
    rep.add("[mini] import succeeds", r["exit"] == 0 and len(vs) == 1, r["err"][-300:], phase="model")
    if not vs:
        return
    v = vs[0]
    info = check_snapshot(rep, "mini", dd, v, files, inp, ch, norm_mode=NORM_MODE[0], downscale=2)
    man = info["man"]
    sc = Counter(c["split"] for c in man["crops"])
    if not all(sc.get(k) for k in ("train", "val", "test")):
        rep.skip("[mini] train/eval", f"a split is empty ({dict(sc)})", phase="model")
        return
    r = ctx.run("mini lock", ["split", "--snapshot", v, "--lock-test"], data_dir=dd)
    phase_train_dry(ctx, rep, out, dd, v, base, files, tag="mini")
    targs = ["train", "--snapshot", v, "--base", base, "--epochs", 2, "--batch-size", 2, "--crop-size", 32, "--allow-small", "--device", "cpu",
             "--num-train-samples", 16, "--max-val-crops", 6]
    r = ctx.run("mini train (2 epochs, cpu)", targs, data_dir=dd, wrapper=True, env={"REHEARSAL_GENERAL_STAND_IN": str(general)})
    mid = (re.search(r"model id: (\S+)", r["out"]) or [None, None])[1]
    rep.add("[mini] tiny CPU fine-tune completes and registers a CANDIDATE", r["exit"] == 0 and bool(mid), r["err"][-400:], phase="model")
    rep.perf["mini_train_s"] = r["wall"]
    if not mid:
        return
    from polony_app.ml.registry import Registry
    entry = Registry(dd).get(mid)
    lock = json.loads((dd / "datasets" / "test_lock.json").read_text())
    if entry:
        sm = entry["split_membership"]
        test_ids = {c["crop_id"] for c in man["crops"] if c["split"] == "test"}
        rep.add("[mini] fine-tune never saw a locked test crop or group (train/val/test crop ids and groups disjoint)",
                not (set(sm["train"]) | set(sm["val"])) & test_ids and not (set(sm["train_groups"]) | set(sm["val_groups"])) & set(lock["groups"]),
                "", phase="model")
        rep.add("[mini] entry is a candidate (never active), dataset_version recorded", entry["status"] == "candidate" and entry["dataset_version"] == v, entry["status"], phase="model")
    env = {"REHEARSAL_GENERAL_STAND_IN": str(general)}
    r = ctx.run("mini eval-model candidate", ["eval-model", "--model", mid, "--device", "cpu"], data_dir=dd, wrapper=True, env=env)
    rep.add("[mini] eval-model <candidate> on the locked test crops", r["exit"] == 0, r["err"][-300:], phase="model")
    r2 = ctx.run("mini eval-model general", ["eval-model", "--model", "general", "--device", "cpu", "--downscale", 2], data_dir=dd, wrapper=True, env=env)
    rep.add("[mini] eval-model general (offline stand-in) on the same locked set", r2["exit"] == 0, r2["err"][-300:], phase="model")
    rep.perf["mini_eval_s"] = r["wall"]
    r3 = ctx.run("mini compare", ["compare", "--models", mid, "general"], data_dir=dd, wrapper=True, env=env)
    rep.add("[mini] compare prints both models side by side", r3["exit"] == 0 and mid in r3["out"] and "general" in r3["out"] and "| model |" in r3["out"], r3["err"][-300:], phase="model")
    r4 = ctx.run("mini promote (must be refused or evidence-based, never forced)", ["promote", mid], data_dir=dd, wrapper=True, env=env)
    rep.add("[mini] promote without --force never silently succeeds on missing/worse evidence (exit 0 only with F1 >= general, else 2)", r4["exit"] in (0, 2), r4["err"][-200:], phase="model")
    # manifest mode, real, with verified zeros and deliberate conflicts
    exdoc = json.loads((M / "examined-fields.json").read_text())
    mdoc = {e["file"]: {int(p): ents for p, ents in e["pages"].items()} for e in exdoc["files"]}
    d2 = mini / "data_manifest_mode"
    r = ctx.run("mini import examined-fields-manifest", args + ["--mode", "examined-fields-manifest", "--examined-fields", M / "examined-fields.json"], data_dir=d2)
    vs2 = snapshots(d2)
    rep.add("[mini-manifest] examined-fields-manifest import succeeds", r["exit"] == 0 and len(vs2) == 1, r["err"][-300:], phase="manifest-mode")
    if vs2:
        check_snapshot(rep, "mini-manifest", d2, vs2[0], files, inp, ch, mode="examined-fields-manifest", mdoc=mdoc, norm_mode=NORM_MODE[0], downscale=2)


NORM_MODE = ["eligible-fields"]


def phase_manifest_dry(ctx, rep, out, inp, M, downscale, files, ch):
    exdoc = json.loads((M / "examined-fields.json").read_text())
    mdoc = {e["file"]: {int(p): ents for p, ents in e["pages"].items()} for e in exdoc["files"]}
    r = ctx.run("dry-run examined-fields-manifest", import_args(inp, M, downscale, 2, ["--mode", "examined-fields-manifest", "--examined-fields", M / "examined-fields.json", "--dry-run"]),
                data_dir=out / "data_dry2")
    exp = ms.expected_status(files, ch, "examined-fields-manifest")
    D = {rel: ms.derive(files[rel], {int(k): v for k, v in files[rel]["mapping_pages"].items()}, "examined-fields-manifest", mdoc.get(rel)) for rel, v in exp.items() if v["status"] == "accepted"}
    tally = Counter(e["status"] for d in D.values() for e in d["tuples"].values())
    m2 = re.search(r"EXCLUDED because .*?: (\d+) \(unknown (\d+), conflict (\d+)\)", r["out"])
    m3 = re.search(r"examined and eligible (\d+) \(positive_marked (\d+), verified_zero (\d+)\)", r["out"])
    rep.add(f"[manifest-mode dry-run] verified zeros ({tally['verified_zero']}), conflicts ({tally['conflict']}), unknown ({tally['unknown']}) and examined counts equal the oracle",
            r["exit"] == 0 and bool(m2) and bool(m3) and tuple(map(int, m2.groups())) == (tally["unknown"] + tally["conflict"], tally["unknown"], tally["conflict"]) and
            tuple(map(int, m3.groups())) == (tally["positive_marked"] + tally["verified_zero"], tally["positive_marked"], tally["verified_zero"]),
            f"{m2.groups() if m2 else r['out'][-200:]} {m3.groups() if m3 else ''} oracle {dict(tally)}", phase="manifest-mode")


def phase_truncated(ctx, rep, out):
    d = out / "probe_truncated"
    if d.exists():
        shutil.rmtree(d)
    sp = ms.Spec(n=12, seed=5, cell_px=64)
    orc = ms.generate(sp, d / "gen")
    good = next(r for r in orc["files"].values() if r["kind"] == "good")
    src = d / "gen" / "input" / good["rel"]
    inp = d / "in"
    inp.mkdir()
    shutil.copy(src, inp / "29-09-2026_T7_AB_ZZ1-A.tif")
    data = src.read_bytes()
    (inp / "29-09-2026_T7_AB_ZZ2-A.tif").write_bytes(data[: int(len(data) * 0.6)])
    for n, spec_ in (("ZZ1", "ZZ1"), ("ZZ2", "ZZ2")):
        pass
    rows = [{"file": f"29-09-2026_T7_AB_{s}-A.tif", "specimen_id": s, "slide_id": f"{s}_A", "assay": "T7_AB", "replicate": "A", "confirmed": True} for s in ("ZZ1", "ZZ2")]
    ms.write_json(d / "ids.json", {"rows": rows})
    chd = json.loads((d / "gen" / "mapping" / "channels.json").read_text())
    chd["rules"], chd["files"] = [], {}                                  # unmatched rules / files would be (correct) blockers
    ch = ms.write_json(d / "channels.json", chd)
    r = ctx.run("probe: truncated TIFF (pixel data cut) real import", ["import-legacy", "--input", inp, "--channel-map", ch, "--identity-map", d / "ids.json", "--seed", 1, "--created-at", T0],
                data_dir=d / "data")
    pub = snapshots(d / "data")
    rep.add("a TIFF whose pixel data is truncated is rejected cleanly (exit 0/2, no traceback, a named reason); the good file is not lost or half-published",
            "Traceback" not in r["err"] and r["exit"] in (0, 2) and ("ZZ2" in r["out"] + r["err"]) and not list((d / "data" / "datasets").glob("ds-*/.staging*")) and len(pub) <= 1,
            f"exit {r['exit']}: {r['err'][-300:]}", known="truncated pixel data passes the header-only scan and fails when the plane is read" if "Traceback" in r["err"] else None,
            phase="probe")


# ====================================================================================== real-size calibration
def phase_calibration(ctx, rep, out, n, workers, n_proj):
    cal = out / "calib"
    sp = ms.Spec(n=n, seed=77, cell_px=316, real_size=True, workers=min(workers, 4))
    t = time.perf_counter()
    orc = ms.generate(sp, cal)
    gen_s = time.perf_counter() - t
    files, inp, M = orc["files"], cal / "input", cal / "mapping"
    ch = json.loads((M / "channels.json").read_text())
    sizes = [r["size"] for r in files.values() if r["kind"] in ("good", "uint8")]
    rd = ctx.run("calibration dry-run (real size)", import_args(inp, M, 4, 1, ["--dry-run"]), data_dir=cal / "data_dry")
    res = {}
    for w in (1, workers):
        dd = cal / f"data_w{w}"
        if dd.exists():
            shutil.rmtree(dd)
        res[w] = ctx.run(f"calibration import workers={w} (real size)", import_args(inp, M, 4, w), data_dir=dd)
    ok = all(r["exit"] == 0 for r in res.values()) and rd["exit"] == 0
    rep.add(f"[calib] real-size sample ({n} files, 2438x1460 uint16) dry-run + import (workers 1 and {workers}) succeed", ok, [r["err"][-200:] for r in res.values()], phase="calibration")
    if not ok:
        return
    v1, vw = snapshots(cal / "data_w1")[0], snapshots(cal / f"data_w{workers}")[0]
    rep.add("[calib] real-size workers 1 vs N identical version", v1 == vw, f"{v1} {vw}", phase="calibration")
    man = json.loads((cal / "data_w1" / "datasets" / v1 / "manifest.json").read_text())
    n_crops = len(man["crops"])
    n_files = len(files)
    snap_b = dir_bytes(cal / "data_w1" / "datasets" / v1)
    in_b = dir_bytes(inp)
    sha_s = _sha_throughput(inp)
    pf_scan = rd["wall"] / n_files
    r1, rw = res[1], res[workers]
    per_crop_bytes = snap_b / max(1, n_crops)
    crops_per_file = n_crops / max(1, len([1 for r in files.values() if r["kind"] in ("good", "uint8", "rescan")]))
    rss_w = rw["rss_mb"]
    P = dict(real_size_files=n_files, real_size_input_MB=in_b / 1e6, mean_file_MB=float(np.mean(sizes)) / 1e6, generated_s=gen_s, dry_run_s=rd["wall"], dry_run_s_per_file=pf_scan,
             import_s_w1=r1["wall"], import_s_per_file_w1=r1["wall"] / n_files, import_s_w_n=rw["wall"], import_s_per_file_wn=rw["wall"] / n_files, workers=workers,
             crops=n_crops, labels=man["label_counts"]["positives_total"], crops_per_s_w1=n_crops / r1["wall"], crops_per_s_wn=n_crops / rw["wall"],
             snapshot_MB=snap_b / 1e6, snapshot_bytes_per_crop=per_crop_bytes, crops_per_accepted_file=crops_per_file, peak_rss_MB_dry=rd["rss_mb"], peak_rss_MB_w1=r1["rss_mb"],
             peak_rss_MB_wn=rss_w, sha256_MB_per_s=sha_s)
    # PROJECTION (an ESTIMATE): linear in the number of files; the real directory is ~1500 files of this size
    P["projection_for_files"] = n_proj
    P["ESTIMATE_dry_run_min"] = pf_scan * n_proj / 60
    P["ESTIMATE_import_min_w1"] = P["import_s_per_file_w1"] * n_proj / 60
    P["ESTIMATE_import_min_wn"] = P["import_s_per_file_wn"] * n_proj / 60
    P["ESTIMATE_snapshot_GB_at_measured_mark_density"] = per_crop_bytes * crops_per_file * n_proj / 1e9
    P["ESTIMATE_snapshot_GB_upper_bound_28_cells_x_page0"] = per_crop_bytes * 28 * n_proj / 1e9
    P["ESTIMATE_input_scan_read_GB"] = in_b / n_files * n_proj / 1e9
    P["ESTIMATE_peak_RSS_GB_import"] = rss_w / 1024 + (rd["rss_mb"] - 0) * 0 / 1024
    P["basis"] = ("measured on %d synthetic files with the real geometry (2438x1460 uint16 x2 pages, 4x7 grid of 316.23 px cells, ~%.0f crops/accepted file); "
                  "projection is linear in the file count; peak RSS is per-process (planes of the sources being cut are held per worker) and does not grow with the file count except for the "
                  "per-file scan metadata" % (n_files, crops_per_file))
    rep.perf["calibration_real_size"] = P


def _sha_throughput(root: Path) -> float:
    fs = [p for p in sorted(Path(root).rglob("*.tif"))][:8]
    t = time.perf_counter()
    tot = 0
    for p in fs:
        ms.file_sha256(p)
        tot += p.stat().st_size
    return tot / 1e6 / max(1e-9, time.perf_counter() - t)


# ====================================================================================== main
def rehearse(a) -> int:
    out = Path(a.out).resolve()
    if out.exists() and not a.keep_existing:
        subprocess.run(["chmod", "-R", "u+w", str(out)])
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    ctx = Ctx(out, rep)
    t0 = time.time()
    meta = dict(command=" ".join(sys.argv), n=a.n, cell_px=a.cell_px, workers=a.workers, seed=a.seed, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"== generate N={a.n} cell_px={a.cell_px}", flush=True)
    spec = ms.Spec(n=a.n, seed=a.seed, cell_px=a.cell_px, workers=a.workers)
    t = time.perf_counter()
    orc = ms.generate(spec, out)
    rep.perf["generate_s"] = time.perf_counter() - t
    files, inp, M = orc["files"], out / "input", out / "mapping"
    ch = json.loads((M / "channels.json").read_text())
    rep.perf["input_MB"] = dir_bytes(inp) / 1e6
    rep.perf["oracle_counts"] = orc["counts"]
    rep.perf["examined_manifest_plan"] = orc["examined_notes"]
    exp = ms.expected_status(files, ch)
    D = {rel: ms.derive(files[rel], {int(k): v for k, v in files[rel]["mapping_pages"].items()}) for rel, v in exp.items() if v["status"] == "accepted"}
    n_exp_crops = sum(len(d["crops"]) for d in D.values())
    n_src_crops = sum(1 for d in D.values() if d["crops"])
    rep.perf["oracle_expected"] = dict(crops=n_exp_crops, sources_with_crops=n_src_crops, accepted=len(D), tuples=sum(len(d["tuples"]) for d in D.values()),
                                       labels=sum(d["labels_total"] for d in D.values()))
    print(f"   {len(files)} files, expected {n_exp_crops} crops from {n_src_crops} sources", flush=True)
    # capability sniff: which product behaviour is present (architect changes land during the rehearsal)
    from polony_app.ml.cli import build_parser
    has_mask = "--valid-mask" in build_parser().format_help() or "--valid-mask" in str(build_parser()._subparsers._group_actions[0].choices["import-legacy"].format_help())
    NORM_MODE[0] = "chamber" if has_mask else "eligible-fields"
    rep.notes.append(f"normalisation oracle mode: {NORM_MODE[0]} (product {'has' if has_mask else 'lacks'} --valid-mask)")
    print("== templates", flush=True)
    phase_templates(ctx, rep, out, files, inp, a.fast)
    print("== refusals", flush=True)
    phase_refusals(ctx, rep, out, orc["error_variants"], inp, M, a.fast)
    print("== dry-run", flush=True)
    rd = phase_dry_run(ctx, rep, out, inp, M, 4, files, ch, NORM_MODE[0])
    rep.perf["dry_run"] = dict(wall_s=rd["wall"], peak_rss_MB=rd["rss_mb"])
    print("== import", flush=True)
    info = phase_import(ctx, rep, out, inp, M, 4, a.workers, files, ch, NORM_MODE[0], n_src_crops, a.fast)
    d1, v1 = info["d1"], info["v1"]
    rep.perf["import"] = dict(wall_s_w1=info["r1"]["wall"], wall_s_wn=info["r4"]["wall"], peak_rss_MB_w1=info["r1"]["rss_mb"], peak_rss_MB_wn=info["r4"]["rss_mb"], snapshot=v1,
                              snapshot_MB=dir_bytes(d1 / "datasets" / v1) / 1e6, input_MB=rep.perf["input_MB"])
    man_info = check_snapshot(rep, "full", d1, v1, files, inp, ch, norm_mode=NORM_MODE[0], downscale=4, pixel_files_limit=a.pixel_files)
    rep.perf["import"].update(crops=man_info["n_crops"], labels=sum(len(v["labels"]) for v in man_info["exp_crops"].values()),
                              crops_per_s_w1=man_info["n_crops"] / info["r1"]["wall"], crops_per_s_wn=man_info["n_crops"] / info["r4"]["wall"])
    rd2 = d1  # lock phase uses d1
    print("== manifest-mode dry-run", flush=True)
    if not a.fast:
        phase_manifest_dry(ctx, rep, out, inp, M, 4, files, ch)
    print("== kill -9 / resume, disk", flush=True)
    phase_kill_resume(ctx, rep, out, inp, M, 4, info, n_exp_crops, n_src_crops, a.fast)
    phase_disk(ctx, rep, out, inp, M, 4, info, n_exp_crops, a.fast)
    base, general = make_base_models(ctx, out)
    print("== train --dry-run", flush=True)
    phase_train_dry(ctx, rep, out, d1, v1, base, files, fast=a.fast)
    print("== lock + extras", flush=True)
    phase_lock(ctx, rep, out, info, files, inp, M, ch, 4, a.workers, spec, NORM_MODE[0], a.fast)
    if not a.fast:
        phase_train_dry(ctx, rep, out, d1, v1, base, files, tag="full-after-lock")
    if not a.no_model:
        print("== model stage (mini snapshot)", flush=True)
        phase_model(ctx, rep, out, a.mini_n, a.seed, a.workers, base, general, a.cell_px)
    if not a.no_probe:
        phase_truncated(ctx, rep, out)
    if a.calibration:
        print("== real-size calibration", flush=True)
        phase_calibration(ctx, rep, out, a.calibration, a.workers, a.project_files)
    rep.perf["total_wall_s"] = time.time() - t0
    rep.perf["disk_used_MB_out_dir"] = dir_bytes(out) / 1e6
    rep.write(out, meta)
    if not a.keep:
        for p in ("input", "input_extra", "data_w2", "data_kill", "data_enospc", "data_disk", "data_dry", "data_dry2", "calib", "probe_truncated"):
            q = out / p
            if q.exists():
                subprocess.run(["chmod", "-R", "u+w", str(q)])
                shutil.rmtree(q, ignore_errors=True)
    cnt = Counter(c["status"] for c in rep.checks)
    print(f"\nRESULT {dict(cnt)}  report: {out}/REHEARSAL.md", flush=True)
    return 1 if rep.failed else 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--cell-px", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the generated TIFFs and intermediate data dirs (snapshots of the main run are always kept)")
    ap.add_argument("--keep-existing", action="store_true", help="do not wipe --out first")
    ap.add_argument("--fast", action="store_true", help="run only a representative subset of the refusal variants")
    ap.add_argument("--no-probe", action="store_true", help="skip the truncated-TIFF probe")
    ap.add_argument("--no-model", action="store_true", help="skip the tiny fine-tune stage")
    ap.add_argument("--mini-n", type=int, default=40, help="files of the small snapshot used for the fine-tune stage")
    ap.add_argument("--calibration", type=int, default=20, help="real-size (2438x1460) files for the calibration batch (0 = skip)")
    ap.add_argument("--project-files", type=int, default=1500, help="size of the real directory for the projection")
    ap.add_argument("--pixel-files", type=int, default=None, help="limit the bit-exact pixel/label comparison to the first N files (default: all)")
    return ap


def main(argv=None) -> int:
    return rehearse(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
