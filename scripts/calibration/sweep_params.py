#!/usr/bin/env python3
"""Per-batch parameter sweep: which (threshold_scale, spot_radius_px) makes the app's red counts
closest to the lab's embedded ImageJ marks?

Drives a RUNNING app through its HTTP API, exactly like the Batch screen does:
upload TIFF -> channels (page 0 = red, page 1 = green) -> auto chamber -> import the researcher's
fields -> POST /detect for every parameter pair -> GET /fields/compare (plane 0, red_positive).

Start the app first (scripts/run_app.sh), then e.g.:

    python scripts/calibration/sweep_params.py /path/to/tiffs --out sweep_out

Needs `requests` and `numpy`. Resumable: raw results are appended to <out>/sweep_raw.csv and
combinations already there are skipped. Nothing is reviewed or approved; analyses are deleted at the
end unless --keep is given.

Reading the result: the reference is the lab's count. Pick parameters per batch type by the
leave-one-file-out column (a setting chosen on the OTHER files, scored on the held-out one), not by the
in-sample best, and confirm on a batch that did not take part. Only red / plane 0 is tuned (green and
yellow have too few marks); both channels get the same parameters so detection runs normally.
"""
import argparse
import csv
import itertools
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

RAW_HEADER = ["file", "batch", "threshold_scale", "spot_radius_px", "field", "reference_count", "app_count", "detect_s"]


class Api:
    def __init__(self, base):
        self.base = base.rstrip("/") + "/api"
        self.s = requests.Session()

    def call(self, method, path, **kw):
        r = self.s.request(method, self.base + path, timeout=600, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def wait_job(self, job, poll=0.5):
        while job["status"] in ("queued", "running"):
            time.sleep(poll)
            job = self.call("GET", f"/jobs/{job['id']}")
        if job["status"] != "done":
            raise RuntimeError(f"detection {job['status']}: {job.get('error')}")
        # the analysis may still be flipping out of 'detecting'
        for _ in range(20):
            if self.call("GET", f"/analyses/{job['analysis_id']}")["status"] != "detecting":
                break
            time.sleep(0.3)


def pick_files(paths, batch_of, per_batch):
    """Deterministic, evenly spread subset per batch (0 = all)."""
    by = defaultdict(list)
    for p in sorted(paths):
        by[batch_of(p.name)].append(p)
    out = []
    for b, ps in sorted(by.items()):
        if per_batch and len(ps) > per_batch:
            idx = np.linspace(0, len(ps) - 1, per_batch).round().astype(int)
            ps = [ps[i] for i in sorted(set(idx))]
        out += ps
    return out


def prepare(api, path, rim):
    """Upload + map channels + chamber + import marks. Returns analysis id or None (no marks)."""
    with open(path, "rb") as fh:
        a = api.call("POST", "/analyses", files=[("files", (path.name, fh))], data={"name": path.stem})
    aid = a["id"]
    api.call("PUT", f"/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    try:
        ch = api.call("POST", f"/analyses/{aid}/regions/auto-chamber", json={"rim_px": rim})
        region = {"id": "chamber", "kind": "counting", "name": f"Chamber (rim {rim}px)", "polygon": ch["polygon"],
                  "approved": False, "approved_at": None, "approval_fingerprint": None, "inspected": False, "field_index": None}
        api.call("PUT", f"/analyses/{aid}/regions", json={"regions": [region]})
    except RuntimeError as e:
        print(f"   ! chamber failed ({e}); counting whole image", file=sys.stderr)
    ref = api.call("GET", f"/analyses/{aid}/reference")
    if not (ref.get("available") and ref.get("grid")):
        api.call("DELETE", f"/analyses/{aid}")
        return None
    api.call("POST", f"/analyses/{aid}/fields/import", json={"mode": "counted", "shift": [0, 0], "plane": None, "replace": False})
    return aid


def detect(api, aid, scale, radius):
    p = {"spot_radius_px": radius, "threshold_scale": scale}
    t0 = time.time()
    job = api.call("POST", f"/analyses/{aid}/detect", json={"engine": "bigfish", "preview_window": None, "red": p, "green": p})
    api.wait_job(job)
    dt = time.time() - t0
    cmp = api.call("GET", f"/analyses/{aid}/fields/compare", params={"plane": 0, "count_type": "red_positive"})
    return dt, [r for r in cmp["rows"] if r["reference_count"] is not None and r["app_count"] is not None]


def load_raw(path):
    rows = []
    if path.exists():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(r, threshold_scale=float(r["threshold_scale"]), spot_radius_px=float(r["spot_radius_px"]),
                                 field=int(r["field"]), reference_count=float(r["reference_count"]), app_count=float(r["app_count"])))
    return rows


def metrics(ref, app):
    ref, app = np.asarray(ref, float), np.asarray(app, float)
    if ref.sum() == 0 or len(ref) == 0:
        return None
    return dict(n=len(ref), total_ratio=app.sum() / ref.sum(), wape=np.abs(app - ref).sum() / ref.sum(),
                exact=float(np.mean(app == ref)), within1=float(np.mean(np.abs(app - ref) <= 1)),
                r=float(np.corrcoef(ref, app)[0, 1]) if len(ref) > 2 and ref.std() > 0 and app.std() > 0 else float("nan"))


def analyse(rows, out_dir, base_combo):
    """Per batch: score every combo, add leave-one-file-out validation, write sweep_summary.csv."""
    by = defaultdict(lambda: defaultdict(list))  # batch -> combo -> rows
    for r in rows:
        by[r["batch"]][(r["threshold_scale"], r["spot_radius_px"])].append(r)
    summary, report = [], []
    for batch, combos in sorted(by.items()):
        files = sorted({r["file"] for rs in combos.values() for r in rs})
        scores = {}
        for c, rs in combos.items():
            m = metrics([r["reference_count"] for r in rs], [r["app_count"] for r in rs])
            if m:
                scores[c] = m
                summary.append(dict(batch=batch, threshold_scale=c[0], spot_radius_px=c[1], n_files=len({r["file"] for r in rs}), **m))
        if not scores:
            continue
        best = min(scores, key=lambda c: scores[c]["wape"])
        # leave-one-file-out: choose on the other files, score on the held-out file (pooled over files)
        held_ref, held_app = [], []
        if len(files) > 2:
            for f in files:
                tr = {c: metrics([r["reference_count"] for r in rs if r["file"] != f], [r["app_count"] for r in rs if r["file"] != f])
                      for c, rs in combos.items()}
                tr = {c: m for c, m in tr.items() if m}
                if not tr:
                    continue
                pick = min(tr, key=lambda c: tr[c]["wape"])
                for r in combos[pick]:
                    if r["file"] == f:
                        held_ref.append(r["reference_count"])
                        held_app.append(r["app_count"])
        cv = metrics(held_ref, held_app) if held_ref else None
        base = scores.get(base_combo)
        report.append((batch, len(files), base, best, scores[best], cv))
    fields = ["batch", "threshold_scale", "spot_radius_px", "n_files", "n", "total_ratio", "wape", "exact", "within1", "r"]
    with open(out_dir / "sweep_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in sorted(summary, key=lambda s: (s["batch"], s["wape"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print("\n=== Result per batch (reference = the lab's count; WAPE = sum|app-ref| / sum ref, lower is better) ===")
    for batch, nf, base, best, bm, cv in report:
        print(f"\n[{batch}]  {nf} files")
        if base:
            print(f"  defaults  scale={base_combo[0]:g} radius={base_combo[1]:g}: total x{base['total_ratio']:.2f}  WAPE {base['wape']*100:.0f}%  "
                  f"exact {base['exact']*100:.0f}%  +-1 {base['within1']*100:.0f}%")
        print(f"  BEST      scale={best[0]:g} radius={best[1]:g}: total x{bm['total_ratio']:.2f}  WAPE {bm['wape']*100:.0f}%  "
              f"exact {bm['exact']*100:.0f}%  +-1 {bm['within1']*100:.0f}%  r={bm['r']:.2f}   (in-sample)")
        if cv:
            print(f"  held-out  (leave-one-file-out): WAPE {cv['wape']*100:.0f}%  exact {cv['exact']*100:.0f}%  +-1 {cv['within1']*100:.0f}%  <- the honest number")
    print(f"\nFull table: {out_dir / 'sweep_summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiff_dir", type=Path, help="folder with the .tif files (searched recursively)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="running app (default %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("sweep_out"))
    ap.add_argument("--scales", default="0.6,0.8,1,1.3,1.7,2.2,3", help="threshold_scale values (>1 = stricter, fewer spots)")
    ap.add_argument("--radii", default="4,5,6", help="spot_radius_px values (LoG sigma in px)")
    ap.add_argument("--rim", type=int, default=40, help="chamber rim px (Batch screen default 40)")
    ap.add_argument("--batch-regex", default=r"^\d+-\d+-\d+_([A-Za-z0-9]+)", help="group 1 = batch id taken from the file name (default: 31-03-2026_D_.. -> D)")
    ap.add_argument("--per-batch", type=int, default=8, help="files per batch, evenly spread (0 = all; the full set takes hours)")
    ap.add_argument("--keep", action="store_true", help="do not delete the analyses at the end")
    a = ap.parse_args()

    scales = [float(x) for x in a.scales.split(",")]
    radii = [float(x) for x in a.radii.split(",")]
    rx = re.compile(a.batch_regex)
    batch_of = lambda n: (rx.match(n).group(1) if rx.match(n) else "all")
    paths = [p for p in a.tiff_dir.rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
    files = pick_files(paths, batch_of, a.per_batch)
    if not files:
        sys.exit(f"no TIFF files under {a.tiff_dir}")
    a.out.mkdir(parents=True, exist_ok=True)
    raw_path = a.out / "sweep_raw.csv"
    rows = load_raw(raw_path)
    done = {(r["file"], r["threshold_scale"], r["spot_radius_px"]) for r in rows}
    combos = list(itertools.product(scales, radii))
    print(f"{len(files)} files x {len(combos)} parameter pairs = {len(files) * len(combos)} detections (~10 s each)")

    api = Api(a.url)
    api.call("GET", "/analyses")  # fail early if the app is not running
    new = not raw_path.exists()
    with open(raw_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(RAW_HEADER)
        created = []
        for i, p in enumerate(files, 1):
            todo = [c for c in combos if (p.name, c[0], c[1]) not in done]
            if not todo:
                continue
            print(f"[{i}/{len(files)}] {p.name}  ({len(todo)} runs)", flush=True)
            try:
                aid = prepare(api, p, a.rim)
                if aid is None:
                    print("   no embedded marks, skipped")
                    continue
                created.append(aid)
                for scale, radius in todo:
                    dt, res = detect(api, aid, scale, radius)
                    for r in res:
                        row = [p.name, batch_of(p.name), scale, radius, r["field"], r["reference_count"], r["app_count"], round(dt, 1)]
                        w.writerow(row)
                        rows.append(dict(zip(RAW_HEADER, row), reference_count=float(r["reference_count"]), app_count=float(r["app_count"])))
                    fh.flush()
            except Exception as e:  # one bad file must not stop the sweep
                print(f"   ! {p.name}: {e}", file=sys.stderr)
        if not a.keep:
            for aid in created:
                try:
                    api.call("DELETE", f"/analyses/{aid}")
                except Exception:
                    pass
    analyse(rows, a.out, (1.0, 5.0))


if __name__ == "__main__":
    main()
