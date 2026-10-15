# Fine-tuning Spotiflow on the historical ImageJ-annotated TIFFs

Operator guide for `python -m polony_app.ml import-legacy`: how ~1,500 historical two-page TIFFs with embedded ImageJ marks enter the
**same** snapshot -> split -> lock-test -> train -> eval-model -> compare -> promote lifecycle as newly reviewed analyses
([FINE_TUNING.md](FINE_TUNING.md)). Same page-level `valid_unsaturated` normalisation, same block-mean downscale and exact coordinate mapping, same
content-addressed snapshots and locked test set. This is **fine-tuning** of the pretrained Spotiflow model (`--base general`), not training from scratch.

> **What the marks are.** ImageJ marks are one person's clicks (median ~5-6 px from the intensity peak), not unquestionable ground truth. A missing mark is **not**
> a negative: the file cannot tell "examined, no polonies" from "not examined". Every metric computed from these labels is *agreement with this annotator*.

## 1. The rule: only fields the researcher examined, per (TIFF, field, page)

A crop is exported only when there is **explicit evidence that the researcher examined that exact source TIFF, grid field and page**:

| status | evidence | training use |
|---|---|---|
| `positive_marked` | embedded ImageJ marks on **that page** inside **that field** | crop with its marks as labels |
| `verified_zero` | explicit sidecar / manifest entry stating the researcher counted that field on that page with count 0 | crop with an empty label file |
| `unknown` | no marks and no explicit record (the grid overlay alone is **not** evidence; marks on the other page are **not** evidence; application detections are **not** evidence) | **excluded**, never a negative |
| `conflict` | a stated count that disagrees with the marks (or a stated zero with marks) | **excluded** and reported |

* Eligibility is decided **independently** for every `(TIFF, field, page)`: marks on page 0 never make the same field on page 1 eligible.
* Modes (`--mode`): **`marked-only`** (default; only field/pages with at least one embedded mark) and **`examined-fields-manifest`** (only tuples listed in `--examined-fields`; verified zeros allowed).
  There is **no** "all grid cells are negative unless marked" mode and no override that enables one.
* A cell that contains only "stray" points whose ImageJ counter belongs to another cell (clicked across a border) is not examined evidence for that cell; it is excluded and recorded.
* A mark with **no usable plane** (ImageJ position missing, 0 or beyond the file's planes) is evidence for no page: every field/page of the cell that contains it becomes `conflict` and is excluded (dropped reason `plane_out_of_range`).
* A mark belongs to the crop whose integer **pixel footprint** contains its rounded centre (`floor(x + 0.5)`), for the evidence count, the labels and the counter->cell mapping, so a mark within half a pixel of a grid line can never be background in one crop and a label in another.
* Points that are out of bounds, duplicates, outside the grid, on a page that is not trained, with an unspecified plane or inside a non-examined cell are **never** labels: they are listed as `dropped:<reason>` in `points.csv` and counted in the manifest. Nothing is clamped.
* Every exported crop carries `provenance`: `researcher_examined: true`, `examination_source`, `evidence`, source TIFF + sha256, page, `imagej_position`, confirmed channel role and where the mapping came from, counter(s), grid row/col/cell polygon, positive-marked or verified-zero, number of marks, coordinate frame, assumptions. `train` refuses a snapshot where any crop lacks `researcher_examined == true`.
* The researcher-comparison report in the app follows the same rule (unknown fields are reported and excluded from totals, MAE, slope and r): [FIELDS.md](FIELDS.md).

## 2. What the real files showed (differences from the working assumptions)

Measured on the 8 research files (7 two-page stacks + 1 RGB composite) and `test_img.tif`; reproduce with `import-legacy --dry-run`.

* **Confirmed:** two-page `uint16` ImageJ stacks (ZYX), 2438x1460 (`test_img` 2450x1438); ROI = one POINT blob with counters and positions; the counter is a **field**, not a class; ImageJ position 1/2 = page 0/1 (no position out of range); grid overlay = 28 cells of ~316.23 px.
* **Differs / new:**
  * the "eighth" file is an 8-bit RGB **composite** (LZW) whose blue plane is a saturation mask, not a colour; the importer rejects it (`unsupported_layout`). No `AI-A` stack exists in the extracted set (the earlier docs mention one).
  * **`FS-B_D` is consistent with its own stored grid** (origin (3, 123)); on the (158, 158) grid its counters would span 3-4 cells. Other files: origin (158, 158). The bridge always uses the file's own overlay.
  * `AI-B`: three counters have one point one cell over, one point lies 1.5 px outside the grid, and counter 12 is a second out-of-order counter. `AS-B`: counter 11 out of order. `AM-B`: counters 3 and 19 absent (the lab CSV numbers fields without gaps, so "field = counter + 1" **fails after counter 19**). `AS-A`: counter 7 absent.
  * Page 1 carries marks in only **4 of 7** stacks (13 field/pages in total); under the rule above page 1 is therefore almost entirely `unknown`.
  * File names say 29-09-2026, scanner metadata says 2026/09/30; AI/AM/AS were scanned with PMT 550 / laser 1.97, FS with PMT 500 / 3.30. No scan-session id exists; `scan_session` stays empty unless you supply it.
* Under `marked-only` with page 0 trained: **196 field/pages, 108 examined (all positive_marked), 88 excluded as unproven** (87 "no marks", 1 "only straddling points"); 1,098 ImageJ points -> 1,078 labels, 20 dropped.

## 2b. Normalisation: the same mask the app uses
Page-level `lo/hi` (percentiles 1 / 99.8 of the valid, unsaturated pixels) must be computed over the **same mask the app uses at inference**, otherwise training and evaluation run on an input scale the app never produces (measured on the research files: the hi of the examined cells alone differs from the whole chamber by 1.5-11x). `--valid-mask` (default **`chamber`**) = the app's automatic chamber (rim 40 px, all planes) turned into the app's valid mask, computed once per source; if no chamber is found the union of **all grid cells** is used with a recorded warning (`valid_mask_used` per page in the manifest). `grid-cells` forces that union; `eligible-fields` (statistics over the examined fields only) is kept for experiments and is **not** inference parity. This uses pixel statistics only (no labels, nothing from the test set). The chamber `hi` differs a lot between slides (e.g. 3.4k to 44k on the research files), which is why the same normalisation at inference matters.

## 3. Files you provide

All three are written as **templates** by `import-legacy --input DIR --write-templates OUT` (no pixel data is read). Nothing in a template is confirmed.

### 3.1 Channel map (`channels.json`) - required for a real import
The researcher's statement of what each page shows. Roles are never inferred from file names or from the `635 [W1]` label.
```json
{ "schema": "polony-channel-map", "version": 1,
  "confirmed_by": "<who confirmed this mapping>", "confirmed_on": "2026-10-14",
  "source_of_truth": "<on what basis: protocol, scanner export, researcher's statement>",
  "defaults": { "pages": { "0": {"role": "red", "train": true}, "1": {"role": "green", "train": false} } },
  "rules": [ { "id": "fs-scans", "match": {"glob": "*_FS-*.tif"}, "pages": { "0": {"role": "red", "train": true} } } ],
  "files": { "some/dir/odd_file.tif": { "pages": { "0": {"role": "green", "train": true} } } } }
```
**Strict:** unknown keys anywhere (e.g. a typo such as `"trian"`) are rejected; every page block must state `role` **and** `train` explicitly; every `files` key must be the exact relative posix path of a scanned file (`./x`, a different case or a backslash is an error that names the nearest existing file); a rule or a `files` entry that matches no scanned file is a **blocker** (real import refuses; the dry-run lists it under BLOCKERS). Precedence: `files` > `rules` > `defaults`; rule `match` keys: `glob`, `assay`, `sha256`, `specimen_id`, `slide_id`. Two different rules matching one file, a page index beyond the file's planes, an empty or duplicate role within a file: **refused** with the offending rule/file named. A map with empty `confirmed_by` or `source_of_truth` is refused ("fill in who confirmed this mapping and on what basis"). The template ships with the default rule page 0 = `red` (train), page 1 = `green` (not trained) - the layout validated on the research stacks - which becomes active only after you fill in those fields. The origin (`default` / `rule:<id>` / `file`) is recorded per file in the manifest. The dry-run also prints an **information-only** check of the ImageJ `Info` label (e.g. `635 [W1]`) per file and lists deviating files; it never decides a role.

### 3.2 Identity map (`identities.json` or `.csv`) - required for a split
```json
{ "rows": [ { "file": "29-09-2026_T7_AB_AM-A.tif", "specimen_id": "AM", "slide_id": "AM_A",
              "assay": "T7_AB", "replicate": "A", "scan_session": null, "confirmed": true } ] }
```
`file` = path relative to `--input` (or a unique base name). `confirmed` must be stated explicitly (a missing key or an empty CSV cell is an error). `specimen_id` groups replicate slides (kept in one split) and is **required** unless you pass `--level slide`; rows that share a specimen id (case-insensitively) but differ in `assay` are refused; `AM`/`am` are merged to one spelling with a warning. Rows with `confirmed: false`, or files not in the map, are **inferred** from file names and **refused for splits** unless `--allow-unconfirmed-identity` (recorded in the manifest). Byte-identical files under different names are merged into one group.

### 3.3 Examined-fields manifest (optional; only for verified zeros) - `--mode examined-fields-manifest --examined-fields F.json`
```json
{ "schema": "polony-examined-fields", "version": 1,
  "evidence": "T7B_15-09-2026.csv: blank = not counted, 0 = counted zero (confirmed by <name> on <date>)",
  "source_kind": "sidecar_csv", "examined_by": "<name>", "recorded_on": "2026-10-14",
  "files": [ { "file": "29-09-2026_T7_AB_AM-B.tif", "pages": { "0": [ {"counter": 3, "count": 0}, {"row": 1, "col": 2, "count": 0}, {"counter": 5} ] } } ] }
```
A field entry names the field by `counter`, `field` (lab field number; read as counter N-1 only with `--acknowledge-field-equals-counter-plus-one`) or `row`+`col`. `count: 0` (or `counted_zero: true`) is a verified zero; an entry **without** a stated count and without marks stays `unknown`; a stated count that differs from the marks is a `conflict`. A CSV form (`file,page,counter|field|row+col,count,counted_zero,note`, first line `# evidence: ...`) is also accepted.
**Lab CSV caution:** the lab sheets number fields without gaps, so a naive "field = counter + 1" mapping mislabels fields after an absent counter (real example: `AM-B`, counter 19). `--lab-csv/--lab-series/--acknowledge-lab-csv-field-numbering` exist, but entries that cannot be located unambiguously are listed under `unresolved_manifest_entries` and never guessed; prefer `row`+`col` entries confirmed by the researcher.

## 4. Operator workflow for the ~1,500-file directory

`D=/path/to/historical-tiffs`, `W=/path/to/work` (templates, reports) and `--data-dir` = the app data directory (snapshots live in `<data-dir>/datasets`, models in `<data-dir>/models`).
Everything below is explicit: nothing trains, locks or promotes by itself, and no step asks an interactive question.

1. **Audit filenames and metadata** (writes only templates; reads no pixel data):
   `python -m polony_app.ml --data-dir DATA import-legacy --input $D --write-templates $W/tpl`
   -> `identities.template.json` (inferred specimen/slide, `confirmed:false`), `channels.template.json` (default rule page 0 = red/train, page 1 = green/not trained, **unconfirmed**), `examined-fields.template.json`.
2. **Create the mapping files.** Copy the templates to `identities.json` / `channels.json`; check every identity row against the lab records and set `confirmed: true`; fill `confirmed_by`, `confirmed_on`, `source_of_truth` in `channels.json` (section 3.1) and add `rules`/`files` entries for files that deviate. Optional `examined-fields.json` for verified zeros (section 3.3).
2b. **Pre-flight check** (read-only, exit 0/1/2): `python -m polony_app.ml --data-dir DATA preflight --input $D --channel-map $W/channels.json --identity-map $W/identities.json` - cross-checks the scan independently and flags problems before any import. Before trusting a new code version on the full set, the synthetic rehearsal `PYTHONPATH=src python3 scripts/preflight/rehearsal.py --n 1500 --cell-px 64 --workers 4 --out DIR --keep` (see `scripts/preflight/README.md`) checks the whole pipeline against an independent oracle (143 checks passed at N=1500; ~30 min, ~1 GB RAM, 880 MB input). It does not test real scanner data or GPU speed.
3. **Import dry-run** (writes nothing):
   `python -m polony_app.ml --data-dir DATA import-legacy --input $D --channel-map $W/channels.json --identity-map $W/identities.json --downscale 4 --normaliser valid_unsaturated --fractions 0.7,0.15,0.15 --batch-size 4 --epochs 10 --dry-run`
4. **Review the report** (section 5): rejected files and reasons, field/page tuples excluded because examination could not be proven (and why), dropped annotations, identity blockers, group/split table, disk/RAM estimates. Fix the mapping files and repeat step 3 until `BLOCKERS` is empty.
5. **Build the immutable snapshot** (same command without `--dry-run`; add `--workers 4` to write crops concurrently - the result is byte-identical for any worker count; an interrupted run resumes where it stopped):
   `python -m polony_app.ml --data-dir DATA import-legacy --input $D --channel-map $W/channels.json --identity-map $W/identities.json --downscale 4 --workers 4`
   then `python -m polony_app.ml --data-dir DATA verify <version>`. Re-importing identical input prints `already imported as <version>` (exit 0, nothing rewritten, no leftover staging; the plan key is stored in `import_summary.plan_key`). The plan key includes a hash of the import code: after a code update the same input builds a new snapshot. Symlinks that resolve outside `--input` are never followed (listed as skipped).
6. **Inspect and lock the test split:** `python -m polony_app.ml --data-dir DATA split --snapshot <version>` (group -> split table; with too few independent groups a split stays empty rather than being filled from the same specimen), then `... split --snapshot <version> --lock-test`. The lock is permanent and refuses a snapshot built with an `--acknowledge-*` assumption (no ImageJ counters, field = counter+1, lab-CSV numbering) unless you pass `--allow-acknowledged-assumptions-in-test-lock` (recorded in the lock); locked identities are matched case-insensitively on specimen and slide id **without the assay**, and `train` also refuses train/val data that shares such an identity. later builds carry the locked test crops over unchanged and `train` refuses locked material.
7. **ROCm GPU smoke test** (AMD Ryzen AI Max+ 395 / Radeon 8060S; PyTorch ROCm exposes the GPU through `torch.cuda`, so `cuda`, `rocm` (an alias) and `auto` all mean "the Radeon GPU"):
   `python -m polony_app.ml gpu-smoke --device rocm` (prints torch / HIP versions, device, memory; runs a matmul + backward and a conv2d forward/backward, compares the gradients with the CPU). Equivalent one-liner:
   ```
   python - <<'PY'
   import torch
   print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
   print("available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       p = torch.cuda.get_device_properties(0)
       print(p.name, getattr(p, "gcnArchName", ""), p.total_memory)
       a = torch.randn(512, 512, device="cuda", requires_grad=True)
       (a @ a).sum().backward(); torch.cuda.synchronize()
       print("backward OK, grad finite:", bool(torch.isfinite(a.grad).all()))
   PY
   ```
   If `torch.version.hip` is set but no device is visible the command says so and lists things to *check* (`/dev/kfd` access, `video`/`render` groups, supported gfx target, `HSA_OVERRIDE_GFX_VERSION`); an explicitly requested GPU never falls back to the CPU silently (`--device auto` may, with a printed notice).
8. **1-epoch timing benchmark** (also a dry-run first: `train ... --dry-run` prints the actual scale):
   `python -m polony_app.ml --data-dir DATA train --snapshot <version> --base general --device rocm --batch-size 4 --epochs 1 --num-workers 4`
   Note seconds/epoch; the total is seconds/epoch x epochs (plus validation over all validation crops each epoch and twice more for the threshold optimisation).
9. **First candidate, ~10 epochs with early stopping:**
   `python -m polony_app.ml --data-dir DATA train --snapshot <version> --base general --device rocm --batch-size 4 --epochs 10 --early-stopping-patience 3 --num-workers 4`
   `--num-train-samples N` bounds an epoch (N crops drawn with replacement instead of one pass over all training crops). The BEST validation checkpoint is registered (the last epoch is only reported); the entry is a `candidate`, never active.
10. **Evaluate the candidate and `general` on the same locked test set:**
    `python -m polony_app.ml --data-dir DATA eval-model --model <candidate-id> --device rocm` and `... eval-model --model general --downscale 4 --device rocm`
11. **Compare:** `python -m polony_app.ml --data-dir DATA compare --models <candidate-id> general` (precision / recall / F1 at 5, 10, 15 px, vs *this annotator's marks*).
12. **Promote only if the evidence supports it:** `python -m polony_app.ml --data-dir DATA promote <candidate-id> --reason "..."` (refuses a lower F1 or a configuration mismatch; `rollback` restores the previous model). Never promote on a thin or `preliminary` test set.

### What `--dry-run` of `train` prints (real numbers from your snapshot)
sources with crops, eligible crops and the train/val/test split, epochs, batch size, `num_train_samples`, steps per epoch, **estimated optimiser steps**, validation batches per pass and in total (Spotiflow validates with batch size 1; there are also 2 sanity-check batches and a final threshold optimisation over the validation set), prepared in-memory size vs available RAM, the load mode, the native snapshot size, then `Training will not start because --dry-run was supplied.` and recommendations (1-epoch benchmark first, then ~10 epochs with early stopping). Sample for the 8 research files at downscale 4:
```
sources (source images): 8 in the manifest, 7 with crops
eligible crops: 108  (train 52, val 17, test 39 - locked/held out, never trained on)
epochs: 10   batch size: 4   downscale: 4   device: cpu   workers: 0
steps per epoch: 13   estimated optimizer steps: 130
validation: 17 batch(es) per pass x 10 pass(es) = 170 batches, + 2 sanity-check, + 34 for the final threshold optimisation
prepared in-memory size (float32 network inputs): 1.6 MiB; load mode: eager
```

## 5. Reading the import dry-run
`IMPORT-LEGACY DRY RUN` lists: **sources** (discovered / unique / accepted / rejected by reason / byte-identical duplicates merged), **pages and channel mapping** with the origin of every mapping and the information-only `Info`-label check, **field/page tuples** by status with the exact number **excluded because examination could not be proven** (`unknown` + `conflict`, with reasons), positives per page/role, **annotations not used as labels** by reason, **groups and splits**, **estimates** (native snapshot size, prepared in-memory size, batches per epoch), identity table (inferred vs confirmed) and `BLOCKERS`. A dry-run exits 0 even when a real import would be refused for identity or mapping gaps (they are printed as blockers); malformed mapping/manifest files exit 2.

## 6. Disk, RAM and time - how to size the real run
* **Disk:** native crops are stored uncompressed at the original dtype: about `h x w x 2` bytes per crop (316x316 uint16 = ~200 KB) -> 54,000 crops ~ 10.8 GB (the dry-run prints the exact estimate; the build refuses early when free space is clearly insufficient, with a 10 % margin).
* **RAM:** the network inputs are float32 after the downscale `d`: ~ `(316/d)^2 x 4` bytes per crop (d = 4: ~25 KB; d = 1: ~400 KB). 54,000 crops: ~1.3 GB at d = 4 but ~21 GB at d = 1; above 8 GiB (or 25 % of free RAM) `train` switches to bounded lazy loading (`--load-mode`), identical to eager (tested bit-for-bit), or refuses an explicit `eager` that does not fit unless `--allow-large-memory`.
* **Time:** not measured on the Radeon GPU. CPU smoke benchmarks only (`scripts/benchmarks/ml/`): eager vs lazy overhead ~0-12 %, `eval-model` ~0.15-0.44 s per crop on a 4-core CPU (8,000 test crops ~20-60 min on CPU); predictor batching was measured and **not adopted** (identical predictions but no consistent speed-up). Use the 1-epoch benchmark to learn the real seconds/epoch.
* **Validation cost:** Spotiflow validates with batch size 1 and `--val-batch-size` cannot be offered without changing validation metrics; `--max-val-crops N` (a deterministic, recorded subset ordered by `sha256(crop_id)`) limits it if a pass becomes too slow.
* `--num-workers N` is forwarded to the train and validation loaders; runs with different worker counts are not bit-identical (the count is recorded in the registry entry).

## 7. Interface decision
The bridge is `ml import-legacy` (not a separate `dataset build` flavour or a new training system): it produces the **same** snapshot format through the shared `assemble_snapshot`, so `verify`, `split`, `--lock-test`, `train`, `eval-model`, `compare`, `promote` and `rollback` need no changes and keep every guarantee (content-addressed immutable snapshots with sha256 for every file, atomic publication, deterministic output, grouped splits, locked test set, page-level normalisation, recorded downscale/normaliser in snapshot and registry). Two outputs are new: `manifest.import_summary` and per-crop `provenance`.

## 8. Robustness behaviour you may notice
* **Refusals, not tracebacks:** environment problems (disk full, permission denied, a source file deleted or truncated mid-import, unwritable data dir) end with `REFUSED: ...` (exit 2) naming the path and the OS reason; staging is kept so a rerun resumes. Ctrl-C is never swallowed.
* **Mapping files are validated strictly:** size (8 MB) and nesting limits, **duplicate JSON keys are refused** (never "last wins"), unknown keys, wrong types and page keys other than 1-4 ASCII digits are refused with the offending key. A channel-map `files` entry or an examined-manifest entry that names no scanned TIFF refuses the import (exit 2, also in `--dry-run`), with the nearest existing file name.
* **File names that are not valid UTF-8** are skipped with the structured reason `filename_not_utf8` (dry-run and import) and listed; they are never mangled into provenance.
* **Concurrency and cleanup:** each staging directory is protected by a lock under `<data-dir>/.locks/`; the loser of a publication race gets a clean `REFUSED: snapshot ... already exists ... published by another run`. After a successful publish other `.staging-*` directories with a free lock are reclaimed, and any older than 7 days at the start of an import.
* **A source changed during the import is never exported with the old hash:** size, mtime, inode and ctime are re-checked and the file is re-hashed when its first plane is read.
* **`verify` / `train` gates:** a snapshot whose file or crop paths leave the snapshot directory fails `verify` (exit 3); `train` refuses unknown split names, duplicate crop ids, impossible shapes, non-finite or non-increasing `norm_lo/norm_hi`, and any bridge import (recognised by `options.legacy_import`, a non-empty `import_summary` **or** crop provenance) whose crops lack `researcher_examined == true`. Known gap: a manifest from which all three markers were deleted and re-sealed cannot be told apart from a snapshot built from reviewed analyses.
