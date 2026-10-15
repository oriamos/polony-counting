# Model training — start here

This guide explains what PoCo trains, which data it uses, and how to read the results. Training is run from a terminal with `python -m polony_app.ml`; there is no training button in the current React interface. The overnight **experiment Run** button runs detection, not training.

## 1. What is being trained?

PoCo **fine-tunes Spotiflow**: it starts with an existing spot detector (`general`) and adjusts its weights using the researcher's marked polonies. Big-FISH is a separate classical detector and is not trained by this pipeline.

The network receives a single-channel image and learns spot positions. This is not a three-colour classifier: channel mapping and the app's red/green matching determine the final red/green/yellow counts. A folder named `yellow` does not by itself mean the network learned a yellow class.

## 2. Which parts of an image become examples?

A **field** is a grid square. A **crop** is the image cut from one field on one TIFF page. A TIFF can contribute several crops; crop count is not slide count.

There are two input routes:

| Your labels | Command | What can enter the dataset |
|---|---|---|
| Historical ImageJ marks embedded in TIFFs | `import-legacy` | Only field/page pairs with marks on that page, or explicitly verified zero counts in an examined-fields manifest |
| Corrections made in the app | `build` | Fully reviewed rectangular regions with a confirmed channel mapping; exclusions and unresolved points can make a crop ineligible |

For historical files, **no marks does not mean zero**. A grid square without evidence of examination is left out. Marks on page 0 do not prove the same square on page 1 was examined. In default `marked-only` mode, there are no verified-zero examples. To include a genuinely empty counted field, provide explicit evidence using the manifest route in [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md).

Inside an included crop, the supplied marks are the positive labels. Other image content is not labelled as a polony: if the researcher missed a real spot, training can therefore teach the model to suppress it. “Examined” is an eligibility rule, not proof of perfect annotation.

## 3. Why are there three data sets?

| Split | Purpose | Does it update weights? |
|---|---|---|
| **Train** | Examples the network learns from | Yes |
| **Validation** (`val`) | Select the best saved epoch, control early stopping and optimise the detection threshold | No, but it influences model selection |
| **Test** | Compare the selected candidate with the baseline after training | No |

The default target fractions are 70% / 15% / 15%. They apply to independent groups, so the crop counts need not have exactly those percentages. Crops of the same group stay together. Specimen/slide identities and byte-identical files are used to prevent related material crossing splits; correct identity mapping matters.

A **snapshot** freezes the exported crops, labels, settings and split assignments. A **test lock** freezes the test material across later snapshots. Repeatedly choosing new training settings because they improve the same test results makes that test part of model development; reserve new independent data for a final claim.

## 4. What do the settings mean?

| Setting | Plain meaning |
|---|---|
| `--base general` | Start from pretrained Spotiflow weights |
| `--epochs 10` | Request up to 10 training epochs; early stopping may end sooner |
| `--batch-size 4` | Process four training examples per optimiser step, except a smaller final batch |
| `--lr 1e-4` | Learning rate: the size of the weight updates |
| `--downscale 4` on the snapshot | Reduce each spatial dimension by four before training/evaluation; original crops remain saved at native resolution |
| `--crop-size 256` on training | Spotiflow's training patch size in the prepared image; separate from native grid-field size and downscale |
| `--normaliser valid_unsaturated` | Use recorded page-level intensity statistics; do not independently stretch each crop |
| `--early-stopping-patience 3` | Stop after the configured period without validation-loss improvement; the run still keeps the best checkpoint |
| `--device auto` | Choose an available GPU, otherwise CPU; inspect the printed resolved device |
| `--dry-run` | Check the plan and print its size without fitting a model; an import dry-run can still print blockers while exiting 0 |

Train and run a candidate with the same downscale and normaliser. Downscale 4 is the setting of the saved run below, not a guarantee that it is optimal for every scanner.

## 5. How to read the training screen

**Loss is an optimisation score, not an accuracy percentage.** Lower validation loss means the model fits its validation objective better. A loss of `0.432` does not mean 43.2% error or 56.8% accuracy, and a 4.4% loss reduction does not imply 4.4 percentage points more detection accuracy.

The saved run's validation loss fell from `0.451747` at the first logged epoch to a best of `0.431678` (about 4.44% lower). The best epoch is index **7**, meaning the **eighth** epoch. The tenth/final epoch reached `0.441449`; `best.pt`, rather than the final weights, is registered. A lower loss alone does not establish that fine-tuning beats `general`.

After training, evaluate the candidate and baseline on the same locked test crops:

- **Precision:** what fraction of predicted spots match a researcher mark.
- **Recall:** what fraction of researcher marks are found.
- **F1:** balances precision and recall; it is not a general “accuracy” percentage.
- **Count MAE:** average absolute difference in spots counted per evaluated field/page.
- **Count bias:** average predicted count minus reference count; positive means over-counting.

Matching is one-to-one and permits a stated distance between a prediction and a mark. Read 5, 10 and 15 native-pixel tolerances together. A missing human mark can count as a false positive, and equal total counts do not prove the correct spots were found.

## 6. Results actually saved in this repository

Source: [`artifacts/registry.json`](../../artifacts/registry.json), [`snapshot_manifest.json`](../../artifacts/test-set/snapshot_manifest.json), [`test_lock.json`](../../artifacts/test-set/test_lock.json) and the [checkpoint files](../../artifacts/ft-7e4c89cb-8edc09e3). These are recorded results from 15 October 2026, not a fresh rerun on the original images.

Candidate: `ft-7e4c89cb-8edc09e3`; base: `general`; downscale: 4; normaliser: `valid_unsaturated`. The run completed 10 epochs on **CPU**, despite requesting `auto`, in about 32.6 minutes. It remains a **candidate** and the saved registry has no active model.

| Split | Crops | Groups | Labelled points before preparation |
|---|---:|---:|---:|
| Train | 9,253 | 454 | 89,456 |
| Validation | 1,985 | 98 | 19,421 |
| Test | 1,980 | 97 | 17,623 |

All **13,218** crops carry `researcher_examined: true`; all are page 0 mapped as `red`. The import summary records **14,611** unknown field/page pairs excluded and **zero** verified-zero fields. There are 1,252 source records in the manifest and 1,041 distinct contributing source-image hashes; neither number is the number of independent specimens. The import identity summary reports 1,041 confirmed contributing identities and zero unconfirmed ones, although `allow_unconfirmed_identity` was enabled. This records the importer's checks; the mapping cannot be independently confirmed against laboratory records from these artifacts alone.

At **10 native-pixel tolerance**, on the same lock and configuration:

| Measure | Pretrained `general` | Fine-tuned candidate |
|---|---:|---:|
| Precision vs marks | 64.89% | 75.14% |
| Recall vs marks | 73.95% | 91.36% |
| F1 | 69.13% | 82.46% |
| Count MAE, spots per field/page | 3.282 | 2.014 |
| Count bias, spots per field/page | +1.243 | +1.921 |

The candidate has higher recorded F1 and lower count MAE, but **more positive count bias**. Its F1 gain is **13.33 percentage points**, not proof of 82.46% accuracy on new laboratory images. Both evaluations pass `prob_thresh=None`, which uses each model's own threshold; the candidate's saved best threshold is 0.3. This is a comparison of the recorded operating points, not a controlled same-threshold experiment. Test groups have not been independently verified against original samples here, only against the saved records. There are no raw test crops in the repository to rerun inference or inspect annotation completeness.

The older eight-slide / leave-specimen-out experiments in the other documents concern different runs. Their findings should not be presented as the result of this larger run, or as disproven by it.

## 7. Complete terminal workflow — historical ImageJ TIFFs

This is the complete path used for the large historical dataset. Run one block at a time and do not continue when a check reports a blocker. Commands below are for macOS/Linux Bash. Windows users can run the same Python commands in PowerShell after creating the environment; path and environment-variable syntax differs.

### 7.1 Choose permanent directories

Use a disk with enough free space. The historical TIFF directory, dataset snapshots and models are deliberately separate. Replace the four example paths once, here:

```bash
cd /path/to/polony-counting

export POLONY_RUN_ROOT=/data/polony-training
export POLONY_TIFF_DIR=/data/polony-training/tiffs
export POLONY_WORK_DIR=/data/polony-training/work
export POLONY_DATA_DIR=/data/polony-training/appdata
export POLONY_DEVICE=auto
export POLONY_WORKERS=4

mkdir -p "$POLONY_RUN_ROOT" "$POLONY_TIFF_DIR" "$POLONY_WORK_DIR" "$POLONY_DATA_DIR"
```

`POLONY_DATA_DIR` is important: snapshots, the test lock and the model registry must all use this same directory. Keep the original TIFF directory unchanged after starting the import. For the historical set, allow roughly 60 GB free for input, snapshots, staging and models; the dry-run prints an estimate for the actual files.

### 7.2 Create the Python environment and install Spotiflow

Python 3.10–3.13 is supported by this repository. The helper installs a matching PyTorch/torchvision pair and Spotiflow. With `--cuda auto`, it uses NVIDIA CUDA when detected, Apple MPS on Apple Silicon, and otherwise CPU.

```bash
python3 -m venv "$POLONY_RUN_ROOT/venv"
export POLONY_PY="$POLONY_RUN_ROOT/venv/bin/python"

"$POLONY_PY" -m pip install --upgrade pip
"$POLONY_PY" -m pip install -r requirements-app.txt
"$POLONY_PY" install/ai_extra.py install --cuda auto
"$POLONY_PY" install/ai_extra.py check
```

On an AMD ROCm machine, use the ROCm-specific installation in [`scripts/runbook/real_training.sh`](../../scripts/runbook/real_training.sh) instead of `ai_extra.py`; then use `POLONY_DEVICE=rocm`. In this CLI, `rocm` is an alias for PyTorch's `cuda` device API. An explicitly requested unavailable GPU is refused rather than silently using CPU. `auto` may fall back to CPU and prints the resolved device.

Define a short command for the remaining steps:

```bash
ml() {
  PYTHONPATH=src "$POLONY_PY" -m polony_app.ml --data-dir "$POLONY_DATA_DIR" "$@"
}

ml --help
```

Expected: the help lists `import-legacy`, `preflight`, `verify`, `split`, `gpu-smoke`, `train`, `eval-model`, `compare`, `promote` and `rollback`.

### 7.3 Put the TIFFs in the input directory

Preserve the original directory tree and file names. Do not add the research images to Git.

```bash
rsync -ah --info=progress2 /path/to/source-tiffs/ "$POLONY_TIFF_DIR/"

find "$POLONY_TIFF_DIR" -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l
```

The importer searches recursively. Archives are not opened.

### 7.4 Generate and complete the mapping templates

```bash
ml import-legacy \
  --input "$POLONY_TIFF_DIR" \
  --write-templates "$POLONY_WORK_DIR/templates"

cp "$POLONY_WORK_DIR/templates/channels.template.json" "$POLONY_WORK_DIR/channels.json"
cp "$POLONY_WORK_DIR/templates/identities.template.json" "$POLONY_WORK_DIR/identities.json"
```

Now edit the two copied JSON files before continuing:

- In `channels.json`, fill `confirmed_by`, `confirmed_on` and `source_of_truth`; confirm which TIFF page is red/green and whether it is trained. Do not infer colour from a folder or file name.
- In `identities.json`, verify every `specimen_id`, `slide_id`, assay and replicate against laboratory records and set `confirmed: true` only after checking it.
- If the researcher has explicit records of fields counted as zero, also copy and complete `examined-fields.template.json`, then use the alternate command in section 7.6. Do not convert blank/unmarked fields into zeroes.

The exact schemas and examples are in [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md#3-files-you-provide).

### 7.5 Run the independent preflight

```bash
ml preflight \
  --input "$POLONY_TIFF_DIR" \
  --channel-map "$POLONY_WORK_DIR/channels.json" \
  --identity-map "$POLONY_WORK_DIR/identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15
```

Expected: exit code 0. Fix every reported mapping, identity or file-layout problem before continuing. To see the exit code immediately after the command, run `echo $?` (`0` means success).

If using verified zeros, add `--mode examined-fields-manifest --examined-fields "$POLONY_WORK_DIR/examined-fields.json"` to this preflight as well.

### 7.6 Dry-run the import

Default, safe `marked-only` mode: only field/page pairs with at least one embedded mark are eligible.

```bash
ml import-legacy \
  --input "$POLONY_TIFF_DIR" \
  --channel-map "$POLONY_WORK_DIR/channels.json" \
  --identity-map "$POLONY_WORK_DIR/identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --batch-size 4 \
  --epochs 10 \
  --dry-run
```

Read the complete report. In particular, confirm:

- `BLOCKERS: none`;
- the expected TIFF count and rejected-file reasons;
- the page-to-channel mapping;
- how many field/page pairs are included and how many are excluded as unproven;
- confirmed identities and the train/validation/test group counts;
- estimated disk and memory use.

Important: the import dry-run can exit 0 while still printing `BLOCKERS`; exit 0 only means the report was produced successfully. Do not continue until the report itself says there are no blockers.

Only when an examined-fields manifest contains genuine evidence for zero-counted fields, replace the command above with:

```bash
ml import-legacy \
  --input "$POLONY_TIFF_DIR" \
  --mode examined-fields-manifest \
  --examined-fields "$POLONY_WORK_DIR/examined-fields.json" \
  --channel-map "$POLONY_WORK_DIR/channels.json" \
  --identity-map "$POLONY_WORK_DIR/identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --batch-size 4 \
  --epochs 10 \
  --dry-run
```

Use exactly the same mode and files for the real import.

### 7.7 Build the immutable snapshot

For the default `marked-only` workflow, rerun the dry-run command without `--dry-run` and add workers:

```bash
ml import-legacy \
  --input "$POLONY_TIFF_DIR" \
  --channel-map "$POLONY_WORK_DIR/channels.json" \
  --identity-map "$POLONY_WORK_DIR/identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --workers "$POLONY_WORKERS"
```

For the verified-zero route, the corresponding real import is:

```bash
ml import-legacy \
  --input "$POLONY_TIFF_DIR" \
  --mode examined-fields-manifest \
  --examined-fields "$POLONY_WORK_DIR/examined-fields.json" \
  --channel-map "$POLONY_WORK_DIR/channels.json" \
  --identity-map "$POLONY_WORK_DIR/identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --workers "$POLONY_WORKERS"
```

The final output prints a version such as `ds-7e4c89cb...`. Copy the exact value:

```bash
export POLONY_SNAPSHOT=ds-REPLACE_WITH_THE_PRINTED_VERSION

ml verify "$POLONY_SNAPSHOT"
ml split --snapshot "$POLONY_SNAPSHOT"
```

Expected: `verify` prints `intact`; `split` shows the group-to-split table. Stop if train or validation is empty, identities look wrong, or related slides appear in different splits.

### 7.8 Lock the test set — irreversible without an explicit replacement

After inspecting the split:

```bash
ml split --snapshot "$POLONY_SNAPSHOT" --lock-test
```

Expected: `LOCKED test set`. Do this once. Later snapshots recognize locked specimens, slides and byte-identical files and keep them out of train/validation. Replacing a lock requires `--replace-lock --reason "..."` and changes the scientific test set, so it should be exceptional and documented.

### 7.9 Check the selected compute device

```bash
ml gpu-smoke --device "$POLONY_DEVICE"
```

Expected: device information followed by successful forward/backward checks. If `POLONY_DEVICE=auto` resolves to CPU, training is still valid but may be slower. The saved large run took about 32.6 minutes on a 20-core CPU; your time depends on hardware and dataset size.

### 7.10 Validate the training plan without training

```bash
ml train \
  --snapshot "$POLONY_SNAPSHOT" \
  --base general \
  --device "$POLONY_DEVICE" \
  --batch-size 4 \
  --epochs 10 \
  --early-stopping-patience 3 \
  --num-workers "$POLONY_WORKERS" \
  --dry-run
```

Expected: snapshot integrity, no test leakage, train/validation crop counts, device, memory mode, steps per epoch and validation cost, ending with `Training will not start because --dry-run was supplied.` A dry-run creates no model.

### 7.11 Optional one-epoch timing run

```bash
ml train \
  --snapshot "$POLONY_SNAPSHOT" \
  --base general \
  --device "$POLONY_DEVICE" \
  --batch-size 4 \
  --epochs 1 \
  --num-workers "$POLONY_WORKERS" \
  --notes "one-epoch timing benchmark"
```

This is a real training run and registers a separate candidate. It does not resume into the final run and should not be mistaken for it. Use its duration only to estimate runtime.

### 7.12 Train the candidate

```bash
ml train \
  --snapshot "$POLONY_SNAPSHOT" \
  --base general \
  --device "$POLONY_DEVICE" \
  --batch-size 4 \
  --lr 1e-4 \
  --epochs 10 \
  --early-stopping-patience 3 \
  --num-workers "$POLONY_WORKERS" \
  --notes "historical ImageJ marks; downscale 4; reviewed mappings"
```

Expected: a final line beginning `registered candidate ft-...`. The run saves the checkpoint with the best validation loss; the last epoch is reported separately. Copy the candidate ID, not the ID of the one-epoch benchmark:

```bash
export POLONY_CANDIDATE=ft-REPLACE_WITH_THE_FINAL_CANDIDATE_ID

ml list models
```

Expected: the chosen ID has status `candidate`, its dataset is `$POLONY_SNAPSHOT`, and its configuration is downscale 4 / `valid_unsaturated`. It is not active yet.

### 7.13 Evaluate candidate and baseline on the same locked test set

```bash
ml eval-model \
  --model "$POLONY_CANDIDATE" \
  --device "$POLONY_DEVICE"

ml eval-model \
  --model general \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --device "$POLONY_DEVICE"

ml compare --models "$POLONY_CANDIDATE" general
```

The candidate automatically uses the configuration stored at training. The explicit baseline settings make `general` see the same downscale and normalisation. Compare precision, recall, F1, count MAE and count bias at 5, 10 and 15 native pixels. Do not decide from validation loss alone.

For a stricter same-operating-threshold comparison, rerun both `eval-model` commands with the same explicit `--prob-thresh VALUE`; keep that result distinct from the models' own optimised-threshold comparison.

### 7.14 Promote only after accepting the evidence

If the comparison supports using the candidate and the app will run it at downscale 4 with `valid_unsaturated`:

```bash
ml promote "$POLONY_CANDIDATE" \
  --app-downscale 4 \
  --app-normaliser valid_unsaturated \
  --reason "better locked-test F1 and count MAE; reviewed count bias and failure cases"

ml list models
```

Do not use `--force` merely to bypass a failed comparison. Promotion changes only the registry inside `$POLONY_DATA_DIR`; it does not rewrite previous analyses.

**Current app-integration limitation:** `promote` updates the ML registry, but the current detection pipeline does not call the registry's `resolve_model()` function. The React model picker also lists Spotiflow's official pretrained names, not registry candidates. Therefore `active` and the promoted candidate do **not** automatically replace `general` in app detection today.

The current low-level workaround is to expose the registry model folders as Spotiflow's local model directory before launching the app, and send the **exact candidate ID** as `spotiflow.model_name` through the API (the present model picker cannot select it):

```bash
export POLONY_MODEL_DIR="$POLONY_DATA_DIR/models"

# Start the app from this same shell/environment.
PYTHONPATH=src "$POLONY_PY" -m polony_app --data-dir "$POLONY_DATA_DIR" --port 8000
```

The detection request must use:

```json
{
  "engine": "spotiflow",
  "spotiflow": {
    "model_name": "ft-REPLACE_WITH_THE_FINAL_CANDIDATE_ID",
    "downscale": 4,
    "normalizer": "valid_unsaturated",
    "prob_thresh": null
  }
}
```

`POLONY_MODEL_DIR="$POLONY_DATA_DIR/models"` lets the adapter locate the candidate folder by its exact ID. `prob_thresh: null` uses the threshold stored with the model. Treat this as an expert/API path until the registry is wired into the app UI. Run a small real-slide smoke check and confirm the recorded engine run contains the candidate ID, downscale 4, `valid_unsaturated` and the candidate checksum before an unattended experiment.

To undo a promotion:

```bash
ml rollback
ml list models
```

### 7.15 Resume and inspect safely

- An interrupted historical import keeps a staging directory and resumes when the identical import command is rerun. Do not change input files or mappings between attempts.
- A failed training run is not registered as a candidate. Rerun after fixing the reported problem.
- `ml list datasets` lists snapshot versions; `ml show "$POLONY_SNAPSHOT"` prints the complete manifest; `ml list models` lists candidates and the active model.
- Never train with `--allow-small`, `--allow-unconfirmed-identity`, an acknowledgement flag or a replaced test lock merely to silence a warning. Each changes what can legitimately be concluded and is recorded in provenance.
- Keep the exact Git commit, mapping files, snapshot ID, test-lock checksum, candidate ID and comparison output with the experiment record.
- Promotion currently governs the ML registry and comparison history; it does not by itself route React/app detection to that model. See section 7.14.

## 8. Short form using the runbook script

The repository also contains [`scripts/runbook/real_training.sh`](../../scripts/runbook/real_training.sh), which wraps the same stages. Read and set its `BASE`, `D`, `W`, `DATA`, `DEVICE` and GPU installation variables first. Then run one stage at a time; arguments shown as `<...>` must be replaced with IDs printed by earlier stages:

```bash
export BASE=/data/polony-training

bash scripts/runbook/real_training.sh install
bash scripts/runbook/real_training.sh upload
bash scripts/runbook/real_training.sh templates
bash scripts/runbook/real_training.sh mapping
# Edit and verify the generated mappings here.
bash scripts/runbook/real_training.sh preflight
bash scripts/runbook/real_training.sh dryrun
# Read the report and require BLOCKERS: none.
bash scripts/runbook/real_training.sh import
bash scripts/runbook/real_training.sh verify <snapshot-version>
bash scripts/runbook/real_training.sh split <snapshot-version>
bash scripts/runbook/real_training.sh lock <snapshot-version>
bash scripts/runbook/real_training.sh smoke
bash scripts/runbook/real_training.sh traindry <snapshot-version>
bash scripts/runbook/real_training.sh bench <snapshot-version>
bash scripts/runbook/real_training.sh train <snapshot-version>
bash scripts/runbook/real_training.sh eval <candidate-id>
bash scripts/runbook/real_training.sh compare <candidate-id>
```

The script intentionally does not promote a model. Use the explicit promotion command from section 7.14 after reviewing the comparison.

## 9. Alternative source: corrections reviewed in the app

If the labels come from analyses reviewed inside PoCo rather than historical ImageJ TIFFs, installation and sections 7.8–7.15 stay the same. Replace sections 7.3–7.7 with a snapshot built from the reviewed analysis IDs:

```bash
ml build \
  --analyses ANALYSIS_ID_1 ANALYSIS_ID_2 ANALYSIS_ID_3 \
  --identity-map "$POLONY_WORK_DIR/reviewed-identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --require-finalized \
  --dry-run

# After reading and accepting the dry-run, repeat without --dry-run:
ml build \
  --analyses ANALYSIS_ID_1 ANALYSIS_ID_2 ANALYSIS_ID_3 \
  --identity-map "$POLONY_WORK_DIR/reviewed-identities.json" \
  --downscale 4 \
  --normaliser valid_unsaturated \
  --fractions 0.7,0.15,0.15 \
  --require-finalized
```

The identity file for this route is JSON keyed by analysis ID; see [FINE_TUNING.md](FINE_TUNING.md#2-build-a-snapshot--build). The same rules still apply: only eligible reviewed regions become crops, train/validation/test groups must be independent, and the locked test set is never trained on.

Copying a checkpoint directory alone does not register or activate it. The repository's `artifacts/` directory is an evidence bundle, not an automatically installed model or a complete training dataset.
