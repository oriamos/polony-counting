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

## 7. Run it yourself

Use the Python environment containing the app and optional AI dependencies. Run commands from the repository with `PYTHONPATH=src` if the package is not installed. Keep **one explicit `--data-dir` before the subcommand** throughout; otherwise a command may use another database/model registry.

1. For historical TIFFs, follow [LEGACY_FINETUNING.md](LEGACY_FINETUNING.md), starting with channel and identity mapping, preflight and import dry-run. For app-reviewed corrections, follow [FINE_TUNING.md](FINE_TUNING.md).
2. Verify the snapshot, inspect its grouped split and lock the test set.
3. Run a training dry-run, then a one-epoch timing run, then the intended candidate training. A timing run creates its own candidate; it does not resume into the next run.
4. Evaluate both models at the intended configuration, then compare. For a snapshot trained at downscale 4 with `valid_unsaturated`, replace `DATA` and `CANDIDATE_ID` below with your actual directory and model ID:

```bash
python -m polony_app.ml --data-dir DATA eval-model --model CANDIDATE_ID
python -m polony_app.ml --data-dir DATA eval-model --model general --downscale 4 --normaliser valid_unsaturated
python -m polony_app.ml --data-dir DATA compare --models CANDIDATE_ID general
```

5. Promotion is a separate decision. The code checks locked-test evaluations, F1 and configuration compatibility; it does **not** establish scientific validity or automatically reject every thin test set. Supply the configuration you will actually use (for this example, `--app-downscale 4 --app-normaliser valid_unsaturated`). Promotion updates that data directory's registry; app analyses must resolve the intended model and use its matching settings.

Copying a checkpoint directory alone does not register or activate it. The repository's `artifacts/` directory is an evidence bundle, not an automatically installed model or a complete training dataset.
