# Shipped model and locked test set

| Path | What |
|---|---|
| `ft-7e4c89cb-8edc09e3/best.pt`, `last.pt` | Spotiflow weights fine-tuned on historical ImageJ-marked field/page crops (`config.yaml`, `thresholds.yaml`, `train_config.yaml` next to them) |
| `registry.json` | model registry entry and evaluation log, as written by `python -m polony_app.ml` |
| `test-set/test_lock.json` | the locked test set (specimen groups, image hashes, lock checksum) used for every evaluation |
| `test-set/snapshot_manifest.json` | the dataset snapshot the model was trained from: every crop with its source image hash, split and label counts |

These files are a provenance record of the model that ships with the repository; the app itself loads models from its own
data directory (`<data_dir>/models`, see [docs/app/FINE_TUNING.md](../docs/app/FINE_TUNING.md)). Copying
`ft-7e4c89cb-8edc09e3/` alone does not register or activate it. The saved registry records it as a candidate, with no active model.
See [TRAINING_GUIDE.md](../docs/app/TRAINING_GUIDE.md) for the recorded results, configuration and limitations; the test crops and original images needed to rerun evaluation are not included. Slide file names inside the JSON files
are the laboratory's own file names (sample codes and dates); the images themselves are not part of the repository.
