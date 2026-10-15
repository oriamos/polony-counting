# Shipped model and locked test set

| Path | What |
|---|---|
| `ft-7e4c89cb-8edc09e3/best.pt`, `last.pt` | Spotiflow weights fine-tuned on reviewed polony slides (`config.yaml`, `thresholds.yaml`, `train_config.yaml` next to them) |
| `registry.json` | model registry entry and evaluation log, as written by `python -m polony_app.ml` |
| `test-set/test_lock.json` | the locked test set (specimen groups, image hashes, lock checksum) used for every evaluation |
| `test-set/snapshot_manifest.json` | the dataset snapshot the model was trained from: every crop with its source image hash, split and label counts |

These files are a provenance record of the model that ships with the repository; the app itself loads models from its own
data directory (`<data_dir>/models`, see [docs/app/FINE_TUNING.md](../docs/app/FINE_TUNING.md)). To use the shipped model, copy
`ft-7e4c89cb-8edc09e3/` into `<data_dir>/models/` and register it as described there. Slide file names inside the JSON files
are the laboratory's own file names (sample codes and dates); the images themselves are not part of the repository.
