# Attribution for the legacy import

| Item | Value |
|---|---|
| Original repository | https://github.com/kirill-push/polony-counting (identical to `main` of this repository, commit `65cb245`, v0.6.5) |
| Licence | MIT, "Copyright (c) [2023] [Kirill Pushkarev]" (see `/LICENSE`) |
| Code copied verbatim | **None** |

Behaviour taken from the legacy code as a *reference* and re-implemented (new code, new tests):

| Legacy file | Behaviour referenced | What is different here |
|---|---|---|
| `src/polony/data/utils.py` `get_roi_coordinates` | ImageJ `counter_positions` select the plane, `counters` identify the field | No fixed list index (`roiread(path)[1]`); per-blob error isolation; explicit canonical coordinate frame; no class assumption |
| `src/polony/data/utils.py` `grid_to_squares` | Grid lines come from the `Overlays` composite shape, spacing = line distance, cells scanned row/column | Lines parsed with a proper shape-path reader, kept as floats, cells carry provenance; no silent clamping, no square skipping rules, empty cells flagged ambiguous |
| `src/polony/config/config.yaml`, `path_dict.json` | Grid origin 158 / 316 px, file naming `<batch>-<n>-<A/B>.tif` | Used only to document and to parse the old naming (`identity.py`) |
| `src/polony/data/make_dataset.py` | Sequential 90/10 split (documented as a defect) | Replaced by group-based, deterministic splitting |

Not copied: checkpoints (`src/polony/checkpoints/*.pth`), `path_dict.json`, datasets (not available / redistribution not verified).
If code is later copied from the legacy repository, add the file path, commit and a description of the modifications here.
