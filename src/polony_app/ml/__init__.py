"""Fine-tuning data and model lifecycle: dataset snapshots, grouped splits, training, registry, evaluation.

Everything is explicit (CLI ``python -m polony_app.ml``); the app never trains or promotes a model by itself.
Importing this package does not import torch or spotiflow."""
from .registry import ModelRef, Registry, resolve_model  # noqa: F401
