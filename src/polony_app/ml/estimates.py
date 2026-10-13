"""Pure arithmetic: what a fine-tuning run will cost, BEFORE it is started (no torch, no I/O).

Used by ``ml import-legacy --dry-run`` and meant to be reused by ``ml train --dry-run``.

Formulas (verified against Spotiflow 0.6.5 ``Spotiflow.fit`` / ``SpotiflowTrainingWrapper.generate_dataloaders``, see
``model/trainer.py``):

* an *epoch* draws ``num_train_samples`` images (with replacement) when that option is given, otherwise it is one pass
  over the ``n_train`` training crops; the training dataloader has ``batch_size`` and does not drop the last batch, so
  ``steps_per_epoch = ceil(samples_per_epoch / batch_size)``;
* ``optimizer_steps = steps_per_epoch * epochs`` (one optimiser step per batch, no gradient accumulation);
* the validation dataloader uses batch size 1: every validation pass evaluates ``n_val`` batches, one pass at the end of
  every epoch (plus Lightning's 2-batch sanity check before the first epoch);
* ``train`` loads ALL train + val crops into memory first (page-normalised float32, block-mean downscaled), so the
  resident size is ``(n_train + n_val) * ceil-trimmed(h / d) * (w / d) * 4`` bytes (the bottom/right remainder of a crop
  that is not a multiple of the downscale ``d`` is trimmed, never padded).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence


def prepared_bytes_of(shapes_hw: Sequence[Sequence[int]], downscale: int = 1, dtype_bytes: int = 4) -> int:
    """Memory of the network inputs of crops with the given native (h, w) shapes after the block-mean downscale."""
    d = max(1, int(downscale))
    return int(sum((int(h) // d) * (int(w) // d) * int(dtype_bytes) for h, w in shapes_hw))


def estimate_training(n_train_crops: int, n_val_crops: int, batch_size: int, epochs: int,
                      num_train_samples: Optional[int] = None, downscale: int = 1,
                      crop_hw: Sequence[int] = (316, 316), dtype_bytes: int = 4,
                      train_shapes_hw: Optional[Sequence[Sequence[int]]] = None,
                      val_shapes_hw: Optional[Sequence[Sequence[int]]] = None) -> dict:
    """Optimiser steps, batches and prepared (in-memory) size of a fine-tuning run.

    ``crop_hw`` is the native shape assumed for every crop unless exact ``train_shapes_hw`` / ``val_shapes_hw`` are given."""
    n_train, n_val = max(0, int(n_train_crops)), max(0, int(n_val_crops))
    bs, ep = max(1, int(batch_size)), max(0, int(epochs))
    samples = int(num_train_samples) if num_train_samples else n_train
    steps_epoch = math.ceil(samples / bs) if samples > 0 else 0
    d = max(1, int(downscale))
    h, w = int(crop_hw[0]), int(crop_hw[1])
    tr_shapes = train_shapes_hw if train_shapes_hw is not None else [(h, w)] * n_train
    va_shapes = val_shapes_hw if val_shapes_hw is not None else [(h, w)] * n_val
    tr_bytes = prepared_bytes_of(tr_shapes, d, dtype_bytes)
    va_bytes = prepared_bytes_of(va_shapes, d, dtype_bytes)
    return {
        "n_train_crops": n_train, "n_val_crops": n_val, "batch_size": bs, "epochs": ep,
        "num_train_samples": int(num_train_samples) if num_train_samples else None,
        "samples_per_epoch": samples,
        "steps_per_epoch": steps_epoch,
        "batches_per_epoch": steps_epoch,
        "optimizer_steps": steps_epoch * ep,
        "validation_batches": n_val,                   # per validation pass (batch size 1)
        "validation_passes": ep,                       # one per epoch (+ a 2-batch sanity check before epoch 1)
        "validation_batches_total": n_val * ep,
        "downscale": d,
        "prepared_train_bytes": tr_bytes, "prepared_val_bytes": va_bytes,
        "prepared_bytes": tr_bytes + va_bytes,
        "prepared_dtype_bytes": int(dtype_bytes),
        "assumed_crop_hw": [h, w] if train_shapes_hw is None else None,
        "basis": "Spotiflow 0.6.5: steps/epoch = ceil(samples/batch_size), validation batch size 1, all crops resident as float32",
    }
