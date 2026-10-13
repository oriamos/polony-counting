"""Wiring of the review fixes: identity-based lock check inside ``train`` and the lock flag of ``split``."""
import pytest

from polony_app.ml import dataset as ds
from polony_app.ml.cli import build_parser
from polony_app.ml.common import MLError
from polony_app.ml.train import TrainOptions, plan_training
from test_ml_splits import build, make_world


def test_train_refuses_when_a_source_shares_the_locked_identity(tmp_path, monkeypatch):
    store, ids = make_world(tmp_path)
    d = tmp_path / "d"
    s_a = build(store, d, sorted(ids.values()), seed=1)
    ds.lock_test_set(d, s_a.version)
    s_b = build(store, d, sorted(ids.values()), seed=2, fractions={"train": 0.75, "val": 0.25, "test": 0.0})
    monkeypatch.setattr("polony_app.ml.spotiflow_io.describe_base",
                        lambda b, dd: {"kind": "folder", "name": "x", "path": "/x", "registry_id": None})
    # group / sha / frozen checks are not what trips here: the identity check is
    monkeypatch.setattr("polony_app.ml.train._locked_sources", lambda dd, lock: (set(), set()))
    monkeypatch.setattr("polony_app.ml.train.lock_identity_conflicts", lambda m, lock: ["source X shares its specimen 'AS'"])
    with pytest.raises(MLError, match="identity"):
        plan_training(d, s_b.version, TrainOptions(base="x", allow_small=True))


def test_split_has_the_acknowledged_assumptions_flag():
    a = build_parser().parse_args(["split", "--snapshot", "v", "--lock-test", "--allow-acknowledged-assumptions-in-test-lock"])
    assert a.allow_acknowledged_assumptions_in_test_lock is True
    assert build_parser().parse_args(["split", "--snapshot", "v"]).allow_acknowledged_assumptions_in_test_lock is False
