"""Model registry: promote / rollback / refusals, resolve_model."""
import json
import threading

import pytest

from polony_app.ml.common import MLError, models_dir, sha256_file
from polony_app.ml.registry import DEFAULT_PRETRAINED, Registry, pretrained_id, resolve_model

LOCK = "a" * 64


CFG = {"downscale": 1, "normaliser": "valid_unsaturated"}


def ev(f1, lock=LOCK, p=None, r=None, config=CFG):
    return {"lock_sha256": lock, "params": {}, "label": "preliminary", "config": config,
            "metrics": {"precision": p or f1, "recall": r or f1, "f1": f1, "mean_match_distance_px": 1.0}}


def add_ft(reg, data_dir, mid, weights=b"w", config=CFG):
    d = models_dir(data_dir) / mid
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(weights)
    reg.add({"id": mid, "kind": "finetuned", "base": {"name": "general"}, "dataset_version": "ds-1", "artifact": mid,
             "weights_file": "best.pt", "weights_sha256": sha256_file(d / "best.pt"), "inference_config": config})
    return d


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path)


def test_empty_registry_and_default_resolution(tmp_path, reg):
    assert reg.list() == [] and reg.active_id() is None
    r = resolve_model("general", tmp_path)
    assert (r.kind, r.name, r.path) == ("pretrained", "general", None)
    assert resolve_model("active", tmp_path).name == DEFAULT_PRETRAINED
    assert not (tmp_path / "models" / "registry.json").exists()      # resolving never writes


def test_candidate_is_not_active_and_resolves_by_id(tmp_path, reg):
    d = add_ft(reg, tmp_path, "ft-1")
    e = reg.get("ft-1")
    assert e["status"] == "candidate" and reg.active_id() is None
    r = resolve_model("ft-1", tmp_path)
    assert r.kind == "finetuned" and r.path == d and r.weights_sha256 == e["weights_sha256"] and r.registry_id == "ft-1"
    assert resolve_model("active", tmp_path).kind == "pretrained"     # still the default
    with pytest.raises(MLError, match="already exists"):
        add_ft(reg, tmp_path, "ft-1") if False else reg.add({"id": "ft-1", "kind": "finetuned"})


def test_resolve_detects_missing_or_modified_weights(tmp_path, reg):
    d = add_ft(reg, tmp_path, "ft-1")
    (d / "best.pt").write_bytes(b"other")
    with pytest.raises(MLError, match="do not match"):
        resolve_model("ft-1", tmp_path)
    (d / "best.pt").unlink()
    with pytest.raises(MLError, match="missing"):
        resolve_model("ft-1", tmp_path)


def test_promote_requires_lock_and_evaluations(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-1")
    with pytest.raises(MLError, match="no locked test set"):
        reg.promote("ft-1", lock_sha256=None)
    with pytest.raises(MLError, match="no evaluation on the current locked test set"):
        reg.promote("ft-1", lock_sha256=LOCK)
    reg.add_evaluation("ft-1", ev(0.9))
    with pytest.raises(MLError, match="nothing to compare against"):
        reg.promote("ft-1", lock_sha256=LOCK)                           # the default model has no evaluation
    reg.ensure_pretrained("general")
    reg.add_evaluation(pretrained_id("general"), ev(0.8, lock="b" * 64))     # evaluated on a DIFFERENT lock
    with pytest.raises(MLError, match="nothing to compare against"):
        reg.promote("ft-1", lock_sha256=LOCK)
    assert reg.active_id() is None


def test_promote_with_comparison_then_rollback(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-1")
    add_ft(reg, tmp_path, "ft-2", b"w2")
    reg.ensure_pretrained("general")
    reg.add_evaluation("pretrained-general", ev(0.70))
    reg.add_evaluation("ft-1", ev(0.80))
    reg.add_evaluation("ft-2", ev(0.85))
    e1 = reg.promote("ft-1", lock_sha256=LOCK)
    assert reg.active_id() == "ft-1" and e1["previous_active"] is None and e1["status"] == "active" and e1["promoted_at"]
    assert e1["promotion"]["comparison"]["baseline"] == "pretrained-general"
    assert e1["promotion"]["comparison"]["delta"]["f1"] == pytest.approx(0.10)
    assert resolve_model("active", tmp_path).registry_id == "ft-1"
    e2 = reg.promote("ft-2", lock_sha256=LOCK)               # compared against the ACTIVE model ft-1
    assert e2["previous_active"] == "ft-1" and e2["promotion"]["comparison"]["baseline"] == "ft-1"
    assert reg.get("ft-1")["status"] == "retired" and reg.active_id() == "ft-2"
    r = reg.rollback()
    assert r == {"rolled_back": "ft-2", "restored": "ft-1"}
    assert reg.active_id() == "ft-1" and reg.get("ft-1")["status"] == "active" and reg.get("ft-2")["status"] == "retired"
    reg.rollback()
    assert reg.active_id() is None                           # back to the app default
    with pytest.raises(MLError, match="nothing to roll back"):
        reg.rollback()


def test_worse_candidate_refused_unless_forced_with_reason(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-1")
    reg.ensure_pretrained("general")
    reg.add_evaluation("pretrained-general", ev(0.9))
    reg.add_evaluation("ft-1", ev(0.7))
    with pytest.raises(MLError, match="WORSE"):
        reg.promote("ft-1", lock_sha256=LOCK)
    with pytest.raises(MLError, match="requires --reason"):
        reg.promote("ft-1", lock_sha256=LOCK, force=True)
    e = reg.promote("ft-1", lock_sha256=LOCK, force=True, reason="researcher wants it for a trial")
    assert e["promotion"]["forced"] and e["promotion"]["reason"] and e["promotion"]["overridden_checks"]
    log = reg.load()["log"]
    assert any(x["event"] == "promote" and x["forced"] for x in log)


def test_force_works_without_any_evaluation(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-1")
    e = reg.promote("ft-1", lock_sha256=None, force=True, reason="no test set yet")
    assert e["status"] == "active" and e["promotion"]["forced"]


def test_atomic_json_and_concurrent_writers(tmp_path, reg):
    def worker(i):
        reg.add({"id": f"m{i}", "kind": "finetuned", "dataset_version": "ds-1"})
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(json.loads((tmp_path / "models" / "registry.json").read_text())["models"]) == 8
    assert not list((tmp_path / "models").glob("*.tmp"))


def test_unknown_model_message(tmp_path, reg):
    with pytest.raises(MLError, match="not in the registry"):
        reg.promote("nope", lock_sha256=LOCK)
    with pytest.raises(MLError, match="not in the registry"):
        reg.get("nope")


# ---------------------------------------------------------------- configuration identity (downscale + normaliser)
def test_resolve_model_exposes_the_training_configuration(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-4", config={"downscale": 4, "normaliser": "valid_unsaturated"})
    r = resolve_model("ft-4", tmp_path)
    assert r.inference_config == {"downscale": 4, "normaliser": "valid_unsaturated"} and r.trained_downscale == 4
    assert "downscale 1 (trained at 4)" in r.config_warning(1, "valid_unsaturated")
    assert "normaliser 'auto'" in r.config_warning(4, "auto")
    assert r.config_warning(4, "valid_unsaturated") is None
    p = resolve_model("general", tmp_path)
    assert p.inference_config is None and p.trained_downscale is None and p.config_warning(4) is None


def test_legacy_model_without_recorded_configuration_is_not_silently_current(tmp_path, reg):
    d = models_dir(tmp_path) / "ft-old"
    d.mkdir(parents=True)
    (d / "best.pt").write_bytes(b"w")
    reg.add({"id": "ft-old", "kind": "finetuned", "artifact": "ft-old", "weights_file": "best.pt",
             "weights_sha256": sha256_file(d / "best.pt")})
    assert resolve_model("ft-old", tmp_path).inference_config == {"downscale": 1, "normaliser": "legacy_per_crop"}


def _setup_pair(tmp_path, reg, cand_cfg, base_cfg_eval=CFG, cand_eval_cfg=None):
    add_ft(reg, tmp_path, "ft-1", config=cand_cfg)
    reg.ensure_pretrained("general")
    reg.add_evaluation("pretrained-general", ev(0.70, config=base_cfg_eval))
    reg.add_evaluation("ft-1", ev(0.80, config=cand_eval_cfg or cand_cfg))


def test_promotion_refused_when_candidate_config_differs_from_the_app(tmp_path, reg):
    c4 = {"downscale": 4, "normaliser": "valid_unsaturated"}
    _setup_pair(tmp_path, reg, c4, base_cfg_eval=CFG)
    with pytest.raises(MLError, match="downscale 4, normaliser valid_unsaturated.*app's current inference configuration is downscale 1"):
        reg.promote("ft-1", lock_sha256=LOCK, app_config=CFG)
    assert reg.active_id() is None
    e = reg.promote("ft-1", lock_sha256=LOCK, app_config=CFG, acknowledge_config=True)
    cmp = e["promotion"]["comparison"]
    assert reg.active_id() == "ft-1" and cmp["candidate_config"] == c4 and cmp["baseline_config"] == CFG
    assert cmp["config_acknowledged"] and "config_mismatch_acknowledged" in cmp


def test_promotion_refused_when_the_normaliser_differs_from_the_app_default(tmp_path, reg):
    _setup_pair(tmp_path, reg, CFG)
    with pytest.raises(MLError, match="normaliser auto"):
        reg.promote("ft-1", lock_sha256=LOCK, app_config={"downscale": 1, "normaliser": "auto"})
    assert reg.promote("ft-1", lock_sha256=LOCK, app_config=CFG)["status"] == "active"


def test_each_model_is_compared_at_its_own_recorded_configuration(tmp_path, reg):
    c4 = {"downscale": 4, "normaliser": "valid_unsaturated"}
    # the candidate was evaluated at a configuration other than its own: that evaluation does not count
    _setup_pair(tmp_path, reg, c4, cand_eval_cfg=CFG)
    with pytest.raises(MLError, match="no evaluation on the current locked test set at downscale 4"):
        reg.promote("ft-1", lock_sha256=LOCK)
    reg.add_evaluation("ft-1", ev(0.80, config=c4, p=0.8))
    e = reg.promote("ft-1", lock_sha256=LOCK)            # app configuration not supplied: not compared, and said so
    assert e["status"] == "active" and "not compared" in e["promotion"]["comparison"]["app_config_note"]
    # the baseline pretrained model counts at the app's configuration; one made at another downscale is not a comparison
    reg2 = Registry(tmp_path / "other")
    add_ft(reg2, tmp_path / "other", "ft-1", config=c4)
    reg2.ensure_pretrained("general")
    reg2.add_evaluation("pretrained-general", ev(0.70, config=CFG))
    reg2.add_evaluation("ft-1", ev(0.80, config=c4))
    with pytest.raises(MLError, match=r"downscale 4.*nothing to compare against.*--downscale 4"):
        reg2.promote("ft-1", lock_sha256=LOCK, app_config=c4)


def test_promotion_compares_f1_at_10_px(tmp_path, reg):
    add_ft(reg, tmp_path, "ft-1")
    reg.ensure_pretrained("general")
    cand = ev(0.9)
    cand["metrics"]["by_tolerance"] = {"5": {"f1": 0.9}, "10": {"f1": 0.60}, "15": {"f1": 0.9}}
    reg.add_evaluation("pretrained-general", ev(0.70))
    reg.add_evaluation("ft-1", cand)
    with pytest.raises(MLError, match="WORSE"):
        reg.promote("ft-1", lock_sha256=LOCK)
