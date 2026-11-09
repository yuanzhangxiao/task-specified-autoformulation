"""Frozen-model forecasts respect the split, full roster and numerical failures."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import prefix_initialization as prefix
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicSplit
from scripts import iclr_frozen_t2_forecast as study
from tests.test_prefix_initialization import problem, row


def source(clipped=False):
    prob = problem().model_dump(mode="json")
    if clipped:
        prob["base_candidate"]["state_equations"][0]["rhs"] = "-theta*x+z+max(r,0)"
    point = {
        "theta": 0.7,
        "init_z_phi": 1.3,
        "trajectory:a:init_r_value": 0.4,
        "trajectory:b:init_r_value": 1.2,
    }
    return {
        "problem": prob,
        "model_sha256": content_sha256(prob),
        "task": "child",
        "best_sampled_admissible": {
            "parameters": point,
            "parameter_sha256": content_sha256(point),
            "sampled_domain_passed": True,
            "training_nmse": 0.011,
        },
    }


def test_only_frozen_shared_parameters_and_pooled_root_initialization_travel():
    record = source()
    result = study.freeze_fit(record, ["a", "b"])
    assert result["shared"] == {"theta": 0.7, "init_z_phi": 1.3}
    assert result["starts"]["training_root_median"] == {"init_r_value": 0.8}
    assert "trajectory:" not in json.dumps(result)
    invalid = deepcopy(record)
    invalid["best_sampled_admissible"]["parameters"]["theta"] = 15
    with pytest.raises(ValueError, match="coefficient vector"):
        study.freeze_fit(invalid, ["a", "b"])
    with pytest.raises(ValueError, match="Incomplete shared"):
        study.freeze_fit(record, ["a", "unexpected"])


@pytest.mark.parametrize("clipped", [False, True])
def test_unseen_roots_and_forecasts_are_invariant_to_suffix_outcomes(clipped):
    model = study.freeze_fit(source(clipped), ["a", "b"])
    e = json.loads(study.CONFIG.read_text())["public_evaluation"]
    original = row("new", 1.7)
    policy = prefix.PrefixPolicy(duration=0.8)
    observed = prefix.make_prefix(original, policy)
    sched = prefix.make_forecast_schedule(original, policy)
    poisoned = original.model_dump(mode="json")
    poisoned["targets"]["y"][9:] = [1e8] * (len(original.time) - 9)
    changed = type(original).model_validate(poisoned)
    assert prefix.make_prefix(changed, policy) == observed
    assert prefix.make_forecast_schedule(changed, policy) == sched
    out = study.forecast_row(model, observed, sched, {"y": 1.0}, e)
    assert out["forecast_success"], out
    assert out["inference"]["selected"]["roots"]["init_r_value"] == pytest.approx(
        1.7, abs=1e-4
    )
    assert (
        prefix.score_forecast(out, original, 0.8, {"y": 1.0})["normalized_mse"] < 1e-8
    )
    assert prefix.score_forecast(out, changed, 0.8, {"y": 1.0})["normalized_mse"] > 1e14
    with pytest.raises(TypeError, match="typed prefix"):
        study.forecast_row(model, original, sched, {"y": 1.0}, e)
    with pytest.raises(ValueError, match="Different prefix"):
        study.forecast_row(
            model,
            observed,
            sched.model_copy(update={"trajectory_id": "other"}),
            {"y": 1.0},
            e,
        )


def test_failed_or_interrupted_row_stays_in_denominator_on_resume(
    tmp_path, monkeypatch
):
    src = source()
    frozen = study.freeze_fit(src, ["a", "b"])
    models = [frozen, {**frozen, "task": "control"}]
    dest = tmp_path / "result"
    val = PublicSplit(
        name="val", fingerprint="fixture", rows=(row("new", 1.7), row("another", 1.9))
    )
    vf = tmp_path / "input/val.json"
    seal(vf, val.model_dump(mode="json"))
    c = json.loads(study.CONFIG.read_text())
    c["revised_method"]["tasks"] = ["child", "control"]
    c["public_evaluation"].update(
        primary_child="child",
        parent_control="control",
        validation_file_sha256=study.sha(vf),
    )
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(c))
    monkeypatch.setattr(study, "CONFIG", cfg)
    sealed_write(dest / "models.json", {"models": models})
    plan = sealed_write(
        dest / "plan.json",
        {
            "design": c,
            "runtime": study.runtime(),
            "models_sha256": content_sha256(models),
            "prefix": {"duration": 0.8},
            "target_scales": {"y": 1.0},
            "training_ids": ["a", "b"],
        },
    )
    sealed_write(
        dest / "rows/child/00.started.json",
        {
            "plan": plan["artifact_sha256"],
            "validation": content_sha256(val),
            "id": "new",
            "task": "child",
        },
    )
    summary = study.run(dest, vf)
    assert (
        summary["models"]["child"]["planned"] == 2
        and summary["models"]["child"]["available"] == 1
    )
    assert summary["models"]["child"]["pooled_nmse"] is None
    assert summary["models"]["control"]["available"] == 2
    assert summary["paired_child_minus_control"]["new"] is None
    assert (
        sealed_read(dest / "rows/child/00.json")["forecast"]["reason"] == "interrupted"
    )
    monkeypatch.setattr(study, "forecast_row", lambda *a: pytest.fail("cache missed"))
    assert study.run(dest, vf) == summary
    other = tmp_path / "invalid/val.json"
    seal(other, val.model_dump(mode="json") | {"name": "train"})
    with pytest.raises(ValueError, match="pinned public validation"):
        study.run(dest, other)


def test_persistence_and_pooled_scores_use_forecast_sample_counts():
    plan = {
        "design": {
            "public_evaluation": {"primary_child": "a", "parent_control": "b"},
            "revised_method": {"tasks": ["a", "b"]},
        },
        "target_scales": {"y": 1.0},
    }
    rows = [
        {
            "task": task,
            "id": str(i),
            "score": {
                "available": True,
                "sample_count": i + 1,
                "normalized_mse": loss,
                "per_target_normalized_mse": {"y": loss},
            },
            "minima": {"y": 0},
        }
        for task in ("a", "b", "persistence")
        for i, loss in enumerate((1.0, 4.0))
    ]
    out = study.summary(plan, rows)
    assert out["models"]["a"]["pooled_nmse"] == 3.0
    assert out["models"]["a"]["trajectory"] == {"median": 2.5, "mad": 1.5}
    assert np.allclose(list(out["paired_child_minus_control"].values()), 0.0)
