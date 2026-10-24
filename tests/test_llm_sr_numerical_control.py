"""Fixed supplied replay, causal initialization, and strict source boundaries."""

import hashlib

import numpy as np
import pytest

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.schemas.public_fitting import PublicSplit


def example():
    source = "  not executable Python; this must only be hashed\n"
    item = control.Transcription.model_validate(
        {
            "task": 4,
            "benchmark_id": "synthetic",
            "repetition": 0,
            "programs": {
                "x": {
                    "function_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "parameters": [0.7, 1.2],
                }
            },
            "parameters": {"a": 0.7, "b": 1.2},
            "hidden_states": {
                "z": {
                    "rhs": "-b*z",
                    "initial": "0.5*x",
                    "program_name": "supplied filter",
                    "rule": "filter",
                }
            },
            "targets": {"x": "-a*x+z"},
            "rules": ["filter"],
        }
    )
    programs = {
        "x": control.SourceProgram(
            function=source, parameters=(0.7, 1.2), variables=("x",)
        )
    }
    time = np.linspace(0, 2, 21)
    train = PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "synthetic",
            "rows": [
                {
                    "trajectory_id": "train0",
                    "time": time.tolist(),
                    "targets": {
                        "x": (2 * np.exp(-0.7 * time) - np.exp(-1.2 * time)).tolist()
                    },
                }
            ],
        }
    )
    return item, programs, ValidationContext(targets=("x",)), train


def test_supplied_replay_analytic_initials_and_no_native_execution():
    item, programs, context, training = example()
    control.bind_programs(item, programs)
    result = control.replay_training(item, context, training, control.ReplaySettings())
    assert result["status"] == "complete"
    assert result["normalized_mse"] < 1e-10
    assert result["supplied_parameters"] == item.parameters
    assert result["optimizer_calls"] == 0
    assert not result["refit_ready"]
    assert not result["native_equivalence_certified"]
    assert not result["native_program_executed"]


@pytest.mark.parametrize("kind", ["source", "parameters", "targets"])
def test_program_binding_fails_closed(kind):
    item, programs, _, _ = example()
    if kind == "source":
        programs["x"] = programs["x"].model_copy(update={"function": "changed"})
    elif kind == "parameters":
        programs["x"] = programs["x"].model_copy(update={"parameters": (1.0, 1.2)})
    else:
        programs = {}
    with pytest.raises(ValueError, match="differ"):
        control.bind_programs(item, programs)


def test_nontraining_and_target_forcing_rejected_before_simulation(monkeypatch):
    item, _, context, training = example()
    monkeypatch.setattr(control, "simulate_trajectory", lambda *a, **k: pytest.fail())
    with pytest.raises(ValueError, match="training only"):
        control.replay_training(
            item,
            context,
            training.model_copy(update={"name": "val"}),
            control.ReplaySettings(),
        )
    with pytest.raises(ValueError, match="target forcing"):
        control.compile_transcription(
            item, ValidationContext(targets=("x",), lagged_targets=("x",))
        )


def test_restricted_expression_cannot_execute_python():
    item, _, context, _ = example()
    item = item.model_copy(update={"targets": {"x": "__import__('os').getcwd()"}})
    with pytest.raises(ModelValidationError):
        control.compile_transcription(item, context)


def test_observation_future_is_used_only_for_scoring():
    item, _, context, training = example()
    changed = training.model_dump(mode="json")
    changed["rows"][0]["targets"]["x"][1:] = [100.0] * 20
    settings = control.ReplaySettings(method="DOP853")
    first = control.replay_training(item, context, training, settings)
    second = control.replay_training(
        item, context, PublicSplit.model_validate(changed), settings
    )
    assert (
        first["trajectories"][0]["predictions"]
        == second["trajectories"][0]["predictions"]
    )


def test_timeout_is_unavailable_not_a_penalty_score():
    item, _, context, training = example()
    result = control.replay_training(
        item, context, training, control.ReplaySettings(seconds=1e-9)
    )
    assert result["status"] == "replay_failed" and result["budget_exhausted"]
    assert result["normalized_mse"] is None
    assert result["per_target_normalized_mse"] == {}
    assert result["trajectories"][0]["per_target_failures"].keys() == {"x"}


def test_domain_audit_reports_guarded_denominator_without_certifying_domain():
    item, _, context, training = example()
    item = item.model_copy(update={"targets": {"x": "0/(a-a)"}})
    result = control.replay_training(item, context, training, control.ReplaySettings())
    audit = result["trajectories"][0]["domain_audit"]
    assert audit["possible_singularity"]
    assert not audit["continuous_domain_certified"]
    assert next(iter(audit["denominators"].values()))["guarded_samples"] == 21
