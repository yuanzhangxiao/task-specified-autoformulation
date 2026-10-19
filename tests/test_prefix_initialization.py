"""Causal prefix inference, joint root sensitivities and honest rank evidence."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as p
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def problem(*, shared=False, ambiguous=False):
    """A measured decay receives two latent transients with a linked initial DAG."""
    return p.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "prefix_test",
                "parent_candidate_id": None,
                "states": [
                    {"name": name, "kind": "latent"} for name in ("x", "z", "r")
                ],
                "state_equations": [
                    {"state": "x", "rhs": "-theta*x" if ambiguous else "-theta*x+z+r"},
                    {"state": "z", "rhs": "-z"},
                    {"state": "r", "rhs": "-2*r"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [
                    {
                        "name": "theta",
                        "scope": "global",
                        "role": "rate",
                        "bounds": {"lower": 0.1, "upper": 2},
                    }
                ],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z", "r")
                ],
            },
            "context": {"targets": ["y"]},
            "parameter_guesses": {"theta": 0.5},
            "graph": {
                "rules": {
                    "r": {
                        "initial": {
                            "mode": "value",
                            "scope": "shared_training" if shared else "trajectory",
                            "guess": 0.8,
                        }
                    },
                    "z": {
                        "initial": {
                            "mode": "map",
                            "expression": "phi*root",
                            "parameters": [{"name": "phi", "guess": 0.9}],
                            "state_initials": {"root": "r"},
                        }
                    },
                }
            },
        }
    )


def row(name, root, *, ambiguous=False, shift=0.0):
    time = np.linspace(0, 4, 41)
    exponential = np.exp(-0.7 * time)
    values = (
        exponential
        if ambiguous
        else exponential
        + root
        * (
            1.3 * (exponential - np.exp(-time)) / 0.3
            + (exponential - np.exp(-2 * time)) / 1.3
        )
    )
    return PublicTrajectory(
        trajectory_id=name, time=tuple(time + shift), targets={"y": tuple(values)}
    )


def training(*, ambiguous=False):
    return PublicSplit(
        name="train",
        fingerprint="synthetic-only",
        rows=(row("a", 0.4, ambiguous=ambiguous), row("b", 1.7, ambiguous=ambiguous)),
    )


@pytest.fixture(scope="module")
def fitted():
    model = problem()
    policy = p.NumericalPolicy(seconds=30, maximum_calls=100)
    result = p.fit_training(model, training(), policy)
    assert result["fit_success"], result
    return model, result, policy


def test_joint_training_recovers_shared_model_and_different_roots(fitted):
    _, result, _ = fitted
    assert result["normalized_mse"] < 1e-14
    assert result["shared_parameters"] == pytest.approx(
        {"theta": 0.7, "init_z_phi": 1.3}, abs=1e-5
    )
    assert result["training_roots"]["a"]["init_r_value"] == pytest.approx(0.4, abs=1e-5)
    assert result["training_roots"]["b"]["init_r_value"] == pytest.approx(1.7, abs=1e-5)
    assert result["rank_diagnostic"]["full_column_rank"]
    assert (
        result["verification_calls_started"]
        == result["verification_calls_completed"]
        == 2
    )


def test_joint_dag_and_scattered_trajectory_root_derivatives():
    oracle = p.TrainingOracle(problem(), training(), p.NumericalPolicy())
    vector = oracle.vector(oracle.start)
    residual, derivative = oracle.evaluate(vector, monotonic() + 20)
    for index in range(len(vector)):
        step = np.eye(len(vector))[index] * 1e-5
        plus = oracle.evaluate(vector + step, monotonic() + 20, jacobian=False)[0]
        minus = oracle.evaluate(vector - step, monotonic() + 20, jacobian=False)[0]
        assert derivative[:, index] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)
    root_a = oracle.names.index("trajectory:a:init_r_value")
    root_b = oracle.names.index("trajectory:b:init_r_value")
    half = len(residual) // 2
    assert derivative[:half, root_b] == pytest.approx(0)
    assert derivative[half:, root_a] == pytest.approx(0)
    assert np.linalg.norm(derivative[:half, root_a]) > 1


def test_prefix_infers_new_root_and_free_forecast_freezes_shared_parameters(fitted):
    model, fit, numerical = fitted
    before = content_sha256(fit)
    heldout = row("heldout", 2.3)
    policy = p.PrefixPolicy(duration=0.8)
    prefix = p.make_prefix(heldout, policy)
    oracle = p.PrefixOracle(model, fit, prefix, numerical)
    assert oracle.start == {"init_r_value": 0.8}
    assert oracle.names == ("init_r_value",)
    result = p.infer_prefix(model, fit, prefix, numerical)
    assert result["fit_success"]
    assert result["roots"]["init_r_value"] == pytest.approx(2.3, abs=1e-5)
    assert result["shared_parameters"] == fit["shared_parameters"]
    prediction = p.forecast(
        model, fit, result, p.make_forecast_schedule(heldout, policy), numerical
    )
    score = p.score_forecast(prediction, heldout, prefix.cutoff, fit["training_scales"])
    assert score["normalized_mse"] < 1e-14
    assert score["sample_count"] == 32
    assert content_sha256(fit) == before


def test_suffix_poison_cannot_change_prefix_inference_or_forecast(fitted):
    model, fit, numerical = fitted
    original = row("poison", 1.4)
    policy = p.PrefixPolicy(duration=0.5)
    payload = original.model_dump(mode="json")
    payload["targets"]["y"][6:] = [9000.0] * 35
    poisoned = PublicTrajectory.model_validate(payload)
    first, second = [p.make_prefix(item, policy) for item in (original, poisoned)]
    assert first == second
    assert p.make_forecast_schedule(original, policy) == p.make_forecast_schedule(
        poisoned, policy
    )
    inference = p.infer_prefix(model, fit, first, numerical)
    prediction = p.forecast(
        model, fit, inference, p.make_forecast_schedule(poisoned, policy), numerical
    )
    assert (
        p.score_forecast(prediction, original, first.cutoff, fit["training_scales"])[
            "normalized_mse"
        ]
        < 1e-14
    )
    assert (
        p.score_forecast(prediction, poisoned, first.cutoff, fit["training_scales"])[
            "normalized_mse"
        ]
        > 1e6
    )


def test_time_zero_prefix_can_fit_but_reports_unidentified_root(fitted):
    model, fit, numerical = fitted
    prefix = p.make_prefix(row("zero", 2), p.PrefixPolicy(duration=0))
    result = p.infer_prefix(model, fit, prefix, numerical)
    assert result["fit_success"]
    assert result["rank_diagnostic"]["rank"] == 0
    assert not result["rank_diagnostic"]["full_column_rank"]
    assert result["roots"] == {"init_r_value": 0.8}


def test_unobserved_roots_remain_rank_deficient_without_preventing_fit():
    result = p.fit_training(
        problem(ambiguous=True), training(ambiguous=True), p.NumericalPolicy()
    )
    assert result["fit_success"] and result["normalized_mse"] < 1e-14
    assert not result["rank_diagnostic"]["full_column_rank"]
    assert result["rank_diagnostic"]["rank"] == 1


def test_shared_root_baseline_cannot_explain_same_initial_different_trajectories():
    result = p.fit_training(problem(shared=True), training(), p.NumericalPolicy())
    assert result["fit_success"]
    assert result["normalized_mse"] > 0.1
    assert result["training_roots"] == {"a": {}, "b": {}}


def test_typed_prefix_rejects_suffix_full_targets_and_invalid_times():
    prefix = p.make_prefix(row("a", 1), p.PrefixPolicy(duration=0.5))
    payload = prefix.model_dump(mode="json")
    payload["time"][-1] = 0.7
    with pytest.raises(ValidationError, match="after cutoff"):
        p.PrefixView.model_validate(payload)
    payload = prefix.model_dump(mode="json")
    payload["hidden_states"] = {"z": [1] * 6}
    with pytest.raises(ValidationError):
        p.PrefixView.model_validate(payload)
    schedule = p.make_forecast_schedule(row("a", 1), prefix.policy).model_dump(
        mode="json"
    )
    schedule["targets"] = {"y": [1] * 41}
    with pytest.raises(ValidationError):
        p.ForecastSchedule.model_validate(schedule)


def test_future_auxiliary_forcing_and_unallowlisted_inputs_rejected(fitted):
    model, _, numerical = fitted
    payload = model.model_dump(mode="json")
    payload["context"]["auxiliaries"] = ["a"]
    payload["base_candidate"]["state_equations"][0]["rhs"] += "+a"
    changed = p.PrefixFitProblem.model_validate(payload)
    system = p._compile(changed, numerical)[0]
    prefix = p.make_prefix(row("a", 1), p.PrefixPolicy(duration=0.5))
    with pytest.raises(ValueError, match="auxiliary"):
        p._validate_future(system, prefix)
    with pytest.raises(ValueError, match="supplied external"):
        p.make_prefix(row("a", 1), p.PrefixPolicy(duration=0.5, future_inputs=("y",)))
    with pytest.raises(ValidationError):
        p.PrefixPolicy(duration=-1)


def test_cutoff_strict_scoring_and_empty_horizon(fitted):
    model, fit, numerical = fitted
    heldout = row("cutoff", 1)
    policy = p.PrefixPolicy(duration=0.5)
    inference = p.infer_prefix(model, fit, p.make_prefix(heldout, policy), numerical)
    prediction = p.forecast(
        model, fit, inference, p.make_forecast_schedule(heldout, policy), numerical
    )
    prediction["predictions"]["y"][5] += 1e8
    assert (
        p.score_forecast(prediction, heldout, 0.5, fit["training_scales"])[
            "normalized_mse"
        ]
        < 1e-14
    )
    prediction["cutoff"] = 4
    assert not p.score_forecast(prediction, heldout, 4, fit["training_scales"])[
        "available"
    ]


def test_changed_initial_boundary_or_training_fit_cannot_reuse_inference(fitted):
    model, fit, numerical = fitted
    heldout = row("sealed", 1)
    policy = p.PrefixPolicy(duration=0.5)
    inference = p.infer_prefix(model, fit, p.make_prefix(heldout, policy), numerical)
    schedule = p.make_forecast_schedule(heldout, policy)
    changed = schedule.model_copy(update={"initial_targets": {"y": 2.0}})
    with pytest.raises(ValueError, match="boundary"):
        p.forecast(model, fit, inference, changed, numerical)
    changed_fit = deepcopy(fit)
    changed_fit["shared_parameters"]["theta"] += 0.1
    with pytest.raises(ValueError, match="frozen fit"):
        p.forecast(model, changed_fit, inference, schedule, numerical)


def test_budget_and_started_call_checkpoint_precede_rollout(monkeypatch):
    oracle = p.TrainingOracle(problem(), training(), p.NumericalPolicy())
    saved = []
    original = oracle.evaluate

    def evaluate(*args, **kwargs):
        assert saved and saved[-1]["calls"] >= 1
        return original(*args, **kwargs)

    monkeypatch.setattr(oracle, "evaluate", evaluate)
    result = p._solve(
        oracle,
        p.NumericalPolicy(maximum_calls=1),
        lambda value: saved.append(deepcopy(value)),
    )
    assert result["search"]["calls"] == 1
    assert result["search"]["stop_reason"] == "budget_limited"


def test_wrong_split_and_incomplete_shared_guesses_fail_closed():
    data = training().model_copy(update={"name": "val"})
    with pytest.raises(ValueError, match="training only"):
        p.fit_training(problem(), data, p.NumericalPolicy())
    with pytest.raises(ValueError, match="exactly dynamics"):
        p.TrainingOracle(
            problem().model_copy(update={"parameter_guesses": {}}),
            training(),
            p.NumericalPolicy(),
        )


def test_forecast_binds_prefix_input_history_and_verifies_independently(fitted):
    original_model, original_fit, numerical = fitted
    payload = original_model.model_dump(mode="json")
    payload["context"]["external_inputs"] = ["u"]
    model = p.PrefixFitProblem.model_validate(payload)
    fit = deepcopy(original_fit)
    fit["problem_sha256"] = content_sha256(model)
    raw = row("input_history", 1.2).model_dump(mode="json")
    raw["external_inputs"] = {"u": list(np.linspace(0, 1, 41))}
    heldout = PublicTrajectory.model_validate(raw)
    policy = p.PrefixPolicy(duration=0.5, future_inputs=("u",))
    prefix = p.make_prefix(heldout, policy)
    inferred = p.infer_prefix(model, fit, prefix, numerical)
    schedule = p.make_forecast_schedule(heldout, policy)
    prediction = p.forecast(model, fit, inferred, schedule, numerical)
    assert prediction["forecast_success"]
    assert (
        prediction["verification_calls_started"]
        == prediction["verification_calls_completed"]
        == 2
    )
    assert set(prediction["solver_counters"]) == {"DOP853", "Radau"}
    changed = schedule.model_dump(mode="json")
    changed["external_inputs"]["u"][3] += 10
    with pytest.raises(ValueError, match="time/input history"):
        p.forecast(
            model, fit, inferred, p.ForecastSchedule.model_validate(changed), numerical
        )
    changed = schedule.model_dump(mode="json")
    changed["time"][3] += 0.001
    with pytest.raises(ValueError, match="time/input history"):
        p.forecast(
            model, fit, inferred, p.ForecastSchedule.model_validate(changed), numerical
        )
    changed = schedule.model_dump(mode="json")
    changed["external_inputs"]["u"][10] += 10
    assert p.forecast(
        model, fit, inferred, p.ForecastSchedule.model_validate(changed), numerical
    )["forecast_success"]


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), 100.0])
def test_frozen_shared_parameters_must_remain_inside_declared_domain(fitted, invalid):
    model, fit, numerical = fitted
    changed = deepcopy(fit)
    changed["shared_parameters"]["theta"] = invalid
    prefix = p.make_prefix(row("bad_parameter", 1), p.PrefixPolicy(duration=0.5))
    with pytest.raises(ValueError, match="physical parameters outside"):
        p.infer_prefix(model, changed, prefix, numerical)


def test_forecast_rejects_root_domain_violation(fitted):
    original_model, original_fit, numerical = fitted
    payload = original_model.model_dump(mode="json")
    payload["graph"]["rules"]["r"]["initial"]["role"] = "nonnegative_coefficient"
    model = p.PrefixFitProblem.model_validate(payload)
    fit = deepcopy(original_fit)
    fit["problem_sha256"] = content_sha256(model)
    heldout = row("root_domain", 1)
    policy = p.PrefixPolicy(duration=0.5)
    inference = p.infer_prefix(model, fit, p.make_prefix(heldout, policy), numerical)
    inference["roots"]["init_r_value"] = -0.1
    with pytest.raises(ValueError, match="physical parameters outside"):
        p.forecast(
            model, fit, inference, p.make_forecast_schedule(heldout, policy), numerical
        )


def test_failed_independent_forecast_is_unavailable(fitted, monkeypatch):
    model, fit, numerical = fitted
    heldout = row("failed_forecast", 1)
    policy = p.PrefixPolicy(duration=0.5)
    inferred = p.infer_prefix(model, fit, p.make_prefix(heldout, policy), numerical)
    original = p.symbolic_rollout

    def fail_second(system, trajectory, vector, settings, *args, **kwargs):
        if settings.integration_method == "Radau":
            raise RuntimeError("independent solver failure")
        return original(system, trajectory, vector, settings, *args, **kwargs)

    monkeypatch.setattr(p, "symbolic_rollout", fail_second)
    prediction = p.forecast(
        model, fit, inferred, p.make_forecast_schedule(heldout, policy), numerical
    )
    assert not prediction["forecast_success"]
    assert prediction["verification_calls_started"] == 2
    assert prediction["verification_calls_completed"] == 1
    assert prediction["predictions"] == {}
    assert not p.score_forecast(prediction, heldout, 0.5, fit["training_scales"])[
        "available"
    ]


def test_all_known_parameter_free_problem_verifies_without_optimizer():
    model = p.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "all_known",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [{"state": "x", "rhs": "1"}],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 0}
                ],
            },
            "context": {"targets": ["y"]},
            "graph": {"rules": {}},
        }
    )
    data = PublicSplit(
        name="train",
        fingerprint="known",
        rows=(
            PublicTrajectory(
                trajectory_id="known", time=(0, 1, 2), targets={"y": (2, 3, 4)}
            ),
        ),
    )
    fit = p.fit_training(model, data, p.NumericalPolicy())
    assert fit["fit_success"]
    assert fit["search"]["calls"] == 0
    assert fit["realized_initial_states"] == {"known": {"x": 2}}
    prefix = p.make_prefix(data.rows[0], p.PrefixPolicy(duration=0))
    inference = p.infer_prefix(model, fit, prefix, p.NumericalPolicy())
    assert inference["fit_success"] and inference["roots"] == {}
