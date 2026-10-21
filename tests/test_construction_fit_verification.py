"""Frozen endpoints receive state/output checks, with honest costs and failures."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicRolloutTimeout
from autoformalism.research import construction_fit_verification as saved
from tests.test_prefix_initialization import problem, row, training


@pytest.fixture
def case():
    """A known joint endpoint with two distinct trajectory roots; no fitting."""
    return (
        problem(),
        training(),
        {
            "theta": 0.7,
            "init_z_phi": 1.3,
            "trajectory:a:init_r_value": 0.4,
            "trajectory:b:init_r_value": 1.7,
        },
        kernel.NumericalPolicy(),
    )


def run(case, execution=None, checkpoint=None):
    return saved.verify_saved_parameters(
        *case, execution or verification.ExecutionPolicy(), checkpoint
    )


class Clock:
    def __init__(self):
        self.wall, self.cpu = 100.0, 10.0

    def advance(self, wall, cpu=0.0):
        self.wall += wall
        self.cpu += cpu


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    for module in (saved, verification):
        monkeypatch.setattr(module, "monotonic", lambda: value.wall)
        monkeypatch.setattr(module, "process_time", lambda: value.cpu)
    return value


def fake_rollout(monkeypatch, clock, *, error=None, elapsed=0.1, difference=0.0):
    calls = []

    def rollout(system, trajectory, vector, settings, deadline, *, sensitivities):
        assert sensitivities is False
        assert settings.relative_tolerance == 1e-9
        assert settings.absolute_tolerance == 1e-11
        calls.append((settings.integration_method, trajectory.trajectory_id, deadline))
        clock.advance(elapsed, 0.02)
        if error is not None:
            raise error
        values = np.column_stack([trajectory.targets[n] for n in system.channels])
        if settings.integration_method == "Radau":
            values = values + difference
        return (
            values,
            None,
            np.zeros((len(trajectory.time), system.state_count)),
            {"nfev": 7, "njev": 2, "nlu": 3, "segments": 1},
        )

    monkeypatch.setattr(kernel, "symbolic_rollout", rollout)
    return calls


def test_exact_joint_endpoint_succeeds_without_optimizer_or_sensitivities(
    case, monkeypatch
):
    original = kernel.symbolic_rollout
    calls = []

    def rollout(*args, **kwargs):
        calls.append((args[3].integration_method, args[1].trajectory_id))
        assert kwargs == {"sensitivities": False}
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("search, parameter sensitivity or rank computation")

    monkeypatch.setattr(kernel, "symbolic_rollout", rollout)
    monkeypatch.setattr(kernel.recovery_numerics, "search", forbidden)
    monkeypatch.setattr(kernel.TrainingOracle, "evaluate", forbidden)
    monkeypatch.setattr(kernel, "_rank", forbidden)
    monkeypatch.setattr(SymbolicODE, "initial_sensitivity", forbidden)
    before = tuple(content_sha256(value) for value in case)
    result = run(case)
    assert result["success"] and result["fit_success"]
    assert result["normalized_mse"] < 1e-16
    assert result["per_target_normalized_mse"]["y"] == result["normalized_mse"]
    assert result["parameters"] == case[2]
    assert result["parameters_sha256"] == before[2]
    assert before == tuple(content_sha256(value) for value in case)
    assert calls == [(m, r) for m in ("DOP853", "Radau") for r in ("a", "b")]
    assert result["verification_calls_started"] == 2
    assert result["verification_calls_completed"] == 2
    assert result["rollout_calls_started"] == result["rollout_calls_completed"] == 4
    assert result["rollout_calls_failed"] == 0
    assert (
        result["calls"] == result["completed_calls"] == result["optimizer_calls"] == 0
    )
    assert result["live_llm_calls"] == 0 and not result["test_data_opened"]
    assert result["no_search_performed"]
    assert result["rank_diagnostic"] == {
        "assessed": False,
        "reason": "state_output_only_verification",
    }
    for methods in result["trajectory_diagnostics"].values():
        for receipt in methods.values():
            assert receipt["started"] and receipt["completed"]
            assert receipt["solver_counters"]["nfev"] > 0
            assert receipt["elapsed_wall_seconds"] > 0
            assert receipt["elapsed_cpu_seconds"] > 0


def test_poor_saved_endpoint_is_scored_exactly_and_never_adjusted(case):
    model, data, parameters, numerical = case
    parameters["trajectory:a:init_r_value"] = 2.4
    result = run((model, data, parameters, numerical))
    scale = np.std(np.concatenate([r.targets["y"] for r in data.rows]))
    expected_a = (
        (np.asarray(row("a", 2.4).targets["y"]) - data.rows[0].targets["y"]) / scale
    ) ** 2
    assert result["success"]
    assert result["normalized_mse"] == pytest.approx(np.mean(expected_a) / 2, rel=1e-8)
    assert result["normalized_mse"] > 1
    assert result["per_row_scores"]["a"]["normalized_mse"] == pytest.approx(
        np.mean(expected_a), rel=1e-8
    )
    assert result["predictions"]["a"]["y"] == pytest.approx(
        row("a", 2.4).targets["y"], abs=2e-8
    )
    assert result["parameters"] == parameters


def test_multiple_targets_and_unequal_rows_use_training_sample_weights(case):
    model, data, parameters, numerical = case
    payload = model.model_dump(mode="json")
    payload["context"]["targets"].append("v")
    payload["base_candidate"]["observation_mappings"].append(
        {"channel": "v", "expression": "2*x"}
    )
    model = kernel.PrefixFitProblem.model_validate(payload)
    payload = data.model_dump(mode="json")
    for index, values in enumerate(payload["rows"]):
        count = 41 if index == 0 else 19
        values["time"] = values["time"][:count]
        values["targets"]["y"] = values["targets"]["y"][:count]
        values["targets"]["v"] = (
            2 * np.asarray(values["targets"]["y"])
            + 0.2 * np.asarray(values["time"])
        ).tolist()
    data = type(data).model_validate(payload)
    result = run((model, data, parameters, numerical))
    assert result["success"]
    squared = {}
    for name in ("y", "v"):
        squared[name] = np.concatenate(
            [
                (
                    (
                        np.asarray(result["predictions"][r.trajectory_id][name])
                        - r.targets[name]
                    )
                    / result["training_scales"][name]
                ) ** 2
                for r in data.rows
            ]
        )
    assert result["per_target_normalized_mse"] == pytest.approx(
        {name: np.mean(values) for name, values in squared.items()}
    )
    assert result["normalized_mse"] == pytest.approx(
        np.mean(np.concatenate(list(squared.values())))
    )
    assert result["per_target_normalized_mse"]["v"] > 1e-3
    assert result["per_target_normalized_mse"]["y"] < 1e-16
    assert result["per_row_scores"]["b"]["sample_count"] == 19


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "renamed_root", "nan", "inf", "bool", "text", "domain"],
)
def test_malformed_or_different_vector_is_refused_before_integration(
    case, change, monkeypatch
):
    parameters = case[2]
    if change == "missing":
        parameters.pop("trajectory:a:init_r_value")
    elif change == "extra":
        parameters["other"] = 1.0
    elif change == "renamed_root":
        parameters["trajectory:other:init_r_value"] = parameters.pop(
            "trajectory:a:init_r_value"
        )
    else:
        parameters["theta"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "bool": True,
            "text": "0.7",
            "domain": -1.0,
        }[change]
    monkeypatch.setattr(
        kernel, "symbolic_rollout", lambda *a, **k: pytest.fail("integration")
    )
    with pytest.raises(ValueError) as caught:
        run(case)
    receipt = caught.value.verification_receipt
    assert receipt["rollout_calls_started"] == 0
    assert not receipt["success"]
    assert receipt["timing"]["numerical_active_wall_seconds"] >= 0


@pytest.mark.parametrize("field", ["rtol", "atol", "solver_tolerance"])
def test_tolerances_cannot_be_relaxed(case, field, monkeypatch):
    monkeypatch.setattr(
        kernel, "symbolic_rollout", lambda *a, **k: pytest.fail("integration")
    )
    policy = case[3].model_copy(update={field: getattr(case[3], field) * 10})
    with pytest.raises(ValueError, match="frozen tolerances"):
        run((*case[:3], policy))


def test_validation_split_is_rejected_before_integration(case, monkeypatch):
    monkeypatch.setattr(
        kernel, "symbolic_rollout", lambda *a, **k: pytest.fail("integration")
    )
    with pytest.raises(ValueError, match="training only"):
        run((case[0], case[1].model_copy(update={"name": "val"}), *case[2:]))


def test_per_rollout_counters_and_callbacks_have_disjoint_costs(
    case, monkeypatch, clock
):
    calls = fake_rollout(monkeypatch, clock)
    snapshots = []

    def checkpoint(value):
        snapshots.append(deepcopy(value))
        clock.advance(2, 0.2)

    result = run(case, verification.ExecutionPolicy(numerical_seconds=1), checkpoint)
    assert result["success"]
    assert len(calls) == 4
    assert calls[0][2] == 103
    assert calls[2][2] == 105
    timing = result["timing"]
    assert timing["elapsed_wall_seconds"] == pytest.approx(6.4)
    assert timing["callback_wall_seconds"] == 6
    assert timing["numerical_wall_seconds"] == pytest.approx(0.4)
    assert timing["numerical_active_wall_seconds"] == pytest.approx(0.4)
    assert timing["elapsed_cpu_seconds"] == pytest.approx(0.68)
    assert timing["callback_cpu_seconds"] == pytest.approx(0.6)
    assert snapshots[0]["verification_calls_attempted"] == 1
    assert snapshots[0]["verification_calls_started"] == 0
    assert snapshots[1]["rollout_calls_completed"] == 2
    assert snapshots[2]["success"]
    for methods in result["trajectory_diagnostics"].values():
        for receipt in methods.values():
            assert receipt["elapsed_wall_seconds"] == pytest.approx(0.1)
            assert receipt["elapsed_cpu_seconds"] == pytest.approx(0.02)
            assert receipt["solver_counters"] == {
                "nfev": 7,
                "njev": 2,
                "nlu": 3,
                "segments": 1,
            }


def test_method_disagreement_is_unavailable(case, monkeypatch, clock):
    fake_rollout(monkeypatch, clock, difference=1e-3)
    result = run(case)
    assert not result["success"]
    assert result["normalized_mse"] is None
    assert not result["per_row_scores"] and not result["predictions"]
    assert result["solver_difference"] > 1e-6
    assert "disagree" in result["verification_error"]
    assert result["verification_calls_completed"] == 2


def test_timeout_keeps_original_row_method_diagnostic(case, monkeypatch, clock):
    diagnostic = {
        "trajectory_id": "a",
        "last_rhs_attempt": {"time": 0.3, "states": {"x": 1.0}},
        "sensitivities": False,
    }
    calls = fake_rollout(
        monkeypatch,
        clock,
        error=SymbolicRolloutTimeout("forced timeout", diagnostic),
    )
    result = run(case)
    assert not result["success"] and result["normalized_mse"] is None
    assert len(calls) == 1
    assert result["verification_calls_started"] == 1
    assert result["verification_calls_failed"] == 1
    assert result["rollout_calls_failed"] == 1
    assert result["rollout_calls_completed"] == 0
    receipt = result["trajectory_diagnostics"]["a"]["DOP853"]
    assert receipt["diagnostic"] == diagnostic
    assert receipt["solver_counters"] is None
    assert receipt["elapsed_wall_seconds"] == pytest.approx(0.1)
    assert result["timing"]["timeout_reason"] == "integration_timeout"


def test_returned_overrun_stops_before_next_trajectory(case, monkeypatch, clock):
    calls = fake_rollout(monkeypatch, clock, elapsed=2)
    result = run(case, verification.ExecutionPolicy(numerical_seconds=1))
    assert not result["success"]
    assert len(calls) == 1
    assert result["rollout_calls_completed"] == 1
    assert result["verification_calls_completed"] == 0
    assert result["verification_calls_failed"] == 1
    assert result["timing"]["numerical_wall_seconds"] == 2


@pytest.mark.parametrize("mode", ["timeout", "failure", "interrupt"])
def test_final_callback_never_preserves_false_success(case, monkeypatch, clock, mode):
    fake_rollout(monkeypatch, clock)
    snapshots = []

    def checkpoint(value):
        snapshots.append(value)
        if len(snapshots) == 3:
            clock.advance(7, 0.5)
            if mode == "failure":
                raise OSError("storage unavailable")
            if mode == "interrupt":
                raise KeyboardInterrupt()

    execution = verification.ExecutionPolicy(numerical_seconds=1, overall_seconds=5)
    if mode == "interrupt":
        with pytest.raises(KeyboardInterrupt) as caught:
            run(case, execution, checkpoint)
        result = caught.value.verification_receipt
    else:
        result = run(case, execution, checkpoint)
    assert not result["success"] and not result["fit_success"]
    assert result["normalized_mse"] is None
    assert not result["predictions"] and result["parameters"] is None
    assert result["timing"]["callback_wall_seconds"] == 7
    assert result["timing"]["callback_failures"] == int(mode != "timeout")
    assert result["timing"]["timeout_reason"] == "overall_wall_limit"
    assert len(snapshots) == 3


def test_setup_cost_consumes_budget_and_is_retained_on_interrupt(
    case, monkeypatch, clock
):
    original = kernel.TrainingOracle

    def slow(*args):
        clock.advance(2, 0.4)
        return original(*args)

    monkeypatch.setattr(kernel, "TrainingOracle", slow)
    calls = fake_rollout(monkeypatch, clock)
    result = run(case, verification.ExecutionPolicy(numerical_seconds=1))
    assert not result["success"] and not calls
    assert result["timing"]["numerical_active_wall_seconds"] == 2
    assert result["timing"]["elapsed_cpu_seconds"] == pytest.approx(0.4)

    def interrupted(*args):
        clock.advance(0.5, 0.1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(kernel, "TrainingOracle", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        run(case)
    receipt = caught.value.verification_receipt
    assert receipt["timing"]["elapsed_wall_seconds"] == 0.5
    assert receipt["timing"]["elapsed_cpu_seconds"] == pytest.approx(0.1)
    assert receipt["rollout_calls_started"] == 0


def test_numerical_interrupt_retains_started_rollout_costs(case, monkeypatch, clock):
    fake_rollout(monkeypatch, clock, error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as caught:
        run(case)
    result = caught.value.verification_receipt
    assert not result["success"]
    assert result["rollout_calls_started"] == result["rollout_calls_failed"] == 1
    assert result["verification_calls_failed"] == 1
    assert result["timing"]["numerical_wall_seconds"] == pytest.approx(0.1)
