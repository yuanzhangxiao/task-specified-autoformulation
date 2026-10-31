"""Fresh affine prefix recovery preserves roots, readouts, domains and budgets."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio_forecast as recovery
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.sensitivity_probe import SymbolicRolloutFailure
from autoformalism.schemas.public_fitting import PublicTrajectory
from tests.test_prefix_initialization import problem


def case(*, rate=2.0, nonlinear_map=False, nonlinear_output=False, nonnegative=False):
    """Exact synthetic outputs include a composite channel and mapped initial."""
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["state_equations"][1]["rhs"] = f"-{rate}*z"
    payload["base_candidate"]["observation_mappings"].append(
        {"channel": "v", "expression": "z*z+2*r" if nonlinear_output else "z+2*r"}
    )
    payload["context"]["targets"] = ["y", "v"]
    if nonlinear_map:
        payload["graph"]["rules"]["z"]["initial"]["expression"] = "phi*root*root"
    if nonnegative:
        payload["graph"]["rules"]["r"]["initial"]["role"] = "nonnegative_coefficient"
    model = kernel.PrefixFitProblem.model_validate(payload)
    root, phi, theta = 1.5, 1.3, 0.7
    initial_z = phi * (root**2 if nonlinear_map else root)
    time = np.linspace(0, 2, 9)
    first = np.exp(-theta * time)
    z, r = initial_z * np.exp(-rate * time), root * np.exp(-2 * time)
    values = first + initial_z * (first - np.exp(-rate * time)) / (rate - theta)
    values += root * (first - np.exp(-2 * time)) / (2 - theta)
    row = PublicTrajectory(
        trajectory_id="fresh",
        time=tuple(time),
        targets={
            "y": tuple(values),
            "v": tuple(z * z + 2 * r if nonlinear_output else z + 2 * r),
        },
    )
    fit = {
        "problem_sha256": content_sha256(model),
        "fit_success": True,
        "shared_parameters": {"theta": theta, "init_z_phi": phi},
        "training_scales": {"y": 1.0, "v": 2.0},
    }
    numerical = kernel.NumericalPolicy(seconds=15, verification_seconds=15)
    prefix = kernel.make_prefix(row, kernel.PrefixPolicy(duration=0.75))
    return model, fit, prefix, row, numerical


def infer(values, checkpoint=None, **kwargs):
    model, fit, prefix, _, numerical = values
    return recovery.infer_prefix(
        model,
        fit,
        prefix,
        numerical,
        execution.SearchExecutionPolicy(),
        checkpoint,
        **kwargs,
    )


def forecast(values, inference, checkpoint=None):
    model, fit, prefix, row, numerical = values
    return recovery.forecast(
        model,
        fit,
        inference,
        kernel.make_forecast_schedule(row, prefix.policy),
        numerical,
        verification.ExecutionPolicy(numerical_seconds=15),
        checkpoint,
    )


@pytest.mark.parametrize("rate", [2.0, 9.15e7])
def test_stiff_affine_roots_composite_outputs_maps_and_forecast(rate, monkeypatch):
    values = case(rate=rate)
    before = [content_sha256(value) for value in values]
    original, calls = kernel.symbolic_rollout, []

    def recorded(*args, **kwargs):
        calls.append(args[3].integration_method)
        assert args[3].integration_method == "Radau"
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "symbolic_rollout", recorded)
    inferred = infer(values)
    assert inferred["fit_success"], inferred
    assert inferred["root_residual_certificate"]["certified"]
    assert inferred["search"]["method"] == "bounded_linear_least_squares"
    assert inferred["search"]["budget_exhausted"] is False
    assert inferred["roots"] == pytest.approx({"init_r_value": 1.5}, abs=2e-9)
    assert inferred["shared_parameters"] == values[1]["shared_parameters"]
    assert inferred["search_calls_started"] == inferred["search_calls_completed"] == 2
    assert inferred["verification_calls_completed"] == 2
    assert inferred["realized_initial_states"]["fresh"]["z"] == pytest.approx(1.95)
    predicted = forecast(values, inferred)
    assert predicted["forecast_success"], predicted
    assert set(predicted["predictions"]) == {"y", "v"}
    assert set(predicted["states"]) == {"x", "z", "r"}
    score = kernel.score_forecast(
        predicted, values[3], values[2].cutoff, values[1]["training_scales"]
    )
    assert score["available"] and score["normalized_mse"] < 1e-16
    assert calls == ["Radau", "Radau"]
    assert before == [content_sha256(value) for value in values]
    audit = inferred["affine_audit"]["transition_precision"]
    assert (
        audit["multiprecision_completed"] > 0
        if rate > 1e6
        else audit["double_completed"] > 0
    )


@pytest.mark.parametrize(
    "option,condition",
    [
        ("nonlinear_map", "initial_state_affine_in_roots"),
        ("nonlinear_output", "readouts_jointly_affine_in_state_and_roots"),
    ],
)
def test_nonlinear_map_or_output_uses_bounded_affine_search(option, condition):
    inferred = infer(case(**{option: True}))
    assert inferred["fit_success"], inferred
    certificate = inferred["root_residual_certificate"]
    assert not certificate["certified"]
    assert not certificate["conditions"][condition]
    assert inferred["search"]["method"] == "bounded_nonlinear_affine_propagation"
    assert inferred["roots"]["init_r_value"] == pytest.approx(1.5, abs=1e-7)


def test_prefix_boundary_and_schedule_forbid_suffix_observations():
    values = case()
    model, fit, prefix, row, numerical = values
    poisoned = row.model_dump(mode="json")
    for outputs in poisoned["targets"].values():
        outputs[4:] = [9e99] * 5
    altered = PublicTrajectory.model_validate(poisoned)
    assert kernel.make_prefix(altered, prefix.policy) == prefix
    first, second = infer(values), infer((model, fit, prefix, altered, numerical))
    assert first["roots"] == second["roots"]
    assert (
        forecast(values, first)["predictions"]
        == forecast((model, fit, prefix, altered, numerical), second)["predictions"]
    )
    with pytest.raises(TypeError, match="prefix-only"):
        recovery.infer_prefix(
            model, fit, row, numerical, execution.SearchExecutionPolicy()
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        kernel.ForecastSchedule.model_validate(
            kernel.make_forecast_schedule(row, prefix.policy).model_dump()
            | {"targets": row.targets}
        )


def test_declared_nonnegative_roots_and_frozen_fit_checks():
    values = case(nonnegative=True)
    with pytest.raises(ValueError, match="outside declared domain"):
        infer(values, root_guesses={"init_r_value": -1})
    inferred = infer(values)
    altered = deepcopy(inferred)
    altered["shared_parameters"]["theta"] += 0.1
    with pytest.raises(ValueError, match="frozen values"):
        forecast(values, altered)
    altered = deepcopy(inferred)
    altered["roots"]["init_r_value"] = -1
    with pytest.raises(ValueError, match="outside declared domain"):
        forecast(values, altered)


def test_structured_original_equation_failure_is_retained(monkeypatch):
    values = case()
    diagnostic = {"trajectory_id": "fresh", "last_rhs_attempt": {"time": 0.25}}

    def fail(*args, **kwargs):
        raise SymbolicRolloutFailure("test rollout failed", diagnostic)

    monkeypatch.setattr(kernel, "symbolic_rollout", fail)
    inferred = infer(values)
    assert not inferred["fit_success"]
    assert inferred["roots"] is inferred["parameters"] is None
    assert inferred["search"]["best"] is not None
    assert inferred["verification_calls_started"] == 2
    assert inferred["verification_calls_completed"] == 1
    assert inferred["verification_calls_failed"] == 1
    assert inferred["failures"][0]["diagnostic"] == diagnostic


def test_disagreement_cannot_produce_forecast(monkeypatch):
    values = case()
    inferred = infer(values)
    original = kernel.symbolic_rollout

    def disagree(*args, **kwargs):
        output, derivative, states, counters = original(*args, **kwargs)
        return output + 1, derivative, states, counters

    monkeypatch.setattr(kernel, "symbolic_rollout", disagree)
    predicted = forecast(values, inferred)
    assert not predicted["forecast_success"]
    assert predicted["predictions"] == predicted["states"] == {}
    assert "disagree" in predicted["verification_error"]


def test_search_checkpoint_failure_prevents_calls(monkeypatch):
    def fail(value):
        raise OSError("cannot persist")

    result = infer(case(), fail)
    assert not result["fit_success"]
    assert result["search_calls_attempted"] == 1
    assert result["search_calls_started"] == 0
    assert result["verification_calls_started"] == 0
    assert result["search_timing"]["callback_failures"] == 1


def test_final_checkpoint_failure_clears_qualified_result():
    values = case()

    def fail(value):
        if value.get("fit_success") or value.get("forecast_success"):
            raise OSError("final persistence failure")

    result = infer(values, fail)
    assert not result["fit_success"] and result["roots"] is None
    assert result["verification_timing"]["callback_failures"] == 1
    predicted = forecast(values, infer(values), fail)
    assert not predicted["forecast_success"] and predicted["predictions"] == {}
    assert predicted["timing"]["callback_failures"] == 1


def test_late_affine_call_cannot_create_best(monkeypatch):
    values = case()
    clock = [100.0]
    for module in (recovery, execution, verification):
        monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    def late(oracle, vector, deadline, **kwargs):
        clock[0] += 20
        return np.ones(8), np.ones((8, 1))

    monkeypatch.setattr(recovery.affine.PrefixOracle, "evaluate", late)
    result = infer(values)
    assert not result["fit_success"]
    assert result["search"]["best"] is None
    assert result["search"]["budget_exhausted"]
    assert result["search_calls_started"] == result["search_calls_completed"] == 1
    assert result["verification_calls_started"] == 0
    assert result["search_timing"]["timeout_reason"] == "numerical_active_wall_limit"


def test_interrupt_receipt_retains_truthful_started_calls():
    def interrupt(value):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as caught:
        infer(case(), interrupt)
    result = caught.value.execution_receipt
    assert result["search_calls_attempted"] == 1
    assert result["search_calls_started"] == 0
    assert not result["fit_success"]


def test_parameter_free_roots_only_require_independent_verification():
    model, fit, prefix, row, numerical = case()
    payload = model.model_dump(mode="json")
    payload["graph"]["rules"]["r"]["initial"]["scope"] = "shared_training"
    model = kernel.PrefixFitProblem.model_validate(payload)
    fit = deepcopy(fit)
    fit["problem_sha256"] = content_sha256(model)
    fit["shared_parameters"]["init_r_value"] = 1.5
    values = model, fit, prefix, row, numerical
    inferred = infer(values)
    assert inferred["fit_success"], inferred
    assert inferred["roots"] == {}
    assert inferred["search"]["stop_reason"] == "no_free_coordinates"
    assert inferred["search_calls_started"] == 0
    assert inferred["verification_calls_completed"] == 2
    assert forecast(values, inferred)["forecast_success"]


def test_bounded_linear_solution_preserves_nonnegative_root_domain():
    model, fit, prefix, row, numerical = case(nonnegative=True)
    values = row.model_dump(mode="json")
    decay = np.exp(-0.7 * np.asarray(row.time))
    values["targets"]["y"] = (
        decay - (np.asarray(row.targets["y"]) - decay) / 1.5
    ).tolist()
    values["targets"]["v"] = (-np.asarray(row.targets["v"]) / 1.5).tolist()
    row = PublicTrajectory.model_validate(values)
    prefix = kernel.make_prefix(row, prefix.policy)
    inferred = infer((model, fit, prefix, row, numerical))
    assert inferred["fit_success"], inferred
    assert 0 <= inferred["roots"]["init_r_value"] < 1e-9
    assert inferred["normalized_mse"] > 0
    assert inferred["search"]["linear_solver_success"]


def test_synchronous_checkpoints_are_excluded_from_numerical_allowances(monkeypatch):
    clock = [100.0]
    for module in (recovery, execution, verification):
        monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    def evaluate(oracle, vector, deadline, **kwargs):
        assert deadline > clock[0]
        clock[0] += 0.1
        return vector - 1.5, np.ones((1, 1)) if kwargs.get("jacobian", True) else None

    monkeypatch.setattr(recovery.affine.PrefixOracle, "evaluate", evaluate)
    snapshots = []

    def checkpoint(result):
        snapshots.append(result)
        clock[0] += 1.0

    result = infer(case(), checkpoint)
    assert result["fit_success"], result
    assert result["search_timing"]["numerical_active_wall_seconds"] == pytest.approx(
        0.2
    )
    assert result["verification_timing"][
        "numerical_active_wall_seconds"
    ] == pytest.approx(0.2)
    assert result["overall_timing"]["elapsed_wall_seconds"] == pytest.approx(
        0.4 + len(snapshots)
    )
    assert result["search_calls_started"] == 2
    assert result["verification_calls_started"] == 2
