"""Frozen-parameter affine inference and target-free original-equation checks."""

from copy import deepcopy
from time import sleep

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.research import construction_paired_evaluation as c
from tests.test_prefix_initialization import problem, row, training


@pytest.fixture
def case():
    """Known shared coefficients, distinct unseen root and a dependent initial map."""
    model, numerical = problem(), kernel.NumericalPolicy()
    oracle = affine.TrainingOracle(model, training(), numerical)
    fit = {
        "problem_sha256": content_sha256(model),
        "fit_success": True,
        "shared_parameters": {"theta": 0.7, "init_z_phi": 1.3},
        "training_scales": oracle.scales,
    }
    heldout = row("unseen", 1.1)
    policy = kernel.PrefixPolicy(duration=1)
    return model, fit, heldout, numerical, policy


def infer(case, save=lambda value: None):
    model, fit, heldout, numerical, policy = case
    return c.infer(
        model, fit, kernel.make_prefix(heldout, policy), numerical, 10, 10, 100, save
    )


def forecast(case, inference, save=lambda value: None):
    model, fit, heldout, numerical, policy = case
    return c.forecast(
        model,
        fit,
        inference,
        kernel.make_forecast_schedule(heldout, policy),
        numerical,
        10,
        save,
    )


def test_recover_unseen_root_and_verify_original_equations(case, monkeypatch):
    original = kernel.symbolic_rollout
    methods = []

    def observed(*args, **kwargs):
        methods.append((args[3].integration_method, kwargs["sensitivities"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "symbolic_rollout", observed)
    before = deepcopy(case[1])
    inference = infer(case)
    assert inference["fit_success"]
    assert inference["roots"] == pytest.approx({"init_r_value": 1.1}, abs=1e-8)
    assert inference["shared_parameters"] == before["shared_parameters"]
    assert inference["verification_calls_started"] == 2
    result = forecast(case, inference)
    assert result["forecast_success"]
    score = kernel.score_forecast(result, case[2], 1, before["training_scales"])
    assert score["normalized_mse"] < 1e-16
    assert score["sample_count"] == 30
    assert methods == [("Radau", False), ("Radau", False)]
    assert case[1] == before
    assert inference["search_timing"]["numerical_active_wall_seconds"] > 0
    assert inference["verification_timing"]["numerical_active_wall_seconds"] > 0
    assert result["overall_timing"]["elapsed_wall_seconds"] > 0


def test_suffix_poison_never_reaches_root_inference_or_forecast(case):
    poisoned = case[2].model_copy(
        update={
            "targets": {
                "y": tuple(
                    value if time <= 1 else 1e12
                    for time, value in zip(
                        case[2].time, case[2].targets["y"], strict=True
                    )
                )
            }
        }
    )
    changed = (*case[:2], poisoned, *case[3:])
    a, b = infer(case), infer(changed)
    assert a["prefix_sha256"] == b["prefix_sha256"]
    assert a["roots"] == b["roots"]
    first, second = forecast(case, a), forecast(changed, b)
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["predictions"] == second["predictions"]


def test_analytic_map_case_verifies_without_root_search(case, monkeypatch):
    model = problem(shared=True)
    fit = deepcopy(case[1])
    fit.update(problem_sha256=content_sha256(model))
    fit["shared_parameters"]["init_r_value"] = 1.1

    def forbidden(*args, **kwargs):
        pytest.fail("shared constants must not be reoptimized on held-out data")

    monkeypatch.setattr(c.recovery_numerics, "search", forbidden)
    modified = (model, fit, *case[2:])
    inference = infer(modified)
    assert inference["fit_success"]
    assert inference["roots"] == {}
    assert inference["search"]["stop_reason"] == "no_free_coordinates"
    assert inference["search_calls_started"] == 0
    assert forecast(modified, inference)["forecast_success"]


def test_requires_physically_restricted_views(case):
    model, fit, heldout, numerical, _ = case
    with pytest.raises(TypeError, match="prefix-only"):
        c.infer(model, fit, heldout, numerical, 10, 10, 10, lambda value: None)
    inference = infer(case)
    with pytest.raises(TypeError, match="target-free"):
        c.forecast(model, fit, inference, heldout, numerical, 10, lambda value: None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit_sha256", "wrong"),
        ("problem_sha256", "wrong"),
        ("boundary_sha256", "wrong"),
        ("prefix_forcing_sha256", "wrong"),
        ("prefix_end_time", 2),
        ("cutoff", 2),
        ("shared_parameters", {"theta": 0.9, "init_z_phi": 1.3}),
        ("training_scales", {"y": 999}),
    ],
)
def test_forecast_binds_fit_boundary_forcing_and_frozen_parameters(case, field, value):
    inference = infer(case)
    with pytest.raises(ValueError):
        forecast(case, inference | {field: value})


def test_tighter_radau_level_can_recover_agreement(case, monkeypatch):
    original = kernel.PrefixOracle.evaluate

    def perturbed(self, *args, **kwargs):
        residual, derivative = original(self, *args, **kwargs)
        if kwargs.get("method") == "Radau" and self.settings.relative_tolerance > 1e-10:
            residual = residual + 1e-4
        return residual, derivative

    monkeypatch.setattr(kernel.PrefixOracle, "evaluate", perturbed)
    result = infer(case)
    assert result["fit_success"]
    assert result["verification_calls_started"] == 3
    assert result["replays"][1]["original_equations_verified"] is False
    assert result["replays"][2]["original_equations_verified"] is True
    assert result["replays"][2]["rtol"] == 1e-11


def test_no_usable_roots_after_all_independent_checks_disagree(case, monkeypatch):
    original = kernel.PrefixOracle.evaluate

    def perturbed(self, *args, **kwargs):
        residual, derivative = original(self, *args, **kwargs)
        return residual + (0.01 if kwargs.get("method") == "Radau" else 0), derivative

    monkeypatch.setattr(kernel.PrefixOracle, "evaluate", perturbed)
    result = infer(case)
    assert result["fit_success"] is False
    assert result["roots"] is None
    assert result["verification_calls_started"] == 4
    assert result["search"]["best"] is not None


def test_failed_checkpoint_prevents_verification(case, monkeypatch):
    calls = []

    def failed(value):
        calls.append(value)
        raise OSError("disk full")

    result = infer(case, failed)
    assert result["fit_success"] is False
    assert result["roots"] is None
    assert result["search_timing"]["callback_failures"] == 1
    assert result.get("verification_calls_started", 0) == 0
    assert len(calls) == 1


def test_last_verification_checkpoint_failure_revokes_forecast(case):
    inference = infer(case)
    calls = []

    def fail_after_success(value):
        calls.append(value)
        if value["forecast_success"]:
            raise OSError("cannot persist verified result")

    result = forecast(case, inference, fail_after_success)
    assert result["forecast_success"] is False
    assert result["predictions"] == {}
    assert result["verification_timing"]["callback_failures"] == 1


def test_nonfinite_affine_forecast_is_unavailable(case, monkeypatch):
    inference = infer(case)

    def invalid(self, row, vector, deadline, **kwargs):
        return np.full((len(row.time), 1), np.nan), None

    monkeypatch.setattr(affine._Propagation, "rollout", invalid)
    result = forecast(case, inference)
    assert not result["forecast_success"]
    assert result["predictions"] == {}
    assert result["verification_calls_failed"] == 1


def test_forecast_deadline_overrun_revokes_success(case, monkeypatch):
    inference = infer(case)
    original = affine._Propagation.rollout

    def late(*args, **kwargs):
        sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(affine._Propagation, "rollout", late)
    model, fit, heldout, numerical, policy = case
    result = c.forecast(
        model,
        fit,
        inference,
        kernel.make_forecast_schedule(heldout, policy),
        numerical,
        0.01,
        lambda value: None,
    )
    assert not result["forecast_success"]
    assert result["predictions"] == {}
    assert "Timeout" in result["error"]


def test_interruption_has_charged_receipt_and_no_prediction(case, monkeypatch):
    inference = infer(case)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(affine._Propagation, "rollout", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        forecast(case, inference)
    receipt = caught.value.execution_receipt
    assert not receipt["forecast_success"]
    assert receipt["verification_calls_started"] == 1
    assert receipt["verification_calls_failed"] == 1
    assert receipt["verification_timing"]["elapsed_wall_seconds"] > 0


def test_nonaffine_system_fails_without_silent_linearization(case):
    payload = case[0].model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] = "-theta*x*x+z+r"
    model = kernel.PrefixFitProblem.model_validate(payload)
    fit = case[1] | {"problem_sha256": content_sha256(model)}
    result = infer((model, fit, *case[2:]))
    assert not result["fit_success"]
    assert result["search_calls_started"] == 0
    assert "affine" in result["search_error"]


def test_bad_root_domain_blocks_forecast(case):
    inference = infer(case)
    result = forecast(case, inference | {"roots": {"init_r_value": float("nan")}})
    assert not result["forecast_success"]
    assert result["predictions"] == {}
