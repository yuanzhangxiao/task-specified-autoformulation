"""Scoped affine propagation preserves outputs, initial DAGs and root columns."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory
from tests.test_prefix_initialization import problem, training


@pytest.fixture
def coupled_case():
    """Multiple outputs, nonlinear coefficients, two roots and an initial map."""
    payload = problem().model_dump(mode="json")
    candidate = payload["base_candidate"]
    candidate["parameters"].append(
        {
            "name": "gain",
            "scope": "global",
            "role": "coefficient",
            "bounds": {"lower": -2, "upper": 3},
        }
    )
    payload["parameter_guesses"]["gain"] = 0.8
    candidate["state_equations"] = [
        {"state": "x", "rhs": "-theta*theta*x+z+r+gain*u+0.5*w+c"},
        {"state": "z", "rhs": "-z+theta*r+w"},
        {"state": "r", "rhs": "-2*r+0.3*u"},
    ]
    candidate["observation_mappings"].append(
        {"channel": "v", "expression": "z+gain*r+0.1*u+theta"}
    )
    payload["context"].update(
        targets=["y", "v"],
        external_inputs=["u"],
        auxiliaries=["w"],
        fixed_covariates=["c"],
    )
    rows = []
    for index, time in enumerate(
        ([0, 0.07, 0.2, 0.21, 0.8, 1.35, 2], [0, 0.1, 0.11, 0.4, 0.9, 1.7])
    ):
        t = np.asarray(time)
        rows.append(
            PublicTrajectory(
                trajectory_id=f"row{index}",
                time=tuple(time),
                targets={"y": tuple(1 + np.sin(t)), "v": tuple(2 + t**2)},
                external_inputs={"u": tuple(np.where(t <= 0.2, 3 * t, 0.6 - t))},
                auxiliaries={"w": tuple(np.cos(3 * t))},
                fixed_covariates={"c": 0.2 + index * 0.1},
            )
        )
    model = kernel.PrefixFitProblem.model_validate(payload)
    data = PublicSplit(name="train", fingerprint="synthetic-affine", rows=tuple(rows))
    oracle = affine.TrainingOracle(model, data, kernel.NumericalPolicy())
    start = oracle.start | {
        "trajectory:row0:init_r_value": 0.4,
        "trajectory:row1:init_r_value": 1.7,
    }
    return model, data, oracle, oracle.vector(start)


def test_multistate_multitarget_roots_maps_and_nonlinear_parameters(coupled_case):
    model, data, oracle, vector = coupled_case
    original = kernel.TrainingOracle(model, data, kernel.NumericalPolicy())
    assert oracle.audit == original.audit
    assert oracle.names == original.names
    assert oracle.start == original.start
    assert oracle.scales == original.scales
    assert oracle.lower == pytest.approx(original.lower)
    assert oracle.upper == pytest.approx(original.upper)
    before = content_sha256(model), content_sha256(data)
    residual, jacobian = oracle.evaluate(vector, monotonic() + 20)
    expected, derivative = original.evaluate(vector, monotonic() + 20)
    assert residual == pytest.approx(expected, abs=3e-8)
    assert jacobian == pytest.approx(derivative, abs=3e-8)
    assert before == (content_sha256(model), content_sha256(data))
    for index in range(len(vector)):
        step = np.eye(len(vector))[index] * 1e-5
        plus = oracle.evaluate(vector + step, monotonic() + 10, jacobian=False)[0]
        minus = oracle.evaluate(vector - step, monotonic() + 10, jacobian=False)[0]
        assert jacobian[:, index] == pytest.approx((plus - minus) / 2e-5, abs=2e-8)
    boundary = len(data.rows[0].time) * 2
    first = oracle.names.index("trajectory:row0:init_r_value")
    second = oracle.names.index("trajectory:row1:init_r_value")
    assert jacobian[:boundary, second] == pytest.approx(0)
    assert jacobian[boundary:, first] == pytest.approx(0)
    assert np.linalg.norm(jacobian[:, first]) > 0.1
    assert oracle.affine_audit["channel_order"] == ["y", "v"]
    assert oracle.affine_audit["generator_parameters"] == ["theta", "gain"]


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_explicit_method_delegates_to_original_equations(
    coupled_case, monkeypatch, method
):
    _, _, oracle, vector = coupled_case
    expected = oracle.evaluate(vector, monotonic() + 10, jacobian=False)[0]
    calls = []
    original = kernel.symbolic_rollout

    def observe(*args, **kwargs):
        calls.append((args[3].integration_method, kwargs["sensitivities"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "symbolic_rollout", observe)
    actual, derivative = oracle.evaluate(
        vector, monotonic() + 20, jacobian=False, method=method
    )
    assert derivative is None
    assert actual == pytest.approx(expected, abs=3e-8)
    assert calls == [(method, False), (method, False)]


def test_prefix_freezes_globals_and_propagates_only_root_derivatives(
    coupled_case, monkeypatch
):
    model, data, training_oracle, vector = coupled_case
    parameters = training_oracle.parameters(vector)
    fit = {
        "problem_sha256": content_sha256(model),
        "fit_success": True,
        "shared_parameters": {n: parameters[n] for n in training_oracle.shared},
        "training_scales": training_oracle.scales,
    }
    prefix = kernel.make_prefix(
        data.rows[0],
        kernel.PrefixPolicy(
            duration=0.8, future_inputs=("u",), future_auxiliaries=("w",)
        ),
    )
    oracle = affine.PrefixOracle(model, fit, prefix, kernel.NumericalPolicy())
    original = kernel.PrefixOracle(model, fit, prefix, kernel.NumericalPolicy())
    point = oracle.vector({"init_r_value": 1.2})
    before = deepcopy(fit)

    def forbidden(*args, **kwargs):
        pytest.fail("root-only inference must not compute shared Frechet derivatives")

    monkeypatch.setattr(affine, "expm_frechet", forbidden)
    residual, derivative = oracle.evaluate(point, monotonic() + 10)
    other, other_derivative = original.evaluate(point, monotonic() + 10)
    assert residual == pytest.approx(other, abs=2e-8)
    assert derivative == pytest.approx(other_derivative, abs=2e-8)
    assert derivative.shape == (len(prefix.time) * 2, 1)
    assert oracle.affine_audit["sensitivity_parameters"] == ["init_r_value"]
    assert fit == before
    assert oracle.frozen == fit["shared_parameters"]
    verified, _ = oracle.evaluate(
        point, monotonic() + 10, jacobian=False, method="Radau"
    )
    assert residual == pytest.approx(verified, abs=2e-8)
    poisoned = data.rows[0].model_dump(mode="json")
    for values in poisoned["targets"].values():
        values[-2:] = [99999, 99999]
    assert (
        kernel.make_prefix(PublicTrajectory.model_validate(poisoned), prefix.policy)
        == prefix
    )


def test_nonlinear_observations_and_initial_maps_keep_exact_chain_rule(coupled_case):
    model, data, _, _ = coupled_case
    payload = model.model_dump(mode="json")
    payload["base_candidate"]["observation_mappings"][1]["expression"] = "z*z+gain*r"
    payload["graph"]["rules"]["z"]["initial"]["expression"] = "phi*root*root"
    oracle = affine.TrainingOracle(
        kernel.PrefixFitProblem.model_validate(payload), data, kernel.NumericalPolicy()
    )
    vector = oracle.vector(oracle.start)
    residual, derivative = oracle.evaluate(vector, monotonic() + 10)
    other, other_derivative = oracle.evaluate(vector, monotonic() + 10, method="DOP853")
    assert residual == pytest.approx(other, abs=3e-8)
    assert derivative == pytest.approx(other_derivative, abs=3e-8)


@pytest.mark.parametrize("shared", [False, True])
def test_shared_or_trajectory_root_with_no_forcing(shared):
    model, data = problem(shared=shared), training()
    oracle = affine.TrainingOracle(model, data, kernel.NumericalPolicy())
    point = oracle.vector(oracle.start)
    result = oracle.evaluate(point, monotonic() + 10)
    original = oracle.evaluate(point, monotonic() + 10, method="DOP853")
    assert result[0] == pytest.approx(original[0], abs=3e-8)
    assert result[1] == pytest.approx(original[1], abs=3e-8)


def test_parameter_free_and_time_zero_prefix_have_empty_jacobians():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["parameters"] = []
    payload["base_candidate"]["states"] = payload["base_candidate"]["states"][:1]
    payload["base_candidate"]["initial_conditions"] = payload["base_candidate"][
        "initial_conditions"
    ][:1]
    payload["base_candidate"]["state_equations"] = [{"state": "x", "rhs": "-x"}]
    payload["parameter_guesses"] = {}
    payload["graph"]["rules"] = {}
    model = kernel.PrefixFitProblem.model_validate(payload)
    oracle = affine.TrainingOracle(model, training(), kernel.NumericalPolicy())
    residual, jacobian = oracle.evaluate(np.empty(0), monotonic() + 10)
    assert jacobian.shape == (len(residual), 0)
    fit = {
        "problem_sha256": content_sha256(model),
        "fit_success": True,
        "shared_parameters": {},
        "training_scales": oracle.scales,
    }
    prefix = kernel.make_prefix(training().rows[0], kernel.PrefixPolicy(duration=0))
    fixed = affine.PrefixOracle(model, fit, prefix, kernel.NumericalPolicy())
    residual, jacobian = fixed.evaluate(np.empty(0), monotonic() + 10)
    assert residual == pytest.approx([0])
    assert jacobian.shape == (1, 0)


@pytest.mark.parametrize("extra", ["+x*x", "+u*x", "+u*u", "+t*x", "+t"])
def test_nonlinear_or_time_varying_rhs_is_rejected(coupled_case, extra):
    model, data, _, _ = coupled_case
    payload = model.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][0]["rhs"] += extra
    with pytest.raises(ValueError, match="autonomous jointly affine"):
        affine.TrainingOracle(
            kernel.PrefixFitProblem.model_validate(payload),
            data,
            kernel.NumericalPolicy(),
        )


def test_irregular_forcing_corners_match_exact_trapezoidal_integral():
    payload = problem().model_dump(mode="json")
    payload["base_candidate"]["parameters"] = []
    payload["base_candidate"]["states"] = payload["base_candidate"]["states"][:1]
    payload["base_candidate"]["initial_conditions"] = payload["base_candidate"][
        "initial_conditions"
    ][:1]
    payload["parameter_guesses"] = {}
    payload["graph"]["rules"] = {}
    payload["base_candidate"]["state_equations"] = [{"state": "x", "rhs": "u"}]
    payload["context"]["external_inputs"] = ["u"]
    time = np.array([0.0, 0.1, 0.10000001, 0.10000003, 0.7, 2.0])
    forcing = np.array([0.0, 0.0, 10000.0, -10000.0, 0.0, 3.0])
    target = np.r_[
        1.0, 1.0 + np.cumsum(np.diff(time) * (forcing[:-1] + forcing[1:]) / 2)
    ]
    row = PublicTrajectory(
        trajectory_id="corners",
        time=tuple(time),
        targets={"y": tuple(target)},
        external_inputs={"u": tuple(forcing)},
    )
    data = PublicSplit(name="train", fingerprint="sharp-pulse", rows=(row,))
    oracle = affine.TrainingOracle(
        kernel.PrefixFitProblem.model_validate(payload), data, kernel.NumericalPolicy()
    )
    residual, _ = oracle.evaluate(np.empty(0), monotonic() + 10)
    assert residual == pytest.approx(0, abs=2e-14)


def test_transition_cache_reuses_dynamics_across_roots_but_not_calls(monkeypatch):
    oracle = affine.TrainingOracle(problem(), training(), kernel.NumericalPolicy())
    vector = oracle.vector(oracle.start)
    calls = []
    original = affine.expm

    def counted(matrix):
        calls.append(matrix.copy())
        return original(matrix)

    monkeypatch.setattr(affine, "expm", counted)
    first = oracle.evaluate(vector, monotonic() + 10, jacobian=False)[0]
    intervals = {float(dt) for row in oracle.rows for dt in np.diff(row.time)}
    assert len(calls) == len(intervals)
    changed = vector.copy()
    changed[oracle.names.index("theta")] += 0.2
    second = oracle.evaluate(changed, monotonic() + 10, jacobian=False)[0]
    assert len(calls) == 2 * len(intervals)
    assert not np.allclose(first, second)


def test_invalid_coordinates_and_deadline_refused_before_matrix_work(
    coupled_case, monkeypatch
):
    _, _, oracle, vector = coupled_case
    monkeypatch.setattr(affine, "expm", lambda *a, **k: pytest.fail("matrix work"))
    with pytest.raises(TimeoutError, match="wall-clock"):
        oracle.evaluate(vector, monotonic() - 1)
    with pytest.raises(ValueError, match="deadline"):
        oracle.evaluate(vector, np.nan)
    invalid = vector.copy()
    invalid[oracle.names.index("theta")] = -1
    with pytest.raises(ValueError, match="outside declared domain"):
        oracle.evaluate(invalid, monotonic() + 10)


@pytest.mark.parametrize("operation", ["expm", "expm_frechet"])
def test_blocking_matrix_overrun_is_rejected(coupled_case, monkeypatch, operation):
    _, _, oracle, vector = coupled_case
    clock = [100.0]
    original = getattr(affine, operation)
    monkeypatch.setattr(affine, "monotonic", lambda: clock[0])

    def overrun(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 102.0
        return value

    monkeypatch.setattr(affine, operation, overrun)
    with pytest.raises(TimeoutError, match="wall-clock"):
        oracle.evaluate(vector, 101.0)


def test_nonfinite_transition_cannot_produce_available_predictions(
    coupled_case, monkeypatch
):
    _, _, oracle, vector = coupled_case
    monkeypatch.setattr(affine, "expm", lambda a: np.full_like(a, np.nan))
    with pytest.raises(ValueError, match="nonfinite scoped affine transition"):
        oracle.evaluate(vector, monotonic() + 10)
