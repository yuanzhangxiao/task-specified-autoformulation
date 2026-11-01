"""Conditional timescale charts preserve original equations and profile nuisances."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from tests.test_prefix_gain_reduction import oracle_for
from tests.test_prefix_structural_reduction import training


def coupled(*, rhs="-x-a*x+g*q", z_rhs="c*x-d*z", bounded=False):
    source = oracle_for(
        law="b*z",
        rhs=rhs,
        observation="x+z",
        amplitude={"bounds": {"lower": 0, "upper": 2}} if bounded else None,
        extra_parameters=[
            {"name": n, "scope": "global", "role": "coefficient"}
            for n in ("b", "c", "d")
        ],
        extra_processes=[{"name": "temporary_declarations", "expression": "a+c+d"}],
    )
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["state_equations"][1]["rhs"] = z_rhs
    problem = PrefixFitProblem.model_validate(payload)
    return kernel.TrainingOracle(problem, training(source), kernel.NumericalPolicy())


def wrapped(source=None, retained=None):
    source = source or coupled()
    return timescale.make_oracle(
        source.problem,
        training(source),
        kernel.NumericalPolicy(),
        retained or source.start,
        states=("x", "z"),
    )


def test_affine_offset_gain_chart_and_original_domain_lift():
    source = coupled()
    original, oracle = wrapped(source)
    assert oracle.start["timescale:x"] == 0.5
    assert oracle.start["timescale:z"] == 1.0
    assert oracle.start["equilibrium_gain:x:z"] == 0.5
    for tau in (1e-11, 1e-3, 0.1, 10):
        point = oracle.vector(oracle.start | {"timescale:x": tau})
        lift = oracle.reduction.lift(point)
        parameters = original.parameters(lift)
        assert 1 + parameters["a"] == pytest.approx(1 / tau)
        assert parameters["g"] * parameters["b"] == pytest.approx(0.5 / tau)
        assert oracle.reduction.reduce(lift) == pytest.approx(point)
        assert original.vector(parameters) == pytest.approx(lift)
    receipt = oracle.reduction.receipt
    assert receipt["coupled_system_stability_certified"] is False
    assert "unassessed" in receipt["zero_and_unstable_branches"]
    assert oracle.lower[oracle.names.index("timescale:x")] == 0


def test_exact_rhs_and_jacobian_chain_rule_at_two_points():
    original, oracle = wrapped()
    for tau in (0.5, 0.1):
        point = oracle.vector(oracle.start | {"timescale:x": tau})
        residual, derivative = oracle.evaluate(point, monotonic() + 10)
        expected, original_derivative = original.evaluate(
            oracle.reduction.lift(point), monotonic() + 10
        )
        assert residual == pytest.approx(expected, abs=1e-10)
        assert derivative == pytest.approx(
            original_derivative @ oracle.reduction.lift_jacobian(point), abs=1e-9
        )
        for index in range(len(point)):
            step = np.eye(len(point))[index] * 1e-6
            plus = oracle.evaluate(point + step, monotonic() + 10, jacobian=False)[0]
            minus = oracle.evaluate(point - step, monotonic() + 10, jacobian=False)[0]
            assert derivative[:, index] == pytest.approx(
                (plus - minus) / 2e-6, abs=1e-7
            )


def test_independent_equivalence_and_original_endpoint_verification():
    original, oracle = wrapped()
    saved = []
    result = timescale.chart_equivalence(original, oracle, 10, saved.append)
    assert result["passed"] and result["original_equations_verified"]
    assert not result["jacobian_qualification_claimed"]
    assert result["calls_started"] == result["calls_completed"] == 3
    result = portfolio.verify_endpoint(original, oracle, oracle.start, 10, saved.append)
    assert result["original_equations_verified"]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_zero_negative_and_nonfinite_timescales_fail_closed(value):
    _, oracle = wrapped()
    with pytest.raises(ValueError):
        oracle.vector(oracle.start | {"timescale:x": value})


@pytest.mark.parametrize("a", [-1, -2])
def test_zero_or_unstable_original_self_coefficient_is_not_reclassified(a):
    source = coupled()
    with pytest.raises(ValueError, match="conditional stable"):
        wrapped(source, source.start | {"a": a})
    # This diagnostic rejection does not alter the original model's domain.
    source.vector(source.start | {"a": a})


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bounded": True}, "unrestricted coefficient domains"),
        ({"z_rhs": "a*x-d*z"}, "overlapping"),
        ({"rhs": "-x-a*x+g*q*x"}, "affine"),
        ({"rhs": "-x+g*q"}, "self-relaxation"),
    ],
)
def test_unsupported_algebra_and_domains_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wrapped(coupled(**kwargs))


def test_grid_is_deterministic_and_uses_native_time_without_lower_floor():
    _, oracle = wrapped()
    grid = timescale.profile_grid(oracle)
    assert grid == timescale.profile_grid(oracle)
    assert len(grid) == 10
    assert [row["tau"] for row in grid[:5]] == [0.5, 0.001, 0.1, 1.0, 10.0]
    assert grid[0]["role"] == "anchor"


def test_profile_optimizes_all_nuisances_without_jacobians_and_retains_control(
    monkeypatch,
):
    _, oracle = wrapped()
    parameters = deepcopy(oracle.start)
    spec = timescale.profile_grid(oracle)[1]
    free_name = "equilibrium_gain:x:z"
    target = parameters[free_name] + 0.25
    evaluations = []

    def objective(point, deadline, *, jacobian=True, method=None):
        assert not jacobian
        values = oracle.parameters(point)
        assert values[spec["coordinate"]] == spec["tau"]
        evaluations.append(values)
        return np.asarray([values[free_name] - target]), None

    monkeypatch.setattr(oracle, "evaluate", objective)
    saved = []
    result = timescale.profile_search(oracle, spec, 10, 120, saved.append)
    assert result["nuisance_optimization_performed"]
    assert not result["derivatives_used"]
    assert result["unoptimized_control"]["training_nmse"] == pytest.approx(0.25**2)
    assert result["best"]["training_nmse"] < 1e-12
    assert result["best"]["parameters"][spec["coordinate"]] == spec["tau"]
    assert set(result["best"]["parameters"]) == set(oracle.names)
    assert any(n.startswith("trajectory:") for n in result["nuisance_coordinates"])
    assert result["calls_started"] == result["calls_completed"] == len(evaluations)
    assert not result["global_minimum_certified"]
    assert oracle.start == parameters
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )


def test_profile_timeout_preserves_valid_incumbent_and_failed_call_cost(monkeypatch):
    _, oracle = wrapped()
    calls = []

    def objective(point, deadline, *, jacobian=True, method=None):
        calls.append(point.copy())
        if len(calls) > 1:
            raise TimeoutError("bounded synthetic timeout")
        return np.asarray([2.0]), None

    monkeypatch.setattr(oracle, "evaluate", objective)
    result = timescale.profile_search(
        oracle, timescale.profile_grid(oracle)[0], 10, 10, lambda _: None
    )
    assert result["best"]["training_nmse"] == 4
    assert result["budget_exhausted"]
    assert result["calls_started"] == 2
    assert result["calls_completed"] == result["calls_failed"] == 1


def test_profile_rejects_unfrozen_grid_budget_and_bad_initial(monkeypatch):
    _, oracle = wrapped()
    grid = timescale.profile_grid(oracle)
    for seconds, calls in ((121, 10), (10, 121), (0, 1), (1, 0)):
        with pytest.raises(ValueError, match="120"):
            timescale.profile_search(oracle, grid[0], seconds, calls, lambda _: None)
    with pytest.raises(ValueError, match="predetermined"):
        timescale.profile_search(oracle, grid[0] | {"tau": 42}, 10, 10, lambda _: None)
    monkeypatch.setattr(
        oracle,
        "evaluate",
        lambda *a, **k: (_ for _ in ()).throw(
            NumericalInvalidTrial("invalid original lift")
        ),
    )
    result = timescale.profile_search(oracle, grid[0], 10, 10, lambda _: None)
    assert result["best"] is None
    assert result["calls_failed"] == 1
    assert not result["nuisance_optimization_performed"]


def test_additive_product_chart_composition_preserves_gain_and_derivative():
    source = coupled()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["parameters"].append(
        {"name": "h", "scope": "global", "role": "coefficient"}
    )
    payload["base_candidate"]["state_equations"][0]["rhs"] += "+h*z"
    payload["parameter_guesses"]["h"] = -0.25
    source = kernel.TrainingOracle(
        PrefixFitProblem.model_validate(payload),
        training(source),
        kernel.NumericalPolicy(),
    )
    original, oracle = wrapped(source, source.start | {"b": 2.0, "g": 1.5})
    assert oracle.reduction.base.groups
    assert oracle.start["equilibrium_gain:x:z"] == pytest.approx(1.375)
    point = oracle.vector(oracle.start)
    residual, derivative = oracle.evaluate(point, monotonic() + 10)
    expected, full = original.evaluate(oracle.reduction.lift(point), monotonic() + 10)
    assert residual == pytest.approx(expected, abs=1e-10)
    assert derivative == pytest.approx(
        full @ oracle.reduction.lift_jacobian(point), abs=1e-9
    )


def test_residual_lift_does_not_compute_unused_overflowing_chart_derivative():
    _, oracle = wrapped()
    point = oracle.vector(oracle.start | {"timescale:x": 1e-160})
    assert np.isfinite(oracle.reduction.lift(point)).all()
    with pytest.raises(NumericalInvalidTrial, match="nonrepresentable"):
        oracle.reduction.lift_jacobian(point)
    with pytest.raises(NumericalInvalidTrial, match="nonrepresentable"):
        oracle.reduction.lift(oracle.vector(oracle.start | {"timescale:x": 1e-320}))


def test_checkpoint_failure_stops_before_new_oracle_call(monkeypatch):
    _, oracle = wrapped()
    calls = []
    monkeypatch.setattr(oracle, "evaluate", lambda *a, **k: calls.append(a))

    def fail(_):
        raise OSError("checkpoint unavailable")

    result = timescale.profile_search(
        oracle, timescale.profile_grid(oracle)[0], 10, 10, fail
    )
    assert not calls
    assert result["status"] == "incomplete"
    assert result["calls_attempted"] == 1
    assert (
        result["calls_started"]
        == result["calls_completed"]
        == result["calls_failed"]
        == 0
    )
    assert result["timing"]["callback_failures"] == 1
