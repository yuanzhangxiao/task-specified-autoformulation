"""Joint sensor/initial nuisance projection never implies statistical confidence."""

from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import drift_information as info
from autoformalism.fitting import measurement_fixtures
from autoformalism.fitting.robust_measurement import Loss
from autoformalism.schemas.sensor_drift import DriftProblem, SensorDrift


def drift_problem(case="exact_varied"):
    """Append one declared sensor alternative without changing observations."""
    return DriftProblem(
        base=measurement_fixtures.make(case, 0)[0],
        drift=SensorDrift(
            channel="y",
            lower=-0.12,
            upper=0.12,
            start=0.0,
            incumbent=0.0,
            scale=0.1,
            unit="m/min",
            sources=("calibration",),
            rationale="Shared instrument trend is a candidate measurement explanation.",
        ),
    )


def test_exact_drift_gain_confounding_retains_only_sum_information():
    time = np.linspace(0, 5, 21)
    j = np.column_stack([time, time])
    result = info.decompose(j, [1.0, 0.1], ["gain", "drift"], ["drift"])
    assert result["joint"]["rank"] == 1
    assert result["physical_before_nuisance"]["rank"] == 1
    assert result["physical_after_nuisance"]["rank"] == 0
    direction = result["joint"]["null_directions"][0]
    physical = np.array([direction["gain"], 0.1 * direction["drift"]])
    assert np.linalg.norm(j @ physical) < 1e-12
    assert result["physical_conditional_gram"][0][0] < 1e-20


def test_varied_inputs_separate_gain_and_drift():
    time = np.tile(np.linspace(0, 5, 21), 2)
    forcing = np.repeat([0.4, 1.1], 21)
    j = np.column_stack([time * forcing, time])
    result = info.decompose(j, [1.0, 0.1], ["gain", "drift"], ["drift"])
    assert result["joint"]["rank"] == 2
    assert result["physical_after_nuisance"]["rank"] == 1
    assert result["physical_conditional_gram"][0][0] > 1


def test_initial_and_sensor_directions_project_jointly_without_normal_equations():
    time = np.linspace(0, 5, 25)
    j = np.column_stack([time**2 + 2 * time + 3, np.ones(25), time])
    result = info.decompose(
        j, [1, 1, 0.1], ["physical", "initial", "drift"], ["initial", "drift"]
    )
    nuisance = np.column_stack([np.ones(25), 0.1 * time])
    projected = j[:, 0] - nuisance @ np.linalg.lstsq(nuisance, j[:, 0], rcond=None)[0]
    assert result["physical_conditional_gram"][0][0] == pytest.approx(
        projected @ projected
    )
    assert result["joint"]["rank"] == 3


def test_no_nuisance_matches_full_information():
    j = np.array([[1, 2], [3, 2], [2, 8]], float)
    result = info.decompose(j, [0.2, 4], ["a", "b"], [])
    assert result["nuisance"]["rank"] == 0
    assert result["physical_before_nuisance"] == result["physical_after_nuisance"]


@pytest.mark.parametrize(
    "units,names,nuisance",
    [
        ([0], ["a"], []),
        ([float("nan")], ["a"], []),
        ([1], ["a"], ["b"]),
        ([1, 1], ["a", "a"], []),
        ([1], ["a"], ["a", "a"]),
    ],
)
def test_invalid_coordinate_contract(units, names, nuisance):
    with pytest.raises(ValueError):
        info.decompose(np.ones((3, len(names))), units, names, nuisance)


def test_nonfinite_sensitivity_is_not_rank_evidence():
    with pytest.raises(ValueError):
        info.decompose([[float("nan")]], [1], ["a"], [])


@pytest.mark.parametrize("jacobian", [np.ones((4, 1)), np.ones((0, 2)), np.ones(2)])
def test_malformed_jacobian_cannot_broadcast_into_a_valid_shape(jacobian):
    with pytest.raises(ValueError, match="dimensions"):
        info.decompose(jacobian, [1, 1], ["physical", "drift"], ["drift"])


def test_large_scale_projection_roundoff_cannot_create_information():
    t = np.linspace(0, 5, 21) * 1e9
    result = info.decompose(
        np.column_stack([t, t]), [1, 0.1], ["physical", "drift"], ["drift"]
    )
    assert result["joint"]["rank"] == 1
    assert result["physical_after_nuisance"]["rank"] == 0
    assert result["physical_after_nuisance"]["weak_under_declared_convention"]
    assert result["physical_after_nuisance"]["rank_reference_scale"] == pytest.approx(
        np.linalg.norm(t)
    )


def test_near_collinearity_is_not_certified_uniqueness():
    t = np.linspace(0, 5, 50)
    result = info.decompose(
        np.column_stack([t, t + 1e-7 * t**2]), [1, 1], ["a", "b"], ["b"]
    )
    assert result["joint"]["weak_under_declared_convention"]


@pytest.mark.parametrize("case", ["exact_varied", "shared_initial", "nonlinear_decay"])
@pytest.mark.parametrize("kind", ["gaussian", "huber"])
def test_actual_rollout_information_includes_all_declared_nuisances(case, kind):
    problem = drift_problem(case)
    ledger = []
    result = info.evaluate(
        problem,
        problem.start,
        loss=Loss(kind=kind),
        deadline=monotonic() + 20,
        save=ledger.append,
    )
    assert result["derivatives_passed"]
    assert result["information_status"] == "locally_full_rank"
    assert result["information"]["joint"]["rank"] == len(problem.start)
    assert result["sensor_nuisance_names"] == ["sensor_drift"]
    assert result["initial_nuisance_names"] == (
        ["x0"] if case == "shared_initial" else []
    )
    assert result["calls"] == 1 + 4 * len(problem.start)
    assert result["completed_calls"] == result["calls"]
    assert ledger[-1] == {"calls": result["calls"], "completed_calls": result["calls"]}
    assert result["confidence_probability"] is None
    assert result["global_identifiability_certified"] is False
    if kind == "huber":
        assert "not Huber likelihood" in result["information_interpretation"]


def test_no_drift_hypothesis_is_not_silently_expanded():
    problem = measurement_fixtures.make("shared_initial", 0)[0]
    result = info.evaluate(
        problem, problem.start, loss=Loss(kind="gaussian"), deadline=monotonic() + 20
    )
    assert result["sensor_nuisance_names"] == []
    assert result["information"]["joint"]["names"] == ["g", "x0"]


def test_active_bound_uses_one_sided_checks_and_withholds_interior_conclusions():
    problem = drift_problem()
    result = info.evaluate(
        problem,
        dict(problem.start) | {"sensor_drift": 0.12},
        loss=Loss(kind="gaussian"),
        deadline=monotonic() + 20,
    )
    assert result["derivatives_passed"]
    assert result["information_status"] == "boundary_limited"
    assert result["active_bounds"] == ["sensor_drift"]
    assert result["derivative_columns"][-1]["schemes"] == ["backward_second_order"] * 2


def test_expired_diagnostic_cannot_begin_a_rollout():
    problem, ledger = drift_problem(), []
    with pytest.raises(TimeoutError):
        info.evaluate(
            problem,
            problem.start,
            loss=Loss(kind="gaussian"),
            deadline=monotonic() - 1,
            save=ledger.append,
        )
    assert ledger[-1] == {"calls": 0, "completed_calls": 0}


def test_failed_derivatives_withhold_information(monkeypatch):
    problem = measurement_fixtures.make("exact_varied", 0)[0]
    original = info.measurement_rollout.MeasurementRollouts.evaluate

    def wrong(self, *args, **kwargs):
        residual, derivative = original(self, *args, **kwargs)
        return residual, derivative * 2 if derivative is not None else None

    monkeypatch.setattr(info.measurement_rollout.MeasurementRollouts, "evaluate", wrong)
    result = info.evaluate(
        problem, problem.start, loss=Loss(kind="gaussian"), deadline=monotonic() + 20
    )
    assert not result["derivatives_passed"]
    assert result["information"] is None
    assert result["information_status"] == "derivatives_unqualified"
