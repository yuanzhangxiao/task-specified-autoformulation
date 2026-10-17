"""Nonlinear profile numerical evidence, failures, support limits and recovery."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import measurement_fit
from autoformalism.fitting import nonlinear_practical_fixtures as fixtures
from autoformalism.fitting import nonlinear_practical_profile as profile
from autoformalism.schemas.sensor_drift import DriftProblem

SMALL = profile.Policy(
    grid_points=3, reference_points=11, nuisance_starts=2, local_sampling=False
)
POINT = {"g": 1.3, "k": 0.65, "x0": 0.4, "sensor_drift": 0.06}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Use an actual independently verified retained measurement-backend fit."""
    problem = fixtures.make(0.5, 0.01, 0)[0]
    fit = measurement_fit.run(
        problem,
        measurement_fit.Policy(
            search_seconds=5, verification_seconds=5, maximum_calls=50, restart_count=0
        ),
        tmp_path_factory.mktemp("nonlinear-profile-fit"),
        strategy="direct",
    )
    assert fit["status"] == "complete"
    return problem, fit["parameters"]


def exact_problem(separation=0.5, parameters=None):
    """Construct noiseless unit-test readings without using evaluator metadata."""
    parameters = POINT if parameters is None else parameters
    problem = fixtures.make(separation, 0.2, 0)[0]
    arrays = profile._public_arrays(problem)
    prediction = profile._residual(arrays, parameters) * arrays[3] + arrays[2]
    payload, index = problem.model_dump(mode="json"), 0
    for row in payload["base"]["training"]["rows"]:
        for reading in row["readings"]:
            if reading["channel"] == "y":
                reading["raw_value"] = float((prediction[index] - 0.03) / 0.01)
                index += 1
    return DriftProblem.model_validate(payload)


def test_real_backend_endpoint_matches_reference_and_frozen_slice(fitted):
    problem, parameters = fitted
    result = profile.run(problem, parameters, policy=SMALL)
    assert result["status"] == "complete", result["reason"]
    assert result["independent_rollout"]["passed"]
    assert result["independent_rollout"]["maximum_standardized_difference"] < 1e-6
    qualification = result["endpoint_qualification"]
    assert qualification["numerically_qualified"]
    assert not qualification["retained_fit_inadequate"]
    assert not result["global_optimum_proved"]
    assert result["confidence_probability"] is None
    assert result["calls"] == result["completed_calls"]
    assert result["nodes_completed"] == len(result["grid"]) == 4
    assert all(
        node["profiled_objective"] <= node["frozen_objective"] + 1e-5
        for node in result["nodes"]
    )
    assert all(
        abs(node["objective_gap"]) <= node["objective_tolerance"]
        for node in result["nodes"]
    )
    for support in result["supports"]:
        assert support["profiled"]["width"] is None
        assert support["profiled"]["unresolved_between_nodes"]
        assert support["profiled"]["boundary_brackets"]
    assert result["curvature"]["joint_rank"] == 4
    assert (
        result["curvature"]["conditional_information"]
        < result["curvature"]["frozen_information"]
    )
    assert "share exact exponential" in result["reference_independence"]


def test_local_sampling_resolves_samples_without_inventing_continuous_width(fitted):
    problem, parameters = fitted
    result = profile.run(
        problem, parameters, policy=SMALL.model_copy(update={"local_sampling": True})
    )
    assert result["status"] == "complete"
    assert len(result["grid"]) == 12
    assert len(result["supports"][1]["profiled"]["sampled_values"]) > 1
    assert result["supports"][1]["profiled"]["width"] is None


def test_weak_input_still_has_local_full_rank_but_more_compensation():
    information = []
    for separation in (0.5, 0.02, 0.0):
        arrays = profile._public_arrays(exact_problem(separation))
        curvature = profile._curvature(arrays, POINT, SMALL)
        assert curvature["joint_rank"] == 4
        information.append(curvature["conditional_information"] * 0.2**2)
    assert information == pytest.approx([1.45969548, 0.63300867, 0.60818218], rel=1e-7)


def test_equilibrium_control_has_actual_gain_rate_ambiguity():
    first = POINT | {"g": 0.26}
    second = first | {"g": 0.4, "k": 1.0}
    arrays = profile._public_arrays(exact_problem(0.0, first))
    assert profile._residual(arrays, first) == pytest.approx(
        profile._residual(arrays, second), abs=1e-13
    )
    curvature = profile._curvature(arrays, first, SMALL)
    assert curvature["joint_rank"] == 3
    assert curvature["conditional_information"] == 0


def test_bound_endpoint_is_reported_as_domain_clipped():
    parameters = POINT | {"g": 3.0}
    result = profile.run(exact_problem(parameters=parameters), parameters, policy=SMALL)
    assert result["status"] == "complete", result["reason"]
    assert result["endpoint_qualification"]["numerically_qualified"]
    for support in result["supports"]:
        assert support["profiled"]["domain_clipped"]
        assert support["profiled"]["upper"] == 3.0
        assert support["profiled"]["width"] is None


def test_better_reference_marks_retained_fit_inadequate_without_replacing_it(fitted):
    problem, _ = fitted
    parameters = dict(problem.start)
    result = profile.run(problem, parameters, policy=SMALL)
    assert result["status"] == "unqualified"
    assert result["endpoint_qualification"]["retained_fit_inadequate"]
    assert result["endpoint_qualification"]["objective_gap"] > 100
    assert result["identity"]["parameters"] == parameters
    assert result["supports"] == []


def test_failed_multistart_is_unresolved_not_weak_information(fitted, monkeypatch):
    problem, parameters = fitted
    monkeypatch.setattr(
        profile,
        "least_squares",
        lambda *a, **kw: SimpleNamespace(success=False, nfev=1, status=0),
    )
    result = profile.run(problem, parameters, policy=SMALL)
    assert result["status"] == "partial"
    assert result["nodes_completed"] == 0
    assert result["unresolved_nodes"] == len(result["grid"])
    assert result["supports"] == []
    assert all("failed to converge" in node["reason"] for node in result["nodes"])


def test_reference_failure_blocks_support(fitted, monkeypatch):
    problem, parameters = fitted
    monkeypatch.setattr(
        profile,
        "minimize_scalar",
        lambda *a, **kw: SimpleNamespace(success=False, nfev=1),
    )
    result = profile.run(problem, parameters, policy=SMALL)
    assert result["status"] == "partial"
    assert not result["endpoint_qualification"]["numerically_qualified"]
    assert result["endpoint_qualification"]["reference_search"] is None
    assert result["supports"] == []
    assert "reference" in result["reason"]


def test_unresolved_node_can_disprove_retained_fit_optimality(fitted, monkeypatch):
    problem, _ = fitted
    parameters = dict(problem.start)
    arrays = profile._public_arrays(problem)
    retained = profile._residual(arrays, parameters)
    original = profile._dense_reference

    def missed_global(arrays, gain, policy, work):
        if gain is None:
            return {
                "parameters": parameters,
                "objective": float(retained @ retained / 2),
            }
        return original(arrays, gain, policy, work)

    def frozen_primary(arrays, gain, parameters, policy, work):
        point = parameters | {"g": gain}
        residual = work.call(lambda: profile._residual(arrays, point))
        return {"parameters": point, "objective": float(residual @ residual / 2)}

    monkeypatch.setattr(profile, "_dense_reference", missed_global)
    monkeypatch.setattr(profile, "_multistart", frozen_primary)
    result = profile.run(problem, parameters, policy=SMALL)
    assert result["unresolved_nodes"] > 0
    assert result["endpoint_qualification"]["retained_fit_inadequate"]
    assert result["endpoint_qualification"]["objective_gap"] > 100
    assert result["supports"] == []


def test_reference_refines_all_sampled_basins_and_keeps_endpoints(monkeypatch):
    def two_basins(arrays, k, gain):
        residual = np.asarray([(k - 0.2) * (k - 1.2)])
        return POINT | {"k": k}, residual, {}

    monkeypatch.setattr(profile, "_project", two_basins)
    state = profile._initial_state({}, [])
    work = profile._Execution(state, SMALL, None)
    reference = profile._dense_reference(None, 1.3, SMALL, work)
    assert reference["sampled_local_basins"] == 2
    assert len(reference["refinements"]) == 2
    assert reference["endpoints_checked"]
    assert reference["objective"] < 1e-15
    assert not reference["global_optimum_proved"]


def test_call_budget_is_total_and_cannot_reset_on_resume(fitted):
    problem, parameters = fitted
    policy = SMALL.model_copy(update={"maximum_calls": 4})
    result = profile.run(problem, parameters, policy=policy)
    assert result["status"] == "partial"
    assert result["calls"] == 4
    assert result["supports"] == []
    resumed = profile.run(problem, parameters, policy=policy, state=result)
    assert resumed["calls"] == 4
    assert resumed["operations"] == result["operations"]
    assert resumed["budget_charge_seconds"] >= result["budget_charge_seconds"]


def test_finished_resume_is_exact_and_checks_content_and_identity(fitted):
    problem, parameters = fitted
    result = profile.run(problem, parameters, policy=SMALL)
    assert profile.run(problem, parameters, policy=SMALL, state=result) == result
    altered = deepcopy(result)
    altered["nodes"][0]["profiled_objective"] -= 1
    with pytest.raises(ValueError, match="identity or content"):
        profile.run(problem, parameters, policy=SMALL, state=altered)
    with pytest.raises(ValueError, match="identity or content"):
        profile.run(
            problem,
            parameters | {"g": parameters["g"] + 0.01},
            policy=SMALL,
            state=result,
        )
    with pytest.raises(ValueError, match="identity or content"):
        profile.run(
            problem,
            parameters,
            policy=SMALL.model_copy(update={"maximum_calls": 20000}),
            state=result,
        )


def test_completed_node_snapshot_resumes_deterministically(fitted):
    problem, parameters = fitted
    ledger = []

    def stop(state):
        ledger.append(state)
        if state["event"] == "operation_completed" and state["operations"][-1][
            "key"
        ].startswith("node:"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        profile.run(problem, parameters, policy=SMALL, save=stop)
    snapshot = ledger[-1]
    resumed = profile.run(problem, parameters, policy=SMALL, state=snapshot)
    ordinary = profile.run(problem, parameters, policy=SMALL)
    assert resumed["status"] == "complete"
    assert resumed["calls"] == ordinary["calls"]
    assert resumed["nodes"] == ordinary["nodes"]
    assert resumed["supports"] == ordinary["supports"]
    assert not resumed["accounting_complete"]  # Lost callback duration stays explicit.
    assert resumed["budget_charge_seconds"] >= SMALL.callback_seconds


def test_started_node_interruption_is_charged_and_never_retried(fitted):
    problem, parameters = fitted
    ledger = []

    def stop(state):
        ledger.append(state)
        if state["pending"] and state["pending"]["key"].startswith("node:"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        profile.run(problem, parameters, policy=SMALL, save=stop)
    saved = ledger[-1]
    resumed = profile.run(problem, parameters, policy=SMALL, state=saved)
    assert resumed["status"] == "partial"
    assert resumed["interrupted_calls_charged"] == SMALL.operation_calls
    assert resumed["interrupted_operations"] == 1
    assert not resumed["accounting_complete"]
    assert resumed["budget_charge_seconds"] >= SMALL.operation_seconds
    assert resumed["unresolved_nodes"] == 1
    assert resumed["supports"] == []
    assert (
        len(
            [op for op in resumed["operations"] if op["key"] == saved["pending"]["key"]]
        )
        == 1
    )


def test_callback_failure_stops_work_and_records_its_cost(fitted):
    problem, parameters = fitted

    def fail(_):
        raise OSError("durable store unavailable")

    result = profile.run(problem, parameters, policy=SMALL, save=fail)
    assert result["status"] == "partial"
    assert "save callback failed" in result["reason"]
    assert result["calls"] == 0
    assert result["timing"]["save_callback_failures"] == 1
    assert result["timing"]["save_callback_calls"] == 1
    assert (
        result["timing"]["total_wall_seconds"]
        >= result["timing"]["save_callback_wall_seconds"]
    )


def test_final_callback_wall_time_is_included_and_exhausts_budget(fitted, monkeypatch):
    problem, parameters = fitted
    original = profile.monotonic
    offset = 0

    def save(state):
        nonlocal offset
        if state["event"] == "diagnostic_finished":
            offset += 100

    monkeypatch.setattr(profile, "monotonic", lambda: original() + offset)
    result = profile.run(problem, parameters, policy=SMALL, save=save)
    assert result["status"] == "partial"
    assert result["timing"]["save_callback_wall_seconds"] >= 100
    assert result["timing"]["total_wall_seconds"] >= 100
    assert result["budget_charge_seconds"] >= 100
    assert result["supports"] == []


def test_public_structure_mismatch_and_untrusted_fields_fail_closed(fitted):
    problem, parameters = fitted
    payload = problem.model_dump(mode="json")
    payload["base"]["request"]["base_candidate"]["state_equations"][0]["rhs"] = "g*u"
    other = DriftProblem.model_validate(payload)
    with pytest.raises(ValueError, match="structure or domains"):
        profile.run(other, parameters, policy=SMALL)
    payload = problem.model_dump(mode="json")
    payload["truth"] = POINT
    with pytest.raises(ValidationError):
        DriftProblem.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_calls": 0},
        {"grid_points": True},
        {"reference_points": 2},
        {"nuisance_starts": 1},
        {"wall_seconds": float("nan")},
        {"thresholds": [0.5, 3.0]},
        {"unexpected": 4},
    ],
)
def test_policy_is_frozen_and_bounded(change):
    with pytest.raises(ValidationError):
        profile.Policy(**change)
    with pytest.raises(ValidationError):
        SMALL.maximum_calls = 100
