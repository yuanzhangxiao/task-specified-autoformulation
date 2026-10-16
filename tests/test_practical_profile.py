"""Exact affine objective support, confounding, bounds and incomplete work."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import lsq_linear

from autoformalism.fitting import measurement_fit, measurement_fixtures
from autoformalism.fitting import practical_profile as profile
from autoformalism.fitting import practical_uncertainty_fixtures as fixtures
from autoformalism.fitting import robust_measurement as robust
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from autoformalism.schemas.sensor_drift import DriftProblem


def problem(separation=0.1, sigma=0.05, *, drift_bounds=None):
    """Use only the public acquisition returned by the synthetic generator."""
    value = fixtures._build(separation, sigma, 0)[0]
    if drift_bounds:
        payload = value.model_dump(mode="json")
        payload["drift"].update(lower=drift_bounds[0], upper=drift_bounds[1])
        value = DriftProblem.model_validate(payload)
    return value


def endpoint(value, parameters=None):
    """Construct a verified-record control using an independent bounded solve."""
    loss = robust.Loss(kind="gaussian")
    oracle = robust.build(value, loss)[0].base
    vector = oracle.vector(value.start)
    residual, design = oracle.evaluate(vector, monotonic() + 10)
    if parameters is None:
        solved = lsq_linear(
            design,
            -(residual - design @ vector),
            bounds=(oracle.lower, oracle.upper),
            method="bvls",
            tol=1e-14,
        )
        assert solved.success
        vector = np.clip(solved.x, oracle.lower, oracle.upper)
    else:
        vector = oracle.vector(parameters)
    parameters = oracle.parameters(vector)
    residual, _ = oracle.evaluate(vector, monotonic() + 10)
    objective = float(residual @ residual / 2)
    drift = isinstance(value, DriftProblem)
    definition = loss.model_dump(mode="json")
    return {
        "protocol": "sensor-drift-fit-result-1"
        if drift
        else "robust-measurement-fit-result-1",
        "identity": {
            "protocol": "sensor-drift-fitting-execution-1"
            if drift
            else "robust-measurement-fitting-execution-1",
            "problem_sha256": content_sha256(value.model_dump(mode="json")),
            "base_request_sha256": content_sha256(
                value.request.model_dump(mode="json")
            ),
            "objective_definition": definition,
        },
        "status": "complete",
        "parameters": parameters,
        "objective_definition": definition,
        "objective_half_sum_squares": objective,
        "objective_value": objective,
        "verification": {
            "parameters": parameters,
            "passed": True,
            "standardized_residuals": residual.tolist(),
            "primary_half_sum_squares": objective,
            "primary_objective_value": objective,
            "objective_definition": definition,
        },
    }


def evaluate(value, **kwargs):
    return profile.evaluate(value, endpoint(value), deadline=monotonic() + 20, **kwargs)


@pytest.mark.parametrize("separation", [0.5, 0.1, 0.02, 0.005])
def test_nuisance_projection_has_raw_unit_analytic_curvature(separation):
    result = evaluate(problem(separation))
    assert result["status"] == "complete", result["reason"]
    assert result["curvature"]["conditional_information"] == pytest.approx(
        187 * separation**2 / 0.05**2, rel=1e-10
    )
    assert result["curvature"]["frozen_information"] == pytest.approx(
        93.5 * (3 + 2 * separation**2) / 0.05**2
    )
    assert result["curvature"]["conditional_rank"] == 1
    assert result["curvature"]["joint_rank"] == 3
    assert result["confidence_probability"] is None
    assert result["endpoint_qualification"]["passed"]
    for support in result["supports"]:
        assert support["profiled"]["width"] >= support["frozen"]["width"]
        if support["profiled"]["classification"] == "interior_data_limited":
            assert support["profiled"]["width"] == pytest.approx(
                2
                * np.sqrt(
                    2
                    * support["threshold"]
                    / result["curvature"]["conditional_information"]
                ),
                rel=1e-7,
            )
    for node in result["nodes"]:
        assert node["profiled_objective"] <= node["frozen_objective"] + 1e-7
        assert (
            node["solver_audit"]["kkt_violation"]
            <= node["solver_audit"]["kkt_tolerance"]
        )


def test_complete_confounding_is_broad_domain_limited_not_false_narrow_support():
    result = evaluate(problem(0.0))
    assert result["status"] == "complete", result["reason"]
    assert result["curvature"]["conditional_information"] == 0
    assert result["curvature"]["conditional_rank"] == 0
    assert result["curvature"]["joint_rank"] == 2
    for support in result["supports"]:
        assert support["profiled"]["lower"] == 0.05
        assert support["profiled"]["upper"] == 3.0
        assert support["profiled"]["width"] == pytest.approx(2.95)
        assert support["profiled"]["classification"] == "gain_domain_clipped"
        assert support["profiled"]["domain_clipped"]
        assert support["frozen"]["width"] < 0.1


def test_exact_confounding_can_have_nuisance_domain_assisted_finite_support():
    result = evaluate(problem(0.0, drift_bounds=(-0.03, 0.03)))
    assert result["status"] == "complete", result["reason"]
    assert result["curvature"]["conditional_information"] == 0
    assert result["curvature"]["joint_rank"] == 2
    for support in result["supports"]:
        interval = support["profiled"]
        assert interval["classification"] == "nuisance_domain_assisted"
        assert not interval["domain_clipped"]
        assert interval["width"] < 0.2
        for end, side in ((interval["left"], "upper"), (interval["right"], "lower")):
            assert end["status"] == "threshold_crossed"
            assert end["nuisance_active_bounds"] == [
                {
                    "name": "sensor_drift",
                    "side": side,
                    "value": 0.03 if side == "upper" else -0.03,
                }
            ]


def test_known_sigma_scales_information_and_interior_support():
    small, large = evaluate(problem(0.5, 0.01)), evaluate(problem(0.5, 0.05))
    assert small["status"] == large["status"] == "complete"
    assert small["curvature"]["conditional_information"] / large["curvature"][
        "conditional_information"
    ] == pytest.approx(25)
    for first, second in zip(small["supports"], large["supports"], strict=True):
        assert second["profiled"]["width"] / first["profiled"][
            "width"
        ] == pytest.approx(5, rel=1e-6)


def test_parameter_scaling_cannot_change_raw_unit_support_or_information():
    original = problem()
    payload = original.model_dump(mode="json")
    payload["base"]["coordinates"]["parameters"]["g"]["scale"] = 0.003
    payload["base"]["coordinates"]["parameters"]["x0"]["scale"] = 150
    payload["drift"]["scale"] = 0.0002
    changed = DriftProblem.model_validate(payload)
    first, second = evaluate(original), evaluate(changed)
    assert first["curvature"] == second["curvature"]
    assert first["supports"] == second["supports"]


def test_boundary_retained_endpoint_keeps_one_sided_domain_clipping():
    value = problem(0.5)
    payload = value.model_dump(mode="json")
    gain = payload["base"]["request"]["base_candidate"]["parameters"][0]
    gain["bounds"]["upper"] = 0.8
    value = DriftProblem.model_validate(payload)
    result = evaluate(value)
    assert result["status"] == "complete", result["reason"]
    assert result["retained_parameters"]["g"] == 0.8
    for support in result["supports"]:
        assert support["profiled"]["right"]["status"] == "domain_limited"
        assert support["profiled"]["upper"] == 0.8
        assert support["profiled"]["left"]["status"] == "threshold_crossed"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda e: e["identity"].update(problem_sha256="0" * 64), "problem identity"),
        (
            lambda e: e["identity"].update(base_request_sha256="0" * 64),
            "candidate identity",
        ),
        (lambda e: e.update(status="retained_unverified"), "verified complete"),
        (lambda e: e["verification"].update(passed=False), "verification"),
        (lambda e: e["verification"].update(parameters={}), "verification"),
        (lambda e: e["verification"].update(standardized_residuals=[0]), "residuals"),
        (lambda e: e.update(objective_value=12345), "objective"),
        (
            lambda e: e["identity"]["objective_definition"].update(kind="huber"),
            "Gaussian",
        ),
        (lambda e: e.update(protocol="not-a-fit"), "protocol"),
    ],
)
def test_detached_or_unqualified_retained_endpoints_have_no_support(mutation, reason):
    value = problem()
    fitted = deepcopy(endpoint(value))
    mutation(fitted)
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "unqualified"
    assert reason in result["reason"]
    assert result["supports"] == []


def test_arbitrary_verified_local_endpoint_is_not_a_global_profile_reference():
    value = problem()
    fitted = endpoint(value, value.start)
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "unqualified"
    assert "global own-loss optimum" in result["reason"]
    assert not result["endpoint_qualification"]["passed"]
    assert not result["nodes"]


def test_nonaffine_family_is_explicitly_unqualified():
    value = measurement_fixtures.make("nonlinear_decay", 0)[0]
    fitted = endpoint(value, value.start)
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "unqualified"
    assert "globally affine" in result["reason"]
    assert result["rollout_calls"] == 0
    assert result["supports"] == []


def test_no_nuisance_and_legacy_gaussian_endpoint_remain_eligible():
    value = measurement_fixtures.make("exact_varied", 0)[0]
    fitted = endpoint(value)
    fitted["protocol"] = "measurement-fit-result-1"
    fitted["identity"]["protocol"] = "measurement-fitting-execution-1"
    for record in (fitted, fitted["identity"], fitted["verification"]):
        record.pop("objective_definition")
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "complete", result["reason"]
    assert result["nuisance_names"] == []
    assert (
        result["curvature"]["conditional_information"]
        == result["curvature"]["frozen_information"]
    )
    for support in result["supports"]:
        assert support["profiled"]["width"] == support["frozen"]["width"]


def test_call_budget_preserves_completed_nodes_without_extrapolated_support():
    value, ledger = problem(), []
    result = evaluate(value, policy=profile.Policy(maximum_calls=5), save=ledger.append)
    assert result["status"] == "partial"
    assert result["calls"] == result["completed_calls"] == 5
    assert result["solve_calls"] == 4
    assert result["rollout_calls"] == 1
    assert result["nodes_completed"] == 3
    assert result["supports"] == []
    assert result["nodes"][0]["value"] == result["retained_parameters"]["g"]
    assert [x["event"]["status"] for x in ledger].count("started") == 6
    assert ledger[-1]["event"]["status"] == "partial"


def test_expired_allowance_has_zero_started_work_and_no_support():
    value, ledger = problem(), []
    result = profile.evaluate(
        value, endpoint(value), deadline=monotonic() - 1, save=ledger.append
    )
    assert result["status"] == "partial"
    assert result["calls"] == result["completed_calls"] == 0
    assert result["supports"] == []
    assert ledger[-1]["event"]["status"] == "partial"


def test_failed_inner_solver_cannot_claim_optimum_and_keeps_known_work(monkeypatch):
    value, fitted, ledger = problem(), None, []
    fitted = endpoint(value)
    monkeypatch.setattr(
        profile,
        "lsq_linear",
        lambda *a, **k: SimpleNamespace(
            success=False, message="iteration limit", nit=200
        ),
    )
    result = profile.evaluate(
        value, fitted, deadline=monotonic() + 20, save=ledger.append
    )
    assert result["status"] == "unqualified"
    assert result["failed_calls"] == 1
    assert result["inner_iterations"] == 200
    assert result["inner_iterations_complete"]
    assert result["calls"] == result["completed_calls"] == 2
    assert any(event["event"]["status"] == "failed" for event in ledger)
    assert result["supports"] == []


def test_mid_grid_timeout_keeps_failure_and_never_invents_missing_nodes(monkeypatch):
    value, fitted = problem(), None
    fitted = endpoint(value)
    original, calls = profile._solve, 0

    def timeout(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise TimeoutError("synthetic interrupted solve")
        return original(*args, **kwargs)

    monkeypatch.setattr(profile, "_solve", timeout)
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "partial"
    assert result["calls"] == result["completed_calls"] == 5
    assert result["failed_calls"] == 1
    assert result["nodes_completed"] == 2
    assert result["supports"] == []


def test_false_global_optimum_contradicted_by_node_is_rejected(monkeypatch):
    value, fitted = problem(), None
    fitted = endpoint(value)
    original, calls = profile._solve, 0

    def inconsistent(*args, **kwargs):
        nonlocal calls
        calls += 1
        values, audit = original(*args, **kwargs)
        if calls > 1:
            audit["objective"] -= 100
        return values, audit

    monkeypatch.setattr(profile, "_solve", inconsistent)
    result = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert result["status"] == "partial"
    assert "contradicts" in result["reason"]
    assert result["supports"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"thresholds": [0]},
        {"thresholds": [2, 0.5]},
        {"thresholds": [0.5, 0.5]},
        {"thresholds": [float("nan")]},
        {"grid_points": True},
        {"maximum_calls": 0},
    ],
)
def test_policy_rejects_unfrozen_or_invalid_support_conventions(change):
    with pytest.raises(ValidationError):
        profile.Policy(**change)


def test_real_fit_endpoint_contract_and_deterministic_output(tmp_path):
    value = problem()
    fitted = measurement_fit.run(
        value,
        measurement_fit.Policy(
            search_seconds=5, verification_seconds=5, maximum_calls=20, restart_count=0
        ),
        tmp_path / "fit",
        strategy="direct",
        loss=robust.Loss(kind="gaussian"),
    )
    first = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    second = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    assert first["status"] == second["status"] == "complete"
    assert first["endpoint_sha256"] == content_sha256(fitted)
    assert first["nodes"] == second["nodes"]
    assert first["supports"] == second["supports"]
    assert first["rollout_calls"] == 1
    assert first["solve_calls"] == first["nodes_completed"] + 1


def test_revalidation_prevents_untyped_hidden_evaluator_fields():
    payload = problem().base.model_dump(mode="json")
    payload["truth"] = {"g": 1.3}
    with pytest.raises(ValidationError):
        MeasurementProblem.model_validate(payload)


@pytest.mark.parametrize(
    "separation,drift_bounds",
    [(0.5, None), (0.02, None), (0.0, None), (0.0, (-0.03, 0.03))],
)
def test_opt_in_exact_mode_caches_preserve_support_and_reduce_bounded_solves(
    separation,
    drift_bounds,
):
    value = problem(separation, drift_bounds=drift_bounds)
    fitted = endpoint(value)
    legacy = profile.evaluate(value, fitted, deadline=monotonic() + 20)
    ledger = []
    efficient = profile.evaluate(
        value,
        fitted,
        deadline=monotonic() + 20,
        execution=profile.ExecutionPolicy(),
        save=ledger.append,
    )
    assert efficient["status"] == legacy["status"] == "complete"
    assert efficient["supports"] == legacy["supports"]
    assert efficient["curvature"] == legacy["curvature"]
    assert efficient["endpoint_sha256"] == legacy["endpoint_sha256"]
    assert efficient["policy"] == legacy["policy"] == profile.POLICY
    assert efficient["protocol"] == legacy["protocol"] == profile.PROTOCOL
    assert efficient["solve_calls"] < legacy["solve_calls"]
    assert efficient["calls"] == efficient["completed_calls"]
    assert efficient["calls"] == sum(
        efficient[name] for name in ("solve_calls", "rollout_calls", "frozen_calls")
    )
    assert efficient["frozen_calls"] == efficient["frozen_nodes_completed"]
    assert efficient["solve_calls"] == efficient["nodes_completed"] + 1
    assert all(count > 0 for count in efficient["cache_reuses"].values())
    assert "execution" not in legacy and "timing" not in legacy
    assert "frozen_nodes" not in legacy
    assert all("profiled_objective" not in node for node in efficient["frozen_nodes"])
    assert all("solver_audit" in node for node in efficient["nodes"])
    starts = [row["event"] for row in ledger if row["event"]["status"] == "started"]
    assert len(starts) == efficient["calls"] + 1
    for frozen in efficient["frozen_nodes"]:
        assert frozen["mode"] == "frozen"
        for name in efficient["nuisance_names"]:
            assert frozen["parameters"][name] == fitted["parameters"][name]
    assert efficient["timing"]["save_callback_calls"] == len(ledger)
    if drift_bounds is not None:
        assert efficient["curvature"]["conditional_information"] == 0
        for support in efficient["supports"]:
            interval = support["profiled"]
            assert interval["classification"] == "nuisance_domain_assisted"
            assert not interval["domain_clipped"]
            assert interval["width"] < 0.2
            for side in ("left", "right"):
                assert interval[side]["status"] == "threshold_crossed"
                assert interval[side]["nuisance_active_bounds"]


def test_exact_target_cache_does_not_round_adjacent_values_or_confuse_curves(
    monkeypatch,
):
    value = problem()
    fitted = endpoint(value)
    original = profile.brentq
    requests = []

    def probing_root(function, left, right, **kwargs):
        point = left + (right - left) / 3
        adjacent = np.nextafter(point, right)
        requests.extend([point, adjacent])
        assert function(point) == function(point)
        function(adjacent)
        return original(function, left, right, **kwargs)

    monkeypatch.setattr(profile, "brentq", probing_root)
    result = profile.evaluate(
        value, fitted, deadline=monotonic() + 20, execution=profile.ExecutionPolicy()
    )
    assert result["status"] == "complete"
    nodes = {node["value"]: node for node in result["nodes"]}
    assert len(nodes) == len(result["nodes"])
    assert requests[0] != requests[1]
    assert requests[0] in nodes and requests[1] in nodes
    assert (
        nodes[requests[0]]["profiled_objective"]
        < nodes[requests[0]]["frozen_objective"]
    )


class SimulatedClock:
    """Advance deterministic elapsed and CPU counters only in instrumented work."""

    def __init__(self):
        self.wall, self.cpu = monotonic(), 20.0

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu

    def patch(self, monkeypatch):
        monkeypatch.setattr(profile, "monotonic", lambda: self.wall)
        monkeypatch.setattr(profile, "process_time", lambda: self.cpu)


def test_opt_in_timing_separates_numerical_and_all_callback_costs(monkeypatch):
    value = problem()
    fitted = endpoint(value)
    clock, ledger = SimulatedClock(), []
    clock.patch(monkeypatch)
    original = profile._solve

    def solve(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            clock.advance(0.25, 0.125)

    def save(row):
        ledger.append(row)
        clock.advance(0.2, 0.05)

    monkeypatch.setattr(profile, "_solve", solve)
    result = profile.evaluate(
        value,
        fitted,
        deadline=clock.wall + 1000,
        policy=profile.Policy(maximum_calls=5),
        execution=profile.ExecutionPolicy(),
        save=save,
    )
    timing = result["timing"]
    assert result["status"] == "partial"
    assert timing["numerical_wall_seconds"] == pytest.approx(1.0)
    assert timing["numerical_cpu_seconds"] == pytest.approx(0.5)
    assert timing["save_callback_wall_seconds"] == pytest.approx(0.2 * len(ledger))
    assert timing["save_callback_cpu_seconds"] == pytest.approx(0.05 * len(ledger))
    assert timing["orchestration_wall_seconds"] == pytest.approx(0, abs=1e-10)
    assert timing["orchestration_cpu_seconds"] == pytest.approx(0, abs=1e-10)
    assert timing["total_wall_seconds"] == pytest.approx(1.0 + 0.2 * len(ledger))
    assert timing["total_cpu_seconds"] == pytest.approx(0.5 + 0.05 * len(ledger))
    assert timing["save_callback_calls"] == len(ledger)
    assert ledger[-1]["timing"]["save_callback_calls"] == len(ledger) - 1
    assert timing["numerical_by_kind"]["solve"]["started"] == result["solve_calls"]
    assert result["elapsed_seconds"] == timing["total_wall_seconds"]


def test_failed_numerical_operation_keeps_wall_cpu_and_iteration_cost(monkeypatch):
    value = problem()
    fitted = endpoint(value)
    clock = SimulatedClock()
    clock.patch(monkeypatch)

    def failed_solve(*args, **kwargs):
        clock.advance(2.0, 0.5)
        raise profile._SolveFailure("synthetic inner failure", 7)

    monkeypatch.setattr(profile, "_solve", failed_solve)
    result = profile.evaluate(
        value, fitted, deadline=clock.wall + 1000, execution=profile.ExecutionPolicy()
    )
    timing = result["timing"]
    assert result["status"] == "unqualified"
    assert result["calls"] == result["completed_calls"] == 2
    assert result["failed_calls"] == 1 and result["inner_iterations"] == 7
    assert timing["numerical_operations_failed"] == 1
    assert timing["numerical_wall_seconds"] == pytest.approx(2.0)
    assert timing["numerical_cpu_seconds"] == pytest.approx(0.5)
    assert timing["save_callback_calls"] == 0
    assert timing["numerical_by_kind"]["solve"]["failed"] == 1


@pytest.mark.parametrize(
    "failure_phase", ["diagnostic", "affine_design", "global_optimum", "terminal"]
)
def test_failed_save_is_timed_and_cannot_start_unjournaled_work(
    monkeypatch, failure_phase
):
    value = problem()
    fitted = endpoint(value)
    clock, ledger = SimulatedClock(), []
    clock.patch(monkeypatch)

    def save(row):
        ledger.append(row)
        clock.advance(0.2, 0.05)
        event = row["event"]
        terminal = event["phase"] == "diagnostic" and event["status"] == "complete"
        if (failure_phase == "terminal" and terminal) or (
            failure_phase != "terminal" and event["phase"] == failure_phase
        ):
            raise OSError("synthetic persistence failure")

    result = profile.evaluate(
        value,
        fitted,
        deadline=clock.wall + 1000,
        execution=profile.ExecutionPolicy(),
        save=save,
    )
    timing = result["timing"]
    assert result["status"] == (
        "partial" if failure_phase == "terminal" else "unqualified"
    )
    assert "save callback failed" in result["reason"]
    assert not result["journal_persistence_complete"]
    assert timing["save_callback_failures"] == 1
    assert timing["save_callback_calls"] == len(ledger)
    assert timing["save_callback_wall_seconds"] == pytest.approx(0.2 * len(ledger))
    assert timing["save_callback_cpu_seconds"] == pytest.approx(0.05 * len(ledger))
    if failure_phase != "terminal":
        assert ("rollout" in timing["numerical_by_kind"]) == (
            failure_phase == "global_optimum"
        )
        assert "solve" not in timing["numerical_by_kind"]
        assert result["inner_iterations_complete"]
        assert result["calls"] == result["completed_calls"]
        assert result["supports"] == []


def test_slow_start_save_exhausts_allowance_before_numerical_work(monkeypatch):
    value = problem()
    fitted = endpoint(value)
    clock = SimulatedClock()
    clock.patch(monkeypatch)

    def save(row):
        if row["event"]["phase"] == "affine_design":
            clock.advance(2, 0.01)

    result = profile.evaluate(
        value,
        fitted,
        deadline=clock.wall + 1,
        execution=profile.ExecutionPolicy(),
        save=save,
    )
    assert result["status"] == "partial"
    assert result["calls"] == result["completed_calls"] == 1
    assert result["failed_calls"] == 1
    assert "rollout" not in result["timing"]["numerical_by_kind"]
    assert result["timing"]["save_callback_wall_seconds"] >= 2
    assert result["supports"] == []


def test_final_save_cost_is_included_in_opt_in_wall_allowance(monkeypatch):
    value = problem()
    fitted = endpoint(value)
    clock = SimulatedClock()
    clock.patch(monkeypatch)

    def save(row):
        if (
            row["event"]["phase"] == "diagnostic"
            and row["event"]["status"] == "complete"
        ):
            clock.advance(2, 0.01)

    result = profile.evaluate(
        value,
        fitted,
        deadline=clock.wall + 1,
        execution=profile.ExecutionPolicy(),
        save=save,
    )
    assert result["status"] == "partial"
    assert result["reason"] == "practical profile wall allowance exhausted during save"
    assert result["calls"] == result["completed_calls"]
    assert len(result["supports"]) == len(profile.POLICY["thresholds"])
    assert result["timing"]["save_callback_wall_seconds"] == pytest.approx(2)
    assert result["elapsed_seconds"] == pytest.approx(2)


def test_opt_in_execution_policy_is_strict_and_does_not_expand_m39_policy():
    assert profile.EXECUTION_POLICY == {
        "protocol": "practical-profile-execution-2",
        "mode": "separate-exact-target-caches",
    }
    assert "execution" not in profile.POLICY
    with pytest.raises(ValidationError):
        profile.ExecutionPolicy(mode="round-nearby-targets")
    with pytest.raises(ValidationError):
        profile.Policy(execution=profile.EXECUTION_POLICY)
