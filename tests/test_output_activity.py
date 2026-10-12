"""Output closure, all-inner solves, initializer paths and conservative stopping."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import identifiable_cases, nonlinear_rollout, poor_fit
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import scaling_recovery as recovery
from autoformalism.fitting.confidence_checks import TrainingProblem
from tests.test_recovery_numerics import QuadraticOracle


def inputs(*, rhs="a*u", hidden="b*z", scalar=False):
    request = identifiable_cases.request("nonlinear", 0).model_dump(mode="json")
    candidate = request["base_candidate"]
    candidate["processes"] = [{"name": "unused", "expression": "c+d"}]
    candidate["state_equations"] = [
        {"state": "y", "rhs": rhs},
        {"state": "z", "rhs": hidden},
    ]
    request["parameter_guesses"] = dict.fromkeys("abcd", 1.0)
    if scalar:
        candidate["processes"] = []
        candidate["states"] = candidate["states"][:1]
        candidate["state_equations"] = candidate["state_equations"][:1]
        candidate["parameters"] = candidate["parameters"][:1]
        candidate["initial_conditions"] = candidate["initial_conditions"][:1]
        request["initialization_plan"]["rules"] = {}
        request["parameter_guesses"] = {"a": 1.0}
    time = np.linspace(0, 1, 9)
    rows = [
        {
            "trajectory_id": "train_0",
            "time": time.tolist(),
            "targets": {"y": (time**2).tolist()},
            "external_inputs": {"u": np.ones_like(time).tolist()},
            "auxiliaries": {},
        }
    ]
    return bridge.Inputs.model_validate(
        {
            "request": request,
            "training": {"name": "train", "rows": rows, "fingerprint": "synthetic"},
        }
    )


def policy():
    p = bridge.Policy()
    return p.model_copy(
        update={
            "adapter": p.adapter.model_copy(
                update={
                    "search": recovery.Policy(
                        poor_fit=poor_fit.Policy(),
                        block_seconds=5,
                        check_seconds=5,
                        sensitivity_seconds=5,
                    ),
                }
            )
        }
    )


def prepared(data):
    value = bridge.prepare(data, policy())
    p = TrainingProblem.model_validate(value["problem"])
    return p, value, nonlinear_rollout.build(p, reduce_inactive=True)


def test_expanded_process_paths_and_initializer_namespace():
    raw = inputs(rhs="flux").model_dump(mode="json")
    raw["request"]["base_candidate"]["processes"].append(
        {"name": "flux", "expression": "a*tanh(z)"}
    )
    _, _, (oracle, _, route) = prepared(bridge.Inputs.model_validate(raw))
    cert = route["output_activity"]
    assert set(cert["active_parameters"]) == {"a", "b", "init_z_value"}
    assert set(cert["inactive_parameters"]) == {"c", "d"}
    assert cert["output_reachable_states"] == ["y", "z"]
    assert oracle.active_indices == (0, 1, 4)
    # Initial data y is not a dynamical edge from z to the future observed y.
    raw = inputs().model_dump(mode="json")
    raw["request"]["initialization_plan"]["rules"]["z"]["initial"] = {
        "mode": "map",
        "expression": "s*y",
        "parameters": [{"name": "s", "role": "scale", "guess": 0.2}],
    }
    _, _, (_, _, route) = prepared(bridge.Inputs.model_validate(raw))
    assert route["output_activity"]["active_parameters"] == ["a"]
    assert "init_z_s" in route["output_activity"]["inactive_parameters"]


def test_multiple_outputs_keep_other_channel_and_hidden_initial_dependencies():
    raw = inputs().model_dump(mode="json")
    raw["request"]["context"]["targets"].append("other")
    raw["request"]["base_candidate"]["observation_mappings"].append(
        {"channel": "other", "expression": "c*z+d"}
    )
    raw["training"]["rows"][0]["targets"]["other"] = [1.0] * 9
    _, _, (_, _, route) = prepared(bridge.Inputs.model_validate(raw))
    assert route["output_activity"]["inactive_parameters"] == []


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("hidden", ["b*z", "-b*z*z"])
def test_fully_profiled_problem_is_one_call_with_analytic_optimum(scalar, hidden):
    data = inputs(scalar=scalar, hidden=hidden)
    p, _, (oracle, profile, route) = prepared(data)
    assert profile.outer == () and route["certificate"]["fully_profiled_output"]
    result = numerical.search(
        oracle,
        p.start,
        monotonic() + 5,
        lambda _: None,
        calls=40,
        target=1e-12,
        profile=profile,
        telemetry=True,
        normalized_objective=True,
        optimizer_scaling="guarded_relative",
        poor_fit=poor_fit.Policy(),
    )
    time = np.array(data.training.rows[0].time)
    expected = time @ time**2 / (time @ time)
    assert result["stop_reason"] == "output_problem_solved"
    assert result["calls"] == result["completed_calls"] == 1
    assert result["best"]["parameters"]["a"] == pytest.approx(expected, abs=1e-10)
    assert all(
        v == p.start[n] for n, v in result["best"]["parameters"].items() if n != "a"
    )


def test_fixed_output_is_verified_without_optimizing_disconnected_parameters(tmp_path):
    data = inputs(rhs="u", hidden="a+b*z")
    result = bridge.run(data, policy(), tmp_path)
    assert result.fit.assessment.output_activity["active_parameters"] == []
    assert result.fit.assessment.search_status == "output_problem_solved"
    assert result.fit.cost.search_calls_started == 1
    assert result.fit.parameters == result.fit.assessment.unestimated_parameters


def test_recovery_certifies_poor_optimum_and_preserves_resume_and_inactive_values(
    tmp_path,
):
    data, settings = inputs(), policy()
    original = deepcopy(data.model_dump(mode="json"))
    result = bridge.run(data, settings, tmp_path)
    fit = result.fit
    assert fit.assessment.search_status == "output_problem_solved"
    assert fit.assessment.recommended_action == "review_equations"
    assert fit.cost.blocks_used == fit.cost.search_calls_started == 1
    assert not fit.verification.target_reached
    assert fit.assessment.unestimated_parameters.keys() == {
        "b",
        "c",
        "d",
        "init_z_value",
    }
    assert all(
        fit.parameters[n] == v for n, v in fit.assessment.unestimated_parameters.items()
    )
    assert data.model_dump(mode="json") == original
    assert bridge.run(data, settings, tmp_path) == result


def test_failed_independent_verification_never_certifies_output_solution(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(recovery.rollout, "verify", lambda *a, **kw: None)
    result = bridge.run(inputs(), policy(), tmp_path)
    assert result.fit.parameters is None
    assert result.fit.assessment.search_status == "verification_needed"
    assert result.fit.cost.blocks_used == 4


def test_empty_outer_still_validates_all_frozen_parameter_vectors():
    p, prep, (oracle, profile, _) = prepared(inputs())
    starts = deepcopy(prep["starts"])
    starts[1]["a"] = -1.0
    with pytest.raises(ValueError, match="outside declared domain"):
        recovery.validate_starts(oracle, profile, p, starts, reduce_inactive=True)


def test_syntactic_activity_never_drops_small_or_canceling_dependencies():
    _, _, (_, _, route) = prepared(inputs(rhs="a*u+1e-20*z+(c-c)"))
    assert set(route["output_activity"]["active_parameters"]) == {
        "a",
        "b",
        "c",
        "init_z_value",
    }


def test_local_zero_jacobian_stops_start_without_claiming_global_optimum():
    class Flat(QuadraticOracle):
        def evaluate(self, vector, deadline):
            return np.ones(2), np.zeros((2, 2))

    r = numerical.search(
        Flat(),
        {"a": 1.0, "initial": 0.0},
        monotonic() + 5,
        lambda _: None,
        calls=40,
        target=1e-12,
        poor_fit=poor_fit.Policy(),
        telemetry=True,
    )
    assert r["calls"] == 1 and r["stop_reason"] == "stationary_poor_fit"
    assert r["stationarity"]["reason"] == "locally_flat_start"
    assert not r["stationarity"]["global_minimum_certified"]


def test_stationarity_requires_progress_step_and_relative_gradient_checks():
    p = poor_fit.Policy()
    base = {
        "q": np.ones(2),
        "loss": 1.0,
        "gradient": 1e-15,
        "correlation": 1e-15,
        "zero_jacobian": False,
    }
    history = [deepcopy(base) for _ in range(p.window)]
    assert poor_fit.witness(history, p, 1e-12)
    for key, value in [("loss", 1e-8), ("gradient", 1.0), ("correlation", 0.2)]:
        changed = deepcopy(history)
        changed[-1][key] = value
        assert poor_fit.witness(changed, p, 1e-12) is None
    changed = deepcopy(history)
    changed[0]["q"] *= 2
    assert poor_fit.witness(changed, p, 1e-12) is None
    changed = deepcopy(history)
    changed[0]["loss"] *= 2
    assert poor_fit.witness(changed, p, 1e-12) is None
    # Weak sensitivity must not turn a large descent correlation into convergence.
    s = poor_fit.sample(
        np.zeros(1),
        np.ones(1),
        np.ones(2),
        np.ones((2, 1)) * 1e-16,
        np.zeros(1),
        np.ones(1) * 10,
    )
    assert s["gradient"] < 1e-8 and s["correlation"] > 0.99


def test_exact_stationary_start_is_local_and_does_not_enter_degenerate_trust_region():
    class ResidualFloor(QuadraticOracle):
        def evaluate(self, vector, deadline):
            return np.r_[vector - [2, 3], 1.0], np.vstack([np.eye(2), np.zeros(2)])

    result = numerical.search(
        ResidualFloor(),
        {"a": 2.0, "initial": 3.0},
        monotonic() + 5,
        lambda _: None,
        calls=40,
        target=1e-12,
        poor_fit=poor_fit.Policy(),
        telemetry=True,
        normalized_objective=True,
        optimizer_scaling="guarded_relative",
    )
    assert result["calls"] == 1
    assert result["stop_reason"] == "stationary_poor_fit"
    assert result["stationarity"]["reason"] == "first_order_stationary_start"
    assert not result["stationarity"]["global_minimum_certified"]
