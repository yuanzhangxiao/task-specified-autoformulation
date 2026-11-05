"""Nested ingestion gain, identical physical starts, and nonrenewable arm work."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import insulin_meal_comparison as fit
from autoformalism.fitting import public_fitting as public


@pytest.fixture(scope="module")
def fixture():
    meal, insulin = fit.MEAL, "insulin_pmol_per_kg_min"
    parameters = {
        "k_abs": 0.2,
        "k_abs_rate": 1.5,
        "a": 0.1,
        "b": -0.7,
        "d": 0.3,
        "f": 0.05,
        "e": 0.4,
        fit.joint.DEFAULT_DRIFT: 0.02,
    }
    equations = {
        "Gp": f"{meal}-U+k_abs*A",
        "I": f"{meal} - I + a*Gp - b*I + d*{insulin}",
        "A": f"k_abs_rate*{meal}-k_abs*A",
        "X_insulin_action": "f*I-e*X_insulin_action",
        "Q": fit.joint.DEFAULT_DRIFT,
    }
    problem = fit.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "synthetic_meal",
                "parent_candidate_id": None,
                "states": [
                    {
                        "name": name,
                        "kind": "observed" if name in {"I", "Gp"} else "latent",
                    }
                    for name in equations
                ],
                "parameters": [
                    {
                        "name": name,
                        "scope": "global",
                        "role": "nonnegative_coefficient"
                        if name in {"k_abs", "k_abs_rate", fit.joint.DEFAULT_DRIFT}
                        else "coefficient",
                    }
                    for name in parameters
                ],
                "state_equations": [
                    {"state": name, "rhs": rhs} for name, rhs in equations.items()
                ],
                "processes": [{"name": "U", "expression": "Q + X_insulin_action"}],
                "observation_mappings": [
                    {"channel": name, "expression": name} for name in ("Gp", "I", "U")
                ],
                "initial_conditions": [
                    {"state": name, "scope": "global"}
                    | (
                        {"expression": name}
                        if name in ("Gp", "I")
                        else {"fixed_value": 0}
                    )
                    for name in equations
                ],
            },
            "parameter_guesses": parameters,
            "context": {
                "targets": ["Gp", "I", "U"],
                "external_inputs": [meal, insulin],
            },
            "graph": {
                "rules": {
                    "Q": {
                        "initial": {
                            "mode": "known",
                            "value": 1,
                            "justification": "preparation",
                        }
                    },
                    **{
                        name: {
                            "initial": {
                                "mode": "value",
                                "scope": "trajectory",
                                "role": "nonnegative_coefficient",
                                "guess": 1,
                            }
                        }
                        for name in ("X_insulin_action", "A")
                    },
                }
            },
        }
    )
    training = fit.PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "synthetic-insulin-control",
            "rows": [
                {
                    "trajectory_id": "train_" + str(i),
                    "time": list(range(7)),
                    "targets": {
                        "Gp": [3, 3.1, 4, 3, 2, 1, 1],
                        "I": [2, 2.2, 2.5, 2.1, 2, 1.8, 1.5],
                        "U": [1, 2, 3, 2, 1.5, 2, 1],
                    },
                    "external_inputs": {
                        meal: [0, i + 1, 0, 0, 1, 0, 0],
                        insulin: [0, 0, 0, 0, 0.1 + i, 0, 0],
                    },
                }
                for i in range(2)
            ],
        }
    )
    numerical = fit.kernel.NumericalPolicy()
    retained = fit.kernel.TrainingOracle(problem, training, numerical).start
    return problem, training, numerical, retained


def test_nested_child_changes_only_the_direct_insulin_gain(fixture):
    problem, training, numerical, retained = fixture
    before = problem.model_dump(mode="json")
    parent = fit.ComparisonFit(problem, training, numerical, retained, "parent")
    child = fit.ComparisonFit(problem, training, numerical, retained, "child")
    parent_value = parent.original.evaluate(
        parent.original.vector(retained),
        monotonic() + 10,
        method="Radau",
        jacobian=False,
    )[0]
    loss = float(np.mean(parent_value**2))
    nested = child.verify_start(loss, 10)
    assert nested["passed"] and nested["parent_arm_maximum_scaled_discrepancy"] < 1e-8
    assert not child.verify_start(loss + 0.1, 10)["passed"]
    assert nested["parameters"] == retained | {fit.GAIN: 1}
    assert child.problem.graph == parent.problem.graph
    assert problem.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="unit insulin"):
        fit.child_problem(child.problem)
    assert child.oracle.lower[child.oracle.names.index(fit.GAIN)] == 0
    with pytest.raises(ValueError, match=r"domain|bound"):
        child.oracle.vector(child.start | {fit.GAIN: -0.01})
    profile = child.profiler_factory(child.oracle)
    parent_profile = parent.profiler_factory(parent.oracle)
    assert profile.audit["shared_count"] == 3
    assert profile.audit["outer_names"] == parent_profile.audit["outer_names"]
    assert profile.audit["trajectory_root_count"] == 4
    assert fit.GAIN in profile.inner_names
    assert child.oracle.reduction.receipt["blocks"][0][
        "physical_forcing_coordinates"
    ] == {fit.MEAL: fit.GAIN}


@pytest.mark.parametrize("true_gain", [0.35, 0.0])
def test_shared_profile_recovers_known_insulin_meal_gain(fixture, true_gain):
    problem, data, numerical, retained = fixture
    child = fit.ComparisonFit(problem, data, numerical, retained, "child")
    point = child.anchor | {fit.GAIN: true_gain}
    residual, _ = child.original.evaluate(
        child.original.vector(point), monotonic() + 10, jacobian=False
    )
    generated = data.model_dump(mode="json")
    cursor = 0
    for row in generated["rows"]:
        m = len(row["time"]) * 3
        channels = child.original.system.channels
        for i, name in enumerate(channels):
            row["targets"][name] = (
                np.array(row["targets"][name])
                + child.original.scales[name]
                * residual[cursor : cursor + m].reshape(-1, 3)[:, i]
            ).tolist()
        cursor += m
    truth_data = fit.PublicSplit.model_validate(generated)
    run = fit.ComparisonFit(problem, truth_data, numerical, retained, "child")
    q = run.qualify(10, lambda _: None)
    assert q["passed"] and q["joint_coefficient_optimality_checked"]
    assert q["parameters"][fit.GAIN] == pytest.approx(true_gain, abs=1e-7)
    assert q["training_nmse"] < 1e-15
    assert run.verify(q["parameters"], q["training_nmse"], 10, lambda _: None)[
        "original_equations_verified"
    ]


def test_shifted_or_no_longer_literal_input_and_name_collision_fail(fixture):
    problem, *_ = fixture
    payload = problem.model_dump(mode="json")
    changed = deepcopy(payload)
    next(
        row
        for row in changed["base_candidate"]["state_equations"]
        if row["state"] == "I"
    )["rhs"] = "-I"
    with pytest.raises(ValueError, match="unit insulin"):
        fit.child_problem(fit.PrefixFitProblem.model_validate(changed))
    payload["parameter_guesses"][fit.GAIN] = 2
    with pytest.raises(ValueError, match="already"):
        fit.child_problem(fit.PrefixFitProblem.model_validate(payload))


def _fake(monkeypatch, *, interrupted=False):
    calls = []

    class Stub:
        def __init__(self, *args):
            pass

        def verify_start(self, *args):
            return {
                "passed": True,
                "original_equations_verified": True,
                "training_nmse": 1.0,
                "parameters": {},
            }

        def qualify(self, *args):
            return {"passed": True, "parameters": {"x": 1}, "training_nmse": 1.0}

        def verify(self, p, loss, *args):
            return {
                "original_equations_verified": True,
                "parameters": p,
                "training_nmse": loss,
            }

        def search(self, policy, save):
            calls.append(policy.maximum_calls)
            best = {"training_nmse": 0.5, "parameters": {"x": 2}}
            if interrupted:
                save({"best": best, "stop_reason": "interrupted"})
                raise KeyboardInterrupt
            return {
                "best": best,
                "search_calls_started": 2,
                "search_calls_completed": 2,
                "stop_reason": "optimizer_stopped",
                "optimizer_success": True,
                "budget_exhausted": False,
                "optimizer_terminal": {"feasible_evaluation_verified": True},
            }

        def metrics(self, *args):
            return {"source": "original_radau"}

    monkeypatch.setattr(fit, "ComparisonFit", Stub)
    return calls


def test_sealed_arm_keeps_receipts_checks_metrics_and_never_restarts(
    fixture, tmp_path, monkeypatch
):
    p, d, n, v = fixture
    policy = fit.Policy()
    calls = _fake(monkeypatch)
    args = (
        tmp_path,
        {"plan": "abc"},
        p,
        d,
        n,
        {"training_nmse": 1.0, "parameters": v, "original_equations_verified": True},
        "child",
        policy,
    )
    result = fit.run_arm(*args)
    assert result["ready_for_comparison"] and result["metrics"]
    assert result["retained_stage"] == "endpoint_verification"
    assert (
        result == fit.run_arm(*args) == fit.read_arm(tmp_path, args[1], "child", policy)
    )
    assert calls == [2000]
    file = tmp_path / "result.json"
    raw = read_seal(file)
    raw["retained_endpoint"]["training_nmse"] = 0
    public._write(file, {"sha256": public.content_sha256(raw), "value": raw})
    with pytest.raises(ValueError, match="receipts"):
        fit.read_arm(tmp_path, args[1], "child", policy)


def test_interrupted_checkpoint_survives_without_enabling_proposer(
    fixture, tmp_path, monkeypatch
):
    p, d, n, v = fixture
    policy = fit.Policy()
    calls = _fake(monkeypatch, interrupted=True)
    args = (
        tmp_path,
        {},
        p,
        d,
        n,
        {"training_nmse": 1.0, "parameters": v, "original_equations_verified": True},
        "parent",
        policy,
    )
    with pytest.raises(KeyboardInterrupt):
        fit.run_arm(*args)
    result = fit.run_arm(*args)
    assert calls == [2000] and not result["ready_for_comparison"]
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["retained_endpoint"]["training_nmse"] == 0.5
