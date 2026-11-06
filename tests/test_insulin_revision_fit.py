"""Signed matrix feedback, exact gain quotient, and frozen numerical scales."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import insulin_revision_fit as fit
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


@pytest.fixture(scope="module")
def specimen():
    params = {
        fit.AMPLITUDE: 0.03,
        fit.GAIN: 1.0,
        "d": -0.95,
        "e": 0.04,
        "f": 0.08,
        "h": 0.2,
        "c_I": 1.0,
        "k_abs": 0.1,
        "k_abs_rate": 1.0,
        fit.joint.DEFAULT_DRIFT: 0.0,
    }
    meal = fit.comparison.MEAL
    infusion = "insulin_pmol_per_kg_min"
    equations = {
        "Gp": f"{meal}-U+k_abs*A",
        "I": f"c_I*{meal}-I + {fit.GAIN}*production - d*I + h*{infusion}",
        "A": f"k_abs_rate*{meal}-k_abs*A",
        fit.MEMORY: f"e*I-f*{fit.MEMORY}",
        "Q": fit.joint.DEFAULT_DRIFT,
    }
    model = {
        "candidate_id": "feedback_fixture",
        "parent_candidate_id": None,
        "states": [
            {"name": n, "kind": "observed" if n in ("Gp", "I") else "latent"}
            for n in equations
        ],
        "parameters": [
            {
                "name": n,
                "scope": "global",
                "role": "nonnegative_coefficient"
                if n
                in ("k_abs", "k_abs_rate", "c_I", fit.GAIN, fit.joint.DEFAULT_DRIFT)
                else "coefficient",
            }
            for n in params
        ],
        "state_equations": [{"state": n, "rhs": rhs} for n, rhs in equations.items()],
        "processes": [
            {"name": "production", "expression": f"{fit.AMPLITUDE}*Gp"},
            {"name": "U", "expression": f"Q+{fit.MEMORY}"},
        ],
        "observation_mappings": [
            {"channel": n, "expression": n} for n in ("Gp", "I", "U")
        ],
        "initial_conditions": [
            {
                "state": n,
                "scope": "global",
                **({"expression": n} if n in ("Gp", "I") else {"fixed_value": 0}),
            }
            for n in equations
        ],
    }
    parent = PrefixFitProblem.model_validate(
        {
            "base_candidate": model,
            "parameter_guesses": params,
            "context": {
                "targets": ["Gp", "I", "U"],
                "external_inputs": [meal, infusion],
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
                        n: {
                            "initial": {
                                "mode": "value",
                                "scope": "trajectory",
                                "role": "nonnegative_coefficient",
                                "guess": 1,
                            }
                        }
                        for n in ("A", fit.MEMORY)
                    },
                }
            },
        }
    )
    payload = parent.model_dump(mode="json")
    payload["base_candidate"]["processes"][0]["expression"] += (
        f"+{fit.DELAY}*{fit.MEMORY}"
    )
    payload["base_candidate"]["parameters"].append(
        {"name": fit.DELAY, "scope": "global", "role": "coefficient"}
    )
    payload["parameter_guesses"][fit.DELAY] = 1.0
    revised = PrefixFitProblem.model_validate(payload)
    data = PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "synthetic-feedback",
            "rows": [
                {
                    "trajectory_id": f"train_{j:03d}",
                    "time": list(range(7)),
                    "targets": {
                        "Gp": [3, 3.1, 4, 3, 2, 1, 1],
                        "I": [2, 2.2, 2.5, 2.1, 2, 1.8, 1.5],
                        "U": [1, 2, 3, 2, 1.5, 2, 1],
                    },
                    "external_inputs": {
                        meal: [0, j + 1, 0, 0, 1, 0, 0],
                        infusion: [0, 0, 0, 0, 0.1 + j, 0, 0],
                    },
                }
                for j in range(2)
            ],
        }
    )
    numerical = fit.kernel.NumericalPolicy()
    original = fit.kernel.TrainingOracle(parent, data, numerical)
    warm = original.start | {fit.DELAY: 0.0}
    return revised, parent, data, numerical, warm


def test_complete_nested_warm_and_joint_profile_keep_feedback_outer(specimen):
    _, _, data, _, warm = specimen
    run = fit.InsulinRevisionFit(*specimen)
    residual, _ = run.parent_original.evaluate(
        run.parent_original.vector(run.parent_parameters),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    score = float(np.mean(residual**2))
    nested = run.verify_start(score, 20)
    assert nested["passed"] and nested["full_executable_nesting_certified"]
    assert nested["parameters"] == warm
    assert run.setup_audit()["passed"]
    assert run.profile_audit["shared_count"] == 3
    assert run.profile_audit["trajectory_root_count"] == 4
    assert fit.DELAY in run.profile_audit["outer_names"]
    assert fit.DELAY not in run.profile_audit["inner_names"]
    q = run.qualify(10, lambda _: None)
    assert q["passed"] and q["joint_coefficient_optimality_checked"]
    verified = run.verify(q["parameters"], q["training_nmse"], 10, lambda _: None)
    assert verified["original_equations_verified"]
    metrics = run.metrics(verified, 10)
    assert metrics["training_sha256"] == fit.public.content_sha256(data)
    assert metrics["training_nmse"] == pytest.approx(verified["training_nmse"])
    assert metrics["diagnostics"]["kappa_sign"] == "zero"
    assert "Q" in metrics["diagnostics"]["zero_dynamics_rows"]
    assert len(metrics["diagnostics"]["full_state_matrix_eigenvalues"]) == 5


@pytest.mark.parametrize("gain", [0.0, 2.0, 5.0])
def test_gain_section_preserves_full_original_rollout_even_at_zero(specimen, gain):
    run = fit.InsulinRevisionFit(*specimen)
    point = run.warm | {fit.GAIN: gain, fit.DELAY: -0.03}
    lifted = fit.canonical_parameters(run.original, point)
    assert lifted[fit.GAIN] == 1.0
    assert lifted[fit.DELAY] == pytest.approx(-0.03 * gain)
    assert lifted[fit.AMPLITUDE] == pytest.approx(point[fit.AMPLITUDE] * gain)
    values = [
        run.original.evaluate(run.original.vector(p), monotonic() + 10, jacobian=False)[
            0
        ]
        for p in (point, lifted)
    ]
    assert np.allclose(*values, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("alter", ["reuse", "nonreal", "nonzero_warm", "missing_root"])
def test_illegal_gauge_or_incomplete_warm_fails_closed(specimen, alter):
    revised, parent, data, numerical, warm = specimen
    payload = revised.model_dump(mode="json")
    warm = dict(warm)
    if alter == "reuse":
        payload["base_candidate"]["state_equations"][0]["rhs"] += f"+{fit.DELAY}"
    elif alter == "nonreal":
        payload["base_candidate"]["parameters"][-1]["role"] = "nonnegative_coefficient"
    elif alter == "nonzero_warm":
        warm[fit.DELAY] = 1.0
    else:
        warm.pop(next(n for n in warm if n.startswith("trajectory:")))
    with pytest.raises(ValueError):
        fit.InsulinRevisionFit(
            PrefixFitProblem.model_validate(payload), parent, data, numerical, warm
        )


def test_centering_uses_net_damping_and_does_not_change_domains(specimen):
    run = fit.InsulinRevisionFit(*specimen)
    old = run.oracle.vector(run.start)
    centered = run.chart.encode(run.start)
    i = run.oracle.names.index("d")
    assert old[i] == pytest.approx(-0.95)
    assert centered[i] == 0.0
    assert run.chart.units(run.start)[i] == pytest.approx(0.05)
    assert np.allclose(run.chart.decode(centered), old)
    for name in ("d", fit.DELAY):
        j = run.oracle.names.index(name)
        assert run.chart.lower[j] == -np.inf
        assert run.chart.upper[j] == np.inf
    j = run.oracle.names.index(fit.DELAY)
    for value in (-2.0, 2.0):
        test = centered.copy()
        test[j] = value
        assert run.oracle.parameters(run.chart.decode(test))[fit.DELAY] == value
    assert run.chart.units(run.start)[j] == pytest.approx(0.08)


def test_signed_controls_still_verify_nested_zero_and_search_has_fixed_gauge(
    specimen, monkeypatch
):
    run = fit.InsulinRevisionFit(*specimen, delay_start=-0.008)
    residual, _ = run.parent_original.evaluate(
        run.parent_original.vector(run.parent_parameters),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    nested = run.verify_start(float(np.mean(residual**2)), 20)
    assert nested["passed"] and nested["parameters"][fit.DELAY] == 0.0
    assert run.start[fit.DELAY] == -0.008
    calls = {}

    def fake(*args, **kwargs):
        calls.update(kwargs)
        return {"captured": True}

    monkeypatch.setattr(fit.roots, "root_profile_search", fake)
    assert run.search(fit.Policy(search_seconds=1, maximum_calls=1), lambda _: None)[
        "captured"
    ]
    assert calls["fixed"] == {fit.GAIN: 1.0}
    assert calls["coordinate_chart"] is run.chart
    assert calls["profiler_factory"] is run.profiler_factory


def test_changed_parent_does_not_pass_full_nesting_check(specimen):
    revised, parent, data, numerical, warm = specimen
    payload = deepcopy(parent.model_dump(mode="json"))
    payload["base_candidate"]["state_equations"][0]["rhs"] += "+0.01"
    run = fit.InsulinRevisionFit(
        revised, PrefixFitProblem.model_validate(payload), data, numerical, warm
    )
    with pytest.raises(ValueError, match="complete parent"):
        run.verify_start(1.0, 10)


def test_centered_chart_is_accepted_by_real_profile_search(specimen):
    run = fit.InsulinRevisionFit(*specimen)
    result = run.search(fit.Policy(search_seconds=10, maximum_calls=1), lambda _: None)
    assert (
        result["shared_coordinate_certificate"]["protocol"]
        == "centered-net-affine-meal-chart-1"
    )
    assert result["best"]["parameters"] is not None
    assert result["fixed_coordinates"] == {fit.GAIN: 1.0}
