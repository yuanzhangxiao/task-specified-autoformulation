"""Positive meal flux coordinates preserve original equations and domain strata."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_meal_coordinates as meal
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


def fixture(*, reservoir="q*r-k*A", output="r-x+k*A", observation="2*x", bounded=False):
    problem = PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "meal_chart_control",
                "parent_candidate_id": None,
                "states": [
                    {"name": "x", "kind": "observed"},
                    {"name": "A", "kind": "latent"},
                ],
                "parameters": [
                    {
                        "name": name,
                        "scope": "global",
                        "role": "nonnegative_coefficient",
                        **({"bounds": {"lower": 0, "upper": 10}} if bounded else {}),
                    }
                    for name in ("k", "q")
                ],
                "processes": [],
                "state_equations": [
                    {"state": "x", "rhs": output},
                    {"state": "A", "rhs": reservoir},
                ],
                "observation_mappings": [{"channel": "y", "expression": observation}],
                "initial_conditions": [
                    {"state": name, "fixed_value": 0, "scope": "global"}
                    for name in ("x", "A")
                ],
            },
            "context": {"targets": ["y"], "external_inputs": ["r"]},
            "graph": {
                "rules": {
                    "x": {
                        "initial": {
                            "mode": "known",
                            "value": 0.2,
                            "justification": "preparation",
                        }
                    },
                    "A": {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory",
                            "role": "nonnegative_coefficient",
                            "guess": 2,
                        }
                    },
                }
            },
            "parameter_guesses": {"k": 0.1, "q": 2.0},
        }
    )
    time = np.linspace(0, 2, 9)
    data = PublicSplit.model_validate(
        {
            "name": "train",
            "fingerprint": "meal-coordinates",
            "rows": [
                {
                    "trajectory_id": str(i),
                    "time": time.tolist(),
                    "targets": {"y": (np.exp(-time) + i * time).tolist()},
                    "external_inputs": {"r": (np.ones(len(time)) * (i + 1)).tolist()},
                }
                for i in range(2)
            ],
        }
    )
    initial = kernel.TrainingOracle(problem, data, kernel.NumericalPolicy())
    return polish.make_oracle(problem, data, kernel.NumericalPolicy(), initial.start)


def certified(oracle):
    return meal.certify(
        oracle,
        oracle.start,
        state="A",
        rate="k",
        input_gain="q",
        input_name="r",
        output_state="x",
    )


def test_exact_positive_chart_roundtrip_and_all_trajectory_initials():
    original, oracle = fixture()
    chart = certified(oracle)
    before = deepcopy(oracle.start)
    encoded = chart.encode(before)
    assert chart.names == ("k", "meal_flux_gain:A", *oracle.names[2:])
    assert encoded[oracle.names.index("q")] == pytest.approx(0.2)
    assert chart.decode(encoded) == pytest.approx(oracle.vector(before))
    assert len(chart.root_names) == 2
    scaled = chart.scaled_initials(before)
    assert all(value == pytest.approx(0.2) for value in scaled.values())
    changed = chart.reseed(before, 0.05)
    assert changed["q"] == pytest.approx(4)
    assert all(changed[n] == pytest.approx(4) for n in chart.root_names)
    assert chart.scaled_initials(changed) == pytest.approx(scaled)
    assert chart.encode(changed)[oracle.names.index("q")] == pytest.approx(0.2)
    lifted = original.parameters(oracle.reduction.lift(oracle.vector(changed)))
    assert original.vector(lifted) is not None
    assert oracle.start == before
    assert not chart.audit["equation_family_changed"]


def test_changed_rate_recomputes_input_gain_even_when_flux_coordinate_unchanged():
    _, oracle = fixture()
    chart = certified(oracle)
    point = chart.encode(oracle.start)
    point[oracle.names.index("k")] *= 2
    restored = oracle.parameters(chart.decode(point))
    assert restored["q"] == pytest.approx(oracle.start["q"] / 2)
    for name in chart.root_names:
        assert restored[name] == oracle.start[name]


def test_same_start_profile_objective_and_original_rollout_agree():
    original, oracle = fixture()
    chart = certified(oracle)
    profiler = roots.RootProfile(oracle)
    first = profiler.profile(oracle.start, monotonic() + 10)
    decoded = oracle.parameters(chart.decode(chart.encode(oracle.start)))
    other = profiler.profile(decoded, monotonic() + 10)
    assert first["training_nmse"] == pytest.approx(other["training_nmse"], abs=1e-12)
    residual, _ = original.evaluate(
        original.vector(other["original_parameters"]),
        monotonic() + 10,
        jacobian=False,
        method="Radau",
    )
    assert other["training_nmse"] == pytest.approx(np.mean(residual**2), abs=1e-9)


def test_zero_boundary_retains_original_stratum_and_excludes_nonzero_closure():
    _, oracle = fixture()
    chart = certified(oracle)
    zero = oracle.start | {"k": 0}
    audit = chart.zero_stratum(zero)
    assert audit["flux_gain"] == 0
    assert set(audit["scaled_initials"].values()) == {0}
    assert not audit["nonzero_flux_at_zero_rate_permitted"]
    with pytest.raises(NumericalInvalidTrial, match="zero rate"):
        chart.encode(zero)
    point = chart.encode(oracle.start)
    point[oracle.names.index("k")] = 0
    with pytest.raises(NumericalInvalidTrial, match="zero rate"):
        chart.decode(point)
    point[oracle.names.index("q")] = 0
    with pytest.raises(NumericalInvalidTrial, match="zero rate"):
        chart.decode(point)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"reservoir": "q*r-2*k*A"}, "reservoir equation"),
        ({"output": "r-x+2*k*A"}, "exactly"),
        ({"output": "r-x+k*A+q"}, "overlapping"),
        ({"observation": "x+A"}, "additional use"),
        ({"bounded": True}, "unbounded domains"),
    ],
)
def test_unsupported_algebra_and_domains_fail_closed(kwargs, match):
    _, oracle = fixture(**kwargs)
    with pytest.raises(ValueError, match=match):
        certified(oracle)


def test_initializer_dependence_and_negative_root_domain_are_rejected():
    original, _ = fixture()
    for kind in ("dependent", "negative"):
        payload = original.problem.model_dump(mode="json")
        if kind == "dependent":
            payload["graph"]["rules"]["x"]["initial"] = {
                "mode": "map",
                "expression": "a0",
                "state_initials": {"a0": "A"},
            }
        else:
            payload["graph"]["rules"]["A"]["initial"]["role"] = "coefficient"
        problem = PrefixFitProblem.model_validate(payload)
        initial = kernel.TrainingOracle(
            problem, original.training, kernel.NumericalPolicy()
        )
        _, oracle = polish.make_oracle(
            problem, original.training, kernel.NumericalPolicy(), initial.start
        )
        with pytest.raises(ValueError, match=r"additional use|unbounded domains"):
            certified(oracle)


def test_nonfinite_inverse_and_underflow_are_rejected():
    _, oracle = fixture()
    chart = certified(oracle)
    point = chart.encode(oracle.start)
    point[oracle.names.index("k")] = 1e-300
    point[oracle.names.index("q")] = 1e300
    with pytest.raises(NumericalInvalidTrial, match="inverse meal gain"):
        chart.decode(point)
    with pytest.raises(NumericalInvalidTrial, match="flux product"):
        chart.encode(oracle.start | {"k": 1e-300, "q": 1e-300})
    with pytest.raises(ValueError, match="strictly positive"):
        chart.reseed(oracle.start, 0)


def test_search_keeps_original_parameters_and_profiles_roots(monkeypatch):
    _, oracle = fixture()
    chart = certified(oracle)
    saved = []
    original = roots.RootProfile.profile
    calls = []

    def observe(self, parameters, deadline):
        calls.append(dict(parameters))
        return original(self, parameters, deadline)

    monkeypatch.setattr(roots.RootProfile, "profile", observe)
    result = roots.root_profile_search(
        oracle, oracle.start, 10, 8, saved.append, coordinate_chart=chart
    )
    assert result["search_calls_started"] == 8
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )
    assert result["shared_coordinate_certificate"] == chart.audit
    assert "meal_flux_gain:A" in result["outer_coordinates"]
    assert result["outer_coordinate_chart"]["meal_flux_coordinates"]
    assert set(result["best"]["parameters"]) == set(oracle.names)
    assert any(
        p["k"] != oracle.start["k"]
        and p["q"] != oracle.start["q"]
        and p["k"] * p["q"] == pytest.approx(0.2)
        for p in calls
    )
    assert calls[0] == oracle.start


def test_search_rejects_fixed_overlap_and_foreign_chart():
    _, oracle = fixture()
    chart = certified(oracle)
    with pytest.raises(ValueError, match="overlap"):
        chart.validate_use(oracle, {"k": 0})
    _, other = fixture()
    with pytest.raises(ValueError, match="different oracle"):
        chart.validate_use(other, {})
    result = roots.root_profile_search(
        oracle,
        oracle.start,
        5,
        8,
        lambda _: None,
        coordinate_chart=chart,
        fixed={"q": 1},
    )
    assert result["search_calls_started"] == 0
    assert result["best"] is None
