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


def fixture(
    *, reservoir="q*r-k*A", output="r-x+k*A", observation="2*x",
    bounded=False, rate=0.1, input_gain=2.0,
):
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
            "parameter_guesses": {"k": rate, "q": input_gain},
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


@pytest.mark.parametrize("new_rate", [1e-6, 1e-4, 0.01, 1.0])
def test_source_flux_unit_is_invariant_under_rate_reseeding(new_rate):
    """A stale billion-unit q scale must not inflate physically identical g."""
    _, oracle = fixture(rate=1e-10, input_gain=1e9)
    chart = certified(oracle)
    index = oracle.names.index("q")
    source_units = chart.units(oracle.start)
    changed = chart.reseed(oracle.start, new_rate)
    assert chart.encode(changed)[index] == pytest.approx(0.1)
    assert chart.units(changed) == pytest.approx(source_units)
    assert source_units[index] == pytest.approx(0.1)
    record = chart.audit["optimizer_gain_scale"]
    assert record["source_input_gain_unit"] == 1e9
    assert record["source_rate"] == 1e-10
    assert record["source_flux_gain"] == pytest.approx(0.1)
    assert record["rule"] == "source_flux_gain"
    assert not record["zero_gain_fallback_used"]
    assert record["reseed_invariant"]


def test_positive_gain_scale_does_not_follow_stale_original_gain_units():
    _, oracle = fixture()
    oracle.units[oracle.names.index("q")] = 1e15
    chart = certified(oracle)
    assert chart.flux_gain_unit == pytest.approx(0.2)
    oracle.units[oracle.names.index("q")] = 1e20
    assert chart.units(oracle.start)[oracle.names.index("q")] == pytest.approx(0.2)


def test_zero_gain_fallback_is_finite_frozen_and_can_leave_the_boundary():
    _, oracle = fixture(input_gain=0.0)
    chart = certified(oracle)
    index = oracle.names.index("q")
    record = chart.audit["optimizer_gain_scale"]
    assert record["zero_gain_fallback_used"]
    assert record["source_flux_gain"] == 0.0
    assert record["rule"] == "source_rate_times_input_gain_unit"
    assert record["unit"] == pytest.approx(0.1)
    for rate in (1e-6, 0.1, 2.0):
        start = chart.reseed(oracle.start, rate)
        units = chart.units(start)
        assert units[index] == pytest.approx(0.1)
        probe = chart.encode(start)
        probe[index] += units[index]
        lifted = oracle.parameters(chart.decode(probe))
        assert lifted["q"] > 0
        assert lifted["k"] * lifted["q"] == pytest.approx(0.1)


@pytest.mark.parametrize("bad_unit", [0.0, -1.0, np.nan, np.inf])
def test_invalid_source_gain_units_fail_closed(bad_unit):
    _, oracle = fixture(input_gain=0.0)
    oracle.units[oracle.names.index("q")] = bad_unit
    with pytest.raises(NumericalInvalidTrial, match="source meal gain unit"):
        certified(oracle)


def test_zero_gain_fallback_rejects_nonrepresentable_product():
    _, oracle = fixture(input_gain=0.0)
    oracle.units[oracle.names.index("q")] = 1e-100
    with pytest.raises(NumericalInvalidTrial, match="source meal flux unit"):
        meal.certify(
            oracle, oracle.start | {"k": 1e-300}, state="A", rate="k",
            input_gain="q", input_name="r", output_state="x",
        )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_reseeded_characteristic_gain_probes_profile_and_replay(direction):
    """Explicit gain probes cover the direction missed by short preflights."""
    original, oracle = fixture(rate=1e-10, input_gain=1e9)
    chart = certified(oracle)
    start = chart.reseed(oracle.start, 0.01)
    index = oracle.names.index("q")
    point = chart.encode(start)
    unit = max(abs(point[index]), chart.units(start)[index])
    point[index] += direction * unit
    changed = oracle.parameters(chart.decode(point))
    assert changed["q"] == pytest.approx(0.0 if direction < 0 else 20.0)
    assert changed["k"] == start["k"]
    endpoint = roots.RootProfile(oracle).profile(changed, monotonic() + 10)
    residual, _ = original.evaluate(
        original.vector(endpoint["original_parameters"]), monotonic() + 10,
        jacobian=False, method="Radau",
    )
    assert np.isfinite(endpoint["training_nmse"])
    assert endpoint["training_nmse"] == pytest.approx(np.mean(residual**2), abs=1e-8)


def test_units_validate_the_reseeded_point_and_source_identity():
    _, oracle = fixture()
    chart = certified(oracle)
    with pytest.raises(NumericalInvalidTrial, match="zero rate"):
        chart.units(oracle.start | {"k": 0})
    chart.audit["problem_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        chart.units(oracle.start)


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


def test_certificate_rejects_initializer_root_in_output_balance(monkeypatch):
    """The zero representative requires exclusive use even after scope lowering."""
    original, oracle = fixture()
    trees = meal.gains._expressions(original)
    root = original.roots[0]
    trees["rhs:x"] = meal.gains.RestrictedParser().parse(
        f"r-x+k*A+{root}", location="rhs:x"
    ).tree.body
    # Inspect the certificate's executable-expression boundary independently
    # of the proposer schema, which currently disallows generated root names.
    monkeypatch.setattr(meal.gains, "_expressions", lambda _: trees)
    with pytest.raises(ValueError, match="overlapping uses"):
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


@pytest.mark.parametrize("value", [0, -0.1, np.inf, np.nan, True])
def test_positive_fixed_rate_rejects_invalid_boundary(value):
    _, oracle = fixture()
    with pytest.raises(ValueError, match="finite positive"):
        certified(oracle).validate_use(oracle, {"k": value})


def test_fixed_positive_rate_profiles_gain_without_changing_rate(monkeypatch):
    _, oracle = fixture(rate=1e-10, input_gain=1e9)
    chart = certified(oracle)
    start = chart.reseed(oracle.start, 0.05)
    original = roots.RootProfile.profile
    attempts = []

    def capture(self, parameters, deadline):
        attempts.append(dict(parameters))
        return original(self, parameters, deadline)

    monkeypatch.setattr(roots.RootProfile, "profile", capture)
    result = roots.root_profile_search(
        oracle, start, 10, 5, lambda _: None,
        coordinate_chart=chart, fixed={"k": 0.05},
    )
    assert result["search_calls_started"] == 5
    assert all(p["k"] == 0.05 for p in attempts)
    assert "k" not in result["outer_coordinates"]
    assert "meal_flux_gain:A" in result["outer_coordinates"]
    assert any(p["q"] != start["q"] for p in attempts)
    assert result["initial_sweep_completed"]
    assert result["outer_coordinate_chart"]["units"] == pytest.approx([0.1])
    assert any(p["q"] == pytest.approx(0.0) for p in attempts)
    assert any(p["q"] == pytest.approx(2 * start["q"]) for p in attempts)
    assert result["best"]["parameters"]["k"] == 0.05


def test_zero_representative_preserves_outputs_and_removes_flat_gain_search():
    original, oracle = fixture()
    chart = certified(oracle)
    start = oracle.start | {"k": 0.0, "q": 1234.0}
    start.update(dict.fromkeys(chart.root_names, 5000.0))
    reduced = chart.zero_representative(start)
    certificate = reduced["certificate"]
    assert certificate["fixed_shared_coordinates"] == {"k": 0.0, "q": 0.0}
    assert not certificate["hidden_state_trajectories_preserved"]
    canonical = reduced["parameters"]
    assert all(canonical[n] == 0 for n in ("q", *chart.root_names))
    deadline = monotonic() + 10
    before, _ = oracle.evaluate(oracle.vector(start), deadline, jacobian=False)
    after, _ = oracle.evaluate(oracle.vector(canonical), deadline, jacobian=False)
    assert after == pytest.approx(before, abs=1e-10)
    independent, _ = original.evaluate(
        original.vector(reduced["original_parameters"]),
        deadline, jacobian=False, method="Radau",
    )
    assert independent == pytest.approx(before, abs=1e-8)
    result = roots.root_profile_search(
        oracle, canonical, 10, 5, lambda _: None,
        fixed=certificate["fixed_shared_coordinates"],
    )
    assert result["outer_coordinates"] == []
    assert result["best"]["parameters"]["q"] == 0
    assert result["best"]["training_nmse"] == pytest.approx(np.mean(before**2))
    assert start["q"] == 1234.0


def test_zero_representative_rejects_positive_rate_or_changed_source():
    _, oracle = fixture()
    chart = certified(oracle)
    with pytest.raises(ValueError, match="exactly zero"):
        chart.zero_representative(oracle.start)
    chart.audit["problem_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        chart.zero_representative(oracle.start | {"k": 0.0})
