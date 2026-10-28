"""Containment, query semantics, and deterministic bounded baseline searches."""

import json
from collections import Counter
from dataclasses import FrozenInstanceError
from decimal import Context, Decimal, localcontext
from itertools import product

import pytest

from autoformalism.research import premise_baseline as baseline
from autoformalism.research.premise_intervals import Interval

TIMES = tuple(Decimal(value) for value in (1, 2, 3, 5))
SIGMA = Decimal("0.02")
Q = Decimal(10)
TOLERANCES = ("0.1", "0.25", "0.5")


def interval(lo: str, hi: str) -> Interval:
    return Interval(Decimal(lo), Decimal(hi))


def kernel_reference(s: Decimal, time: Decimal) -> Decimal:
    """Evaluate the mathematical kernel in a separate high-precision context."""
    with localcontext(Context(prec=140)):
        return ((-s * time).exp() - (-time).exp()) / (1 - s)


def problem(rate: str = "0.2", contract: str = "coupled"):
    """Create fixed observed decimals without passing a truth label to inference."""
    with localcontext(Context(prec=140)):
        observations = tuple(
            str(kernel_reference(Decimal(rate), time).quantize(Decimal("1e-12")))
            for time in TIMES
        )
    return baseline.Problem("test_observations", observations, contract)


def q_reference(item, amplitude: Decimal, rate: Decimal) -> Decimal:
    """Evaluate a point's residual directly, independently of profiling."""
    with localcontext(Context(prec=140)):
        return sum(
            ((Decimal(y) - amplitude * kernel_reference(rate, time)) / SIGMA) ** 2
            for y, time in zip(item.observations, TIMES, strict=True)
        )


def amplitude_reference(item, rate: Decimal) -> tuple[Decimal, Decimal] | None:
    """Compute the exact-form conditional interval at substantially higher precision."""
    with localcontext(Context(prec=140)):
        values = [kernel_reference(rate, time) for time in TIMES]
        a = sum(value**2 / SIGMA**2 for value in values)
        c = sum(
            value * Decimal(y) / SIGMA**2
            for value, y in zip(values, item.observations, strict=True)
        )
        fitted = c / a
        residual = q_reference(item, fitted, rate)
        if residual > Q:
            return None
        radius = ((Q - residual) / a).sqrt()
        lo, hi = max(Decimal("0.5"), fitted - radius), min(Decimal(2), fitted + radius)
        return None if lo > hi else (lo, hi)


def task_reference(
    item, amplitude: Decimal, rate: Decimal, gamma: Decimal, rho: Decimal
) -> Decimal:
    """Evaluate a complete world's future mean under its declared physical contract."""
    with localcontext(Context(prec=140)):
        before = amplitude * kernel_reference(rate, Decimal(7))
        pulse = (-gamma * 3).exp() * kernel_reference(rate, Decimal(4))
        return before + (
            pulse if item.contract == "coupled" else amplitude * rho * pulse
        )


def feasible_predictions(item) -> list[Decimal]:
    """Sample independent feasible points; sampling never certifies exclusion."""
    values = []
    for a_text, s_text in product(
        ("0.8", "0.9", "1", "1.1", "1.2"),
        ("0", "0.15", "0.2", "0.25", "0.3"),
    ):
        a, s = Decimal(a_text), Decimal(s_text)
        if q_reference(item, a, s) > Q:
            continue
        for gamma, rho in product((Decimal(0), s / 2, s), (Decimal("0.5"), Decimal(2))):
            values.append(task_reference(item, a, s, gamma, rho))
    assert values, "the test reference must contain at least one feasible point"
    return values


def pair(values) -> tuple[Decimal, Decimal]:
    """Read an ordered exact-decimal bound without binary conversion."""
    assert isinstance(values, list) and len(values) == 2
    assert all(isinstance(value, str) for value in values)
    lo, hi = map(Decimal, values)
    assert lo.is_finite() and hi.is_finite() and lo <= hi
    return lo, hi


def test_frozen_roster_pairs_identical_observations_across_semantic_contracts():
    roster = baseline.problems()
    assert len(roster) == 4
    assert Counter(item.contract for item in roster) == {"coupled": 2, "independent": 2}
    assert sorted(Counter(item.observations for item in roster).values()) == [2, 2]
    assert len({item.case_id for item in roster}) >= 2
    with pytest.raises((FrozenInstanceError, AttributeError)):
        roster[0].contract = "independent"


@pytest.mark.parametrize(
    "observations",
    [("1",) * 3, ("1",) * 5, ("NaN", "1", "1", "1"), (0.1, "1", "1", "1")],
)
def test_problem_rejects_invalid_observation_schema(observations):
    with pytest.raises((ValueError, TypeError)):
        baseline.Problem("bad", observations, "coupled")


def test_problem_rejects_unknown_contract():
    with pytest.raises((ValueError, TypeError)):
        baseline.Problem("bad", ("1",) * 4, "unrestricted")


def test_fixed_gaussian_threshold_has_outward_coverage_above_nominal_level():
    result = baseline.coverage()
    with localcontext(Context(prec=140)):
        reference = 1 - 6 * Decimal(-5).exp()
    assert Decimal("0.95") < result.lo <= reference <= result.hi < 1


@pytest.mark.parametrize("contract", ["coupled", "independent"])
@pytest.mark.parametrize("s_text", ["0", "0.15", "0.2", "0.25", "0.6"])
def test_profile_point_contains_high_precision_conditional_amplitude(contract, s_text):
    item = problem(contract=contract)
    rate = Decimal(s_text)
    expected = amplitude_reference(item, rate)
    result = baseline.profile_amplitude(item, Interval.point(rate))
    if expected is not None:
        assert result is not None
        assert result.lo <= expected[0] <= expected[1] <= result.hi
    elif result is not None:
        # A conservative unresolved interval is permitted; claiming feasibility is not.
        assert result.lo <= result.hi


def test_profile_cell_contains_every_sampled_feasible_conditional_interval():
    item = problem()
    result = baseline.profile_amplitude(item, interval("0.15", "0.25"))
    assert result is not None
    included = 0
    for text in ("0.15", "0.175", "0.2", "0.225", "0.25"):
        expected = amplitude_reference(item, Decimal(text))
        if expected is not None:
            assert result.lo <= expected[0] <= expected[1] <= result.hi
            included += 1
    assert included


def test_profile_respects_a_restricted_amplitude_domain():
    item = problem()
    domain = interval("0.99", "1.01")
    result = baseline.profile_amplitude(item, Interval.point("0.2"), domain)
    assert result is not None
    assert domain.lo <= result.lo <= Decimal(1) <= result.hi <= domain.hi


def test_witness_policy_reaches_both_conditional_amplitude_extremes():
    item, rate = problem(), Decimal("0.2")
    expected = amplitude_reference(item, rate)
    assert expected is not None
    cost = {"boxes": 0, "kernels": 0, "point_tests": 0}
    witnesses = baseline._points_at(item, rate, baseline.A_PRIOR, cost)
    assert len(witnesses) == 3
    amplitudes = [Decimal(witness["A"]) for witness in witnesses]
    assert expected[0] <= min(amplitudes) < max(amplitudes) <= expected[1]
    with localcontext(Context(prec=140)):
        assert min(amplitudes) - expected[0] < Decimal("1e-8")
        assert expected[1] - max(amplitudes) < Decimal("1e-8")
    for witness in witnesses:
        assert q_reference(item, Decimal(witness["A"]), rate) <= Q
        assert pair(witness["q"])[1] <= Q
    assert cost["point_tests"] == 3
    assert cost["kernels"] > 0


@pytest.mark.parametrize("contract", ["coupled", "independent"])
def test_q_and_task_boxes_contain_independent_point_evaluations(contract):
    item = problem(contract=contract)
    a_box, s_box = interval("0.8", "1.2"), interval("0.1", "0.3")
    q_box = baseline.q_value(item, a_box, s_box)
    task_box = baseline.task_bounds(item, a_box, s_box)
    for a, s in product(
        (Decimal("0.8"), Decimal(1), Decimal("1.2")),
        (Decimal("0.1"), Decimal("0.2"), Decimal("0.3")),
    ):
        q = q_reference(item, a, s)
        assert q_box.lo <= q <= q_box.hi
        for gamma, rho in product((Decimal(0), s / 2, s), (Decimal("0.5"), Decimal(2))):
            target = task_reference(item, a, s, gamma, rho)
            assert task_box.lo <= target <= task_box.hi


def test_cartesian_uncertainty_contains_but_is_not_the_coupled_contract():
    a, s = Interval.point(1), Interval.point("0.2")
    coupled = baseline.task_bounds(problem(contract="coupled"), a, s)
    independent = baseline.task_bounds(problem(contract="independent"), a, s)
    assert independent.lo < coupled.lo < coupled.hi < independent.hi


@pytest.mark.parametrize("method", ["profiled", "direct"])
@pytest.mark.parametrize("contract", ["coupled", "independent"])
def test_bounded_search_preserves_feasible_world_predictions(method, contract):
    item = problem(contract=contract)
    state = baseline.advance(item, baseline.initialize(item, method), 64)
    assert 0 < state["cost"]["boxes"] <= 64
    assert state["cost"]["kernels"] > 0
    result = baseline.summarize(item, state)
    bounds = result["bounds"]
    assert bounds is not None
    low, high = pair(bounds["L"]), pair(bounds["U"])
    for prediction in feasible_predictions(item):
        assert low[0] <= prediction <= high[1]
    assert result["individual_rejection"] == {
        "constant_ligand": False,
        "constant_capacity": False,
    }
    risk_lo, risk_hi = pair(result["forecast_risk"])
    assert risk_lo >= 0
    for threshold in TOLERANCES:
        status = result["forecast_decisions"][threshold]
        assert status in {"adequate", "inadequate", "unresolved"}
        if status == "adequate":
            assert risk_hi <= Decimal(threshold)
        elif status == "inadequate":
            assert risk_lo > Decimal(threshold)


@pytest.mark.parametrize("method", ["profiled", "direct"])
def test_chunked_budget_resume_equals_uninterrupted_state_and_keeps_evidence(method):
    item = problem()
    partial = baseline.advance(item, baseline.initialize(item, method), 64)
    saved = json.loads(json.dumps(partial))
    before_summary = json.loads(json.dumps(partial))
    baseline.summarize(item, partial)
    assert partial == before_summary
    resumed = baseline.advance(item, saved, 256)
    uninterrupted = baseline.advance(item, baseline.initialize(item, method), 256)
    assert resumed == uninterrupted
    assert resumed["cost"]["boxes"] <= 256
    assert resumed["cost"]["boxes"] >= partial["cost"]["boxes"]
    old, new = baseline.summarize(item, partial), baseline.summarize(item, resumed)
    assert pair(new["bounds"]["L"])[0] >= pair(old["bounds"]["L"])[0]
    assert pair(new["bounds"]["U"])[1] <= pair(old["bounds"]["U"])[1]


@pytest.mark.parametrize(
    "rate,expected", [("0", "not_established"), ("0.2", "established")]
)
def test_common_zero_face_query_agrees_across_methods(rate, expected):
    item = problem(rate)
    results = [
        baseline.summarize(
            item, baseline.advance(item, baseline.initialize(item, m), 64)
        )
        for m in ("profiled", "direct")
    ]
    assert [result["joint_conflict"] for result in results] == [expected, expected]
    for result in results:
        assert not any(result["individual_rejection"].values())


@pytest.mark.parametrize("method", ["profiled", "direct"])
def test_incompatible_family_does_not_issue_forecast_or_premise_claims(method):
    item = baseline.Problem("outside_envelope", ("10",) * 4, "coupled")
    state = baseline.advance(item, baseline.initialize(item, method), 64)
    result = baseline.summarize(item, state)
    assert result["bounds"] is None
    assert result["individual_rejection"] is None
    assert result["joint_conflict"] != "established"
    assert all(value == "unresolved" for value in result["forecast_decisions"].values())


def test_invalid_method_and_budget_fail_closed():
    item = problem()
    with pytest.raises((ValueError, TypeError)):
        baseline.initialize(item, "local_optimizer")
    for budget in (True, -1, 1.5, "64"):
        with pytest.raises((ValueError, TypeError)):
            baseline.advance(item, baseline.initialize(item, "profiled"), budget)


@pytest.mark.parametrize("method", ["profiled", "direct"])
def test_frontier_keeps_all_independently_verified_feasible_grid_points(method):
    item = problem()
    state = baseline.advance(item, baseline.initialize(item, method), 64)
    frontier = [state["nodes"][index] for index in state["frontier"]]
    included = 0
    for a_text, s_text in product(
        ("0.94", "0.98", "1", "1.02", "1.06"),
        ("0.18", "0.19", "0.2", "0.21", "0.22"),
    ):
        a, s = Decimal(a_text), Decimal(s_text)
        if q_reference(item, a, s) > Q:
            continue
        assert any(
            Decimal(node["A_outer"][0]) <= a <= Decimal(node["A_outer"][1])
            and Decimal(node["s"][0]) <= s <= Decimal(node["s"][1])
            for node in frontier
        )
        included += 1
    assert included > 1


@pytest.mark.parametrize("contract", ["coupled", "independent"])
def test_methods_start_from_identical_feasible_model_candidates_and_preparation(
    contract,
):
    item = problem(contract=contract)
    first, second = [
        baseline.initialize(item, method) for method in ("profiled", "direct")
    ]
    assert first["candidates"] == second["candidates"]
    assert first["preparation_cost"] == second["preparation_cost"]
    assert first["zero_face"] == second["zero_face"]
    assert len(first["candidates"]) == 3
    for model in first["candidates"]:
        a, s, gamma = (Decimal(model[key]) for key in ("A", "s", "gamma"))
        assert Decimal("0.5") <= a <= 2 and 0 <= gamma <= s <= Decimal("0.6")
        assert model["rho"] == ("1/A" if contract == "coupled" else "1.25")
        residual = q_reference(item, a, s)
        lower, upper = pair(model["q"])
        assert lower <= residual <= upper <= Q
        prediction = task_reference(item, a, s, gamma, Decimal("1.25"))
        lower, upper = pair(model["prediction"])
        assert lower <= prediction <= upper


@pytest.mark.parametrize("method", ["profiled", "direct"])
@pytest.mark.parametrize("contract", ["coupled", "independent"])
def test_model_risk_encloses_errors_over_independent_feasible_worlds(method, contract):
    item = problem(contract=contract)
    state = baseline.advance(item, baseline.initialize(item, method), 64)
    result = baseline.summarize(item, state)
    predictions = feasible_predictions(item)
    assert len(result["candidate_reports"]) == 3
    for model in result["candidate_reports"]:
        a, s, gamma = (Decimal(model[key]) for key in ("A", "s", "gamma"))
        prediction = task_reference(item, a, s, gamma, Decimal("1.25"))
        risk_lo, risk_hi = pair(model["risk"])
        with localcontext(Context(prec=140)):
            assert all(
                abs(prediction - possible) <= risk_hi for possible in predictions
            )
        assert risk_lo >= 0
        for threshold in TOLERANCES:
            status = model["decisions"][threshold]
            if status == "adequate":
                assert risk_hi <= Decimal(threshold)
            elif status == "inadequate":
                assert risk_lo > Decimal(threshold)
            else:
                assert status == "unresolved"


@pytest.mark.parametrize("method", ["profiled", "direct"])
def test_inadequacy_certificate_has_two_actual_feasible_endpoint_witnesses(method):
    item = problem(contract="independent")
    state = baseline.advance(item, baseline.initialize(item, method), 64)
    result = baseline.summarize(item, state)
    low, high = state["lower_witness"], state["upper_witness"]
    low_a, low_s = Decimal(low["A"]), Decimal(low["s"])
    high_a, high_s = Decimal(high["A"]), Decimal(high["s"])
    assert q_reference(item, low_a, low_s) <= Q
    assert q_reference(item, high_a, high_s) <= Q
    with localcontext(Context(prec=140)):
        attained_gap = (
            task_reference(item, high_a, high_s, Decimal(0), Decimal(2))
            - task_reference(item, low_a, low_s, low_s, Decimal("0.5"))
        ) / 2
    statuses = result["forecast_decisions"]
    assert "inadequate" in statuses.values()
    for threshold, status in statuses.items():
        if status == "inadequate":
            assert attained_gap > Decimal(threshold)


def test_state_cannot_be_advanced_or_reported_as_another_scientific_contract():
    item = problem(contract="coupled")
    state = baseline.initialize(item, "profiled")
    altered = problem(contract="independent")
    with pytest.raises(ValueError):
        baseline.advance(altered, state, 64)
    with pytest.raises(ValueError):
        baseline.summarize(altered, state)
    changed_protocol = json.loads(json.dumps(state))
    changed_protocol["protocol"] = "another-protocol"
    with pytest.raises(ValueError):
        baseline.advance(item, changed_protocol, 64)
    with pytest.raises(ValueError):
        baseline.summarize(item, changed_protocol)
