"""Residual-only controls preserve proof domains, useful points and honest costs."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import prefix_portfolio_residual_search as p
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from tests.test_compact_search_checkpoint import fake_clock
from tests.test_prefix_portfolio_nonlinear_reduction import build, case


def quadratic(monkeypatch, *, source=None):
    original, oracle = build(source or case())
    seed = oracle.vector(oracle.start)
    target = np.clip(seed + 0.37, oracle.lower, oracle.upper)
    seen = []

    def evaluate(point, deadline, *, jacobian=True, method=None):
        assert jacobian is False
        assert np.isfinite(deadline)
        oracle.vector(oracle.parameters(point))
        original.vector(original.parameters(oracle.reduction.lift(point)))
        seen.append(np.asarray(point).copy())
        return np.asarray(point) - target, None

    monkeypatch.setattr(oracle, "evaluate", evaluate)
    return original, oracle, seed, target, seen


def test_qualification_uses_two_fresh_residuals_and_independent_original_radau(
    monkeypatch,
):
    original, oracle = build(case())
    saved, reduced_calls, original_calls = [], [], []
    raw, direct = oracle._call, original.evaluate

    def reduced(*args, **kwargs):
        reduced_calls.append(kwargs)
        return raw(*args, **kwargs)

    def independent(*args, **kwargs):
        original_calls.append(kwargs)
        return direct(*args, **kwargs)

    monkeypatch.setattr(oracle, "_call", reduced)
    monkeypatch.setattr(original, "evaluate", independent)
    result = p.qualify_residual(original, oracle, oracle.start, 10, saved.append)
    assert result["passed"] and result["original_equations_verified"]
    assert reduced_calls == [{"jacobian": False, "method": "BDF"}] * 2
    assert original_calls == [{"jacobian": False, "method": "Radau"}]
    assert (
        result["verification_calls_started"]
        == result["verification_calls_completed"]
        == 3
    )
    assert result["verification_calls_failed"] == 0
    assert result["stage_backend_calls"]["residual_solves_completed"] == 2
    assert result["stage_backend_calls"]["jacobian_solves_started"] == 0
    assert result["repeat_prediction_discrepancy"] == 0
    assert (
        not result["gradient_qualified"]
        and not result["parameter_information_qualified"]
    )
    assert saved[-1]["passed"]


def test_qualification_rejects_changed_certificate_before_numerical_calls():
    original, oracle = build(case())
    oracle.portfolio_audit["reduction"]["proof"]["zero_product_branch_preserved"] = (
        False
    )
    result = p.qualify_residual(original, oracle, oracle.start, 10, lambda _: None)
    assert not result["passed"]
    assert result["verification_calls_started"] == 0
    assert "audit differs" in result["error"]


def test_real_ode_search_smoke_never_integrates_parameter_sensitivities(monkeypatch):
    _, oracle = build(case())
    actual = oracle.oracle.evaluate
    requested = []

    def checked(*args, **kwargs):
        requested.append(kwargs["jacobian"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(oracle.oracle, "evaluate", checked)
    result = p.search(oracle, oracle.start, 10, 3, lambda _: None)
    assert result["best"] is not None
    assert requested == [False, False, False]
    assert result["stage_backend_calls"]["residual_solves_completed"] == 3
    assert result["stage_backend_calls"]["jacobian_solves_started"] == 0


def test_qualification_interruption_keeps_completed_replay_cost(monkeypatch):
    original, oracle = build(case())
    actual = oracle._call
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return actual(*args, **kwargs)

    monkeypatch.setattr(oracle, "_call", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        p.qualify_residual(original, oracle, oracle.start, 10, lambda _: None)
    result = caught.value.execution_receipt
    assert not result["passed"] and not result["original_equations_verified"]
    assert result["verification_calls_started"] == 2
    assert result["verification_calls_completed"] == 1
    assert result["verification_calls_failed"] == 1
    assert len(result["replays"]) == 1


@pytest.mark.parametrize("which", ["repeat", "original"])
def test_residual_disagreement_withholds_qualification(monkeypatch, which):
    original, oracle = build(case())
    owner, attr = (oracle, "_call") if which == "repeat" else (original, "evaluate")
    actual = getattr(owner, attr)
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        r, j = actual(*args, **kwargs)
        return r + (1e-3 if which == "original" or calls == 2 else 0), j

    monkeypatch.setattr(owner, attr, changed)
    result = p.qualify_residual(original, oracle, oracle.start, 10, lambda _: None)
    assert not result["passed"]
    assert result["verification_calls_completed"] == 3
    assert not result["gradient_qualified"]


def test_powell_quadratic_smoke_uses_no_jacobian_and_preserves_history(monkeypatch):
    _, oracle, seed, target, seen = quadratic(monkeypatch)
    records = []
    result = p.search(oracle, oracle.start, 10, 500, records.append)
    assert result["best"]["training_nmse"] < 1e-15
    assert oracle.vector(result["best"]["parameters"]) == pytest.approx(
        target, abs=1e-7
    )
    assert result["initial_sweep_completed"]
    assert len(result["initial_sweep_coordinates_completed"]) == len(oracle.names)
    assert result["accepted_history"][0]["parameters"] == oracle.start
    assert result["last_accepted"] is not None
    assert (
        result["search_calls_started"] == result["search_calls_completed"] == len(seen)
    )
    assert result["search_calls_failed"] == 0
    assert not result["derivatives_used"] and not result["convergence_certified"]
    assert records[-1]["best"] == result["best"]
    assert oracle.vector(oracle.start) == pytest.approx(seed)


def test_balanced_seed_sweep_has_fixed_scales_and_original_unbounded_domains(
    monkeypatch,
):
    _, oracle, seed, _, seen = quadratic(monkeypatch)
    captured = {}

    def stop_after_sweep(objective, q, **kwargs):
        captured.update(kwargs)
        raise TimeoutError("bounded synthetic stop")

    monkeypatch.setattr(p, "minimize", stop_after_sweep)
    result = p.search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["initial_sweep_completed"]
    units = np.maximum(abs(seed), oracle.units)
    expected = [seed]
    for j in range(len(seed)):
        for sign in (-1, 1):
            q = np.clip(
                sign,
                (oracle.lower[j] - seed[j]) / units[j],
                (oracle.upper[j] - seed[j]) / units[j],
            )
            if q:
                point = seed.copy()
                point[j] += q * units[j]
                expected.append(point)
    assert np.asarray(seen) == pytest.approx(np.asarray(expected))
    assert any(
        not np.isfinite(lo) or not np.isfinite(hi) for lo, hi in captured["bounds"]
    )
    assert not result["coordinate_chart"]["artificial_bounds_added"]


def test_original_finite_bounds_are_respected_by_every_trial(monkeypatch):
    source = case(
        amplitude={"bounds": {"lower": 0, "upper": 2}},
        gain={"bounds": {"lower": 0, "upper": 3}},
    )
    _, oracle, _, _, seen = quadratic(monkeypatch, source=source)
    result = p.search(oracle, oracle.start, 10, 80, lambda _: None)
    assert result["best"] is not None
    assert all(np.all(x >= oracle.lower) and np.all(x <= oracle.upper) for x in seen)


def test_fixed_coordinate_is_overlaid_and_excluded_from_all_search_directions(
    monkeypatch,
):
    _, oracle, seed, _, seen = quadratic(monkeypatch)
    name = oracle.names[0]
    held = {name: 0.0}
    original_start = deepcopy(oracle.start)
    result = p.search(oracle, oracle.start, 10, 500, lambda _: None, fixed=held)
    assert result["fixed_coordinates"] == held
    assert result["optimizer_start"] == original_start | held
    assert all(point[0] == 0 for point in seen)
    assert result["best"]["parameters"][name] == 0
    assert result["initial_sweep_completed"]
    for key in (
        "free_coordinates",
        "coordinates_varied",
        "initial_sweep_coordinates_completed",
    ):
        assert name not in result[key]
    assert all(row["parameters"][name] == 0 for row in result["accepted_history"])
    assert oracle.start == original_start
    assert oracle.vector(oracle.start) == pytest.approx(seed)
    assert held == {name: 0.0}


def test_all_fixed_coordinates_evaluate_once_without_calling_optimizer(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)

    def unexpected(*args, **kwargs):
        pytest.fail("no optimizer needed when every coordinate is fixed")

    monkeypatch.setattr(p, "minimize", unexpected)
    held = dict.fromkeys(oracle.names, 0.0)
    result = p.search(oracle, oracle.start, 10, 10, lambda _: None, fixed=held)
    assert result["stop_reason"] == "no_free_coordinates"
    assert result["optimizer_success"]
    assert not result["convergence_certified"]
    assert result["search_calls_attempted"] == len(seen) == 1
    assert result["best"]["parameters"] == held
    assert result["free_coordinates"] == []
    assert result["coordinates_varied"] == []


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "0"])
def test_invalid_fixed_values_fail_before_numerical_work(monkeypatch, value):
    _, oracle, _, _, seen = quadratic(monkeypatch)
    result = p.search(
        oracle, oracle.start, 10, 10, lambda _: None,
        fixed={oracle.names[0]: value},
    )
    assert "finite numeric" in result["error"]
    assert result["best"] is None and not seen
    assert result["search_calls_attempted"] == 0


def test_unknown_fixed_coordinate_fails_before_numerical_work(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)
    result = p.search(
        oracle, oracle.start, 10, 10, lambda _: None, fixed={"unknown": 1.0}
    )
    assert "must exist" in result["error"]
    assert result["best"] is None and not seen


def test_fixed_value_outside_original_domain_is_not_clipped(monkeypatch):
    source = case(amplitude={"bounds": {"lower": -2, "upper": 2}})
    _, oracle, _, _, seen = quadratic(monkeypatch, source=source)
    result = p.search(
        oracle, oracle.start, 10, 10, lambda _: None, fixed={"a": 3.0}
    )
    assert "domain" in result["error"]
    assert result["best"] is None and not seen


def test_fixed_coordinates_require_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        p.search(None, {}, 10, 10, lambda _: None, fixed=[("a", 1.0)])


def test_invalid_numerical_trials_are_bounded_and_do_not_erase_best(monkeypatch):
    _, oracle, seed, _, _ = quadratic(monkeypatch)
    actual = oracle.evaluate
    calls = 0

    def unstable(point, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise NumericalInvalidTrial(f"unstable integration sample {calls}")
        return actual(point, *args, **kwargs)

    monkeypatch.setattr(oracle, "evaluate", unstable)

    def keep_requesting(objective, q, **kwargs):
        while True:
            objective(q)

    monkeypatch.setattr(p, "minimize", keep_requesting)
    result = p.search(oracle, oracle.start, 10, 30, lambda _: None)
    assert oracle.vector(result["best"]["parameters"]) == pytest.approx(seed)
    assert result["invalid_trials"] == 29
    assert result["search_calls_failed"] == 29
    assert len(result["invalid_trial_examples"]) == p.MAX_FAILURE_EXAMPLES
    assert len(result["invalid_trial_histogram"]) <= p.MAX_FAILURE_EXAMPLES + 1
    assert sum(result["invalid_trial_histogram"].values()) == 29
    assert result["budget_exhausted"]


def test_out_of_domain_optimizer_request_never_reaches_oracle(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)

    def invalid_trial(objective, q, **kwargs):
        before = len(seen)
        bad = q.copy()
        bad[0] = np.inf
        assert objective(bad) == np.inf
        assert len(seen) == before
        return SimpleNamespace(success=False, message="synthetic stop", nit=0)

    monkeypatch.setattr(p, "minimize", invalid_trial)
    result = p.search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["invalid_trials"] == 1
    assert result["search_calls_attempted"] == result["search_calls_started"] + 1
    assert result["search_calls_failed"] == 0


def test_call_limit_includes_seed_and_preserves_best(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)
    result = p.search(oracle, oracle.start, 10, 3, lambda _: None)
    assert (
        result["search_calls_attempted"]
        == result["search_calls_started"]
        == len(seen)
        == 3
    )
    assert result["stop_reason"] == "call_limit" and result["budget_exhausted"]
    assert result["best"] is not None and not result["initial_sweep_completed"]


def test_deadline_after_return_cannot_promote_late_trial(monkeypatch):
    _, oracle, seed, _, _ = quadratic(monkeypatch)
    budget = p.execution._budget(10, 130)
    monkeypatch.setattr(p.execution, "_budget", lambda *_: budget)
    actual = oracle.evaluate
    calls = 0

    def late(point, *args, **kwargs):
        nonlocal calls
        calls += 1
        value = actual(point, *args, **kwargs)
        if calls == 2:
            budget.wall -= 20
        return value

    monkeypatch.setattr(oracle, "evaluate", late)
    result = p.search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["budget_exhausted"]
    assert result["search_calls_completed"] == 2
    assert oracle.vector(result["best"]["parameters"]) == pytest.approx(seed)


def test_interrupt_keeps_last_finite_checkpoint_and_does_not_resume(monkeypatch):
    _, oracle, seed, _, _ = quadratic(monkeypatch)
    actual = oracle.evaluate
    calls, snapshots = 0, []

    def interrupted(point, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return actual(point, *args, **kwargs)

    monkeypatch.setattr(oracle, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        p.search(oracle, oracle.start, 10, 100, snapshots.append)
    result = caught.value.execution_receipt
    assert result["stop_reason"] == "interrupted"
    assert result["search_calls_started"] == 2 and result["search_calls_failed"] == 1
    assert oracle.vector(result["best"]["parameters"]) == pytest.approx(seed)
    assert any(x["best"] is not None for x in snapshots)


def test_failed_checkpoint_blocks_further_numerical_work(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)
    writes = []

    def fail_after_best(result):
        writes.append(deepcopy(result))
        if result["best"] is not None:
            raise OSError("disk unavailable")

    result = p.search(oracle, oracle.start, 10, 100, fail_after_best)
    assert len(seen) == 1 and len(writes) == 2
    assert result["best"] is not None
    assert result["timing"]["callback_failures"] == 1
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "seconds,calls", [(0, 1), (1801, 1), (1, 0), (1, True), (1, 2001)]
)
def test_invalid_allowances_rejected(seconds, calls):
    with pytest.raises(ValueError, match="residual search requires"):
        p.search(None, {}, seconds, calls, lambda _: None)


@pytest.mark.parametrize("seconds", [5, 1800])
def test_compact_slow_sink_preserves_objective_sequence_and_best(monkeypatch, seconds):
    def repeat_point(objective, q, *, callback, **kwargs):
        for _ in range(100):
            objective(q)
        callback(q)
        return SimpleNamespace(success=True, message="synthetic stop", nit=1)

    monkeypatch.setattr(p, "minimize", repeat_point)
    outcomes = []
    for compact in (False, True):
        clock = fake_clock(monkeypatch)
        _, oracle, _, _, seen = quadratic(monkeypatch)
        snapshots = []

        def slow_sink(snapshot, snapshots=snapshots, clock=clock):
            snapshots.append(deepcopy(snapshot))
            clock.advance(0.7)

        kwargs = {"checkpoint_policy": p.SearchCheckpointPolicy()} if compact else {}
        result = p.search(oracle, oracle.start, seconds, 500, slow_sink, **kwargs)
        outcomes.append((result, snapshots, seen))
    baseline, compact = outcomes
    assert compact[0]["best"] == baseline[0]["best"]
    assert np.asarray(compact[2][: len(baseline[2])]) == pytest.approx(
        np.asarray(baseline[2])
    )
    if seconds == 1800:
        assert compact[0]["search_calls_completed"] == (
            baseline[0]["search_calls_completed"]
        )
    else:
        assert baseline[0]["timing"]["timeout_reason"] == "overall_wall_limit"
        assert compact[0]["timing"]["timeout_reason"] is None
        assert compact[0]["search_calls_completed"] > (
            baseline[0]["search_calls_completed"]
        )
        assert compact[0]["optimizer_success"]
    assert len(compact[1]) < len(baseline[1]) / 5
    assert compact[1][-1]["best"] == compact[0]["best"]
    assert compact[0]["timing"]["callback_wall_seconds"] < (
        baseline[0]["timing"]["callback_wall_seconds"] / 5
    )
    assert compact[0]["checkpoint_persistence"]["writes_completed"] == len(compact[1])
    assert "certificate" in compact[0] and "certificate" not in compact[1][-1]


def test_compact_interruption_keeps_best_in_durable_progress(monkeypatch):
    _, oracle, seed, _, _ = quadratic(monkeypatch)
    actual = oracle.evaluate
    calls, snapshots = 0, []

    def interrupted(point, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return actual(point, *args, **kwargs)

    monkeypatch.setattr(oracle, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        p.search(
            oracle, oracle.start, 10, 100, snapshots.append,
            checkpoint_policy=p.SearchCheckpointPolicy(),
        )
    result = caught.value.execution_receipt
    assert result["search_calls_started"] == 2
    assert snapshots[-1]["best"] == result["best"]
    assert oracle.vector(result["best"]["parameters"]) == pytest.approx(seed)
    assert snapshots[-1]["compact_progress"]["event"] == "accepted_iteration"


def test_compact_failed_checkpoint_prevents_next_numerical_call(monkeypatch):
    _, oracle, _, _, seen = quadratic(monkeypatch)

    def fail_after_best(snapshot):
        if snapshot["best"] is not None:
            raise OSError("disk unavailable")

    result = p.search(
        oracle, oracle.start, 10, 100, fail_after_best,
        checkpoint_policy=p.SearchCheckpointPolicy(),
    )
    assert len(seen) == 1 and result["best"] is not None
    assert result["checkpoint_persistence"]["writes_failed"] == 1
    assert result["status"] == "incomplete"
