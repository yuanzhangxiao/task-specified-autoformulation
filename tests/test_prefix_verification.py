"""Frozen candidate recovery and disjoint, finite verification timing."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_initialization as p
from autoformalism.fitting import prefix_verification as v
from autoformalism.fitting import sensitivity_probe
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicTrajectory
from tests.test_prefix_initialization import problem, row, training


@pytest.fixture(scope="module")
def case():
    model, numerical = problem(), p.NumericalPolicy()
    fit = p.fit_training(model, training(), numerical)
    assert fit["fit_success"]
    heldout = row("recovery", 1.7)
    prefix = p.make_prefix(heldout, p.PrefixPolicy(duration=0.4))
    inference = p.infer_prefix(
        model, fit, prefix, numerical, root_guesses={"init_r_value": 3.5}
    )
    assert inference["fit_success"]
    return model, fit, prefix, numerical, inference, heldout


def failed(inference):
    result = deepcopy(inference)
    result.update(
        fit_success=False,
        parameters=None,
        roots=None,
        realized_initial_states=None,
        verification_calls_started=1,
        verification_calls_completed=0,
        verification_error="sensitivity rollout wall-clock limit reached",
    )
    return result


class Clock:
    def __init__(self):
        self.wall, self.cpu = 100.0, 10.0

    def advance(self, wall, cpu=0.0):
        self.wall += wall
        self.cpu += cpu


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(v, "monotonic", lambda: clock.wall)
    monkeypatch.setattr(v, "process_time", lambda: clock.cpu)
    return clock


def fake_oracles(monkeypatch, clock, *, elapsed=0.25, error=None):
    calls = []

    def prefix(oracle, vector, deadline, *, jacobian, method):
        calls.append((method, deadline, oracle.parameters(vector)))
        clock.advance(elapsed, 0.1)
        if error is not None:
            raise error
        count = len(oracle.rows[0].time)
        return np.zeros(count), np.ones((count, 1)) if jacobian else None

    def forecast(system, row, vector, settings, deadline):
        calls.append((settings.integration_method, deadline, vector.copy()))
        clock.advance(elapsed, 0.1)
        if error is not None:
            raise error
        return (
            np.zeros((len(row.time), len(system.channels))),
            None,
            np.zeros((len(row.time), system.state_count)),
            {"nfev": 1},
        )

    monkeypatch.setattr(p.PrefixOracle, "evaluate", prefix)
    monkeypatch.setattr(p, "symbolic_rollout", forecast)
    return calls


def run(case, kind, execution, checkpoint=None, old=None):
    model, fit, prefix, numerical, inference, heldout = case
    if kind == "inference":
        return v.verify_inference(
            model,
            fit,
            prefix,
            old if old is not None else failed(inference),
            numerical,
            execution,
            checkpoint,
        )
    return v.forecast(
        model,
        fit,
        inference,
        p.make_forecast_schedule(heldout, prefix.policy),
        numerical,
        execution,
        checkpoint,
    )


@pytest.mark.parametrize("field", ["numerical_seconds", "overall_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_policy_requires_positive_finite_allowances(field, value):
    with pytest.raises(ValidationError):
        v.ExecutionPolicy.model_validate({field: value})


def test_real_reverification_preserves_candidate_search_scales_and_rank(
    case, monkeypatch
):
    model, fit, prefix, numerical, inference, heldout = case
    old = failed(inference)
    before = tuple(content_sha256(x) for x in (model, fit, prefix, old))
    monkeypatch.setattr(
        p.recovery_numerics, "search", lambda *a, **k: pytest.fail("new search")
    )
    monkeypatch.setattr(
        p, "score_forecast", lambda *a, **k: pytest.fail("evaluator access")
    )
    result = run(case, "inference", v.ExecutionPolicy(), old=old)
    assert result["fit_success"]
    for name in (
        "roots",
        "parameters",
        "normalized_mse",
        "rank_diagnostic",
        "root_start",
        "shared_parameters",
        "training_scales",
        "realized_initial_states",
        "search",
    ):
        assert result[name] == inference[name]
    assert result["calls"] == result["completed_calls"] == 0
    assert result["no_search_performed"] and result["historical_search_reused"]
    assert result["original_inference_sha256"] == content_sha256(old)
    assert result["verification_calls_started"] == 2
    assert result["verification_calls_completed"] == 2
    assert before == tuple(content_sha256(x) for x in (model, fit, prefix, old))
    schedule = p.make_forecast_schedule(heldout, prefix.policy)
    original = p.forecast(model, fit, result, schedule, numerical)
    current = v.forecast(model, fit, result, schedule, numerical, v.ExecutionPolicy())
    for name in (
        "predictions",
        "states",
        "solver_difference",
        "solver_counters",
        "inference_sha256",
    ):
        assert current[name] == original[name]
    assert current["forecast_success"]
    result["search"]["best"]["parameters"]["init_r_value"] = 100
    assert old["search"]["best"]["parameters"] == inference["roots"]


@pytest.mark.parametrize("kind", ["inference", "forecast"])
def test_slow_checkpoints_do_not_consume_numerical_allowance(
    case, kind, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock)
    snapshots = []

    def checkpoint(value):
        snapshots.append(value)
        clock.advance(2.0, 0.2)

    result = run(
        case,
        kind,
        v.ExecutionPolicy(numerical_seconds=1, overall_seconds=10),
        checkpoint,
    )
    assert result.get("fit_success", result.get("forecast_success"))
    assert [call[0] for call in calls] == ["DOP853", "Radau"]
    # Each absolute deadline is renewed using the unspent active allowance.
    assert calls[0][1] == 103
    assert calls[1][1] == 105
    assert result["verification_calls_attempted"] == 2
    assert result["verification_calls_started"] == 2
    assert result["verification_calls_completed"] == 2
    timing = result["timing"]
    assert timing["elapsed_wall_seconds"] == pytest.approx(6.5)
    assert timing["callback_wall_seconds"] == pytest.approx(6)
    assert timing["numerical_active_wall_seconds"] == pytest.approx(0.5)
    assert timing["numerical_wall_seconds"] == pytest.approx(0.5)
    assert timing["elapsed_cpu_seconds"] == pytest.approx(0.8)
    assert timing["callback_cpu_seconds"] == pytest.approx(0.6)
    assert timing["numerical_cpu_seconds"] == pytest.approx(0.2)
    assert timing["timeout_reason"] is None
    assert snapshots[0]["verification_calls_attempted"] == 1
    assert snapshots[0]["verification_calls_started"] == 0
    assert snapshots[1]["verification_calls_started"] == 1


@pytest.mark.parametrize("kind", ["inference", "forecast"])
@pytest.mark.parametrize("slow_callback", [1, 3])
def test_overall_expiry_during_initial_or_final_callback_cannot_qualify(
    case, kind, slow_callback, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(value)
        if len(saved) == slow_callback:
            clock.advance(6)

    result = run(
        case,
        kind,
        v.ExecutionPolicy(numerical_seconds=1, overall_seconds=5),
        checkpoint,
    )
    assert not result.get("fit_success", result.get("forecast_success"))
    assert len(calls) == (0 if slow_callback == 1 else 2)
    assert result["verification_calls_started"] == len(calls)
    assert result["verification_calls_completed"] == len(calls)
    assert len(saved) == slow_callback
    assert result["timing"]["callback_wall_seconds"] == 6
    assert result["timing"]["timeout_reason"] == "overall_wall_limit"
    assert result.get("roots") is None
    assert not result.get("predictions")


@pytest.mark.parametrize("kind", ["inference", "forecast"])
def test_returned_integration_overrun_stops_before_second_call(
    case, kind, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock, elapsed=2.0)
    result = run(case, kind, v.ExecutionPolicy(numerical_seconds=1))
    assert not result.get("fit_success", result.get("forecast_success"))
    assert len(calls) == 1
    assert result["verification_calls_started"] == 1
    assert result["verification_calls_completed"] == 1
    assert result["verification_calls_failed"] == 0
    assert result["timing"]["numerical_wall_seconds"] == 2
    assert result["timing"]["timeout_reason"] == "numerical_active_wall_limit"


@pytest.mark.parametrize("kind", ["inference", "forecast"])
def test_real_rollout_obeys_fresh_absolute_deadline(case, kind, monkeypatch, clock):
    original = sensitivity_probe.solve_ivp
    calls = []
    monkeypatch.setattr(sensitivity_probe, "monotonic", lambda: clock.wall)

    def delayed(*args, **kwargs):
        calls.append(1)
        clock.advance(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(sensitivity_probe, "solve_ivp", delayed)
    result = run(case, kind, v.ExecutionPolicy(numerical_seconds=1))
    assert not result.get("fit_success", result.get("forecast_success"))
    assert calls == [1]
    assert result["verification_calls_started"] == 1
    assert result["verification_calls_completed"] == 0
    assert result["verification_calls_failed"] == 1
    assert result["timing"]["numerical_wall_seconds"] == 2
    assert result["timing"]["timeout_reason"] == "numerical_active_wall_limit"


@pytest.mark.parametrize("kind", ["inference", "forecast"])
@pytest.mark.parametrize("error_call", [1, 3])
def test_callback_failure_is_charged_without_retry_or_false_qualification(
    case, kind, error_call, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(value)
        clock.advance(0.5, 0.1)
        if len(saved) == error_call:
            raise OSError("storage unavailable")

    result = run(case, kind, v.ExecutionPolicy(), checkpoint)
    assert not result.get("fit_success", result.get("forecast_success"))
    assert len(calls) == (0 if error_call == 1 else 2)
    assert result["verification_calls_started"] == len(calls)
    assert len(saved) == error_call
    assert "storage unavailable" in result["verification_error"]
    assert result["timing"]["callback_failures"] == 1
    assert result["timing"]["callback_wall_seconds"] == error_call * 0.5
    assert result["timing"]["callback_cpu_seconds"] == pytest.approx(error_call * 0.1)


@pytest.mark.parametrize("kind", ["inference", "forecast"])
@pytest.mark.parametrize("where", ["callback", "numerical"])
def test_interrupt_keeps_final_receipt_for_callers_journal(
    case, kind, where, monkeypatch, clock
):
    calls = fake_oracles(
        monkeypatch, clock, error=KeyboardInterrupt() if where == "numerical" else None
    )

    def checkpoint(value):
        if where == "callback":
            clock.advance(0.5, 0.1)
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as caught:
        run(case, kind, v.ExecutionPolicy(), checkpoint)
    receipt = caught.value.verification_receipt
    assert not receipt.get("fit_success", receipt.get("forecast_success"))
    assert receipt["verification_calls_attempted"] == 1
    assert receipt["verification_calls_started"] == len(calls)
    assert receipt["verification_calls_completed"] == 0
    assert receipt["timing"]["callback_failures"] == int(where == "callback")
    assert receipt["verification_calls_failed"] == int(where == "numerical")
    assert receipt["timing"]["elapsed_wall_seconds"] == (
        0.5 if where == "callback" else 0.25
    )


@pytest.mark.parametrize(
    "field",
    [
        "fit_success",
        "fit_sha256",
        "prefix_sha256",
        "boundary_sha256",
        "prefix_forcing_sha256",
        "cutoff",
        "root_start",
        "parameters",
        "shared_parameters",
        "training_scales",
        "policy",
        "best",
        "best_parameters",
        "best_loss",
    ],
)
def test_invalid_saved_inference_rejected_before_any_call(case, field, monkeypatch):
    inference = failed(case[4])
    if field == "fit_success":
        inference[field] = True
    elif field.endswith("sha256"):
        inference[field] = "a" * 64
    elif field in ("root_start", "shared_parameters", "parameters"):
        inference[field] = {"theta": 1}
    elif field == "training_scales":
        inference[field] = {"y": 100}
    elif field == "policy":
        inference[field]["rtol"] *= 2
    elif field == "best":
        inference["search"]["best"] = None
    elif field == "best_parameters":
        inference["search"]["best"]["parameters"] = {"theta": 1}
    elif field == "best_loss":
        inference["search"]["best"]["training_nmse"] = float("nan")
    else:
        inference[field] += 1
    monkeypatch.setattr(
        p.PrefixOracle,
        "evaluate",
        lambda *a, **k: pytest.fail("oracle before validation"),
    )
    with pytest.raises(ValueError):
        run(case, "inference", v.ExecutionPolicy(), old=inference)


def test_saved_truncated_search_endpoint_is_never_refined(case, monkeypatch):
    old = failed(case[4])
    old["search"]["best"]["parameters"]["init_r_value"] = 0.6505102849
    old["search"]["stop_reason"] = "wall_clock_limit"
    monkeypatch.setattr(
        p.recovery_numerics, "search", lambda *a, **k: pytest.fail("refitted candidate")
    )
    result = run(case, "inference", v.ExecutionPolicy(), old=old)
    assert result["fit_success"]
    assert result["roots"] == {"init_r_value": 0.6505102849}
    assert result["search"] == old["search"]
    assert result["normalized_mse"] > 0.01


def test_suffix_poison_and_evaluator_functions_cannot_enter_verification(
    case, monkeypatch
):
    model, fit, prefix, numerical, inference, heldout = case
    payload = heldout.model_dump(mode="json")
    payload["targets"]["y"][5:] = [9000] * 36
    poisoned = PublicTrajectory.model_validate(payload)
    assert p.make_prefix(poisoned, prefix.policy) == prefix
    assert p.make_forecast_schedule(
        poisoned, prefix.policy
    ) == p.make_forecast_schedule(heldout, prefix.policy)
    monkeypatch.setattr(
        p, "score_forecast", lambda *a, **k: pytest.fail("evaluator access")
    )
    result = v.verify_inference(
        model,
        fit,
        p.make_prefix(poisoned, prefix.policy),
        failed(inference),
        numerical,
        v.ExecutionPolicy(),
    )
    assert result["roots"] == inference["roots"]
    forecast = v.forecast(
        model,
        fit,
        result,
        p.make_forecast_schedule(poisoned, prefix.policy),
        numerical,
        v.ExecutionPolicy(),
    )
    assert forecast["forecast_success"]
    assert max(forecast["predictions"]["y"]) < 10


@pytest.mark.parametrize("kind", ["inference", "forecast"])
def test_overall_cap_also_limits_integration_when_numerical_budget_remains(
    case, kind, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock, elapsed=2)
    result = run(case, kind, v.ExecutionPolicy(numerical_seconds=10, overall_seconds=1))
    assert not result.get("fit_success", result.get("forecast_success"))
    assert len(calls) == 1 and calls[0][1] == 101
    assert result["timing"]["timeout_reason"] == "overall_wall_limit"
    assert result["timing"]["numerical_active_wall_seconds"] == 2


@pytest.mark.parametrize("error_call", [1, 3])
def test_failed_callback_still_reports_overall_expiry(
    case, error_call, monkeypatch, clock
):
    fake_oracles(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(value)
        if len(saved) == error_call:
            clock.advance(61)
            raise OSError("blocked write failed")

    result = run(case, "inference", v.ExecutionPolicy(), checkpoint)
    assert not result["fit_success"]
    assert result["verification_calls_started"] == (0 if error_call == 1 else 2)
    assert len(saved) == error_call
    assert result["timing"]["timeout_reason"] == "overall_wall_limit"
    assert result["timing"]["callback_wall_seconds"] == 61
    assert "blocked write failed" in result["verification_error"]


def test_independent_integrator_disagreement_stays_failed(case, monkeypatch, clock):
    def evaluate(oracle, vector, deadline, *, jacobian, method):
        clock.advance(0.1)
        count = len(oracle.rows[0].time)
        return (
            np.zeros(count) if method == "DOP853" else np.ones(count),
            np.ones((count, 1)) if jacobian else None,
        )

    monkeypatch.setattr(p.PrefixOracle, "evaluate", evaluate)
    result = run(case, "inference", v.ExecutionPolicy())
    assert not result["fit_success"]
    assert result["roots"] is result["parameters"] is None
    assert result["verification_calls_completed"] == 2
    assert result["solver_difference"] == 1
    assert "disagree" in result["verification_error"]


@pytest.mark.parametrize("kind", ["inference", "forecast"])
def test_numerical_exception_has_one_failed_call_and_full_timing(
    case, kind, monkeypatch, clock
):
    calls = fake_oracles(monkeypatch, clock, error=ArithmeticError("invalid solver"))
    result = run(case, kind, v.ExecutionPolicy())
    assert not result.get("fit_success", result.get("forecast_success"))
    assert len(calls) == 1
    assert result["verification_calls_failed"] == 1
    assert result["verification_calls_started"] == 1
    assert result["verification_calls_completed"] == 0
    assert result["timing"]["numerical_wall_seconds"] == 0.25
    assert result["timing"]["numerical_cpu_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize("field", ["root_start", "best_parameters"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_saved_coordinates_rejected_before_oracle(
    case, field, value, monkeypatch
):
    old = failed(case[4])
    target = (
        old["root_start"]
        if field == "root_start"
        else old["search"]["best"]["parameters"]
    )
    target["init_r_value"] = value
    monkeypatch.setattr(
        p.PrefixOracle,
        "evaluate",
        lambda *a, **k: pytest.fail("invalid coordinate reached oracle"),
    )
    with pytest.raises(ValueError, match="domain"):
        run(case, "inference", v.ExecutionPolicy(), old=old)


@pytest.mark.parametrize(
    "field",
    ["fit_sha256", "boundary_sha256", "prefix_forcing_sha256", "roots", "policy"],
)
def test_forecast_rejects_changed_boundaries_roots_and_policy(case, field, monkeypatch):
    modified = list(case)
    modified[4] = deepcopy(case[4])
    if field == "roots":
        modified[4][field] = {"theta": 1}
    elif field == "policy":
        modified[4][field]["rtol"] *= 2
    else:
        modified[4][field] = "a" * 64
    monkeypatch.setattr(
        p,
        "symbolic_rollout",
        lambda *a, **k: pytest.fail("invalid forecast reached oracle"),
    )
    with pytest.raises(ValueError):
        run(modified, "forecast", v.ExecutionPolicy())
