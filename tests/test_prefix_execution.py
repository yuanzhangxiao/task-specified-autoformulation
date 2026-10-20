"""Opt-in search timing preserves persistence costs and historical fit semantics."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import prefix_execution as e
from autoformalism.fitting import prefix_initialization as p
from autoformalism.fitting import prefix_verification as v
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def problem():
    """Two observable independent roots with distinct fixed decay rates."""
    return p.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "execution_two_roots",
                "parent_candidate_id": None,
                "states": [
                    {"name": name, "kind": "latent"} for name in ("x", "z1", "z2")
                ],
                "state_equations": [
                    {"state": "x", "rhs": "z1+z2"},
                    {"state": "z1", "rhs": "-z1"},
                    {"state": "z2", "rhs": "-2*z2"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z1", "z2")
                ],
            },
            "context": {"targets": ["y"]},
            "graph": {
                "rules": {
                    name: {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory",
                            "guess": value,
                        }
                    }
                    for name, value in (("z1", 0.4), ("z2", 0.6))
                }
            },
        }
    )


def row(name, first=1.1, second=0.8):
    """Closed-form public output; no latent trajectories enter the fitter."""
    time = np.linspace(0, 3, 16)
    values = 1 + first * (1 - np.exp(-time)) + second * (1 - np.exp(-2 * time)) / 2
    return PublicTrajectory(
        trajectory_id=name, time=tuple(time), targets={"y": tuple(values)}
    )


@pytest.fixture(scope="module")
def case():
    model, numerical = problem(), p.NumericalPolicy(seconds=30)
    data = PublicSplit(
        name="train", fingerprint="two-root-execution", rows=(row("a"), row("b", 2, 1))
    )
    fit = e.fit_training(model, data, numerical, e.SearchExecutionPolicy())
    assert fit["fit_success"], fit
    prefix = p.make_prefix(row("new", 1.6, 1.3), p.PrefixPolicy(duration=1))
    return model, data, fit, prefix, numerical


class Clock:
    def __init__(self):
        self.wall, self.cpu = 100.0, 10.0

    def advance(self, wall, cpu=0.0):
        self.wall += wall
        self.cpu += cpu


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    for module in (e, v, e.recovery_numerics):
        monkeypatch.setattr(module, "monotonic", lambda: value.wall)
    for module in (e, v):
        monkeypatch.setattr(module, "process_time", lambda: value.cpu)
    return value


def fake_oracle(monkeypatch, clock, *, elapsed=0.1, error=None):
    calls = []

    def evaluate(oracle, vector, deadline, *, jacobian=True, method=None):
        calls.append((method, deadline, vector.copy()))
        clock.advance(elapsed, 0.05)
        if error is not None:
            raise error
        return vector - np.asarray([1.1, 0.8]), np.eye(2) if jacobian else None

    monkeypatch.setattr(p.PrefixOracle, "evaluate", evaluate)
    return calls


def run(case, execution=None, checkpoint=None, *, numerical=None, guesses=None):
    model, _, fit, prefix, original = case
    return e.infer_prefix(
        model,
        fit,
        prefix,
        numerical or original.model_copy(update={"seconds": 1, "maximum_calls": 1}),
        execution or e.SearchExecutionPolicy(),
        checkpoint,
        root_guesses=guesses,
    )


@pytest.mark.parametrize(
    "field", ["search_overall_seconds", "verification_overall_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_execution_requires_positive_finite_allowances(field, value):
    with pytest.raises(ValidationError):
        e.SearchExecutionPolicy.model_validate({field: value})


def test_two_independent_roots_fit_and_forecast_without_mutating_inputs(case):
    model, data, fit, prefix, numerical = case
    before = tuple(content_sha256(value) for value in (model, data, fit, prefix))
    guesses = {"init_z1_value": 2.4, "init_z2_value": 2.6}
    inferred = run(case, numerical=numerical, guesses=guesses)
    assert inferred["fit_success"], inferred
    assert inferred["roots"] == pytest.approx(
        {"init_z1_value": 1.6, "init_z2_value": 1.3}, abs=1e-6
    )
    assert inferred["rank_diagnostic"]["rank"] == 2
    assert inferred["rank_diagnostic"]["full_column_rank"]
    assert fit["rank_diagnostic"]["rank"] == 4
    assert fit["training_roots"]["b"] == pytest.approx(
        {"init_z1_value": 2.0, "init_z2_value": 1.0}, abs=1e-6
    )
    assert inferred["verification_calls_started"] == 2
    assert inferred["verification_calls_completed"] == 2
    assert inferred["search_calls_started"] == inferred["search_calls_completed"]
    assert inferred["root_start"] == guesses
    guesses["init_z1_value"] = 99
    assert inferred["root_start"]["init_z1_value"] == 2.4
    forecast = p.forecast(
        model,
        fit,
        inferred,
        p.make_forecast_schedule(row("new", 1.6, 1.3), prefix.policy),
        numerical,
    )
    assert forecast["forecast_success"]
    assert before == tuple(
        content_sha256(value) for value in (model, data, fit, prefix)
    )
    assert fit["overall_timing"]["elapsed_wall_seconds"] >= (
        fit["search_timing"]["elapsed_wall_seconds"]
        + fit["verification_timing"]["elapsed_wall_seconds"]
    )


def test_historical_kernels_have_unchanged_receipts_and_values(case):
    model, data, fit, prefix, numerical = case
    legacy = p.fit_training(model, data, numerical)
    assert "execution_policy" not in legacy and "search_timing" not in legacy
    assert legacy["parameters"] == pytest.approx(fit["parameters"], abs=1e-10)
    assert legacy["normalized_mse"] == pytest.approx(fit["normalized_mse"], abs=1e-15)
    current = e.infer_prefix(model, fit, prefix, numerical, e.SearchExecutionPolicy())
    old = p.infer_prefix(model, fit, prefix, numerical)
    assert "root_start" not in old and "execution_protocol" not in old
    assert old["roots"] == pytest.approx(current["roots"], abs=1e-10)


def test_slow_checkpoints_are_excluded_from_both_numerical_allowances(
    case, monkeypatch, clock
):
    calls = fake_oracle(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(deepcopy(value))
        clock.advance(2, 0.2)

    result = run(case, checkpoint=checkpoint)
    assert result["fit_success"], result
    assert [call[0] for call in calls] == [None, "DOP853", "Radau"]
    assert calls[0][1] == pytest.approx(103)
    assert result["calls"] == result["search_calls_started"] == 1
    assert result["search_calls_attempted"] == result["search"]["calls"] == 1
    assert result["search_timing"]["numerical_active_wall_seconds"] == pytest.approx(
        0.1
    )
    assert result["verification_timing"]["numerical_active_wall_seconds"] == (
        pytest.approx(0.2)
    )
    assert result["search_timing"]["callback_wall_seconds"] > 1
    assert result["verification_timing"]["callback_wall_seconds"] == 6
    assert saved[0]["search_calls_attempted"] == 1
    assert saved[0]["search_calls_started"] == 0
    assert result["overall_timing"]["elapsed_wall_seconds"] == pytest.approx(
        0.3 + 2 * len(saved)
    )


@pytest.mark.parametrize("stage", ["search", "verification"])
def test_overall_expiry_in_callback_prevents_numerical_work(
    case, stage, monkeypatch, clock
):
    calls = fake_oracle(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(value)
        selected = value["verification_calls_attempted"] > 0
        if selected == (stage == "verification"):
            clock.advance(61)

    result = run(
        case,
        e.SearchExecutionPolicy(
            search_overall_seconds=60, verification_overall_seconds=60
        ),
        checkpoint,
    )
    assert not result["fit_success"]
    assert result["parameters"] is result["roots"] is None
    assert len(calls) == (0 if stage == "search" else 1)
    assert result[f"{stage}_timing"]["timeout_reason"] == "overall_wall_limit"
    assert result["verification_calls_started"] == 0


@pytest.mark.parametrize("stage", ["search", "verification", "final"])
def test_checkpoint_failure_stops_without_retry_and_invalidates_final_result(
    case, stage, monkeypatch, clock
):
    calls = fake_oracle(monkeypatch, clock)
    saved = []

    def checkpoint(value):
        saved.append(value)
        fail = (
            (stage == "search")
            or (stage == "verification" and value["verification_calls_attempted"] == 1)
            or (stage == "final" and value["fit_success"])
        )
        if fail:
            clock.advance(0.25)
            raise OSError("write unavailable")

    result = run(case, checkpoint=checkpoint)
    assert not result["fit_success"]
    assert result["parameters"] is result["roots"] is None
    assert len(calls) == {"search": 0, "verification": 1, "final": 3}[stage]
    assert "write unavailable" in result["verification_error"]
    assert saved[-1]["fit_success"] == (stage == "final")
    budget = result["search_timing" if stage == "search" else "verification_timing"]
    assert budget["callback_failures"] == 1
    assert budget["callback_wall_seconds"] == pytest.approx(0.25)


def test_numerical_search_expiry_verifies_previously_completed_best(
    case, monkeypatch, clock
):
    calls = fake_oracle(monkeypatch, clock, elapsed=0.3)
    numerical = case[-1].model_copy(
        update={"seconds": 0.5, "verification_seconds": 1, "maximum_calls": 20}
    )
    result = run(case, numerical=numerical)
    assert result["fit_success"]
    assert result["search"]["stop_reason"] == "budget_limited"
    assert result["search_timing"]["timeout_reason"] == "numerical_active_wall_limit"
    assert result["verification_timing"]["timeout_reason"] is None
    assert [call[0] for call in calls] == [None, None, "DOP853", "Radau"]
    assert result["search_calls_started"] == result["search_calls_completed"] == 2
    assert result["search"]["completed_calls"] == 1
    assert result["parameters"] == {"init_z1_value": 0.4, "init_z2_value": 0.6}


def test_first_search_call_overrun_cannot_create_success(case, monkeypatch, clock):
    calls = fake_oracle(monkeypatch, clock, elapsed=2)
    result = run(case)
    assert not result["fit_success"]
    assert result["search"]["best"] is None
    assert len(calls) == 1
    assert result["search_calls_completed"] == 1
    assert result["search"]["completed_calls"] == 0
    assert result["verification_calls_started"] == 0


@pytest.mark.parametrize("where", ["search_callback", "search_call", "verification"])
def test_interrupt_retains_started_counts_and_full_lower_bound_receipt(
    case, where, monkeypatch, clock
):
    calls = fake_oracle(
        monkeypatch,
        clock,
        error=KeyboardInterrupt() if where == "search_call" else None,
    )

    def checkpoint(value):
        if where == "search_callback" or (
            where == "verification" and value["verification_calls_attempted"]
        ):
            clock.advance(0.2)
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as caught:
        run(case, checkpoint=checkpoint)
    receipt = caught.value.execution_receipt
    assert not receipt["fit_success"]
    assert receipt["parameters"] is receipt["roots"] is None
    assert receipt["search_calls_attempted"] == 1
    assert receipt["search_calls_started"] == len(calls)
    assert receipt["search_calls_failed"] == int(where == "search_call")
    assert receipt["verification_calls_started"] == 0
    assert receipt["overall_timing"]["elapsed_wall_seconds"] > 0
    assert caught.value.verification_receipt == receipt


def test_final_callback_overall_expiry_cannot_preserve_qualified_result(
    case, monkeypatch, clock
):
    calls = fake_oracle(monkeypatch, clock)

    def checkpoint(value):
        if value["fit_success"]:
            clock.advance(61)

    result = run(case, checkpoint=checkpoint)
    assert len(calls) == 3
    assert not result["fit_success"]
    assert result["roots"] is result["parameters"] is None
    assert result["verification_timing"]["timeout_reason"] == "overall_wall_limit"


def test_invalid_root_start_and_execution_fail_before_search(case, monkeypatch):
    monkeypatch.setattr(
        e.recovery_numerics, "search", lambda *a, **k: pytest.fail("search")
    )
    with pytest.raises(ValueError, match="trajectory roots"):
        run(case, guesses={"init_z1_value": 1})
    with pytest.raises(TypeError, match="execution policy"):
        run(case, execution={"search_overall_seconds": 100})


def test_failed_training_final_checkpoint_clears_qualified_coordinates(case):
    model, data, _, _, numerical = case
    saved = []

    def checkpoint(value):
        saved.append(value)
        if value["fit_success"]:
            raise OSError("terminal receipt unavailable")

    result = e.fit_training(
        model, data, numerical, e.SearchExecutionPolicy(), checkpoint
    )
    assert not result["fit_success"]
    assert result["parameters"] is result["shared_parameters"] is None
    assert result["training_roots"] is result["realized_initial_states"] is None
    assert saved[-1]["fit_success"]
    assert result["verification_timing"]["callback_failures"] == 1


def test_parameter_free_problem_runs_only_independent_verification(case):
    model, data, _, _, numerical = case
    payload = model.model_dump(mode="json")
    for name, value in (("z1", 1.1), ("z2", 0.8)):
        payload["graph"]["rules"][name]["initial"] = {
            "mode": "known",
            "value": value,
            "justification": "Explicit synthetic preparation",
        }
    model = p.PrefixFitProblem.model_validate(payload)
    fit = e.fit_training(
        model,
        data.model_copy(update={"rows": (data.rows[0],)}),
        numerical,
        e.SearchExecutionPolicy(),
    )
    assert fit["fit_success"], fit
    assert fit["search"]["stop_reason"] == "no_free_coordinates"
    assert fit["search_calls_started"] == fit["search_calls_attempted"] == 0
    assert fit["verification_calls_started"] == fit["verification_calls_completed"] == 2
    assert fit["parameters"] == {}


def test_search_active_allowance_accepts_full_legacy_policy_range(case):
    # M45 verification limits stay unchanged; the search policy permits longer
    # declared numerical windows, bounded by its independent overall allowance.
    numerical = case[-1].model_copy(update={"seconds": 301, "maximum_calls": 1})
    result = run(case, numerical=numerical)
    assert result["fit_success"]
    assert result["policy"]["seconds"] == 301
