"""M23 preserves direct warm fitting and allocates precision/screening explicitly."""

from time import monotonic

import pytest

from autoformalism.fitting import fitting_allocation as allocation
from autoformalism.fitting import nonlinear_comparison as fit
from autoformalism.fitting.fitting_assessment import FittingAssessment
from tests.test_nonlinear_comparison import checked, problem
from tests.test_nonlinear_comparison import mocked as mocked
from tests.test_recovery_numerics import QuadraticOracle


def policy(**kwargs):
    return fit.ComparisonPolicy(allocation="measured", **kwargs)


def test_precision_depends_on_decision_not_ultimate_target_alone():
    p = checked({"a": 1}, 0.4)
    p.update(
        solver_loss_discrepancy=1e-12,
        solver_prediction_discrepancy=1e-9,
        alternate_training_nmse=0.4 + 1e-12,
    )
    assert allocation.precision_reason(p, None, policy()) == (
        "ordinary_precision_sufficient_for_search"
    )
    near = p | {
        "training_nmse": 1e-12,
        "alternate_training_nmse": 2e-12,
        "maximum_trajectory_nmse": 3e-12,
    }
    assert allocation.precision_reason(near, None, policy()) == "near_accuracy_target"
    tie = p | {"parameters": {"a": 2}}
    assert allocation.precision_reason(tie, p, policy()) == "retention_interval_overlap"
    assert allocation.precision_reason(p, p, policy()) != "retention_interval_overlap"
    bad = p | {"solver_prediction_discrepancy": 0.2}
    assert (
        allocation.precision_reason(bad, None, policy())
        == "ordinary_numerics_unreliable"
    )


def test_screen_reserve_requires_whole_rollout_and_meaningful_solve():
    assert allocation.screening_plan(60, None, policy())["status"] == "budget_skip"
    assert allocation.screening_plan(60, 25, policy())["status"] == "budget_skip"
    plan = allocation.screening_plan(60, 15, policy())
    assert plan["screen_reserve_seconds"] == 30 and plan["node_seconds"] == 30


@pytest.mark.usefixtures("mocked")
def test_original_warm_identical_and_conditional_only_for_trials(monkeypatch, tmp_path):
    starts, seeds = [], []

    def search(o, start, end, save, **kw):
        starts.append((start, kw["profile"], kw["calls"]))
        return {"best": None, "calls": 1, "stop_reason": "budget_limited"}

    def propose(self, start, end, save, path):
        seeds.append((start, self.screen_cost))
        return {"parameters": start}

    monkeypatch.setattr(fit.numerical, "search", search)
    monkeypatch.setattr(fit.MeasuredConditionalEngine, "propose", propose)
    monkeypatch.setattr(
        fit.rollout,
        "verify",
        lambda o, p, *a, **kw: checked(p, 0.4) | {"integrator_seconds": {"DOP853": 12}},
    )
    a = fit.run(problem(), policy(portfolio_size=1), fit.METHODS[0], tmp_path / "a")
    b = fit.run(problem(), policy(portfolio_size=1), fit.METHODS[1], tmp_path / "b")
    assert starts[:2] == starts[2:] and len(seeds) == 1 and seeds[0][1] == 12
    assert not any(o["operation"] == "warm/conditional" for o in b["operations"])
    assert a["retained_parameters"] == b["retained_parameters"] == problem().incumbent
    assert b == fit.run(
        problem(), policy(portfolio_size=1), fit.METHODS[1], tmp_path / "b"
    )
    FittingAssessment.model_validate(b["assessment"])


@pytest.mark.usefixtures("mocked")
def test_poor_fit_avoids_tight_check_and_failed_near_retry_preserves_incumbent(
    monkeypatch, tmp_path
):
    calls = []

    def verify(o, p, *a, tight=False, **kw):
        calls.append(tight)
        if tight:
            raise TimeoutError("retry exhausted")
        value = 0.4 if p["a"] == 1 else 2e-12
        return checked(p, value) | {
            "solver_loss_discrepancy": 3e-13,
            "alternate_training_nmse": value + 3e-13,
            "solver_prediction_discrepancy": 1e-9,
        }

    monkeypatch.setattr(fit.rollout, "verify", verify)
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda *a, **k: {
            "best": {"parameters": {"a": 2.0, "initial": 1.0}},
            "calls": 1,
            "stop_reason": "budget_limited",
        },
    )
    r = fit.run(problem(), policy(portfolio_size=0), fit.METHODS[0], tmp_path)
    assert calls == [False, False, True]
    assert r["selected"]["training_nmse"] == 2e-12
    assert r["precision_decisions"][0]["attempts"] == []
    assert r["assessment"]["recommended_action"] == "verify_numerics"


def test_measured_screening_skips_without_solving_and_uses_full_remaining_window(
    monkeypatch, tmp_path
):
    engine = allocation.MeasuredConditionalEngine(
        QuadraticOracle(), problem(), policy()
    )
    engine.screen_cost = 30
    monkeypatch.setattr(
        allocation, "TrajectoryProblem", lambda *a: pytest.fail("build")
    )
    r = engine.propose(problem().start, monotonic() + 60, lambda _: None, tmp_path)
    assert r["allocation"]["status"] == "budget_skip" and r["calls"] == 0

    engine.policy = policy(
        minimum_node_seconds=0.1, node_targets=(10,), penalties=(100,)
    )
    engine.screen_cost = 0.5
    proposed = {"a": 4.0, "initial": 1.0}
    monkeypatch.setattr(allocation, "TrajectoryProblem", lambda *a: object())
    monkeypatch.setattr(
        allocation,
        "optimize",
        lambda *a, **k: {"parameters": proposed, "trajectories": {}},
    )
    seen = []

    def evaluate(vector, end, **kw):
        seen.append((vector.tolist(), end))
        return vector, None

    monkeypatch.setattr(engine.oracle, "evaluate", evaluate)
    end = monotonic() + 10
    r = engine.propose(problem().start, end, lambda _: None, tmp_path / "funded")
    assert len(seen) == 1 and seen[0][1] == end
    assert r["parameters"] == proposed and not r["control_rollout_repeated"]
    assert not r["incumbent_replacement_authorized"]


@pytest.mark.usefixtures("mocked")
def test_interrupted_conditional_cannot_restart_its_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda *a, **k: {"best": None, "calls": 1, "stop_reason": "budget_limited"},
    )

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(fit.MeasuredConditionalEngine, "propose", interrupted)
    p = policy(portfolio_size=1)
    with pytest.raises(KeyboardInterrupt):
        fit.run(problem(), p, fit.METHODS[1], tmp_path)
    monkeypatch.setattr(
        fit.MeasuredConditionalEngine,
        "propose",
        lambda *a: pytest.fail("interrupted work restarted"),
    )
    r = fit.run(problem(), p, fit.METHODS[1], tmp_path)
    ops = {o["operation"]: o for o in r["operations"]}
    assert ops["trial-0/conditional"]["status"] == "interrupted"
    assert ops["trial-0/conditional"]["budget_charge_seconds"] == 60
    assert ops["trial-0/rollout"]["identity"]["allowance_seconds"] == 140
    assert not r["cost_complete"] and r["retained_parameters"] == problem().incumbent


@pytest.mark.usefixtures("mocked")
def test_failed_precision_retry_is_reused_for_identical_parameters(
    monkeypatch, tmp_path
):
    retries = []

    def verify(o, parameters, *a, tight=False, **kw):
        if tight:
            retries.append(parameters)
            raise TimeoutError("limited")
        return checked(parameters, 2e-12) | {
            "alternate_training_nmse": 2.3e-12,
            "solver_loss_discrepancy": 3e-13,
        }

    monkeypatch.setattr(fit.rollout, "verify", verify)
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda o, start, *a, **k: {
            "best": {"parameters": start},
            "calls": 1,
            "stop_reason": "budget_limited",
        },
    )
    r = fit.run(problem(), policy(portfolio_size=0), fit.METHODS[0], tmp_path)
    assert len(retries) == 1
    assert r["precision_decisions"][-1]["attempts"][0]["reused"]


def test_screen_safety_margin_absorbs_small_node_deadline_overrun(
    monkeypatch, tmp_path
):
    clock = [0.0]
    monkeypatch.setattr(allocation, "monotonic", lambda: clock[0])
    p = policy(node_targets=(10,), penalties=(100,), minimum_node_seconds=0.1)
    engine = allocation.MeasuredConditionalEngine(QuadraticOracle(), problem(), p)
    engine.screen_cost = 2.0
    monkeypatch.setattr(allocation, "TrajectoryProblem", lambda *a: object())

    def optimize(*a, **kw):
        clock[0] = 6.01  # A 10s stage reserves 4s, then overruns by 0.01s.
        return {"parameters": problem().start, "trajectories": {}}

    monkeypatch.setattr(allocation, "optimize", optimize)
    monkeypatch.setattr(engine.oracle, "evaluate", lambda v, end, **kw: (v, None))
    r = engine.propose(problem().start, 10.0, lambda _: None, tmp_path)
    assert r["selected"] is not None and r["calls"] == 1
