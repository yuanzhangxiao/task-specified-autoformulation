"""Isolate graph caching from primal initialization, without reference-based choice."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import reuse_diagnostic as d
from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting import transcription_fit as f
from tests.test_fitting_strategies import small_payload


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


def record(cost=0.2, defect=1e-8, bounded=True, iteration=1):
    return {
        "training_nmse": cost,
        "maximum_scaled_defect": defect,
        "within_parameter_bounds": bounded,
        "primal": [1.0],
        "primal_sha256": str(iteration),
        "iteration": iteration,
        "parameters": {},
        "collocation_nmse": 0.1,
    }


def test_training_gate_and_cold_fallback():
    policy = d.ReuseDiagnosticPolicy()
    good, bad = record(), record(2, iteration=2)
    for arm in ("rebuild_cold", "cache_cold"):
        assert d.warm_start_decision(arm, [good], 1, policy)["record"] is None
    assert (
        d.warm_start_decision("cache_last_primal", [good, bad], 1, policy)["record"]
        == bad
    )
    assert (
        d.warm_start_decision("cache_screened_primal", [good, bad], 1, policy)["record"]
        == good
    )
    # Native success is not needed; unavailable ordinary rollout is not zero loss.
    assert (
        d.warm_start_decision("cache_screened_primal", [good], None, policy)["record"]
        == good
    )
    for candidate in (
        record(None),
        record(float("nan")),
        record(defect=1),
        record(bounded=False),
        record(cost=1),
        record(cost=0.9999),
    ):
        assert (
            d.warm_start_decision("cache_screened_primal", [candidate], 1, policy)[
                "record"
            ]
            is None
        )
    with pytest.raises(ValueError, match="unknown"):
        d.warm_start_decision("wrong", [], None, policy)


def test_pool_keeps_best_feasible_and_latest_in_time_order():
    records = [record(0.2, iteration=1), record(defect=1, iteration=2)]
    records[0]["collocation_nmse"] = 0.2
    records[1]["collocation_nmse"] = 0.0
    selected = d._pool(records)
    assert selected == records


@pytest.mark.parametrize(
    "arm,builds",
    [
        ("rebuild_cold", 2),
        ("cache_cold", 1),
        ("cache_last_primal", 1),
        ("cache_screened_primal", 1),
    ],
)
def test_native_cache_and_cold_reset(inputs, tmp_path, monkeypatch, arm, builds):
    payload = small_payload(inputs, "collocation")
    payload.update(
        arm=arm,
        policy={"seconds": 45},
        reuse_diagnostic={"native_seconds": 10, "iterations": 2, "point_seconds": 1},
    )
    calls = []
    original = d.build_problem

    def build(*args):
        problem = original(*args)
        calls.append(problem)
        return problem

    monkeypatch.setattr(d, "build_problem", build)
    result = d.fit(payload, tmp_path)
    assert result["graph_builds"] == len(calls) == builds
    assert len(result["attempts"]) == 2
    assert result["parameters"] is not None
    first, second = result["attempts"]
    assert first["start_source"] == "ordinary"
    if arm.endswith("cold"):
        assert first["initial_primal_sha256"] == second["initial_primal_sha256"]
        assert first["iterations"] == second["iterations"]
    if arm == "cache_last_primal":
        assert second["start_source"] == "last_finite_primal"
        assert first["initial_primal_sha256"] != second["initial_primal_sha256"]
    for i in range(2):
        started = public._read(tmp_path / f"attempt-{i}/started.json")
        assert started["dual_start"] == "explicit_zero"
        assert started["warm_start_init_point"] == "no"
        assert started["maximum_iterations"] == 2
    assert (
        result["training_nmse"]
        <= public._read(tmp_path / "screens/000.json")["training_nmse"]
    )
    assert (
        len(list((tmp_path / "screens").glob("*.json")))
        == result["actual_residual_calls"]
    )


def test_failed_screens_cannot_become_good_warm_starts(inputs, tmp_path, monkeypatch):
    payload = small_payload(inputs, "collocation")
    payload.update(
        arm="cache_screened_primal",
        policy={"seconds": 30},
        reuse_diagnostic={"native_seconds": 5, "iterations": 2, "point_seconds": 0.1},
    )
    calls = []

    def unavailable(self, values):
        calls.append(self.point_seconds)
        self.budget.take()
        raise TimeoutError("test stalled rollout")

    monkeypatch.setattr(d.GuardedOracle, "__call__", unavailable)
    result = d.fit(payload, tmp_path)
    assert calls and all(seconds == 0.1 for seconds in calls)
    assert result["parameters"] is None
    assert result["attempts"][1]["start_source"] == "ordinary"
    screens = [public._read(p) for p in (tmp_path / "screens").glob("*.json")]
    assert all(s["status"] == "rollout_unavailable" for s in screens)


def test_matched_campaign_truth_separation_and_no_budget_reset(
    inputs, tmp_path, monkeypatch
):
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    old, new = tmp_path / "old", tmp_path / "new"
    c.prepare(old, c.CampaignConfig(starts=1, include_cstr=False, cases=("linear",)))
    old_plan, old_inputs = c.verify(old)
    config = c.CampaignConfig(
        starts=1,
        include_cstr=False,
        cases=("linear",),
        strategy=f.StrategyPolicy(seconds=600),
        reuse_diagnostic=d.ReuseDiagnosticPolicy(),
    )
    assert c.prepare(new, config, matched_source=old)["tasks"] == 4
    plan, data = c.verify(new)
    assert plan["protocol"] == c.DIAGNOSTIC_PROTOCOL
    assert plan["commons"] == old_plan["commons"] and data == old_inputs
    for task in plan["tasks"]:
        worker = c.worker_payload(plan, data, task)
        assert set(worker) == {
            "request",
            "coordinates",
            "nodes",
            "training",
            "arm",
            "policy",
            "reuse_diagnostic",
        }
        assert worker["reuse_diagnostic"]["iterations"] == 200
    assert c.prepare(new, config, matched_source=old)["tasks"] == 4
    seal(
        new / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    task = plan["tasks"][0]
    folder = new / "results" / task["task_id"]
    seal(folder / "started.json", {"plan_sha256": public.content_sha256(plan)})
    monkeypatch.setattr(c, "run", lambda *a: pytest.fail("budget cannot restart"))
    result = c.run_task(new, 0)
    assert result["status"] == "interrupted" and not result["budget_restarted"]
    assert c.run_task(new, 0) == result
    summary = c.report(new)
    assert summary["recorded"] == 1 and summary["expected"] == 4
    assert read_seal(new / "plan.json")["live_llm_calls"] == 0


def test_bad_policy_and_splits_rejected(inputs, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        c.CampaignConfig(include_reuse=True, reuse_diagnostic=d.ReuseDiagnosticPolicy())
    with pytest.raises(ValueError, match="two solves"):
        c.CampaignConfig(reuse_diagnostic=d.ReuseDiagnosticPolicy())
    payload = small_payload(inputs, "collocation")
    payload.update(arm="cache_cold", policy={"seconds": 30}, reuse_diagnostic={})
    payload["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        d.fit(payload, tmp_path)


def test_nonfinite_checkpoint_rejected(inputs, tmp_path):
    problem = d.build_problem(small_payload(inputs, "collocation"), tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        d._snapshot(problem, 0, lambda expression: np.nan)


def test_partial_worker_recovers_attempt_metadata_without_new_solve(
    tmp_path, monkeypatch
):
    payload = {"arm": "cache_cold", "policy": {"seconds": 30}, "reuse_diagnostic": {}}
    folder = tmp_path / "attempt-0"
    folder.mkdir()
    public._write(
        folder / "formulation.json",
        {
            "graph_builds_so_far": 1,
            "graph_seconds": 0.1,
            "start_source": "ordinary",
        },
    )
    public._write(
        tmp_path / "best.json",
        {
            "parameters": {"a": 1},
            "training_nmse": 0.5,
        },
    )
    monkeypatch.setattr(
        f,
        "invoke",
        lambda *a: {
            "wall_timeout": True,
            "elapsed_seconds": 30,
            "returncode": -9,
        },
    )
    result = f.run(payload, tmp_path)
    assert result["parameters"] == {"a": 1}
    assert result["partial_metadata"] and result["budget_exhausted"]
    assert result["graph_builds"] == 1
    assert result["attempts"][0]["status"] == "interrupted"


def test_completed_report_has_separate_timing_and_parameter_accuracy(
    inputs, tmp_path, monkeypatch
):
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(
        tmp_path,
        c.CampaignConfig(
            starts=1,
            include_cstr=False,
            cases=("linear",),
            strategy=f.StrategyPolicy(seconds=600),
            reuse_diagnostic=d.ReuseDiagnosticPolicy(),
        ),
    )
    plan, data = c.verify(tmp_path)
    seal(
        tmp_path / "qualification/result.json",
        {
            "passed": True,
            "plan_sha256": public.content_sha256(plan),
        },
    )
    task = plan["tasks"][0]
    folder = tmp_path / "results" / task["task_id"]
    seal(
        folder / "backend.json",
        {
            "parameters": data["cases"]["linear"]["reference_parameters"],
            "stop_reason": "test",
            "total_seconds": 3,
            "budget_exhausted": False,
            "worker_payload_sha256": public.content_sha256(
                c.worker_payload(plan, data, task)
            ),
            "graph_builds": 2,
            "attempts": [
                {
                    "attempt": i,
                    "graph_seconds": 0.1,
                    "seconds": 1,
                    "start_source": "ordinary",
                }
                for i in range(2)
            ],
        },
    )
    monkeypatch.setattr(
        c,
        "replay",
        lambda *a: {
            "complete": True,
            "maximum_solver_difference": 0,
            "metrics": {"train": 0, "val": 0},
        },
    )
    monkeypatch.setattr(
        c, "run", lambda *a: pytest.fail("completed backend cannot rerun")
    )
    result = c.run_task(tmp_path, 0)
    assert result["coefficients_recovered"]
    assert result == c.run_task(tmp_path, 0)
    summary = c.report(tmp_path)
    group = next(g for g in summary["groups"] if g["arm"] == task["arm"])
    assert group["native_timings_available"] == group["recorded_attempts"] == 2
    assert group["graph_seconds"] == pytest.approx(0.2)
    assert group["second_start_sources"] == {"ordinary": 1}
