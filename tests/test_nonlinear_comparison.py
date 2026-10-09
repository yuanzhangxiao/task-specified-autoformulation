"""Strong common kernel, matched budgets, restart retention and sealed scoring."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import nonlinear_comparison as fit
from autoformalism.fitting import nonlinear_comparison_campaign as campaign
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import trajectory_profile_fit as conditional
from autoformalism.fitting.fitting_assessment import FittingAssessment
from tests.test_recovery_numerics import QuadraticOracle
from tests.test_trajectory_profile import problem as numerical_problem


def problem():
    return SimpleNamespace(
        incumbent={"a": 1.0, "initial": 1.0},
        start={"a": 2.0, "initial": 2.0},
        model_dump=lambda **_: {"fixture": "nonlinear-comparison"},
    )


def checked(parameters, loss, *, worst=None):
    return {
        "parameters": parameters,
        "training_nmse": loss,
        "alternate_training_nmse": loss,
        "solver_loss_discrepancy": 0.0,
        "solver_prediction_discrepancy": 0.0,
        "maximum_trajectory_nmse": loss if worst is None else worst,
    }


@pytest.fixture
def mocked(monkeypatch):
    profile = object()
    monkeypatch.setattr(
        fit.rollout,
        "build",
        lambda _: (QuadraticOracle(), profile, {"selected": "terminal_output_profile"}),
    )
    monkeypatch.setattr(fit.rollout, "verify", lambda o, p, *a, **k: checked(p, p["a"]))
    return profile


def test_verified_incumbent_skips_work_and_finished_resume(
    mocked, monkeypatch, tmp_path
):
    monkeypatch.setattr(fit.rollout, "verify", lambda o, p, *a, **k: checked(p, 1e-15))
    monkeypatch.setattr(fit.numerical, "search", lambda *a, **k: pytest.fail("search"))
    policy = fit.ComparisonPolicy()
    result = fit.run(problem(), policy, fit.METHODS[0], tmp_path)
    assert not result["portfolio_triggered"] and result["trial_count"] == 0
    FittingAssessment.model_validate(result["assessment"])
    monkeypatch.setattr(fit.rollout, "build", lambda _: pytest.fail("restarted"))
    assert fit.run(problem(), policy, fit.METHODS[0], tmp_path) == result
    with pytest.raises(ValueError, match="identity differs"):
        fit.run(problem(), policy, fit.METHODS[1], tmp_path)


def test_both_keep_profiling_portfolio_and_continue_best_distinct_basin(
    mocked, monkeypatch, tmp_path
):
    starts, kernels = [], []

    def search(o, start, end, save, **kw):
        starts.append(start)
        kernels.append(kw["profile"])
        save({"calls": 1})
        a = [1.0, 4.0, 2.0, 3.0, 0.1][(len(starts) - 1) % 5]
        return {
            "best": {"parameters": {"a": a, "initial": 0.0}},
            "calls": 1,
            "stop_reason": "budget_limited",
        }

    monkeypatch.setattr(fit.numerical, "search", search)
    monkeypatch.setattr(
        fit.ConditionalEngine, "propose", lambda self, start, *a: {"parameters": start}
    )
    outputs = [
        fit.run(problem(), fit.ComparisonPolicy(), m, tmp_path / m) for m in fit.METHODS
    ]
    assert kernels == [mocked] * 10 and starts[:5] == starts[5:]
    assert starts[4] == {"a": 2.0, "initial": 0.0}
    assert all(o["after_warm"]["training_nmse"] == 1.0 for o in outputs)
    assert all(o["selected"]["training_nmse"] == 0.1 for o in outputs)
    assert all(o["trial_count"] == 3 and o["continuation_run"] for o in outputs)
    assert read_seal(tmp_path / fit.METHODS[0] / "portfolio.json") == read_seal(
        tmp_path / fit.METHODS[1] / "portfolio.json"
    )
    assert not any(
        o["operation"] == "continuation/conditional" for o in outputs[1]["operations"]
    )


def test_interrupted_conditional_is_charged_without_reset(
    mocked, monkeypatch, tmp_path
):
    policy = fit.ComparisonPolicy(portfolio_size=0)
    p = problem()
    identity = {
        "problem_sha256": public.content_sha256(p.model_dump()),
        "method": fit.METHODS[1],
        "policy": policy.model_dump(mode="json"),
        "point": p.incumbent,
        "allowance_seconds": policy.warm_seconds * policy.conditional_fraction,
    }
    seal(tmp_path / "warm/conditional/started.json", identity)
    monkeypatch.setattr(
        fit.ConditionalEngine,
        "propose",
        lambda *a: pytest.fail("conditional budget restarted"),
    )
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda *a, **k: {"best": None, "calls": 1, "stop_reason": "budget_limited"},
    )
    result = fit.run(p, policy, fit.METHODS[1], tmp_path)
    ops = {o["operation"]: o for o in result["operations"]}
    assert ops["warm/conditional"]["status"] == "interrupted"
    assert ops["warm/rollout"]["identity"]["allowance_seconds"] == 420
    assert not result["cost_complete"]
    assert result["retained_parameters"] == p.incumbent


def test_failed_projection_gets_bounded_joint_fallback(mocked, monkeypatch, tmp_path):
    seen = []

    def search(o, start, end, save, **kw):
        seen.append(kw)
        return {"best": None, "calls": 2, "stop_reason": "numerical_failure"}

    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit.run(
        problem(), fit.ComparisonPolicy(portfolio_size=0), fit.METHODS[0], tmp_path
    )
    assert [s["profile"] for s in seen] == [mocked, None]
    assert [s["calls"] for s in seen] == [600, 598]
    assert result["retained_parameters"] == problem().incumbent


def test_worst_trajectory_prevents_early_stop(mocked, monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit.rollout, "verify", lambda o, p, *a, **k: checked(p, 5e-13, worst=2e-11)
    )
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda *a, **k: {"best": None, "calls": 1, "stop_reason": "budget_limited"},
    )
    r = fit.run(
        problem(), fit.ComparisonPolicy(portfolio_size=0), fit.METHODS[0], tmp_path
    )
    assert r["assessment"]["fit_quality"] == "above_numerical_target"
    assert any(o["operation"] == "warm/rollout" for o in r["operations"])


def test_conditional_basin_is_explored_without_promoting_a_worse_fit(
    monkeypatch, tmp_path
):
    from time import monotonic

    proposed = {"a": 4.0, "initial": 0.0}
    monkeypatch.setattr(conditional, "TrajectoryProblem", lambda *a: object())
    monkeypatch.setattr(
        conditional,
        "optimize",
        lambda *a, **k: {
            "parameters": proposed,
            "trajectories": {},
            "rollout_verified": False,
        },
    )
    oracle = QuadraticOracle()
    monkeypatch.setattr(oracle, "evaluate", lambda v, end, **k: (v - [2, 3], None))
    policy = fit.ComparisonPolicy()
    engine = conditional.ConditionalEngine(oracle, problem(), policy)
    start = problem().incumbent
    record = engine.propose(start, monotonic() + 10, lambda _: None, tmp_path)
    assert record["parameters"] == proposed
    assert (
        record["selected"]["training_nmse"] > record["candidates"][0]["training_nmse"]
    )
    assert not record["incumbent_replacement_authorized"]
    monkeypatch.setattr(
        oracle,
        "evaluate",
        lambda v, end, **k: (
            (_ for _ in ()).throw(ValueError("candidate rollout failed"))
            if v[0] == 4
            else (v - [2, 3], None)
        ),
    )
    assert (
        engine.propose(start, monotonic() + 10, lambda _: None, tmp_path / "failed")[
            "parameters"
        ]
        == start
    )


@pytest.fixture
def exported(tmp_path, monkeypatch):
    p = numerical_problem().model_dump(mode="json")
    data = {"commons": {f"alien_hard_s{i}": deepcopy(p) for i in range(3)}}
    monkeypatch.setattr(campaign, "INPUT_DIGEST", public.content_sha256(data))
    monkeypatch.setattr(campaign.source, "bases", lambda d: d["commons"])
    path = tmp_path / "inputs.json"
    seal(path, data)
    return path


def test_six_unfiltered_tasks_allowlist_and_sealed_scoring(
    exported, tmp_path, monkeypatch
):
    data = read_seal(exported)
    assert len(campaign.roster(data)) == 6
    assert set(campaign.bases(data)["alien_hard_s0"].model_dump()) == {
        "request",
        "training",
        "coordinates",
        "start",
        "incumbent",
    }
    bad = deepcopy(data)
    bad["commons"].pop("alien_hard_s0")
    with pytest.raises(ValueError, match="unchanged frozen"):
        campaign.bases(bad)
    root = tmp_path / "run"
    campaign.prepare(root, exported, fit.ComparisonPolicy())
    assert campaign.report(root)["recorded"] == 0

    def run(problem, policy, method, folder):
        point = {"parameters": problem.incumbent}
        return {
            "status": "complete",
            "after_warm": point,
            "selected": point,
            "assessment": {"recommended_action": "assess_parameter_uncertainty"},
            "trajectory_assessment": {},
            "routing": {"selected": "terminal_output_profile"},
            "portfolio_triggered": False,
            "trial_count": 0,
            "verified_trial_count": 0,
            "continuation_run": False,
            "additional_wall_seconds": 1.0,
            "additional_cpu_seconds": 1.0,
            "additional_budget_charge_seconds": 1.0,
            "cost_complete": True,
            "rollout_calls_observed": 1,
            "operations": [],
        }

    def score(folder, *a):
        assert (folder.parent / "backend.json").exists()
        return {"status": "complete", "accuracy_passed": True}

    monkeypatch.setattr(fit, "run", run)
    monkeypatch.setattr(campaign.screening_replay, "_evaluation", score)
    r = campaign.run_task(root, 0)
    assert campaign.run_task(root, 0) == r
    assert (
        campaign.report(root)["groups"][0]["totals"]["selected"]["accuracy_passed"] == 1
    )
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.run_task(root, 1)


def test_cpu_submission_identity_and_ambiguous_reply(exported, tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_generic_recovery as scheduler
    from scripts import submit_phase_c_nonlinear_comparison as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    config = tmp_path / "config.json"
    config.write_text(fit.ComparisonPolicy().model_dump_json())
    kw = {"account": "bibo-delta-cpu", "concurrency": 3, "inputs": exported}
    r = submit.submit(tmp_path / "run", config, **kw)
    assert submit.submit(tmp_path / "run", config, **kw) == r
    assert len(calls) == 3 and r["gpus"] == 0 and r["array_tasks"] == 6
    assert "--array=0-5%3" in calls[1] and "--time=01:30:00" in calls[1]
    monkeypatch.setattr(
        scheduler.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr="Socket timed out"
        ),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submit.submit(tmp_path / "uncertain", config, **kw)
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submit.submit(tmp_path / "uncertain", config, **kw)
