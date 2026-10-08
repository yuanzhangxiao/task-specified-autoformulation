"""Adaptive routing, identical portfolios, preserved incumbents and sealed scoring."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import assessed_recovery as fit
from autoformalism.fitting import assessed_recovery_campaign as campaign
from autoformalism.fitting import confidence_campaign as original
from autoformalism.fitting import larger_coupled_inputs as controls
from autoformalism.fitting import public_fitting as public


class Oracle:
    names = ("a", "initial")
    units = np.ones(2)
    lower, upper = np.array([0.01, -10.0]), np.array([30.0, 10.0])

    def __init__(self, *args):
        self.audit = {"fixture": True}

    def vector(self, parameters):
        return np.array([parameters[n] for n in self.names])

    def parameters(self, v):
        return dict(zip(self.names, v.tolist(), strict=True))

    def evaluate(self, v, deadline):
        return v, np.eye(2)


def problem():
    return SimpleNamespace(
        incumbent={"a": 1.0, "initial": 1.0},
        start={"a": 2.0, "initial": 2.0},
        model_dump=lambda **_: {"test": "assessed-recovery"},
    )


def checked(parameters, loss, discrepancy=0.0):
    return {
        "parameters": parameters,
        "training_nmse": loss,
        "alternate_training_nmse": loss + discrepancy,
        "solver_loss_discrepancy": abs(discrepancy),
        "solver_prediction_discrepancy": 1e-10,
    }


def test_accurate_incumbent_skips_search_and_exact_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    monkeypatch.setattr(
        fit.numerical, "check_point", lambda o, p, *a, **k: checked(p, 1e-22)
    )
    monkeypatch.setattr(
        fit.numerical, "search", lambda *a, **k: pytest.fail("unnecessary search")
    )
    p, policy = problem(), fit.RecoveryPolicy()
    result = fit.run(p, policy, "joint", tmp_path)
    assert result["selected"]["parameters"] == p.incumbent
    assert result["trial_count"] == 0 and not result["portfolio_triggered"]
    assert result["assessment"]["recommended_action"] == "retain_with_local_evidence"
    monkeypatch.setattr(fit, "AffineRollouts", lambda *a: pytest.fail("rerun"))
    assert fit.run(p, policy, "joint", tmp_path) == result
    with pytest.raises(ValueError, match="identity differs"):
        fit.run(p, policy, "profiled", tmp_path)


def test_warm_then_three_distinct_trials_and_best_trial_continuation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    monkeypatch.setattr(fit, "AffineProfile", lambda o: SimpleNamespace(audit={}))
    monkeypatch.setattr(
        fit.numerical, "check_point", lambda o, p, *a, **k: checked(p, p["a"])
    )
    starts, calls = [], []

    def search(o, start, end, save, *, calls: int, target, profile):
        starts.append(start)
        save({"calls": 1})
        # All trial optima are worse than the incumbent, but the most promising
        # distinct basin must still receive continuation (without replacing it).
        a = [1.0, 4.0, 2.0, 3.0, 0.1][(len(starts) - 1) % 5]
        return {
            "best": {"parameters": {"a": a, "initial": 0.0}},
            "calls": 1,
            "stop_reason": "budget_limited",
        }

    def counted(o, start, end, save, **kwargs):
        calls.append(kwargs["calls"])
        return search(o, start, end, save, **kwargs)

    monkeypatch.setattr(fit.numerical, "search", counted)
    policy = fit.RecoveryPolicy()
    outputs = []
    for method in fit.METHODS:
        outputs.append(fit.run(problem(), policy, method, tmp_path / method))
    assert starts[:5] == starts[5:]
    assert calls == [400, 150, 150, 150, 400] * 2
    assert starts[4] == {"a": 2.0, "initial": 0.0}
    assert len({tuple(p.values()) for p in starts[1:4]}) == 3
    assert outputs[0]["selected"]["training_nmse"] == 0.1
    assert outputs[0]["after_warm"]["training_nmse"] == 1.0
    assert all(r["trial_count"] == 3 and r["continuation_run"] for r in outputs)
    assert read_seal(tmp_path / "joint/portfolio.json") == read_seal(
        tmp_path / "profiled/portfolio.json"
    )


def test_loose_incumbent_is_retained_after_worse_strict_proposal(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)

    def verify(o, p, *args, **kwargs):
        return checked(p, 0.01, 4.8e-13) if p["a"] == 1 else checked(p, 0.02)

    monkeypatch.setattr(fit.numerical, "check_point", verify)
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda *a, **k: {
            "best": {"parameters": {"a": 2, "initial": 1}},
            "stop_reason": "budget_limited",
        },
    )
    r = fit.run(problem(), fit.RecoveryPolicy(portfolio_size=0), "joint", tmp_path)
    assert r["retained_parameters"] == problem().incumbent
    assert r["assessment"]["numerical_status"] == "fit_usable_uncertainty_unresolved"
    assert r["assessment"]["recommended_action"] == "verify_numerics"


def test_failed_checks_keep_original_and_interrupted_search_is_not_restarted(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    p, policy = problem(), fit.RecoveryPolicy(portfolio_size=0)
    identity = {
        "problem_sha256": public.content_sha256(p.model_dump()),
        "policy": policy.model_dump(mode="json"),
        "method": "joint",
        "point": {"start": p.incumbent, "calls": policy.warm_calls},
        "allowance_seconds": policy.warm_seconds,
    }
    seal(tmp_path / "warm/started.json", identity)
    monkeypatch.setattr(
        fit.numerical, "search", lambda *a, **k: pytest.fail("restarted allowance")
    )
    monkeypatch.setattr(
        fit.numerical,
        "check_point",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("unavailable")),
    )
    r = fit.run(p, policy, "joint", tmp_path)
    assert r["status"] == "retained_unverified" and r["selected"] is None
    assert r["retained_parameters"] == p.incumbent and not r["cost_complete"]
    op = next(o for o in r["operations"] if o["operation"] == "warm")
    assert op["status"] == "interrupted" and not op["budget_restarted"]
    assert op["budget_charge_seconds"] == policy.warm_seconds


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("assessed-source")
    controls.export_inputs(root / "linear.json")
    data = read_seal(root / "linear.json")
    endpoints = [
        {"task_id": f"{k}_{arm}", "common": k, "arm": arm, "parameters": b["start"]}
        for k, b in data["commons"].items()
        for arm in ("rollout_only", "coupled_profiled_rollout")
    ]
    source_data = {"data": data, "endpoints": endpoints}
    tasks = [{k: e[k] for k in ("task_id", "common", "arm")} for e in endpoints]
    plan = {"inputs_sha256": public.content_sha256(source_data), "tasks": tasks}
    patch = pytest.MonkeyPatch()
    patch.setattr(campaign.source, "SOURCE_PLAN", public.content_sha256(plan))
    converted = []
    for e, task in zip(endpoints, tasks, strict=True):
        p = original.training_problem(source_data, e).model_dump(mode="json")
        converted.append(
            task
            | {
                "problem": p
                | {"anchor": p["incumbent"], "profiles": [], "alternatives": []},
                "source_evaluation": {"status": "complete"},
                "source_cost": {},
            }
        )
    seal(
        root / "inputs.json",
        {
            "protocol": campaign.source.INPUT_PROTOCOL,
            "source_plan": plan,
            "source_data": source_data,
            "endpoints": converted,
            "test_data_opened": False,
        },
    )
    yield root / "inputs.json"
    patch.undo()


def test_roster_no_filter_training_allowlist_and_identity(exported, tmp_path):
    data = read_seal(exported)
    assert len(campaign.roster(data)) == 48
    payload = campaign.training_problem(data["endpoints"][0]).model_dump()
    assert set(payload) == {"request", "training", "coordinates", "start", "incumbent"}
    campaign.prepare(tmp_path, exported, fit.RecoveryPolicy())
    r = campaign.report(tmp_path)
    assert r["expected"] == 48 and r["recorded"] == 0 and not r["cost_complete"]
    table = (tmp_path / "assessment.md").read_text()
    assert table.count("await_result") == 48
    assert "correctness probabilities" in table
    bad = deepcopy(data)
    bad["endpoints"][0]["problem"]["training"]["rows"][0]["targets"]["y"][0] += 1
    with pytest.raises(ValueError, match="training problem differs"):
        campaign.roster(bad)


def test_score_only_after_seal_and_report_identity(exported, tmp_path, monkeypatch):
    campaign.prepare(tmp_path, exported, fit.RecoveryPolicy())

    def backend(problem, policy, method, folder):
        point = {"parameters": problem.incumbent}
        return {
            "status": "complete",
            "after_warm": point,
            "selected": point,
            "assessment": {
                "recommended_action": "assess_parameter_uncertainty",
                "parameter_information": "weak_sensitivity",
            },
            "additional_wall_seconds": 1.0,
            "additional_cpu_seconds": 1.0,
            "additional_budget_charge_seconds": 1.0,
            "cost_complete": True,
            "rollout_calls_observed": 2,
            "stage_costs": {},
            "trial_count": 0,
            "portfolio_triggered": False,
            "continuation_run": False,
        }

    monkeypatch.setattr(fit, "run", backend)

    def score(folder, *args):
        assert (folder.parent / "backend.json").is_file()
        return {"status": "complete", "accuracy_passed": True}

    monkeypatch.setattr(campaign.screening_replay, "_evaluation", score)
    r = campaign.run_task(tmp_path, 0)
    assert campaign.run_task(tmp_path, 0) == r
    assert (
        campaign.report(tmp_path)["groups"][0]["totals"]["selected"]["accuracy_passed"]
        == 1
    )
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.run_task(tmp_path, 1)


def test_submission_cpu_idempotence_uncertain_reply(exported, tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_assessed_recovery as submit
    from scripts import submit_phase_c_generic_recovery as scheduler

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    config = tmp_path / "policy.json"
    config.write_text(fit.RecoveryPolicy().model_dump_json())
    kwargs = {"account": "bibo-delta-cpu", "concurrency": 6, "inputs": exported}
    r = submit.submit(tmp_path / "run", config, **kwargs)
    assert submit.submit(tmp_path / "run", config, **kwargs) == r
    assert len(calls) == 3 and r["gpus"] == 0 and r["array_tasks"] == 48
    assert "--array=0-47%6" in calls[1] and "--dependency=afterok:101" in calls[1]
    assert "--time=01:00:00" in calls[1]
    monkeypatch.setattr(
        scheduler.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="Socket timed out",
        ),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submit.submit(tmp_path / "uncertain", config, **kwargs)
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submit.submit(tmp_path / "uncertain", config, **kwargs)


def test_launch_scripts_parse():
    import subprocess

    for kind in ("submit", "run", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_assessed_recovery_delta.sh"],
            check=True,
        )
