"""Budgeted branching, common starts, independent retention and resumable accounting."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import scaling_campaign as campaign
from autoformalism.fitting import scaling_recovery as fit
from tests.test_nonlinear_comparison import checked
from tests.test_nonlinear_comparison import exported as exported_fixture
from tests.test_recovery_numerics import QuadraticOracle
from tests.test_trajectory_profile import problem as ode_problem

exported = exported_fixture


@pytest.fixture
def small(monkeypatch):
    p = ode_problem()
    # Use the real typed training boundary while substituting a cheap local oracle.
    p = p.model_copy(
        update={
            "start": {"a": 1.0, "initial": 0.0},
            "incumbent": {"a": 1.0, "initial": 0.0},
        }
    )
    oracle = QuadraticOracle()
    monkeypatch.setattr(fit.rollout, "build", lambda _: (oracle, None, {}))
    monkeypatch.setattr(
        fit.rollout, "verify", lambda o, s, *a, **kw: checked(s, s["a"])
    )
    monkeypatch.setattr(fit.checks, "sensitivity", lambda *a: None)
    starts = [p.start, *[{"a": float(k), "initial": 0.0} for k in (3, 4, 5)]]
    return p, starts


def response(start, loss, *, progress=True, stop="budget_limited"):
    return {
        "calls": 10,
        "completed_calls": 10,
        "stop_reason": stop,
        "best": {"parameters": start | {"a": loss}, "training_nmse": loss},
        "evaluations": [
            {"best_training_nmse": loss * (2 if progress and k < 9 else 1)}
            for k in range(10)
        ],
    }


def test_progress_extends_then_reserves_every_frozen_start_and_retains_best(
    small, monkeypatch, tmp_path
):
    p, starts = small
    histories = []
    for method in fit.METHODS:
        seen = []

        def search(o, initial, *a, seen=seen, **kw):
            seen.append((initial, kw))
            return response(initial, [0.5, 0.4, 0.6, 0.7, 0.8, 0.9][len(seen) - 1])

        monkeypatch.setattr(fit.numerical, "search", search)
        result = fit.run(p, starts, fit.Policy(), method, tmp_path / method)
        assert result["blocks_used"] == 6 and result["search_calls_started"] == 60
        assert result["retained_parameters"]["a"] == 0.4
        assert [s[0]["a"] for s in seen] == [1, 0.5, 3, 0.6, 4, 5]
        assert all(s[1]["optimizer_scaling"] == method for s in seen)
        assert all(s[1]["normalized_objective"] for s in seen)
        assert all(s[1]["calls"] == 40 for s in seen)
        assert "reserve_unvisited_starts" in [d["reason"] for d in result["decisions"]]
        histories.append([s[0] for s in seen])
    assert histories[0] == histories[1] == histories[2]


def test_plateau_and_native_stop_diversify_without_spending_unused_ceiling(
    small, monkeypatch, tmp_path
):
    p, starts = small
    seen = []

    def search(o, initial, *a, **kw):
        seen.append(initial)
        return response(initial, 0.5, progress=False)

    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit.run(p, starts, fit.Policy(), "guarded_relative", tmp_path)
    assert seen == starts and result["blocks_used"] == 4
    assert all(d["reason"] == "plateau" for d in result["decisions"])
    assert (
        fit.continuation_reason(
            response(starts[0], 0.1, stop="step_or_gradient_converged"), fit.Policy()
        )
        == "native_stop_or_failure"
    )


def test_verified_target_early_stop_and_frozen_resume(small, monkeypatch, tmp_path):
    p, starts = small
    monkeypatch.setattr(fit.rollout, "verify", lambda o, s, *a, **kw: checked(s, 1e-15))
    monkeypatch.setattr(fit.numerical, "search", lambda *a, **kw: pytest.fail("search"))
    policy = fit.Policy()
    result = fit.run(p, starts, policy, "guarded_relative", tmp_path)
    assert result["accuracy_verified"] and result["blocks_used"] == 0
    monkeypatch.setattr(fit.rollout, "build", lambda *a: pytest.fail("repeated setup"))
    assert fit.run(p, starts, policy, "guarded_relative", tmp_path) == result
    with pytest.raises(ValueError, match="identity"):
        fit.run(
            p,
            starts,
            policy.model_copy(update={"objective_multiplier": 2}),
            "guarded_relative",
            tmp_path,
        )


def test_target_claim_requires_independent_and_worst_trajectory_checks(
    small, monkeypatch, tmp_path
):
    p, starts = small
    monkeypatch.setattr(
        fit.numerical,
        "search",
        lambda o, s, *a, **kw: response(s, 0.1, stop="numerical_target_reached"),
    )
    monkeypatch.setattr(
        fit.rollout, "verify", lambda o, s, *a, **kw: checked(s, 1e-14, worst=1)
    )
    result = fit.run(p, starts, fit.Policy(), "fixed_coordinates", tmp_path)
    assert not result["accuracy_verified"] and result["blocks_used"] == 4
    assert result["assessment"]["fit_quality"] == "above_numerical_target"


def test_interrupted_block_never_restarts_or_loses_incumbent(
    small, monkeypatch, tmp_path
):
    p, starts = small
    monkeypatch.setattr(
        fit.numerical, "search", lambda o, s, *a, **kw: response(s, 2, progress=False)
    )
    policy = fit.Policy()
    source = tmp_path / "complete"
    fit.run(p, starts, policy, "fixed_coordinates", source)
    target = tmp_path / "interrupted"
    seal(
        target / "start-0/block-0/rollout/started.json",
        read_seal(source / "start-0/block-0/rollout/started.json"),
    )
    seen = []

    def search(o, s, *a, **kw):
        seen.append(s)
        return response(s, 2, progress=False)

    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit.run(p, starts, policy, "fixed_coordinates", target)
    assert seen == starts[1:]
    assert result["retained_parameters"] == starts[0]
    assert result["search_calls_started"] is None and not result["cost_complete"]
    interrupted = result["operations"][1]
    assert interrupted["status"] == "interrupted"
    assert interrupted["budget_charge_seconds"] == policy.block_seconds


def test_restart_distinction_checks_outer_coordinates(small):
    p, starts = small
    with pytest.raises(ValueError, match="distinct"):
        fit.validate_starts(QuadraticOracle(), None, p, [starts[0], starts[0]])
    with pytest.raises(ValueError, match="original"):
        fit.validate_starts(QuadraticOracle(), None, p, starts[1:])


def test_joint_fallback_cannot_reset_block_call_or_time_budget(
    small, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    p, starts = small
    profile = SimpleNamespace(outer=[0, 1])
    monkeypatch.setattr(
        fit.rollout, "build", lambda _: (QuadraticOracle(), profile, {})
    )
    seen = []

    def search(o, start, end, checkpoint, **kw):
        seen.append((kw["profile"], kw["calls"], end))
        if kw["profile"] is not None:
            return response(start, 0.5, stop="numerical_failure")
        assert kw["calls"] == 30
        assert end <= seen[-2][2] + 0.02
        return response(start, 0.25, stop="step_or_gradient_converged")

    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit.run(p, starts, fit.Policy(), "guarded_relative", tmp_path)
    assert len(seen) == 8 and result["blocks_used"] == 4
    assert result["retained_parameters"]["a"] == 0.25
    assert result["search_calls_started"] == 80
    assert all(d["reason"] == "native_stop_or_failure" for d in result["decisions"])


@pytest.fixture
def prepared(exported, monkeypatch, tmp_path):
    monkeypatch.setattr(
        campaign,
        "portfolios",
        lambda d, p: {k: [v.start] * 4 for k, v in campaign.source.bases(d).items()},
    )
    root = tmp_path / "campaign"
    campaign.prepare(root, exported, campaign.Policy())
    return root


def test_all_nine_tasks_training_boundary_sealed_evaluation_and_resume(
    prepared, monkeypatch
):
    plan, _ = campaign.verify(prepared)
    assert len(plan["tasks"]) == 9
    assert campaign.report(prepared)["recorded"] == 0
    seen = []

    def run(problem, starts, policy, method, folder):
        assert set(problem.model_dump()) == {
            "request",
            "training",
            "coordinates",
            "start",
            "incumbent",
        }
        assert problem.incumbent == problem.start
        seen.append((starts, policy.model_dump(), method))
        result = dict.fromkeys(
            (
                "search_calls_started",
                "search_calls_completed",
                "maximum_search_calls",
                "search_budget_charge_seconds",
                "additional_wall_seconds",
                "additional_cpu_seconds",
                "additional_budget_charge_seconds",
                "blocks_used",
            ),
            0,
        )
        return result | {
            "status": "complete",
            "assessment": {},
            "accuracy_verified": False,
            "decisions": [],
            "cost_complete": True,
            "search_call_accounting_complete": True,
            "retained_parameters": problem.start,
            "operations": [],
        }

    monkeypatch.setattr(campaign.fitting, "run", run)

    def evaluate(folder, *a):
        assert (folder.parent / "backend.json").exists()
        return {"training_nmse": 0.1, "validation_nmse": 0.2}

    monkeypatch.setattr(campaign.screening_replay, "_evaluation", evaluate)
    rows = [campaign.run_task(prepared, i) for i in range(3)]
    assert seen[0][:2] == seen[1][:2] == seen[2][:2]
    assert len({r["starts_sha256"] for r in rows}) == 1
    monkeypatch.setattr(campaign.fitting, "run", lambda *a: pytest.fail("repeated fit"))
    assert campaign.run_task(prepared, 0) == rows[0]
    assert campaign.report(prepared)["recorded"] == 3
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, 9)


def test_export_excludes_warm_endpoints_and_configuration_is_immutable(
    prepared, monkeypatch, tmp_path
):
    _, data = campaign.verify(prepared)
    monkeypatch.setattr(
        campaign.stagnation_campaign,
        "verify",
        lambda *a, **kw: (
            {},
            {
                "source_inputs": {"data": data},
                "parents": {"forbidden": "warm endpoint"},
            },
        ),
    )
    path = tmp_path / "export.json"
    assert campaign.export_inputs(prepared, path)["original_starts"] == 3
    assert read_seal(path) == data
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, path, campaign.Policy(objective_multiplier=10))
    bad = deepcopy(data)
    bad["extra"] = "changed"
    seal(tmp_path / "changed.json", bad)
    with pytest.raises(ValueError, match="unchanged frozen"):
        campaign.prepare(tmp_path / "bad", tmp_path / "changed.json", campaign.Policy())


def test_delta_submission_is_idempotent_and_uncertain_reply_never_retried(
    prepared, monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    from scripts import submit_phase_c_fitting_scaling as submitter

    config = tmp_path / "config.json"
    config.write_text(fit.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    # Avoid patching subprocess.run used internally by check_output for git.
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    args = {"account": "test-cpu", "concurrency": 3, "inputs": prepared / "inputs.json"}
    first = submitter.submit(prepared, config, **args)
    assert first["array_tasks"] == 9 and first["gpus"] == 0
    assert "--array=0-8%3" in calls[1] and "--time=02:30:00" in calls[1]
    assert submitter.submit(prepared, config, **args) == first
    assert len(calls) == 3
    unknown = tmp_path / "unknown"
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="timeout"),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(unknown, config, **args)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "run", lambda *a, **kw: pytest.fail("resubmit")
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(unknown, config, **args)
