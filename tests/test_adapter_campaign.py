"""Matched qualification, evaluation isolation, report denominators and receipts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import adapter_campaign as campaign
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting.fitting_assessment import assess
from tests.test_broad_scaling import data as data_fixture
from tests.test_nonlinear_comparison import checked

data = data_fixture


@pytest.fixture
def prepared(data, tmp_path):
    inputs = tmp_path / "input.json"
    value = {
        "protocol": campaign.INPUT_PROTOCOL,
        "development": data,
        "starts": campaign.previous.portfolios(data, campaign.previous.Policy()),
        "fixed_endpoint": {
            "common": "basin_coupled_s0",
            "parameters": data["commons"]["basin_coupled_s0"]["start"],
        },
        "test_data_opened": False,
    }
    seal(inputs, value)
    root = tmp_path / "campaign"
    assert campaign.prepare(root, inputs, campaign.Policy())["tasks"] == 31
    return root


def fake_backend(problem, starts, policy, method, folder):
    assert "validation" not in problem.model_dump()
    assert "reference_parameters" not in problem.model_dump()
    selected = checked(problem.start, 0.1) | {"origin": "original/check"}
    return {
        "selected": selected,
        "status": "complete",
        "retained_parameters": problem.start,
        "routing": {"selected": "joint_rollout"},
        "assessment": assess(selected, None, [], [], policy.target_nmse),
        "accuracy_verified": False,
        "blocks_used": 1,
        "maximum_search_calls": policy.maximum_blocks * policy.block_calls,
        "search_calls_started": 1,
        "search_calls_completed": 1,
        "search_budget_charge_seconds": 1.0,
        "additional_wall_seconds": 2.0,
        "additional_cpu_seconds": 1.0,
        "additional_budget_charge_seconds": 2.0,
        "cost_complete": True,
        "search_call_accounting_complete": True,
    }


def test_training_only_paired_fit_is_sealed_before_evaluation_and_resume(
    prepared, monkeypatch
):
    seen = []

    def backend(problem, starts, policy, method, folder):
        seen.append((starts, policy, method))
        return fake_backend(problem, starts, policy, method, folder)

    def evaluation(data, task, parameters, policy, folder):
        assert (folder.parent / "fit/result.json").exists()
        assert (folder.parent / "fit/backend.json").exists()
        return {
            "status": "unverified",
            "solver_agreement_passed": False,
            "accounting_complete": True,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(adapter.recovery, "run", backend)
    monkeypatch.setattr(campaign, "evaluate", evaluation)
    a, b = [campaign.run_task(prepared, i) for i in range(2)]
    assert seen[0][:2] == seen[1][:2]
    assert seen[0][2] != seen[1][2]
    assert a["fit"]["parameters"] == b["fit"]["parameters"]
    monkeypatch.setattr(adapter.recovery, "run", lambda *a: pytest.fail("resume fit"))
    assert campaign.run_task(prepared, 0) == a
    summary = campaign.report(prepared)
    assert summary["expected"] == 31 and summary["recorded"] == 2
    assert not summary["cost_complete"]
    assert len(summary["groups"]) == 6
    assert all(g["expected"] == 5 for g in summary["groups"])
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, 31)


def test_fixed_endpoint_is_never_sent_to_fitter(prepared, monkeypatch):
    monkeypatch.setattr(adapter, "run", lambda *a: pytest.fail("diagnostic fit"))
    seen = []
    monkeypatch.setattr(
        campaign,
        "evaluate",
        lambda d, t, p, policy, f: seen.append(p)
        or {"status": "verified", "solver_agreement_passed": True},
    )
    row = campaign.run_task(prepared, 30)
    _, data = campaign.verify(prepared)
    assert seen == [data["fixed_endpoint"]["parameters"]]
    assert row["fit"] is None and not row["parameter_fitting_performed"]


def test_inputs_validation_is_not_in_fitter_and_changed_roster_is_rejected(prepared):
    _, data = campaign.verify(prepared)
    a = campaign.source.bases(data["development"])
    changed = deepcopy(data)
    for case in changed["development"]["cases"].values():
        case["validation"] = "forbidden"
        case["reference_parameters"] = "forbidden"
    assert campaign.source.bases(changed["development"]) == a
    changed["starts"]["alien_hard_s0"][0]["a"] += 1
    with pytest.raises(ValueError, match="identity"):
        campaign.roster(changed)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared,
            prepared / "inputs.json",
            campaign.Policy(search=campaign.previous.Policy(block_calls=20)),
        )


def test_coefficient_error_and_solver_agreement_are_separate(prepared, monkeypatch):
    plan, data = campaign.verify(prepared)
    task = plan["tasks"][0]
    monkeypatch.setattr(
        campaign.verified_replay,
        "run",
        lambda *a: {
            "status": "unverified",
            "solver_agreement_passed": False,
            "metrics": {"train": 1e-15, "val": 1e-15},
        },
    )
    monkeypatch.setattr(
        campaign,
        "coefficient_metrics",
        lambda *a: {"maximum_coefficient_relative_error": 1e-8},
    )
    answer = campaign.evaluate(data, task, {"x": 1}, campaign.Policy().replay, prepared)
    assert answer["coefficient_error_within_threshold"]
    assert not answer["prediction_accuracy_passed"]


def test_submit_is_idempotent_and_uncertain_receipt_blocks_duplicate(
    prepared, monkeypatch, tmp_path
):
    import sys

    from scripts import submit_phase_c_rollout_adapter as submitter

    config = tmp_path / "policy.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    kw = {"account": "test-cpu", "concurrency": 3, "inputs": prepared / "inputs.json"}
    first = submitter.submit(prepared, config, **kw)
    assert first["array_tasks"] == 31 and first["gpus"] == 0
    assert "--array=0-30%3" in calls[1] and "--time=02:30:00" in calls[1]
    assert submitter.submit(prepared, config, **kw) == first and len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="timeout"),
    )
    unknown = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(unknown, config, **kw)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "run", lambda *a, **kw: pytest.fail("resubmit")
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(unknown, config, **kw)
