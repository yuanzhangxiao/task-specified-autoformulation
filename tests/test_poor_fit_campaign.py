"""Matched input/ceiling policy, post-fit evaluation and durable paired launch."""

import sys
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import poor_fit_campaign as campaign
from autoformalism.fitting import public_rollout as bridge
from tests.test_adapter_campaign import fake_backend
from tests.test_public_integration import prepared as source_prepared

prepared_source = source_prepared


@pytest.fixture
def prepared(prepared_source, tmp_path):
    root = tmp_path / "paired"
    campaign.prepare(root, prepared_source / "inputs.json", campaign.Policy())
    return root


def test_policy_changes_only_handling_and_records_all_twenty_two_entries(prepared):
    plan, _ = campaign.verify(prepared)
    assert len(plan["tasks"]) == 22
    assert campaign.report(prepared)["status_counts"] == {"missing": 22}
    policy = campaign.Policy()
    a, b = [
        campaign.arm_policy(policy, n).model_dump(mode="json") for n in campaign.ARMS
    ]
    b["fitting"]["adapter"]["search"]["poor_fit"] = None
    assert a == b
    for i in (4, 5, 10, 11):
        row = campaign.run_task(prepared, i)
        assert row["status"] == "source_unavailable"
        assert campaign.source.costs(row)["wall_seconds"] == 0
    assert campaign.report(prepared)["recorded"] == 4
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, 22)


def test_matched_starts_postfit_evaluation_and_exact_resume(prepared, monkeypatch):
    monkeypatch.setattr(bridge.adapter.recovery, "run", fake_backend)

    def evaluate(record, parameters, policy, folder):
        assert (folder.parent / "fit/result.json").exists()
        saved = bridge.completed(folder.parent / "fit")
        assert parameters == saved.fit.parameters
        assert "validation" not in saved.preparation["value"]["problem"]
        return {"status": "verified", "solver_agreement_passed": True}

    monkeypatch.setattr(campaign.source, "evaluate", evaluate)
    rows = [campaign.run_task(prepared, i) for i in (0, 1)]
    result = campaign.report(prepared)
    assert result["pairs"][0]["starts_and_problem_match"] is True
    monkeypatch.setattr(bridge, "run", lambda *a: pytest.fail("repeat fitting"))
    assert campaign.run_task(prepared, 1) == rows[1]
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared, prepared / "inputs.json", campaign.Policy(handling={"window": 6})
        )


def test_tampered_child_cannot_be_resumed(prepared):
    campaign.run_task(prepared, 4)
    path = prepared / "results/legacy__cell02_seed0_full/result.json"
    value = read_seal(path)
    value["status"] = "complete"
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError, match="sealed child"):
        campaign.report(prepared)


def test_scheduler_idempotence_and_unknown_receipt_preserves_intent(
    prepared, tmp_path, monkeypatch
):
    from scripts import submit_phase_c_poor_fit as submitter

    config = tmp_path / "config.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test"
    )
    kwargs = {
        "account": "test-cpu",
        "concurrency": 3,
        "inputs": prepared / "inputs.json",
    }
    first = submitter.submit(prepared, config, **kwargs)
    assert first["array_tasks"] == 22 and first["gpus"] == 0
    assert "--array=0-21%3" in calls[1] and "--time=02:30:00" in calls[1]
    assert submitter.submit(prepared, config, **kwargs) == first and len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="timeout"),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(tmp_path / "unknown", config, **kwargs)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate")
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(tmp_path / "unknown", config, **kwargs)
