"""M33 packaging is immutable, portable and never silently resubmits work."""

import json
import shutil
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import diagnostic_supplement_campaign as campaign
from autoformalism.fitting import public_fitting as public
from scripts import submit_phase_c_diagnostic_supplement as scheduler


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Small archive stand-in; numerical verification has its own regression suite."""
    root = tmp_path / "m32"
    plan = {"tasks": [{"task_id": campaign.TARGET}, {"task_id": "unavailable"}]}
    seal(root / "plan.json", plan)
    seal(root / "inputs.json", {"records": [], "fixtures": []})
    monkeypatch.setattr(campaign.previous, "ARCHIVED", (campaign.TARGET, "unavailable"))
    monkeypatch.setattr(campaign.previous.fixtures, "roster", lambda: [])
    for task in plan["tasks"]:
        seal(
            root / "results" / task["task_id"] / "result.json",
            {
                **task,
                "status": "complete"
                if task["task_id"] == campaign.TARGET
                else "source_unavailable",
                "feedback": None,
            },
        )
    monkeypatch.setattr(
        campaign.previous,
        "verify",
        lambda path, runtime=False: (
            read_seal(path / "plan.json"),
            read_seal(path / "inputs.json"),
        ),
    )
    monkeypatch.setattr(
        campaign.previous,
        "completed",
        lambda folder, *args: read_seal(folder / "result.json"),
    )
    monkeypatch.setattr(
        campaign.previous,
        "_qualification",
        lambda rows: {
            "qualification_status": "passed",
            "qualification_gates": {"all_recorded": True},
        },
    )
    monkeypatch.setattr(public, "_source_identity", lambda: "fixture-code")
    return root


def test_snapshot_audit_keeps_unavailable_and_all_source_bytes(source, tmp_path):
    root = tmp_path / "m33"
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    campaign.prepare(root, source, campaign.Policy())
    report = campaign.report(root)
    assert report["status"] == "incomplete"
    assert report["portable_audit"]["verified"] == 2
    assert report["portable_audit"]["rows"][1]["status"] == "source_unavailable"
    assert not report["fitting_performed"]
    assert before == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    moved = tmp_path / "downloaded"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    shutil.rmtree(source)
    assert campaign.report(moved) == report


def test_snapshot_changed_bytes_rejected_even_if_resealed(source, tmp_path):
    root = tmp_path / "m33"
    campaign.prepare(root, source, campaign.Policy())
    path = root / "source_snapshot/results" / campaign.TARGET / "result.json"
    path.unlink()
    seal(path, {"status": "different"})
    with pytest.raises(ValueError, match="snapshot/identity"):
        campaign.verify(root)


def test_resume_freezes_policy_and_source(source, tmp_path):
    root = tmp_path / "m33"
    first = campaign.prepare(root, source, campaign.Policy())
    again = campaign.prepare(root, source, campaign.Policy())
    assert first["identity"] == again["identity"]
    with pytest.raises(ValueError, match="source/policy"):
        campaign.prepare(
            root,
            source,
            campaign.Policy(supplement=campaign.supplement.Policy(extra_seconds=100)),
        )
    path = source / "results/unavailable/result.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source/policy"):
        campaign.prepare(root, source, campaign.Policy())


def test_partial_or_invalid_source_not_launched(source, tmp_path, monkeypatch):
    (source / "results/unavailable/result.json").unlink()
    with pytest.raises(ValueError, match="fully recorded"):
        campaign.prepare(tmp_path / "m33", source, campaign.Policy())
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(source / "new", source, campaign.Policy())
    monkeypatch.setattr(
        campaign.previous,
        "_qualification",
        lambda rows: {"qualification_status": "failed"},
    )
    seal(source / "results/unavailable/result.json", {"status": "unavailable"})
    with pytest.raises(ValueError, match="qualification"):
        campaign.prepare(tmp_path / "m33", source, campaign.Policy())


def test_shortened_roster_is_not_a_complete_campaign(source, tmp_path):
    path = source / "plan.json"
    plan = read_seal(path)
    plan["tasks"] = plan["tasks"][:1]
    path.unlink()
    seal(path, plan)
    with pytest.raises(ValueError, match="23-outcome"):
        campaign.prepare(tmp_path / "m33", source, campaign.Policy())


@pytest.mark.parametrize("wrong", ["path", "runtime"])
def test_report_rejects_substituted_predecessor_or_runtime(
    source, tmp_path, monkeypatch, wrong
):
    root = tmp_path / "m33"
    campaign.prepare(root, source, campaign.Policy())
    folder = root / "results" / campaign.TARGET
    seal(folder / "result.json", {"placeholder": True})
    seal(folder / "supplement.json", {"placeholder": True})
    expected = root / "source_snapshot/results" / campaign.TARGET / "diagnostics"
    subplan = SimpleNamespace(
        source_path=str(expected if wrong != "path" else tmp_path / "another"),
        runtime={"packages": public._runtime() if wrong != "runtime" else {}},
    )
    monkeypatch.setattr(campaign.supplement, "completed", lambda *a: object())
    monkeypatch.setattr(
        campaign.supplement,
        "SupplementPlan",
        SimpleNamespace(model_validate=lambda *a: subplan),
    )
    with pytest.raises(ValueError, match="predecessor/runtime"):
        campaign.report(root)


def test_only_frozen_target_and_policy_reach_runner(source, tmp_path, monkeypatch):
    root = tmp_path / "m33"
    campaign.prepare(root, source, campaign.Policy())
    calls = []

    def run(target, folder, policy):
        calls.append((target, folder, policy))
        return SimpleNamespace(model_dump=lambda mode: {"ok": True})

    monkeypatch.setattr(campaign.supplement, "run", run)
    with pytest.raises(ValueError, match="unknown"):
        campaign.run_task(root, 1)
    assert campaign.run_task(root, 0) == {"ok": True}
    assert calls == [
        (
            root / "source_snapshot/results" / campaign.TARGET / "diagnostics",
            root / "results" / campaign.TARGET,
            campaign.supplement.Policy(),
        )
    ]


def test_scheduler_records_one_cpu_task_and_does_not_duplicate(
    source, tmp_path, monkeypatch
):
    root, config = tmp_path / "m33", tmp_path / "policy.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", __import__("sys").executable)
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        scheduler.scheduler.subprocess, "check_output", lambda *a, **k: "commit"
    )
    first = scheduler.submit(root, source, config, account="bibo-delta-cpu")
    assert scheduler.submit(root, source, config, account="bibo-delta-cpu") == first
    assert len(calls) == 3
    assert "--array=0-0%1" in calls[1]
    assert "--cpus-per-task=1" in calls[1]
    assert "--time=00:45:00" in calls[1]
    assert first["gpus"] == first["live_llm_calls"] == 0
    (root / "submission/run.id").unlink()
    with pytest.raises(ValueError, match="Unconfirmed run"):
        scheduler.submit(root, source, config, account="bibo-delta-cpu")
    assert len(calls) == 3
    assert (
        json.loads((root / "submission/run.reply.json").read_text())["stdout"] == "102"
    )
