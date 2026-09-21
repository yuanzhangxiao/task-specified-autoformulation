"""Fresh requests, matched budgets, audit gates and complete paired reporting."""

import json
import shutil

import pytest

from autoformalism.rebuttal import function_dependency_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_function_dependency_confirmation as smoke
from scripts import submit_detention_process_pilot as submitter


@pytest.fixture
def historical(tmp_path):
    return smoke.historical_fixture(tmp_path / "historical")


def rewrite(path, **updates):
    value = sealed_read(path)
    value.pop("artifact_sha256")
    value.update(updates)
    path.unlink()
    return sealed_write(path, value)


def test_freeze_preserves_inputs_and_source_and_resumes(historical, tmp_path):
    source, audit, old = historical
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    assert experiment.previous._matched(plan) == experiment.previous._matched(old)
    assert plan["function_dependency_policy"] == experiment.dep.POLICY
    assert "handoff_confirmation" not in plan
    assert plan["artifact_sha256"] != old["artifact_sha256"]
    assert {p.name for p in root.iterdir()} == {".lock", "plan.json", "baseline.json"}
    assert experiment.freeze(source, audit, root) == plan
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    shutil.rmtree(source)
    shutil.rmtree(audit)
    assert experiment.pilot.verify(root) == plan
    report = experiment.report(root)
    assert report["current_status_counts"] == {"missing": 16}
    assert report["current_accounting"]["constructions_without_usage"] == 8
    assert all(r["current"]["validation"] is None for r in report["rows"])
    assert not report["automatic_followup"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("previously_accepted_now_blocked", [{"task": "regression"}]),
        ("unavailable_saved_attempts", 1),
        ("llm_calls", 1),
        ("optimizer_calls", 1),
        ("test_data_opened", True),
        ("whole_models_recovered", 1),
        ("constructions", []),
    ],
)
def test_audit_gate_rejects_regressions_or_wrong_source(
    historical, tmp_path, field, value
):
    source, audit, _ = historical
    rewrite(audit / "summary.json", **{field: value})
    with pytest.raises(ValueError, match="audit gate"):
        experiment.freeze(source, audit, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_source_and_package_drift_and_baseline_tampering(
    historical, tmp_path, monkeypatch
):
    source, audit, _ = historical
    runtime = experiment.public._runtime
    monkeypatch.setattr(experiment.public, "_runtime", lambda: {"python": "wrong"})
    with pytest.raises(ValueError, match="runtime differs"):
        experiment.freeze(source, audit, tmp_path / "new")
    monkeypatch.setattr(experiment.public, "_runtime", runtime)
    root = tmp_path / "new"
    experiment.freeze(source, audit, root)
    rewrite(root / "baseline.json", rows=[])
    with pytest.raises(ValueError, match="baseline"):
        experiment.pilot.verify(root)
    path = next(source.glob("construction/results/*/review/function_stage.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="audited source"):
        experiment.freeze(source, audit, tmp_path / "changed")


def test_new_policy_cannot_be_changed_on_resume(historical, tmp_path):
    source, audit, _ = historical
    root = tmp_path / "new"
    experiment.freeze(source, audit, root)
    rewrite(root / "plan.json", function_dependency_policy="strict")
    with pytest.raises(ValueError, match="confirmation inputs"):
        experiment.pilot.verify(root)


def test_overlap_execution_artifacts_and_missing_historical_results(
    historical, tmp_path
):
    source, audit, _ = historical
    with pytest.raises(ValueError, match="separate"):
        experiment.freeze(source, audit, source / "new")
    root = tmp_path / "new"
    (root / "construction").mkdir(parents=True)
    with pytest.raises(ValueError, match="execution artifacts"):
        experiment.freeze(source, audit, root)
    next(source.glob("results/*/result.json")).unlink()
    with pytest.raises(FileNotFoundError):
        experiment.freeze(source, audit, tmp_path / "missing")


def test_both_repairs_reach_gain_handoff_with_fresh_requests(historical, tmp_path):
    source, audit, old = historical
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    tasks, common, derived, calls, client = smoke.construct_pair(root, plan)
    old_calls = {p.stem for p in source.glob("construction/results/*/calls/*.json")}
    new_calls = {p.stem for p in root.glob("construction/results/*/calls/*.json")}
    assert new_calls and not old_calls & new_calls
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(calls)
    assert not common["fallback_used"]
    assert len({d["common_proposal_sha256"] for d in derived}) == 1
    assert (
        experiment.pilot.construct(
            root / "construction", plan, tasks[0]["construction_task"], client
        )
        == common
    )
    count = len(calls)
    report = experiment.report(root)
    assert report["current_accounting"]["physical_calls"] == count
    assert report["current_accounting"]["observed_total_tokens"] == count * 100
    assert report["current_accounting"]["atomic_function_repairs"] == {
        "physical_calls": 1,
        "observed_total_tokens": 100,
        "unmeasured_calls": 0,
        "missing_call_records": 0,
    }
    fresh = next(
        r["current"]
        for r in report["construction_pairs"]
        if r["current"]["status"] == "constructed"
    )
    repairs = fresh["function_repairs"]["routes"][0]
    assert [r["kind"] for r in repairs["dependency_revisions"]] == [
        "proposer_revision",
        "public_fixed_covariates",
    ]
    assert not repairs["connectivity"]["without_target_path"]
    assert experiment.report(root) == report
    alias = tmp_path / "campaign-alias"
    alias.symlink_to(root, target_is_directory=True)
    assert experiment.report(alias) == report
    assert len(calls) == count
    assert plan["config"] == old["config"]
    # A missing repair usage record is unavailable, never fabricated as observed.
    functions = root / "construction/results" / tasks[0]["construction_task"]["task_id"]
    (
        functions
        / "calls"
        / (
            next(
                e["request_hash"]
                for e in sealed_read(functions / "review/function_stage.json")[
                    "result"
                ]["events"]
                if e["step"].startswith("atomic_repair_")
            )
            + ".json"
        )
    ).unlink()
    accounting = experiment.report(root)["current_accounting"][
        "atomic_function_repairs"
    ]
    assert accounting["missing_call_records"] == 1


def test_paired_submission_is_bounded_and_idempotent(historical, tmp_path, monkeypatch):
    source, audit, _ = historical
    root = tmp_path / "new"
    experiment.freeze(source, audit, root)
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        path = tmp_path / key
        path.touch()
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("AF_COMMIT", "a" * 40)
    monkeypatch.setattr(submitter, "source_commit", lambda repo: "a" * 40)
    calls = []

    def queue(directory, key, options, worker, stage, index):
        calls.append((stage, options))
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", queue)
    manifest = submitter.submit(root)
    assert manifest["experiment"] == experiment.PROTOCOL
    assert manifest["fresh_constructions"] == 8
    assert "--array=0-15%4" in calls[2][1]
    assert "--dependency=afterany:103" in calls[3][1]
    assert submitter.submit(root) == manifest
    assert len(calls) == 4
