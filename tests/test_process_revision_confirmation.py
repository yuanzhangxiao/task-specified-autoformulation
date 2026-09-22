"""Audit-gated fresh construction, matched inputs, accounting and bounded submit."""

import json
import shutil

import pytest

from autoformalism.rebuttal import process_revision_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_process_revision_confirmation as smoke
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


def test_freeze_matches_inputs_and_resumes_without_historical_mounts(
    historical, tmp_path
):
    source, audit, old = historical
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    assert experiment._matched(plan) == experiment._matched(old)
    assert plan["process_assembly_policy"] == experiment.assembly.POLICY
    assert plan["function_dependency_policy"] == experiment.dep.POLICY
    assert "assembly_confirmation" not in plan
    assert experiment.freeze(source, audit, root) == plan
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert {p.name for p in root.iterdir()} == {".lock", "plan.json", "baseline.json"}
    shutil.rmtree(source)
    shutil.rmtree(audit)
    assert experiment.pilot.verify(root) == plan
    report = experiment.report(root)
    assert report["current_status_counts"] == {"missing": 16}
    assert report["current_accounting"]["constructions_without_usage"] == 8
    assert report["historical_accounting"]["identified_delivery"]["counters"] is None
    assert not report["automatic_followup"]
    markdown = (root / "COMPARISON.md").read_bytes()
    assert experiment.report(root) == report
    assert (root / "COMPARISON.md").read_bytes() == markdown


@pytest.mark.parametrize(
    "field,value",
    [
        ("unexpected_acceptance_regressions", [{"task": "bad"}]),
        ("unavailable_saved_attempts", 1),
        ("missing_constructions", ["bad"]),
        ("llm_calls", 1),
        ("optimizer_calls", 1),
        ("test_data_opened", True),
        ("whole_models_recovered", 1),
        ("constructions", []),
    ],
)
def test_audit_gate_rejects_incomplete_or_changed_evidence(
    historical, tmp_path, field, value
):
    source, audit, _ = historical
    rewrite(audit / "summary.json", **{field: value})
    with pytest.raises(ValueError, match="audit gate"):
        experiment.freeze(source, audit, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_source_runtime_and_baseline_drift(historical, tmp_path, monkeypatch):
    source, audit, _ = historical
    runtime = experiment.public._runtime
    monkeypatch.setattr(experiment.public, "_runtime", lambda: {"python": "wrong"})
    with pytest.raises(ValueError, match="runtime differ"):
        experiment.freeze(source, audit, tmp_path / "wrong-runtime")
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_assembly_policy", "legacy"),
        ("function_delivery_policy", "legacy"),
        ("function_dependency_policy", "strict"),
    ],
)
def test_policy_cannot_change_on_resume(historical, tmp_path, field, value):
    source, audit, _ = historical
    root = tmp_path / "new"
    experiment.freeze(source, audit, root)
    rewrite(root / "plan.json", **{field: value})
    with pytest.raises(ValueError, match="confirmation inputs"):
        experiment.pilot.verify(root)


def test_refuse_overlap_prior_execution_and_missing_fit(historical, tmp_path):
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


def test_real_repair_sign_and_two_gain_handoffs_are_fresh_and_counted_once(
    historical, tmp_path
):
    source, audit, old = historical
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    tasks, common, derived, calls, client = smoke.construct_pair(root, plan)
    old_calls = {p.stem for p in source.glob("construction/results/*/calls/*.json")}
    new_calls = {p.stem for p in root.glob("construction/results/*/calls/*.json")}
    assert new_calls and not old_calls & new_calls
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(calls)
    assert not common["fallback_used"]
    assert len({p["common_proposal_sha256"] for p in derived}) == 1
    count = len(calls)
    assert (
        experiment.pilot.construct(
            root / "construction", plan, tasks[0]["construction_task"], client
        )
        == common
    )
    assert len(calls) == count
    report = experiment.report(root)
    accounting = report["current_accounting"]
    assert accounting["physical_calls"] == count
    assert accounting["observed_total_tokens"] == count * 100
    counters = accounting["identified_delivery"]["counters"]
    assert counters["repair_calls"] == 1
    assert counters["assignment_labels_removed"] == 1
    assert counters["delivery_diagnostics"] > 0
    assert accounting["atomic_function_repairs"]["physical_calls"] == 1
    assert plan["config"] == old["config"]
    # Missing saved usage is visible, not counted as a measured zero-cost repair.
    repair = next(source for source in calls if "selected_slot" in source["payload"])
    assert repair
    base = root / "construction/results" / tasks[0]["construction_task"]["task_id"]
    f = sealed_read(base / "review/function_stage.json")["result"]
    key = next(
        e["request_hash"]
        for e in f["events"]
        if e["step"].startswith("assembly_revision_")
    )
    (base / "calls" / f"{key}.json").unlink()
    assert (
        experiment.report(root)["current_accounting"]["atomic_function_repairs"][
            "missing_call_records"
        ]
        == 1
    )


def test_empty_optional_process_remains_valid(historical, tmp_path):
    source, audit, _ = historical
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    task = next(
        t["construction_task"] for t in plan["tasks"] if t["case"] == "independent"
    )
    client = experiment.pilot.make_client(
        root / "construction",
        plan,
        task,
        "offline",
        transport=smoke.transport_for([], independent=True),
    )
    proposal = experiment.pilot.construct(root / "construction", plan, task, client)
    assert proposal["status"] == "constructed", proposal
    assert not proposal["fallback_used"]


def test_submit_is_four_jobs_and_idempotent(historical, tmp_path, monkeypatch):
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
    assert manifest["tasks"] == 16
    assert [c[0] for c in calls] == ["prepare", "propose", "fit", "report"]
    assert "--array=0-15%4" in calls[2][1]
    assert "--dependency=afterany:103" in calls[3][1]
    assert submitter.submit(root) == manifest
    assert len(calls) == 4
