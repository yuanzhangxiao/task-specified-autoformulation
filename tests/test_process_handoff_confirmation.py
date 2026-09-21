"""Matched inputs, fresh calls, immutable baseline and safe submission."""

import json
import shutil

import pytest

from autoformalism.rebuttal import process_handoff_confirmation as experiment
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import smoke_process_handoff_confirmation as smoke
from scripts import smoke_signed_processes as signed
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


def test_frozen_inputs_only_and_resume(historical, tmp_path):
    source, audit, old = historical
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    assert experiment._matched(plan) == experiment._matched(old)
    assert plan["artifact_sha256"] != old["artifact_sha256"]
    assert {p.name for p in root.iterdir()} == {".lock", "plan.json", "baseline.json"}
    assert experiment.freeze(source, audit, root) == plan
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    # Workers consume their frozen root; historical directories may be offline.
    shutil.rmtree(source)
    shutil.rmtree(audit)
    assert experiment.pilot.verify(root) == plan
    report = experiment.report(root)
    assert report["historical_status_counts"] == {"construction_failed": 16}
    assert report["current_status_counts"] == {"missing": 16}
    assert report["current_accounting"]["constructions_without_usage"] == 8
    assert all(r["current"]["validation"] is None for r in report["rows"])
    assert not report["automatic_followup"]
    assert not report["scientific_compliance_certified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("previously_accepted_now_blocked", [{"task": "regression"}]),
        ("unavailable_saved_attempts", 1),
        ("llm_calls", 1),
        ("optimizer_calls", 1),
        ("test_data_opened", True),
    ],
)
def test_unsuccessful_audit_blocks_submission(historical, tmp_path, field, value):
    source, audit, _ = historical
    rewrite(audit / "summary.json", **{field: value})
    with pytest.raises(ValueError, match="audit gate"):
        experiment.freeze(source, audit, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_source_drift_and_runtime_drift(historical, tmp_path, monkeypatch):
    source, audit, _ = historical
    monkeypatch.setattr(experiment.public, "_runtime", lambda: {"python": "different"})
    with pytest.raises(ValueError, match="runtime differs"):
        experiment.freeze(source, audit, tmp_path / "new")
    path = next((source / "results").glob("*/result.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from the audited"):
        experiment.freeze(source, audit, tmp_path / "new")


def test_separate_root_and_frozen_baseline(historical, tmp_path):
    source, audit, _ = historical
    with pytest.raises(ValueError, match="separate"):
        experiment.freeze(source, audit, source / "new")
    root = tmp_path / "new"
    (root / "construction").mkdir(parents=True)
    with pytest.raises(ValueError, match="already contains"):
        experiment.freeze(source, audit, root)
    (root / "construction").rmdir()
    experiment.freeze(source, audit, root)
    rewrite(root / "baseline.json", rows=[])
    with pytest.raises(ValueError, match="baseline"):
        experiment.pilot.verify(root)


def test_fresh_constructed_pair_counted_once(historical, tmp_path):
    source, audit, _ = historical
    root = tmp_path / "new"
    plan = experiment.freeze(source, audit, root)
    tasks, common, derived, calls, client = signed.construct_pair(root, plan)
    assert calls
    assert len({p["common_proposal_sha256"] for p in derived}) == 1
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
    assert report["current_accounting"]["constructions_without_usage"] == 7
    assert report["current_status_counts"] == {"missing": 16}
    constructed = [
        r["current"]
        for r in report["rows"]
        if r["current"]["proposal_status"] == "constructed"
    ]
    assert len(constructed) == 2
    assert all(r["equation_diagnostics"] is not None for r in constructed)
    assert len(calls) == count
    assert experiment.report(root) == report
    assert json.loads((root / "comparison.json").read_text()) == report


def test_confirmation_scheduler_is_paired_and_idempotent(
    historical, tmp_path, monkeypatch
):
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
