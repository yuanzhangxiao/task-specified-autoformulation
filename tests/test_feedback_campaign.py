"""Training-only archive export, fixed denominators, exact replay and submission."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import feedback_campaign as campaign
from autoformalism.fitting import poor_fit_campaign as historical
from autoformalism.fitting import public_rollout as bridge
from tests.test_adapter_campaign import fake_backend
from tests.test_public_integration import prepared as source_prepared

prepared_source = source_prepared


@pytest.fixture
def exported(prepared_source, tmp_path, monkeypatch):
    source = tmp_path / "m30"
    historical.prepare(source, prepared_source / "inputs.json", historical.Policy())

    def backend(problem, starts, policy, method, folder):
        return fake_backend(problem, starts, policy, method, folder) | {
            "identity": {
                "problem_sha256": campaign.public.content_sha256(
                    problem.model_dump(mode="json")
                ),
                "starts": starts,
                "policy": policy.model_dump(mode="json"),
            },
            "operations": [],
        }

    monkeypatch.setattr(bridge.adapter.recovery, "run", backend)
    monkeypatch.setattr(
        historical.source,
        "evaluate",
        lambda *a: {"validation": "EVALUATOR_SECRET", "reference_values": "FORBIDDEN"},
    )
    for index in (1, 5, 11):
        historical.run_task(source, index)
    snapshot = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "export.json"
    result = campaign.export_inputs(source, output)
    assert result["archived"] == 11 and result["fixtures"] == 5
    assert snapshot == {
        str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    return output, source


@pytest.fixture
def prepared(exported, tmp_path):
    root = tmp_path / "feedback"
    campaign.prepare(root, exported[0], campaign.Policy())
    return root


def test_export_excludes_evaluator_data_and_keeps_all_source_outcomes(exported):
    output, source = exported
    data = read_seal(output)
    text = json.dumps(data)
    assert "EVALUATOR_SECRET" not in text and "FORBIDDEN" not in text
    assert '"evaluation"' not in text and '"validation"' not in text
    assert '"reference_parameters"' not in text
    assert [r["task_id"] for r in data["records"]] == list(campaign.ARCHIVED)
    assert data["records"][0]["fit"] is not None
    assert data["records"][2]["source_status"] == "source_unavailable"
    assert data["records"][1]["source_status"] == "source_result_missing"
    assert data["records"][0]["source_identity"]["arm"] == "output_aware"
    assert campaign.export_inputs(source, output)["tasks"] == 16
    with pytest.raises(ValueError, match="outside"):
        campaign.export_inputs(source, source / "export.json")


def test_export_rejects_source_tampering(exported, tmp_path):
    _, source = exported
    path = source / "arms/output_aware/results/cell00_seed0_full/fit/fit/backend.json"
    value = json.loads(path.read_text())
    value["value"]["retained_parameters"]["a"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        campaign.export_inputs(source, tmp_path / "tampered.json")


def test_inputs_reject_evaluator_injection_and_mismatched_backend(exported):
    data = read_seal(exported[0])
    bad = deepcopy(data)
    bad["records"][0]["problem"]["validation"] = {"untrusted": True}
    with pytest.raises(ValueError, match="Extra inputs"):
        campaign.roster(bad)
    bad = deepcopy(data)
    bad["records"][0]["backend"]["retained_parameters"]["a"] += 1
    with pytest.raises(ValueError, match="backend identity"):
        campaign.roster(bad)
    bad = deepcopy(data)
    bad["records"].pop()
    with pytest.raises(ValueError, match="roster"):
        campaign.roster(bad)


def test_all_denominators_and_unavailable_resume_without_any_fitting(
    prepared, monkeypatch
):
    monkeypatch.setattr(bridge, "run", lambda *a: pytest.fail("unexpected fitting"))
    monkeypatch.setattr(campaign.feedback, "run", lambda *a: pytest.fail("no input"))
    summary = campaign.report(prepared)
    assert summary["status_counts"] == {"missing": 16}
    assert all(value is None for value in summary["diagnostic_costs"].values())
    row = campaign.run_task(prepared, 2)
    assert row["status"] == "source_unavailable" and row["feedback"] is None
    assert not row["fitting_performed"]
    assert campaign.run_task(prepared, 2) == row
    assert campaign.report(prepared)["recorded"] == 1
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, 16)


def test_archive_replay_does_not_refit_and_feedback_resume_is_exact(
    prepared, monkeypatch
):
    monkeypatch.setattr(bridge, "run", lambda *a: pytest.fail("archived fit repeated"))
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "complete" and not row["fitting_performed"]
    assert row["evaluator_labels"] is None
    assert row["original_fit_cost"]["search_calls_started"] == 1
    monkeypatch.setattr(
        campaign.feedback, "run", lambda *a: pytest.fail("repeat check")
    )
    assert campaign.run_task(prepared, 0) == row
    summary = campaign.report(prepared)
    assert summary["diagnostic_costs"][row["task_id"]] == row["feedback"]["cost"]
    path = prepared / "results" / row["task_id"] / "diagnostic_inputs.json"
    bad = read_seal(path)
    bad["fit"]["parameters"]["a"] += 1
    path.unlink()
    seal(path, bad)
    with pytest.raises(ValueError, match="diagnostic input identity"):
        campaign.report(prepared)


def test_changed_policy_inputs_and_runtime_cannot_resume(prepared, monkeypatch):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared,
            prepared / "inputs.json",
            campaign.Policy(feedback={"check_seconds": 181}),
        )
    monkeypatch.setattr(campaign.public, "_runtime", lambda: {"changed": True})
    with pytest.raises(ValueError, match="runtime"):
        campaign.verify(prepared)
    assert campaign.report(prepared)["expected"] == 16


def test_fixture_labels_are_separate_and_added_after_feedback(prepared, monkeypatch):
    _, data = campaign.verify(prepared)
    values = {key: data["records"][0][key] for key in ("problem", "fit", "backend")}
    seen = []

    def fit_fixture(case_id, folder):
        seen.append(case_id)
        return values

    def labels(case_id):
        assert (prepared / "results/fixture__adequate_drainage/feedback.json").exists()
        return None, None, {"explicit_claim": "EVALUATOR_ONLY"}

    monkeypatch.setattr(campaign.fixtures, "fit_fixture", fit_fixture)
    monkeypatch.setattr(campaign.fixtures, "make", labels)
    row = campaign.run_task(prepared, 11)
    assert seen == ["adequate_drainage"]
    assert row["fitting_performed"]
    assert row["evaluator_labels"] == {"explicit_claim": "EVALUATOR_ONLY"}
    assert "EVALUATOR_ONLY" not in json.dumps(row["feedback"])
    saved = read_seal(
        prepared / "results/fixture__adequate_drainage/diagnostic_inputs.json"
    )
    assert set(saved) == {"problem", "fit", "backend"}
    assert campaign.run_task(prepared, 11) == row and len(seen) == 1


def test_unknown_and_partial_diagnostic_costs_remain_unknown():
    rows = [
        {
            "status": "complete",
            "feedback": {
                "cost": {
                    "wall_seconds": None,
                    "cpu_seconds": None,
                    "budget_charge_seconds": 180,
                    "calls": None,
                    "accounting_complete": False,
                }
            },
        },
        {
            "status": "complete",
            "feedback": {
                "cost": {
                    "wall_seconds": 2,
                    "cpu_seconds": 1,
                    "budget_charge_seconds": 2,
                    "calls": 3,
                    "accounting_complete": True,
                }
            },
        },
        {"status": "source_unavailable", "feedback": None},
    ]
    totals = campaign._cost_totals(rows)
    assert totals["wall_seconds"] is None and totals["calls"] is None
    assert totals["known_wall_seconds"] == 2 and totals["known_calls"] == 3
    assert totals["budget_charge_seconds"] == 182
    assert not totals["accounting_complete"]
    rows.append({"status": "missing", "feedback": None})
    assert campaign._cost_totals(rows)["budget_charge_seconds"] is None


def test_fixture_gate_distinguishes_recorded_wrong_and_passed_outcomes():
    cost = {
        "wall_seconds": 2,
        "cpu_seconds": 1,
        "budget_charge_seconds": 2,
        "calls": 3,
        "accounting_complete": True,
    }
    fit_cost = {
        "wall_seconds": 5,
        "cpu_seconds": 4,
        "budget_charge_seconds": 5,
        "search_calls_started": 8,
        "search_calls_completed": 7,
        "accounting_complete": True,
    }
    row = {
        "task_id": "fixture__false_no_drainage_claim",
        "kind": "synthetic_fixture",
        "status": "complete",
        "feedback": {
            "diagnosis": "solved_restricted_problem_poor_fit",
            "specification_verdict": "not_assessed",
            "automatic_specification_revision_allowed": False,
            "diagnostics": {"status": "complete"},
            "cost": cost,
        },
        "original_fit_cost": fit_cost,
        "evaluator_labels": {
            "expected_diagnosis": "solved_restricted_problem_poor_fit",
            "incompatibility_certificate": {"evaluator_only": "CLAIM_CERTIFICATE"},
        },
    }
    rows = [
        row,
        {
            "task_id": "archive__missing_source",
            "kind": "archived",
            "status": "source_unavailable",
            "feedback": None,
        },
    ]
    result = campaign._qualification(rows)
    assert result["qualification_status"] == "passed"
    assert result["fixture_check_status"] == "passed"
    assert result["fixture_checks"][0]["diagnosis_matches"] is True
    assert result["packet_boundary_checks"][1]["status"] == "not_applicable"
    assert "CLAIM_CERTIFICATE" not in json.dumps(result)
    row["feedback"]["diagnostics"]["status"] = "unavailable"
    unavailable = campaign._qualification(rows)
    assert unavailable["qualification_status"] == "failed"
    assert unavailable["diagnostic_check_status"] == "failed"
    assert unavailable["fixture_checks"][0]["diagnosis_matches"] is True
    assert unavailable["fixture_checks"][0]["diagnostics_complete"] is False
    row["feedback"]["diagnostics"]["status"] = "complete"
    row["feedback"]["diagnosis"] = "search_incomplete"
    failed = campaign._qualification(rows)
    assert failed["qualification_status"] == "failed"
    assert failed["qualification_gates"]["all_outcomes_recorded"]
    row["feedback"]["diagnosis"] = "solved_restricted_problem_poor_fit"
    row["feedback"]["automatic_specification_revision_allowed"] = True
    assert campaign._qualification(rows)["packet_boundary_check_status"] == "failed"
    row["feedback"]["automatic_specification_revision_allowed"] = False
    row["feedback"]["cost"]["accounting_complete"] = False
    assert campaign._qualification(rows)["qualification_status"] == "failed"
    row["status"], row["feedback"] = "missing", None
    row.pop("original_fit_cost")
    pending = campaign._qualification(rows)
    assert pending["qualification_status"] == "pending"
    assert pending["fixture_check_status"] == "pending"
    assert pending["fixture_checks"][0]["diagnosis_matches"] is None


def test_fixture_fit_cost_excludes_archive_and_preserves_unknown():
    cost = {
        "wall_seconds": 5,
        "cpu_seconds": 4,
        "budget_charge_seconds": 5,
        "search_calls_started": 8,
        "search_calls_completed": 7,
        "accounting_complete": True,
    }
    rows = [
        {"kind": "synthetic_fixture", "status": "complete", "original_fit_cost": cost},
        {
            "kind": "archived",
            "status": "complete",
            "original_fit_cost": {
                key: 100000 if type(value) is int else value
                for key, value in cost.items()
            },
        },
    ]
    totals = campaign._fixture_cost_totals(rows)
    assert totals["wall_seconds"] == 5 and totals["search_calls_started"] == 8
    assert totals["search_calls_completed"] == 7 and totals["accounting_complete"]
    cost["search_calls_started"] = None
    unknown = campaign._fixture_cost_totals(rows)
    assert unknown["search_calls_started"] is None
    assert not unknown["accounting_complete"]
    rows.append({"kind": "synthetic_fixture", "status": "missing"})
    partial = campaign._fixture_cost_totals(rows)
    assert partial["wall_seconds"] is None and partial["known_wall_seconds"] == 5
    assert partial["missing_tasks"] == 1


def test_scheduler_cpu_short_job_idempotence_and_uncertain_receipts(
    prepared, tmp_path, monkeypatch
):
    from scripts import submit_phase_c_fitting_feedback as submitter

    config = tmp_path / "config.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kwargs):
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
    assert first["array_tasks"] == 16 and first["gpus"] == 0
    assert "--array=0-15%3" in calls[1] and "--time=00:30:00" in calls[1]
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
