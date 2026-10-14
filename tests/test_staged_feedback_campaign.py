"""M32 sealed export, fixed-vector replay, cost boundaries and durable submission."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import feedback_campaign as historical
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import staged_feedback_campaign as campaign
from tests.test_feedback_campaign import exported as m30_exported
from tests.test_feedback_campaign import prepared_source

__all__ = ["m30_exported", "prepared_source"]


@pytest.fixture
def source(m30_exported, tmp_path, monkeypatch):
    """Make a sealed M31 archive with all five previous fixtures already fitted."""
    root = tmp_path / "m31"
    historical.prepare(root, m30_exported[0], historical.Policy())
    data = read_seal(m30_exported[0])
    values = {key: data["records"][0][key] for key in ("problem", "fit", "backend")}
    monkeypatch.setattr(historical.fixtures, "fit_fixture", lambda *args: values)
    for index in range(16):
        historical.run_task(root, index)
    return root


@pytest.fixture
def exported(source, tmp_path, monkeypatch):
    """Export old fits without invoking either the bridge or fixture fitting."""

    def fail(*args, **kwargs):
        pytest.fail("historical refitting prohibited")

    monkeypatch.setattr(bridge, "run", fail)
    monkeypatch.setattr(historical.fixtures, "fit_fixture", fail)
    monkeypatch.setattr(campaign.fixtures, "fit_fixture", fail)
    snapshot = {
        str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    output = tmp_path / "staged-inputs.json"
    campaign.export_inputs(source, output)
    assert snapshot == {
        str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    return output


@pytest.fixture
def prepared(exported, tmp_path):
    root = tmp_path / "m32"
    campaign.prepare(root, exported, campaign.Policy())
    return root


def test_export_keeps_all_m31_fits_statuses_and_excludes_labels(exported, source):
    data = read_seal(exported)
    assert [record["task_id"] for record in data["records"]] == list(campaign.ARCHIVED)
    assert len(data["records"]) == 16
    old_fixtures = [
        record
        for record in data["records"]
        if record["source_kind"] == "synthetic_fixture"
    ]
    assert len(old_fixtures) == 5 and all(
        record["fit"] is not None for record in old_fixtures
    )
    assert data["records"][2]["source_status"] == "source_unavailable"
    assert data["records"][0]["prior_feedback"]["protocol"] == "fitting-feedback-1"
    text = json.dumps(data)
    assert "evaluator_labels" not in text and "expected_diagnosis" not in text
    assert "EVALUATOR_SECRET" not in text and "reference_parameters" not in text
    assert '"validation"' not in text
    assert campaign.export_inputs(source, exported)["tasks"] == 16 + len(
        campaign.fixtures.roster()
    )
    with pytest.raises(ValueError, match="outside"):
        campaign.export_inputs(source, source / "export.json")


def test_export_rejects_changed_sealed_source(source, tmp_path):
    path = source / "results/fixture__adequate_drainage/diagnostic_inputs.json"
    value = json.loads(path.read_text())
    value["value"]["fit"]["parameters"]["a"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        campaign.export_inputs(source, tmp_path / "rejected.json")


def test_input_schema_rejects_labels_backend_packet_and_roster_tampering(exported):
    original = read_seal(exported)
    for field, value, message in (
        ("evaluator_labels", {"forbidden": True}, "Extra inputs"),
        (
            "problem",
            original["records"][0]["problem"] | {"validation": {}},
            "Extra inputs",
        ),
    ):
        data = deepcopy(original)
        data["records"][0][field] = value
        with pytest.raises(ValueError, match=message):
            campaign.roster(data)
    data = deepcopy(original)
    data["records"][0]["backend"]["retained_parameters"]["a"] += 1
    with pytest.raises(ValueError, match="backend identity"):
        campaign.roster(data)
    data = deepcopy(original)
    data["records"][0]["prior_feedback"]["identity"]["fit_sha256"] = "changed"
    with pytest.raises(ValueError, match="feedback identity"):
        campaign.roster(data)
    data = deepcopy(original)
    data["records"].pop()
    with pytest.raises(ValueError, match="roster"):
        campaign.roster(data)


def test_unavailable_rows_remain_in_denominator_and_resume_without_diagnostics(
    prepared, monkeypatch
):
    monkeypatch.setattr(
        campaign.feedback, "run", lambda *args: pytest.fail("unavailable source")
    )
    summary = campaign.report(prepared)
    assert summary["expected"] == 16 + len(campaign.fixtures.roster())
    assert summary["recorded"] == 0
    row = campaign.run_task(prepared, 2)
    assert row["status"] == "source_unavailable" and row["feedback"] is None
    assert row["fitting_performed"] is False
    assert campaign.run_task(prepared, 2) == row
    assert campaign.report(prepared)["recorded"] == 1
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, summary["expected"])


def test_archive_and_old_fixture_replay_never_refit_and_tamper_fails(
    prepared, monkeypatch
):
    row = campaign.run_task(prepared, 11)
    assert row["kind"] == "archived_fixture"
    assert row["fitting_performed"] is False
    assert row["historical_fit_cost"] is not None
    assert row["new_fixture_fit_cost"] is None
    assert row["prior_feedback"] is not None
    monkeypatch.setattr(
        campaign.feedback, "run", lambda *args: pytest.fail("diagnostic repeated")
    )
    assert campaign.run_task(prepared, 11) == row
    path = prepared / "results" / row["task_id"] / "diagnostic_inputs.json"
    value = read_seal(path)
    value["fit"]["parameters"]["a"] += 1
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError, match="diagnostic input identity"):
        campaign.report(prepared)


def test_new_fixture_labels_are_added_only_after_feedback(prepared, monkeypatch):
    _, data = campaign.verify(prepared)
    values = {key: data["records"][0][key] for key in ("problem", "fit", "backend")}
    calls = []

    def fit_fixture(case_id, folder):
        calls.append(case_id)
        return values

    def labels(case_id):
        assert (
            prepared / "results" / ("fixture__" + case_id) / "feedback.json"
        ).exists()
        return None, None, {"private_claim": "EVALUATOR_ONLY"}

    monkeypatch.setattr(campaign.fixtures, "fit_fixture", fit_fixture)
    monkeypatch.setattr(campaign.fixtures, "make", labels)
    row = campaign.run_task(prepared, 16)
    assert row["fitting_performed"]
    assert (
        row["historical_fit_cost"] is None and row["new_fixture_fit_cost"] is not None
    )
    assert "EVALUATOR_ONLY" not in json.dumps(row["feedback"])
    saved = read_seal(prepared / "results" / row["task_id"] / "diagnostic_inputs.json")
    assert set(saved) == {"problem", "fit", "backend"}
    assert campaign.run_task(prepared, 16) == row and len(calls) == 1


def test_changed_policy_and_runtime_cannot_resume(prepared, monkeypatch):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared,
            prepared / "inputs.json",
            campaign.Policy(feedback={"primary_seconds": 121}),
        )
    monkeypatch.setattr(campaign.public, "_runtime", lambda: {"changed": True})
    with pytest.raises(ValueError, match="runtime"):
        campaign.verify(prepared)
    assert campaign.report(prepared)["expected"] >= 16


def _row(*, derivative="unsupported_kink"):
    cost = {
        "wall_seconds": 2,
        "cpu_seconds": 1,
        "budget_charge_seconds": 2,
        "calls": 3,
        "calls_completed": 2,
        "accounting_complete": True,
    }
    return {
        "task_id": "fixture__piecewise_kink",
        "kind": "synthetic_fixture",
        "status": "complete",
        "feedback": {
            "diagnosis": "accurate_unresolved",
            "recommended_action": "assess_parameter_uncertainty",
            "specification_verdict": "not_assessed",
            "automatic_specification_revision_allowed": False,
            "diagnostics": {
                "output_status": "verified",
                "derivative_scope": "piecewise_branch_convention",
                "piecewise_assessment": {"status": derivative},
            },
            "cost": cost,
        },
        "new_fixture_fit_cost": {
            "wall_seconds": 1,
            "cpu_seconds": 1,
            "budget_charge_seconds": 1,
            "search_calls_started": 0,
            "search_calls_completed": 0,
            "accounting_complete": True,
        },
        "evaluator_labels": {
            "expected_diagnosis": "accurate_unresolved",
            "expected_derivative_support": "unsupported_kink",
            "private_claim": "EVALUATOR_ONLY",
        },
    }


def test_qualification_preserves_negative_piecewise_control_and_required_output_gate():
    row = _row()
    passed = campaign._qualification([row])
    assert passed["qualification_status"] == "passed"
    assert "EVALUATOR_ONLY" not in json.dumps(passed)
    row["feedback"]["diagnostics"]["output_status"] = "unavailable"
    failed = campaign._qualification([row])
    assert failed["qualification_status"] == "failed"
    assert failed["qualification_gates"]["required_outputs_verified"] == "failed"
    row["feedback"]["diagnostics"]["output_status"] = "verified"
    row["feedback"]["diagnostics"]["piecewise_assessment"]["status"] = "supported"
    assert campaign._qualification([row])["qualification_status"] == "failed"


def test_new_costs_exclude_historical_timeout_and_keep_unknown_usage():
    row = _row()
    row["prior_feedback"] = {
        "cost": {
            "wall_seconds": 180,
            "cpu_seconds": 175,
            "budget_charge_seconds": 180,
            "calls": 2,
            "accounting_complete": True,
        }
    }
    assert campaign._cost_totals([row])["wall_seconds"] == 2
    row["feedback"]["cost"]["calls_completed"] = None
    total = campaign._cost_totals([row])
    assert total["calls_completed"] is None and not total["accounting_complete"]
    assert total["known_wall_seconds"] == 2
    total = campaign._cost_totals([row, {"status": "missing", "feedback": None}])
    assert total["wall_seconds"] is None


def test_delta_cpu_resources_idempotence_and_uncertain_receipts(
    prepared, tmp_path, monkeypatch
):
    from scripts import submit_phase_c_staged_feedback as submitter

    config = tmp_path / "config.json"
    config.write_text(campaign.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *args, **kwargs: "test"
    )
    kwargs = {
        "account": "test-cpu",
        "concurrency": 3,
        "inputs": prepared / "inputs.json",
    }
    first = submitter.submit(prepared, config, **kwargs)
    assert (
        first["array_tasks"] == 16 + len(campaign.fixtures.roster())
        and first["gpus"] == 0
    )
    assert f"--array=0-{first['array_tasks'] - 1}%3" in calls[1]
    assert "--time=03:00:00" in calls[1]
    assert "--mem=16G" in calls[1] and "--cpus-per-task=1" in calls[1]
    assert submitter.submit(prepared, config, **kwargs) == first and len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr="timeout"
        ),
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(tmp_path / "unknown", config, **kwargs)
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("duplicate"),
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(tmp_path / "unknown", config, **kwargs)
