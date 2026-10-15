"""Real local controls, sealed resume and the standalone interaction-feedback CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import interaction_feedback_qualification as qualification
from scripts import phase_c_interaction_feedback as cli


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    """Keep concurrent collaborators' unrelated edits outside fixture identities."""
    monkeypatch.setattr(
        qualification.public,
        "_source_identity",
        lambda: "m37-qualification-test-source",
    )


@pytest.fixture(scope="module")
def qualified(tmp_path_factory):
    """Fit and diagnose all four independent controls once through production code."""
    root = tmp_path_factory.mktemp("interaction-qualification")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            qualification.public,
            "_source_identity",
            lambda: "m37-qualification-test-source",
        )
        summary = qualification.qualify(root)
    return root, summary


def _replace(path: Path, value: dict) -> None:
    path.unlink()
    seal(path, value)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_real_control_qualification_and_separate_costs(qualified):
    root, summary = qualified
    assert summary["qualification_status"] == "passed", summary
    assert summary["expected"] == summary["recorded"] == summary["passed"] == 4
    assert summary["failed"] == summary["pending"] == 0
    assert [row["case_id"] for row in summary["rows"]] == list(qualification.CASES)
    assert not summary["test_data_opened"] and not summary["validation_used"]
    assert not summary["benchmark_claim"] and summary["live_llm_calls"] == 0
    for row in summary["rows"]:
        assert all(row["checks"].values())
        assert row["fit_cost"]["accounting_complete"]
        assert row["diagnostic_cost"]["optimization_calls"] == 0
        overhead = row["adapter_cost"]
        assert overhead["export_calls"] == overhead["verification_calls"] == 1
        assert overhead["export_wall_seconds"] > 0
        assert overhead["verification_wall_seconds"] > 0
        assert overhead["optimization_calls"] == overhead["rollout_calls"] == 0
        assert overhead["llm_calls"] == 0
    inputs = read_seal(root / "inputs.json")
    assert "expected_" not in json.dumps(inputs)
    assert "expected_diagnosis" in json.dumps(
        read_seal(root / "evaluator_expectations.json")
    )
    for case in qualification.CASES:
        folder = root / "results" / case
        for name in (
            "diagnostic_inputs.json",
            "diagnostics/context.json",
            "interaction/request.json",
        ):
            assert "expected_diagnosis" not in json.dumps(read_seal(folder / name))


def test_finished_resume_and_report_do_not_execute_work(qualified, monkeypatch):
    root, expected = qualified
    before = _snapshot(root)

    def unexpected(*args, **kwargs):
        pytest.fail("completed qualification attempted new computational work")

    monkeypatch.setattr(qualification.fixtures, "fit_fixture", unexpected)
    monkeypatch.setattr(qualification.staged, "run", unexpected)
    monkeypatch.setattr(qualification.feedback, "export_packet", unexpected)
    assert qualification.qualify(root) == expected
    assert qualification.report(root) == expected
    assert _snapshot(root) == before


def test_prepared_report_retains_entire_denominator_without_work(tmp_path, monkeypatch):
    qualification.prepare(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("report performed numerical work")

    monkeypatch.setattr(qualification.fixtures, "fit_fixture", unexpected)
    monkeypatch.setattr(qualification.staged, "run", unexpected)
    monkeypatch.setattr(qualification.feedback, "export_packet", unexpected)
    before = _snapshot(tmp_path)
    summary = qualification.report(tmp_path)
    assert summary["expected"] == summary["pending"] == 4
    assert summary["recorded"] == summary["passed"] == summary["failed"] == 0
    assert summary["qualification_status"] == "pending"
    assert _snapshot(tmp_path) == before


def test_failed_control_is_checkpointed_and_never_silently_retried(
    tmp_path, monkeypatch
):
    qualification.prepare(tmp_path)
    attempts = []

    def fail(case_id, folder):
        attempts.append(case_id)
        raise TimeoutError("bounded synthetic control unavailable")

    monkeypatch.setattr(qualification.fixtures, "fit_fixture", fail)
    first = qualification.run_task(tmp_path, 1)
    assert first["status"] == "failed"
    assert first["failure"]["stage"] == "fixture"
    assert first["failure"]["error_type"] == "TimeoutError"
    assert first["adapter_cost"]["export_calls"] == 0
    assert qualification.run_task(tmp_path, 1) == first
    assert attempts == ["stationary_poor_retained"]
    summary = qualification.report(tmp_path)
    assert summary["expected"] == 4 and summary["recorded"] == 1
    assert summary["failed"] == 1 and summary["pending"] == 3
    assert summary["qualification_status"] == "failed"
    failed = summary["rows"][1]
    assert failed["fit_cost"] is failed["diagnostic_cost"] is None
    for name in ("fit", "diagnostic"):
        accounting = failed[name + "_cost_accounting"]
        assert accounting["status"] == "unavailable"
        assert not accounting["accounting_complete"]
        assert not accounting["unrecorded_work_assumed_zero"]


@pytest.mark.parametrize("stage", ["diagnostics", "export"])
def test_later_failure_preserves_verified_costs_and_partial_provenance(
    tmp_path, monkeypatch, stage
):
    qualification.prepare(tmp_path)
    if stage == "diagnostics":
        real_operation = qualification.staged.checks.operation

        def fail_after_primary(folder, *args, **kwargs):
            if folder.name == "alternate" and folder.parent.name == "stages":
                raise RuntimeError("diagnostic task failed after its primary rollout")
            return real_operation(folder, *args, **kwargs)

        monkeypatch.setattr(
            qualification.staged.checks, "operation", fail_after_primary
        )
    else:

        def fail_export(*args, **kwargs):
            raise ValueError("export byte cap exceeded")

        monkeypatch.setattr(qualification.feedback, "export_packet", fail_export)
    saved = qualification.run_task(tmp_path, 0)
    assert saved["status"] == "failed" and saved["failure"]["stage"] == stage
    folder = tmp_path / "results/initializer_path"
    before = _snapshot(folder)
    summary = qualification.report(tmp_path)
    assert summary["failed"] == 1 and summary["pending"] == 3
    row = summary["rows"][0]
    assert row["fit_cost"] == read_seal(folder / "fixture/fit/result.json")["cost"]
    assert row["fit_cost"]["wall_seconds"] > 0
    fit_accounting = row["fit_cost_accounting"]
    assert fit_accounting["status"] == "verified"
    assert fit_accounting["accounting_complete"]
    assert "fixture/fit/backend.json" in fit_accounting["artifact_sha256"]
    diagnostic_accounting = row["diagnostic_cost_accounting"]
    assert (
        "diagnostics/stages/primary/record.json"
        in diagnostic_accounting["artifact_sha256"]
    )
    if stage == "diagnostics":
        assert row["diagnostic_cost"] is None
        assert diagnostic_accounting["status"] == "unavailable"
        assert not diagnostic_accounting["accounting_complete"]
        assert (
            diagnostic_accounting["artifact_verification"]
            == "seals_only_not_complete_accounting"
        )
    else:
        expected = qualification.staged.completed(folder / "diagnostics")
        assert row["diagnostic_cost"] == expected.cost.model_dump(mode="json")
        assert row["diagnostic_cost"]["wall_seconds"] > 0
        assert diagnostic_accounting["status"] == "verified"
        assert diagnostic_accounting["accounting_complete"]
    assert not diagnostic_accounting["unrecorded_work_assumed_zero"]
    assert qualification.run_task(tmp_path, 0) == saved
    assert qualification.report(tmp_path) == summary
    assert _snapshot(folder) == before


@pytest.mark.parametrize("index", [-1, 4])
def test_invalid_index_creates_no_experiment(tmp_path, index):
    with pytest.raises(ValueError, match="task index"):
        qualification.qualify(tmp_path, index)
    assert not (tmp_path / "plan.json").exists()


def test_interrupted_task_resumes_existing_fixture_checkpoint(tmp_path, monkeypatch):
    qualification.prepare(tmp_path)
    real_fit = qualification.fixtures.fit_fixture

    def interrupt(case_id, folder):
        real_fit(case_id, folder)
        raise KeyboardInterrupt

    monkeypatch.setattr(qualification.fixtures, "fit_fixture", interrupt)
    with pytest.raises(KeyboardInterrupt):
        qualification.run_task(tmp_path, 0)
    folder = tmp_path / "results/initializer_path"
    assert not (folder / "result.json").exists()
    fitted = _snapshot(folder / "fixture")
    monkeypatch.setattr(qualification.fixtures, "fit_fixture", real_fit)
    assert qualification.run_task(tmp_path, 0)["status"] == "complete"
    assert _snapshot(folder / "fixture") == fitted
    summary = qualification.report(tmp_path)
    assert summary["passed"] == 1 and summary["pending"] == 3


@pytest.mark.parametrize("change", ["source", "runtime"])
def test_changed_runtime_blocks_execution_but_preserves_read_only_report(
    qualified, monkeypatch, change
):
    root, expected = qualified
    if change == "source":
        monkeypatch.setattr(qualification.public, "_source_identity", lambda: "changed")
    else:
        monkeypatch.setattr(qualification.public, "_runtime", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source/runtime"):
        qualification.run_task(root, 0)
    assert qualification.report(root) == expected


@pytest.mark.parametrize(
    "artifact",
    [
        "plan.json",
        "inputs.json",
        "evaluator_expectations.json",
        "results/initializer_path/result.json",
    ],
)
def test_resealed_extra_fields_are_rejected(qualified, tmp_path, artifact):
    shutil.copytree(qualified[0], tmp_path, dirs_exist_ok=True)
    path = tmp_path / artifact
    value = read_seal(path)
    _replace(path, value | {"untrusted_evaluator_channel": "injected"})
    with pytest.raises(ValueError, match="Extra inputs"):
        qualification.report(tmp_path)


def test_resealed_expectation_and_diagnostic_input_changes_are_rejected(
    qualified, tmp_path
):
    shutil.copytree(qualified[0], tmp_path, dirs_exist_ok=True)
    path = tmp_path / "evaluator_expectations.json"
    original = read_seal(path)
    changed = json.loads(json.dumps(original))
    changed["controls"][0]["expected_diagnosis"] = "invented"
    _replace(path, changed)
    with pytest.raises(ValueError, match="input/expectation identity"):
        qualification.report(tmp_path)
    _replace(path, original)
    path = tmp_path / "results/initializer_path/diagnostic_inputs.json"
    value = read_seal(path)
    value["fit"]["parameters"]["init_z_value"] += 0.1
    _replace(path, value)
    with pytest.raises(ValueError, match="fitted inputs"):
        qualification.report(tmp_path)


def test_cli_prepare_report_and_single_task(tmp_path, monkeypatch, capsys):
    for command in ("prepare", "report", "qualify"):
        argv = ["phase_c_interaction_feedback.py", command, "--root", str(tmp_path)]
        if command == "qualify":
            argv += ["--index", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        result = json.loads(capsys.readouterr().out)
        if command == "prepare":
            assert result["tasks"] == 4
        elif command == "report":
            assert result["pending"] == 4
        else:
            assert result["passed"] == 1 and result["pending"] == 3


def test_cli_export_and_verify_use_verified_source(
    qualified, tmp_path, monkeypatch, capsys
):
    source = qualified[0] / "results/initializer_path/diagnostics"
    packets = []
    for command in ("export", "verify"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "phase_c_interaction_feedback.py",
                command,
                "--diagnostics",
                str(source),
                "--output",
                str(tmp_path),
            ],
        )
        cli.main()
        packets.append(json.loads(capsys.readouterr().out))
    assert packets[0] == packets[1]
    assert packets[0]["status"] == "ready_for_advisory_review"


def test_cli_is_independently_runnable():
    result = subprocess.run(
        [sys.executable, "scripts/phase_c_interaction_feedback.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "export,verify,prepare,qualify,report" in result.stdout
