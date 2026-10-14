"""M34 immutable controls, evaluator isolation, failed denominators and HPC receipts."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import measurement_campaign as campaign
from autoformalism.fitting import measurement_fixtures as fixtures
from autoformalism.schemas.measurement_fitting import analysis_layout, initial_values


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    result = campaign.prepare(root, None, campaign.Policy())
    assert result["tasks"] == 36
    return root


def fake_fit(problem, policy, folder, *, strategy):
    """Return a deliberately poor fit to test campaign accounting, not numerics."""
    assert "truth" not in problem.model_dump()
    assert "evaluation" not in problem.model_dump()
    return {
        "status": "complete",
        "identity": {
            "protocol": "measurement-fitting-execution-1",
            "source_sha256": campaign.public._source_identity(),
            "runtime": campaign.public._runtime(),
            "problem_sha256": campaign.public.content_sha256(
                problem.model_dump(mode="json")
            ),
            "base_request_sha256": campaign.public.content_sha256(
                problem.request.model_dump(mode="json")
            ),
            "policy": policy.model_dump(mode="json"),
            "strategy": strategy,
        },
        "parameters": dict(problem.start),
        "objective_half_sum_squares": 12.0,
        "verification": {"passed": True},
        "route": {"selected": strategy},
        "cost": {"wall_seconds": 0.1, "accounting_complete": True},
        "confidence_probability": None,
    }


def test_generated_roster_deterministic_sigma_units_and_raw_hashes():
    first, second = fixtures.make_inputs(), fixtures.make_inputs()
    assert first == second
    assert len(campaign.roster(first)) == 36
    assert len(first["problems"]) == 18
    for digest, artifact in first["raw_artifacts"].items():
        assert campaign.public.content_sha256(artifact) == digest
    for name, payload in first["problems"].items():
        problem = campaign.training_problem(first, {"common": name})
        assert len(set(problem.manifest.noise.sigma)) > 1
        assert "truth" not in payload and "analytic_parameters" not in payload
        assert problem.manifest.channels[0].gain == 0.01
        assert problem.manifest.time_bases[0].factor == 1 / 60
        for row in problem.training.rows:
            for reading in row.readings:
                source = first["raw_artifacts"][reading.artifact_sha256]
                original = next(r for r in source["records"] if r["id"] == reading.id)
                assert original["raw_value"] == reading.raw_value
                assert original["condition"] == reading.condition


@pytest.mark.parametrize("case", fixtures.CASES)
@pytest.mark.parametrize("seed", fixtures.SEEDS)
def test_separate_analytic_reference_and_missing_initials(case, seed):
    problem, evaluator = fixtures.make(case, seed)
    layout = analysis_layout(problem)
    assert sum(layout.included) == len(problem.manifest.noise.sigma)
    if case == "nonlinear_decay":
        assert evaluator["analytic_objective"] is None
    else:
        optimum = evaluator["analytic_parameters"]
        answer = fixtures.evaluate(evaluator, optimum)
        assert answer["analytic_objective_passed"]
        assert answer["analytic_objective_gap"] == pytest.approx(0, abs=1e-9)
        assert evaluator["analytic_design_rank"] == len(optimum)
    if case == "active_gain_bound":
        assert evaluator["analytic_active_gain_bound"]
        assert evaluator["analytic_parameters"]["g"] == 0.8
    if case == "missing_readings":
        assert sum(not included for included in layout.included) == 8
        for row in problem.training.rows:
            first = next(r for r in row.readings if r.channel == "y")
            assert first.condition == "missing" and first.raw_value is None
            initial = initial_values(problem, row.trajectory_id, problem.start)["x"]
            assert initial in (0.3, 1.1)
    if case == "shared_initial":
        assert problem.initial_rules[0].shared_coordinate == "x0"
        for row in problem.training.rows:
            assert initial_values(problem, row.trajectory_id, {"g": 1, "x0": 0.2}) == {
                "x": 0.2
            }
    assert np.isfinite(evaluator["observations"][0]["value"])


def test_training_boundary_paired_roster_and_missing_denominators(prepared):
    plan, data = campaign.verify(prepared)
    problem = campaign.training_problem(data, plan["tasks"][0])
    data["evaluation"] = dict.fromkeys(data["evaluation"], "forbidden")
    assert campaign.training_problem(data, plan["tasks"][0]) == problem
    assert campaign.training_problem(data, plan["tasks"][1]) == problem
    summary = campaign.report(prepared)
    assert summary["recorded"] == 0 and summary["expected"] == 36
    assert summary["status_counts"] == {"missing": 36}
    assert len(summary["groups"]) == 12
    assert all(group["expected"] == 3 for group in summary["groups"])
    assert not summary["cost_complete"]
    assert summary["confidence_probability"] is None


def test_objective_masks_calibration_and_config_changes_cannot_resume(prepared):
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(
            prepared,
            prepared / "inputs.json",
            campaign.Policy(maximum_calls=101),
        )
    plan, data = campaign.verify(prepared)
    changed = deepcopy(data)
    key = plan["tasks"][0]["common"]
    changed["problems"][key]["manifest"]["noise"]["sigma"][0] *= 2
    other = prepared.parent / "other-inputs.json"
    seal(other, changed)
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(prepared, other, campaign.Policy())
    data["problems"].pop(key)
    with pytest.raises(ValueError, match="roster"):
        campaign.roster(data)


def test_raw_record_tampering_is_rejected_before_fitting(prepared):
    plan, data = campaign.verify(prepared)
    key = plan["tasks"][0]["common"]
    data["problems"][key]["training"]["rows"][0]["readings"][0]["raw_value"] += 1
    with pytest.raises(ValueError, match="reading differs from raw artifact"):
        campaign.roster(data)


def test_selected_endpoint_sealed_before_evaluator_and_terminal_resume(
    prepared, monkeypatch
):
    calls = []
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    original = campaign.evaluate

    def score(data, task, backend):
        folder = prepared / "results" / task["task_id"]
        assert read_seal(folder / "backend.json") == backend
        calls.append(task["task_id"])
        return original(data, task, backend)

    monkeypatch.setattr(campaign, "evaluate", score)
    first = campaign.run_task(prepared, 0)
    assert first["evaluation"]["analytic_objective_passed"] is False
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("resumed fit")
    )
    assert campaign.run_task(prepared, 0) == first
    assert len(calls) == 1
    assert campaign.report(prepared)["recorded"] == 1
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(prepared, -1)


def test_fit_exception_counted_and_evaluator_interruption_preserves_unknown_cost(
    prepared, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("deliberate numerical failure")

    monkeypatch.setattr(campaign.measurement_fit, "run", broken)
    row = campaign.run_task(prepared, 0)
    assert row["status"] == "failed"
    assert row["evaluation"]["status"] == "no_parameters"
    assert campaign.report(prepared)["status_counts"] == {"failed": 1, "missing": 35}
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interrupted evaluation")

    monkeypatch.setattr(campaign, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt, match="interrupted evaluation"):
        campaign.run_task(prepared, 1)
    partial = campaign.report(prepared)
    assert partial["fit_wall_seconds_observed"] >= 0.1
    assert partial["status_counts"]["missing"] == 35
    assert partial["evaluation_status_counts"]["unfinished"] == 1
    assert not partial["cost_complete"]
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("repeated fit")
    )
    monkeypatch.setattr(
        campaign, "evaluate", lambda *a, **kw: pytest.fail("repeated evaluation")
    )
    resumed = campaign.run_task(prepared, 1)
    assert resumed["status"] == "complete"
    assert resumed["evaluation"]["status"] == "interrupted"
    assert resumed["evaluation"]["wall_seconds"] is None
    assert resumed["evaluation"]["accounting_complete"] is False
    assert campaign.run_task(prepared, 1) == resumed


@pytest.mark.parametrize(
    "field", ["strategy", "problem_sha256", "policy", "source_sha256"]
)
def test_intermediate_backend_identity_mismatch_rejected(prepared, monkeypatch, field):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    row = campaign.run_task(prepared, 0)
    folder = prepared / "results" / row["task_id"]
    (folder / "result.json").unlink()
    backend = read_seal(folder / "backend.json")
    backend["identity"][field] = "wrong"
    (folder / "backend.json").unlink()
    seal(folder / "backend.json", backend)
    monkeypatch.setattr(
        campaign.measurement_fit, "run", lambda *a, **kw: pytest.fail("refitted")
    )
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 0)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.report(prepared)


def test_copied_other_arm_backend_and_wrong_endpoint_evaluation_rejected(
    prepared, monkeypatch
):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    original = campaign.evaluate
    direct = campaign.run_task(prepared, 0)
    plan, _ = campaign.verify(prepared)
    folder = prepared / "results" / plan["tasks"][1]["task_id"]
    seal(folder / "backend.json", direct["backend"])
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 1)
    (folder / "backend.json").unlink()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 1)
    seal(folder / "evaluation.json", direct["evaluation"])
    monkeypatch.setattr(campaign, "evaluate", original)
    with pytest.raises(ValueError, match="evaluation identity differs"):
        campaign.run_task(prepared, 1)


def test_terminal_reuse_validates_nested_execution_identity(prepared, monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    row = campaign.run_task(prepared, 0)
    folder = prepared / "results" / row["task_id"]
    row["backend"]["identity"]["strategy"] = "profiled"
    row["backend_sha256"] = campaign.public.content_sha256(row["backend"])
    for filename, value in (("backend.json", row["backend"]), ("result.json", row)):
        (folder / filename).unlink()
        seal(folder / filename, value)
    with pytest.raises(ValueError, match="backend identity differs"):
        campaign.run_task(prepared, 0)


def test_failed_evaluation_is_retained_with_measured_cost(prepared, monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)

    def failed(*args, **kwargs):
        raise ValueError("deliberate evaluator error")

    monkeypatch.setattr(campaign, "evaluate", failed)
    row = campaign.run_task(prepared, 0)
    assert row["backend"]["parameters"] is not None
    assert row["evaluation"]["status"] == "failed"
    assert row["evaluation"]["accounting_complete"] is True
    assert row["evaluation"]["wall_seconds"] >= 0
    monkeypatch.setattr(
        campaign, "evaluate", lambda *a, **kw: pytest.fail("repeated evaluation")
    )
    assert campaign.run_task(prepared, 0) == row


def test_partial_fitter_result_cost_is_counted_without_terminal_row(prepared):
    plan, data = campaign.verify(prepared)
    task = plan["tasks"][0]
    backend = fake_fit(
        campaign.training_problem(data, task),
        campaign.Policy.model_validate(plan["policy"]),
        prepared,
        strategy=task["strategy"],
    )
    seal(prepared / "results" / task["task_id"] / "fit/result.json", backend)
    summary = campaign.report(prepared)
    assert summary["recorded"] == 0 and summary["status_counts"] == {"missing": 36}
    assert summary["fit_wall_seconds_observed"] == 0.1
    assert not summary["cost_complete"]


def test_modified_backend_seal_is_rejected(prepared, monkeypatch):
    monkeypatch.setattr(campaign.measurement_fit, "run", fake_fit)
    first = campaign.run_task(prepared, 0)
    path = prepared / "results" / first["task_id"] / "backend.json"
    backend = read_seal(path)
    backend["parameters"]["g"] += 0.1
    path.unlink()
    seal(path, backend)
    with pytest.raises(ValueError, match="evaluation identity differs"):
        campaign.run_task(prepared, 0)


def test_submission_idempotent_one_cpu_and_uncertain_receipt(prepared, monkeypatch):
    from scripts import submit_phase_c_measurement_fitting as submitter

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "test-commit"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(200 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", sbatch)
    config = Path("configs/phase_c_measurement_fitting_v1.json")
    first = submitter.submit(prepared, config, account="test", concurrency=2)
    assert first["array_tasks"] == 36 and first["gpus"] == 0
    assert "--array=0-35%2" in calls[1]
    assert "--cpus-per-task=1" in calls[1] and "--time=00:10:00" in calls[1]
    assert "--dependency=afterany:201:202" in calls[2]
    assert submitter.submit(prepared, config, account="test", concurrency=2) == first
    assert len(calls) == 3
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr="uncertain"),
    )
    unknown = prepared.parent / "uncertain"
    with pytest.raises(ValueError, match="unconfirmed"):
        submitter.submit(unknown, config, account="test")
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("resubmitted"),
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(unknown, config, account="test")


def test_hpc_shell_syntax_and_cli_help():
    for kind in ("submit", "run", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_measurement_fitting_delta.sh"],
            check=True,
        )
    result = subprocess.run(
        [sys.executable, "scripts/phase_c_measurement_fitting.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "prepare,verify,run,report" in result.stdout
