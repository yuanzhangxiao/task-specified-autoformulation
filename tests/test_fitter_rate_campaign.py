"""Historical import provenance, paired starts, and frozen-parameter retry."""

from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.rebuttal import fitter_methods as old
from autoformalism.rebuttal import fitter_rate_refinement as new
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.fitter_offset import verify_replays
from autoformalism.staged_topology import content_hash
from tests.test_fitter_methods import small_plan


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Small generated fixture; production importer still pins full v4 hashes."""
    plan = small_plan().model_copy(
        update={"protocol": "fitter-methods-4", "guard_seconds": 100}
    )
    root = tmp_path / "old"
    frozen = old.prepare_methods(plan, root)
    guard = old.execute_methods(root, 0)
    assert guard["status"] == "complete", guard
    task = next(t for t in frozen["tasks"] if t["kind"] == "fit")
    truth = frozen["cases"][0]["truth"]
    identity = content_hash([frozen["freeze_sha256"], task])
    _, _, model, _, _, dataset, scale, _ = old._fit_inputs(root, frozen, task)
    replays = verify_replays(
        root / "results" / task["name"] / "replays",
        identity,
        model,
        dataset,
        truth,
        plan.settings(),
        plan,
        scale,
    )
    result = {
        **replays,
        "identity": identity,
        "task": task,
        "parameters": truth,
        "refinement_start": truth,
        "initializer": {"seconds": 1, "parameters": truth, "success": True},
    }
    write_json(root / "results" / task["name"] / "result.json", result)
    monkeypatch.setattr(
        new,
        "SOURCE",
        {
            "source_sha256": frozen["runtime"]["source_sha256"],
            "launcher": frozen["launcher"],
            "plan": content_hash(frozen["plan"]),
            "tasks": content_hash(frozen["tasks"]),
        },
    )
    return root, task, result


def test_import_pair_resume_and_array_tampering(source, tmp_path, monkeypatch):
    root, task, original = source
    output = tmp_path / "new"
    plan = new.RatePlan(
        source_tasks=(task["name"],), guard_seconds=30, replay_seconds=10
    )
    frozen = new.prepare_rate(plan, root, output)
    result = new.execute_rate(output, 0)
    assert result["status"] == "complete", result
    assert result["collocation_rerun"] is False
    physical, rate = result["arms"]["physical"], result["arms"]["rates"]
    assert physical["nominal_start"] == original["refinement_start"]
    assert rate["nominal_start"]["rate_tau"] == 1 / physical["nominal_start"]["tau"]
    assert (
        physical["refinement_budget_seconds"] == rate["refinement_budget_seconds"] == 29
    )
    assert rate["clean_signal_nmse"]["validation"] < 1e-8
    assert new.summarize_rate(output)["rows"][0]["recovered"]

    def unexpected(*args, **kwargs):
        raise AssertionError("completed fit restarted")

    monkeypatch.setattr(new, "refine", unexpected)
    assert new.execute_rate(output, 0) == read_json(
        output / "results" / frozen["tasks"][0]["name"] / "result.json"
    )
    assert new.prepare_rate(plan, root, output) == frozen
    candidate = output / "source/candidate_moderate.json"
    candidate.write_text(candidate.read_text() + " ")
    with pytest.raises(ValueError, match="source artifact"):
        new.verify_rate(output)


def test_retry_only_missing_trajectory_and_preserves_source(
    source, tmp_path, monkeypatch
):
    root, task, original = source
    original["status"] = "replay_unverified"
    old_record = original["replays"]["Radau_tight/train"]["trajectories"][0]
    old_record.update(status="timeout", error="fitting wall-clock limit reached")
    original["replays"]["Radau_tight/train"].update(
        status="incomplete", normalized_mse=None
    )
    original_path = root / "results" / task["name"] / "result.json"
    write_json(original_path, original)
    before = sha256(original_path)
    output = tmp_path / "retry"
    new.prepare_rate(
        new.RatePlan(source_tasks=(task["name"],), replay_seconds=10), root, output
    )

    def no_fit(*args, **kwargs):
        raise AssertionError("retry must not fit")

    monkeypatch.setattr(new, "refine", no_fit)
    from autoformalism.rebuttal import fitter_offset

    real_simulate = fitter_offset.simulate_trajectory
    calls = []

    def count(model, row, parameters, *args, **kwargs):
        calls.append(row.trajectory_id)
        assert parameters == original["parameters"]
        return real_simulate(model, row, parameters, *args, **kwargs)

    monkeypatch.setattr(fitter_offset, "simulate_trajectory", count)
    retried = new.execute_rate(output, 1)
    assert retried["status"] == "complete", retried
    assert calls == [old_record["trajectory_id"]]
    assert retried["fit_rerun"] is False
    assert sha256(original_path) == before
    new.execute_rate(output, 1)
    assert len(calls) == 1


def test_import_rejects_missing_selection_and_wrong_source(source, tmp_path):
    root, task, _ = source
    with pytest.raises(ValueError, match="distinct v4"):
        new.prepare_rate(
            new.RatePlan(source_tasks=("not_a_task",)), root, tmp_path / "invalid"
        )
    path = root / "freeze.json"
    frozen = read_json(path)
    frozen["runtime"]["source_sha256"] = "changed"
    frozen["freeze_sha256"] = content_hash(
        {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    )
    write_json(path, frozen)
    with pytest.raises(ValueError, match="pinned v4"):
        new.prepare_rate(
            new.RatePlan(source_tasks=(task["name"],)), root, tmp_path / "wrong"
        )


def test_scheduler_is_cpu_only_and_partial_submission_is_guarded():
    root = Path(__file__).resolve().parents[1]
    shell = (root / "scripts/hpc/submit_fitter_rate_refinement_delta.sh").read_text()
    worker = (root / "scripts/hpc/fitter_rate_refinement_delta.slurm").read_text()
    assert "submission.intent" in shell and "submission_complete" in shell
    assert "--cpus-per-task=1" in worker and "--gres" not in worker
    assert "00:45:00" in worker
    assert new.RatePlan().worker_seconds("pair") < 45 * 60
    assert "--time=00:30:00" in shell


def test_stopping_pair_changes_only_ftol_and_preserves_missing_counts(
    source, tmp_path, monkeypatch
):
    root, task, original = source
    original["status"] = "replay_unverified"
    write_json(root / "results" / task["name"] / "result.json", original)
    output = tmp_path / "stopping"
    plan = new.RatePlan(
        protocol="fitter-rate-stopping-1",
        source_tasks=(task["name"],),
        guard_seconds=30,
        replay_seconds=10,
    )
    frozen = new.prepare_rate(plan, root, output)
    assert len(frozen["tasks"]) == 1  # Previous verification retry is already resolved.
    missing = new.summarize_rate(output)
    assert {r["arm"] for r in missing["rows"]} == {"rates_default", "rates_no_ftol"}
    assert all(
        g["total"] == 1 and g["recovered"] == 0
        for g in missing["recovery_by_initializer"]
    )
    options = []
    real = new.least_squares

    def spy(fun, x, **kwargs):
        options.append(
            {k: kwargs[k] for k in ("ftol", "xtol", "gtol", "x_scale", "max_nfev")}
        )
        return real(fun, x, **kwargs)

    monkeypatch.setattr(new, "least_squares", spy)
    result = new.execute_rate(output, 0)
    assert result["status"] == "complete", result
    left, right = (result["arms"][n] for n in ("rates_default", "rates_no_ftol"))
    assert left["nominal_start"] == right["nominal_start"]
    assert left["first_evaluated_parameters"] == right["first_evaluated_parameters"]
    assert left["refinement_budget_seconds"] == right["refinement_budget_seconds"] == 29
    assert options[0] == {**options[1], "ftol": 1e-8}
    assert options[1]["ftol"] is None
    assert left["parameter_order"] == right["parameter_order"]
    assert left["raw_gradient_inf_norm"] is not None
    summary = new.summarize_rate(output)
    assert all(r["recovered"] for r in summary["rows"])
    assert "ftol" in (output / "summary.md").read_text()

    def unexpected(*args, **kwargs):
        raise AssertionError("checkpointed optimizer restarted")

    monkeypatch.setattr(new, "least_squares", unexpected)
    assert new.execute_rate(output, 0) == read_json(
        output / "results" / frozen["tasks"][0]["name"] / "result.json"
    )
    with pytest.raises(ValueError, match="plan differs"):
        new.prepare_rate(
            plan.model_copy(update={"protocol": "fitter-rate-refinement-1"}),
            root,
            output,
        )


def test_recovery_groups_do_not_confuse_joint_and_alternating():
    rows = [
        {
            "source": "fit_separated_noise1_rep2_" + method,
            "arm": "rates_no_ftol",
            "status": "complete",
            "recovered": index == 0,
        }
        for index, method in enumerate(
            (
                "collocation_sensitivity",
                "joint_collocation_sensitivity",
                "alternating_collocation_sensitivity",
            )
        )
    ]
    groups = new.recovery_counts(rows)
    assert [g["method"] for g in groups] == ["C+S", "J+S", "A+S"]
    assert [g["recovered"] for g in groups] == [1, 0, 0]


def test_one_optimizer_failure_does_not_suppress_the_other_arm(
    source, tmp_path, monkeypatch
):
    root, task, _ = source
    output = tmp_path / "failed_physical"
    new.prepare_rate(
        new.RatePlan(source_tasks=(task["name"],), guard_seconds=30, replay_seconds=10),
        root,
        output,
    )
    real_fit = new.instrumented_fit

    def fail_physical(oracle, *args, **kwargs):
        if "tau" in oracle.names:
            raise ValueError("physical coordinate overflow")
        return real_fit(oracle, *args, **kwargs)

    monkeypatch.setattr(new, "instrumented_fit", fail_physical)
    result = new.execute_rate(output, 0)
    assert result["arms"]["physical"]["status"] == "fit_failed"
    assert result["arms"]["rates"]["status"] == "complete"
    assert result["status"] == "pair_unverified"


def test_supervisor_timeout_keeps_partial_fit_checkpoint(source, tmp_path, monkeypatch):
    from scripts import run_fitter_rate_refinement as runner

    root, task, _ = source
    output = tmp_path / "timeout"
    frozen = new.prepare_rate(new.RatePlan(source_tasks=(task["name"],)), root, output)
    partial = (
        output / "results" / frozen["tasks"][0]["name"] / "physical/fit_started.json"
    )
    write_json(partial, {"budget_seconds": 29})

    class Stalled:
        pid = 123456

        def wait(self, timeout=None):
            if timeout is not None:
                raise runner.subprocess.TimeoutExpired("test", timeout)
            return -9

    killed = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Stalled())
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append(pid))
    result = runner.run_supervised(output, 0)
    assert result["status"] == "timeout" and killed == [123456]
    assert read_json(partial) == {"budget_seconds": 29}
