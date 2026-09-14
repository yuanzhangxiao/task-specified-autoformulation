"""Final campaign isolation, finite budgets, meaningful recovery and resume."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.data import TrainingScaler
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.matching_probe import latent_start
from autoformalism.fitting.stagnation import RolloutOracle
from autoformalism.rebuttal import final_fitter as final
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.final_fitter_models import (
    alternative_starts,
    smaller_problem,
    training_prefix,
)
from autoformalism.rebuttal.final_fitter_smoke import smoke
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.staged_topology import content_hash
from tests.test_parameter_freedom import prepared


def plan():
    return final.FinalFitterPlan.model_validate_json(
        Path("configs/fitter_final_alternatives_v1.json").read_text()
    )


def frozen_at(tmp_path):
    _, source, old = prepared(tmp_path)
    # The older preparation fixture deliberately omits scoring labels; this
    # campaign validates their coverage before any cluster work is submitted.
    problem = read_json(source / "inputs/synthetic.json")
    clean = {
        key: {r["trajectory_id"]: r["targets"]["v01"] for r in split["rows"]}
        for key, split in problem["splits"].items()
    }
    write_json(source / "inputs/clean.json", clean)
    old["assets"]["inputs/clean.json"] = sha256(source / "inputs/clean.json")
    old["identity"] = content_hash({k: v for k, v in old.items() if k != "identity"})
    write_json(source / "freeze.json", old)
    write_json(source / "gate/result.json", {"identity": old["identity"], "pass": True})
    output = tmp_path / "final"
    return source, output, final.prepare(source, output, plan())


def test_nine_tasks_no_near_starts_and_exact_source_provenance(tmp_path):
    source, output, frozen = frozen_at(tmp_path)
    assert len(frozen["tasks"]) == 9
    assert frozen["no_automatic_followup"] and not frozen["historical_estimates_used"]
    assert not frozen["test_data_opened"] and not frozen["proposer_access"]
    for index, source_index in ((0, 0), (4, 2)):
        original = read_json(source / f"problems/{source_index:03d}.json")
        for offset in range(4):
            task = frozen["tasks"][index + offset]
            assert task["starts"][0] == original["start"]
            assert not task["oracle_weight_start"] and task["oracle_initials"]
            assert read_json(output / f"problems/{index + offset:03d}.json") == original
    small = frozen["tasks"][8]
    assert small["states"] == 3 and small["parameters"] == 13
    assert not small["oracle_shapes"] and not small["oracle_initials"]
    assert final.prepare(source, output, plan()) == frozen
    write_json(output / "problems/000.json", {})
    with pytest.raises(ValueError, match="frozen asset"):
        final.verify(output)


def test_smaller_model_uses_no_hidden_boundaries_or_validation_for_design():
    p, _ = synthetic_problem("shared", 0, 0)
    small = smaller_problem(p["splits"])
    altered = deepcopy(p["splits"])
    for key, split in altered.items():
        for row in split["rows"]:
            row["fixed_covariates"] = {"known_z1": 12345}
            if key == "val":
                row["targets"]["v01"] = [999 for _ in row["time"]]
    other = smaller_problem(altered)
    for name in ("candidate", "context", "start", "initialization_plan", "design"):
        assert small[name] == other[name]
    system, guesses = system_for(small)
    assert set(system.initial_parameter_names) == set(guesses)
    assert len(guesses) == 4
    assert all(
        not row["fixed_covariates"]
        for s in small["splits"].values()
        for row in s["rows"]
    )
    assert small["context"]["fixed_covariates"] == []


def test_prefix_preserves_initials_and_original_input_interpolation():
    p, _ = synthetic_problem("shared", 0, 0)
    original = deepcopy(p)
    cropped = training_prefix(p, 0.25)
    for full, prefix in zip(p["splits"]["train"]["rows"], cropped["rows"], strict=True):
        n = len(prefix["time"])
        assert 2 <= n < len(full["time"])
        assert prefix["time"] == full["time"][:n]
        assert prefix["targets"]["v01"] == full["targets"]["v01"][:n]
        assert prefix["external_inputs"]["u01"] == full["external_inputs"]["u01"][:n]
        assert prefix["fixed_covariates"] == full["fixed_covariates"]
    assert p == original
    assert training_prefix(p, 1)["rows"] == p["splits"]["train"]["rows"]
    with pytest.raises(ValueError, match="prefix"):
        training_prefix(p, 0)


def test_start_design_is_deterministic_and_domain_valid(tmp_path):
    p, _ = synthetic_problem("shared", 0, 0)
    system, guesses = system_for(p)
    p["start"].update(guesses)
    starts = alternative_starts(p, 103)
    assert starts == alternative_starts(p, 103)
    assert starts[1] != alternative_starts(p, 104)[1]
    train = unpack_split(p["splits"]["train"])
    layout = RolloutOracle(
        system.model, train, {"v01": 1}, plan().fit.fit_config(), tmp_path, None
    )
    for point in starts:
        layout.vector(point)


def test_interrupted_stage_retains_best_without_new_calls(tmp_path):
    write_json(tmp_path / "started.json", {"identity": "abc"})
    best = {"parameters": {"x": 1.0}, "cost": 5.0}
    write_json(tmp_path / "calls/best_evaluated.json", best)
    result = final.checkpoint_stage(
        tmp_path, "abc", lambda: pytest.fail("fresh budget")
    )
    assert result["status"] == "interrupted" and result["best"] == best
    assert (
        final.checkpoint_stage(tmp_path, "abc", lambda: pytest.fail("repeated"))
        == result
    )
    with pytest.raises(ValueError):
        final.checkpoint_stage(tmp_path, "changed", lambda: {})


def test_horizon_continuation_never_compares_prefix_cost_with_full_cost(
    tmp_path, monkeypatch
):
    p, _ = synthetic_problem("shared", 0, 0)
    _, guesses = system_for(p)
    p["start"].update(guesses)
    task = {"strategy": "horizon_continuation", "starts": [p["start"]]}
    calls, horizons, starts = [], [], []

    def stage(problem, payload, point, scale, config, seconds, cap, root):
        starts.append(point.copy())
        calls.append((seconds, cap, scale))
        horizons.append(len(payload["rows"][0]["time"]))
        updated = {**point, "rate": point["rate"] + 1}
        return {
            "status": "finished",
            "best": {"parameters": updated, "cost": 0.001 if len(calls) < 3 else 10.0},
        }

    monkeypatch.setattr(final, "sensitivity_stage", stage)
    # Full training evaluation finds the original start better than every final
    # full rollout; tiny prefix costs must not overwrite it.
    monkeypatch.setattr(
        final,
        "select_full_training",
        lambda *args: {
            "status": "finished",
            "best": {"parameters": p["start"], "cost": 2.0},
        },
    )
    result = final.fit_task(p, task, plan(), tmp_path, "horizon")
    assert horizons[0] < horizons[1] < horizons[2]
    assert starts[1]["rate"] == starts[0]["rate"] + 1
    assert starts[2]["rate"] == starts[1]["rate"] + 1
    assert result["parameters"] == p["start"]
    assert sum(c[0] for c in calls) == pytest.approx(plan().search_seconds)
    assert sum(c[1] for c in calls) == plan().search_calls
    assert len({c[2] for c in calls}) == 1  # fixed full-training scale
    final.fit_task(p, task, plan(), tmp_path, "horizon")
    assert len(calls) == 3


def test_multistart_refines_each_start_and_retains_best_if_selection_fails(
    tmp_path, monkeypatch
):
    p, _ = synthetic_problem("shared", 0, 0)
    _, guesses = system_for(p)
    p["start"].update(guesses)
    starts = alternative_starts(p, 10)
    seen = []

    def stage(problem, payload, point, *args):
        seen.append(point)
        return {
            "status": "finished",
            "best": {"parameters": point, "cost": (8, 1, 3)[len(seen) - 1]},
        }

    monkeypatch.setattr(final, "sensitivity_stage", stage)
    monkeypatch.setattr(
        final, "select_full_training", lambda *a: {"status": "failed", "best": None}
    )
    result = final.fit_task(
        p,
        {"strategy": "direct_multistart", "starts": starts},
        plan(),
        tmp_path,
        "multi",
    )
    assert seen == starts and result["parameters"] == starts[1]


def test_completed_fit_replay_resume_and_numerical_pass_is_not_recovery(
    tmp_path, monkeypatch
):
    _, output, frozen = frozen_at(tmp_path)
    write_json(
        output / "gate/result.json", {"identity": frozen["identity"], "pass": True}
    )
    task = frozen["tasks"][0]
    identity = content_hash([frozen["identity"], task])
    root = output / "results/task_000"
    write_json(
        root / "fit.json",
        {"identity": identity, "fit": {"parameters": task["starts"][0]}},
    )
    monkeypatch.setattr(final, "fit_task", lambda *a: pytest.fail("refitting"))
    monkeypatch.setattr(
        final,
        "checked_replay",
        lambda *a, **k: {
            "pass": True,
            "scores": {"Radau": {"clean_nmse": 0.8}},
        },
    )
    result = final.execute(output, 0)
    assert result["numerical_replay_pass"] and not any(result["fit_bands"].values())
    monkeypatch.setattr(final, "checked_replay", lambda *a, **k: pytest.fail("replay"))
    assert final.execute(output, 0) == result
    for i in (-1, 9):
        with pytest.raises(ValueError, match="task index"):
            final.execute(output, i)


def test_limited_memory_policy_rejects_exact_piecewise(tmp_path):
    p, _ = synthetic_problem("piecewise", 0, 0)
    system, guesses = system_for(p)
    train = unpack_split(p["splits"]["train"])
    config = plan().fit.fit_config()
    layout = RolloutOracle(system.model, train, {"v01": 1}, config, tmp_path, None)
    with pytest.raises(ValueError, match="first-order"):
        latent_start(
            system,
            train,
            layout.lower,
            layout.upper,
            layout.vector({**p["start"], **guesses}),
            1,
            config,
            "collocation_init",
            5,
            tmp_path,
            hessian_approximation="exact",
        )


def test_real_sensitivity_stage_estimates_parameters_and_initial_value(tmp_path):
    p, _ = synthetic_problem("shared", 0, 0)
    system, guesses = system_for(p)
    p["start"].update(dict.fromkeys(guesses, 1.5))
    p["start"].update(rate=0.45, gain=1.1)
    train = unpack_split(p["splits"]["train"])
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    result = final.sensitivity_stage(
        p, p["splits"]["train"], p["start"], scale, plan().fit, 20, 35, tmp_path
    )
    assert result["best"]["cost"] < 1e-10, result
    assert result["actual_residual_calls"] <= 35
    assert result["best"]["parameters"][
        system.initial_parameter_names[0]
    ] == pytest.approx(2, abs=1e-5)


def test_numerical_smoke_all_four_routes(tmp_path):
    result = smoke(tmp_path, plan())
    assert result["pass"], result
    for strategy, hessian in (
        ("collocation_exact", "exact"),
        ("collocation_limited", "limited-memory"),
    ):
        phase = read_json(
            tmp_path / strategy / "coupled/fit/collocation/phase_timing.json"
        )
        assert phase["hessian_approximation"] == hessian


def test_report_with_no_numerical_runtime_keeps_missing_results(tmp_path):
    _, output, frozen = frozen_at(tmp_path)
    write_json(
        output / "gate/result.json", {"identity": frozen["identity"], "pass": False}
    )
    r = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/run_final_fitter.py",
            "summarize",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert read_json(output / "summary.json")["counts"] == {"blocked_by_gate": 9}


def test_launcher_has_one_matrix_and_no_duplicate_submission(tmp_path):
    source, _, _ = frozen_at(tmp_path)
    binpath = tmp_path / "bin"
    binpath.mkdir()
    (binpath / "git").write_text(
        '#!/bin/sh\n[ "$1" = rev-parse ] && echo pinned-test\nexit 0\n'
    )
    (binpath / "git").chmod(0o755)
    calls = tmp_path / "calls.json"
    sbatch = binpath / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(calls)!r})\n"
        "r=json.loads(p.read_text()) if p.exists() else []\n"
        "r.append(sys.argv[1:]);p.write_text(json.dumps(r));print(50000+len(r))\n"
    )
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binpath) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(Path.cwd()),
        "AF_PYTHON": sys.executable,
        "AF_CASADI_ROOT": "",
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(tmp_path / "jobs"),
    }
    for _ in range(2):
        r = subprocess.run(
            ["bash", "scripts/hpc/submit_final_fitter_delta.sh"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
    submissions = json.loads(calls.read_text())
    assert len(submissions) == 3
    assert "--array=0-8%2" in submissions[1]
    assert "--dependency=afterok:50001" in submissions[1]
    assert "--dependency=afterany:50002:50001" in submissions[2]
    result = read_json(tmp_path / "jobs/submission.json")
    assert result["fit_memory_gb"] == 64 and result["submission_complete"]


def test_budget_validation():
    p = plan()
    assert p.search_calls + p.selection_calls == 120
    assert p.search_seconds + p.selection_seconds == 4800
    with pytest.raises(ValueError, match="selection"):
        final.FinalFitterPlan.model_validate(
            {
                **p.model_dump(mode="json"),
                "selection_seconds": 240,
                "total_fit_seconds": 100,
            }
        )
    assert CollocationSensitivityConfig().collocation_hessian == "auto"


def test_real_execute_saves_fit_then_independent_replays(tmp_path, monkeypatch):
    p, clean = synthetic_problem("shared", 0, 0)
    _, guesses = system_for(p)
    p["start"] = {"rate": 0.3, "gain": 0.8, **dict.fromkeys(guesses, 2.0)}
    raw = plan().model_dump(mode="json")
    raw.update(total_fit_seconds=18, selection_seconds=3, replay_seconds=10)
    raw["fit"].update(initializer_seconds=1, recovery_sensitivity_seconds=5)
    frozen = {
        "identity": "real-runtime-fixture",
        "plan": raw,
        "tasks": [
            {
                "index": 0,
                "family": "fixed_shapes",
                "strategy": "direct_multistart",
                "starts": alternative_starts(p, 2),
            }
        ],
        "bands": {"strict": 1e-4, "good": 0.01, "practical": 0.1},
    }
    monkeypatch.setattr(final, "verify", lambda *a: frozen)
    write_json(tmp_path / "problems/000.json", p)
    write_json(tmp_path / "inputs/clean.json", clean)
    write_json(
        tmp_path / "gate/result.json", {"identity": frozen["identity"], "pass": True}
    )
    result = final.execute(tmp_path, 0)
    assert result["status"] == "complete", result
    assert result["fit_bands"]["strict"]
    assert set(result["replays"]["train"]["scores"]) == {"BDF", "Radau"}
    assert (tmp_path / "results/task_000/fit.json").exists()
    monkeypatch.setattr(final, "fit_task", lambda *a: pytest.fail("refit"))
    assert final.execute(tmp_path, 0) == result


def test_source_missing_clean_samples_fails_before_submission(tmp_path):
    source, _, _ = frozen_at(tmp_path)
    old = read_json(source / "freeze.json")
    write_json(source / "inputs/clean.json", {})
    old["assets"]["inputs/clean.json"] = sha256(source / "inputs/clean.json")
    old["identity"] = content_hash({k: v for k, v in old.items() if k != "identity"})
    write_json(source / "freeze.json", old)
    write_json(source / "gate/result.json", {"identity": old["identity"], "pass": True})
    with pytest.raises(ValueError, match="clean reference"):
        final.prepare(source, tmp_path / "invalid", plan())
