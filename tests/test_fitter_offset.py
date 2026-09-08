"""Paired role intervention, synthetic recovery, split budgets and provenance."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal import fitter_offset as campaign
from autoformalism.rebuttal.fitter_diagnostic import (
    read_json,
    runtime_identity,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_runtime import RuntimePlan
from autoformalism.rebuttal.staged_fit_probe import prepare_probe
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash
from scripts import run_fitter_offset as runner
from tests.test_fitter_diagnostic import bundle as diagnostic_bundle  # noqa: F401
from tests.test_staged_fit_probe import staged_bundle  # noqa: F401


def test_production_pin_matches_serialized_runtime_plan():
    """Pin the actual freeze representation, including float coercion of budgets."""
    configs = Path(__file__).resolve().parents[1] / "configs"
    raw = read_json(configs / "fitter_runtime_v2.json")
    serialized = RuntimePlan.model_validate(raw).model_dump(mode="json")
    plan = campaign.OffsetPlan.model_validate(
        read_json(configs / "fitter_signed_offset_v1.json")
    )
    assert plan.source_plan_sha256 == content_hash(serialized)
    assert plan.source_plan_sha256 != content_hash(raw)


@pytest.fixture
def offset_bundle(staged_bundle, tmp_path):  # noqa: F811
    """Make a source-pinned synthetic decay plus negative offset, with no test data."""
    staged_plan, paths = staged_bundle
    public = paths["public_root"] / "phase_b_v1" / staged_plan.benchmark_id
    for split in ("train", "validation"):
        path = public / f"{split}.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["v01"] = str(float(row["v01"]) - 2)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    manifest = read_json(public / "manifest.json")
    for split in ("train", "validation"):
        manifest["splits"][split] = sha256(public / f"{split}.csv")
    write_json(public / "manifest.json", manifest)
    source_root = paths["function_results"] / staged_plan.function_task_id
    source = read_json(source_root / "result.json")
    source["candidate"] = CandidateModel.model_validate(
        {
            "candidate_id": "signed_offset_fixture",
            "parent_candidate_id": None,
            "states": [{"name": "x", "kind": "latent"}],
            "state_equations": [{"state": "x", "rhs": "-decay*x"}],
            "processes": [{"name": "readout", "expression": "gain*x+c"}],
            "observation_mappings": [{"channel": "v01", "expression": "readout"}],
            "parameters": [
                {"name": "decay", "role": "rate", "scope": "global"},
                {"name": "gain", "role": "nonnegative_coefficient", "scope": "global"},
                {"name": "c", "role": "nonnegative_coefficient", "scope": "global"},
            ],
            "initial_conditions": [{"state": "x", "fixed_value": 1, "scope": "global"}],
        }
    ).model_dump(mode="json")
    write_json(source_root / "result.json", source)
    terminal = read_json(source_root / "terminal.json")
    terminal["result"] = source
    write_json(source_root / "terminal.json", terminal)
    staged_plan = staged_plan.model_copy(
        update={"function_result_sha256": content_hash(source)}
    )
    original = prepare_probe(staged_plan, **paths)
    root = paths["output"]
    write_json(root / "source_freeze.json", original)
    previous = {"plan": {"source_freeze_sha256": sha256(root / "source_freeze.json")}}
    write_json(root / "previous_freeze.json", previous)
    runtime_plan = RuntimePlan(
        source_code_sha256="0" * 64,
        source_plan_sha256=content_hash(previous["plan"]),
        anchor_hashes=dict.fromkeys(("all_ones", "intermediate", "late"), "0" * 64),
    )
    parent = {
        "plan": runtime_plan.model_dump(mode="json"),
        "runtime": runtime_identity(),
        "assets": {
            **original["assets"],
            "source_freeze.json": sha256(root / "source_freeze.json"),
            "previous_freeze.json": sha256(root / "previous_freeze.json"),
        },
        "tasks": list(campaign.PARENT_TASKS),
    }
    parent["freeze_sha256"] = content_hash(parent)
    write_json(root / "freeze.json", parent)
    for arm in campaign.OLD_ARMS:
        task = next(t for t in campaign.PARENT_TASKS if t["name"] == arm)
        record = {
            "identity": content_hash([parent["freeze_sha256"], task]),
            "fit": {"parameters": {"c": 0.0, "decay": 0.72, "gain": 0.0}},
            "test_data_opened": False,
            "private_reference_opened": False,
        }
        write_json(root / "results" / arm / "fit.json", record)
        write_json(root / "results" / arm / "result.json", record)
    plan = campaign.OffsetPlan(
        source_code_sha256=parent["runtime"]["source_sha256"],
        source_plan_sha256=content_hash(parent["plan"]),
        source_candidate_sha256=sha256(root / "candidate.json"),
        zero_gain_parameters=("gain",),
        profile_trajectories=1,
        fit_seconds=10,
        replay_seconds=5,
        guard_seconds=30,
        grace_seconds=5,
        maximum_nfev=80,
    )
    return plan, root, tmp_path / "offset"


def test_paired_recovery_changes_only_offset_domain(offset_bundle):
    plan, source, output = offset_bundle
    before = sha256(source / "candidate.json")
    frozen = campaign.prepare_offset(plan, source, output)
    original = read_json(output / "candidate.json")
    variant = read_json(output / "signed_offset.json")
    c = next(p for p in variant["parameters"] if p["name"] == "c")
    assert (c["role"], c["domain"]) == ("offset", "real")
    c.update(role="nonnegative_coefficient", domain="nonnegative")
    assert original == variant
    results = [campaign.execute_offset(output, i) for i in range(7)]
    assert all(r["status"] == "complete" for r in results)
    assert len(results[0]["cases"]["all_ones"]["replays"]) == 4
    assert not list(output.rglob("test.csv"))
    for index in (1, 2, 3, 4):
        r = results[index]
        assert r["fit"]["initial_parameters"] == frozen["starts"][r["task"]["start"]]
        assert len(r["replays"]) == 6
        assert all(c["pass"] for c in r["checks"])
        if r["task"]["variant"] == "signed_offset":
            assert r["parameters"]["c"] == pytest.approx(-2, abs=1e-4)
            assert r["parameters"]["decay"] == pytest.approx(0.72, abs=1e-3)
            assert r["replays"]["BDF_tight/validation"]["normalized_mse"] < 1e-7
        else:
            assert r["parameters"]["c"] >= 0
            assert r["replays"]["Radau_tight/train"]["normalized_mse"] > 1
    assert (
        sha256(source / "candidate.json") == sha256(output / "candidate.json") == before
    )
    summary = runner.write_summary(output)
    assert summary["baselines"]["train"]["training_mean_nmse"] == pytest.approx(1)
    assert not summary["model_selection_performed"]
    assert "BDF_tight/validation" in (output / "summary.md").read_text()


def test_resume_reuses_optimizer_and_per_trajectory_replays(offset_bundle, monkeypatch):
    plan, source, output = offset_bundle
    campaign.prepare_offset(plan, source, output)
    campaign.execute_offset(output, 0)
    first = campaign.execute_offset(output, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed numerical work repeated")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    monkeypatch.setattr(campaign, "simulate_trajectory", forbidden)
    root = output / "results" / first["task"]["name"]
    (root / "result.json").unlink()
    for name in first["replays"]:
        (root / "replays" / name / "result.json").unlink()
    assert campaign.execute_offset(output, 2) == first


@pytest.mark.parametrize(
    "field", ["source_code_sha256", "source_plan_sha256", "source_candidate_sha256"]
)
def test_wrong_source_pins_are_rejected(offset_bundle, field):
    plan, source, output = offset_bundle
    with pytest.raises(ValueError, match="differ") as error:
        campaign.prepare_offset(
            plan.model_copy(update={field: "0" * 64}), source, output
        )
    if field != "source_candidate_sha256":
        assert f"{field} expected={'0' * 64}, actual={getattr(plan, field)}" in str(
            error.value
        )


def test_modified_runtime_tasks_are_rejected_with_both_hashes(offset_bundle):
    plan, source, output = offset_bundle
    parent = read_json(source / "freeze.json")
    expected = content_hash(list(campaign.PARENT_TASKS))
    parent["tasks"][0]["anchor"] = "different_anchor"
    actual = content_hash(parent["tasks"])
    parent["freeze_sha256"] = content_hash(
        {k: v for k, v in parent.items() if k != "freeze_sha256"}
    )
    write_json(source / "freeze.json", parent)
    with pytest.raises(ValueError, match="source runtime tasks differ") as error:
        campaign.prepare_offset(plan, source, output)
    assert f"expected_sha256={expected}, actual_sha256={actual}" in str(error.value)
    assert not output.exists()


@pytest.mark.parametrize(
    "artifact", ["candidate.json", "source_freeze.json", "previous_freeze.json"]
)
def test_original_snapshot_cannot_be_rebound(offset_bundle, artifact):
    plan, source, output = offset_bundle
    value = read_json(source / artifact)
    value["changed"] = True
    write_json(source / artifact, value)
    parent = read_json(source / "freeze.json")
    parent["assets"][artifact] = sha256(source / artifact)
    parent["freeze_sha256"] = content_hash(
        {k: v for k, v in parent.items() if k != "freeze_sha256"}
    )
    write_json(source / "freeze.json", parent)
    if artifact == "previous_freeze.json":
        # Arbitrary extra provenance metadata does not change a source pin;
        # changing the bound original snapshot does.
        value["plan"]["source_freeze_sha256"] = "0" * 64
        write_json(source / artifact, value)
    with pytest.raises(ValueError, match="differ"):
        campaign.prepare_offset(plan, source, output)


def test_missing_guard_blocks_fitting_and_stays_in_summary(offset_bundle, monkeypatch):
    plan, source, output = offset_bundle
    campaign.prepare_offset(plan, source, output)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit should be guarded")

    monkeypatch.setattr(campaign, "instrumented_fit", forbidden)
    result = campaign.execute_offset(output, 1)
    assert result["status"] == "accuracy_guard_failed"
    assert result["test_data_opened"] is False
    assert len(campaign.summarize_offset(output)["rows"]) == 7


def test_training_timeout_does_not_consume_validation_budget(
    offset_bundle, monkeypatch
):
    plan, source, output = offset_bundle
    frozen = campaign.prepare_offset(plan, source, output)
    original = campaign.simulate_trajectory

    def training_timeout(model, trajectory, *args, **kwargs):
        if trajectory.trajectory_id.startswith("train"):
            raise TimeoutError("synthetic training timeout")
        return original(model, trajectory, *args, **kwargs)

    monkeypatch.setattr(campaign, "simulate_trajectory", training_timeout)
    result = campaign.execute_offset(output, 5)
    assert result["status"] == "replay_unverified"
    for case in campaign.CASES:
        assert result["replays"][f"{case}/train"]["normalized_mse"] is None
        assert (
            result["replays"][f"{case}/train"]["failures"][0]["message"]
            == "synthetic training timeout"
        )
        assert result["replays"][f"{case}/validation"]["status"] == "complete"
    assert frozen["previous_parameters"][campaign.OLD_ARMS[0]] == result["parameters"]
    runner.write_summary(output)
    assert "synthetic training timeout" in (output / "summary.md").read_text()


def test_changed_numerical_checkpoint_is_rejected(offset_bundle):
    plan, source, output = offset_bundle
    campaign.prepare_offset(plan, source, output)
    record = campaign.execute_offset(output, 5)
    root = output / "results" / record["task"]["name"]
    (root / "result.json").unlink()
    path = root / "replays/Radau_tight/train/trajectory-0.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        campaign.execute_offset(output, 5)


def test_large_finite_metrics_do_not_become_missing_success(offset_bundle):
    plan, source, output = offset_bundle
    frozen = campaign.prepare_offset(plan, source, output)
    parent, dataset, model = campaign.load_problem(output, "original")
    scale = frozen["baselines"]["training_scale"]
    record = campaign.replay_split(
        output / "large",
        "large",
        model,
        dataset.train,
        {"gain": 0.0, "decay": 1.0, "c": scale * 1e154},
        scale,
        parent.fit_config,
        5,
    )
    assert record["status"] == "complete"
    assert np.isfinite(record["normalized_mse"])
    assert record["normalized_mse"] == pytest.approx(1e308)


def test_supervisor_cli_and_resume(offset_bundle):
    plan, source, output = offset_bundle
    config = output.parent / "offset_config.json"
    write_json(config, plan.model_dump(mode="json"))
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}

    def run(*args):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_fitter_offset.py",
                *args,
                "--output",
                str(output),
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert "tasks 7" in run("prepare", "--source", str(source), "--config", str(config))
    assert "complete" in run("run", "--task-index", "0")
    assert "complete" in run("run", "--task-index", "2")
    assert "complete" in run("run", "--task-index", "2")
    assert "summary.md" in run("summarize")


@pytest.mark.parametrize("fail_second", [False, True])
def test_slurm_dependencies_and_partial_submission_do_not_duplicate(
    offset_bundle, tmp_path, fail_second
):
    plan, source, output = offset_bundle
    config = tmp_path / "offset_config.json"
    write_json(config, plan.model_dump(mode="json"))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text(
        '#!/bin/sh\nif [ "$1" = "rev-parse" ]; then '
        "echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; fi\n"
    )
    git.chmod(0o755)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
path=Path(os.environ['FAKE_SUBMISSIONS'])
rows=path.read_text().splitlines() if path.exists() else []
with path.open('a') as stream:
    stream.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAKE_FAIL_SECOND')=='1' and len(rows)==1:
    sys.exit(42)
print(10000+len(rows))
"""
    )
    sbatch.chmod(0o755)
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / "submissions.jsonl"
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": sys.executable,
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(output),
        "AF_CONFIG": str(config),
        "FAKE_SUBMISSIONS": str(log),
        "FAKE_FAIL_SECOND": "1" if fail_second else "0",
    }

    def run():
        return subprocess.run(
            ["bash", "scripts/hpc/submit_fitter_offset_delta.sh"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    first = run()
    assert first.returncode == (42 if fail_second else 0), first.stderr
    submissions = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--array=0%2" in submissions[0]
    assert "--array=1-6%2" in submissions[1]
    assert "--dependency=afterany:10000" in submissions[1]
    if not fail_second:
        assert "--dependency=afterany:10001" in submissions[2]
    assert run().returncode == 0
    assert len(log.read_text().splitlines()) == len(submissions)
