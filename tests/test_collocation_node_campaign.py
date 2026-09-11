"""Portable public export, paired policies, split boundaries and resume."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.rebuttal import collocation_node_campaign as campaign
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.staged_topology import content_hash
from scripts import export_collocation_node_cases as exporter
from scripts import run_collocation_node_campaign as runner
from tests.test_fitter_diagnostic import bundle as diagnostic_bundle  # noqa: F401


@pytest.fixture
def exported(diagnostic_bundle, tmp_path):  # noqa: F811
    plan, paths, run = diagnostic_bundle
    cell = plan.cells[0]
    source = tmp_path / "multiround"
    ledger = {}
    for name in exporter.PUBLIC:
        relative = f"frozen/public/phase_b_v1/{cell.benchmark_id}/{name}"
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            paths["public_root"] / "phase_b_v1" / cell.benchmark_id / name, destination
        )
        ledger[relative] = sha256(destination)
    candidate = read_json(run / "candidate.json")
    write_json(source / "frozen/candidates/parent.json", candidate)
    task = {
        "task_id": "source_seed0",
        "seed": 0,
        "benchmark_id": cell.benchmark_id,
        "tier": cell.tier,
        "candidate_path": "frozen/candidates/parent.json",
        "candidate_file_sha256": sha256(source / "frozen/candidates/parent.json"),
    }
    parent = {
        "schema_version": "scientific-staged-multiround-feedback-plan-3",
        "tasks": [task],
        "public_asset_ledger": ledger,
        "public_asset_ledger_sha256": content_hash(ledger),
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    parent["plan_sha256"] = content_hash(parent)
    write_json(source / "plan.json", parent)
    revised = {**candidate, "candidate_id": "revised"}
    write_json(source / "results/source_seed0/round_001/candidate.json", revised)
    terminal = {
        "identity": content_hash([parent["plan_sha256"], task]),
        "result": {
            "test_data_opened": False,
            "private_reference_opened": False,
            "status": "complete",
            "rounds": [
                {
                    "round_index": 1,
                    "parent_candidate_sha256": content_hash(candidate),
                    "candidate_sha256": content_hash(revised),
                    "fit": {
                        "status": "fit_failed",
                        "training": None,
                        "validation": None,
                    },
                }
            ],
        },
    }
    write_json(source / "results/source_seed0/terminal.json", terminal)
    output = tmp_path / "bundle"
    bundle = exporter.export_cases(source, output)
    return source, output, bundle


def small_plan():
    return campaign.NodeCampaignPlan(
        fit={
            "initializer_seconds": 15,
            "refinement_seconds": 10,
            "maximum_function_evaluations": 20,
        }
    )


def test_export_keeps_failed_round_and_never_copies_test(exported):
    source, output, bundle = exported
    assert [c["source_round"] for c in bundle["cases"]] == [0, 1]
    assert exporter.export_cases(source, output) == bundle
    assert not list(output.rglob("test.csv"))
    assert campaign.verify_bundle(output) == bundle
    path = source / "results/source_seed0/round_001/candidate.json"
    payload = read_json(path)
    payload["candidate_id"] = "tampered"
    write_json(path, payload)
    with pytest.raises(ValueError, match="candidate differs"):
        exporter.export_cases(source, output)


@pytest.mark.parametrize("corruption", ["asset", "identity", "escape", "test"])
def test_bundle_corruption_fails_closed(exported, corruption):
    _, root, bundle = exported
    if corruption == "asset":
        (root / bundle["cases"][0]["candidate"]).write_text("{}")
    else:
        if corruption == "identity":
            bundle["identity"] = "0" * 64
        else:
            bundle["files"][
                "../secret.json"
                if corruption == "escape"
                else "public/phase_b_v1/fake/test.csv"
            ] = "0" * 64
            bundle["identity"] = content_hash(
                {k: v for k, v in bundle.items() if k != "identity"}
            )
        write_json(root / "bundle.json", bundle)
    with pytest.raises(ValueError):
        campaign.verify_bundle(root)


def test_paired_starts_run_and_restore_completed_fit(exported, tmp_path, monkeypatch):
    _, source, _ = exported
    root = tmp_path / "campaign"
    frozen = campaign.prepare_nodes(small_plan(), source, root)
    assert len(frozen["tasks"]) == 10  # Two saved candidates + three controls.
    assert [t["policy"] for t in frozen["tasks"][:2]] == list(campaign.POLICIES)
    assert frozen["tasks"][0]["case"] == frozen["tasks"][1]["case"]
    result = campaign.execute_nodes(root, 1)
    assert result["status"] == "complete", result
    assert result["fit"]["training"]["normalized_mse"] < 1e-7
    assert campaign.prepare_nodes(small_plan(), source, root) == frozen
    task_root = root / "results" / frozen["tasks"][1]["name"]
    (task_root / "result.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("saved fit must not be optimized again")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    restored = campaign.execute_nodes(root, 1)
    assert restored["fit"] == read_json(task_root / "fit.json")["fit"]
    summary = campaign.summarize_nodes(root)
    assert len(summary["rows"]) == 10
    assert len([r for r in summary["rows"] if r["group"] == "parent"]) == 2
    assert len([r for r in summary["rows"] if r["group"] == "accepted_round"]) == 2
    assert sum(r["status"] == "missing" for r in summary["rows"]) == 9


def test_interrupted_fit_never_receives_new_budget(exported, tmp_path, monkeypatch):
    _, source, _ = exported
    root = tmp_path / "interrupted"
    frozen = campaign.prepare_nodes(small_plan(), source, root)
    task = frozen["tasks"][0]
    identity = content_hash([frozen["identity"], task])
    write_json(
        root / "results" / task["name"] / "fit_started.json", {"identity": identity}
    )
    assert campaign.execute_nodes(root, 0)["status"] == "interrupted_fit"


def test_supervisor_timeout_is_terminal(exported, tmp_path, monkeypatch):
    _, source, _ = exported
    root = tmp_path / "timeout"
    campaign.prepare_nodes(small_plan(), source, root)

    class Stalled:
        pid = 999999

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    killed = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Stalled())
    monkeypatch.setattr(runner.os, "killpg", lambda *args: killed.append(args))
    assert runner.supervised(root, 0)["status"] == "timeout"
    assert runner.supervised(root, 0)["status"] == "timeout"
    assert len(killed) == 1


def test_cli_prepare_run_resume(exported, tmp_path):
    _, source, _ = exported
    config = tmp_path / "config.json"
    write_json(config, small_plan().model_dump(mode="json"))
    output = tmp_path / "cli"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_collocation_node_campaign.py"
    )
    environment = {**os.environ, "PYTHONPATH": str(script.parents[1] / "src")}
    base = [sys.executable, str(script)]
    for args in (
        ["prepare", "--source", str(source), "--config", str(config)],
        ["run", "--task-index", "1"],
        ["run", "--task-index", "1"],
        ["summarize"],
    ):
        subprocess.run(
            [*base, *args, "--output", str(output)],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=40,
        )
    summary = read_json(output / "summary.json")
    assert summary["rows"][1]["status"] == "complete"


def test_scheduler_resources_and_budget():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/hpc/collocation_node_delta.slurm").read_text()
    submit = (root / "scripts/hpc/submit_collocation_node_delta.sh").read_text()
    assert "--cpus-per-task=1" in worker and "--gres" not in worker
    assert "submission.intent" in submit and "submission_complete" in submit
    assert campaign.NodeCampaignPlan().worker_seconds < 30 * 60
    with pytest.raises(ValueError, match="scheduler limit"):
        campaign.NodeCampaignPlan(fit={"refinement_seconds": 900})
