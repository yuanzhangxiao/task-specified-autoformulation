"""Known-equation assistance, local qualification, matched arms and safe resume."""

import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks import challenging_fitting_inputs as inputs
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.detention import GEOMETRY
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as campaign
from autoformalism.fitting.reuse_diagnostic import ReuseDiagnosticPolicy
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.transcription_fit import StrategyPolicy
from autoformalism.schemas.public_fitting import PublicFitRequest


@pytest.fixture
def witness():
    equations = [
        f"-0.1*x{i}+0.3*x{(i + 1) % 5}+0.4*tanh(1.2*x{(i + 2) % 5}-0.2)"
        f"+{0.7 if i < 2 else 0.0}*tanh(1.1*u01)"
        for i in range(5)
    ] + [
        "-0.1*x5+0.2*tanh(1.3*x0)-0.3*tanh(1.1*x1)"
        "+(-0.4)*tanh(1.2*x2)+0.5*tanh(1.4*x3)*tanh(1.5*x4)"
    ]
    # The witness emitter uses additions with signed factors, not subtraction.
    equations[-1] = equations[-1].replace("-0.3*tanh", "+(-0.3)*tanh")
    return PublicFitRequest.model_validate(
        {
            "base_candidate": {
                "candidate_id": "witness",
                "parent_candidate_id": None,
                "states": [{"name": f"x{i}", "kind": "latent"} for i in range(6)],
                "state_equations": [
                    {"state": f"x{i}", "rhs": rhs} for i, rhs in enumerate(equations)
                ],
                "observation_mappings": [{"channel": "v01", "expression": "x5"}],
                "initial_conditions": [
                    {"state": f"x{i}", "scope": "global", "fixed_value": 0.0}
                    for i in range(6)
                ],
            },
            "context": {"targets": ["v01"], "external_inputs": ["u01"]},
            "initialization_plan": {
                "rules": {
                    f"x{i}": {"initial": {"mode": "map", "expression": "0.0"}}
                    for i in range(5)
                }
            },
            "profile": "general-rollout-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": "witness",
                "artifact_sha256": "a" * 64,
            },
        }
    ).model_dump(mode="json")


def test_lift_retains_exact_equations_signs_and_shared_initials(witness):
    req, truth = inputs.alien_request(witness)
    original = SymbolicODE(public._lower(PublicFitRequest.model_validate(witness))[0])
    free = SymbolicODE(public._lower(req)[0])
    assert len(truth) == 18
    assert len(free.initial_parameter_names) == 5
    assert not free.rhs_affine  # Unknown input tanh scale is genuinely nonlinear.
    theta = [truth[n] for n in free.names]
    rng = np.random.default_rng(17)
    for _ in range(10):
        x, u = rng.normal(size=6), rng.normal(size=1)
        assert np.allclose(original.rhs(0, x, [], u), free.rhs(0, x, theta, u))
    assert all(v == 0 for n, v in truth.items() if n.startswith("init_"))
    assert all(
        p.role.value == "nonnegative_coefficient" for p in req.base_candidate.parameters
    )
    bad = deepcopy(witness)
    bad["context"]["auxiliaries"] = ["v02"]
    with pytest.raises(ValueError, match="layout"):
        inputs.alien_request(bad)


@pytest.fixture
def release(tmp_path, witness):
    root = tmp_path / "release"
    plan = {"identity": "synthetic-test-release"}
    seal(root / "plan.json", plan)
    records = []
    for name, cell in inputs.CELLS.items():
        folder = root / "public" / cell
        for split in ("train", "val"):
            alien = name == "alien_hard"
            row = {
                "trajectory_id": f"{split}_000",
                "time": [0, 1, 2],
                "targets": {"v01" if alien else "h_down": [0.2, 0.3, 0.4]},
                "external_inputs": {"u01": [0, 1, 0]}
                if alien
                else {"inflow_up": [0, 1, 0], "inflow_down": [0.2, 0.2, 0.2]},
                "fixed_covariates": {} if alien else {**GEOMETRY, "initial_up": 0.5},
            }
            seal(
                folder / f"{split}.json",
                {"name": split, "fingerprint": split, "rows": [row]},
            )
        seal(folder / "specification.json", {"cell": cell})
        (folder / "proposer_prompt.txt").write_text("Unknown shared preparation.")
        files = list(folder.iterdir())
        if name == "alien_hard":
            path = root / "diagnostic" / cell / "reference_request.json"
            seal(path, witness)
            files.append(path)
        records.append(
            {
                "cell": cell,
                "ready_for_development": True,
                "files": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            }
        )
    seal(
        root / "summary.json",
        {
            "protocol": "phase-c-development-2",
            "whole_phase_c_roster_ready": True,
            "plan_sha256": public.content_sha256(plan),
            "test_generated": False,
            "cells": records,
        },
    )
    return root


@pytest.fixture
def exported(release, tmp_path):
    path = tmp_path / "inputs.json"
    inputs.export_inputs(release, path)
    return path


def config():
    return campaign.CampaignConfig(
        include_cstr=False,
        include_challenging=True,
        starts=1,
        strategy=StrategyPolicy(seconds=120),
        challenging_reuse=ReuseDiagnosticPolicy(native_seconds=30, point_seconds=2),
    )


def test_export_checks_bytes_never_reads_test(release, exported, monkeypatch):
    original = inputs.read_seal
    calls = []

    def read(path):
        assert "test" not in path.name
        calls.append(path)
        return original(path)

    monkeypatch.setattr(inputs, "read_seal", read)
    inputs.export_inputs(release, exported)
    assert calls
    assert set(read_seal(exported)["cases"]) == set(inputs.CELLS)
    prompt = release / "public" / inputs.CELLS["alien_hard"] / "proposer_prompt.txt"
    prompt.write_text("changed")
    with pytest.raises(ValueError, match="asset differs"):
        inputs.export_inputs(release, exported)
    with pytest.raises(ValueError, match="outside"):
        inputs.export_inputs(release, release / "new.json")


def test_paired_starts_no_truth_no_validation_no_warm_endpoint(exported, tmp_path):
    case = read_seal(exported)["cases"]["alien_hard"]
    req = inputs.request(case, 0)
    alternate = deepcopy(case)
    alternate["reference_parameters"] = dict.fromkeys(case["reference_parameters"], 99)
    assert req == inputs.request(alternate, 0)
    assert req != inputs.request(case, 1)
    root = tmp_path / "campaign"
    assert campaign.prepare(root, config(), exported)["tasks"] == 12
    plan, data = campaign.verify(root)
    assert plan["protocol"] == campaign.CHALLENGING_PROTOCOL
    for common in plan["commons"]:
        workers = [
            campaign.worker_payload(plan, data, t)
            for t in plan["tasks"]
            if t["common"] == common
        ]
        assert len(workers) == 6
        for w in workers:
            assert w["request"] == workers[0]["request"]
            assert w["nodes"] == workers[0]["nodes"]
            assert w["training"]["name"] == "train"
            assert "validation" not in w and "reference_parameters" not in w
            assert ("reuse_diagnostic" in w) == w["arm"].startswith("cache_")
    assert campaign.prepare(root, config(), exported)["tasks"] == 12
    summary = campaign.report(root)
    assert summary["recorded"] == 0
    assert all(g["coefficient_passes"] == 0 for g in summary["groups"])
    with pytest.raises(ValueError, match="fresh"):
        campaign.prepare(root, config(), exported, root)


def test_rank_failure_blocks_fit_and_preserves_diagnostic(
    exported, tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.prepare(root, config(), exported)
    monkeypatch.setattr(
        campaign,
        "replay",
        lambda *a: {
            "complete": True,
            "metrics": {"train": 0, "val": 0},
            "maximum_solver_difference": 0,
        },
    )
    monkeypatch.setattr(
        campaign, "sensitivity_audit", lambda *a: {"full_local_rank": False}
    )
    with pytest.raises(ValueError, match="excitation gate"):
        campaign.qualify(root)
    assert not read_seal(root / "qualification/result.json")["passed"]
    with pytest.raises(ValueError, match="qualification"):
        campaign.run_task(root, 0)


def test_new_endpoint_errors_resume_and_interruption(exported, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.prepare(root, config(), exported)
    plan, data = campaign.verify(root)
    seal(
        root / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )

    def fit(payload, directory):
        # Test double only: evaluator's exact parameters simulate an endpoint.
        return {
            "parameters": data["cases"]["alien_hard"]["reference_parameters"],
            "worker_payload_sha256": public.content_sha256(payload),
            "stop_reason": "test",
            "total_seconds": 1,
            "budget_exhausted": False,
        }

    monkeypatch.setattr(campaign, "run", fit)
    monkeypatch.setattr(
        campaign,
        "replay",
        lambda *a: {
            "complete": True,
            "metrics": {"train": 0, "val": 0},
            "maximum_solver_difference": 0,
        },
    )
    i = next(i for i, t in enumerate(plan["tasks"]) if t["common"] == "alien_hard_s0")
    row = campaign.run_task(root, i)
    assert row["accuracy_passed"] and row["coefficients_recovered"]
    assert row["initials_recovered"] and row["recovery_passed"] is None
    monkeypatch.setattr(campaign, "run", lambda *a: pytest.fail("no duplicate budget"))
    assert campaign.run_task(root, i) == row
    j = i + 1
    seal(root / "results" / plan["tasks"][j]["task_id"] / "started.json", {})
    assert campaign.run_task(root, j)["status"] == "interrupted"
    assert campaign.report(root)["recorded"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_challenging": True},
        {"include_cstr": False, "cases": ("alien_hard",)},
        {"include_cstr": False, "challenging_reuse": {}},
        {"include_cstr": False, "include_challenging": True, "cases": ("shape",)},
    ],
)
def test_invalid_campaign_configuration(kwargs):
    with pytest.raises(ValueError):
        campaign.CampaignConfig(**kwargs)


def test_scheduler_resources_and_exact_receipt_resume(exported, tmp_path, monkeypatch):
    from scripts import submit_phase_c_fitting_strategies as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "frozen\n")
    calls = []

    def sbatch(argv, **kw):
        calls.append(argv)
        assert "--mem=16G" in argv
        assert "--partition=cpu" in argv
        assert not any("--gres" in v or "--gpus" in v for v in argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    cfg = tmp_path / "config.json"
    cfg.write_text(config().model_dump_json())
    args = {"account": "test-cpu", "concurrency": 2, "inputs": exported}
    root = tmp_path / "campaign"
    first = submit.submit(root, cfg, **args)
    assert len(calls) == 3 and "--time=00:25:00" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert "--array=0-11%2" in calls[1]
    assert submit.submit(root, cfg, **args) == first and len(calls) == 3


@pytest.mark.parametrize("matched", [None, "/tmp/a matched source"])
def test_shell_launcher_optional_source_on_system_bash(tmp_path, matched):
    """Exercise actual shell parsing, including macOS Bash's empty-array nounset."""
    executable, log = tmp_path / "python-stub", tmp_path / "arguments.txt"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$AF_STUB_LOG"\n')
    executable.chmod(0o755)
    env = dict(os.environ)
    env.update(AF_PYTHON=str(executable), AF_STUB_LOG=str(log))
    env.pop("AF_MATCHED_SOURCE", None)
    if matched:
        env["AF_MATCHED_SOURCE"] = matched
    path = Path(__file__).resolve().parents[1] / (
        "scripts/hpc/submit_phase_c_fitting_strategies_delta.sh"
    )
    result = subprocess.run(
        ["bash", str(path)], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    arguments = log.read_text().splitlines()
    assert ("--matched-source" in arguments) == bool(matched)
    if matched:
        assert arguments[arguments.index("--matched-source") + 1] == matched
