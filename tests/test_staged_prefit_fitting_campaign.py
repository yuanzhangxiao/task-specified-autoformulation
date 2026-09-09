"""Frozen six-candidate fitting route, provenance, and resume tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autoformalism.rebuttal.staged_prefit_fitting_campaign as campaign
from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext
from autoformalism.schemas import CandidateModel
from autoformalism.staged_topology import content_hash


def _candidate() -> CandidateModel:
    return CandidateModel.model_validate(
        {
            "candidate_id": "synthetic",
            "parent_candidate_id": None,
            "states": [
                {"name": "x", "kind": "observed"},
                {"name": "z", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "x", "rhs": "b * z"},
                {"state": "z", "rhs": "-a * z + u"},
            ],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "parameters": [
                {"name": "a", "scope": "global", "role": "rate"},
                {"name": "b", "scope": "global", "role": "coefficient"},
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"},
                {"state": "z", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _data(provenance: DerivativeProvenance) -> DevelopmentDataset:
    def split(name: SplitName, fingerprint: str) -> DatasetSplit:
        time = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 0.5, 1.0])
        trajectory = Trajectory(
            trajectory_id=f"{name.value}-0",
            time=time,
            targets={"x": values.copy()},
            auxiliaries={},
            external_inputs={"u": np.ones(3)},
            fixed_covariates={},
            derivatives={"x": np.array([0.5, 0.5, 0.5])},
            derivative_provenance=provenance,
        )
        return DatasetSplit(name, (trajectory,), fingerprint)

    return DevelopmentDataset(
        benchmark_id="synthetic",
        tier="easy",
        roles=TierRoles(targets=("x",)),
        train=split(SplitName.TRAIN, "train"),
        validation=split(SplitName.VALIDATION, "validation"),
    )


def _context() -> ValidationContext:
    return ValidationContext(
        targets=("x",),
        external_inputs=("u",),
        forcing_bounds={"u": (1.0, 1.0)},
    )


def test_router_requires_structure_and_exact_public_provenance() -> None:
    estimated = campaign.fitting_route(
        _candidate(), _data(DerivativeProvenance.ESTIMATED), _context()
    )
    exact = campaign.fitting_route(
        _candidate(), _data(DerivativeProvenance.EXACT), _context()
    )

    assert estimated["profiled_structural_compatible"]
    assert not estimated["exact_training_derivative_contract_available"]
    assert estimated["selected_backend"] == "bounded_nonlinear"
    assert not estimated["validation_used_for_routing"]
    assert exact["exact_training_derivative_contract_available"]
    assert exact["selected_backend"] == "profiled_latent_basis_linear_ridge"
    assert exact["profiled_affine_parameters"] == ["b"]
    assert exact["profiled_outer_parameters"] == ["a"]


def _source_bundle(root: Path) -> tuple[Path, Path, Path]:
    config = json.loads(
        Path("configs/staged_prefit_fitting_handoff_v1.json").read_text()
    )
    source_root = root / "source"
    public_root = root / "public"
    tasks = []
    candidate = _candidate().model_dump(mode="json")
    result = {
        "status": "complete",
        "complete_model": True,
        "candidate": candidate,
        "batch_term_audits": [{"atomic_repair_attempted": True}],
        "deterministic_prefit_certificate": {"passed": True},
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    for benchmark_id in config["public_cells"]:
        directory = public_root / "phase_b_v1" / benchmark_id
        directory.mkdir(parents=True)
        for name in campaign.PUBLIC_FILES:
            payload = (
                "Synthetic scientific context\n\nF. Required response\ncontract\n"
                if name == "proposer_prompt.txt"
                else "fixture\n"
            )
            (directory / name).write_text(payload)
        for seed in config["seeds"]:
            task_id = f"{benchmark_id}_seed{seed}_functions"
            task = {
                "task_id": task_id,
                "benchmark_id": benchmark_id,
                "seed": seed,
                "brief": {"scientific_context": "Synthetic scientific context"},
            }
            tasks.append(task)
    plan = {"tasks": tasks}
    plan["plan_sha256"] = content_hash(plan)
    (source_root / "results").mkdir(parents=True)
    (source_root / "plan.json").write_text(json.dumps(plan))
    summary = {
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "overall_result": "pass",
        "terminal_results": 6,
        "complete_model_rate": 1.0,
        "deterministic_prefit_pass_rate": 1.0,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    summary_path = source_root / "results" / "summary.json"
    summary_path.write_text(json.dumps(summary))
    for task in tasks:
        task_root = source_root / "results" / task["task_id"]
        task_root.mkdir()
        (task_root / "result.json").write_text(json.dumps(result))
        terminal = {
            "identity": content_hash([plan["plan_sha256"], task]),
            "result": result,
        }
        (task_root / "terminal.json").write_text(json.dumps(terminal))
    config["source_plan_sha256"] = plan["plan_sha256"]
    config["source_summary_file_sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    config["public_asset_sha256"] = {
        benchmark_id: {
            name: hashlib.sha256(
                (public_root / "phase_b_v1" / benchmark_id / name).read_bytes()
            ).hexdigest()
            for name in campaign.PUBLIC_FILES
        }
        for benchmark_id in config["public_cells"]
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, source_root, public_root


def test_freeze_fit_resume_and_summary_are_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source, public = _source_bundle(tmp_path)
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *args: (_data(DerivativeProvenance.ESTIMATED), _context()),
    )
    output = tmp_path / "output"
    frozen = campaign.freeze_campaign(config, source, public, output)
    plan = json.loads((output / "plan.json").read_text())

    assert frozen["task_count"] == 6
    assert all(
        task["route"]["selected_backend"] == "bounded_nonlinear"
        for task in plan["tasks"]
    )
    assert (output / "frozen" / "candidate_audit.md").is_file()

    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(success=True, message=None)

    monkeypatch.setattr(campaign, "fit_candidate", fit)
    monkeypatch.setattr(
        campaign,
        "fit_result_payload",
        lambda fit: {
            "success": fit.success,
            "training_normalized_mse": 1.0,
            "validation_normalized_mse": 2.0,
            "validation_per_target_normalized_mse": {"x": 2.0},
            "diagnostics": [],
        },
    )
    first = campaign.run_task(output, 0)
    assert first["fit_success"]
    assert not first["validation_used_for_parameter_fitting"]
    assert campaign.run_task(output, 0) == first
    assert len(calls) == 1
    for index in range(1, 6):
        campaign.run_task(output, index)
    summary = campaign.summarize_campaign(output)
    assert summary["status"] == "complete"
    assert summary["fit_success_rate"] == 1.0
    assert summary["median_validation_normalized_mse_conditional_on_success"] == 2.0
    assert summary["selected_backend_counts"] == {"bounded_nonlinear": 6}
    assert not summary["candidate_regeneration_performed"]
    assert not summary["test_data_opened"]

    source_path = output / plan["tasks"][0]["source_path"]
    source_path.write_text("{}")
    with pytest.raises(ValueError, match="candidate source differs"):
        campaign.run_task(output, 0)


def test_config_rejects_estimated_derivative_profiled_route() -> None:
    config = json.loads(
        Path("configs/staged_prefit_fitting_handoff_v1.json").read_text()
    )
    config["profiled_fit_config"]["allow_derivative_regression"] = False
    with pytest.raises(ValueError, match="profiled route"):
        campaign.PrefitFittingConfig.model_validate(config)
