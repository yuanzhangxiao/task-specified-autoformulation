"""Frozen-source, warm-start, screening, and rescue campaign tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import autoformalism.rebuttal.staged_fitter_rescue_campaign as campaign
from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import (
    ExactDerivativeFitError,
    FitConfig,
    estimate_profiled_warm_start_from_public_derivatives,
)
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
                {
                    "name": "b",
                    "scope": "global",
                    "role": "nonnegative_coefficient",
                },
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "x"},
                {"state": "z", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )


def _dataset(provenance: DerivativeProvenance) -> DevelopmentDataset:
    def split(name: SplitName, fingerprint: str) -> DatasetSplit:
        time = np.linspace(0.0, 2.0, 9)
        values = 2.0 * (time - 1.0 + np.exp(-time))
        trajectory = Trajectory(
            trajectory_id=f"{name.value}-0",
            time=time,
            targets={"x": values},
            auxiliaries={},
            external_inputs={"u": np.ones_like(time)},
            fixed_covariates={},
            derivatives={"x": 2.0 * (1.0 - np.exp(-time))},
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
    return ValidationContext(targets=("x",), external_inputs=("u",))


def test_estimated_derivatives_are_accepted_only_by_initializer() -> None:
    model = compile_candidate(_candidate(), _context())
    config = FitConfig(
        parameter_fit_strategy="profiled_latent_basis_linear_ridge",
        allow_derivative_regression=True,
        maximum_function_evaluations=6,
        maximum_wall_time_seconds=10.0,
    )
    estimated = _dataset(DerivativeProvenance.ESTIMATED)
    unavailable = _dataset(DerivativeProvenance.UNAVAILABLE)

    result = estimate_profiled_warm_start_from_public_derivatives(
        model,
        estimated.train,
        config,
        initial_global_parameters={"a": 1.0, "b": 1.0},
    )

    assert set(result.global_parameters) == {"a", "b"}
    assert np.isfinite(list(result.global_parameters.values())).all()
    with pytest.raises(ValueError, match="requires training data"):
        estimate_profiled_warm_start_from_public_derivatives(
            model,
            estimated.validation,
            config,
        )
    with pytest.raises(ExactDerivativeFitError, match="estimated or exact"):
        estimate_profiled_warm_start_from_public_derivatives(
            model,
            unavailable.train,
            config,
        )


def _source_bundle(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    candidate = _candidate()
    tasks = []
    public_ledger = {}
    cells = ("cell_easy", "cell_hard")
    for cell in cells:
        for name in (
            "manifest.json",
            "proposer_prompt.txt",
            "train.csv",
            "validation.csv",
        ):
            path = source / "frozen" / "public" / "phase_b_v1" / cell / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{cell}:{name}\n")
            public_ledger[str(path.relative_to(source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for index, (cell, seed) in enumerate(
        (cell, seed) for cell in cells for seed in range(3)
    ):
        candidate_path = (
            source / "frozen" / "candidates" / f"candidate_{index:03d}.json"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(candidate.model_dump_json())
        tasks.append(
            {
                "task_index": index,
                "task_id": f"{cell}_seed{seed}_fit",
                "benchmark_id": cell,
                "tier": cell.rsplit("_", 1)[-1],
                "seed": seed,
                "candidate_path": str(candidate_path.relative_to(source)),
                "candidate_file_sha256": hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
            }
        )
    plan = {
        "schema_version": "source",
        "tasks": tasks,
        "public_asset_ledger": public_ledger,
        "candidate_regeneration_performed": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    (source / "plan.json").write_text(json.dumps(plan))
    for task in tasks:
        path = source / "tasks" / f"task_{task['task_index']:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "plan_sha256": plan["plan_sha256"],
                    "candidate_file_sha256": task["candidate_file_sha256"],
                    "fit_success": False,
                    "error": "synthetic timeout",
                    "fit": {"global_parameters": {"a": 1.0, "b": 1.0}},
                }
            )
        )
    summary = {
        "status": "complete",
        "terminal_results": 6,
        "candidate_regeneration_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    summary_path = source / "summary" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary))
    config = json.loads(Path("configs/staged_fitter_rescue_v1.json").read_text())
    config["source_fitting_plan_sha256"] = plan["plan_sha256"]
    config["source_fitting_summary_file_sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, source


def test_freeze_run_resume_and_summary_are_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, source = _source_bundle(tmp_path)
    output = tmp_path / "output"
    frozen = campaign.freeze_campaign(config, source, output)
    dataset = _dataset(DerivativeProvenance.ESTIMATED)
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *args: (dataset, _context()),
    )
    monkeypatch.setattr(
        campaign,
        "_derivative_initialization",
        lambda *args: {
            "attempted": True,
            "usable": True,
            "parameters": {"a": 1.0, "b": 2.0},
            "fit": {},
            "wall_seconds": 0.0,
            "error_type": None,
            "error": None,
        },
    )
    monkeypatch.setattr(
        campaign,
        "_score_split",
        lambda *args: {
            "complete": True,
            "normalized_mse": 0.5,
            "per_target_normalized_mse": {"x": 0.5},
            "failed_trajectory_count": 0,
            "failures": [],
            "wall_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        campaign,
        "_score_train_then_validation",
        lambda *args: {
            "complete": False,
            "training": None,
            "validation": None,
            "reason": "synthetic source failure",
        },
    )

    def fit(*args, initial_global_parameters=None, **kwargs):
        assert args[1] is args[2]
        if initial_global_parameters["b"] == 2.0:
            raise RuntimeError("synthetic first-start failure")
        return SimpleNamespace(
            success=True,
            message=None,
            global_parameters=dict(initial_global_parameters),
        )

    monkeypatch.setattr(campaign, "fit_candidate", fit)
    monkeypatch.setattr(
        campaign,
        "fit_result_payload",
        lambda result: {
            "success": result.success,
            "global_parameters": result.global_parameters,
            "diagnostics": [],
        },
    )

    assert frozen["status"] == "frozen_before_rescue"
    first = campaign.run_task(output, 0)
    assert first["rescue_score"]["complete"]
    assert first["rescue_attempts"][0]["fresh_training_score"] is None
    assert first["rescue_attempts"][1]["fresh_training_score"]["complete"]
    assert first["attribution"] == "fitter_initialization_or_search_limited"
    assert campaign.run_task(output, 0) == first
    for index in range(1, 6):
        campaign.run_task(output, index)
    summary = campaign.summarize_campaign(output)
    assert summary["status"] == "complete"
    assert summary["rescue_complete_rate"] == 1.0
    assert summary["estimated_derivatives_used_only_for_initialization"]
    assert not summary["validation_used_for_start_selection"]
    assert not summary["multiple_round_search_performed"]

    source_result = output / "frozen" / "source_results" / "task_000.json"
    source_result.write_text("{}")
    with pytest.raises(ValueError, match="frozen source artifact differs"):
        campaign.run_task(output, 1)


def test_nonfinite_rollout_error_is_not_a_stable_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compile_candidate(_candidate(), _context())
    dataset = _dataset(DerivativeProvenance.ESTIMATED)
    monkeypatch.setattr(
        campaign,
        "simulate_trajectory",
        lambda *args, **kwargs: SimpleNamespace(
            success=True,
            predictions={"x": np.full(9, 1e308)},
            message=None,
        ),
    )

    score = campaign._score_split(
        model,
        dataset.train,
        {"a": 1.0, "b": 1.0},
        {"x": 1.0},
        FitConfig(),
        1.0,
    )

    assert not score["complete"]
    assert score["failed_trajectory_count"] == 1
    assert "non-finite normalized squared residuals" in score["failures"][0]["message"]
