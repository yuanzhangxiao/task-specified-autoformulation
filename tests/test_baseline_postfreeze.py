"""Tests for classical final-model freezing and sealed predictive endpoints."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import (
    DatasetSplit,
    DerivativeProvenance,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.baseline_postfreeze import (
    FrozenBaselineModel,
    equations_sha256,
    evaluate_frozen_baseline_predictively,
    freeze_baseline_model,
)
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
)


def _trajectory(identifier: str, scale: float = 1.0) -> Trajectory:
    time = np.asarray((0.0, 1.0, 2.0), dtype=float)
    values = scale * np.exp(-0.5 * time)
    return Trajectory(
        trajectory_id=identifier,
        time=time,
        targets={"x": values},
        auxiliaries={},
        external_inputs={},
        fixed_covariates={},
        derivatives={"x": -0.5 * values},
        derivative_provenance=DerivativeProvenance.ESTIMATED,
    )


def _development() -> DevelopmentDataset:
    return DevelopmentDataset(
        benchmark_id="fixture",
        tier="easy",
        roles=TierRoles(targets=("x",)),
        train=DatasetSplit(
            SplitName.TRAIN,
            (_trajectory("train-0"), _trajectory("train-1", 2.0)),
            "train-fingerprint",
        ),
        validation=DatasetSplit(
            SplitName.VALIDATION,
            (_trajectory("validation-0", 1.5),),
            "validation-fingerprint",
        ),
    )


def _context() -> ValidationContext:
    return ValidationContext(targets=("x",), lagged_targets=("x",))


def _development_result(method: str) -> BaselineDevelopmentResult:
    return BaselineDevelopmentResult(
        method=method,
        benchmark_id="fixture",
        tier="easy",
        seed=0,
        equations={"x": "-0.5 * x" if method != "persistence" else "x"},
        selected_hyperparameters=(
            {"threshold": 0.01} if method == "sindy" else {}
        ),
        training_normalized_mse=0.1,
        validation_normalized_mse=0.2,
    )


def _freeze(method: str) -> FrozenBaselineModel:
    return freeze_baseline_model(
        task_index=0,
        result=_development_result(method),
        development=_development(),
        context=_context(),
        source_development_result_sha256="a" * 64,
        source_development_freeze_sha256="b" * 64,
    )


def test_sindy_final_fit_uses_selected_threshold_before_test() -> None:
    model = _freeze("sindy")

    assert model.finalization_protocol == (
        "selected_threshold_refit_on_train_plus_validation"
    )
    assert model.derivative_provenance == "estimated_numpy_gradient"
    assert model.equations_sha256 == equations_sha256(model.equations)
    assert model.equations["x"] != "0"
    assert model.test_data_opened is False


def test_pysr_final_fit_preserves_selected_equation_exactly() -> None:
    model = _freeze("pysr")

    assert model.equations == {"x": "-0.5 * x"}
    assert model.finalization_protocol == "selected_equations_preserved"


def test_predictive_endpoint_separates_persistence_and_symbolic_protocols() -> None:
    test = DatasetSplit(
        SplitName.TEST,
        (_trajectory("test-0", 1.25),),
        "test-fingerprint",
    )
    persistence = _freeze("persistence")
    symbolic = _freeze("pysr")

    persistence_result = evaluate_frozen_baseline_predictively(
        persistence,
        test,
        _context(),
        frozen_model_sha256="c" * 64,
    )
    symbolic_result = evaluate_frozen_baseline_predictively(
        symbolic,
        test,
        _context(),
        frozen_model_sha256="d" * 64,
    )

    assert persistence_result.status == "available"
    assert persistence_result.evaluation_protocol == "causal_previous_observation"
    assert symbolic_result.status == "available"
    assert symbolic_result.evaluation_protocol == (
        "causal_one_step_observed_state_reset"
    )
    assert symbolic_result.normalized_mse < persistence_result.normalized_mse


def test_frozen_symbolic_model_is_supported_by_common_adapter(tmp_path: Path) -> None:
    model = _freeze("pysr")
    source = tmp_path / "task_000.json"
    source.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")

    subject = adapt_source(
        SourceAdapterRequest(
            request_id="frozen-pysr",
            source_kind="pysr",
            source_path=source,
        ),
        _context(),
    )

    assert subject.method == "pysr"
    assert subject.candidate.state_equations[0].rhs == "-0.5 * x"
    assert subject.target_prediction.status == "missing"
    assert subject.source_provenance.source_sha256 == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_delta_postfreeze_scripts_are_cpu_only_and_dependency_bound() -> None:
    root = Path(__file__).parents[1]
    scripts = (
        root / "scripts/hpc/phase_b_public_baseline_postfreeze_prepare.slurm",
        root / "scripts/hpc/phase_b_public_baseline_predictive_test.slurm",
        root / "scripts/hpc/phase_b_public_baseline_predictive_summary.slurm",
        root / "scripts/hpc/phase_b_public_baseline_common_postfreeze.slurm",
        root / "scripts/hpc/phase_b_public_baseline_common_finalize.slurm",
        root / "scripts/hpc/phase_b_public_baseline_postfreeze_readiness.slurm",
        root / "scripts/hpc/submit_phase_b_public_baseline_postfreeze_delta.sh",
    )
    for path in scripts:
        subprocess.run(["/bin/bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "API_KEY" not in text
        assert "bibo-delta-cpu" in text
        assert "gpu" not in text.lower() or path.name.startswith("submit_")
    submit = scripts[-1].read_text(encoding="utf-8")
    assert 'dependency="afterok:${AF_DEVELOPMENT_READINESS_JOB_ID}"' in submit
    assert 'dependency="afterok:${prepare_job_id}"' in submit
    assert '--array="0-359%${AF_PREDICTIVE_CONCURRENCY}"' in submit
    assert "test_data_opened_at_submission: false" in submit
    assert "private_reference_opened: false" in submit
