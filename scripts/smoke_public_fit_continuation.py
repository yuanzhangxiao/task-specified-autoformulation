#!/usr/bin/env python3
"""Real numerical continuation control with explicitly synthetic parent history.

The parent vector and scores are real, while its four timeout-history entries
are a test fixture. This checks the continuation path, not its policy efficacy.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke_public_fitting import control

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.continuation_numerics import _score
from autoformalism.fitting.fit_continuation import (
    execute_continuation,
    prepare_continuation,
)
from autoformalism.schemas.fit_continuation import ContinuationSelection
from autoformalism.schemas.public_fitting import PublicFitResult


def fixture_parent(directory: Path, *, real_scores: bool = False):
    """Seal a synthetic parent; never export this fixture as experiment evidence."""
    request, train, val = control("collocation-feasible-v1")
    public.prepare_fit(request, train, val, directory)
    frozen = public._read(directory / "freeze.json")
    parameters = {"a": 0.9, "b": 0.8, "init_z_scale": 0.3}
    metrics = {
        "normalized_mse": 0.2,
        "per_target_normalized_mse": {"v01": 0.2},
        "failed_trajectories": [],
    }
    train_metrics, val_metrics = metrics, metrics
    cost = 4.2  # 0.5 * 42 training observations * 0.2 NMSE.
    if real_scores:
        from autoformalism.data import TrainingScaler
        from autoformalism.fitting.collocation_sensitivity import (
            CollocationSensitivityConfig,
        )

        model, _, _ = public._lower(request)
        training = public.unpack_split(train)
        scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
        settings = CollocationSensitivityConfig.model_validate(
            frozen["settings"]
        ).fit_config()
        train_metrics = _score(model, training, parameters, scale, settings)["metrics"]
        val_metrics = _score(
            model, public.unpack_split(val), parameters, scale, settings
        )["metrics"]
        cost = (
            train_metrics["normalized_mse"]
            * 0.5
            * sum(len(row.time) for row in train.rows)
        )
    stage = {
        "parameters": parameters,
        "cost": cost,
        "optimizer_status": -2,
        "optimizer_native_success": False,
        "best_evaluated": {"parameters": parameters, "cost": cost, "call": 5},
        "iterations": [
            {
                "iteration": i,
                "nfev": i + 2,
                "cost": cost * factor,
                "parameters": parameters,
            }
            for i, factor in enumerate((1.3, 1.2, 1.1, 1.0))
        ],
    }
    raw = {
        "parameters": parameters,
        "training": train_metrics,
        "validation": val_metrics,
        "refinement": {
            "budget_exhausted": True,
            "actual_residual_calls": 6,
            "stages": [{"mode": "sensitivity", "result": stage}],
        },
        "synthetic_test_history": True,
    }
    result = PublicFitResult(
        **public._result_base(frozen, request),
        **public._evidence(request, raw),
        status="complete",
        backend_result_sha256=public.content_sha256(raw),
        message="Synthetic history fixture; real parent scores only when requested.",
    )
    public._write(directory / "backend_result.json", raw)
    public._write(
        directory / "result.json",
        {
            "result": result.model_dump(mode="json"),
            "sha256": public.content_sha256(result),
        },
    )
    return ContinuationSelection(
        parent_lowered_candidate_sha256=result.lowered_candidate_sha256,
        parent_backend_result_sha256=result.backend_result_sha256,
    )


def main() -> None:
    with TemporaryDirectory(prefix="public-fit-continuation-") as temporary:
        root = Path(temporary)
        parent, output = root / "parent/fit", root / "extension"
        selection = fixture_parent(parent, real_scores=True)
        before = {p.name: p.read_bytes() for p in parent.glob("*.json")}
        prepared = prepare_continuation(parent, selection, output)
        assert prepared["gate"]["eligible"]
        result = execute_continuation(output)
        assert result.status == "complete", result.message
        assert result.selected == "extension"
        assert result.training.normalized_mse < 1e-8
        assert result.validation.normalized_mse < 1e-8
        # This output admits equivalent rate/latent scalings; require that the
        # latent-map parameter was fitted, not a unique generating value.
        assert abs(result.parameters["init_z_scale"] - 0.3) > 1e-3
        assert execute_continuation(output) == result
        assert before == {p.name: p.read_bytes() for p in parent.glob("*.json")}
        print(
            json.dumps(
                {
                    "status": "passed",
                    "synthetic_parent_history": True,
                    "real_numerical_extension": True,
                    "parent_unchanged": True,
                    "training_nmse": result.training.normalized_mse,
                    "validation_nmse": result.validation.normalized_mse,
                    "all_parameters_carried": True,
                    "resume_unchanged": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
