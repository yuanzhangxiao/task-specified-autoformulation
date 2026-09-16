#!/usr/bin/env python3
"""Synthetic open-rollout smoke: no benchmark files, fitting, or LLM calls."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.baseline_validation import (
    ReplaySettings,
    evaluate_validation,
)
from autoformalism.rebuttal.final_evaluation_adapters import (
    SourceAdapterRequest,
    adapt_source,
)


def main() -> None:
    """Validate an analytic decay with initial values varying across trajectories."""

    def trajectory(identifier: str, initial: float) -> Trajectory:
        time = np.linspace(0, 2, 21)
        return Trajectory(
            trajectory_id=identifier,
            time=time,
            targets={"x": initial * np.exp(-time)},
            auxiliaries={},
            external_inputs={},
            fixed_covariates={},
            derivatives={},
        )

    train = DatasetSplit(SplitName.TRAIN, (trajectory("train", 1.0),), "train")
    validation = DatasetSplit(
        SplitName.VALIDATION, (trajectory("v1", 2.0), trajectory("v2", 3.0)), "val"
    )
    with TemporaryDirectory() as directory:
        source = Path(directory) / "result.json"
        source.write_text(
            BaselineDevelopmentResult(
                method="sindy",
                benchmark_id="synthetic",
                tier="easy",
                seed=0,
                equations={"x": "-x"},
                selected_hyperparameters={},
                training_normalized_mse=0,
                validation_normalized_mse=0,
            ).model_dump_json()
        )
        subject = adapt_source(
            SourceAdapterRequest(
                request_id="smoke", source_kind="sindy", source_path=source
            ),
            ValidationContext(targets=("x",), lagged_targets=("x",)),
        )
        result = evaluate_validation(subject, train, validation, ReplaySettings())
    assert result["status"] == "complete"
    assert result["normalized_mse"] < 1e-10
    print(
        json.dumps(
            {
                "status": "pass",
                "validation_nmse": result["normalized_mse"],
                "live_llm_calls": 0,
                "test_data_opened": False,
                "parameter_refit_applied": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
