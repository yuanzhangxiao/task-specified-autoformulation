#!/usr/bin/env python3
"""CPU-only transaction-to-fitter smoke; no live LLM or benchmark reference."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.repair_comparison import RepairComparisonConfig, _assess
from autoformalism.rebuttal.repair_scientific_judge import review_request
from autoformalism.rebuttal.repair_transactions import (
    RepairAction,
    commit_action,
    default_initialization,
)
from autoformalism.schemas import CandidateModel


def main():
    problem, _ = synthetic_problem("shared", 0.0, 0)
    candidate = CandidateModel.model_validate(problem["candidate"])
    context = ValidationContext.model_validate(problem["context"])
    initial = default_initialization(candidate, context)
    revised, initial, audit = commit_action(
        candidate,
        initial,
        RepairAction.model_validate(
            {
                "scope": "model",
                "hypothesis": "Equivalent rate parameterization in an interface test",
                "equations": [
                    {
                        "component": "m",
                        "expression": "-new_rate*m+gain*u01",
                        "parameters": [{"name": "new_rate"}],
                    }
                ],
                "keep": ["y", "u01"],
            }
        ),
        context,
        memory_targets=("v01",),
    )
    config = RepairComparisonConfig(judge_revision="0" * 40, nonlinear_targets=())
    config = config.model_copy(
        update={
            "fit": config.fit.model_copy(
                update={
                    "initializer_seconds": 10,
                    "refinement_seconds": 15,
                    "recovery_probe_seconds": 1,
                    "recovery_max_starts": 3,
                }
            )
        }
    )
    dataset = SimpleNamespace(
        train=unpack_split(problem["splits"]["train"]),
        validation=unpack_split(problem["splits"]["val"]),
    )
    with tempfile.TemporaryDirectory(prefix="repair-smoke-") as temporary:
        directory = Path(temporary)
        assessment = _assess(
            directory,
            {"arm": "redesigned_runtime", "seed": 0},
            directory,
            candidate,
            revised,
            initial,
            dataset,
            context,
            "Synthetic input-memory control",
            config,
        )
        resumed = _assess(
            directory,
            {"arm": "redesigned_runtime", "seed": 0},
            directory,
            candidate,
            revised,
            initial,
            dataset,
            context,
            "Synthetic input-memory control",
            config,
        )
        assert assessment == resumed
    fit = assessment["fit"]
    assert audit["status"] == "committed" and fit["status"] == "complete", fit
    assert fit["validation"]["normalized_mse"] < 1e-5, fit
    request = review_request(
        candidate, revised, context, "Synthetic public control", 0, "0" * 40
    )
    assert not {"fit", "training", "validation", "runtime_findings"}.intersection(
        request
    )
    print(
        json.dumps(
            {
                "schema_version": "repair-comparison-smoke-1",
                "status": "pass",
                "transaction": audit["status"],
                "fit_status": fit["status"],
                "training_nmse": fit["training"]["normalized_mse"],
                "validation_nmse": fit["validation"]["normalized_mse"],
                "fitted_parameters": fit["parameters"],
                "resume_identical": True,
                "llm_calls": 0,
                "private_reference_opened": False,
                "test_data_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
