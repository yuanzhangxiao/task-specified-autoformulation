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
from autoformalism.rebuttal.repair_drafts import RepairActionV2, advance, empty_draft
from autoformalism.rebuttal.repair_evidence import decision_report, numerical_findings
from autoformalism.rebuttal.repair_feedback import initialization_facts
from autoformalism.rebuttal.repair_fit_reporting import fit_outcomes
from autoformalism.rebuttal.repair_scientific_judge import review_request
from autoformalism.rebuttal.repair_transactions import (
    default_initialization,
)
from autoformalism.schemas import CandidateModel


def main():
    problem, _ = synthetic_problem("shared", 0.0, 0)
    candidate = CandidateModel.model_validate(problem["candidate"])
    context = ValidationContext.model_validate(problem["context"])
    initial = default_initialization(candidate, context)
    revised, initial, draft, diagnostics = advance(
        candidate,
        initial,
        context,
        empty_draft(),
        RepairActionV2.model_validate(
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
            }
        ),
        memory_targets=("v01",),
    )
    assert not diagnostics, diagnostics
    audit = draft["audit"]
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
    facts = initialization_facts(revised, initial, context, fit)
    priority = decision_report(
        revised,
        [],
        [],
        numerical_findings(revised, fit),
        [],
        routing_policy="evidence_strength",
        fit=fit,
    )
    assert priority["objective_category"] == "prediction_refinement"
    assert priority["numerical_reliability"]["selected_training_replay_verified"]
    assert not priority["numerical_reliability"]["global_infeasibility_established"]
    assert all(b["selected_values_available"] for b in facts["latent_boundaries"])
    request = review_request(
        candidate, revised, context, "Synthetic public control", 0, "0" * 40
    )
    assert not {"fit", "training", "validation", "runtime_findings"}.intersection(
        request
    )
    print(
        json.dumps(
            {
                "schema_version": "repair-comparison-smoke-3",
                "initialization_facts": facts,
                "priority_policy": priority["priority_evidence"]["protocol"],
                "numerical_support": priority["numerical_reliability"]["support"],
                "status": "pass",
                "transaction": audit["status"],
                "fit_status": fit["status"],
                "training_nmse": fit["training"]["normalized_mse"],
                "validation_nmse": fit["validation"]["normalized_mse"],
                "fitted_parameters": fit["parameters"],
                "fit_outcomes": fit_outcomes(fit),
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
