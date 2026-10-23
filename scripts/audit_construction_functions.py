#!/usr/bin/env python3
"""Audit a sealed train-only model roster without optimization or LLM calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import (
    construction_function_consistency,
    construction_parameter_audit,
    construction_revision_consistency,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.topology_compatibility import (
    TrainingCompatibilityEvidence,
    validate_evidence,
)
from autoformalism.staged_topology import content_hash

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Model(StrictSchema):
    """One original model identity; never include fitted parameters or scores."""

    model_id: str = Field(min_length=1, max_length=200)
    source_role: Literal["parent", "child", "candidate"]
    parent_id: str | None = None
    problem: PrefixFitProblem


class Inputs(StrictSchema):
    """Positive data boundary: only training evidence and unfitted candidates."""

    protocol: Literal["construction-function-science-inputs-1"]
    models: tuple[Model, ...] = Field(min_length=1, max_length=256)
    evidence: TrainingCompatibilityEvidence
    source_sha256: Digest
    evidence_sha256: Digest
    test_data_opened: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        validate_evidence(self.evidence)
        if len({m.model_id for m in self.models}) != len(self.models):
            raise ValueError("duplicate model identity")
        if any(m.problem.context != self.evidence.context for m in self.models):
            raise ValueError("model and training evidence contexts differ")
        if content_hash(self.evidence.model_dump(mode="json")) != self.evidence_sha256:
            raise ValueError("training evidence digest differs")
        return self


def _assess_model(item: Model, evidence: dict) -> dict:
    """Retain a candidate's expected semantic validation failure as unavailable.

    The outer envelope, channel context and evidence hashes have already passed
    validation. Programming errors, I/O errors and unsupported exception classes
    still propagate; only candidate validation failures become report rows.
    """
    row = {
        "model_id": item.model_id,
        "source_role": item.source_role,
        "parent_id": item.parent_id,
        "audit_scope": "supplied_model",
        "problem_sha256": content_hash(item.problem.model_dump(mode="json")),
    }
    stage = "matched_condition_equality"
    try:
        equality = construction_revision_consistency.assess(item.problem, evidence)
        stage = "function_domain_consistency"
        functions = construction_function_consistency.assess(item.problem, evidence)
        stage = "parameter_bookkeeping"
        parameters = construction_parameter_audit.assess(item.problem)
        if parameters.get("status") == "unavailable":
            raise ValueError(parameters.get("reason", "parameter audit unavailable"))
    except (ModelValidationError, ValueError) as error:
        diagnostic = {
            "code": "CANDIDATE_ASSESSMENT_UNAVAILABLE",
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error)[:4000],
        }
        if isinstance(error, ModelValidationError):
            diagnostic["model_diagnostics"] = [
                {"code": item.code, "location": item.location, "message": item.message}
                for item in error.diagnostics
            ]
        return {
            **row,
            "status": "unavailable",
            "diagnostic": diagnostic,
            "parameter_audit": {
                "status": "unavailable",
                "reason": diagnostic["message"],
            },
            "training_consistency": {
                "status": "unavailable",
                "checks": [],
                "conflicts": [],
                "unavailable": [diagnostic],
                "scientific_correctness_established": False,
            },
        }
    status = (
        "blocked"
        if equality["conflicts"] or functions["conflicts"]
        else "deferred"
        if "deferred" in (equality["status"], functions["status"])
        else "no_conflict_found"
    )
    return {
        **row,
        "status": status,
        "parameter_audit": parameters,
        "training_consistency": {
            **equality,
            "equality_status": equality["status"],
            "status": "conflict" if status == "blocked" else status,
            "function_consistency": functions,
        },
    }


def audit(source: Path, output: Path) -> dict:
    """Validate the train-only envelope, then retain each model's audit outcome."""
    from collections import Counter

    from autoformalism.research.construction_science_report import render

    raw = read_seal(source)
    value = Inputs.model_validate(raw)
    evidence = value.evidence.model_dump(mode="json")
    models = [_assess_model(item, evidence) for item in value.models]
    summary = {
        "protocol": "construction-function-science-audit-1",
        "source_sha256": value.source_sha256,
        "input_sha256": content_hash(raw),
        "evidence_sha256": value.evidence_sha256,
        "runtime_source_sha256": runtime_source_hash(),
        "model_count": len(models),
        "unavailable_model_count": sum(m["status"] == "unavailable" for m in models),
        "status_counts": dict(Counter(m["status"] for m in models)),
        "optimizer_calls": 0,
        "llm_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "fitted_metrics_used": False,
        "scientific_compliance_assessed": False,
        "interpretation": "Finite necessary-condition checks; no_conflict_found "
        "and deferred do not certify scientific validity or achievable NMSE.",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "audit.json", {"summary": summary, "models": models})
    render(output / "REPORT.html", summary, models, title="Function construction audit")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
