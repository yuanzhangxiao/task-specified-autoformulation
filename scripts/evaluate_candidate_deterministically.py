#!/usr/bin/env python3
"""Evaluate public deterministic endpoints for one frozen candidate model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    FrozenParameterization,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
    evaluate_frozen_subject,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation-context", type=Path, required=True)
    parser.add_argument("--mechanism-spec", type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = CandidateModel.model_validate_json(
        args.candidate.read_text(encoding="utf-8")
    )
    candidate_sha256 = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    context = ValidationContext.model_validate_json(
        args.validation_context.read_text(encoding="utf-8")
    )
    mechanism_spec = (
        None
        if args.mechanism_spec is None
        else MechanismEvaluationSpec.model_validate_json(
            args.mechanism_spec.read_text(encoding="utf-8")
        )
    )
    subject = FrozenEvaluationSubject(
        subject_id=args.subject_id,
        method=args.method,
        benchmark_id=args.benchmark_id,
        tier=args.tier,
        repetition=args.repetition,
        private_metrics_opened_after_freeze=False,
        source_provenance=SourceArtifactProvenance(
            adapter="direct_candidate",
            request_id=args.subject_id,
            source_path=str(args.candidate.resolve()),
            source_sha256=candidate_sha256,
            candidate_sha256=hashlib.sha256(
                candidate.model_dump_json().encode("utf-8")
            ).hexdigest(),
        ),
        candidate=candidate,
        parameterization=FrozenParameterization(
            status=(
                "not_required"
                if not candidate.parameters
                and not any(
                    item.initialization_range is not None
                    for item in candidate.initial_conditions
                )
                else "missing"
            )
        ),
        validation_context=context,
        target_prediction=TargetPredictionEndpoint(status="missing"),
    )
    record = evaluate_frozen_subject(subject, mechanism_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote deterministic evaluation for {args.subject_id} to {args.output}")


if __name__ == "__main__":
    main()
