#!/usr/bin/env python3
"""Offline three-property audit of explicit complete training evidence and topology.

Use --example for a synthetic smoke run, or --input for a JSON object with
exactly {draft, evidence}. Older bounded prompt packets are not complete evidence.
This script never calls a model, performs fitting, or changes campaign results.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    TrainingCompatibilityEvidence,
    build_evidence,
    compact,
    diagnose,
)


class AuditInput(StrictSchema):
    """The model and independently sourced evidence, never a campaign mutation."""

    draft: Draft
    evidence: TrainingCompatibilityEvidence


def example() -> AuditInput:
    """Illustrate all three checks using known synthetic contradictions."""
    train = DatasetSplit(
        SplitName.TRAIN,
        tuple(
            Trajectory(
                trajectory_id=f"synthetic_{i}",
                time=np.array([0.0, 1.0, 2.0]),
                targets={
                    "a": np.array([0.0, peak, 0.0]),
                    "b": np.array([0.0, peak, 0.0]),
                },
                auxiliaries={},
                external_inputs={"u": np.zeros(3)},
                fixed_covariates={},
                derivatives={},
            )
            for i, peak in enumerate((1.0, 2.0))
        ),
        "synthetic-only",
    )
    context = ValidationContext(targets=("a", "b"), external_inputs=("u",))
    contract = EvidenceContract(
        provenance="Synthetic demonstration; all hypotheses explicitly imposed.",
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        interpolation={"u": "linear_complete_knots"},
        absolute_error_bounds={"a": 0, "b": 0, "u": 0},
        initial_rules={"b": InitialRule(mode="observed_initial")},
    )
    model = Draft.model_validate(
        {
            "variables": [
                {"name": "a", "definition": "algebraic", "scientific_role": "output"},
                {"name": "b", "definition": "differential", "scientific_role": "state"},
            ],
            "equations": [
                {
                    "name": name,
                    "terms": [
                        {
                            "sources": sources,
                            "outer_weight_sign": "positive",
                            "scientific_role": "response",
                        }
                    ],
                }
                for name, sources in (("a", ["u"]), ("b", ["b", "u"]))
            ],
        }
    )
    return AuditInput(draft=model, evidence=build_evidence(train, context, contract))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--example", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = (
        example()
        if args.example
        else AuditInput.model_validate_json(args.input.read_text(encoding="utf-8"))
    )
    outputs = [
        args.output / name for name in ("input.json", "report.json", "REPORT.txt")
    ]
    if args.input and args.input.resolve() in {p.resolve() for p in outputs}:
        raise ValueError("output must not overwrite the source evidence")
    report = diagnose(value.draft, value.evidence)
    saved = value.model_dump(mode="json")
    if outputs[0].exists() and json.loads(outputs[0].read_text()) != saved:
        raise ValueError(
            "existing audit directory has different input; choose a new directory"
        )
    atomic_json(outputs[0], saved)
    atomic_json(outputs[1], report.model_dump(mode="json"))
    outputs[2].write_text(compact(report) + "\n", encoding="utf-8")
    print(compact(report))


if __name__ == "__main__":
    main()
