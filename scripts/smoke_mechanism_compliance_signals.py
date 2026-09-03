#!/usr/bin/env python3
"""Exercise the independent graph and annotation mechanism endpoints."""

from __future__ import annotations

import json

from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    MechanismRequirement,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel


def main() -> None:
    """Require valid graph science to survive missing proposer annotations."""
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "mechanism_signal_smoke",
            "parent_candidate_id": None,
            "states": [
                {
                    "name": "target",
                    "kind": "observed",
                    "mechanisms": [],
                }
            ],
            "state_equations": [
                {"state": "target", "rhs": "input_u - target"}
            ],
            "observation_mappings": [
                {"channel": "target", "expression": "target"}
            ],
            "initial_conditions": [
                {"state": "target", "scope": "global", "fixed_value": 0.0}
            ],
        }
    )
    spec = MechanismEvaluationSpec(
        benchmark_id="synthetic",
        tier="smoke",
        required_mechanisms=(
            MechanismRequirement(
                id="input_response",
                required_drivers=("input_u",),
                required_targets=("target",),
            ),
        ),
    )
    result = evaluate_mechanisms(candidate, spec)
    passed = (
        result.graph_mechanism_compliance == 1.0
        and result.graph_mechanism_compliance_complete
        and result.mechanism_annotation_compliance == 0.0
        and result.mechanism_annotation_compliance_complete
        and result.mechanism_compliance == result.graph_mechanism_compliance
        and len(result.annotation_repairs) == 1
        and result.annotation_repairs[0].status == "unambiguous"
    )
    payload = {
        "schema_version": "mechanism-compliance-signal-smoke-1",
        "status": "pass" if passed else "fail",
        "graph_mechanism_compliance": result.graph_mechanism_compliance,
        "graph_mechanism_compliance_complete": (
            result.graph_mechanism_compliance_complete
        ),
        "mechanism_annotation_compliance": (
            result.mechanism_annotation_compliance
        ),
        "mechanism_annotation_compliance_complete": (
            result.mechanism_annotation_compliance_complete
        ),
        "legacy_alias_matches_graph": (
            result.mechanism_compliance == result.graph_mechanism_compliance
            and result.mechanism_compliance_complete
            == result.graph_mechanism_compliance_complete
        ),
        "annotation_repairs": [
            item.model_dump(mode="json") for item in result.annotation_repairs
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
