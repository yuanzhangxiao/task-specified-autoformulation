#!/usr/bin/env python3
"""Offline smoke for compact staged proposals and feedback routing."""

from __future__ import annotations

import json

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
)
from autoformalism.search import (
    CandidateFeedbackEvidence,
    RevisionStage,
    TargetValidationMetric,
    route_proposer_feedback,
)
from autoformalism.staging import (
    enrich_functional_proposal,
    enrich_topology_proposal,
    expand_staged_candidate,
)


def main() -> None:
    """Construct and validate one small staged latent-state model."""
    context = ValidationContext(
        targets=("target",),
        external_inputs=("input_u",),
    )
    proposed_topology = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "smoke_graph",
            "change_summary": "A latent response drives a measured storage.",
            "states": [{"name": "x"}, {"name": "z"}],
            "interactions": [
                {
                    "interaction_id": "latent_drive",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["input_u"],
                },
                {
                    "interaction_id": "latent_decay",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                    "polarity": "subtractive",
                },
                {
                    "interaction_id": "storage_drive",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                },
                {
                    "interaction_id": "storage_decay",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                    "polarity": "subtractive",
                },
            ],
            "target_mappings": [{"channel": "target", "source": "x"}],
        }
    )
    topology = enrich_topology_proposal(proposed_topology, context)
    proposed_functional = ProposedFunctionalCandidate.model_validate(
        {
            "candidate_id": "smoke_functions",
            "change_summary": "Assign stable linear interaction functions.",
            "interaction_functions": [
                {"interaction_id": "latent_drive", "expression": "input_u"},
                {"interaction_id": "latent_decay", "expression": "k_z * z"},
                {"interaction_id": "storage_drive", "expression": "gain * z"},
                {"interaction_id": "storage_decay", "expression": "k_x * x"},
            ],
            "parameters": [
                {"name": "k_z", "role": "rate"},
                {"name": "gain", "role": "scale"},
                {"name": "k_x", "role": "rate"},
            ],
            "latent_initials": [
                {"state": "z", "initial": {"fixed_value": 0.0}}
            ],
        }
    )
    functional = enrich_functional_proposal(proposed_functional, topology)
    expansion = expand_staged_candidate(topology, functional, context)
    feedback = route_proposer_feedback(
        CandidateFeedbackEvidence(
            graph_mechanism_failures=("Missing delayed pathway.",),
            deterministic_validation_failures=("Undefined symbol typo.",),
            validation_metrics=(
                TargetValidationMetric(target="target", normalized_mse=2.5),
            ),
            integration_failures=("Candidate became non-finite.",),
        )
    )
    report = {
        "schema_version": "staged-feedback-routing-smoke-1",
        "status": "pass",
        "state_kinds": {
            state.name: state.kind.value for state in topology.states
        },
        "external_symbols": list(topology.external_symbols),
        "parameter_roles": {
            item.name: item.role.value for item in functional.parameters
        },
        "parameter_ranges_supplied": any(
            item.bounds is not None or item.initialization_range is not None
            for item in functional.parameters
        ),
        "topology_feedback_sources": sorted(
            {
                item["source"]
                for item in feedback.for_stage(RevisionStage.TOPOLOGY)["items"]
            }
        ),
        "functional_feedback_sources": sorted(
            {
                item["source"]
                for item in feedback.for_stage(RevisionStage.FUNCTIONAL_FORM)[
                    "items"
                ]
            }
        ),
        "candidate_identity": expansion.candidate_identity.model_dump(mode="json"),
        "topology_commitment_sha256": expansion.topology_commitment_sha256,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
