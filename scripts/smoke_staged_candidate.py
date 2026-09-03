#!/usr/bin/env python3
"""Exercise topology-to-functional-to-executable expansion without data or LLMs."""

from __future__ import annotations

import json

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import FunctionalCandidate, TopologyCandidate
from autoformalism.staging import (
    expand_staged_candidate,
    topology_commitment_sha256,
)


def main() -> None:
    """Build and validate a small partially observed staged ODE model."""
    topology = TopologyCandidate.model_validate(
        {
            "candidate_id": "smoke_topology",
            "change_summary": "Driven latent state controls a measured storage.",
            "states": [
                {"name": "x", "kind": "observed"},
                {"name": "z", "kind": "latent"},
            ],
            "processes": [{"name": "generated_flux"}],
            "external_symbols": ["input_u"],
            "interactions": [
                {
                    "interaction_id": "latent_drive",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["input_u"],
                },
                {
                    "interaction_id": "latent_relaxation",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                    "polarity": "subtractive",
                },
                {
                    "interaction_id": "flux_generation",
                    "target": "generated_flux",
                    "target_kind": "algebraic_process",
                    "sources": ["z"],
                },
                {
                    "interaction_id": "storage_source",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["generated_flux"],
                },
                {
                    "interaction_id": "storage_sink",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                    "polarity": "subtractive",
                },
            ],
            "observation_mappings": [
                {"channel": "target", "source": "x"}
            ],
        }
    )
    functional = FunctionalCandidate.model_validate(
        {
            "candidate_id": "smoke_functional",
            "change_summary": "Assign affine response functions.",
            "topology_commitment_sha256": topology_commitment_sha256(topology),
            "interaction_functions": [
                {"interaction_id": "latent_drive", "expression": "input_u"},
                {
                    "interaction_id": "latent_relaxation",
                    "expression": "latent_decay * z",
                },
                {
                    "interaction_id": "flux_generation",
                    "expression": "gain * z",
                },
                {
                    "interaction_id": "storage_source",
                    "expression": "generated_flux",
                },
                {
                    "interaction_id": "storage_sink",
                    "expression": "storage_decay * x",
                },
            ],
            "parameters": [
                {
                    "name": name,
                    "scope": "global",
                    "bounds": {"lower": 0.01, "upper": 5.0},
                    "initialization_range": {"lower": 0.1, "upper": 1.0},
                }
                for name in ("latent_decay", "gain", "storage_decay")
            ],
            "initial_conditions": [
                {"state": "x", "scope": "global", "expression": "target"},
                {"state": "z", "scope": "global", "fixed_value": 0.0},
            ],
        }
    )
    result = expand_staged_candidate(
        topology,
        functional,
        ValidationContext(
            targets=("target",),
            external_inputs=("input_u",),
        ),
    )
    print(
        json.dumps(
            {
                "schema_version": "staged-candidate-smoke-1",
                "status": "pass",
                "topology_commitment_sha256": (
                    result.topology_commitment_sha256
                ),
                "candidate_identity": result.candidate_identity.model_dump(
                    mode="json"
                ),
                "state_count": len(result.candidate.states),
                "latent_state_count": sum(
                    state.kind.value == "latent"
                    for state in result.candidate.states
                ),
                "algebraic_process_count": len(result.candidate.processes),
                "state_equations": {
                    equation.state: equation.rhs
                    for equation in result.candidate.state_equations
                },
                "observation_mappings": {
                    mapping.channel: mapping.expression
                    for mapping in result.candidate.observation_mappings
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
