#!/usr/bin/env python3
"""Offline smoke test for action compilation and conditional beam diversity."""

from __future__ import annotations

import json

from autoformalism.construction import (
    ConstructionTranspositionTable,
    apply_functional_actions,
    apply_topology_actions,
    assess_functional_compatibility,
    finalize_functional_draft,
    finalize_topology_draft,
    functional_draft_sha256,
    select_conditional_beam,
    topology_draft_sha256,
)
from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    ConditionalBeamEntry,
    ConstructionIntent,
    FunctionalDraft,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    TopologyDraft,
)
from autoformalism.staging import topology_commitment_sha256


def main() -> None:
    """Compile one action-built model without any LLM or data access."""
    context = ValidationContext(
        targets=("target",),
        external_inputs=("input_u",),
    )
    intent = ConstructionIntent(
        objective="initial_construction",
        requirement_ids=("input_response",),
        target_channels=("target",),
    )
    topology_application = apply_topology_actions(
        TopologyDraft(),
        intent,
        ProposedTopologyActionTransaction.model_validate(
            {
                "actions": [
                    {"action": "add_state", "name": "x"},
                    {"action": "add_state", "name": "z"},
                    {
                        "action": "add_interaction",
                        "interaction_id": "z_drive",
                        "target": "z",
                        "sources": ["input_u"],
                    },
                    {
                        "action": "add_interaction",
                        "interaction_id": "z_decay",
                        "target": "z",
                        "sources": ["z"],
                        "polarity": "subtractive",
                    },
                    {
                        "action": "add_interaction",
                        "interaction_id": "x_drive",
                        "target": "x",
                        "sources": ["z"],
                    },
                    {
                        "action": "add_interaction",
                        "interaction_id": "x_decay",
                        "target": "x",
                        "sources": ["x"],
                        "polarity": "subtractive",
                    },
                    {
                        "action": "set_target_mapping",
                        "channel": "target",
                        "source": "x",
                    },
                ],
            }
        ),
        context,
        allowed_requirement_ids=("input_response",),
    )
    topology = finalize_topology_draft(topology_application.draft, context)
    commitment = topology_commitment_sha256(topology)
    functional_application = apply_functional_actions(
        FunctionalDraft(topology_commitment_sha256=commitment),
        intent,
        ProposedFunctionalActionTransaction.model_validate(
            {
                "actions": [
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "z_drive",
                        "expression": "input_u",
                    },
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "z_decay",
                        "expression": "k_z * z",
                        "parameters": [{"name": "k_z", "role": "rate"}],
                    },
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "x_drive",
                        "expression": "gain * z",
                        "parameters": [{"name": "gain", "role": "scale"}],
                    },
                    {
                        "action": "set_interaction_function",
                        "interaction_id": "x_decay",
                        "expression": "k_x * x",
                        "parameters": [{"name": "k_x", "role": "rate"}],
                    },
                    {
                        "action": "set_latent_initial",
                        "state": "z",
                        "initial": {"fixed_value": 0.0},
                    },
                ],
            }
        ),
        topology,
        context,
        allowed_requirement_ids=("input_response",),
    )
    compatibility = assess_functional_compatibility(
        topology, functional_application.draft
    )
    expansion = finalize_functional_draft(
        topology,
        functional_application.draft,
        context,
    )
    topology_sha = topology_draft_sha256(topology_application.draft)
    functional_sha = functional_draft_sha256(functional_application.draft)
    table = ConstructionTranspositionTable()
    first_registration = table.register("topology", topology_sha)
    duplicate_registration = table.register("topology", topology_sha)
    selected = select_conditional_beam(
        (
            ConditionalBeamEntry(
                topology_sha256=topology_sha,
                functional_sha256=functional_sha,
                score=0.1,
            ),
            ConditionalBeamEntry(
                topology_sha256="a" * 64,
                functional_sha256="b" * 64,
                score=0.2,
            ),
        ),
        beam_size=2,
        maximum_functions_per_topology=2,
    )
    print(
        json.dumps(
            {
                "schema_version": "incremental-construction-smoke-1",
                "status": "pass",
                "compatibility": compatibility.status,
                "topology_sha256": topology_sha,
                "functional_sha256": functional_sha,
                "candidate_identity": expansion.candidate_identity.model_dump(
                    mode="json"
                ),
                "state_equations": {
                    item.state: item.rhs
                    for item in expansion.candidate.state_equations
                },
                "first_transposition_registration": first_registration,
                "duplicate_transposition_registration": duplicate_registration,
                "selected_topology_count": len(
                    {item.topology_sha256 for item in selected}
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
