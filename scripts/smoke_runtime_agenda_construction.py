#!/usr/bin/env python3
"""Offline smoke for runtime agenda selection and phased topology actions."""

from __future__ import annotations

import json

from autoformalism.construction import (
    apply_topology_actions,
    finalize_topology_draft,
)
from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    ConstructionIntent,
    ProposedTopologyActionTransaction,
    TopologyConstructionPhase,
    TopologyDraft,
)
from autoformalism.search import (
    CandidateFeedbackEvidence,
    TargetValidationMetric,
    route_proposer_feedback,
    select_construction_agenda,
    select_next_revision_stage,
)


def main() -> None:
    """Compile one phase-ordered graph without LLM, fitting, or data access."""
    context = ValidationContext(targets=("target",), external_inputs=("input_u",))
    feedback = route_proposer_feedback(
        CandidateFeedbackEvidence(
            graph_mechanism_failures=("Required response path is missing.",),
            validation_metrics=(
                TargetValidationMetric(target="target", normalized_mse=9.0),
            ),
        )
    )
    next_stage = select_next_revision_stage(feedback)
    agenda = select_construction_agenda(
        stage="topology",
        feedback=feedback,
        allowed_requirement_ids=("input_response", "output_readout"),
        target_channels=context.targets,
    )
    intent = ConstructionIntent(
        objective=agenda.objective,
        requirement_ids=agenda.eligible_requirement_ids,
        target_channels=agenda.eligible_target_channels,
    )
    draft = TopologyDraft()
    transactions = (
        (
            TopologyConstructionPhase.COMPONENT_SPECIFICATION,
            {
                "actions": [
                    {"action": "add_state", "name": "x"},
                    {"action": "add_process", "name": "predicted_target"},
                ]
            },
        ),
        (
            TopologyConstructionPhase.DYNAMIC_TOPOLOGY,
            {
                "actions": [
                    {
                        "action": "add_interaction",
                        "interaction_id": "input_drive",
                        "target": "x",
                        "sources": ["input_u"],
                    }
                ]
            },
        ),
        (
            TopologyConstructionPhase.ALGEBRAIC_READOUT_TOPOLOGY,
            {
                "actions": [
                    {
                        "action": "add_interaction",
                        "interaction_id": "readout",
                        "target": "predicted_target",
                        "sources": ["x"],
                    },
                    {
                        "action": "set_target_mapping",
                        "channel": "target",
                        "source": "predicted_target",
                    },
                ]
            },
        ),
    )
    for phase, payload in transactions:
        application = apply_topology_actions(
            draft,
            intent,
            ProposedTopologyActionTransaction.model_validate(payload),
            context,
            allowed_requirement_ids=agenda.eligible_requirement_ids,
            topology_phase=phase,
            attach_intent_mechanisms=False,
        )
        draft = application.draft
    topology = finalize_topology_draft(draft, context)
    print(
        json.dumps(
            {
                "schema_version": "runtime-agenda-construction-smoke-1",
                "status": "pass",
                "next_revision_stage": next_stage,
                "selected_feedback_category": agenda.feedback_category,
                "selected_mechanism_count": len(intent.requirement_ids),
                "state_count": len(topology.states),
                "process_count": len(topology.processes),
                "mechanism_annotations_attached": any(
                    item.mechanisms
                    for item in (*topology.states, *topology.interactions)
                ),
                "topology_phases": [phase.value for phase, _ in transactions],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
