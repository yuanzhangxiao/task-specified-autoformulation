"""Incremental action compiler and conditional beam tests."""

from __future__ import annotations

import pytest

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
    FunctionalDraft,
    ProposedFunctionalActionTransaction,
    ProposedTopologyActionTransaction,
    TopologyDraft,
)
from autoformalism.staging import topology_commitment_sha256


def _context() -> ValidationContext:
    return ValidationContext(targets=("target",), external_inputs=("input_u",))


def _topology_transaction(
    *, reverse: bool = False
) -> ProposedTopologyActionTransaction:
    actions = [
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
    ]
    if reverse:
        actions = [*reversed(actions[:2]), *reversed(actions[2:6]), actions[6]]
    return ProposedTopologyActionTransaction.model_validate(
        {
            "intent": {
                "objective": "initial_construction",
                "requirement_ids": ["input_response"],
                "target_channels": ["target"],
            },
            "actions": actions,
        }
    )


def _functional_transaction(
    topology_sha256: str,
    *,
    x_drive_expression: str = "gain * z",
) -> ProposedFunctionalActionTransaction:
    return ProposedFunctionalActionTransaction.model_validate(
        {
            "topology_commitment_sha256": topology_sha256,
            "intent": {
                "objective": "initial_construction",
                "requirement_ids": ["input_response"],
                "target_channels": ["target"],
            },
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
                    "expression": x_drive_expression,
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
    )


def test_topology_actions_compile_to_existing_immutable_topology() -> None:
    application = apply_topology_actions(
        TopologyDraft(),
        _topology_transaction(),
        _context(),
        allowed_requirement_ids=("input_response",),
    )
    topology = finalize_topology_draft(application.draft, _context())

    assert application.changed is True
    assert application.added_nodes == ("x", "z")
    assert {item.name: item.kind.value for item in topology.states} == {
        "x": "observed",
        "z": "latent",
    }
    assert all(
        item.mechanisms == ("input_response",)
        for item in (*topology.states, *topology.interactions)
    )
    assert topology.target_mappings[0].source == "x"


def test_topology_hash_and_transposition_ignore_action_order() -> None:
    first = apply_topology_actions(
        TopologyDraft(), _topology_transaction(), _context()
    ).draft
    second = apply_topology_actions(
        TopologyDraft(), _topology_transaction(reverse=True), _context()
    ).draft
    table = ConstructionTranspositionTable()

    assert topology_draft_sha256(first) == topology_draft_sha256(second)
    assert table.register("topology", topology_draft_sha256(first)) is True
    assert table.register("topology", topology_draft_sha256(second)) is False
    assert table.snapshot() == (("topology", topology_draft_sha256(first)),)


def test_topology_deletion_does_not_silently_cascade() -> None:
    draft = apply_topology_actions(
        TopologyDraft(), _topology_transaction(), _context()
    ).draft
    removal = ProposedTopologyActionTransaction.model_validate(
        {
            "intent": {
                "objective": "simplification",
                "target_channels": ["target"],
            },
            "actions": [{"action": "remove_generated_node", "name": "z"}],
        }
    )

    with pytest.raises(ValueError, match="unavailable sources"):
        apply_topology_actions(draft, removal, _context())


def test_partial_functions_are_conditioned_on_the_exact_topology() -> None:
    topology = finalize_topology_draft(
        apply_topology_actions(
            TopologyDraft(), _topology_transaction(), _context()
        ).draft,
        _context(),
    )
    commitment = topology_commitment_sha256(topology)
    partial_transaction = ProposedFunctionalActionTransaction.model_validate(
        {
            "topology_commitment_sha256": commitment,
            "intent": {
                "objective": "function_repair",
                "target_channels": ["target"],
            },
            "actions": [
                {
                    "action": "set_interaction_function",
                    "interaction_id": "z_drive",
                    "expression": "input_u",
                }
            ],
        }
    )
    application = apply_functional_actions(
        FunctionalDraft(topology_commitment_sha256=commitment),
        partial_transaction,
        topology,
        _context(),
    )

    report = assess_functional_compatibility(topology, application.draft)

    assert report.status == "incomplete"
    assert report.missing_interaction_ids == (
        "x_decay",
        "x_drive",
        "z_decay",
    )
    assert report.missing_latent_initial_states == ("z",)


def test_compatible_function_actions_expand_to_an_executable_candidate() -> None:
    topology = finalize_topology_draft(
        apply_topology_actions(
            TopologyDraft(), _topology_transaction(), _context()
        ).draft,
        _context(),
    )
    commitment = topology_commitment_sha256(topology)
    application = apply_functional_actions(
        FunctionalDraft(topology_commitment_sha256=commitment),
        _functional_transaction(commitment),
        topology,
        _context(),
        allowed_requirement_ids=("input_response",),
    )

    report = assess_functional_compatibility(topology, application.draft)
    expansion = finalize_functional_draft(
        topology, application.draft, _context()
    )

    assert report.status == "compatible"
    assert application.changed is True
    assert expansion.candidate_identity.functional_sha256
    assert {item.state: item.rhs for item in expansion.candidate.state_equations} == {
        "x": "-(k_x * x) + (gain * z)",
        "z": "-(k_z * z) + (input_u)",
    }
    assert functional_draft_sha256(application.draft)


def test_incompatible_function_does_not_invalidate_its_topology() -> None:
    topology = finalize_topology_draft(
        apply_topology_actions(
            TopologyDraft(), _topology_transaction(), _context()
        ).draft,
        _context(),
    )
    commitment = topology_commitment_sha256(topology)
    application = apply_functional_actions(
        FunctionalDraft(topology_commitment_sha256=commitment),
        _functional_transaction(
            commitment,
            x_drive_expression="gain * input_u",
        ),
        topology,
        _context(),
    )

    report = assess_functional_compatibility(topology, application.draft)

    assert report.status == "incompatible"
    assert [item.code for item in report.diagnostics] == [
        "TOPOLOGY_SOURCE_MISMATCH"
    ]
    assert topology_commitment_sha256(topology) == commitment


def test_conditional_beam_reserves_one_child_per_topology_before_fill() -> None:
    a = "a" * 64
    b = "b" * 64
    entries = (
        ConditionalBeamEntry(
            topology_sha256=a, functional_sha256="1" * 64, score=0.1
        ),
        ConditionalBeamEntry(
            topology_sha256=a, functional_sha256="2" * 64, score=0.2
        ),
        ConditionalBeamEntry(
            topology_sha256=b, functional_sha256="3" * 64, score=0.15
        ),
        ConditionalBeamEntry(
            topology_sha256=b, functional_sha256="4" * 64, score=0.16
        ),
        # A worse duplicate must not replace the identical branch.
        ConditionalBeamEntry(
            topology_sha256=a, functional_sha256="1" * 64, score=9.0
        ),
    )

    selected = select_conditional_beam(
        entries,
        beam_size=3,
        maximum_functions_per_topology=2,
    )

    assert [(item.topology_sha256, item.functional_sha256) for item in selected] == [
        (a, "1" * 64),
        (b, "3" * 64),
        (b, "4" * 64),
    ]
