"""Staged topology, functional assignment, and expansion tests."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.schemas import (
    FunctionalCandidate,
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
    TopologyCandidate,
)
from autoformalism.search import candidate_identity
from autoformalism.staging import (
    enrich_functional_proposal,
    enrich_topology_proposal,
    expand_staged_candidate,
    normalize_topology_proposal,
    topology_commitment_sha256,
)


def _topology_payload() -> dict[str, Any]:
    return {
        "candidate_id": "topology_0",
        "parent_candidate_id": None,
        "change_summary": "A driven latent response controls a measured state.",
        "states": [
            {
                "name": "x",
                "kind": "observed",
                "unit": "relative",
                "description": "Measured storage.",
            },
            {
                "name": "z",
                "kind": "latent",
                "unit": "relative",
                "description": "Delayed response.",
            },
        ],
        "processes": [
            {
                "name": "flux",
                "unit": "relative/time",
                "description": "Generated response flux.",
            }
        ],
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
                "target": "flux",
                "target_kind": "algebraic_process",
                "sources": ["z"],
            },
            {
                "interaction_id": "storage_source",
                "target": "x",
                "target_kind": "state_derivative",
                "sources": ["flux"],
            },
            {
                "interaction_id": "storage_sink",
                "target": "x",
                "target_kind": "state_derivative",
                "sources": ["x"],
                "polarity": "subtractive",
            },
        ],
        "target_mappings": [
            {"channel": "target", "source": "x", "unit": "relative"}
        ],
    }


def _functional_payload(topology: TopologyCandidate) -> dict[str, Any]:
    return {
        "candidate_id": "functional_0",
        "parent_candidate_id": None,
        "change_summary": "Assign linear response and relaxation functions.",
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
            {"interaction_id": "storage_source", "expression": "flux"},
            {
                "interaction_id": "storage_sink",
                "expression": "storage_decay * x",
            },
        ],
        "parameters": [
            {
                "name": name,
                "scope": "global",
                "role": role,
                "bounds": {"lower": 0.01, "upper": 5.0},
                "initialization_range": {"lower": 0.1, "upper": 1.0},
            }
            for name, role in (
                ("latent_decay", "rate"),
                ("gain", "scale"),
                ("storage_decay", "rate"),
            )
        ],
        "initial_conditions": [
            {"state": "x", "scope": "global", "expression": "target"},
            {"state": "z", "scope": "global", "fixed_value": 0.0},
        ],
    }


def _context() -> ValidationContext:
    return ValidationContext(targets=("target",), external_inputs=("input_u",))


def _proposed_topology_payload() -> dict[str, Any]:
    payload = _topology_payload()
    payload["schema_version"] = "proposed-topology-candidate-2"
    payload.pop("external_symbols")
    payload["states"] = [
        {"name": item["name"], "mechanisms": item.get("mechanisms", [])}
        for item in payload["states"]
    ]
    payload["processes"] = [
        {"name": item["name"], "mechanisms": item.get("mechanisms", [])}
        for item in payload["processes"]
    ]
    return payload


def _proposed_functional_payload() -> dict[str, Any]:
    return {
        "candidate_id": "functional_compact_0",
        "change_summary": "Assign functions to a fixed graph.",
        "interaction_functions": [
            {"interaction_id": "latent_drive", "expression": "input_u"},
            {
                "interaction_id": "latent_relaxation",
                "expression": "latent_decay * z",
            },
            {"interaction_id": "flux_generation", "expression": "gain * z"},
            {"interaction_id": "storage_source", "expression": "flux"},
            {
                "interaction_id": "storage_sink",
                "expression": "storage_decay * x",
            },
        ],
        "parameters": [
            {"name": "latent_decay", "role": "rate"},
            {"name": "gain", "role": "scale"},
            {"name": "storage_decay", "role": "rate"},
        ],
        "latent_initials": [
            {"state": "z", "initial": {"fixed_value": 0.0}}
        ],
    }


def test_staged_expansion_builds_valid_executable_and_identity_link() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    functional = FunctionalCandidate.model_validate(_functional_payload(topology))

    result = expand_staged_candidate(topology, functional, _context())

    assert result.topology_commitment_sha256 == topology_commitment_sha256(topology)
    assert result.candidate_identity == candidate_identity(result.candidate)
    assert result.candidate.parent_candidate_id is None
    assert {item.name: item.expression for item in result.candidate.processes} == {
        "flux": "(gain * z)"
    }
    assert {item.state: item.rhs for item in result.candidate.state_equations} == {
        "x": "(flux) - (storage_decay * x)",
        "z": "(input_u) - (latent_decay * z)",
    }


def test_compact_staged_proposals_are_enriched_from_runtime_context() -> None:
    proposed_topology = ProposedTopologyCandidate.model_validate(
        _proposed_topology_payload()
    )
    topology = enrich_topology_proposal(proposed_topology, _context())
    proposed_functional = ProposedFunctionalCandidate.model_validate(
        _proposed_functional_payload()
    )
    functional = enrich_functional_proposal(proposed_functional, topology)

    result = expand_staged_candidate(topology, functional, _context())

    assert topology.external_symbols == ("input_u",)
    assert {item.name: item.kind.value for item in topology.states} == {
        "x": "observed",
        "z": "latent",
    }
    assert {item.state: item.expression for item in functional.initial_conditions} == {
        "x": "target",
        "z": None,
    }
    assert all(item.scope.value == "global" for item in functional.parameters)
    assert all(item.bounds is None for item in functional.parameters)
    assert result.candidate_identity == candidate_identity(result.candidate)


def test_observability_uses_auxiliary_measurements_not_target_status() -> None:
    payload = _proposed_topology_payload()
    payload["states"].append({"name": "measured_driver"})
    payload["state_measurements"] = [
        {"state": "measured_driver", "channel": "driver_aux"}
    ]
    payload["interactions"].append(
        {
            "interaction_id": "measured_driver_dynamics",
            "target": "measured_driver",
            "target_kind": "state_derivative",
            "sources": ["input_u"],
        }
    )
    context = ValidationContext(
        targets=("target",),
        auxiliaries=("driver_aux",),
        external_inputs=("input_u",),
    )

    topology = enrich_topology_proposal(
        ProposedTopologyCandidate.model_validate(payload), context
    )

    assert {item.name: item.kind.value for item in topology.states} == {
        "measured_driver": "observed",
        "x": "observed",
        "z": "latent",
    }


def test_topology_normalization_applies_only_unambiguous_repairs() -> None:
    proposal = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "repairable_graph",
            "states": [{"name": "x"}, {"name": "driver_alias"}],
            "processes": [{"name": "predicted_y"}],
            "interactions": [
                {
                    "interaction_id": "drive",
                    "target": "x",
                    "target_kind": "algebraic_process",
                    "sources": ["driver_alias"],
                },
                {
                    "interaction_id": "predict",
                    "target": "predicted_y",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                },
            ],
            "state_measurements": [
                {"state": "driver_alias", "channel": "driver_aux"},
                {"state": "public_aux", "channel": "public_aux"},
            ],
            "target_mappings": [
                {"channel": "y", "source": "predicted_y"},
                {"channel": "y", "source": "predicted_y"},
            ],
        }
    )
    context = ValidationContext(
        targets=("y",),
        auxiliaries=("driver_aux", "public_aux"),
    )

    normalized, repair = normalize_topology_proposal(proposal, context)
    topology = enrich_topology_proposal(normalized, context)

    assert [item.name for item in normalized.states] == ["x"]
    assert normalized.state_measurements == ()
    assert len(normalized.target_mappings) == 1
    assert {
        item.interaction_id: item.target_kind.value
        for item in normalized.interactions
    } == {"drive": "state_derivative", "predict": "algebraic_process"}
    assert normalized.interactions[0].sources == ("driver_aux",)
    assert topology.external_symbols == ("driver_aux",)
    assert repair == {
        "schema_version": "staged-topology-repair-1",
        "exact_duplicate_target_mappings_removed": ["y"],
        "interaction_target_kinds_corrected": [
            "drive:algebraic_process->state_derivative",
            "predict:state_derivative->algebraic_process",
        ],
        "forcing_alias_states_collapsed": ["driver_alias->driver_aux"],
        "redundant_state_measurements_removed": [
            "driver_alias->driver_aux",
            "public_aux->public_aux",
        ],
    }


def test_topology_normalization_rejects_conflicting_target_mappings() -> None:
    proposal = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "ambiguous_graph",
            "states": [{"name": "x"}, {"name": "z"}],
            "interactions": [
                {
                    "interaction_id": "x_dynamics",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x"],
                },
                {
                    "interaction_id": "z_dynamics",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["z"],
                },
            ],
            "target_mappings": [
                {"channel": "y", "source": "x"},
                {"channel": "y", "source": "z"},
            ],
        }
    )

    with pytest.raises(ValueError, match="conflicting duplicate target channel"):
        normalize_topology_proposal(
            proposal,
            ValidationContext(targets=("y",)),
        )


def test_parameterized_target_process_does_not_reveal_internal_state() -> None:
    proposal = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "nonidentity_measurement",
            "states": [{"name": "x"}],
            "processes": [{"name": "predicted_y"}],
            "interactions": [
                {
                    "interaction_id": "x_dynamics",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["input_u"],
                },
                {
                    "interaction_id": "measurement_function",
                    "target": "predicted_y",
                    "target_kind": "algebraic_process",
                    "sources": ["x"],
                },
            ],
            "target_mappings": [{"channel": "y", "source": "predicted_y"}],
        }
    )
    context = ValidationContext(targets=("y",), external_inputs=("input_u",))
    topology = enrich_topology_proposal(proposal, context)
    functional_proposal = ProposedFunctionalCandidate.model_validate(
        {
            "candidate_id": "nonidentity_functions",
            "interaction_functions": [
                {"interaction_id": "x_dynamics", "expression": "input_u"},
                {
                    "interaction_id": "measurement_function",
                    "expression": "theta * x",
                },
            ],
            "parameters": [{"name": "theta", "role": "scale"}],
            "latent_initials": [
                {"state": "x", "initial": {"fixed_value": 0.0}}
            ],
        }
    )
    functional = enrich_functional_proposal(functional_proposal, topology)

    result = expand_staged_candidate(topology, functional, context)

    assert topology.states[0].kind.value == "latent"
    assert result.candidate.observation_mappings[0].expression == "predicted_y"


def test_target_state_can_depend_on_a_latent_state() -> None:
    proposal = ProposedTopologyCandidate.model_validate(
        {
            "candidate_id": "latent_drives_target",
            "states": [{"name": "x"}, {"name": "z"}],
            "interactions": [
                {
                    "interaction_id": "x_memory",
                    "target": "x",
                    "target_kind": "state_derivative",
                    "sources": ["x", "input_u"],
                },
                {
                    "interaction_id": "z_response",
                    "target": "z",
                    "target_kind": "state_derivative",
                    "sources": ["x", "input_u"],
                },
            ],
            "target_mappings": [{"channel": "z_data", "source": "z"}],
        }
    )

    topology = enrich_topology_proposal(
        proposal,
        ValidationContext(
            targets=("z_data",), external_inputs=("input_u",)
        ),
    )

    assert {item.name: item.kind.value for item in topology.states} == {
        "x": "latent",
        "z": "observed",
    }


def test_compact_topology_cannot_invent_external_channels_or_targets() -> None:
    unknown_source = _proposed_topology_payload()
    unknown_source["interactions"][0]["sources"] = ["private_signal"]
    proposal = ProposedTopologyCandidate.model_validate(unknown_source)
    with pytest.raises(ValueError, match="unavailable external symbols"):
        enrich_topology_proposal(proposal, _context())

    wrong_target = _proposed_topology_payload()
    wrong_target["target_mappings"][0]["channel"] = "other_target"
    proposal = ProposedTopologyCandidate.model_validate(wrong_target)
    with pytest.raises(ValueError, match="mappings differ from public targets"):
        enrich_topology_proposal(proposal, _context())


def test_compact_functional_requires_exact_latent_initial_set() -> None:
    topology = enrich_topology_proposal(
        ProposedTopologyCandidate.model_validate(_proposed_topology_payload()),
        _context(),
    )
    payload = _proposed_functional_payload()
    payload["latent_initials"] = []
    proposal = ProposedFunctionalCandidate.model_validate(payload)

    with pytest.raises(ValueError, match=r"missing=\['z'\]"):
        enrich_functional_proposal(proposal, topology)


def test_topology_commitment_is_stable_across_json_round_trip() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    restored = TopologyCandidate.model_validate_json(topology.model_dump_json())

    assert topology_commitment_sha256(restored) == topology_commitment_sha256(
        topology
    )


def test_functional_candidate_must_reference_exact_topology() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    payload = _functional_payload(topology)
    payload["topology_commitment_sha256"] = "0" * 64
    functional = FunctionalCandidate.model_validate(payload)

    with pytest.raises(ValueError, match="different topology commitment"):
        expand_staged_candidate(topology, functional, _context())


def test_function_binding_cannot_change_declared_sources() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    payload = _functional_payload(topology)
    payload["interaction_functions"][0]["expression"] = "input_u + z"
    functional = FunctionalCandidate.model_validate(payload)

    with pytest.raises(ValueError, match="changes topology"):
        expand_staged_candidate(topology, functional, _context())


def test_function_bindings_must_cover_topology_exactly() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    payload = _functional_payload(topology)
    payload["interaction_functions"].pop()
    functional = FunctionalCandidate.model_validate(payload)

    with pytest.raises(ValueError, match="bindings differ from topology"):
        expand_staged_candidate(topology, functional, _context())


def test_topology_owned_sign_rejects_signed_scalar_coefficient_role() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    payload = _functional_payload(topology)
    for parameter in payload["parameters"]:
        if parameter["name"] == "gain":
            parameter["role"] = "coefficient"
            parameter.pop("domain", None)
    functional = FunctionalCandidate.model_validate(payload)

    with pytest.raises(ValueError, match="topology owns the outer polarity"):
        expand_staged_candidate(topology, functional, _context())


def test_algebraic_cycle_is_rejected_before_function_assignment() -> None:
    payload = _topology_payload()
    payload["processes"].append(
        {"name": "feedback", "description": "Cyclic helper."}
    )
    for interaction in payload["interactions"]:
        if interaction["interaction_id"] == "flux_generation":
            interaction["sources"] = ["feedback"]
    payload["interactions"].append(
        {
            "interaction_id": "feedback_generation",
            "target": "feedback",
            "target_kind": "algebraic_process",
            "sources": ["flux"],
        }
    )

    with pytest.raises(ValidationError, match="cyclic algebraic process"):
        TopologyCandidate.model_validate(payload)


def test_dynamic_feedback_loop_is_allowed() -> None:
    payload = _topology_payload()
    for interaction in payload["interactions"]:
        if interaction["interaction_id"] == "latent_drive":
            interaction["sources"] = ["x"]
    payload["external_symbols"] = []

    topology = TopologyCandidate.model_validate(payload)

    assert topology.states[0].name == "x"


def test_topology_rejects_undefined_source_and_uncovered_node() -> None:
    undefined = _topology_payload()
    undefined["interactions"][0]["sources"] = ["typo"]
    with pytest.raises(ValidationError, match="undefined sources"):
        TopologyCandidate.model_validate(undefined)

    uncovered = copy.deepcopy(_topology_payload())
    uncovered["interactions"] = [
        item
        for item in uncovered["interactions"]
        if item["target"] != "flux"
    ]
    with pytest.raises(ValidationError, match="without defining interactions"):
        TopologyCandidate.model_validate(uncovered)


def test_topology_external_symbols_must_be_available_at_runtime() -> None:
    topology = TopologyCandidate.model_validate(_topology_payload())
    functional = FunctionalCandidate.model_validate(_functional_payload(topology))
    context = ValidationContext(targets=("target",))

    with pytest.raises(ValueError, match="unavailable external symbols"):
        expand_staged_candidate(topology, functional, context)
