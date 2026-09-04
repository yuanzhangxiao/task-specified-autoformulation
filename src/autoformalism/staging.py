"""Deterministic expansion of staged candidates into executable models."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import ConfigDict

from autoformalism.expressions import (
    CandidateValidator,
    RestrictedParser,
    ValidationContext,
)
from autoformalism.schemas import (
    CandidateModel,
    FunctionalCandidate,
    InitialConditionSpec,
    InteractionPolarity,
    InteractionTargetKind,
    ObservationMapping,
    ParameterRole,
    ParameterScope,
    ParameterSpec,
    ProcessSpec,
    ProposedFunctionalCandidate,
    ProposedTopologyCandidate,
    StateEquation,
    StateKind,
    StateSpec,
    TopologyCandidate,
    TopologyProcessSpec,
)
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged import Sha256Digest
from autoformalism.search.identity import CandidateIdentity, candidate_identity


class StagedCandidateExpansion(StrictSchema):
    """Executable result plus commitments linking all representation levels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["staged-candidate-expansion-1"] = (
        "staged-candidate-expansion-1"
    )
    topology_candidate_id: Identifier
    functional_candidate_id: Identifier
    topology_commitment_sha256: Sha256Digest
    candidate_identity: CandidateIdentity
    candidate: CandidateModel


def topology_commitment_sha256(topology: TopologyCandidate) -> str:
    """Commit to the exact validated topology artifact using canonical JSON."""
    payload = json.dumps(
        topology.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_topology_proposal(
    proposal: ProposedTopologyCandidate,
    context: ValidationContext,
) -> tuple[ProposedTopologyCandidate, dict[str, object]]:
    """Apply only unambiguous representation repairs before enrichment.

    This normalization never chooses between competing scientific target
    mappings, invents dynamics, or makes an unavailable channel available. It
    removes exact duplicates, derives interaction target-kind tags from the
    declared node namespace, and collapses dynamics-free aliases of supplied
    forcing channels back to those channels.
    """
    states = {item.name: item for item in proposal.states}
    processes = {item.name: item for item in proposal.processes}

    target_mappings = []
    target_by_channel = {}
    removed_target_mappings: list[str] = []
    for mapping in proposal.target_mappings:
        previous = target_by_channel.get(mapping.channel)
        if previous is None:
            target_by_channel[mapping.channel] = mapping
            target_mappings.append(mapping)
        elif previous == mapping:
            removed_target_mappings.append(mapping.channel)
        else:
            raise ValueError(
                "conflicting duplicate target channel: "
                f"{mapping.channel}; sources={previous.source},{mapping.source}"
            )

    corrected_kinds: list[str] = []
    interactions = []
    for interaction in proposal.interactions:
        expected = None
        if interaction.target in states:
            expected = InteractionTargetKind.STATE_DERIVATIVE
        elif interaction.target in processes:
            expected = InteractionTargetKind.ALGEBRAIC_PROCESS
        if expected is not None and interaction.target_kind is not expected:
            corrected_kinds.append(
                f"{interaction.interaction_id}:"
                f"{interaction.target_kind.value}->{expected.value}"
            )
            interaction = interaction.model_copy(update={"target_kind": expected})
        interactions.append(interaction)

    derivative_targets = {
        item.target
        for item in interactions
        if item.target_kind is InteractionTargetKind.STATE_DERIVATIVE
    }
    target_sources = {item.source for item in target_mappings}
    measurement_by_state = {
        item.state: item for item in proposal.state_measurements
    }
    forcing_channels = set(context.forcing_channels)
    forcing_aliases: dict[str, str] = {}
    for state in proposal.states:
        if state.name in derivative_targets or state.name in target_sources:
            continue
        measurement = measurement_by_state.get(state.name)
        if measurement is not None and measurement.channel in forcing_channels:
            forcing_aliases[state.name] = measurement.channel
        elif state.name in forcing_channels:
            forcing_aliases[state.name] = state.name

    normalized_interactions = []
    for interaction in interactions:
        sources = tuple(
            dict.fromkeys(
                forcing_aliases.get(item, item) for item in interaction.sources
            )
        )
        if sources != interaction.sources:
            interaction = interaction.model_copy(update={"sources": sources})
        normalized_interactions.append(interaction)

    retained_states = tuple(
        item for item in proposal.states if item.name not in forcing_aliases
    )
    retained_state_names = {item.name for item in retained_states}
    removed_measurements: list[str] = []
    retained_measurements = []
    for measurement in proposal.state_measurements:
        if measurement.state in forcing_aliases:
            removed_measurements.append(
                f"{measurement.state}->{measurement.channel}"
            )
            continue
        if (
            measurement.state not in retained_state_names
            and measurement.state == measurement.channel
            and measurement.channel in set(context.auxiliaries)
        ):
            removed_measurements.append(
                f"{measurement.state}->{measurement.channel}"
            )
            continue
        retained_measurements.append(measurement)

    normalized = ProposedTopologyCandidate.model_validate(
        {
            **proposal.model_dump(mode="json"),
            "states": [item.model_dump(mode="json") for item in retained_states],
            "interactions": [
                item.model_dump(mode="json") for item in normalized_interactions
            ],
            "state_measurements": [
                item.model_dump(mode="json") for item in retained_measurements
            ],
            "target_mappings": [
                item.model_dump(mode="json") for item in target_mappings
            ],
        }
    )
    return normalized, {
        "schema_version": "staged-topology-repair-1",
        "exact_duplicate_target_mappings_removed": removed_target_mappings,
        "interaction_target_kinds_corrected": corrected_kinds,
        "forcing_alias_states_collapsed": [
            f"{state}->{channel}"
            for state, channel in sorted(forcing_aliases.items())
        ],
        "redundant_state_measurements_removed": removed_measurements,
    }


def enrich_topology_proposal(
    proposal: ProposedTopologyCandidate,
    context: ValidationContext,
) -> TopologyCandidate:
    """Add public/runtime-owned graph metadata to a compact topology proposal.

    The proposer names generated nodes, dependencies, direct auxiliary-state
    measurements, and target outputs. The runtime validates every channel
    against the public context and derives state observability from identity
    measurements. Target status is not itself the definition of observability.
    """
    state_names = {item.name for item in proposal.states}
    process_names = {item.name for item in proposal.processes}
    generated = state_names | process_names
    referenced = {
        source
        for interaction in proposal.interactions
        for source in interaction.sources
    }
    external_symbols = referenced - generated
    available_external = set(context.forcing_channels) | {context.time_symbol}
    unknown_external = external_symbols - available_external
    if unknown_external:
        raise ValueError(
            "topology references unavailable external symbols: "
            f"{sorted(unknown_external)}"
        )

    channels = {item.channel for item in proposal.target_mappings}
    missing_targets = set(context.targets) - channels
    extra_targets = channels - set(context.targets)
    if missing_targets or extra_targets:
        raise ValueError(
            "topology target mappings differ from public targets: "
            f"missing={sorted(missing_targets)}, extra={sorted(extra_targets)}"
        )

    measured_channels = {item.channel for item in proposal.state_measurements}
    unavailable_measurements = measured_channels - set(context.auxiliaries)
    if unavailable_measurements:
        raise ValueError(
            "state measurements must reference supplied auxiliary channels: "
            f"{sorted(unavailable_measurements)}"
        )
    unknown_measured_states = {
        item.state for item in proposal.state_measurements
    } - state_names
    if unknown_measured_states:
        raise ValueError(
            "state measurements reference undeclared states: "
            f"{sorted(unknown_measured_states)}"
        )

    direct_state_channels: dict[str, list[str]] = {}
    for measurement in proposal.state_measurements:
        direct_state_channels.setdefault(measurement.state, []).append(
            measurement.channel
        )
    for mapping in proposal.target_mappings:
        if mapping.source in state_names:
            direct_state_channels.setdefault(mapping.source, []).append(
                mapping.channel
            )
    ambiguous_states = {
        state: channels
        for state, channels in direct_state_channels.items()
        if len(channels) != 1
    }
    if ambiguous_states:
        raise ValueError(
            "directly observed state maps to multiple target channels: "
            f"{ambiguous_states}"
        )

    interaction_mechanisms: dict[str, set[str]] = {
        name: set() for name in generated
    }
    for interaction in proposal.interactions:
        interaction_mechanisms.setdefault(interaction.target, set()).update(
            interaction.mechanisms
        )

    topology = TopologyCandidate(
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.parent_candidate_id,
        change_summary=proposal.change_summary,
        states=tuple(
            StateSpec(
                name=item.name,
                kind=(
                    StateKind.OBSERVED
                    if item.name in direct_state_channels
                    else StateKind.LATENT
                ),
                mechanisms=tuple(
                    sorted(
                        set(item.mechanisms)
                        | interaction_mechanisms.get(item.name, set())
                    )
                ),
            )
            for item in proposal.states
        ),
        processes=tuple(
            TopologyProcessSpec(
                name=item.name,
                mechanisms=tuple(
                    sorted(
                        set(item.mechanisms)
                        | interaction_mechanisms.get(item.name, set())
                    )
                ),
            )
            for item in proposal.processes
        ),
        external_symbols=tuple(sorted(external_symbols)),
        interactions=proposal.interactions,
        state_measurements=proposal.state_measurements,
        target_mappings=proposal.target_mappings,
    )
    if context.forbid_latent_states and any(
        item.kind is StateKind.LATENT for item in topology.states
    ):
        raise ValueError("topology contains latent states but they are forbidden")
    return topology


def enrich_functional_proposal(
    proposal: ProposedFunctionalCandidate,
    topology: TopologyCandidate,
) -> FunctionalCandidate:
    """Bind compact interaction functions to an immutable topology artifact."""
    latent_states = {
        item.name for item in topology.states if item.kind is StateKind.LATENT
    }
    supplied_latent_states = {item.state for item in proposal.latent_initials}
    missing = latent_states - supplied_latent_states
    extra = supplied_latent_states - latent_states
    if missing or extra:
        raise ValueError(
            "functional latent initializers differ from topology: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    observed_state_names = {
        state.name
        for state in topology.states
        if state.kind is StateKind.OBSERVED
    }
    direct_observed_channels = {
        item.state: item.channel for item in topology.state_measurements
    }
    direct_observed_channels.update(
        {
            mapping.source: mapping.channel
            for mapping in topology.target_mappings
            if mapping.source in observed_state_names
        }
    )
    observed_initials = tuple(
        InitialConditionSpec(
            state=state.name,
            scope=ParameterScope.GLOBAL,
            expression=direct_observed_channels[state.name],
        )
        for state in topology.states
        if state.kind is StateKind.OBSERVED
    )
    latent_initials = tuple(
        InitialConditionSpec(
            state=item.state,
            scope=ParameterScope.GLOBAL,
            fixed_value=item.initial.fixed_value,
            expression=item.initial.expression,
        )
        for item in proposal.latent_initials
    )
    return FunctionalCandidate(
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.parent_candidate_id,
        change_summary=proposal.change_summary,
        topology_commitment_sha256=topology_commitment_sha256(topology),
        interaction_functions=proposal.interaction_functions,
        parameters=tuple(
            ParameterSpec(
                name=item.name,
                scope=ParameterScope.GLOBAL,
                role=item.role,
                domain=item.role.domain,
            )
            for item in proposal.parameters
        ),
        initial_conditions=(*observed_initials, *latent_initials),
    )


def expand_staged_candidate(
    topology: TopologyCandidate,
    functional: FunctionalCandidate,
    context: ValidationContext,
    *,
    parser: RestrictedParser | None = None,
) -> StagedCandidateExpansion:
    """Validate a functional assignment and build one executable candidate.

    Functional expressions may use only the sources declared by their topology
    interaction and parameters declared by the functional candidate. The final
    candidate is then passed through the ordinary deterministic validator, so
    staged construction cannot bypass the executable safety boundary.
    """
    commitment = topology_commitment_sha256(topology)
    if functional.topology_commitment_sha256 != commitment:
        raise ValueError(
            "functional candidate references a different topology commitment"
        )

    allowed_external = set(context.forcing_channels) | {context.time_symbol}
    unknown_external = set(topology.external_symbols) - allowed_external
    if unknown_external:
        raise ValueError(
            "topology declares unavailable external symbols: "
            f"{sorted(unknown_external)}"
        )

    interactions = {
        item.interaction_id: item for item in topology.interactions
    }
    functions = {
        item.interaction_id: item for item in functional.interaction_functions
    }
    missing = set(interactions) - set(functions)
    extra = set(functions) - set(interactions)
    if missing or extra:
        raise ValueError(
            "functional interaction bindings differ from topology: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    restricted_parser = parser or RestrictedParser()
    parameter_names = {item.name for item in functional.parameters}
    signed_coefficient_names = {
        item.name
        for item in functional.parameters
        if item.role is ParameterRole.COEFFICIENT
    }
    terms_by_target: dict[
        tuple[InteractionTargetKind, str],
        list[tuple[InteractionPolarity, str]],
    ] = {}
    for interaction in topology.interactions:
        binding = functions[interaction.interaction_id]
        parsed = restricted_parser.parse(
            binding.expression,
            location=f"interaction:{interaction.interaction_id}",
        )
        used_sources = set(parsed.symbols) - parameter_names
        expected_sources = set(interaction.sources)
        if used_sources != expected_sources:
            raise ValueError(
                f"interaction {interaction.interaction_id} changes topology: "
                f"missing_sources={sorted(expected_sources - used_sources)}, "
                f"extra_sources={sorted(used_sources - expected_sources)}"
            )
        ambiguous_signed_coefficients = sorted(
            parsed.symbols & signed_coefficient_names
        )
        if ambiguous_signed_coefficients:
            raise ValueError(
                f"interaction {interaction.interaction_id} uses signed scalar "
                "coefficient roles even though topology owns the outer polarity: "
                f"{ambiguous_signed_coefficients}; use nonnegative_coefficient, "
                "rate, scale, or another scientifically appropriate typed role"
            )
        terms_by_target.setdefault(
            (interaction.target_kind, interaction.target), []
        ).append((interaction.polarity, binding.expression))

    state_equations = tuple(
        StateEquation(
            state=state.name,
            rhs=_sum_terms(
                terms_by_target[
                    (InteractionTargetKind.STATE_DERIVATIVE, state.name)
                ]
            ),
        )
        for state in topology.states
    )
    processes = tuple(
        ProcessSpec(
            name=process.name,
            expression=_sum_terms(
                terms_by_target[
                    (InteractionTargetKind.ALGEBRAIC_PROCESS, process.name)
                ]
            ),
            unit=process.unit,
            description=process.description,
            mechanisms=process.mechanisms,
        )
        for process in topology.processes
    )
    candidate = CandidateModel(
        candidate_id=functional.candidate_id,
        parent_candidate_id=functional.parent_candidate_id,
        change_summary=functional.change_summary,
        states=topology.states,
        processes=processes,
        state_equations=state_equations,
        observation_mappings=(
            *(
                ObservationMapping(
                    channel=mapping.channel,
                    expression=mapping.source,
                    unit=mapping.unit,
                )
                for mapping in topology.target_mappings
            ),
            *(
                ObservationMapping(
                    channel=measurement.channel,
                    expression=measurement.state,
                    unit=measurement.unit,
                )
                for measurement in topology.state_measurements
            ),
        ),
        parameters=functional.parameters,
        initial_conditions=functional.initial_conditions,
        constraints=functional.constraints,
    )
    CandidateValidator(restricted_parser).validate(candidate, context)
    return StagedCandidateExpansion(
        topology_candidate_id=topology.candidate_id,
        functional_candidate_id=functional.candidate_id,
        topology_commitment_sha256=commitment,
        candidate_identity=candidate_identity(candidate),
        candidate=candidate,
    )


def _sum_terms(terms: list[tuple[InteractionPolarity, str]]) -> str:
    rendered: list[str] = []
    for index, (polarity, expression) in enumerate(terms):
        if index == 0:
            prefix = "-" if polarity is InteractionPolarity.SUBTRACTIVE else ""
        else:
            prefix = " - " if polarity is InteractionPolarity.SUBTRACTIVE else " + "
        rendered.append(f"{prefix}({expression})")
    return "".join(rendered)
