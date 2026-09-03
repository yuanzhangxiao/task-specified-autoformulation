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
    InteractionPolarity,
    InteractionTargetKind,
    ObservationMapping,
    ProcessSpec,
    StateEquation,
    TopologyCandidate,
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
        observation_mappings=tuple(
            ObservationMapping(
                channel=mapping.channel,
                expression=mapping.source,
                unit=mapping.unit,
            )
            for mapping in topology.observation_mappings
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
