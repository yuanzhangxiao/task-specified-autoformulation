"""Runtime-owned observability inferred from public channel mappings."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from autoformalism.expressions.diagnostics import RuntimeExpressionError
from autoformalism.expressions.validation import ValidatedCandidate


@dataclass(frozen=True)
class EffectiveObservability:
    """Effective observed/latent state partition used by numerical methods.

    The proposer-declared state kind is retained in the candidate artifact, but
    an identity mapping from a public target or auxiliary channel is
    authoritative evidence that the corresponding state is observed at fit
    time. Nonidentity mappings never reveal their internal states.
    """

    state_channels: Mapping[str, str]
    observed_state_names: frozenset[str]
    latent_state_names: frozenset[str]
    runtime_inferred_observed_state_names: frozenset[str]


def infer_effective_observability(
    validated: ValidatedCandidate,
) -> EffectiveObservability:
    """Infer observed states from identity mappings to public data channels."""
    state_names = tuple(item.name for item in validated.candidate.states)
    state_name_set = frozenset(state_names)
    available = frozenset(validated.context.targets + validated.context.auxiliaries)
    auxiliary_channels = frozenset(validated.context.auxiliaries)
    channels: dict[str, str] = {
        state_name: state_name
        for state_name in state_names
        if state_name in auxiliary_channels
    }
    for channel, expression in validated.observation_expressions.items():
        body = expression.tree.body
        if (
            channel not in available
            or not isinstance(body, ast.Name)
            or body.id not in state_name_set
        ):
            continue
        existing = channels.get(body.id)
        if existing is not None and existing != channel:
            raise RuntimeExpressionError(
                f"state {body.id} has conflicting observed channels: "
                f"{existing}, {channel}"
            )
        channels[body.id] = channel

    observed = frozenset(channels)
    declared_observed = frozenset(
        item.name
        for item in validated.candidate.states
        if item.kind.value == "observed"
    )
    return EffectiveObservability(
        state_channels=MappingProxyType(channels),
        observed_state_names=observed,
        latent_state_names=state_name_set - observed,
        runtime_inferred_observed_state_names=observed - declared_observed,
    )
