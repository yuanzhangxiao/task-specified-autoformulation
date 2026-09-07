"""Atomic function assignments using scientific names and the restricted compiler."""

from __future__ import annotations

import ast
from collections.abc import Mapping

from autoformalism.construction import (
    apply_functional_actions,
    assess_functional_compatibility,
)
from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.schemas.construction import (
    ConstructionIntent,
    FunctionalDraft,
    ProposedFunctionalActionTransaction,
    SetInteractionFunctionAction,
    SetLatentInitialAction,
)
from autoformalism.schemas.proposal import ProposedInitialValue, ProposedParameter
from autoformalism.schemas.staged import TopologyCandidate
from autoformalism.schemas.staged_functions import (
    InteractionFunctionReply,
    LatentInitialReply,
)


def rename_expression(expression: str, aliases: Mapping[str, str]) -> str:
    """Rename parsed scalar identifiers without executing provider text."""
    parsed = RestrictedParser().parse(expression, location="function")
    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Name) and node.id in aliases:
            node.id = aliases[node.id]
    return ast.unparse(parsed.tree)


def initial_symbols(
    context: ValidationContext, aliases: Mapping[str, str]
) -> tuple[str, ...]:
    """Expose causal supplied data, excluding modeled auxiliaries."""
    return tuple(
        sorted(
            (set(context.auxiliaries) - set(aliases))
            | set(context.external_inputs)
            | set(context.fixed_covariates)
            | {context.time_symbol}
        )
    )


def apply_function_reply(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    selected_id: str,
    reply: InteractionFunctionReply,
    context: ValidationContext,
    aliases: Mapping[str, str],
) -> FunctionalDraft:
    """Bind one reply and reject incompatible changes before accepting a new draft."""
    reserved = (
        {item.name for item in topology.states}
        | {item.name for item in topology.processes}
        | set(context.targets)
        | set(context.forcing_channels)
        | {context.time_symbol}
        | set(aliases)
        | set(APPROVED_FUNCTION_ARITY)
    )
    for parameter in reply.parameters:
        if parameter.name in reserved or parameter.name.startswith("af_internal_"):
            raise ValueError(
                f"parameter collides with a variable or reserved name: {parameter.name}"
            )
    selected = next(
        (item for item in topology.interactions if item.interaction_id == selected_id),
        None,
    )
    if selected is None:
        raise ValueError("unknown selected interaction")
    inverse = {value: key for key, value in aliases.items()}
    expected = {inverse.get(name, name) for name in selected.sources}
    parsed = RestrictedParser().parse(reply.expression, location="function")
    actual = set(parsed.symbols) - {item.name for item in reply.parameters}
    if actual != expected:
        raise ValueError(
            f"source mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    action = SetInteractionFunctionAction(
        interaction_id=selected_id,
        expression=rename_expression(reply.expression, aliases),
        parameters=tuple(
            ProposedParameter(name=item.name, role=item.role)
            for item in reply.parameters
        ),
    )
    candidate = apply_functional_actions(
        draft,
        ConstructionIntent(
            objective="initial_construction", target_channels=context.targets
        ),
        ProposedFunctionalActionTransaction(actions=(action,)),
        topology,
        context,
    ).draft
    report = assess_functional_compatibility(topology, candidate)
    if report.status == "incompatible":
        raise ValueError(
            "; ".join(f"{item.code}: {item.message}" for item in report.diagnostics)
        )
    return candidate


def apply_initial_reply(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    selected_state: str,
    reply: LatentInitialReply,
    context: ValidationContext,
    aliases: Mapping[str, str],
) -> FunctionalDraft:
    """Check causal symbols before binding the selected latent initializer."""
    initial = ProposedInitialValue.model_validate(reply.initial.model_dump())
    if initial.expression is not None:
        parsed = RestrictedParser().parse(initial.expression, location="initial")
        unknown = set(parsed.symbols) - set(initial_symbols(context, aliases))
        if unknown:
            raise ValueError(f"unavailable initialization symbols: {sorted(unknown)}")
    return apply_functional_actions(
        draft,
        ConstructionIntent(
            objective="initial_construction", target_channels=context.targets
        ),
        ProposedFunctionalActionTransaction(
            actions=(SetLatentInitialAction(state=selected_state, initial=initial),)
        ),
        topology,
        context,
    ).draft
