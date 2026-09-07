"""Atomic function assignments using scientific names and the restricted compiler."""

from __future__ import annotations

import ast
import hashlib
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
    EquationFunctionBatchReply,
    FunctionParameter,
    InteractionFunctionObligation,
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
    obligation: InteractionFunctionObligation | None = None,
) -> FunctionalDraft:
    """Bind one reply and reject incompatible changes before accepting a new draft."""
    candidate, _ = bind_function_reply(
        topology,
        draft,
        selected_id,
        reply,
        context,
        aliases,
        obligation,
    )
    return candidate


def bind_function_reply(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    selected_id: str,
    reply: InteractionFunctionReply,
    context: ValidationContext,
    aliases: Mapping[str, str],
    obligation: InteractionFunctionObligation | None = None,
) -> tuple[FunctionalDraft, InteractionFunctionReply]:
    """Validate, normalize and bind one reply while exposing its stored identity."""
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
    active_obligation = obligation or InteractionFunctionObligation()
    if (
        active_obligation.requires_nonlinear_source_dependence
        and not has_nonlinear_source_dependence(parsed.tree, expected)
    ):
        raise ValueError(
            "NONLINEAR_SOURCE_DEPENDENCE_REQUIRED: selected interaction's "
            "scientific role explicitly requires nonlinear dependence on its source"
        )
    normalized = _normalize_parameter_identities(
        selected_id,
        reply,
        policy=active_obligation.parameter_identity_policy,
    )
    action = SetInteractionFunctionAction(
        interaction_id=selected_id,
        expression=rename_expression(normalized.expression, aliases),
        parameters=tuple(
            ProposedParameter(name=item.name, role=item.role)
            for item in normalized.parameters
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
    return candidate, normalized


def apply_equation_function_reply(
    topology: TopologyCandidate,
    draft: FunctionalDraft,
    selected_ids: tuple[str, ...],
    reply: EquationFunctionBatchReply,
    context: ValidationContext,
    aliases: Mapping[str, str],
) -> FunctionalDraft:
    """Validate and bind one complete same-LHS function batch atomically."""
    if len(reply.functions) != len(selected_ids):
        raise ValueError(
            "equation function count mismatch: "
            f"expected={len(selected_ids)}, actual={len(reply.functions)}"
        )
    candidate = draft
    for selected_id, function in zip(selected_ids, reply.functions, strict=True):
        candidate = apply_function_reply(
            topology,
            candidate,
            selected_id,
            function,
            context,
            aliases,
        )
    return candidate


def derive_interaction_function_obligation(
    scientific_role: str,
    *,
    parameter_identity_policy: str = "preserve",
) -> InteractionFunctionObligation:
    """Derive narrow syntax obligations from the frozen interaction role text."""
    lowered = scientific_role.lower()
    markers = tuple(
        marker
        for marker in ("nonlinear", "saturat", "sigmoid", "threshold")
        if marker in lowered
    )
    return InteractionFunctionObligation(
        requires_nonlinear_source_dependence=bool(markers),
        parameter_identity_policy=parameter_identity_policy,
        provenance=tuple(f"scientific_role_contains:{marker}" for marker in markers),
    )


def has_nonlinear_source_dependence(tree: ast.AST, sources: set[str]) -> bool:
    """Detect explicit nonlinear dependence on at least one scientific source."""

    def source_names(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in sources
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and source_names(node):
            return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and source_names(node.left)
            and (
                not isinstance(node.right, ast.Constant) or node.right.value != 1
            )
        ):
            return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and source_names(node.right)
        ):
            return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and source_names(node.left)
            and source_names(node.right)
        ):
            return True
    return False


def _normalize_parameter_identities(
    selected_id: str,
    reply: InteractionFunctionReply,
    *,
    policy: str,
) -> InteractionFunctionReply:
    """Namespace local parameter mnemonics without changing their expression use."""
    if policy == "preserve":
        return reply
    if policy != "interaction_local":
        raise ValueError(f"unsupported parameter identity policy: {policy}")
    names = {
        parameter.name: _interaction_local_parameter_name(
            selected_id,
            parameter.name,
        )
        for parameter in reply.parameters
    }
    return InteractionFunctionReply(
        expression=rename_expression(reply.expression, names),
        parameters=tuple(
            FunctionParameter(name=names[item.name], role=item.role)
            for item in reply.parameters
        ),
    )


def _interaction_local_parameter_name(interaction_id: str, name: str) -> str:
    """Build a deterministic valid identifier bounded by the schema limit."""
    suffix = hashlib.sha256(f"{interaction_id}:{name}".encode()).hexdigest()[:10]
    stem = f"{name}_{interaction_id}"
    if len(stem) <= 117:
        return f"{stem}_{suffix}"
    return f"{name[:106]}_{suffix}"


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
