"""Exact, selected-interaction parameter inheritance for saved RHS revisions."""

from __future__ import annotations

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas.staged_functions import (
    FunctionParameter,
    InteractionFunctionReply,
)
from autoformalism.staged_functions import (
    _normalize_parameter_identities,
    rename_expression,
)

POLICY = "selected-interaction-parameter-inheritance-1"


def inherit_parameters(
    slot: dict, reply: InteractionFunctionReply
) -> tuple[InteractionFunctionReply, list[dict]]:
    """Recover only exact local/canonical identities from the selected parent slot.

    Canonical names are transported back to their recorded local names before
    binding. No fuzzy matching, values, foreign parameters or invented roles.
    """
    old = InteractionFunctionReply.model_validate(slot["accepted_reply"])
    canonical = {p["name"]: p for p in slot["canonical_function"]["parameters"]}
    normalized = _normalize_parameter_identities(
        slot["interaction_id"],
        old,
        policy=slot["selected_term"]["functional_obligation"][
            "parameter_identity_policy"
        ],
    )
    if {p.name for p in normalized.parameters} != set(canonical):
        raise ValueError("parent local/canonical parameter identities differ")
    matches: dict[str, tuple[str, str, str]] = {}
    for local, bound in zip(old.parameters, normalized.parameters, strict=True):
        item = (local.name, bound.name, canonical[bound.name]["role"])
        for name in (local.name, bound.name):
            if name in matches and matches[name] != item:
                raise ValueError("ambiguous parent parameter identity")
            matches[name] = item
    used = set(RestrictedParser().parse(reply.expression, location="revision").symbols)
    declared = {p.name: p for p in reply.parameters}
    aliases, records = {}, []
    parameters: dict[str, FunctionParameter] = {}
    for name in sorted(used | set(declared)):
        if name not in matches:
            if name in declared:
                parameters[name] = declared[name]
            continue
        local, bound, role = matches[name]
        if name in declared and declared[name].role.value != role:
            raise ValueError(f"reused parameter role conflicts with parent: {name}")
        aliases[name] = local
        parameters[local] = FunctionParameter(name=local, role=role)
        if name not in declared or name != local:
            records.append(
                {
                    "policy": POLICY,
                    "interaction_id": slot["interaction_id"],
                    "referenced_name": name,
                    "local_name": local,
                    "canonical_name": bound,
                    "role": role,
                    "declaration_inherited": name not in declared,
                    "numeric_value_fixed": False,
                }
            )
    order = [p.name for p in old.parameters if p.name in parameters]
    order.extend(p.name for p in reply.parameters if p.name not in matches)
    return InteractionFunctionReply(
        expression=rename_expression(reply.expression, aliases),
        parameters=tuple(parameters[name] for name in order),
    ), records
