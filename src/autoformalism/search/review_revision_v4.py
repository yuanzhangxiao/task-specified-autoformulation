"""One declaration per new model parameter; inherited roles remain immutable."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.rebuttal.repair_transactions import signed_tree
from autoformalism.rebuttal.revision_decision import parameter_aliases, translate_names
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    ComponentRevision,
    RevisionContractError,
    RevisionParameter,
    _effective_revision_parameter_roles,
)
from autoformalism.schemas import CandidateModel, ParameterSpec
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.search import review_revision_v3 as previous


class EquationContent(StrictSchema):
    """Equations reference parameters but do not redeclare their meanings."""

    component: Identifier
    kind: Literal["preserve", "dynamic", "algebraic"] = "preserve"
    expression: str = Field(min_length=1, max_length=4096)


class ScientificRevision(previous.ScientificRevision):
    equations: tuple[EquationContent, ...] = Field(default=(), max_length=6)
    new_parameters: tuple[RevisionParameter, ...] = Field(default=(), max_length=32)


SYSTEM_PROMPT = (
    previous.SYSTEM_PROMPT.replace(
        "Each equation gives component, kind, a complete scalar RHS expression, "
        "and NEW\nparameter declarations.",
        "Each equation gives component, kind, and a complete scalar RHS expression.\n"
        "Declare genuinely new equation/output parameters ONCE in new_parameters at\n"
        "the top level. Equations have NO parameters field. Existing parameter names\n"
        "are references with immutable roles; do not redeclare them. Initial causal\n"
        "maps retain their own local parameter declarations inside causal_map.",
    )
    .replace(
        "and initializers. No action/scope/route",
        "initializers, and new_parameters. No action/scope/route",
    )
    .replace(
        "shape/coefficient/offset are real; "
        "positive_shape/scale/rate/time_constant positive.",
        "shape is real; positive_shape/scale/rate/time_constant are positive. "
        "A generic\ncoefficient or offset role is insufficient for an internal "
        "nonlinear parameter.\n"
        "A shared parameter has one domain across every use. Use distinct names if\n"
        "two coefficients are scientifically independent. Never resolve a conflict by\n"
        "changing an existing parameter's meaning.",
    )
)


def payload(bundle, packet, parameters, retry=None) -> dict:
    value = previous.payload(bundle, packet, parameters, retry)
    value["protocol"] = "scientific-content-revision-4"
    value["editable_objects"]["new_parameters"] = (
        "Declare once at patch top level; existing parameters inherit their roles"
    )
    return value


def _resolve(
    bundle: dict, reply: ScientificRevision
) -> tuple[tuple[ParameterSpec, ...], dict]:
    """Resolve every new role against all edited RHS and output occurrences."""
    parent = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    aliases = parameter_aliases(parent)
    inverse = {v: k for k, v in aliases.items()}
    existing = {p.name: p for p in parent.parameters}
    declarations, inherited, redundant = {}, [], []
    for parameter in reply.new_parameters:
        name = inverse.get(parameter.name, parameter.name)
        if name in existing:
            inherited.append(
                {
                    "parameter": name,
                    "requested_role": parameter.role.value if parameter.role else None,
                    "effective_role": existing[name].role.value,
                }
            )
            continue
        prior = declarations.get(name)
        if prior is not None:
            if prior.role and parameter.role and prior.role != parameter.role:
                raise RevisionContractError(
                    "NEW_PARAMETER_DECLARATION_CONFLICT",
                    "Declare one role for a shared new parameter, "
                    "or use distinct names.",
                    parameter=name,
                    roles=[prior.role.value, parameter.role.value],
                )
            redundant.append(name)
        if prior is None or parameter.role is not None:
            declarations[name] = parameter
    expressions = [(e.component, e.expression) for e in reply.equations]
    if reply.output_expression is not None:
        expressions.append(("public_output", reply.output_expression))
    symbols = {s.name for s in parent.states} | {p.name for p in parent.processes}
    symbols |= {e.component for e in reply.equations}
    context = ValidationContext.model_validate(bundle["context"])
    symbols |= (
        set(context.forcing_channels) | set(context.targets) | {context.time_symbol}
    )
    if symbols.intersection(declarations):
        raise ValueError("new parameter collides with a scientific symbol")
    occurrences = {name: [] for name in declarations}
    derivations = []
    for component, expression in expressions:
        expression = translate_names(expression, inverse)
        parsed = RestrictedParser().parse(expression, location=component)
        local = tuple(p for name, p in declarations.items() if name in parsed.symbols)
        # Match the transaction compiler: -a*m is a signed outer gain, not an
        # ambiguous internal parameter. Normalize only the analysis tree.
        role_tree = signed_tree(parsed.tree)
        roles, audit = _effective_revision_parameter_roles(
            ComponentRevision(
                component=component, expression=expression, parameters=local
            ),
            role_tree,
            scientific_symbols=symbols,
            parent_parameters=existing,
        )
        derivations.extend(audit)
        for name, role in roles.items():
            occurrences[name].append({"component": component, "role": role.value})
    specs = []
    positive = {"positive_shape", "scale", "rate", "time_constant"}
    for name, uses in occurrences.items():
        if not uses:
            raise RevisionContractError(
                "UNUSED_NEW_PARAMETER", "New declaration is unused.", parameter=name
            )
        roles = {u["role"] for u in uses}
        explicit = declarations[name].role
        chosen = next(iter(roles)) if len(roles) == 1 else None
        if (
            explicit
            and explicit.value in positive
            and roles <= {explicit.value, "nonnegative_coefficient"}
        ):
            chosen = explicit.value
        if explicit and explicit.value == "shape" and roles <= {"shape", "offset"}:
            chosen = "shape"
        if chosen is None:
            raise RevisionContractError(
                "NEW_PARAMETER_USE_CONFLICT",
                "A shared new parameter has incompatible roles across its uses; "
                "resolve its meaning or split independent coefficients.",
                parameter=name,
                uses=uses,
            )
        specs.append(ParameterSpec(name=name, scope="global", role=chosen))
    return tuple(specs), {
        "inherited_declarations_ignored": inherited,
        "redundant_new_declarations": redundant,
        "new_parameter_uses": occurrences,
        "resolved_parameters": [p.model_dump(mode="json") for p in specs],
        "role_derivations": derivations,
    }


def apply_edits(bundle: dict, packet: dict, raw: dict) -> dict:
    reply = ScientificRevision.model_validate(raw)
    specs, audit = _resolve(bundle, reply)
    result = previous.apply_edits(
        bundle,
        packet,
        reply.model_dump(mode="json", exclude={"new_parameters"}),
        parameter_specs=specs,
    )
    result["provenance"].update(
        protocol="scientific-content-revision-4",
        parameter_declaration_audit=audit,
    )
    return result


def feedback(bundle, packet, parameters, raw, error) -> dict:
    result = previous.feedback(bundle, packet, parameters, raw, error)
    result["parameter_contract"] = {
        "existing": "Reference the displayed name; role and domain are inherited.",
        "new": "Declare once in new_parameters; equations only reference the name.",
        "internal_roles": ["shape", "positive_shape", "rate", "scale", "time_constant"],
        "initial_maps": "Local map parameters remain inside causal_map.",
    }
    return result


def migrate_saved(raw: dict) -> dict:
    """Move v3 declarations without choosing between conflicting meanings."""
    reply = previous.ScientificRevision.model_validate(raw)
    value = reply.model_dump(mode="json")
    declarations = []
    for equation in value["equations"]:
        declarations.extend(equation.pop("parameters"))
    value["new_parameters"] = declarations
    return value
