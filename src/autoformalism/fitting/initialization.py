"""Versioned scientific initialization choices lowered to shared fitted parameters.

No observations are read during lowering. Boundary observations enter only when
the compiled initializer is evaluated at a trajectory's first sample.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.schemas import CandidateModel, ParameterRole
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema


class InitialMapParameter(StrictSchema):
    """A map-local scientific parameter; its numerical domain follows its role."""

    name: Identifier
    role: ParameterRole = ParameterRole.COEFFICIENT
    guess: FiniteFloat = 0.1


class InitialValueGuess(StrictSchema):
    """One shared unknown initial value, with a suggested optimizer start."""

    mode: Literal["value"] = "value"
    guess: FiniteFloat
    role: Literal["coefficient", "nonnegative_coefficient"] = "coefficient"


class InitialCausalMap(StrictSchema):
    """A restricted map of initial public observations and shared parameters."""

    mode: Literal["map"] = "map"
    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[InitialMapParameter, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def unique_parameters(self):
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate initializer parameter")
        return self


class KnownInitialValue(StrictSchema):
    """Explicit exception for an initial value established by task preparation."""

    mode: Literal["known"] = "known"
    value: FiniteFloat
    justification: str = Field(min_length=1, max_length=2000)


class LatentInitializationReply(StrictSchema):
    """One runtime-selected state; no proposer-owned state identifiers or scope."""

    initial: Annotated[
        InitialValueGuess | InitialCausalMap | KnownInitialValue,
        Field(discriminator="mode"),
    ]


class LatentInitializationPlan(StrictSchema):
    """Explicit, hashable opt-in; omitted latent rules retain their old meaning."""

    protocol: Literal["shared-latent-initialization-1"] = (
        "shared-latent-initialization-1"
    )
    scope: Literal["shared_training"] = "shared_training"
    rules: Mapping[Identifier, LatentInitializationReply] = Field(default_factory=dict)


INITIALIZATION_SYSTEM_PROMPT = """Choose initialization for the selected latent state.
Return exactly the supplied JSON schema. Normally choose mode=value with a finite
guess, or mode=map with a restricted scalar expression and its local parameters.
A value is an optimizer guess, not a prescribed physical value. Its fitted value
is shared across training trajectories and frozen for validation. A map is
evaluated only at the first sample, using the displayed public channel names,
fixed covariates and time. Map parameters are also shared, fitted on training and
frozen. A channel name denotes its initial observation, never its future path.
Use mode=known only when the public preparation establishes that latent initial
value, and give the scientific justification. Zero input or zero observed output
alone does not establish zero hidden states. Directly observed state initials
are bound by the runtime to their observations and are not part of this call.
Use only the displayed restricted expression grammar and allowed symbols. No
hidden reference states, trajectory identifiers, future observations, indexing,
arbitrary calls, or recursive references to other generated initial states.
Declare every map-local parameter exactly once with name, role and an optional
guess. Existing equation parameters are not implicitly shared with map-local
parameters. Do not emit numerical ranges, state names, fitted values or prose.
"""


def apply_initialization_plan(
    model: CompiledModel, plan: LatentInitializationPlan
) -> tuple[CompiledModel, dict[str, float], dict]:
    """Lower an explicit plan without changing any RHS or observation expression."""
    direct = model.direct_state_observation_channels
    latent = set(model.state_names) - set(direct)
    unknown = set(plan.rules) - latent
    if unknown:
        raise ValueError(
            f"initialization rules must name latent states: {sorted(unknown)}"
        )
    payload = model.validated.candidate.model_dump(mode="json")
    originals = {item["state"]: item for item in payload["initial_conditions"]}
    reserved = (
        set(model.state_names)
        | set(model.parameter_names)
        | {p.name for p in model.validated.candidate.processes}
        | set(model.validated.context.targets)
        | set(model.validated.context.forcing_channels)
        | {model.validated.context.time_symbol}
    )
    allowed = (
        set(model.validated.context.targets)
        | set(model.validated.context.forcing_channels)
        | {model.validated.context.time_symbol}
    )
    guesses, bindings = {}, {}
    for state in model.state_names:
        if state in direct:
            originals[state] = {
                "state": state,
                "scope": "global",
                "expression": direct[state],
            }
            continue
        if state not in plan.rules:
            continue
        rule = plan.rules[state].initial
        base = {"state": state, "scope": "global"}
        if isinstance(rule, KnownInitialValue):
            originals[state] = {**base, "fixed_value": rule.value}
            bindings[state] = {"mode": "known", "parameters": []}
            continue
        if isinstance(rule, InitialValueGuess):
            parameters = (
                InitialMapParameter(name="value", role=rule.role, guess=rule.guess),
            )
            expression = "value"
        else:
            parameters, expression = rule.parameters, rule.expression
        parsed = RestrictedParser().parse(expression, location=f"initial:{state}")
        local = {p.name for p in parameters}
        if (
            local & allowed
            or parsed.symbols - local - allowed
            or local - parsed.symbols
        ):
            raise ValueError("initializer has unavailable, ambiguous or unused symbols")
        rename = {}
        for parameter in parameters:
            name = f"init_{state}_{parameter.name}"
            if name in reserved:
                raise ValueError(f"initializer parameter name collision: {name}")
            reserved.add(name)
            rename[parameter.name] = name
            payload["parameters"].append(
                {"name": name, "scope": "global", "role": parameter.role.value}
            )
            guesses[name] = parameter.guess

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node, names=rename):
                return ast.copy_location(
                    ast.Name(id=names.get(node.id, node.id), ctx=ast.Load()), node
                )

        transformed = ast.unparse(Rename().visit(parsed.tree))
        originals[state] = {**base, "expression": transformed}
        bindings[state] = {"mode": rule.mode, "parameters": list(rename.values())}
    payload["initial_conditions"] = [originals[name] for name in model.state_names]
    context = model.validated.context.model_copy(update={"fitted_initialization": True})
    compiled = compile_candidate(CandidateModel.model_validate(payload), context)
    return (
        compiled,
        guesses,
        {
            "protocol": plan.protocol,
            "scope": plan.scope,
            "bindings": bindings,
            "observed_initial_channels": dict(direct),
            "validation_initials_fitted": False,
        },
    )
