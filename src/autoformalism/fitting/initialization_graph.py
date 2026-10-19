"""Explicit acyclic initial-state graphs for the opt-in prefix fitting protocol.

Lowering reads no data. Independent trajectory roots are represented by ordinary
compiled parameters for simulation, with their scientific scope retained in the
audit. Only the new runner may assign those parameters separately per trajectory.
Graphs containing no trajectory roots are directly replayable by the existing
shared-initialization backend.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import CompiledModel, compile_candidate
from autoformalism.expressions.parser import (
    APPROVED_FUNCTION_ARITY,
    ParsedExpression,
    RestrictedParser,
)
from autoformalism.fitting.initialization import InitialMapParameter, KnownInitialValue
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import FiniteFloat, Identifier, StrictSchema


class InitialGraphValue(StrictSchema):
    """One independent scalar root, shared or inferred per trajectory."""

    mode: Literal["value"] = "value"
    guess: FiniteFloat
    role: Literal["coefficient", "nonnegative_coefficient"] = "coefficient"
    scope: Literal["shared_training", "trajectory"] = "shared_training"


class InitialGraphMap(StrictSchema):
    """A dependent initial value using explicit aliases and shared coefficients."""

    mode: Literal["map"] = "map"
    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[InitialMapParameter, ...] = Field(default=(), max_length=32)
    state_initials: Mapping[Identifier, Identifier] = Field(
        default_factory=dict, max_length=64
    )
    public_initials: Mapping[Identifier, Identifier] = Field(
        default_factory=dict, max_length=64
    )

    @model_validator(mode="after")
    def unambiguous_symbols(self) -> InitialGraphMap:
        """Reject duplicate local names and aliases that shadow math functions."""
        names = (
            [parameter.name for parameter in self.parameters]
            + list(self.state_initials)
            + list(self.public_initials)
        )
        if len(names) != len(set(names)):
            raise ValueError("duplicate or ambiguous initializer alias/parameter")
        if set(names) & APPROVED_FUNCTION_ARITY.keys():
            raise ValueError("initializer aliases cannot shadow approved functions")
        return self


class InitialGraphRule(StrictSchema):
    """A latent state's explicit initial-value definition."""

    initial: Annotated[
        InitialGraphValue | InitialGraphMap | KnownInitialValue,
        Field(discriminator="mode"),
    ]


class InitializationGraph(StrictSchema):
    """Versioned complete latent initialization contract; observed rules are fixed."""

    protocol: Literal["latent-initialization-graph-1"] = "latent-initialization-graph-1"
    rules: Mapping[Identifier, InitialGraphRule] = Field(max_length=64)


def apply_initialization_graph(
    model: CompiledModel, graph: InitializationGraph
) -> tuple[CompiledModel, dict[str, float], dict[str, Any]]:
    """Validate and expand a complete graph without changing model dynamics.

    The returned guesses cover generated parameters only. ``shared_parameter_names``
    lists generated shared initial parameters; existing equation parameters retain
    their original scope. ``trajectory_root_parameters`` maps root states to the
    parameter names that the new runner must supply independently for each row.
    Numerical bounds continue to follow the existing parameter-role policy.
    """
    observed = dict(model.direct_state_observation_channels)
    latent = set(model.state_names) - observed.keys()
    if set(graph.rules) != latent:
        raise ValueError(
            "initialization graph must cover exactly latent states: "
            f"missing={sorted(latent - graph.rules.keys())}, "
            f"extra={sorted(graph.rules.keys() - latent)}"
        )
    context = model.validated.context
    public = (
        set(context.targets) | set(context.forcing_channels) | {context.time_symbol}
    )
    parsed = _validate_maps(model, graph, public)
    order = _topological_order(model.state_names, graph)
    payload = model.validated.candidate.model_dump(mode="json")
    reserved = (
        set(model.state_names)
        | set(model.parameter_names)
        | {process.name for process in model.validated.candidate.processes}
        | public
    )
    guesses: dict[str, float] = {}
    shared: list[str] = []
    roots: dict[str, str] = {}
    renamed: dict[str, dict[str, str]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    parameter_domains: dict[str, dict[str, Any]] = {}
    for state in model.state_names:
        if state in observed:
            continue
        rule = graph.rules[state].initial
        parameters = _parameters(rule)
        rename: dict[str, str] = {}
        scope = rule.scope if isinstance(rule, InitialGraphValue) else "shared_training"
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
            if scope == "trajectory":
                roots[state] = name
            else:
                shared.append(name)
            parameter_domains[name] = {
                "role": parameter.role.value,
                "domain": parameter.role.domain.value,
                "scope": scope,
                "bounds": None,
                "bound_policy": "existing_parameter_role_policy",
            }
        renamed[state] = rename
        bindings[state] = {
            "mode": rule.mode,
            "parameters": list(rename.values()),
            "state_initials": dict(rule.state_initials)
            if isinstance(rule, InitialGraphMap)
            else {},
            "public_initials": dict(rule.public_initials)
            if isinstance(rule, InitialGraphMap)
            else {},
        }
        if isinstance(rule, InitialGraphValue):
            bindings[state]["scope"] = rule.scope

    expanded: dict[str, ast.expr] = {
        state: ast.Name(id=channel, ctx=ast.Load())
        for state, channel in observed.items()
    }
    initial_conditions: dict[str, dict[str, Any]] = {
        state: {"state": state, "scope": "global", "expression": channel}
        for state, channel in observed.items()
    }
    for state in order:
        rule = graph.rules[state].initial
        initial: dict[str, Any] = {"state": state, "scope": "global"}
        if isinstance(rule, KnownInitialValue):
            expanded[state] = ast.Constant(value=rule.value)
            initial["fixed_value"] = rule.value
        else:
            replacements: dict[str, ast.expr] = {
                local: ast.Name(id=name, ctx=ast.Load())
                for local, name in renamed[state].items()
            }
            if isinstance(rule, InitialGraphValue):
                source = RestrictedParser().parse("value", location=f"initial:{state}")
            else:
                source = parsed[state]
                replacements.update(
                    (alias, expanded[target])
                    for alias, target in rule.state_initials.items()
                )
                replacements.update(
                    (alias, ast.Name(id=channel, ctx=ast.Load()))
                    for alias, channel in rule.public_initials.items()
                )
            result = _expand(source, replacements, state)
            expanded[state] = result.tree.body
            initial["expression"] = result.source
        initial_conditions[state] = initial
        bindings[state]["expanded_expression"] = ast.unparse(expanded[state])
    payload["initial_conditions"] = [
        initial_conditions[name] for name in model.state_names
    ]
    compiled = compile_candidate(
        CandidateModel.model_validate(payload),
        context.model_copy(update={"fitted_initialization": True}),
    ).with_nonbinding_parameter_bounds(model.nonbinding_parameter_bounds)
    return (
        compiled,
        guesses,
        {
            "protocol": graph.protocol,
            "bindings": bindings,
            "topological_order": list(order),
            "observed_initial_channels": observed,
            "shared_parameter_names": shared,
            "trajectory_parameter_names": list(roots.values()),
            "trajectory_root_parameters": roots,
            "parameter_domains": parameter_domains,
            "legacy_shared_compatible": not roots,
        },
    )


def _parameters(
    rule: InitialGraphValue | InitialGraphMap | KnownInitialValue,
) -> tuple[InitialMapParameter, ...]:
    """Return independent numerical coordinates; maps have no output coordinate."""
    if isinstance(rule, InitialGraphValue):
        return (InitialMapParameter(name="value", role=rule.role, guess=rule.guess),)
    return rule.parameters if isinstance(rule, InitialGraphMap) else ()


def _validate_maps(
    model: CompiledModel, graph: InitializationGraph, public: set[str]
) -> dict[str, ParsedExpression]:
    """Resolve declaration namespaces before expanding any expressions."""
    parsed = {}
    for state in model.state_names:
        if state not in graph.rules:
            continue
        rule = graph.rules[state].initial
        if not isinstance(rule, InitialGraphMap):
            continue
        unknown_states = set(rule.state_initials.values()) - set(model.state_names)
        unknown_public = set(rule.public_initials.values()) - public
        if unknown_states or unknown_public:
            raise ValueError(
                f"unavailable initializer references for {state}: "
                f"states={sorted(unknown_states)}, public={sorted(unknown_public)}"
            )
        source = RestrictedParser().parse(rule.expression, location=f"initial:{state}")
        declared = (
            {parameter.name for parameter in rule.parameters}
            | rule.state_initials.keys()
            | rule.public_initials.keys()
        )
        if source.symbols != declared:
            raise ValueError(
                f"initializer {state} has undeclared or unused symbols: "
                f"undeclared={sorted(source.symbols - declared)}, "
                f"unused={sorted(declared - source.symbols)}"
            )
        parsed[state] = source
    return parsed


def _topological_order(
    states: tuple[str, ...], graph: InitializationGraph
) -> tuple[str, ...]:
    """Order latent dependencies deterministically and reject every cycle."""
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state: str) -> None:
        if state in visiting:
            raise ValueError(f"cyclic initializer state dependency at {state}")
        if state in visited or state not in graph.rules:
            return
        visiting.add(state)
        rule = graph.rules[state].initial
        if isinstance(rule, InitialGraphMap):
            dependencies = set(rule.state_initials.values())
            for dependency in states:
                if dependency in dependencies:
                    visit(dependency)
        visiting.remove(state)
        visited.add(state)
        order.append(state)

    for state in states:
        visit(state)
    return tuple(order)


def _expand(
    source: ParsedExpression, replacements: Mapping[str, ast.expr], state: str
) -> ParsedExpression:
    """Substitute validated ASTs while bounding expansion before allocating it."""
    parser = RestrictedParser()
    nodes = list(ast.walk(source.tree))
    expanded_count = len(nodes)
    for node in nodes:
        if isinstance(node, ast.Name) and node.id in replacements:
            expanded_count += sum(1 for _ in ast.walk(replacements[node.id])) - sum(
                1 for _ in ast.walk(node)
            )
    if expanded_count > parser.max_nodes:
        raise ValueError(f"expanded initializer {state} exceeds AST node limit")

    class Substitute(ast.NodeTransformer):
        """Replace symbol reads by copies of already expanded scalar expressions."""

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in replacements:
                return ast.copy_location(deepcopy(replacements[node.id]), node)
            return node

    tree = Substitute().visit(deepcopy(source.tree))
    if parser._depth(tree) > parser.max_depth:
        raise ValueError(f"expanded initializer {state} exceeds AST depth limit")
    rendered = ast.unparse(tree)
    if len(rendered) > parser.max_length:
        raise ValueError(
            f"expanded initializer {state} exceeds expression length limit"
        )
    return parser.parse(rendered, location=f"expanded_initial:{state}")
