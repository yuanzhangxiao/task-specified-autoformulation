"""Public T2 topology construction and deterministic sifting for the breadth pilot."""

from __future__ import annotations

import ast
from graphlib import TopologicalSorter
from typing import Literal

from pydantic import Field

from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.models import FitConfig
from autoformalism.rebuttal import basin_algebra as algebra
from autoformalism.research import insulin_boundary_revision as boundary
from autoformalism.research.construction_repair_identity import family_identity
from autoformalism.research.construction_revision_edits import (
    RevisionValidationError,
    _definitions,
    _path,
    _symbols,
    _validate_problem,
    public_path_checks,
)
from autoformalism.research.construction_search_policy import InScopeRevisionReply
from autoformalism.research.construction_structural_revision import (
    InitializationEdit,
    _initial_rule,
    _normalize_law,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash


class Component(StrictSchema):
    name: Identifier
    kind: Literal["dynamic", "algebraic"]
    role: str = Field(min_length=1, max_length=500)
    sources: tuple[Identifier, ...] = Field(max_length=32)
    represents_auxiliary: Identifier | None = None


class Topology(StrictSchema):
    hypothesis: str = Field(min_length=1, max_length=1500)
    components: tuple[Component, ...] = Field(min_length=3, max_length=32)

    def aliases(self) -> dict[str, str]:
        """Bind each modeled public auxiliary to its generated identifier."""
        aliases = {}
        for node in self.components:
            if channel := node.represents_auxiliary:
                if channel in aliases:
                    raise ValueError(f"Two components represent {channel}.")
                aliases[channel] = node.name
        return aliases


class Formula(StrictSchema):
    component: Identifier
    expression: str = Field(min_length=1, max_length=4096)


class Laws(StrictSchema):
    formulas: tuple[Formula, ...] = Field(min_length=3, max_length=32)
    parameters: tuple[FunctionParameter, ...] = Field(max_length=64)
    latent_initials: tuple[InitializationEdit, ...] = Field(max_length=32)


class FreshCorrection(StrictSchema):
    """Allow a formula-stage correction to change the topology as well."""

    topology: Topology
    laws: Laws


class FiveGlucoseRepairs(StrictSchema):
    alternatives: tuple[InScopeRevisionReply, ...] = Field(min_length=5, max_length=5)


def check_topology(
    topology: Topology, context: ValidationContext, brief: PublicScientificBrief
):
    """Check declared graph edges before spending a call on functions."""
    nodes = {x.name: x for x in topology.components}
    states = {name for name, x in nodes.items() if x.kind == "dynamic"}
    targets = set(context.targets)
    supplied = set(context.forcing_channels) | {context.time_symbol}
    if len(nodes) != len(topology.components) or not states or targets - nodes.keys():
        raise ValueError(
            "Declare each generated quantity once, including all targets and at "
            "least one state."
        )
    if collisions := nodes.keys() & (supplied | APPROVED_FUNCTION_ARITY.keys()):
        raise ValueError(
            f"Name already used by a channel or function: {sorted(collisions)}"
        )
    aliases = topology.aliases()
    if aliases.keys() - set(context.auxiliaries) or set(aliases.values()) & targets:
        raise ValueError(
            "Only a private generated quantity may represent a public auxiliary."
        )
    graph = {}
    algebraics = nodes.keys() - states
    for node in nodes.values():
        if both := set(node.sources) & aliases.keys():
            raise ValueError(
                f"Use the generated names: {[aliases[s] for s in sorted(both)]}"
            )
        if unknown := set(node.sources) - nodes.keys() - supplied:
            raise ValueError(f"{node.name} has undeclared sources {sorted(unknown)}.")
        for source in node.sources:
            graph.setdefault(source, set()).add(node.name)
    tuple(
        TopologicalSorter(
            {
                name: sorted(set(nodes[name].sources) & algebraics)
                for name in sorted(algebraics)
            }
        ).static_order()
    )
    for name in nodes:
        if not any(_path(graph, name, t) for t in targets):
            raise ValueError(f"{name} has no path to an output.")
    for req in brief.requirements:
        for src in req.drivers:
            for dst in req.targets:
                if not _path(graph, src, dst) or (
                    req.requires_dynamic_memory
                    and not any(
                        _path(graph, src, state) and _path(graph, state, dst)
                        for state in states - {src}
                    )
                ):
                    raise ValueError(
                        f"Missing required path {src} -> {dst}. "
                        f"Requires memory: {req.requires_dynamic_memory}."
                    )
    for dep in brief.target_dependencies:
        if not any(
            _path(graph, aliases.get(src, src), dep.target)
            for src in dep.acceptable_sources
        ):
            raise ValueError(
                f"{dep.target} needs a path from one of {list(dep.acceptable_sources)}."
            )


def assemble(
    topology: Topology, laws: Laws, context: ValidationContext
) -> PrefixFitProblem:
    """Lower complete, newly proposed equations and causal initials for fitting."""
    nodes = {c.name: c for c in topology.components}
    eqs = {row.component: row.expression for row in laws.formulas}
    if set(eqs) != set(nodes) or len(eqs) != len(laws.formulas):
        raise ValueError(
            "Supply exactly one formula for each declared graph component."
        )
    parameters = {p.name for p in laws.parameters}
    allowed = nodes.keys() | set(context.forcing_channels) | {context.time_symbol}
    if len(parameters) != len(laws.parameters) or parameters & (
        allowed | APPROVED_FUNCTION_ARITY.keys()
    ):
        raise ValueError(
            "Use unique parameter names distinct from variables and functions."
        )
    states, dynamics, processes, initials = [], [], [], []
    used = set()
    latent = set()
    for name, node in nodes.items():
        expression, actual = _normalize_law(
            eqs[name],
            name,
            node.kind,
            node.sources,
            time_symbol=context.time_symbol,
            parameters=parameters,
            allowed_sources=allowed,
            changes=[],
        )
        if set(actual) != set(node.sources):
            raise ValueError(
                f"{name} declares {sorted(node.sources)}. Formula sources: {actual}."
            )

        used.update(_symbols(expression, name) & parameters)
        if node.kind == "dynamic":
            measured = name in context.targets
            states.append(
                {
                    "name": name,
                    "kind": "observed" if measured else "latent",
                    "description": node.role,
                }
            )
            dynamics.append({"state": name, "rhs": expression})
            initials.append(
                {
                    "state": name,
                    "scope": "global",
                    **({"expression": name} if measured else {"fixed_value": 0.0}),
                }
            )
            if not measured:
                latent.add(name)
        else:
            processes.append(
                {"name": name, "expression": expression, "description": node.role}
            )
    if parameters != used:
        raise ValueError(f"Unused model parameters: {sorted(parameters - used)}")
    rules = {
        row.state: _initial_rule(row.state, row.initial, allow_trajectory_roots=True)
        for row in laws.latent_initials
    }
    if rules.keys() != latent or len(rules) != len(laws.latent_initials):
        raise ValueError(f"Supply one initializer per hidden state: {sorted(latent)}")
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "fresh_"
            + content_hash([topology.model_dump(), laws.model_dump()])[:24],
            "parent_candidate_id": None,
            "change_summary": topology.hypothesis,
            "states": states,
            "state_equations": dynamics,
            "processes": processes,
            "parameters": [
                {"name": p.name, "role": p.role, "scope": "global"}
                for p in laws.parameters
            ],
            "initial_conditions": initials,
            "observation_mappings": [
                {"channel": t, "expression": t} for t in context.targets
            ],
        }
    )
    model = compile_candidate(candidate, context)
    guesses = {}
    for p in candidate.parameters:
        variable = _parameter_variable(model, p, FitConfig())
        guesses[p.name] = (variable.start_lower + variable.start_upper) / 2
    problem = PrefixFitProblem.model_validate(
        {
            "base_candidate": candidate,
            "context": context,
            "graph": {"rules": rules},
            "parameter_guesses": guesses,
        }
    )
    _validate_problem(problem)
    return problem


def check_paths(problem, brief, aliases):
    rows = public_path_checks(problem, brief, scientific_aliases=aliases)
    if any(row["status"] != "pass" for row in rows):
        raise RevisionValidationError(
            "MISSING_PUBLIC_PATH", "Required dependency missing", checks=rows
        )
    return rows


def glucose_balance(problem: PrefixFitProblem) -> dict:
    """Identify retained zero-meal net source on the defined polynomial balance."""
    candidate = problem.base_candidate
    readouts = {m.channel: m.expression for m in candidate.observation_mappings}
    defs = _definitions(candidate)
    if (
        readouts.get("Gp", "").strip() != "Gp"
        or defs.get("Gp", (None,))[0] != "dynamic"
    ):
        return {
            "status": "deferred",
            "reason": "glucose readout is not an identity-mapped differential state",
        }
    processes = {p.name: p.expression for p in candidate.processes}
    try:
        # Keep the measured disposal as one symbol first. Cancellation of the
        # same readout holds wherever that readout is defined, even if nonlinear.
        if readouts["U"].strip() == "U":
            named = {k: v for k, v in processes.items() if k != "U"}
            held = algebra.expanded(f"({defs['Gp'][1]}) + U", named)
            polynomial = boundary._polynomial(held)
            if polynomial is not None and not boundary._at_zero(
                polynomial, {boundary.MEAL}
            ):
                return {"status": "identically_zero", "zero_meal_dGp_dt_plus_U": "0"}
        total = algebra.expanded(f"({defs['Gp'][1]}) + ({readouts['U']})", processes)
        poly = boundary._polynomial(total)
        if poly is None:
            return {
                "status": "deferred",
                "reason": "outside the total-polynomial proof",
            }
        residual = boundary._at_zero(poly, {boundary.MEAL})
        return {
            "status": "nonzero_expression" if residual else "identically_zero",
            "zero_meal_dGp_dt_plus_U": algebra.render(residual),
        }
    except (ValueError, ModelValidationError, RecursionError) as exc:
        return {"status": "deferred", "reason": str(exc)[:300]}


def balance_view(problem: PrefixFitProblem) -> dict:
    """Group literally equal expanded dynamics, readouts, domains and initial laws.

    Only total polynomial arithmetic is canonicalized. Other expressions retain
    their restricted syntax, including divisions. Private names are preserved.
    """
    c = problem.base_candidate
    processes = {p.name: p.expression for p in c.processes}

    def expanded(text):
        tree = algebra.expanded(text, processes)
        poly = boundary._polynomial(tree)
        return ast.unparse(tree) if poly is None else algebra.render(poly)

    try:
        balances = {e.state: expanded(e.rhs) for e in c.state_equations}
        observations = {
            m.channel: expanded(m.expression) for m in c.observation_mappings
        }
        # An unused process can still have a singularity when compiled. Only
        # merge expanded models when every removed definition is total.
        total = all(
            boundary._polynomial(algebra.expanded(p.expression, processes)) is not None
            for p in c.processes
        )
        description = {
            "derivatives": balances,
            "readouts": observations,
            "states": sorted((s.name, s.kind.value) for s in c.states),
            "parameters": {
                p.name: [
                    p.role.value,
                    p.domain.value,
                    p.bounds.model_dump() if p.bounds else None,
                ]
                for p in c.parameters
            },
            "constraints": [x.model_dump(mode="json") for x in c.constraints],
            "base_initials": {
                row.state: row.model_dump(mode="json") for row in c.initial_conditions
            },
            "scoped_initials": problem.graph.model_dump(mode="json"),
        }
        return {
            "expanded": balances,
            "readouts": observations,
            "balance_sha256": content_hash(description) if total else None,
        }
    except (ValueError, ModelValidationError, RecursionError) as exc:
        return {"balance_sha256": None, "reason": str(exc)[:300]}


def alternative_identity(parent, problem):
    """A deferred family comparison cannot merge two different options."""
    return family_identity(parent, problem).get("sha256")
