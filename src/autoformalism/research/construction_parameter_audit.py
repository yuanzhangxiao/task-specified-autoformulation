"""Read-only equation/parameter bookkeeping for construction feedback.

This is a syntactic audit, not an identifiability or scientific-validity theorem.
Restricted ASTs are expanded without evaluating proposer text or fitting data.
Locally composite coefficients are review candidates, never automatic rewrites:
the image of the original domains and all sharing must survive any later change.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from typing import Any

from autoformalism.expressions import compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY, RestrictedParser
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.staged_topology import content_hash

PROTOCOL = "construction-parameter-bookkeeping-1"
MAX_EXPANDED_NODES = 4096
MAX_EXPANSION_DEPTH = 128
MAX_LOCATIONS_PER_PARAMETER = 128


class _ExpansionLimit(ValueError):
    """The bounded presentation cannot expand this equation completely."""


class _NonAffine(ValueError):
    """The expression is not syntactically affine in states and public forcing."""


def _names(node: ast.AST) -> set[str]:
    """Return value symbols; approved function names are never parameters."""
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and item.id not in APPROVED_FUNCTION_ARITY
    }


def _expand(root: ast.expr, definitions: dict[str, ast.expr]) -> ast.expr:
    """Expand an acyclic process DAG with explicit size/depth limits."""
    remaining = MAX_EXPANDED_NODES

    def visit(node: ast.AST, depth: int, path: frozenset[str]) -> ast.AST:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_EXPANSION_DEPTH:
            raise _ExpansionLimit("expanded expression exceeds bounded audit limits")
        if isinstance(node, ast.Name) and node.id in definitions:
            if node.id in path:
                raise _ExpansionLimit("cyclic process definitions")
            return visit(definitions[node.id], depth + 1, path | {node.id})
        result = deepcopy(node)
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(result, field, visit(value, depth + 1, path))
            elif isinstance(value, list):
                setattr(result, field, [visit(item, depth + 1, path) for item in value])
        return result

    return visit(root, 0, frozenset())  # type: ignore[return-value]


def _number(node: ast.expr, value: int) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _binary(op: type[ast.operator], left: ast.expr, right: ast.expr) -> ast.expr:
    """Elide neutral constants only; do not cancel unknown denominators or terms."""
    if op in (ast.Add, ast.Sub) and _number(right, 0):
        return left
    if op is ast.Add and _number(left, 0):
        return right
    if op is ast.Sub and _number(left, 0):
        return ast.UnaryOp(op=ast.USub(), operand=right)
    if op is ast.Mult:
        if _number(left, 0) or _number(right, 0):
            return ast.Constant(value=0)
        if _number(left, 1):
            return right
        if _number(right, 1):
            return left
    if op is ast.Div and _number(right, 1):
        return left
    return ast.BinOp(left=left, op=op(), right=right)


def _tidy(node: ast.expr) -> ast.expr:
    """Remove neutral factors from coefficient display without merging parameters."""
    if isinstance(node, ast.BinOp):
        return _binary(type(node.op), _tidy(node.left), _tidy(node.right))
    if isinstance(node, ast.UnaryOp):
        operand = _tidy(node.operand)
        return (
            operand
            if isinstance(node.op, ast.UAdd)
            else ast.UnaryOp(op=ast.USub(), operand=operand)
        )
    return node


def _simple_coordinate(node: ast.expr) -> bool:
    """An ordinary coefficient or inverse time constant needs no renaming advice."""
    if isinstance(node, ast.UnaryOp):
        return _simple_coordinate(node.operand)
    return isinstance(node, (ast.Name, ast.Constant)) or (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _number(node.left, 1)
        and isinstance(node.right, ast.Name)
    )


def _affine(root: ast.expr, parameters: set[str]) -> dict[str | None, ast.expr]:
    """Collect a constant and linear coefficients without symbolic cancellation."""
    if _names(root) <= parameters:
        return {None: _tidy(root)}
    if isinstance(root, ast.Name):
        return {root.id: ast.Constant(value=1)}
    if isinstance(root, ast.UnaryOp):
        values = _affine(root.operand, parameters)
        if isinstance(root.op, ast.UAdd):
            return values
        return {
            name: ast.UnaryOp(op=ast.USub(), operand=x) for name, x in values.items()
        }
    if isinstance(root, ast.BinOp):
        left, right = _affine(root.left, parameters), _affine(root.right, parameters)
        if isinstance(root.op, (ast.Add, ast.Sub)):
            return {
                name: _binary(
                    type(root.op),
                    left.get(name, ast.Constant(value=0)),
                    right.get(name, ast.Constant(value=0)),
                )
                for name in dict.fromkeys((*left, *right))
            }
        if isinstance(root.op, ast.Mult):
            if set(left) <= {None}:
                return {
                    name: _binary(ast.Mult, left[None], coefficient)
                    for name, coefficient in right.items()
                }
            if set(right) <= {None}:
                return {
                    name: _binary(ast.Mult, coefficient, right[None])
                    for name, coefficient in left.items()
                }
        if isinstance(root.op, ast.Div) and set(right) <= {None}:
            return {
                name: _binary(ast.Div, coefficient, right[None])
                for name, coefficient in left.items()
            }
        if isinstance(root.op, ast.Pow) and _number(root.right, 1):
            return left
    raise _NonAffine("nonlinear dependence on a state, time or supplied quantity")


def _closure(symbols: set[str], definitions: dict[str, ast.expr]) -> set[str]:
    """Follow process definitions; states are terminal since every RHS is a root."""
    pending, seen = set(symbols), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in definitions:
            pending.update(_names(definitions[name]) - seen)
    return seen


def _limited_locations(locations: set[str]) -> dict[str, Any]:
    return {
        "locations": sorted(locations)[:MAX_LOCATIONS_PER_PARAMETER],
        "count": len(locations),
        "truncated": len(locations) > MAX_LOCATIONS_PER_PARAMETER,
    }


def _sources(candidate) -> dict[str, str]:
    """Every executable definition plus constraint subjects, without prose parsing."""
    return {
        **{f"process:{p.name}": p.expression for p in candidate.processes},
        **{f"rhs:{e.state}": e.rhs for e in candidate.state_equations},
        **{f"output:{o.channel}": o.expression for o in candidate.observation_mappings},
        **{
            f"initial:{i.state}": i.expression
            for i in candidate.initial_conditions
            if i.expression is not None
        },
        **{f"constraint:{i}": c.subject for i, c in enumerate(candidate.constraints)},
    }


def _equations(trees, definitions, parameters):
    """Render bounded full equations and exact affine coefficient collections."""
    rows, slots = [], {}
    for location, node in trees.items():
        if not location.startswith(("rhs:", "output:")):
            continue
        category, target = location.split(":", 1)
        row = {
            "location": location,
            "lhs": f"d({target})/dt" if category == "rhs" else target,
            "declared_expression": ast.unparse(node),
        }
        try:
            expanded = _expand(node, definitions)
        except _ExpansionLimit as error:
            rows.append(row | {"status": "expansion_deferred", "reason": str(error)})
            continue
        row["expanded_expression"] = ast.unparse(expanded)
        try:
            coefficients = _affine(expanded, parameters)
        except _NonAffine as error:
            rows.append(row | {"status": "nonlinear_deferred", "reason": str(error)})
            continue
        if (
            sum(sum(1 for _ in ast.walk(value)) for value in coefficients.values())
            > MAX_EXPANDED_NODES
        ):
            rows.append(
                row
                | {
                    "status": "affine_collection_deferred",
                    "reason": "collected coefficients exceed the audit size limit",
                }
            )
            continue
        row["status"] = "affine_collected"
        row["coefficients"] = {
            name: ast.unparse(coefficient)
            for name, coefficient in coefficients.items()
            if name is not None
        }
        row["offset"] = ast.unparse(coefficients.get(None, ast.Constant(value=0)))
        slots.update(
            {
                (location, name): coefficient
                for name, coefficient in coefficients.items()
            }
        )
        rows.append(row)
    return rows, slots


def assess(problem: PrefixFitProblem) -> dict[str, Any]:
    """Audit unchanged equations, all parameter consumers and scoped initializers.

    No observations, fitted values, optimizer or numerical integrator are read.
    Inactivity is a dependency certificate for mathematical model expressions,
    not permission to remove potentially eagerly evaluated unused definitions.
    """
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "problem_sha256": content_hash(problem.model_dump(mode="json")),
        "automatic_rewrite_permitted": False,
        "parameter_fitting_performed": False,
        "numerical_rollout_performed": False,
        "test_data_opened": False,
        "limits": {
            "expanded_ast_nodes_per_equation": MAX_EXPANDED_NODES,
            "expansion_depth": MAX_EXPANSION_DEPTH,
            "listed_locations_per_parameter": MAX_LOCATIONS_PER_PARAMETER,
        },
    }
    try:
        base = compile_candidate(problem.base_candidate, problem.context)
        compiled, _, initializer_audit = apply_initialization_graph(base, problem.graph)
    except (ModelValidationError, ValueError) as error:
        return result | {"status": "unavailable", "reason": str(error)[:2000]}
    candidate = compiled.validated.candidate
    parameters = {p.name: p for p in candidate.parameters}
    sources = _sources(candidate)
    parser = RestrictedParser()
    trees = {
        location: parser.parse(source, location=location).tree.body
        for location, source in sources.items()
    }
    definitions = {
        location.split(":", 1)[1]: node
        for location, node in trees.items()
        if location.startswith("process:")
    }
    roots = {
        location: _closure(_names(node), definitions)
        for location, node in trees.items()
        if not location.startswith("process:")
    }
    active_symbols = set().union(*roots.values())
    inactive_definitions = sorted(set(definitions) - active_symbols)
    direct: dict[str, set[str]] = defaultdict(set)
    for location, node in trees.items():
        for name in _names(node):
            direct[name].add(location)
    # The graph replaces historical base initials. Retain them for provenance,
    # without falsely certifying them as active scoped-initializer dependencies.
    legacy_initials = {
        f"base_initial:{i.state}": i.expression
        for i in problem.base_candidate.initial_conditions
        if i.expression is not None
    }
    legacy_names = {
        location: parser.parse(source, location=location).symbols
        for location, source in legacy_initials.items()
    }
    parameter_rows = []
    for name, spec in parameters.items():
        active_roots = {location for location, names in roots.items() if name in names}
        parameter_rows.append(
            {
                "parameter": name,
                "declaration": spec.model_dump(mode="json"),
                "direct_uses": _limited_locations(direct[name]),
                "active_consumers": _limited_locations(active_roots),
                "historical_base_initial_uses": sorted(
                    location
                    for location, names in legacy_names.items()
                    if name in names
                ),
                "structurally_inactive": not active_roots,
            }
        )
    equations, slots = _equations(trees, definitions, set(parameters))
    candidates, retained, fixed_scaling = [], [], []
    state_units = {item.name: item.unit for item in candidate.states}
    output_units = {item.channel: item.unit for item in candidate.observation_mappings}
    for (location, symbol), coefficient in slots.items():
        names = _names(coefficient) & parameters.keys()
        literal = RestrictedParser._signed_number(coefficient)
        if literal is not None and literal not in (-1, 0, 1):
            category, target = location.split(":", 1)
            fixed_scaling.append(
                {
                    "location": location,
                    "argument": symbol,
                    "fixed_coefficient": ast.unparse(coefficient),
                    "target_declared_unit": (
                        state_units.get(target, "unspecified")
                        if category == "rhs"
                        else output_units.get(target, "unspecified")
                    ),
                    "argument_declared_unit": state_units.get(symbol, "unspecified"),
                    "time_unit": "unavailable in this interface",
                    "status": "advisory_units_review",
                    "instruction": (
                        "Explain this fixed scaling using declared units or public "
                        "preparation. Missing unit metadata is unresolved, not a "
                        "dimensional contradiction; do not invent a conversion."
                    ),
                }
            )
        if not names or _simple_coordinate(coefficient):
            continue
        external = {
            where for where, used in roots.items() if where != location and names & used
        }
        external.update(
            f"{other_location}/coefficient:{other_symbol}"
            for (other_location, other_symbol), other in slots.items()
            if (other_location, other_symbol) != (location, symbol)
            and names & _names(other)
        )
        external.update(where for where, used in legacy_names.items() if names & used)
        item = {
            "location": location,
            "argument": symbol,
            "effective_coefficient": ast.unparse(coefficient),
            "parameters": sorted(names),
            "external_uses": sorted(external),
            "original_domains": {
                name: parameters[name].model_dump(mode="json") for name in sorted(names)
            },
            "domain_image_derived": False,
            "automatic_rewrite_permitted": False,
        }
        if external:
            retained.append(item | {"reason": "parameters have external uses"})
        else:
            candidates.append(
                item
                | {
                    "reason": "private coefficient; no external parameter uses",
                    "kind": (
                        "single_parameter_reexpression"
                        if len(names) == 1
                        else "multiple_parameter_combination"
                    ),
                    "parameter_count": len(names),
                    "potential_parameter_count_reduction": len(names) - 1,
                    "repair_instruction": (
                        "Consider one effective parameter only if its domain image, "
                        "units, scientific meaning and any retained definitions can be "
                        "preserved. Keep an explicit old-to-new parameter map. Do not "
                        "rewrite shared parameters or widen sign/bound constraints."
                    ),
                }
            )
    return result | {
        "status": "assessed",
        "compiled_candidate_sha256": content_hash(candidate.model_dump(mode="json")),
        "equations": equations,
        "parameters": parameter_rows,
        "simplification_candidates": candidates,
        "retained_composite_coefficients": retained,
        "fixed_scaling_review": fixed_scaling,
        "unused_process_definitions": [
            {"name": name, "expression": sources[f"process:{name}"]}
            for name in inactive_definitions
        ],
        "inactive_parameter_certificate": {
            "parameters": [
                r["parameter"] for r in parameter_rows if r["structurally_inactive"]
            ],
            "method": "syntactic absence from all transitive active expressions",
            "roots": sorted(roots),
            "scope": "ODEs, outputs, scoped initializers and constraint subjects",
            "symbolic_cancellation_used": False,
            "execution_domain_safety_proven": False,
            "limitation": (
                "Unused definitions may still be eagerly evaluated. Their domain "
                "validity and original receipts must be preserved before freezing or "
                "removing any optimizer coordinate. This audit changes no coordinates."
            ),
        },
        "initializer_audit": initializer_audit,
        "historical_base_initials": legacy_initials,
        "interpretation": (
            "Collections apply on the original expression domains. "
            "An effective coefficient is an exact algebraic collection, not proof that "
            "parameters are identifiable or the model is scientifically valid. "
            "Nonlinear coefficients, domain images and automatic rewrites defer."
        ),
    }


def prompt_view(audit: dict[str, Any]) -> dict[str, Any]:
    """Bound feedback to assembled equations and at most 12 local audit examples."""
    if audit.get("status") != "assessed":
        return {"status": audit.get("status"), "reason": audit.get("reason")}
    equations, remaining = [], 12000
    for row in audit["equations"]:
        if len(equations) >= 32:
            break
        expression = row.get("expanded_expression")
        if expression is None:
            continue
        rendered = f"{row['lhs']} = {expression}"
        if len(rendered) > remaining:
            continue
        equations.append(rendered)
        remaining -= len(rendered)
    examples = []
    for item in audit["simplification_candidates"][:12]:
        examples.append(
            {
                "equation": item["location"],
                "argument": item["argument"],
                "effective_coefficient": item["effective_coefficient"],
                "kind": item["kind"],
                "parameters": {
                    name: {
                        "domain": spec["domain"],
                        "bounds": spec["bounds"],
                        "role": spec["role"],
                    }
                    for name, spec in item["original_domains"].items()
                },
            }
        )
    inactive = audit["inactive_parameter_certificate"]["parameters"]
    return {
        "status": "assessed",
        "expanded_equations": equations,
        "equation_count": len(audit["equations"]),
        "omitted_equation_count": len(audit["equations"]) - len(equations),
        "private_coefficient_examples": examples,
        "fixed_scaling_examples": audit["fixed_scaling_review"][:8],
        "fixed_scaling_count": len(audit["fixed_scaling_review"]),
        "private_coefficient_count": len(audit["simplification_candidates"]),
        "inactive_parameter_examples": inactive[:24],
        "inactive_parameter_count": len(inactive),
        "instruction": (
            "These are read-only bookkeeping suggestions, not scientific defects. "
            "A sum or product of private parameters can represent one effective "
            "coefficient. A one-parameter reexpression changes no parameter count: "
            "for example 1+gain with gain>=0 still has one unknown, constrained >=1. "
            "Preserve shared uses, domain images, units and original-to-new mappings. "
            "Do not delete or rewrite parameters automatically; unused definitions "
            "may still be evaluated and require domain checks."
        ),
    }
