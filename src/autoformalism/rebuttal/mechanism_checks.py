"""Method-neutral equation predicates, with explicit limits on certification.

Reuse the pre-fit checker's restricted simplification, graph traversal and
nonlinear-source test. No tags, LLM verdicts or private reference are inputs.
"""

from __future__ import annotations

import ast
import copy
import math
from collections import Counter
from typing import Literal

from autoformalism.expressions import RestrictedParser, ValidationContext
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.requirement_feedback import _ancestors, _simplify
from autoformalism.staged_functions import has_nonlinear_source_dependence


class EquationRequirement(StrictSchema):
    """A reviewed operational predicate bound to an exact public requirement."""

    id: str
    public_requirement: str
    kind: Literal[
        "input_path", "dynamic_memory", "nonlinear_feedback", "balance_channel"
    ]
    driver: str
    target: str
    interpretation: str


def counts(rows: list[dict]) -> dict:
    """Retain unresolved predicates in the denominator; never infer correctness."""
    n = Counter(r["status"] for r in rows)
    return {
        "total": len(rows),
        **{s: n[s] for s in ("pass", "fail", "unresolved")},
        "passed_fraction": n["pass"] / len(rows) if rows else None,
    }


def effective_trees(
    candidate: CandidateModel, parameters: dict[str, float] | None = None
) -> dict[str, ast.AST]:
    """Substitute fitted scalars before removing obvious inactive syntax."""

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            if parameters is not None and node.id in parameters:
                value = parameters[node.id]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("nonfinite saved parameter")
                return ast.copy_location(ast.Constant(float(value)), node)
            return node

    expressions = {
        **{e.state: e.rhs for e in candidate.state_equations},
        **{p.name: p.expression for p in candidate.processes},
        **{f"output:{o.channel}": o.expression for o in candidate.observation_mappings},
    }
    parser = RestrictedParser()
    return {
        name: _simplify(
            Substitute().visit(copy.deepcopy(parser.parse(rhs, location=name).tree))
        )
        for name, rhs in expressions.items()
    }


def equation_evidence(
    candidate: CandidateModel,
    context: ValidationContext,
    requirements: list[EquationRequirement],
    *,
    parameters: dict[str, float] | None = None,
    semantics: str = "continuous_time",
) -> dict:
    """Check formal features, not their empirical strength or scientific truth."""
    if parameters is not None and set(parameters) != {
        p.name for p in candidate.parameters
    }:
        raise ValueError("require exactly the complete saved parameter vector")
    trees = effective_trees(candidate, parameters)
    dependencies = {
        name: {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name, tree in trees.items()
    }
    dynamic = {e.state for e in candidate.state_equations} - set(context.auxiliaries)
    rows = []
    for rule in requirements:
        target = f"output:{rule.target}"
        route = _ancestors(target, dependencies)
        driven = {
            name for name in route if rule.driver in _ancestors(name, dependencies)
        }
        evidence = sorted(driven & dynamic)
        path = rule.driver in route
        present = path
        if rule.kind == "dynamic_memory":
            # Memory need not use a particular hidden coordinate or state name.
            present = bool(evidence)
        elif rule.kind == "nonlinear_feedback":
            evidence = []
            for lhs in sorted(driven):
                cycle_sources = {
                    s
                    for s in dependencies.get(lhs, ())
                    if s in trees and (s == lhs or lhs in _ancestors(s, dependencies))
                }
                if cycle_sources and has_nonlinear_source_dependence(
                    trees[lhs], cycle_sources
                ):
                    evidence.append(lhs)
            present = bool(evidence)
        elif rule.kind == "balance_channel":
            # These are public channel-based witnesses, not a proof that a
            # reaction/feed/exchange term has the claimed physical meaning.
            other_drivers = {
                r.driver for r in requirements if r.kind == "balance_channel"
            } - {rule.driver}

            def terms(node, seen=frozenset()):
                if isinstance(node, ast.Expression):
                    return terms(node.body, seen)
                if isinstance(node, ast.UnaryOp):
                    return terms(node.operand, seen)
                if isinstance(node, ast.BinOp) and isinstance(
                    node.op, (ast.Add, ast.Sub)
                ):
                    return terms(node.left, seen) + terms(node.right, seen)
                if (
                    isinstance(node, ast.Name)
                    and node.id in trees
                    and node.id not in dynamic
                    and node.id not in seen
                ):
                    return terms(trees[node.id], seen | {node.id})
                return [node]

            evidence = []
            for lhs in sorted(dynamic & route):
                for term in terms(trees[lhs]):
                    names = {n.id for n in ast.walk(term) if isinstance(n, ast.Name)}
                    expanded, pending = set(names), list(names - dynamic)
                    while pending:
                        current = pending.pop()
                        additions = dependencies.get(current, set()) - expanded
                        expanded.update(additions)
                        pending.extend(additions - dynamic)
                    if rule.driver in expanded and not expanded & other_drivers:
                        evidence.append(f"{lhs}: {ast.unparse(term)}")
            present = path and bool(evidence)
        status = "pass" if present else "fail"
        if rule.kind == "balance_channel" and not present:
            status = "unresolved"  # An alternative internal representation may exist.
        rows.append(
            {
                **rule.model_dump(),
                "status": status,
                "input_to_target_path": path,
                "witness_equations": evidence,
                "scope": "public_channel_witness_only"
                if rule.kind == "balance_channel"
                else "necessary_equation_feature",
            }
        )
    return {
        "counts": counts(rows),
        "requirements": rows,
        "equation_semantics": semantics,
        "continuous_time_representation": "pass"
        if semantics == "continuous_time"
        else "fail",
        "fitted_parameters_used": parameters is not None,
        "effective_expressions": {
            name: ast.unparse(tree) for name, tree in trees.items()
        },
        "scientific_correctness_certified": False,
        "limitation": "Finite syntax checks; general cancellation, physical balance, "
        "stability and identifiability are not certified. An observed dynamic state "
        "can carry memory. Native identity carryover alone is not a feedback witness.",
    }
