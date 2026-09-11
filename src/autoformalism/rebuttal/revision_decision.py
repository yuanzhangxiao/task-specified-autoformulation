"""Public-only decision evidence and truthful local action contracts.

No fitting, hidden simulator, or data loading belongs in this module.  A
certificate of nonlinearity below is only a syntax/path check, not a scientific
mechanism score.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.staged_topology import content_hash


class RevisionDecisionState(StrictSchema):
    """Separate committed search state, numerical evidence, and pending work."""

    schema_version: str = "revision-decision-state-1"
    committed_candidate_sha256: str
    best_evaluated: dict[str, Any] | None = None
    last_numerical_evidence: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    pending_transaction: dict[str, Any] | None = None


def expressions(candidate: CandidateModel) -> dict[str, str]:
    """Return generated definitions, without confusing ODEs with readouts."""
    return {
        **{item.state: item.rhs for item in candidate.state_equations},
        **{item.name: item.expression for item in candidate.processes},
    }


def additive_terms(node: ast.expr, sign: int = 1) -> list[tuple[int, ast.expr]]:
    """Expose the signed outer additive shell, not distributive rewrites."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return additive_terms(node.left, sign) + additive_terms(
            node.right, -sign if isinstance(node.op, ast.Sub) else sign
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return additive_terms(
            node.operand, -sign if isinstance(node.op, ast.USub) else sign
        )
    return [(sign, node)]


def interaction_contract(
    candidate: CandidateModel, component: str
) -> list[dict[str, Any]]:
    """Describe exactly the term-sign/source multiset frozen by local repair."""
    parameters = {item.name for item in candidate.parameters}
    tree = (
        RestrictedParser()
        .parse(expressions(candidate)[component], location=component)
        .tree
    )
    return [
        {
            "slot": index,
            "outer_sign": sign,
            "sources": sorted(
                set(
                    RestrictedParser()
                    .parse(ast.unparse(term), location=component)
                    .symbols
                )
                - parameters
            ),
            "expression": ast.unparse(term),
        }
        for index, (sign, term) in enumerate(additive_terms(tree.body))
    ]


def interaction_diff(
    before: CandidateModel, after: CandidateModel, component: str
) -> dict[str, Any]:
    """Return unmatched hyperedges with multiplicity; order alone is immaterial."""

    def counts(model: CandidateModel) -> Counter:
        return Counter(
            (item["outer_sign"], tuple(item["sources"]))
            for item in interaction_contract(model, component)
        )

    old, new = counts(before), counts(after)

    def rows(value: Counter) -> list[dict[str, Any]]:
        return [
            {"outer_sign": sign, "sources": list(sources), "count": count}
            for (sign, sources), count in sorted(value.items())
        ]

    return {
        "component": component,
        "removed_interactions": rows(old - new),
        "added_interactions": rows(new - old),
        "required_action_if_intentional": "topology_revision",
    }


def parameter_aliases(candidate: CandidateModel) -> dict[str, str]:
    """Choose stable short request-local aliases, avoiding all model symbols."""
    occupied = set(expressions(candidate)) | {p.name for p in candidate.parameters}
    for value in expressions(candidate).values():
        occupied.update(RestrictedParser().parse(value, location="aliases").symbols)
    aliases = {}
    index = 0
    for name in sorted(p.name for p in candidate.parameters):
        while True:
            index += 1
            alias = f"par_{index:03d}"
            if alias not in occupied:
                break
        aliases[name] = alias
        occupied.add(alias)
    return aliases


def translate_names(value: Any, mapping: Mapping[str, str]) -> Any:
    """Rename complete identifier tokens only; never evaluate provider text."""
    if isinstance(value, str):
        return re.sub(
            r"\b[A-Za-z_][A-Za-z_0-9]*\b",
            lambda match: mapping.get(match[0], match[0]),
            value,
        )
    if isinstance(value, list):
        return [translate_names(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            mapping.get(key, key): translate_names(item, mapping)
            for key, item in value.items()
        }
    return value


def nonlinear_target_paths(
    candidate: CandidateModel, targets: tuple[str, ...]
) -> dict[str, bool]:
    """Conservatively detect non-affine generated-variable syntax on a target path.

    This necessary condition cannot certify nonlinear feedback scientifically;
    cancellation and fitted zero gains can still deactivate the pathway.
    """
    definitions = expressions(candidate)
    generated = set(definitions)

    def degree(node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            return int(node.id in generated)
        if isinstance(node, ast.Constant):
            return 0
        if isinstance(node, ast.UnaryOp):
            return degree(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = degree(node.left), degree(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(left, right)
            if isinstance(node.op, ast.Mult):
                return min(2, left + right)
            if isinstance(node.op, ast.Div):
                return 2 if right else left
            if isinstance(node.op, ast.Pow):
                if not left and not right:
                    return 0
                if isinstance(node.right, ast.Constant) and node.right.value == 1:
                    return left
                return 2
        if isinstance(node, ast.Call):
            return 2 if any(degree(arg) for arg in node.args) else 0
        return 2

    dependencies = {
        name: set(RestrictedParser().parse(value, location=name).symbols) & generated
        for name, value in definitions.items()
    }
    nonlinear = {
        name
        for name, value in definitions.items()
        if degree(RestrictedParser().parse(value, location=name).tree.body) > 1
    }
    mappings = {
        item.channel: item.expression for item in candidate.observation_mappings
    }
    result = {}
    for target in targets:
        seen: set[str] = set()
        frontier = list(
            set(RestrictedParser().parse(mappings[target], location=target).symbols)
            & generated
        )
        while frontier:
            node = frontier.pop()
            if node not in seen:
                seen.add(node)
                frontier.extend(dependencies[node] - seen)
        result[target] = bool(seen & nonlinear)
    return result


def decision_snapshot(
    candidate: CandidateModel,
    rounds: list[dict[str, Any]],
    numerical_evidence: Mapping[str, Any],
    pending: Mapping[str, Any] | None = None,
) -> RevisionDecisionState:
    """Rebuild best-so-far and history from immutable round records on resume."""
    history = [
        {
            "round_index": item["round_index"],
            "candidate_sha256": item["candidate_sha256"],
            "fit_status": item["fit"].get("status"),
            "failure_class": item["fit"].get("failure_class"),
            "validation_normalized_mse": item.get("validation_normalized_mse"),
        }
        for item in rounds
    ]
    eligible = [
        item
        for item in rounds
        if item.get("numerically_stable")
        and isinstance(item.get("validation_normalized_mse"), (int, float))
        and math.isfinite(item["validation_normalized_mse"])
    ]
    best = min(
        eligible, key=lambda item: item["validation_normalized_mse"], default=None
    )
    return RevisionDecisionState(
        committed_candidate_sha256=content_hash(candidate.model_dump(mode="json")),
        best_evaluated=(
            {
                "round_index": best["round_index"],
                "candidate_sha256": best["candidate_sha256"],
                "validation_normalized_mse": best["validation_normalized_mse"],
            }
            if best
            else None
        ),
        last_numerical_evidence=dict(numerical_evidence),
        history=history,
        pending_transaction=(
            {
                key: pending.get(key)
                for key in ("parent_sha256", "route", "selected", "pending", "kept")
            }
            if pending and pending.get("status") == "pending"
            else None
        ),
    )
