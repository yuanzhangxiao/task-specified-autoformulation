"""Canonical structure features and seed-stability comparisons."""

from __future__ import annotations

import ast
from itertools import combinations

from pydantic import BaseModel, ConfigDict

from autoformalism.expressions import RestrictedParser
from autoformalism.schemas import CandidateModel


class StructureFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dependency_edges: frozenset[str]
    target_terms: frozenset[str]
    latent_state_count: int


class StructureSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_id: str
    right_id: str
    edge_jaccard: float
    term_jaccard: float


def structure_features(candidate: CandidateModel) -> StructureFeatures:
    """Extract alpha-normalized edges and commutative additive terms."""
    names = _alpha_names(candidate)
    parser = RestrictedParser()
    edges: set[str] = set()
    for process in candidate.processes:
        expression = parser.parse(
            process.expression, location=f"process:{process.name}"
        )
        edges.update(
            f"{names.get(symbol, symbol)}->{names[process.name]}"
            for symbol in expression.symbols
        )
    for equation in candidate.state_equations:
        expression = parser.parse(
            equation.rhs, location=f"equation:{equation.state}"
        )
        edges.update(
            f"{names.get(symbol, symbol)}->{names[equation.state]}"
            for symbol in expression.symbols
        )
    target_states = {
        symbol
        for mapping in candidate.observation_mappings
        for symbol in parser.parse(
            mapping.expression, location=f"observation:{mapping.channel}"
        ).symbols
    }
    equations = {item.state: item.rhs for item in candidate.state_equations}
    terms = {
        _canonical_ast(term, names)
        for state in target_states
        if state in equations
        for term in _additive_terms(ast.parse(equations[state], mode="eval").body)
    }
    return StructureFeatures(
        dependency_edges=frozenset(edges),
        target_terms=frozenset(terms),
        latent_state_count=sum(
            item.kind.value == "latent" for item in candidate.states
        ),
    )


def pairwise_similarities(
    items: tuple[tuple[str, CandidateModel], ...]
) -> tuple[StructureSimilarity, ...]:
    features = {
        identifier: structure_features(candidate)
        for identifier, candidate in items
    }
    return tuple(
        StructureSimilarity(
            left_id=left,
            right_id=right,
            edge_jaccard=_jaccard(
                features[left].dependency_edges,
                features[right].dependency_edges,
            ),
            term_jaccard=_jaccard(
                features[left].target_terms,
                features[right].target_terms,
            ),
        )
        for (left, _), (right, _) in combinations(items, 2)
    )


def _alpha_names(candidate: CandidateModel) -> dict[str, str]:
    names: dict[str, str] = {}
    for prefix, values in (
        ("s", sorted(item.name for item in candidate.states)),
        ("q", sorted(item.name for item in candidate.processes)),
        ("p", sorted(item.name for item in candidate.parameters)),
    ):
        names.update({name: f"{prefix}{index}" for index, name in enumerate(values)})
    return names


def _additive_terms(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _additive_terms(node.left) + _additive_terms(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return (
            *_additive_terms(node.left),
            ast.UnaryOp(ast.USub(), node.right),
        )
    return (node,)


def _canonical_ast(node: ast.AST, names: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return names.get(node.id, node.id)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp):
        return f"{type(node.op).__name__}({_canonical_ast(node.operand, names)})"
    if isinstance(node, ast.BinOp):
        parts = (
            _canonical_ast(node.left, names),
            _canonical_ast(node.right, names),
        )
        if isinstance(node.op, (ast.Add, ast.Mult)):
            parts = tuple(sorted(parts))
        return f"{type(node.op).__name__}({parts[0]},{parts[1]})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = ",".join(_canonical_ast(item, names) for item in node.args)
        return f"{node.func.id}({arguments})"
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
