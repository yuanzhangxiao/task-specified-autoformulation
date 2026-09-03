"""Name-invariant scientific and executable candidate fingerprints.

The fingerprints deliberately describe three different projections of a
candidate.  They are identifiers for canonical summaries, not replacements for
the candidate payload itself.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import ConfigDict, Field

from autoformalism.schemas import CandidateModel, InitialConditionSpec
from autoformalism.schemas.base import StrictSchema

IdentityMode = Literal["topology", "functional", "executable"]


class CandidateIdentity(StrictSchema):
    """Versioned fingerprints for distinct candidate equivalence levels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["candidate-identity-1"] = "candidate-identity-1"
    topology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    functional_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def candidate_identity(candidate: CandidateModel) -> CandidateIdentity:
    """Return topology, functional, and executable fingerprints.

    Topology ignores operators, numeric constants, and fitted parameters while
    preserving signed additive hyperedges. Functional identity adds the exact
    expression structure and fixed numeric constants but excludes optimizer
    ranges. Executable identity additionally includes all numerical fitting and
    constraint metadata that can change execution.

    All three summaries replace proposer-owned state, process, and parameter
    names with iteratively refined structural colors. Public channel names stay
    anchored. Consequently, a pure alpha-renaming has the same fingerprints.
    """

    return CandidateIdentity(
        topology_sha256=_fingerprint(candidate, "topology"),
        functional_sha256=_fingerprint(candidate, "functional"),
        executable_sha256=_fingerprint(candidate, "executable"),
    )


def _fingerprint(candidate: CandidateModel, mode: IdentityMode) -> str:
    labels = _refined_symbol_labels(candidate, mode)
    payload = _invariant_payload(candidate, labels, mode)
    return _sha256(payload)


def _refined_symbol_labels(
    candidate: CandidateModel,
    mode: IdentityMode,
) -> dict[str, str]:
    bases = _declaration_bases(candidate, mode)
    labels = {name: _sha256(base) for name, base in bases.items()}
    definitions = _definitions(candidate)
    parameters = {item.name for item in candidate.parameters}

    # A fixed number of color-refinement rounds is isomorphism invariant and
    # lets information cross cycles in the dynamic dependency graph.
    for _ in range(max(4, 2 * len(labels) + 2)):
        updated: dict[str, str] = {}
        for name, base in bases.items():
            contexts: list[object] = []
            for owner_kind, owner, expression in definitions:
                if name not in _expression_symbols(expression):
                    continue
                owner_label = (
                    labels.get(owner, owner)
                    if owner_kind != "target"
                    else f"target:{owner}"
                )
                contexts.append(
                    (
                        owner_kind,
                        owner_label,
                        _expression_signature(
                            expression,
                            labels,
                            mode,
                            parameters=parameters,
                            self_symbol=name,
                        ),
                    )
                )
            updated[name] = _sha256(
                {
                    "base": base,
                    "contexts": sorted(
                        contexts,
                        key=lambda item: json.dumps(item, sort_keys=True),
                    ),
                }
            )
        labels = updated
    return labels


def _declaration_bases(
    candidate: CandidateModel,
    mode: IdentityMode,
) -> dict[str, object]:
    bases: dict[str, object] = {}
    initials = {item.state: item for item in candidate.initial_conditions}
    constraints: dict[str, list[object]] = defaultdict(list)
    for item in candidate.constraints:
        constraint: list[object] = [
            item.kind.value,
            item.source.value,
            item.enforcement.value,
        ]
        if mode == "executable":
            constraint.append(
                None
                if item.bounds is None
                else [item.bounds.lower, item.bounds.upper]
            )
        else:
            constraint.append(item.bounds is not None)
        constraints[item.subject].append(constraint)

    for item in candidate.states:
        initial = initials.get(item.name)
        initial_payload: object = None
        if initial is not None:
            if mode == "executable":
                initial_payload = {
                    "scope": initial.scope.value,
                    "mode": _initial_mode(initial),
                    "fixed_value": initial.fixed_value,
                    "initialization_range": (
                        None
                        if initial.initialization_range is None
                        else [
                            initial.initialization_range.lower,
                            initial.initialization_range.upper,
                        ]
                    ),
                }
            else:
                initial_payload = {
                    "scope": initial.scope.value,
                    "mode": _initial_mode(initial),
                }
        bases[item.name] = {
            "declaration": "state",
            "kind": item.kind.value,
            "initial": initial_payload,
            "constraints": sorted(
                constraints.get(item.name, []),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
        }
    for item in candidate.processes:
        bases[item.name] = {
            "declaration": "process",
            "constraints": sorted(
                constraints.get(item.name, []),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
        }
    if mode != "topology":
        for item in candidate.parameters:
            payload: dict[str, object] = {
                "declaration": "parameter",
                "scope": item.scope.value,
            }
            if mode == "executable":
                payload.update(
                    bounds=[item.bounds.lower, item.bounds.upper],
                    initialization_range=[
                        item.initialization_range.lower,
                        item.initialization_range.upper,
                    ],
                )
            bases[item.name] = payload
    return bases


def _definitions(candidate: CandidateModel) -> tuple[tuple[str, str, str], ...]:
    return (
        *(
            ("state", item.state, item.rhs)
            for item in candidate.state_equations
        ),
        *(
            ("process", item.name, item.expression)
            for item in candidate.processes
        ),
        *(
            ("target", item.channel, item.expression)
            for item in candidate.observation_mappings
        ),
        *(
            ("initial", item.state, item.expression)
            for item in candidate.initial_conditions
            if item.expression is not None
        ),
    )


def _invariant_payload(
    candidate: CandidateModel,
    labels: dict[str, str],
    mode: IdentityMode,
) -> dict[str, object]:
    parameters = {item.name for item in candidate.parameters}
    definitions = [
        (
            owner_kind,
            (
                f"target:{owner}"
                if owner_kind == "target"
                else labels.get(owner, owner)
            ),
            _expression_signature(
                expression,
                labels,
                mode,
                parameters=parameters,
            ),
        )
        for owner_kind, owner, expression in _definitions(candidate)
    ]
    return {
        "identity_schema": "candidate-identity-1",
        "mode": mode,
        "declarations": sorted(labels.values()),
        "definitions": sorted(
            definitions,
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }


def _expression_signature(
    source: str,
    labels: dict[str, str],
    mode: IdentityMode,
    *,
    parameters: set[str],
    self_symbol: str | None = None,
) -> object:
    parsed = ast.parse(source, mode="eval").body
    if mode == "topology":
        terms = []
        for sign, term in _signed_additive_terms(parsed):
            symbols = []
            for name in _expression_symbols_node(term):
                if name in parameters:
                    continue
                symbols.append(
                    "@self"
                    if name == self_symbol
                    else labels.get(name, f"public:{name}")
                )
            terms.append((sign, sorted(symbols)))
        return sorted(terms, key=lambda item: json.dumps(item, sort_keys=True))
    return _ast_signature(parsed, labels, self_symbol=self_symbol)


def _ast_signature(
    node: ast.AST,
    labels: dict[str, str],
    *,
    self_symbol: str | None,
) -> object:
    if isinstance(node, ast.Name):
        if node.id == self_symbol:
            return ("symbol", "@self")
        return ("symbol", labels.get(node.id, f"public:{node.id}"))
    if isinstance(node, ast.Constant):
        return ("constant", repr(node.value))
    if isinstance(node, ast.BinOp):
        operator = type(node.op).__name__
        if isinstance(node.op, (ast.Add, ast.Mult)):
            operands = [
                _ast_signature(item, labels, self_symbol=self_symbol)
                for item in _flatten_operator(node, type(node.op))
            ]
            return (
                operator,
                sorted(operands, key=lambda item: json.dumps(item, sort_keys=True)),
            )
        return (
            operator,
            _ast_signature(node.left, labels, self_symbol=self_symbol),
            _ast_signature(node.right, labels, self_symbol=self_symbol),
        )
    if isinstance(node, ast.UnaryOp):
        return (
            type(node.op).__name__,
            _ast_signature(node.operand, labels, self_symbol=self_symbol),
        )
    if isinstance(node, ast.Call):
        function = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else ast.dump(node.func)
        )
        return (
            "call",
            function,
            tuple(
                _ast_signature(item, labels, self_symbol=self_symbol)
                for item in node.args
            ),
        )
    if isinstance(node, ast.IfExp):
        return (
            "if",
            _ast_signature(node.test, labels, self_symbol=self_symbol),
            _ast_signature(node.body, labels, self_symbol=self_symbol),
            _ast_signature(node.orelse, labels, self_symbol=self_symbol),
        )
    if isinstance(node, ast.Compare):
        return (
            "compare",
            _ast_signature(node.left, labels, self_symbol=self_symbol),
            tuple(type(item).__name__ for item in node.ops),
            tuple(
                _ast_signature(item, labels, self_symbol=self_symbol)
                for item in node.comparators
            ),
        )
    if isinstance(node, ast.BoolOp):
        values = [
            _ast_signature(item, labels, self_symbol=self_symbol)
            for item in node.values
        ]
        return (
            type(node.op).__name__,
            sorted(values, key=lambda item: json.dumps(item, sort_keys=True)),
        )
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _signed_additive_terms(node: ast.AST, sign: int = 1) -> list[tuple[int, ast.AST]]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [
            *_signed_additive_terms(node.left, sign),
            *_signed_additive_terms(node.right, sign),
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return [
            *_signed_additive_terms(node.left, sign),
            *_signed_additive_terms(node.right, -sign),
        ]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _signed_additive_terms(node.operand, -sign)
    return [(sign, node)]


def _flatten_operator(node: ast.AST, operator: type[ast.operator]) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, operator):
        return [
            *_flatten_operator(node.left, operator),
            *_flatten_operator(node.right, operator),
        ]
    return [node]


def _expression_symbols(source: str) -> set[str]:
    return _expression_symbols_node(ast.parse(source, mode="eval").body)


def _expression_symbols_node(node: ast.AST) -> set[str]:
    result: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call) -> None:
            for argument in call.args:
                self.visit(argument)
            for keyword in call.keywords:
                self.visit(keyword.value)

        def visit_Name(self, name: ast.Name) -> None:
            result.add(name.id)

    Visitor().visit(node)
    return result


def _initial_mode(initial: InitialConditionSpec) -> str:
    if initial.fixed_value is not None:
        return "fixed"
    if initial.expression is not None:
        return "expression"
    if initial.initialization_range is not None:
        return "fitted_range"
    return "contextual"


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
