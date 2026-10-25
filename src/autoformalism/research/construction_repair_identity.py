"""Bounded identities for complete repair families and unchanged failures.

Only consistent renaming of private symbols is ignored. This is not general
algebraic equivalence: AST order, constants, domains, sharing and initialization
remain significant. Numerical starting guesses and descriptive prose are not a
model family. Unsupported or excessively symmetric graphs defer, never collide.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from autoformalism.expressions import compile_candidate
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.staged_topology import content_hash

POLICY = "complete-repair-private-renaming-1"
LIMITS = {"private_symbols": 128, "permutations": 4096, "expression_nodes": 32768}
_PROVENANCE_FIELDS = frozenset(
    {
        "artifact_sha256",
        "parent_problem_sha256",
        "child_problem_sha256",
        "problem_sha256",
        "source_sha256",
        "tool_sha256",
        "runtime_source_sha256",
        "request_hash",
        "reply_sha256",
    }
)
_SYMBOL_SET_FIELDS = frozenset(
    {
        "closed_states",
        "complete_common_forcing",
        "initialized_states",
        "initial_public_channels",
        "forcing_channels",
        "trajectory_roots",
        "possibly_different_initial_states",
        "exact_conditioning_channels",
        "scope",
    }
)


class _Unsupported(ValueError):
    """No conservative family identity is available within the fixed bounds."""


@dataclass(frozen=True)
class _Ref:
    key: str


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _replace(value: Any, labels: dict[str, Any]) -> Any:
    if isinstance(value, _Ref):
        return ["private", labels[value.key]]
    if isinstance(value, dict):
        return {key: _replace(item, labels) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_replace(item, labels) for item in value]
    return value


def _references(value: Any, path: tuple = ()):
    if isinstance(value, _Ref):
        yield value.key, path
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            yield from _references(item, (*path, key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _references(item, (*path, index))


def _color(signatures: dict[str, str]) -> dict[str, int]:
    labels = {
        value: index for index, value in enumerate(sorted(set(signatures.values())))
    }
    return {key: labels[value] for key, value in signatures.items()}


def _partition(colors: dict[str, int]) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = defaultdict(set)
    for key, value in colors.items():
        groups[value].add(key)
    return {frozenset(group) for group in groups.values()}


def _canonical(root: Any, nodes: dict[str, Any]) -> tuple[str, dict[str, int], int]:
    """Refine a colored graph, then enumerate only remaining bounded ambiguity."""
    if len(nodes) > LIMITS["private_symbols"]:
        raise _Unsupported("private-symbol bound exceeded")
    if not nodes:
        return _json({"root": root, "private_definitions": []}), {}, 1
    placeholders = {key: value["type"] for key, value in nodes.items()}
    colors = _color(
        {key: _json(_replace(value, placeholders)) for key, value in nodes.items()}
    )
    incoming: dict[str, list[tuple[str | None, tuple]]] = defaultdict(list)
    for target, path in _references(root):
        incoming[target].append((None, path))
    for source, value in nodes.items():
        for target, path in _references(value):
            incoming[target].append((source, path))
    for _ in range(len(nodes) + 1):
        signatures = {
            key: _json(
                [
                    colors[key],
                    _replace(value, colors),
                    sorted(
                        [
                            ["root" if source is None else colors[source], path]
                            for source, path in incoming[key]
                        ],
                        key=_json,
                    ),
                ]
            )
            for key, value in nodes.items()
        }
        refined = _color(signatures)
        stable = _partition(refined) == _partition(colors)
        colors = refined
        if stable:
            break
    groups: dict[int, list[str]] = defaultdict(list)
    for key, color in colors.items():
        groups[color].append(key)
    ordered = [sorted(groups[color]) for color in sorted(groups)]
    permutations = math.prod(math.factorial(len(group)) for group in ordered)
    if permutations > LIMITS["permutations"]:
        raise _Unsupported("private-symbol symmetry exceeds canonicalization bound")
    best: str | None = None
    best_labels: dict[str, int] = {}
    for choice in itertools.product(
        *(itertools.permutations(group) for group in ordered)
    ):
        order = list(itertools.chain.from_iterable(choice))
        labels = {key: index for index, key in enumerate(order)}
        payload = _json(
            {
                "root": _replace(root, labels),
                "private_definitions": [_replace(nodes[key], labels) for key in order],
            }
        )
        if best is None or payload < best:
            best, best_labels = payload, labels
    if best is None:  # pragma: no cover - the empty case is handled above
        raise _Unsupported("no canonical labeling")
    return best, best_labels, permutations


class _ModelGraph:
    """Translate typed formulas and initial aliases into a finite labeled graph."""

    def __init__(self, parent: PrefixFitProblem, child: PrefixFitProblem):
        self.child = child
        self.nodes: dict[str, Any] = {}
        self.global_names: dict[str, str] = {}
        self.initial_parameters: dict[tuple[str, str], str] = {}
        self.remaining = LIMITS["expression_nodes"]
        self.fixed = {
            item.name
            for field in ("states", "processes", "parameters")
            for item in getattr(parent.base_candidate, field)
        }
        self.fixed |= set(parent.context.targets) | set(parent.context.forcing_channels)
        self.fixed.add(parent.context.time_symbol)
        for field, kind in (
            ("states", "state"),
            ("processes", "process"),
            ("parameters", "parameter"),
        ):
            for item in getattr(child.base_candidate, field):
                if item.name not in self.fixed:
                    key = f"{kind}:{item.name}"
                    self.global_names[item.name] = key
                    self.nodes[key] = {"type": kind}
        for state, rule in child.graph.rules.items():
            if rule.initial.mode == "map":
                for parameter in rule.initial.parameters:
                    key = f"initial_parameter:{state}:{parameter.name}"
                    self.initial_parameters[(state, parameter.name)] = key
                    self.nodes[key] = {
                        "type": "initial_parameter",
                        "owner": self.symbol(state),
                        "role": parameter.role.value,
                        "domain": parameter.role.domain.value,
                    }

    def symbol(self, name: str) -> Any:
        return (
            _Ref(self.global_names[name])
            if name in self.global_names
            else ["fixed", name]
        )

    def expression(self, text: str, *, aliases: dict | None = None) -> Any:
        """Parse the existing restricted grammar; never evaluate model output."""
        tree = RestrictedParser().parse(text, location="repair family identity").tree
        self.remaining -= sum(1 for _ in ast.walk(tree))
        if self.remaining < 0:
            raise _Unsupported("expression-node bound exceeded")

        def visit(node: ast.AST) -> Any:
            if isinstance(node, ast.Name):
                return (
                    aliases[node.id]
                    if aliases is not None and node.id in aliases
                    else self.symbol(node.id)
                )
            if isinstance(node, ast.Constant):
                # Preserve exact literal type, not an inferred algebraic identity.
                return ["number", type(node.value).__name__, node.value]
            if isinstance(node, ast.Call):
                return ["call", node.func.id, *[visit(arg) for arg in node.args]]
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.BinOp):
                return [type(node.op).__name__, visit(node.left), visit(node.right)]
            if isinstance(node, ast.UnaryOp):
                return [type(node.op).__name__, visit(node.operand)]
            raise _Unsupported("unsupported expression node")

        return visit(tree)

    def initial(self, state: str) -> Any:
        rule = self.child.graph.rules.get(state)
        if rule is None:
            return None
        value = rule.initial
        if value.mode == "known":
            return {"mode": "known", "value": value.value}
        if value.mode == "value":
            return {"mode": "value", "scope": value.scope, "role": value.role}
        aliases = {
            alias: ["state_initial", self.symbol(target)]
            for alias, target in value.state_initials.items()
        }
        aliases.update(
            {
                alias: ["public_initial", target]
                for alias, target in value.public_initials.items()
            }
        )
        aliases.update(
            {
                parameter.name: _Ref(self.initial_parameters[(state, parameter.name)])
                for parameter in value.parameters
            }
        )
        return {
            "mode": "map",
            "expression": self.expression(value.expression, aliases=aliases),
        }

    def build(self) -> Any:
        candidate = self.child.base_candidate
        fixed_definitions = []
        equations = {
            equation.state: equation.rhs for equation in candidate.state_equations
        }
        initials = {item.state: item for item in candidate.initial_conditions}
        for kind, field in (
            ("state", "states"),
            ("process", "processes"),
            ("parameter", "parameters"),
        ):
            for item in getattr(candidate, field):
                descriptor = item.model_dump(
                    mode="json", exclude={"name", "description"}
                )
                descriptor["type"] = kind
                if "mechanisms" in descriptor:
                    descriptor["mechanisms"] = sorted(descriptor["mechanisms"])
                if kind == "state":
                    descriptor["rhs"] = self.expression(equations[item.name])
                    descriptor["initial"] = self.initial(item.name)
                    # Latent placeholders are explicitly overridden by the graph.
                    if descriptor["initial"] is None:
                        base = initials.get(item.name)
                        descriptor["base_initial"] = (
                            base.model_dump(mode="json", exclude={"state"})
                            if base
                            else None
                        )
                        if base is not None and base.expression is not None:
                            descriptor["base_initial"]["expression"] = self.expression(
                                base.expression
                            )
                elif kind == "process":
                    descriptor["expression"] = self.expression(item.expression)
                if item.name in self.global_names:
                    self.nodes[self.global_names[item.name]] = descriptor
                else:
                    fixed_definitions.append([item.name, descriptor])
        return {
            "context": self.child.context.model_dump(mode="json"),
            "fixed_symbols": sorted(self.fixed),
            "definitions": sorted(fixed_definitions, key=lambda row: row[0]),
            "outputs": sorted(
                [
                    [item.channel, item.unit, self.expression(item.expression)]
                    for item in candidate.observation_mappings
                ],
                key=lambda row: row[0],
            ),
            "constraints": sorted(
                [
                    {
                        **item.model_dump(
                            mode="json", exclude={"description", "subject"}
                        ),
                        "subject": self.symbol(item.subject),
                    }
                    for item in candidate.constraints
                ],
                # Constraints may involve private symbols; preserve declared order
                # rather than using uncanonicalized symbol spelling to sort them.
                key=lambda row: (row["kind"], row["source"], row["enforcement"]),
            ),
        }


def family_identity(
    parent: PrefixFitProblem | dict, candidate: PrefixFitProblem | dict
) -> dict[str, Any]:
    """Identify an exact private-renaming family of a statically valid model.

    Existing parent identities and public channels stay anchored. New state,
    process and parameter names may change only through one consistent bijection.
    Initial-map aliases are resolved; map coefficients remain separate scoped
    nodes. Fitted starting guesses are intentionally excluded, fixed values are
    retained. A deferred identity must never be treated as an equal family.
    """
    try:
        parent = PrefixFitProblem.model_validate(parent)
        candidate = PrefixFitProblem.model_validate(candidate)
        compiled = compile_candidate(candidate.base_candidate, candidate.context)
        apply_initialization_graph(compiled, candidate.graph)
        graph = _ModelGraph(parent, candidate)
        root = graph.build()
        canonical, labels, permutations = _canonical(root, graph.nodes)
        names = {
            name: f"$private:{labels[key]}$" for name, key in graph.global_names.items()
        }
        for (state, name), key in graph.initial_parameters.items():
            names[f"init_{state}_{name}"] = f"$private:{labels[key]}$"
        for state, rule in candidate.graph.rules.items():
            if rule.initial.mode == "value" and state in graph.global_names:
                names[f"init_{state}_value"] = f"initial_{names[state]}"
        return {
            "policy": POLICY,
            "status": "identified",
            "sha256": content_hash({"policy": POLICY, "canonical": canonical}),
            "private_symbol_mapping": names,
            "private_symbols": len(graph.nodes),
            "canonical_labelings": permutations,
            "limits": dict(LIMITS),
        }
    except (
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        ModelValidationError,
    ) as error:
        return {
            "policy": POLICY,
            "status": "deferred",
            "sha256": None,
            "reason": f"{type(error).__name__}: {error}"[:1000],
            "private_symbol_mapping": {},
            "limits": dict(LIMITS),
        }


def diagnostic_identity(
    diagnostic: dict, *, private_symbol_mapping: dict[str, str] | None = None
) -> str:
    """Hash a failure with witnesses, excluding only named provenance fields.

    Optional names must come from the same identified complete candidate family.
    They normalize identifier tokens, not arbitrary substrings; public labels and
    witness values remain. Comparing this hash alone does not establish repeated
    scientific failure: the caller must also compare the complete family.
    """
    names = private_symbol_mapping or {}

    def text(value: str) -> str:
        return re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            lambda match: names.get(match[0], match[0]),
            value,
        )

    def normalized(value: Any, field: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                text(key): normalized(item, key)
                for key, item in value.items()
                if key not in _PROVENANCE_FIELDS
            }
        if isinstance(value, list):
            if field in _SYMBOL_SET_FIELDS and all(
                isinstance(item, str) for item in value
            ):
                # These producers explicitly serialize symbol sets as sorted
                # strings. Renaming can reverse their lexical order. Witness
                # trajectory/value vectors and every unlisted array retain order.
                return sorted(text(item) for item in value)
            return [normalized(item) for item in value]
        if isinstance(value, str) and names:
            return text(value)
        return value

    return content_hash({"policy": POLICY, "diagnostic": normalized(diagnostic)})
