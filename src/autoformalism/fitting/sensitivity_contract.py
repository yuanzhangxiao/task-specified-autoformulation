"""Small structural certificates for differentiable restricted expressions.

This is not a general symbolic simplifier or an event-aware derivative engine.
Only the identities below may enclose otherwise unsupported kinked primitives.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from autoformalism.expressions import CompiledModel
from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.expressions.parser import ParsedExpression


class SensitivityContractError(ValueError):
    """The candidate needs a feature outside this numerical adapter's contract."""


def _same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _number(node.operand)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _call(node: ast.AST, names: set[str]) -> bool:
    return isinstance(node, ast.Call) and node.func.id in names


def _product_factors(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _product_factors(node.left) + _product_factors(node.right)
    if isinstance(node, ast.UnaryOp):
        sign = 1 if isinstance(node.op, ast.UAdd) else -1
        return [ast.Constant(sign), *_product_factors(node.operand)]
    return [node]


def _bounded_tree(tree: ast.AST) -> None:
    # Process substitution can grow exponentially despite bounded source trees.
    for index, _ in enumerate(ast.walk(tree)):
        if index >= 32768:
            raise SensitivityContractError(
                "sensitivity expression exceeds expanded structural audit budget"
            )


class _Expand(ast.NodeTransformer):
    def __init__(self, processes: dict[str, ast.AST]):
        self.processes = processes

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return self.processes.get(node.id, node)


@dataclass
class SmoothnessAudit:
    """Certify a few complete composites, without changing their function values."""

    location: str = ""
    c1_only: bool = False
    rhs_c1_only: bool = False
    certificates: list[dict] = field(default_factory=list)
    allow_piecewise: bool = False
    piecewise: list[dict] = field(default_factory=list)
    rhs_piecewise: bool = False

    def record(self, node: ast.AST, rule: str, *, c1_only: bool = False) -> None:
        self.c1_only |= c1_only
        self.certificates.append(
            {
                "location": self.location,
                "expression": ast.unparse(node)[:512],
                "rule": rule,
                "c1_only": c1_only,
            }
        )

    def visit(self, node: ast.AST) -> ast.AST:
        """Return a fresh certified tree; never mutate the candidate's source AST."""
        if isinstance(node, ast.Expression):
            return ast.Expression(self.visit(node.body))
        if isinstance(node, ast.Name | ast.Constant):
            return node
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(node.op, self.visit(node.operand))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div) and self._softsign(node):
                # The guarded denominator is >= epsilon everywhere. CasADi's
                # first AD derivative at zero is correct for the complete ratio.
                return node
            if isinstance(node.op, ast.Pow) and self._smooth_power(node):
                argument = node.left.args[0]
                if _call(node.left, {"abs"}):
                    return ast.BinOp(self.visit(argument), node.op, node.right)
                arguments = [self.visit(arg) for arg in node.left.args]
                return ast.BinOp(
                    ast.Call(node.left.func, arguments, []), node.op, node.right
                )
            return ast.BinOp(self.visit(node.left), node.op, self.visit(node.right))
        if isinstance(node, ast.Call):
            name = node.func.id
            if name in {"abs", "min", "max"}:
                values = [_number(arg) for arg in node.args]
                if all(value is not None for value in values):
                    operation = {"abs": lambda *v: abs(v[0]), "min": min, "max": max}
                    self.record(node, "constant_primitive")
                    return ast.Constant(operation[name](*values))
                if name in {"min", "max"} and all(
                    _same(node.args[0], arg) for arg in node.args[1:]
                ):
                    self.record(node, "identical_branches")
                    return self.visit(node.args[0])
                if self.allow_piecewise:
                    arguments = [self.visit(arg) for arg in node.args]
                    self.piecewise.append(
                        {
                            "location": self.location,
                            "expression": ast.unparse(node)[:512],
                            "primitive": name,
                        }
                    )
                    self.c1_only = True  # Never request exact second derivatives.
                    return ast.Call(node.func, arguments, [])
                raise SensitivityContractError(
                    f"{self.location}: uncertified nonsmooth {name} crossing; "
                    "no classical derivative is assigned at abs(0) or unequal "
                    "min/max branch ties; event/generalized sensitivities unsupported"
                )
            if name == "softplus":
                # The existing stable abs/max implementation has correct first
                # AD derivatives, but AD of it twice is not correct at zero.
                self.record(node, "stable_softplus_first_derivative", c1_only=True)
            return ast.Call(node.func, [self.visit(arg) for arg in node.args], [])
        raise AssertionError("restricted parser admitted unsupported syntax")

    def _softsign(self, node: ast.BinOp) -> bool:
        denominator = node.right
        if not isinstance(denominator, ast.BinOp) or not isinstance(
            denominator.op, ast.Add
        ):
            return False
        for constant, magnitude in (
            (denominator.left, denominator.right),
            (denominator.right, denominator.left),
        ):
            offset = _number(constant)
            if (
                offset is None
                or offset < SAFE_DIVISION_EPSILON
                or not _call(magnitude, {"abs"})
            ):
                continue
            argument = magnitude.args[0]
            factors = _product_factors(node.left)
            if not any(_same(factor, argument) for factor in factors):
                continue
            # Validate every factor, including the repeated argument, so this
            # certificate cannot hide another kink or an unsupported function.
            for factor in factors:
                self.visit(factor)
            self.record(node, "weighted_softsign_positive_literal", c1_only=True)
            return True
        return False

    def _smooth_power(self, node: ast.BinOp) -> bool:
        exponent = _number(node.right)
        if not _call(node.left, {"abs", "min", "max"}):
            return False
        if node.left.func.id == "abs" and exponent and exponent % 2 == 0:
            self.record(node, "even_power_of_abs")
            return True
        if exponent != 2 or not _call(node.left, {"min", "max"}):
            return False
        if len(node.left.args) == 2 and any(
            _number(arg) == 0 for arg in node.left.args
        ):
            self.record(node, "squared_rectifier", c1_only=True)
            return True
        return False


def certify_expressions(
    model: CompiledModel,
    *,
    allow_piecewise: bool = False,
) -> tuple[dict[str, ParsedExpression], dict[str, ParsedExpression], SmoothnessAudit]:
    """Resolve process aliases before checking complete equations and mappings."""
    validated = model.validated
    processes: dict[str, ast.AST] = {}
    audit = SmoothnessAudit(allow_piecewise=allow_piecewise)

    def expanded(expression: ParsedExpression) -> ast.Expression:
        tree = _Expand(processes).visit(deepcopy(expression.tree))
        _bounded_tree(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Pow)
                and (not isinstance(node.right, ast.Constant) or node.right.value < 0)
            ):
                raise SensitivityContractError(
                    "probe declines negative or variable powers"
                )
            if _call(node, {"sqrt", "log"}):
                raise SensitivityContractError(
                    f"sensitivity domain not certified for {node.func.id}; "
                    "positive-domain and boundary derivative proof required"
                )
        return tree

    for name in validated.process_order:
        processes[name] = expanded(validated.process_expressions[name]).body

    def certify(
        expressions: Mapping[str, ParsedExpression], kind: str
    ) -> dict[str, ParsedExpression]:
        result = {}
        for name, expression in expressions.items():
            audit.location = f"{kind}:{name}"
            tree = audit.visit(expanded(expression))
            result[name] = ParsedExpression(expression.source, tree, expression.symbols)
        return result

    equations = certify(validated.equation_expressions, "equation")
    audit.rhs_c1_only = audit.c1_only
    audit.rhs_piecewise = bool(audit.piecewise)
    observations = certify(validated.observation_expressions, "observation")
    return equations, observations, audit
