"""Bounded formal algebra over restricted expressions, without symbolic execution.

Only sums, products and powers of monomials are expanded. Other expressions are
opaque factors. An unproved identity stays unverified, never a scientific failure.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.parser import RestrictedParser

Monomial = tuple[tuple[str, int], ...]
Polynomial = dict[Monomial, Fraction]
LIMIT = 512


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError("formal algebra coefficient budget exceeded")
    return value


def add(*values: Polynomial) -> Polynomial:
    """Collect exact rational coefficients with a bounded term inventory."""
    result: dict[Monomial, Fraction] = defaultdict(Fraction)
    for value in values:
        for key, coefficient in value.items():
            result[key] = _bounded(result[key] + coefficient)
        if len(result) > LIMIT:
            raise ValueError("formal algebra term budget exceeded")
    return {k: v for k, v in result.items() if v}


def scale(value: Polynomial, factor: Fraction | int) -> Polynomial:
    """Multiply by an exact scalar."""
    return {k: _bounded(v * factor) for k, v in value.items() if v * factor}


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    """Distribute products without unbounded expression expansion."""
    if len(left) * len(right) > LIMIT:
        raise ValueError("formal algebra product budget exceeded")
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            powers = dict(a)
            for name, power in b:
                powers[name] = powers.get(name, 0) + power
            if any(abs(p) > 1024 for p in powers.values()):
                raise ValueError("formal algebra exponent budget exceeded")
            key = tuple(sorted((n, p) for n, p in powers.items() if p))
            result = add(result, {key: _bounded(x * y)})
    return result


def atom(expression: str) -> Polynomial:
    """Represent a validated subexpression as one opaque factor."""
    return {((expression, 1),): Fraction(1)}


def formal(node: ast.AST) -> Polynomial:
    """Expand a validated AST, retaining unsupported identities as opaque factors."""
    if isinstance(node, ast.Constant):
        return {(): Fraction(str(node.value))} if node.value else {}
    if isinstance(node, ast.Name):
        return atom(node.id)
    if isinstance(node, ast.UnaryOp):
        return scale(formal(node.operand), -1 if isinstance(node.op, ast.USub) else 1)
    if isinstance(node, ast.BinOp):
        a, b = formal(node.left), formal(node.right)
        if isinstance(node.op, ast.Add):
            return add(a, b)
        if isinstance(node.op, ast.Sub):
            return add(a, scale(b, -1))
        if isinstance(node.op, ast.Mult):
            return multiply(a, b)
        if isinstance(node.op, ast.Div) and len(b) == 1:
            ((key, coefficient),) = b.items()
            return multiply(a, {tuple((n, -p) for n, p in key): 1 / coefficient})
        if isinstance(node.op, ast.Pow) and len(a) == 1:
            power = int(RestrictedParser._signed_number(node.right))
            ((key, coefficient),) = a.items()
            if max(
                coefficient.numerator.bit_length(), coefficient.denominator.bit_length()
            ) * abs(power) > 4096 or any(abs(p * power) > 1024 for _, p in key):
                raise ValueError("formal algebra power budget exceeded")
            return {
                tuple((n, p * power) for n, p in key if p * power): coefficient**power
            }
    return atom(ast.unparse(node))


@lru_cache(maxsize=4096)
def factor_symbols(expression: str) -> frozenset[str]:
    """Read dependencies through the same restricted parser as the fitter."""
    return (
        RestrictedParser(max_length=40000, max_nodes=4096)
        .parse(expression, location="basin audit factor")
        .symbols
    )


def symbols(value: Polynomial) -> set[str]:
    """Return dependencies after exact additive cancellation."""
    return {s for key in value for name, _ in key for s in factor_symbols(name)}


def coefficient(value: Polynomial, variable: str) -> Polynomial | None:
    """Extract a linear coefficient; nonlinear dependence is explicitly unknown."""
    result = {}
    for key, scalar in value.items():
        dependent = [(n, p) for n, p in key if variable in factor_symbols(n)]
        if not dependent:
            continue
        if dependent != [(variable, 1)]:
            return None
        result = add(result, {tuple((n, p) for n, p in key if n != variable): scalar})
    return result


def render(value: Polynomial) -> str:
    """Produce a deterministic human-readable identity, not executable Python."""
    terms = []
    for key, scalar in sorted(value.items()):
        factors = [f"({name})**({power})" for name, power in key]
        terms.append(" * ".join([f"({scalar})", *factors]))
    return " + ".join(terms) or "0"


def number(value: Polynomial, environment: dict[str, float]) -> float:
    """Evaluate validated factors using the production scalar interpreter."""
    total = 0.0
    for key, scalar in value.items():
        term = float(scalar)
        for name, power in key:
            parsed = RestrictedParser(max_length=40000, max_nodes=4096).parse(
                name, location="basin audit point"
            )
            term *= _evaluate(parsed, environment) ** power
        total += term
    return total


def expanded(expression: str, processes: dict[str, str]) -> ast.AST:
    """Inline algebraics, rejecting cycles and excessive expansion before allocation."""
    budget = 4096

    def walk(node, visiting):
        nonlocal budget
        budget -= 1
        if budget < 0:
            raise ValueError("algebraic expansion budget exceeded")
        if isinstance(node, ast.Name) and node.id in processes:
            if node.id in visiting:
                raise ValueError("algebraic cycle")
            return walk(parse(processes[node.id]), visiting | {node.id})
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                walk(node.left, visiting), node.op, walk(node.right, visiting)
            )
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(node.op, walk(node.operand, visiting))
        if isinstance(node, ast.Call):
            return ast.Call(node.func, [walk(a, visiting) for a in node.args], [])
        return node

    def parse(text):
        return RestrictedParser().parse(text, location="basin audit equation").tree.body

    return walk(parse(expression), set())
