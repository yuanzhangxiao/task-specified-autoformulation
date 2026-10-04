"""Restricted Python-expression parser with no dynamic code execution."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import dataclass

from autoformalism.expressions.baseline_functions import BASELINE_FUNCTIONS
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    ValidationDiagnostic,
)

APPROVED_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    "abs": (1, 1),
    "exp": (1, 1),
    "log": (1, 1),
    "max": (2, 64),
    "min": (2, 64),
    "sigmoid": (1, 1),
    "sin": (1, 1),
    "softplus": (1, 1),
    "sqrt": (1, 1),
    "tanh": (1, 1),
}

#: Our approved functions plus the external baselines' wider vocabulary. Only
#: ``baseline_parser`` admits it; the default grammar is unchanged.
BASELINE_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    **APPROVED_FUNCTION_ARITY,
    **{name: arity for name, (arity, _) in BASELINE_FUNCTIONS.items()},
}

_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    *_ALLOWED_BINARY_OPERATORS,
    *_ALLOWED_UNARY_OPERATORS,
)


@dataclass(frozen=True)
class ParsedExpression:
    """Validated syntax tree and referenced symbol names."""

    source: str
    tree: ast.Expression
    symbols: frozenset[str]


class RestrictedParser:
    """Parse a small analytic grammar and reject every other Python construct."""

    def __init__(
        self,
        *,
        max_length: int = 4096,
        max_nodes: int = 512,
        max_depth: int = 64,
        max_literal_magnitude: float = 1e12,
        max_integer_power: int = 16,
        functions: Mapping[str, tuple[int, int]] | None = None,
        real_powers: bool = False,
    ) -> None:
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_literal_magnitude = max_literal_magnitude
        self.max_integer_power = max_integer_power
        self.functions = dict(
            APPROVED_FUNCTION_ARITY if functions is None else functions
        )
        # Any exponent expression, evaluated over the reals. Off by default:
        # only the external-baseline grammar enables it.
        self.real_powers = real_powers

    def parse(self, source: str, *, location: str) -> ParsedExpression:
        """Return a validated expression or stable diagnostics."""
        diagnostics: list[ValidationDiagnostic] = []
        if len(source) > self.max_length:
            diagnostics.append(
                ValidationDiagnostic(
                    "EXPRESSION_TOO_LONG",
                    location,
                    f"expression exceeds {self.max_length} characters",
                )
            )
            raise ModelValidationError(tuple(diagnostics))
        try:
            parsed = ast.parse(source, mode="eval")
        except (SyntaxError, ValueError, MemoryError) as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    "SYNTAX_ERROR",
                    location,
                    f"expression cannot be parsed: {type(exc).__name__}",
                )
            )
            raise ModelValidationError(tuple(diagnostics)) from exc
        if not isinstance(parsed, ast.Expression):  # pragma: no cover
            raise AssertionError("expression parser returned an unexpected root")
        parsed = _TimeIndexedSymbolNormalizer(self.functions).visit(parsed)
        ast.fix_missing_locations(parsed)

        nodes = list(ast.walk(parsed))
        if len(nodes) > self.max_nodes:
            diagnostics.append(
                ValidationDiagnostic(
                    "EXPRESSION_TOO_COMPLEX",
                    location,
                    f"expression exceeds {self.max_nodes} AST nodes",
                )
            )
        depth = self._depth(parsed)
        if depth > self.max_depth:
            diagnostics.append(
                ValidationDiagnostic(
                    "EXPRESSION_TOO_DEEP",
                    location,
                    f"expression depth {depth} exceeds {self.max_depth}",
                )
            )

        symbols: set[str] = set()
        called_function_nodes = {
            id(node.func)
            for node in nodes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in nodes:
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                diagnostics.append(
                    ValidationDiagnostic(
                        "UNSUPPORTED_SYNTAX",
                        location,
                        f"node {type(node).__name__} is not allowed",
                    )
                )
                continue
            if isinstance(node, ast.Constant):
                self._validate_constant(node, location, diagnostics)
            elif isinstance(node, ast.Name):
                if not isinstance(node.ctx, ast.Load):
                    diagnostics.append(
                        ValidationDiagnostic(
                            "UNSUPPORTED_SYNTAX",
                            location,
                            "only symbol reads are allowed",
                        )
                    )
                elif node.id in self.functions and id(node) in called_function_nodes:
                    continue
                elif node.id in APPROVED_FUNCTION_ARITY:
                    diagnostics.append(
                        ValidationDiagnostic(
                            "FUNCTION_AS_VALUE",
                            location,
                            f"function {node.id!r} must be called",
                        )
                    )
                else:
                    # The baseline vocabulary names a function only where it is
                    # called: `gamma` is also a common rate parameter.
                    symbols.add(node.id)
            elif isinstance(node, ast.BinOp):
                if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
                    diagnostics.append(
                        ValidationDiagnostic(
                            "UNSUPPORTED_OPERATOR",
                            location,
                            f"operator {type(node.op).__name__} is not allowed",
                        )
                    )
                if isinstance(node.op, ast.Pow) and not self.real_powers:
                    self._validate_power(node, location, diagnostics)
            elif isinstance(node, ast.UnaryOp) and not isinstance(
                node.op, _ALLOWED_UNARY_OPERATORS
            ):
                diagnostics.append(
                    ValidationDiagnostic(
                        "UNSUPPORTED_OPERATOR",
                        location,
                        f"operator {type(node.op).__name__} is not allowed",
                    )
                )
            elif isinstance(node, ast.Call):
                self._validate_call(node, location, diagnostics)

        if diagnostics:
            raise ModelValidationError(tuple(diagnostics))
        return ParsedExpression(source, parsed, frozenset(symbols))

    def _validate_constant(
        self,
        node: ast.Constant,
        location: str,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_LITERAL",
                    location,
                    f"literal type {type(value).__name__} is not allowed",
                )
            )
            return
        numeric = float(value)
        if not math.isfinite(numeric) or abs(numeric) > self.max_literal_magnitude:
            diagnostics.append(
                ValidationDiagnostic(
                    "INVALID_NUMERIC_LITERAL",
                    location,
                    "numeric literal is nonfinite or exceeds the magnitude limit",
                )
            )

    def _validate_call(
        self,
        node: ast.Call,
        location: str,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        if not isinstance(node.func, ast.Name):
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_CALL",
                    location,
                    "function must be a direct approved name",
                )
            )
            return
        function_name = node.func.id
        if function_name not in self.functions:
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_FUNCTION",
                    location,
                    f"function {function_name!r} is not approved",
                )
            )
            return
        if node.keywords:
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_CALL",
                    location,
                    "keyword arguments are not allowed",
                )
            )
        minimum, maximum = self.functions[function_name]
        if not minimum <= len(node.args) <= maximum:
            diagnostics.append(
                ValidationDiagnostic(
                    "INVALID_FUNCTION_ARITY",
                    location,
                    f"{function_name} expects {minimum}..{maximum} arguments",
                )
            )

    def _validate_power(
        self,
        node: ast.BinOp,
        location: str,
        diagnostics: list[ValidationDiagnostic],
    ) -> None:
        exponent = self._signed_number(node.right)
        if exponent is None or not float(exponent).is_integer():
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_POWER",
                    location,
                    "power exponent must be an integer literal",
                )
            )
            return
        if abs(int(exponent)) > self.max_integer_power:
            diagnostics.append(
                ValidationDiagnostic(
                    "UNSUPPORTED_POWER",
                    location,
                    f"integer power magnitude exceeds {self.max_integer_power}",
                )
            )

    @staticmethod
    def _signed_number(node: ast.AST) -> float | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int | float)
            and not isinstance(node.operand.value, bool)
        ):
            value = float(node.operand.value)
            return -value if isinstance(node.op, ast.USub) else value
        return None

    @staticmethod
    def _depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(RestrictedParser._depth(child) for child in children)


class _TimeIndexedSymbolNormalizer(ast.NodeTransformer):
    """Normalize conventional ``symbol(t)`` notation into a symbol read."""

    def __init__(self, functions: Mapping[str, tuple[int, int]]) -> None:
        self.functions = functions

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Rewrite only a direct non-function name called with exactly ``t``."""
        node = self.generic_visit(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in self.functions
            and not node.keywords
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "t"
        ):
            return ast.copy_location(ast.Name(id=node.func.id, ctx=ast.Load()), node)
        return node


def baseline_parser() -> RestrictedParser:
    """The grammar an external baseline's model is read with.

    Wider than ours in vocabulary, exponents and size, never in kind: still
    a closed set of named functions over a fixed node set, with no execution.
    Fitted constants of any finite magnitude are accepted -- an Arrhenius
    prefactor is routinely beyond our own 1e12 bound -- and the size limits
    are raised so that a long regression model is not refused for length.
    """
    return RestrictedParser(
        max_length=65536,
        max_nodes=8192,
        max_depth=256,
        max_literal_magnitude=math.inf,
        functions=BASELINE_FUNCTION_ARITY,
        real_powers=True,
    )
