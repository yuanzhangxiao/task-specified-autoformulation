"""Convert an LLM-SR program into the expression grammar the evaluator parses.

LLM-SR does program synthesis: what it evolves is the body of a Python
function, not a symbolic expression. A body may legitimately contain
assignments, numpy calls, helper functions, loops or conditionals, while the
frozen evaluator parses a single restricted arithmetic expression over public
channels.

This converts the bodies that compute one expression of the current inputs and
refuses the rest by name, so the cost of our restricted grammar is a counted
number rather than an impression. Refusing is not a judgement about the
method: a loop over timesteps is a perfectly good discovery that our evaluator
cannot score.

Nothing here executes the program. A small reader walks the body's syntax
tree, statement by statement, and knows a fixed set of constructs:
assignments, the parameter vector and copies of it, helper functions the body
defines, loops over a known range, and branches whose condition is known
without the data: the length of ``params``, the shapes of the inputs, or a
fitted coefficient. Every other construct is refused by name. What the reader
returns is one expression for the restricted parser.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from autoformalism.expressions.baseline_functions import NUMPY_BASELINE_FUNCTIONS
from autoformalism.expressions.parser import BASELINE_FUNCTION_ARITY

#: Their specification template fixes the parameter vector length.
MAX_NPARAMS = 10

#: numpy spellings that have an approved counterpart in our grammar.
NUMPY_EQUIVALENTS = {
    "abs": "abs",
    "absolute": "abs",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "maximum": "max",
    "minimum": "min",
}

#: Named constants a program may read as ``np.pi`` or ``math.e``; they are
#: written out as numbers, which is what they are. An infinity is read, as a
#: clip bound for instance, but refused if it reaches the expression.
NAMED_CONSTANTS = {"pi": math.pi, "e": math.e, "inf": math.inf}

#: Bounds on reading one body, so that no program can make the reader run
#: long, recurse without end, or build an expression too large to write out.
_MAX_STEPS = 50_000
_MAX_CALL_DEPTH = 32
_MAX_SEQUENCE = 1_000
_MAX_NODES = 5_000
_MAX_DEPTH = 120

#: Python operators the grammar writes as operators, mirroring the parser.
_BINARY = frozenset({ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow})

#: How two known numbers combine, as numpy combines them.
_FOLD = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
    ast.FloorDiv: np.floor_divide,
}
_INTEGER_FOLD = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

#: Builtins a body may call; anything else it names must be defined in it.
_BUILTINS = frozenset({
    "abs", "bool", "enumerate", "float", "int", "len", "list", "max", "min",
    "pow", "range", "round", "sum", "tuple", "zip",
})

#: Modules a body may import, by the name the reader knows them under.
_MODULES = {"numpy": "np", "math": "math"}

#: numpy calls that return their argument's values: conversions to an array
#: or to float64, and copies. `nan_to_num` changes only values that are not
#: finite; a program that relied on that would show as a gap between LLM-SR's
#: score and the refitted error, which the driver records.
_IDENTITY_CALLS = frozenset({
    "array", "asanyarray", "asarray", "ascontiguousarray", "atleast_1d", "copy",
    "double", "float32", "float64", "nan_to_num",
})
_IDENTITY_KEYWORDS = frozenset({"dtype", "copy", "order", "nan", "posinf", "neginf"})

#: Reductions: over a vector they combine its items; over the data they would
#: combine rows, which no equation of the current inputs can.
_REDUCTIONS = {
    "sum": "sum", "nansum": "sum", "mean": "mean", "nanmean": "mean",
    "average": "mean", "prod": "prod", "max": "max", "amax": "max",
    "nanmax": "max", "min": "min", "amin": "min", "nanmin": "min",
}

#: Methods read on a vector, on a data array, or on a known number.
_VECTOR_METHODS = frozenset({
    "astype", "clip", "copy", "flatten", "item", "max", "mean", "min", "prod",
    "ravel", "sum", "tolist",
})
_DATA_METHODS = frozenset({"astype", "clip", "copy"})
_SCALAR_METHODS = frozenset({"astype", "copy", "item"})


class InexpressibleProgram(ValueError):
    """A synthesized program has no equivalent in the restricted grammar."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}{f': {detail}' if detail else ''}")
        self.reason = reason
        self.detail = detail


class CoefficientBranch(InexpressibleProgram):
    """The body branches on a coefficient that is still to be fitted.

    Read with fitted values, such a body is one expression; read with symbolic
    coefficients it is not yet one, so the caller fits it first.
    """


@dataclass(frozen=True)
class ProgramConversion:
    """The converted expression plus what it was found to depend on."""

    expression: str
    used_parameters: tuple[int, ...]
    used_inputs: tuple[str, ...]


class _Vector:
    """A sequence of values: ``params``, a copy or slice of it, or a list.

    ``origins`` names the entry of ``params`` each item came from, so reading
    an item records which coefficients the expression depends on. Arrays and
    lists differ in arithmetic: ``+`` adds arrays elementwise but concatenates
    lists, and ``*`` repeats a list. As with numpy arrays, an item assignment
    changes the vector in place, which every name bound to it sees; a slice is
    read as a copy.
    """

    def __init__(self, items, origins=None, *, array: bool) -> None:
        self.items = list(items)
        self.origins = (
            list(origins) if origins is not None else [None] * len(self.items)
        )
        self.array = array

    def copy(self, *, array: bool | None = None) -> _Vector:
        """A new vector with the same items."""
        return _Vector(
            self.items, self.origins, array=self.array if array is None else array
        )


@dataclass(frozen=True)
class _Function:
    """A helper the body defines with ``def`` or ``lambda``; inlined per call."""

    node: ast.FunctionDef | ast.Lambda
    scope: _Scope
    defaults: tuple[Any, ...]


@dataclass(frozen=True)
class _Module:
    """``np`` or ``math``: only its functions and named constants are read."""

    name: str


@dataclass(frozen=True)
class _Named:
    """A function named by a builtin, a module attribute or an import."""

    name: str
    origin: str


@dataclass(frozen=True)
class _Method:
    """A method looked up on a value, such as ``params.copy``."""

    owner: Any
    name: str


@dataclass(frozen=True)
class _NotFinite:
    """``np.isnan(x)`` or ``np.isinf(x)`` of an expression, or ``isfinite``.

    True only where ``x`` is not finite (with ``negated``, only where it is).
    The one use read is a guard, ``np.where(np.isnan(x), 0.0, x)``, which
    leaves every finite value as it is, as ``nan_to_num`` does.
    """

    value: ast.expr
    negated: bool = False


@dataclass(frozen=True)
class _Missing:
    """A coefficient the fit did not produce; reading it is refused."""

    index: int


class _Marker:
    """A value known only by its kind."""

    def __init__(self, description: str) -> None:
        self.description = description

    def __repr__(self) -> str:
        return self.description


#: The shape every input array has, so any two of them compare equal.
_SHAPE = _Marker("the shape of the inputs")
#: The number of rows, which only the data knows.
_ROWS = _Marker("the number of rows")
#: What a block gives back when it did not return.
_NO_RETURN = _Marker("no return")
#: ``np.finfo(...)``, read only for its constants.
_FINFO = _Marker("np.finfo")


class _Scope:
    """Names bound in one function body, falling back to the enclosing ones."""

    def __init__(self, parent: _Scope | None = None) -> None:
        self.names: dict[str, Any] = {}
        self.parent = parent

    def lookup(self, name: str) -> tuple[bool, Any]:
        """Whether the name is bound here or in an enclosing scope, and to what."""
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.names:
                return True, scope.names[name]
            scope = scope.parent
        return False, None


class _Undecided(Exception):
    """A decision only the data, or a coefficient still to be fitted, can make."""

    def __init__(self, *, on_data: bool) -> None:
        super().__init__("on the data" if on_data else "on a fitted coefficient")
        self.on_data = on_data


def _is_number(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_scalar(value: Any) -> bool:
    """A known number, or an expression of the inputs and coefficients."""
    return _is_number(value) or isinstance(value, ast.expr)


def _source(node: ast.AST) -> str:
    return ast.unparse(node)[:80]


def _constant(value: Any) -> ast.expr:
    """A known number as a grammar constant; a negative one as a negation.

    Writing ``-0.5`` as a negated literal keeps ``(-0.5) ** 2`` from being
    read back as ``-(0.5 ** 2)``.
    """
    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    number = int(value) if _is_integer(value) else float(value)
    if not math.isfinite(number):
        raise InexpressibleProgram("a constant that is not finite", repr(number))
    literal = ast.Constant(value=abs(number))
    return ast.UnaryOp(op=ast.USub(), operand=literal) if number < 0 else literal


def _as_expression(value: Any) -> ast.expr:
    return _constant(value) if _is_number(value) else value


def _call_node(name: str, arguments: list[ast.expr]) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()), args=arguments, keywords=[]
    )


def _names(node: ast.expr) -> frozenset[str]:
    """Every name in an expression, visiting each shared subtree once."""
    seen: set[int] = set()
    names: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ast.Name):
            names.add(current.id)
        stack.extend(ast.iter_child_nodes(current))
    return frozenset(names)


def _measure(node: ast.expr) -> tuple[int, int]:
    """Node count and depth as written out, a shared subtree once per use."""
    sizes: dict[int, tuple[int, int]] = {}
    stack: list[tuple[ast.AST, bool]] = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if id(current) in sizes:
            continue
        children = list(ast.iter_child_nodes(current))
        if not expanded:
            stack.append((current, True))
            stack.extend((child, False) for child in children)
            continue
        sizes[id(current)] = (
            1 + sum(sizes[id(child)][0] for child in children),
            1 + max((sizes[id(child)][1] for child in children), default=0),
        )
    return sizes[id(node)]


def _fold(op: ast.operator, left: Any, right: Any, node: ast.AST) -> Any:
    """Combine two known numbers as the program would: integers exactly,
    everything else as numpy float64."""
    left = int(left) if isinstance(left, (bool, np.bool_)) else left
    right = int(right) if isinstance(right, (bool, np.bool_)) else right
    if _is_integer(left) and _is_integer(right):
        if type(op) in _INTEGER_FOLD and not (
            isinstance(op, (ast.FloorDiv, ast.Mod)) and right == 0
        ):
            return _INTEGER_FOLD[type(op)](int(left), int(right))
        if isinstance(op, ast.Pow) and 0 <= right <= 64 and abs(left) <= 2**32:
            return int(left) ** int(right)
    function = _FOLD.get(type(op))
    if function is None:
        raise InexpressibleProgram("an operator outside the grammar", _source(node))
    with np.errstate(all="ignore"):
        return function(np.float64(left), np.float64(right))


class _Reader:
    """Read one evolved body into an expression, executing none of it.

    ``inputs`` maps the function's argument names to public channels.
    ``parameters`` gives each coefficient a fitted value; ``None`` leaves each
    one a symbol to be fitted.
    """

    def __init__(
        self, inputs: dict[str, str], parameters: dict[int, float] | None
    ) -> None:
        self.inputs = inputs
        self.channels = frozenset(inputs.values())
        self.parameters = parameters
        self.read_parameters: set[int] = set()
        self.steps = 0
        self.depth = 0

    def read(self, statements: list[ast.stmt]) -> ast.expr:
        """The expression the body returns."""
        try:
            value = self._block(statements, self._function_scope())
        except _Undecided as exc:
            raise self._refusal(exc, "a comparison", "") from None
        if value is _NO_RETURN:
            raise InexpressibleProgram("body never returns a value")
        if value is None:
            raise InexpressibleProgram("returns nothing")
        if isinstance(value, _Vector):
            raise InexpressibleProgram("returns several values")
        if not _is_scalar(value):
            raise InexpressibleProgram(
                "returns a value outside the grammar", repr(value)[:80]
            )
        expression = _as_expression(value)
        count, depth = _measure(expression)
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise InexpressibleProgram(
                "the expression is too large to write out",
                f"{count} nodes, depth {depth}",
            )
        return expression

    # --- scopes and bookkeeping ---------------------------------------------

    def _function_scope(self) -> _Scope:
        """The evolved function's own scope, inside its module's."""
        module = _Scope()
        module.names.update({"np": _Module("np"), "MAX_NPARAMS": MAX_NPARAMS})
        scope = _Scope(module)
        scope.names["params"] = self._parameter_vector()
        for argument, channel in self.inputs.items():
            scope.names[argument] = ast.Name(id=channel, ctx=ast.Load())
        return scope

    def _parameter_vector(self) -> _Vector:
        items: list[Any] = []
        for index in range(MAX_NPARAMS):
            if self.parameters is None:
                items.append(ast.Name(id=parameter_symbol(index), ctx=ast.Load()))
            elif index in self.parameters:
                items.append(np.float64(self.parameters[index]))
            else:
                items.append(_Missing(index))
        return _Vector(items, range(MAX_NPARAMS), array=True)

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise InexpressibleProgram(
                "too long to read", f"more than {_MAX_STEPS} steps"
            )

    def _refusal(
        self, exc: _Undecided, construct: str, detail: str
    ) -> InexpressibleProgram:
        if exc.on_data:
            return InexpressibleProgram(f"{construct} on the data", detail)
        return CoefficientBranch(f"{construct} on a fitted coefficient", detail)

    def _on_data(self, value: Any) -> bool:
        """Whether a value depends on the rows of the data."""
        if value is _ROWS or value is _SHAPE or isinstance(value, _NotFinite):
            return True
        if isinstance(value, ast.expr):
            return bool(_names(value) & self.channels)
        if isinstance(value, _Vector):
            return any(self._on_data(item) for item in value.items)
        return False

    # --- statements -----------------------------------------------------------

    def _block(self, statements: list[ast.stmt], scope: _Scope) -> Any:
        for statement in statements:
            self._tick()
            result = self._statement(statement, scope)
            if result is not _NO_RETURN:
                return result
        return _NO_RETURN

    def _statement(self, statement: ast.stmt, scope: _Scope) -> Any:
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise InexpressibleProgram("returns nothing")
            return self._expr(statement.value, scope)
        if isinstance(statement, ast.Assign):
            value = self._expr(statement.value, scope)
            for target in statement.targets:
                self._bind(target, value, scope)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self._bind(statement.target, self._expr(statement.value, scope), scope)
        elif isinstance(statement, ast.AugAssign):
            current = self._expr(statement.target, scope)
            value = self._expr(statement.value, scope)
            self._bind(
                statement.target,
                self._binary(statement.op, current, value, statement),
                scope,
            )
        elif isinstance(statement, ast.If):
            taken = (
                statement.body
                if self._decide(statement.test, scope, "a branch")
                else statement.orelse
            )
            return self._block(taken, scope)
        elif isinstance(statement, ast.For):
            return self._loop(statement, scope)
        elif isinstance(statement, ast.FunctionDef):
            scope.names[statement.name] = self._function(statement, scope)
        elif isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            pass  # a docstring
        elif isinstance(statement, ast.Pass):
            pass
        elif isinstance(statement, ast.Assert):
            self._check(statement, scope)
        elif isinstance(statement, ast.Raise):
            raise InexpressibleProgram("raises an exception", _source(statement))
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            self._import(statement, scope)
        elif isinstance(statement, ast.With):
            return self._with(statement, scope)
        elif isinstance(statement, (ast.While, ast.AsyncFor)):
            raise InexpressibleProgram("a loop", _source(statement))
        elif isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            raise InexpressibleProgram("exception handling", _source(statement))
        else:
            raise InexpressibleProgram(
                "statement has no expression equivalent", _source(statement)
            )
        return _NO_RETURN

    def _loop(self, statement: ast.For, scope: _Scope) -> Any:
        """A loop over a known sequence, read once per item."""
        if statement.orelse:
            raise InexpressibleProgram("a loop with an else clause", _source(statement))
        sequence = self._expr(statement.iter, scope)
        for item in self._sequence(sequence, statement.iter, "a loop"):
            self._bind(statement.target, item, scope)
            result = self._block(statement.body, scope)
            if result is not _NO_RETURN:
                return result
        return _NO_RETURN

    def _check(self, statement: ast.Assert, scope: _Scope) -> None:
        """An assertion held when the program scored; one known to fail cannot have."""
        try:
            holds = self._decide(statement.test, scope, "an assertion")
        except InexpressibleProgram:
            return
        if not holds:
            raise InexpressibleProgram("an assertion that fails", _source(statement))

    def _import(self, statement: ast.Import | ast.ImportFrom, scope: _Scope) -> None:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name not in _MODULES:
                    raise InexpressibleProgram(
                        "imports a module other than numpy or math", alias.name
                    )
                scope.names[alias.asname or alias.name] = _Module(_MODULES[alias.name])
            return
        module = _MODULES.get(statement.module or "")
        if module is None or statement.level:
            raise InexpressibleProgram(
                "imports from a module other than numpy or math", _source(statement)
            )
        for alias in statement.names:
            if alias.name == "*":
                raise InexpressibleProgram("a star import", _source(statement))
            value = NAMED_CONSTANTS.get(alias.name)
            scope.names[alias.asname or alias.name] = (
                _Named(alias.name, module) if value is None else value
            )

    def _with(self, statement: ast.With, scope: _Scope) -> Any:
        """``with np.errstate(...)`` changes warnings, not values."""
        if len(statement.items) == 1 and statement.items[0].optional_vars is None:
            context = statement.items[0].context_expr
            if isinstance(context, ast.Call):
                callee = self._expr(context.func, scope)
                if callee == _Named("errstate", "np"):
                    return self._block(statement.body, scope)
        raise InexpressibleProgram("a with block", _source(statement))

    def _function(
        self, node: ast.FunctionDef | ast.Lambda, scope: _Scope
    ) -> _Function:
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise InexpressibleProgram("a decorated helper", node.name)
        arguments = node.args
        if (
            arguments.vararg
            or arguments.kwarg
            or arguments.kwonlyargs
            or arguments.posonlyargs
        ):
            raise InexpressibleProgram(
                "a helper with variable or keyword-only arguments", _source(node)
            )
        defaults = tuple(self._expr(item, scope) for item in arguments.defaults)
        return _Function(node, scope, defaults)

    # --- assignment -----------------------------------------------------------

    def _bind(self, target: ast.expr, value: Any, scope: _Scope) -> None:
        if isinstance(target, ast.Name):
            scope.names[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if any(isinstance(item, ast.Starred) for item in target.elts):
                raise InexpressibleProgram(
                    "an assignment that unpacks with a star", _source(target)
                )
            items = self._unpack(value, len(target.elts), target)
            for element, item in zip(target.elts, items, strict=True):
                self._bind(element, item, scope)
        elif isinstance(target, ast.Subscript):
            self._store(target, value, scope)
        else:
            raise InexpressibleProgram("assignment is not to a name", _source(target))

    def _unpack(self, value: Any, count: int, node: ast.AST) -> list[Any]:
        if isinstance(value, _Vector) and len(value.items) == count:
            return [self._item(value, position) for position in range(count)]
        if self._on_data(value):
            raise InexpressibleProgram("unpacks the data rows", _source(node))
        raise InexpressibleProgram(
            "unpacks a value that is not a sequence of that length", _source(node)
        )

    def _store(self, target: ast.Subscript, value: Any, scope: _Scope) -> None:
        """``p[i] = v`` or ``p[a:b] = values``: a vector changed in place."""
        container = self._expr(target.value, scope)
        if not isinstance(container, _Vector):
            if self._on_data(container):
                raise InexpressibleProgram("writes into the data rows", _source(target))
            raise InexpressibleProgram(
                "writes into a value that is not a vector", _source(target)
            )
        if isinstance(target.slice, ast.Slice):
            positions = self._slice(container, target.slice, scope)
            if isinstance(value, _Vector):
                pairs = list(zip(value.items, value.origins, strict=True))
                if len(pairs) == 1:
                    pairs = pairs * len(positions)
                if len(pairs) != len(positions):
                    raise InexpressibleProgram(
                        "a slice assignment of a different length", _source(target)
                    )
            else:
                self._require_item(value, target)
                pairs = [(value, None)] * len(positions)
            for position, (item, origin) in zip(positions, pairs, strict=True):
                container.items[position] = item
                container.origins[position] = origin
            return
        index = self._expr(target.slice, scope)
        position = self._position(container, index, target)
        self._require_item(value, target)
        container.items[position] = value
        container.origins[position] = None

    def _require_item(self, value: Any, node: ast.AST) -> None:
        """A vector holds numbers or expressions of coefficients, as an array does."""
        if _is_number(value) or (
            isinstance(value, ast.expr) and not self._on_data(value)
        ):
            return
        if self._on_data(value):
            raise InexpressibleProgram("stores the data in a vector", _source(node))
        raise InexpressibleProgram(
            "stores a value that is not a number in a vector", _source(node)
        )

    # --- vectors --------------------------------------------------------------

    def _slice(self, vector: _Vector, node: ast.Slice, scope: _Scope) -> list[int]:
        bounds: list[int | None] = []
        for part in (node.lower, node.upper, node.step):
            value = None if part is None else self._expr(part, scope)
            if value is not None and not _is_integer(value):
                raise InexpressibleProgram(
                    "a slice bound that is not a known integer", _source(node)
                )
            bounds.append(None if value is None else int(value))
        if bounds[2] == 0:
            raise InexpressibleProgram("a slice with a zero step", _source(node))
        return list(range(len(vector.items)))[slice(*bounds)]

    def _position(self, vector: _Vector, index: Any, node: ast.AST) -> int:
        if not _is_integer(index):
            if self._on_data(index):
                raise InexpressibleProgram("indexes by the data", _source(node))
            raise InexpressibleProgram("index is not a known integer", _source(node))
        position = int(index)
        if position < 0:
            position += len(vector.items)
        if not 0 <= position < len(vector.items):
            raise InexpressibleProgram(
                "index is beyond the declared vector", _source(node)
            )
        return position

    def _item(self, vector: _Vector, position: int) -> Any:
        item = vector.items[position]
        if isinstance(item, _Missing):
            raise InexpressibleProgram(
                "uses a parameter the fit did not produce", f"params[{item.index}]"
            )
        origin = vector.origins[position]
        if origin is not None:
            self.read_parameters.add(origin)
        return item

    def _sequence(self, value: Any, node: ast.AST, construct: str) -> list[Any]:
        """The items a loop or a comprehension visits."""
        if isinstance(value, _Vector):
            return [self._item(value, position) for position in range(len(value.items))]
        if self._on_data(value):
            raise InexpressibleProgram(f"{construct} over the rows", _source(node))
        raise InexpressibleProgram(
            f"{construct} over a value that is not a sequence", _source(node)
        )

    # --- expressions ----------------------------------------------------------

    def _expr(self, node: ast.expr, scope: _Scope) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str)):
                return node.value
            raise InexpressibleProgram("a constant outside the grammar", _source(node))
        if isinstance(node, ast.Name):
            found, value = scope.lookup(node.id)
            if found:
                return value
            if node.id in _BUILTINS:
                return _Named(node.id, "builtin")
            raise InexpressibleProgram("unknown name", node.id)
        if isinstance(node, ast.BinOp):
            left = self._expr(node.left, scope)
            return self._binary(node.op, left, self._expr(node.right, scope), node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node, scope)
        if isinstance(node, ast.Compare):
            return self._compare(node, scope)
        if isinstance(node, ast.BoolOp):
            return self._boolean(node, scope)
        if isinstance(node, ast.IfExp):
            chosen = (
                node.body
                if self._decide(node.test, scope, "a conditional expression")
                else node.orelse
            )
            return self._expr(chosen, scope)
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        if isinstance(node, ast.Attribute):
            return self._attribute(node, scope)
        if isinstance(node, ast.Subscript):
            return self._subscript(node, scope)
        if isinstance(node, (ast.Tuple, ast.List)):
            if any(isinstance(item, ast.Starred) for item in node.elts):
                raise InexpressibleProgram("a sequence with a star", _source(node))
            return _Vector([self._expr(item, scope) for item in node.elts], array=False)
        if isinstance(node, ast.Lambda):
            return self._function(node, scope)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self._comprehension(node, scope)
        if isinstance(node, ast.JoinedStr):
            return ""  # an f-string, which a body only ever uses as a message
        raise InexpressibleProgram("a construct outside the grammar", _source(node))

    def _binary(self, op: ast.operator, left: Any, right: Any, node: ast.AST) -> Any:
        if isinstance(left, _NotFinite) or isinstance(right, _NotFinite):
            return self._either_not_finite(op, left, right, node)
        if isinstance(left, _Vector) or isinstance(right, _Vector):
            return self._vector_binary(op, left, right, node)
        if _is_number(left) and _is_number(right):
            return _fold(op, left, right, node)
        if not (_is_scalar(left) and _is_scalar(right)):
            raise InexpressibleProgram(
                "arithmetic on a value outside the grammar", _source(node)
            )
        a, b = _as_expression(left), _as_expression(right)
        if type(op) in _BINARY:
            return ast.BinOp(left=a, op=type(op)(), right=b)
        if isinstance(op, ast.Mod):
            return _call_node("mod", [a, b])
        if isinstance(op, ast.FloorDiv):
            return _call_node("floor", [ast.BinOp(left=a, op=ast.Div(), right=b)])
        raise InexpressibleProgram("an operator outside the grammar", _source(node))

    def _either_not_finite(
        self, op: ast.operator, left: Any, right: Any, node: ast.AST
    ) -> _NotFinite:
        """``np.isnan(x) | np.isinf(x)``: true where ``x`` is not finite."""
        if (
            isinstance(op, ast.BitOr)
            and isinstance(left, _NotFinite)
            and isinstance(right, _NotFinite)
            and left.value is right.value
            and not left.negated
            and not right.negated
        ):
            return left
        raise InexpressibleProgram("a test of the data", _source(node))

    def _vector_binary(
        self, op: ast.operator, left: Any, right: Any, node: ast.AST
    ) -> _Vector:
        if isinstance(left, _Vector) and isinstance(right, _Vector):
            if not left.array and not right.array:
                if isinstance(op, ast.Add):
                    return _Vector(
                        left.items + right.items,
                        left.origins + right.origins,
                        array=False,
                    )
                raise InexpressibleProgram("arithmetic on lists", _source(node))
            if len(left.items) != len(right.items):
                raise InexpressibleProgram(
                    "arithmetic on vectors of different lengths", _source(node)
                )
            items = [
                self._binary(op, a, b, node)
                for a, b in zip(left.items, right.items, strict=True)
            ]
            origins = [
                a if a is not None else b
                for a, b in zip(left.origins, right.origins, strict=True)
            ]
            return _Vector(items, origins, array=True)
        vector, other = (left, right) if isinstance(left, _Vector) else (right, left)
        if not vector.array:
            if isinstance(op, ast.Mult) and _is_integer(other):
                count = max(int(other), 0)
                if count * len(vector.items) > _MAX_SEQUENCE:
                    raise InexpressibleProgram("a list too long to read", _source(node))
                return _Vector(
                    vector.items * count, vector.origins * count, array=False
                )
            raise InexpressibleProgram("arithmetic on a list", _source(node))
        if not _is_scalar(other) or self._on_data(other):
            raise InexpressibleProgram(
                "arithmetic between a vector and the data", _source(node)
            )
        items = [
            self._binary(op, item, other, node)
            if vector is left
            else self._binary(op, other, item, node)
            for item in vector.items
        ]
        return _Vector(items, vector.origins, array=True)

    def _unary(self, node: ast.UnaryOp, scope: _Scope) -> Any:
        value = self._expr(node.operand, scope)
        if isinstance(node.op, ast.Not):
            return not self._truth(value)
        if isinstance(node.op, ast.Invert) and isinstance(value, _NotFinite):
            return _NotFinite(value.value, not value.negated)
        if isinstance(value, _Vector) and value.array:
            return _Vector(
                [self._sign(node.op, item, node) for item in value.items],
                value.origins,
                array=True,
            )
        return self._sign(node.op, value, node)

    def _sign(self, op: ast.unaryop, value: Any, node: ast.AST) -> Any:
        if isinstance(op, ast.UAdd) and _is_scalar(value):
            return value
        if isinstance(op, ast.USub):
            if _is_number(value):
                return -value
            if isinstance(value, ast.expr):
                return ast.UnaryOp(op=ast.USub(), operand=value)
        raise InexpressibleProgram("an operator outside the grammar", _source(node))

    def _truth(self, value: Any) -> bool:
        """Python's truth value; ``_Undecided`` when only the data or a fit can tell."""
        if value is None:
            return False
        if _is_number(value) or isinstance(value, str):
            return bool(value)
        if isinstance(value, _Vector):
            if not value.array:
                return bool(value.items)
            if len(value.items) == 1:
                return self._truth(value.items[0])
            raise InexpressibleProgram("the truth value of an array")
        if isinstance(value, (ast.expr, _NotFinite)) or value is _ROWS:
            raise _Undecided(on_data=self._on_data(value))
        if isinstance(value, (_Function, _Module, _Named, _Method)):
            return True
        raise InexpressibleProgram(
            "a truth value outside the grammar", repr(value)[:80]
        )

    def _decide(self, test: ast.expr, scope: _Scope, construct: str) -> bool:
        try:
            return self._truth(self._expr(test, scope))
        except _Undecided as exc:
            raise self._refusal(exc, construct, _source(test)) from None

    def _compare(self, node: ast.Compare, scope: _Scope) -> bool:
        left = self._expr(node.left, scope)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self._expr(comparator, scope)
            if not self._compare_pair(op, left, right, node):
                return False
            left = right
        return True

    def _compare_pair(
        self, op: ast.cmpop, left: Any, right: Any, node: ast.AST
    ) -> bool:
        if isinstance(op, (ast.Is, ast.IsNot)):
            if left is None or right is None:
                same = left is None and right is None
            elif isinstance(left, (_Vector, _Function)) or isinstance(
                right, (_Vector, _Function)
            ):
                same = left is right
            else:
                raise InexpressibleProgram(
                    "an identity comparison outside the grammar", _source(node)
                )
            return same if isinstance(op, ast.Is) else not same
        if isinstance(op, (ast.In, ast.NotIn)):
            raise InexpressibleProgram("a membership test", _source(node))
        if _is_number(left) and _is_number(right):
            return bool(_COMPARE[type(op)](left, right))
        if left is right and (left is _SHAPE or left is _ROWS):
            # Every input array has the same shape and length.
            return isinstance(op, (ast.Eq, ast.LtE, ast.GtE))
        if (
            isinstance(left, _Vector)
            and isinstance(right, _Vector)
            and all(_is_number(item) for item in left.items + right.items)
        ):
            return bool(_COMPARE[type(op)](tuple(left.items), tuple(right.items)))
        if isinstance(left, str) and isinstance(right, str):
            return bool(_COMPARE[type(op)](left, right))
        if (_is_scalar(left) or left is _ROWS) and (
            _is_scalar(right) or right is _ROWS
        ):
            raise _Undecided(on_data=self._on_data(left) or self._on_data(right))
        raise InexpressibleProgram("a comparison outside the grammar", _source(node))

    def _boolean(self, node: ast.BoolOp, scope: _Scope) -> Any:
        """``and`` and ``or`` return an operand, as Python's do."""
        conjunction = isinstance(node.op, ast.And)
        value: Any = None
        for operand in node.values:
            value = self._expr(operand, scope)
            if self._truth(value) != conjunction:
                return value
        return value

    def _comprehension(
        self, node: ast.ListComp | ast.GeneratorExp, scope: _Scope
    ) -> _Vector:
        if len(node.generators) != 1 or node.generators[0].is_async:
            raise InexpressibleProgram("a nested comprehension", _source(node))
        generator = node.generators[0]
        sequence = self._expr(generator.iter, scope)
        items = self._sequence(sequence, generator.iter, "a comprehension")
        local = _Scope(scope)
        results: list[Any] = []
        for item in items:
            self._bind(generator.target, item, local)
            if all(
                self._decide(condition, local, "a comprehension filter")
                for condition in generator.ifs
            ):
                results.append(self._expr(node.elt, local))
        return _Vector(results, array=False)

    def _attribute(self, node: ast.Attribute, scope: _Scope) -> Any:
        owner = self._expr(node.value, scope)
        name = node.attr
        if isinstance(owner, _Module):
            if name in NAMED_CONSTANTS:
                return NAMED_CONSTANTS[name]
            return _Named(name, owner.name)
        if owner is _FINFO and name in {"eps", "tiny", "max", "resolution"}:
            return getattr(np.finfo(np.float64), name)
        if isinstance(owner, _Vector):
            if name == "size":
                return len(owner.items)
            if name == "shape":
                return _Vector([len(owner.items)], array=False)
            if name == "ndim":
                return 1
            if name == "T":
                return owner
            if name in _VECTOR_METHODS:
                return _Method(owner, name)
        elif self._on_data(owner):
            if name == "shape":
                return _SHAPE
            if name == "size":
                return _ROWS
            if name == "ndim":
                return 1
            if name in _DATA_METHODS:
                return _Method(owner, name)
        elif _is_scalar(owner):
            if name == "real":
                return owner
            if name in _SCALAR_METHODS:
                return _Method(owner, name)
        raise InexpressibleProgram("an attribute access", _source(node))

    def _subscript(self, node: ast.Subscript, scope: _Scope) -> Any:
        container = self._expr(node.value, scope)
        if isinstance(container, _Vector):
            if isinstance(node.slice, ast.Slice):
                positions = self._slice(container, node.slice, scope)
                return _Vector(
                    [container.items[index] for index in positions],
                    [container.origins[index] for index in positions],
                    array=container.array,
                )
            index = self._expr(node.slice, scope)
            return self._item(container, self._position(container, index, node))
        if self._on_data(container):
            raise InexpressibleProgram("indexes the data rows", _source(node))
        raise InexpressibleProgram(
            "indexes a value that is not a vector", _source(node)
        )

    # --- calls ----------------------------------------------------------------

    def _call(self, node: ast.Call, scope: _Scope) -> Any:
        callee = self._expr(node.func, scope)
        if any(isinstance(item, ast.Starred) for item in node.args) or any(
            item.arg is None for item in node.keywords
        ):
            raise InexpressibleProgram("a call with unpacked arguments", _source(node))
        arguments = [self._expr(item, scope) for item in node.args]
        keywords = {item.arg: self._expr(item.value, scope) for item in node.keywords}
        if isinstance(callee, _Function):
            return self._inline(callee, arguments, keywords, node)
        if isinstance(callee, _Method):
            return self._method(callee, arguments, keywords, node)
        if isinstance(callee, _Named):
            return self._named(callee, arguments, keywords, node)
        raise InexpressibleProgram("call is not a plain function", _source(node))

    def _keywords(self, keywords: dict, allowed: frozenset[str], node: ast.AST) -> None:
        unexpected = sorted(set(keywords) - allowed)
        if unexpected:
            raise InexpressibleProgram(
                "call uses keyword arguments",
                f"{_source(node)} ({', '.join(unexpected)})",
            )

    def _inline(
        self, function: _Function, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        """A helper's result for these arguments, read in a scope of its own."""
        if self.depth >= _MAX_CALL_DEPTH:
            raise InexpressibleProgram("helper calls nested too deeply", _source(node))
        names = [item.arg for item in function.node.args.args]
        if len(arguments) > len(names):
            raise InexpressibleProgram(
                "a helper called with too many arguments", _source(node)
            )
        bound = dict(zip(names, arguments, strict=False))
        for name, value in keywords.items():
            if name not in names or name in bound:
                raise InexpressibleProgram(
                    "a helper called with an unknown or repeated argument",
                    _source(node),
                )
            bound[name] = value
        defaulted = names[len(names) - len(function.defaults):]
        for name, default in zip(defaulted, function.defaults, strict=True):
            bound.setdefault(name, default)
        if len(bound) != len(names):
            raise InexpressibleProgram(
                "a helper called without all its arguments", _source(node)
            )
        local = _Scope(function.scope)
        local.names.update(bound)
        self.depth += 1
        try:
            if isinstance(function.node, ast.Lambda):
                return self._expr(function.node.body, local)
            result = self._block(function.node.body, local)
            return None if result is _NO_RETURN else result
        finally:
            self.depth -= 1

    def _method(
        self, method: _Method, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        owner, name = method.owner, method.name
        if name in {"astype", "copy", "flatten", "ravel"}:
            self._keywords(keywords, frozenset({"dtype", "copy", "order"}), node)
            return owner.copy() if isinstance(owner, _Vector) else owner
        if name == "tolist" and isinstance(owner, _Vector):
            return owner.copy(array=False)
        if name == "item" and not arguments and not keywords:
            if not isinstance(owner, _Vector):
                return owner
            if len(owner.items) == 1:
                return self._item(owner, 0)
        if name == "clip":
            return self._clip(owner, arguments, keywords, node)
        if name in _REDUCTIONS and isinstance(owner, _Vector):
            if arguments or keywords:
                raise InexpressibleProgram("a reduction with an axis", _source(node))
            return self._reduce(_REDUCTIONS[name], owner, node)
        raise InexpressibleProgram("an attribute access", _source(node))

    def _named(
        self, callee: _Named, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        name, origin = callee.name, callee.origin
        if origin == "builtin":
            return self._builtin(name, arguments, keywords, node)
        if origin == "np":
            if name in _IDENTITY_CALLS:
                self._keywords(keywords, _IDENTITY_KEYWORDS, node)
                return self._identity(name, arguments, keywords, node)
            if name in {"zeros", "ones", "empty", "full"}:
                return self._filled(name, arguments, keywords, node)
            if name in {"zeros_like", "ones_like", "empty_like", "full_like"}:
                return self._filled_like(name, arguments, keywords, node)
            if name in _REDUCTIONS:
                return self._reduction(name, arguments, keywords, node)
            if name in {"size", "shape", "ndim"} and len(arguments) == 1:
                return self._dimension(name, arguments[0], node)
            if name == "take" and len(arguments) == 2 and set(keywords) <= {"mode"}:
                return self._take(arguments[0], arguments[1], keywords, node)
            if name in {"isclose", "allclose"} and len(arguments) == 2:
                return self._close(arguments[0], arguments[1], keywords, node)
            if name == "dot" and len(arguments) == 2 and not keywords:
                return self._dot(arguments[0], arguments[1], node)
            if name == "where" and len(arguments) == 3 and not keywords:
                return self._where(*arguments, node)
            if name == "clip" and arguments:
                return self._clip(arguments[0], arguments[1:], keywords, node)
            if name == "finfo":
                return _FINFO
        if name in {"isfinite", "isnan", "isinf"} and len(arguments) == 1:
            return self._finite(name, arguments[0], node)
        if name == "logical_or" and len(arguments) == 2 and not keywords:
            return self._binary(ast.BitOr(), arguments[0], arguments[1], node)
        if name in {"power", "pow"} and len(arguments) == 2 and not keywords:
            return self._binary(ast.Pow(), arguments[0], arguments[1], node)
        approved = NUMPY_EQUIVALENTS.get(name, name)
        if approved not in BASELINE_FUNCTION_ARITY:
            raise InexpressibleProgram("function is not approved", name)
        if keywords:
            raise InexpressibleProgram("call uses keyword arguments", name)
        return self._apply(approved, arguments, node)

    def _builtin(
        self, name: str, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        if keywords and not (name == "enumerate" and set(keywords) == {"start"}):
            raise InexpressibleProgram("call uses keyword arguments", name)
        if name in {"max", "min"}:
            return self._extreme(name, arguments, node)
        if name == "range":
            return self._range(arguments, node)
        if name == "zip":
            return self._zip(arguments, node)
        if name == "pow" and len(arguments) == 2:
            return self._binary(ast.Pow(), arguments[0], arguments[1], node)
        if name == "enumerate" and len(arguments) == 1:
            return self._enumerate(arguments[0], keywords.get("start", 0), node)
        if len(arguments) != 1 and not (name == "round" and len(arguments) == 2):
            raise InexpressibleProgram(
                f"{name}() with unexpected arguments", _source(node)
            )
        value = arguments[0]
        if name == "len":
            if isinstance(value, _Vector):
                return len(value.items)
            if self._on_data(value):
                return _ROWS
        elif name == "bool":
            return self._truth(value)
        elif name in {"float", "int"}:
            if _is_number(value):
                return float(value) if name == "float" else int(value)
            if (
                name == "float"
                and isinstance(value, ast.expr)
                and not self._on_data(value)
            ):
                return value
        elif name == "abs":
            if _is_number(value):
                return abs(value)
            return self._apply("abs", [value], node)
        elif name == "sum":
            return self._reduction("sum", arguments, keywords, node)
        elif name == "round":
            if all(_is_number(item) for item in arguments):
                return round(value, *[int(item) for item in arguments[1:]])
        elif name in {"list", "tuple"} and isinstance(value, _Vector):
            return value.copy(array=False)
        raise InexpressibleProgram(
            f"{name}() of a value outside the grammar", _source(node)
        )

    def _identity(
        self, name: str, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        if not arguments or len(arguments) > 2:
            raise InexpressibleProgram(
                "a conversion with unexpected arguments", _source(node)
            )
        value = arguments[0]
        if isinstance(value, _Vector):
            copies = name in {"array", "copy"} or keywords.get("copy") is True
            return value.copy(array=True) if copies or not value.array else value
        if _is_scalar(value):
            return value
        raise InexpressibleProgram(
            "a conversion of a value outside the grammar", _source(node)
        )

    def _filled(
        self, name: str, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        """``np.zeros(n)`` and kin: a vector of one value, or that value broadcast."""
        self._keywords(keywords, frozenset({"dtype", "order"}), node)
        if not arguments or (name == "full" and len(arguments) < 2):
            raise InexpressibleProgram("an array of unknown length", _source(node))
        shape = arguments[0]
        fill = {"zeros": 0.0, "empty": 0.0, "ones": 1.0}.get(name)
        if fill is None:
            fill = arguments[1]
            self._require_item(fill, node)
        if isinstance(shape, _Vector) and len(shape.items) == 1:
            shape = shape.items[0]
        if _is_integer(shape) and 0 <= shape <= _MAX_SEQUENCE:
            return _Vector([fill] * int(shape), array=True)
        if shape is _ROWS:
            # As long as the data, of one value: that value, broadcast.
            return fill
        raise InexpressibleProgram("an array of unknown length", _source(node))

    def _filled_like(
        self, name: str, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        self._keywords(keywords, frozenset({"dtype", "order"}), node)
        if not arguments or (name == "full_like" and len(arguments) < 2):
            raise InexpressibleProgram("an array of unknown fill", _source(node))
        like = arguments[0]
        fill = {"zeros_like": 0.0, "empty_like": 0.0, "ones_like": 1.0}.get(name)
        if fill is None:
            fill = arguments[1]
            self._require_item(fill, node)
        if isinstance(like, _Vector):
            return _Vector([fill] * len(like.items), array=True)
        if _is_scalar(like):
            return fill
        raise InexpressibleProgram(
            "an array like a value outside the grammar", _source(node)
        )

    def _dimension(self, name: str, value: Any, node: ast.AST) -> Any:
        if isinstance(value, _Vector):
            return {
                "size": len(value.items),
                "shape": _Vector([len(value.items)], array=False),
                "ndim": 1,
            }[name]
        if self._on_data(value):
            return {"size": _ROWS, "shape": _SHAPE, "ndim": 1}[name]
        raise InexpressibleProgram(
            f"np.{name} of a value outside the grammar", _source(node)
        )

    def _range(self, arguments: list, node: ast.AST) -> _Vector:
        if not 1 <= len(arguments) <= 3:
            raise InexpressibleProgram(
                "range() with unexpected arguments", _source(node)
            )
        if any(self._on_data(item) for item in arguments):
            raise InexpressibleProgram("a range over the rows", _source(node))
        if not all(_is_integer(item) for item in arguments):
            raise InexpressibleProgram(
                "a range whose bounds are not known integers", _source(node)
            )
        values = range(*[int(item) for item in arguments])
        if len(values) > _MAX_SEQUENCE:
            raise InexpressibleProgram("a range too long to read", _source(node))
        return _Vector(list(values), array=False)

    def _enumerate(self, value: Any, start: Any, node: ast.AST) -> _Vector:
        if not isinstance(value, _Vector) or not _is_integer(start):
            if self._on_data(value):
                raise InexpressibleProgram("enumerates the rows", _source(node))
            raise InexpressibleProgram(
                "enumerates a value that is not a vector", _source(node)
            )
        return _Vector(
            [
                _Vector([int(start) + position, item], [None, origin], array=False)
                for position, (item, origin) in enumerate(
                    zip(value.items, value.origins, strict=True)
                )
            ],
            array=False,
        )

    def _zip(self, arguments: list, node: ast.AST) -> _Vector:
        if not arguments or not all(isinstance(item, _Vector) for item in arguments):
            if any(self._on_data(item) for item in arguments):
                raise InexpressibleProgram("zips the rows", _source(node))
            raise InexpressibleProgram(
                "zips a value that is not a vector", _source(node)
            )
        length = min(len(item.items) for item in arguments)
        return _Vector(
            [
                _Vector(
                    [item.items[position] for item in arguments],
                    [item.origins[position] for item in arguments],
                    array=False,
                )
                for position in range(length)
            ],
            array=False,
        )

    def _extreme(self, name: str, arguments: list, node: ast.AST) -> Any:
        """Python's ``max`` and ``min``, over arguments or over one vector."""
        if len(arguments) == 1:
            value = arguments[0]
            if isinstance(value, _Vector):
                items = [self._item(value, index) for index in range(len(value.items))]
            elif self._on_data(value):
                raise InexpressibleProgram("a reduction over the rows", _source(node))
            else:
                raise InexpressibleProgram(f"{name}() of a single value", _source(node))
        else:
            items = arguments
        if not items:
            raise InexpressibleProgram(f"{name}() of nothing", _source(node))
        if all(_is_number(item) for item in items):
            return (max if name == "max" else min)(items)
        return items[0] if len(items) == 1 else self._apply(name, items, node)

    def _reduction(
        self, name: str, arguments: list, keywords: dict, node: ast.AST
    ) -> Any:
        if keywords or len(arguments) != 1:
            raise InexpressibleProgram("a reduction with an axis", _source(node))
        value = arguments[0]
        if isinstance(value, _Vector):
            return self._reduce(_REDUCTIONS[name], value, node)
        if self._on_data(value):
            raise InexpressibleProgram("a reduction over the rows", _source(node))
        if _is_scalar(value):
            return value
        raise InexpressibleProgram(
            "a reduction of a value outside the grammar", _source(node)
        )

    def _reduce(self, kind: str, vector: _Vector, node: ast.AST) -> Any:
        items = [self._item(vector, index) for index in range(len(vector.items))]
        if not items:
            raise InexpressibleProgram("a reduction of an empty vector", _source(node))
        if kind in {"max", "min"}:
            return self._extreme(kind, items, node) if len(items) > 1 else items[0]
        step = ast.Mult() if kind == "prod" else ast.Add()
        total = items[0]
        for item in items[1:]:
            total = self._binary(step, total, item, node)
        if kind == "mean":
            total = self._binary(ast.Div(), total, len(items), node)
        return total

    def _dot(self, left: Any, right: Any, node: ast.AST) -> Any:
        if not (isinstance(left, _Vector) and isinstance(right, _Vector)):
            return self._binary(ast.Mult(), left, right, node)
        if len(left.items) != len(right.items) or not left.items:
            raise InexpressibleProgram(
                "a dot product of vectors of different lengths", _source(node)
            )
        total: Any = None
        for index in range(len(left.items)):
            product = self._binary(
                ast.Mult(), self._item(left, index), self._item(right, index), node
            )
            total = (
                product if total is None
                else self._binary(ast.Add(), total, product, node)
            )
        return total

    def _clip(self, value: Any, bounds: list, keywords: dict, node: ast.AST) -> Any:
        """``np.clip`` with an absent or infinite bound is a one-sided max or min."""
        named = dict(keywords)
        lower = named.pop("a_min", named.pop("min", bounds[0] if bounds else None))
        upper = named.pop(
            "a_max", named.pop("max", bounds[1] if len(bounds) > 1 else None)
        )
        if named or len(bounds) > 2:
            raise InexpressibleProgram(
                "clip() with unexpected arguments", _source(node)
            )
        if isinstance(value, _Vector):
            return _Vector(
                [self._clip(item, [lower, upper], {}, node) for item in value.items],
                value.origins,
                array=True,
            )
        if _is_number(lower) and float(lower) == -math.inf:
            lower = None
        if _is_number(upper) and float(upper) == math.inf:
            upper = None
        if lower is None and upper is None:
            return value
        if lower is None:
            return self._apply("min", [value, upper], node)
        if upper is None:
            return self._apply("max", [value, lower], node)
        return self._apply("clip", [value, lower, upper], node)

    def _finite(self, name: str, value: Any, node: ast.AST) -> Any:
        if _is_number(value):
            return bool(getattr(np, name)(value))
        if isinstance(value, ast.expr):
            if self._on_data(value):
                return _NotFinite(value, negated=name == "isfinite")
            raise _Undecided(on_data=False)
        raise InexpressibleProgram(
            f"np.{name} of a value outside the grammar", _source(node)
        )

    def _where(
        self, condition: Any, chosen: Any, otherwise: Any, node: ast.AST
    ) -> Any:
        """``np.where`` on a known condition, or as a guard against non-finite ones."""
        if isinstance(condition, _NotFinite):
            kept = chosen if condition.negated else otherwise
            if kept is condition.value:
                return kept
        try:
            taken = self._truth(condition)
        except _Undecided as exc:
            raise self._refusal(exc, "np.where", _source(node)) from None
        return chosen if taken else otherwise

    def _take(self, vector: Any, index: Any, keywords: dict, node: ast.AST) -> Any:
        """``np.take``: an item, with its mode's handling of an index out of range."""
        if not isinstance(vector, _Vector) or not _is_integer(index):
            raise InexpressibleProgram(
                "np.take of a value outside the grammar", _source(node)
            )
        mode = keywords.get("mode", "raise")
        length = len(vector.items)
        if mode == "clip":
            index = min(max(int(index), 0), length - 1)
        elif mode == "wrap":
            index = int(index) % length
        elif mode != "raise":
            raise InexpressibleProgram("np.take with an unknown mode", _source(node))
        return self._item(vector, self._position(vector, index, node))

    def _close(self, left: Any, right: Any, keywords: dict, node: ast.AST) -> bool:
        """``np.isclose`` of two numbers; of expressions, the data's or a fit's call."""
        self._keywords(keywords, frozenset({"rtol", "atol", "equal_nan"}), node)
        if (
            _is_number(left)
            and _is_number(right)
            and all(_is_number(value) for value in keywords.values())
        ):
            return bool(np.isclose(left, right, **keywords))
        if _is_scalar(left) and _is_scalar(right):
            raise _Undecided(on_data=self._on_data(left) or self._on_data(right))
        raise InexpressibleProgram(
            "np.isclose of a value outside the grammar", _source(node)
        )

    def _apply(self, name: str, arguments: list, node: ast.AST) -> Any:
        """An approved function: computed when every argument is known,
        elementwise over a vector, and otherwise written out."""
        low, high = BASELINE_FUNCTION_ARITY[name]
        if not low <= len(arguments) <= high:
            raise InexpressibleProgram(
                "function takes a different number of arguments",
                f"{name} with {len(arguments)}",
            )
        vectors = [item for item in arguments if isinstance(item, _Vector)]
        if vectors:
            length = len(vectors[0].items)
            if any(len(item.items) != length for item in vectors) or any(
                self._on_data(item)
                for item in arguments
                if not isinstance(item, _Vector)
            ):
                raise InexpressibleProgram(
                    "a function of a vector and the data", _source(node)
                )
            items = [
                self._apply(
                    name,
                    [
                        item.items[index] if isinstance(item, _Vector) else item
                        for item in arguments
                    ],
                    node,
                )
                for index in range(length)
            ]
            return _Vector(items, vectors[0].origins, array=True)
        if all(_is_number(item) for item in arguments):
            with np.errstate(all="ignore"):
                value = _NUMPY_FUNCTIONS[name](*arguments)
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value[()]
            if _is_number(value):
                return value
        if all(_is_scalar(item) for item in arguments):
            return _call_node(name, [_as_expression(item) for item in arguments])
        raise InexpressibleProgram(
            "a function of a value outside the grammar", _source(node)
        )


def _parse_body(body: str) -> list[ast.stmt]:
    try:
        return ast.parse(body.strip()).body
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise InexpressibleProgram("body does not parse", str(exc)) from exc


def convert_program(
    body: str,
    inputs: dict[str, str],
    parameters: dict[int, float] | None = None,
) -> ProgramConversion:
    """Turn one evolved function body into a restricted-grammar expression.

    ``inputs`` maps the synthesized function's argument names to our public
    channel names. ``parameters`` maps ``params`` indices to fitted values;
    passing ``None`` leaves each coefficient as a symbol, which is possible
    when the body does not branch on a coefficient (``CoefficientBranch``
    otherwise).
    """
    statements = _parse_body(body)
    reader = _Reader(inputs, parameters)
    try:
        tree = reader.read(statements)
        expression = ast.unparse(tree)
    except RecursionError:
        raise InexpressibleProgram("nested too deeply to read") from None
    arguments = {channel: argument for argument, channel in inputs.items()}
    return ProgramConversion(
        expression=expression,
        used_parameters=tuple(sorted(reader.read_parameters)),
        used_inputs=tuple(
            sorted(arguments[name] for name in _names(tree) if name in arguments)
        ),
    )


#: One node per operator, evaluated over numpy arrays. Defined here rather
#: than reached for with eval so that recovering a model from LLM-SR never
#: executes proposer-written code, whatever their own evaluator does.
_NUMPY_FUNCTIONS = {
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    # Reduced pairwise: np.maximum(a, b, c) would treat c as the output array.
    "max": NUMPY_BASELINE_FUNCTIONS["Max"],
    "min": NUMPY_BASELINE_FUNCTIONS["Min"],
    "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "softplus": lambda value: np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0),
    # The baseline grammar's wider vocabulary; the entries above are unchanged.
    **{
        name: function
        for name, function in NUMPY_BASELINE_FUNCTIONS.items()
        if name not in {"abs", "exp", "log", "sin", "sqrt", "tanh", "max", "min"}
    },
}


def parameter_symbol(index: int) -> str:
    """The name a fitted coefficient carries once params[i] is symbolic."""
    return f"p_{index}"


def _evaluate_tree(tree: ast.expr, bindings: dict[str, Any]) -> NDArray[np.float64]:
    """Evaluate a converted expression over numpy arrays, without exec.

    Only the node types the grammar approves are handled; anything else raises,
    so a program that slipped through conversion cannot execute here either. A
    subtree that several places share is evaluated once.
    """
    values: dict[int, Any] = {}

    def walk(node: ast.AST) -> Any:
        if id(node) in values:
            return values[id(node)]
        if isinstance(node, ast.Constant):
            result: Any = float(node.value)
        elif isinstance(node, ast.Name):
            if node.id not in bindings:
                raise InexpressibleProgram("unbound name", node.id)
            result = bindings[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)
        ):
            operand = walk(node.operand)
            result = -operand if isinstance(node.op, ast.USub) else operand
        elif isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            for kind, apply in (
                (ast.Add, np.add), (ast.Sub, np.subtract),
                (ast.Mult, np.multiply), (ast.Div, np.divide),
                (ast.Pow, np.power),
            ):
                if isinstance(node.op, kind):
                    result = apply(left, right)
                    break
            else:
                raise InexpressibleProgram("binary operator", ast.unparse(node))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _NUMPY_FUNCTIONS.get(node.func.id)
            if function is None:
                raise InexpressibleProgram("function is not approved", node.func.id)
            result = function(*[walk(item) for item in node.args])
        else:
            raise InexpressibleProgram("node has no numeric meaning", ast.unparse(node))
        values[id(node)] = result
        return result

    with np.errstate(all="ignore"):
        return np.asarray(walk(tree), dtype=float)


def evaluate_expression(
    expression: str, bindings: dict[str, Any]
) -> NDArray[np.float64]:
    """Evaluate a restricted expression over numpy arrays, without exec."""
    return _evaluate_tree(ast.parse(expression, mode="eval").body, bindings)


class _RefitExpired(Exception):
    """The refit ran past its time limit."""


def refit_program(
    body: str,
    inputs: dict[str, str],
    channels: dict[str, NDArray[np.float64]],
    target: NDArray[np.float64],
    *,
    seconds: float = 600.0,
) -> dict[int, float] | None:
    """Re-run the coefficient fit their specification performs and discards.

    Their ``evaluate`` fits all ``MAX_NPARAMS`` coefficients with BFGS from an
    all-ones start, minimizing the mean squared error on the training rows,
    and returns only the loss, so the values behind a score are not saved.
    This repeats that call on the same rows. Each trial vector is read through
    the program as their evaluator runs it, so a branch on a coefficient
    follows the trial values, as it does in theirs; nothing is executed.
    Reading is slower than their running, so the limit is wider than their
    30 seconds; a refit past it is refused rather than left to run.
    """
    statements = _parse_body(body)
    deadline = monotonic() + seconds

    def loss(values: NDArray[np.float64]) -> float:
        if monotonic() > deadline:
            raise _RefitExpired
        try:
            tree = _Reader(inputs, dict(enumerate(values))).read(statements)
            predicted = _evaluate_tree(tree, channels)
        except (InexpressibleProgram, RecursionError):
            return float("inf")
        with np.errstate(all="ignore"):
            return float(np.mean((predicted - target) ** 2))

    try:
        with np.errstate(all="ignore"):
            result = minimize(loss, [1.0] * MAX_NPARAMS, method="BFGS")
    except _RefitExpired:
        raise InexpressibleProgram(
            "the coefficient refit ran past its time limit", f"{seconds:g} s"
        ) from None
    if not np.isfinite(result.fun):
        return None
    return {index: float(value) for index, value in enumerate(result.x)}


def training_error(
    expression: str,
    channels: dict[str, NDArray[np.float64]],
    target: NDArray[np.float64],
) -> float:
    """Mean squared error of a converted expression on the training rows.

    The negative of LLM-SR's own score for the program, when the conversion
    and the refit reproduce what its evaluator ran; the driver records both.
    """
    predicted = evaluate_expression(expression, channels)
    with np.errstate(all="ignore"):
        return float(np.mean((predicted - target) ** 2))


def best_sample(log_dir: Path) -> dict | None:
    """The highest-scoring sample the profiler recorded.

    Their score is negative mean squared error, so larger is better. A sample
    whose score is absent never evaluated and is not a candidate.
    """
    best: dict | None = None
    directory = log_dir / "samples"
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("samples_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        score = payload.get("score")
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            continue
        if best is None or score > best["score"]:
            best = {**payload, "score": float(score)}
    return best


def sample_yield(log_dir: Path) -> dict[str, int]:
    """How many samples the model wrote, and how many of those scored.

    The profiler records the specification's own program as sample 0; every
    later sample is one the model wrote. A sample without a finite score never
    evaluated, which is what an empty or unreadable reply comes to.
    """
    written = scored = 0
    directory = log_dir / "samples"
    paths = directory.glob("samples_*.json") if directory.is_dir() else ()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        order = payload.get("sample_order")
        if not isinstance(order, int) or order < 1:
            continue
        written += 1
        score = payload.get("score")
        if isinstance(score, (int, float)) and np.isfinite(score):
            scored += 1
    return {"model_samples": written, "model_samples_scored": scored}


def equation_body(function_source: str) -> str:
    """The body of the evolved function, without its signature or docstring."""
    try:
        module = ast.parse(textwrap.dedent(function_source))
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise InexpressibleProgram("sample does not parse", str(exc)) from exc
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            statements = [
                item
                for item in node.body
                if not (
                    isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                )
            ]
            if not statements:
                raise InexpressibleProgram("evolved function has an empty body")
            return "\n".join(ast.unparse(item) for item in statements)
    raise InexpressibleProgram("sample contains no function definition")


#: Their published specifications open with a plain-language description of the
#: system, so supplying ours matches their protocol rather than adapting it.
SPECIFICATION_TEMPLATE = '''"""
{description}
"""

import numpy as np

#Initialize parameters
MAX_NPARAMS = {max_params}
params = [1.0]*MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""

    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    {unpack}

    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation({arguments}, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')

    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss


@equation.evolve
def equation({signature}, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for the time derivative of {target}

    Args:
{argument_docs}
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing the time derivative of {target}.
    """
    {initial} = {initial_expression}
    return {initial}
'''


def specification_variable(channel: str, index: int) -> str:
    """A Python identifier for one public channel, stable across a campaign."""
    cleaned = "".join(
        item if item.isalnum() or item == "_" else "_" for item in channel
    )
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"c_{cleaned}"
    return cleaned if cleaned.isidentifier() else f"x{index}"


def build_specification(
    description: str,
    channels: tuple[str, ...],
    target: str,
    *,
    max_params: int = MAX_NPARAMS,
) -> tuple[str, dict[str, str]]:
    """Render one Phase-B specification, and the variable to channel mapping.

    The specification is an input to LLM-SR, not part of it: their repository
    ships one per problem and the runner is pointed at a file. Writing ours is
    therefore unavoidable rather than a modification, and their own examples
    carry a description of the system, so including the public task text
    matches their protocol.
    """
    if target not in channels:
        raise ValueError(f"target {target!r} is not among the channels")
    if len(channels) + 1 > max_params:
        # The starting skeleton spends one parameter per channel plus an
        # intercept; past their cap it would index a parameter that is absent.
        raise ValueError(
            f"{len(channels)} channels need {len(channels) + 1} starting "
            f"parameters, beyond LLM-SR's {max_params}"
        )
    variables = [specification_variable(name, index)
                 for index, name in enumerate(channels)]
    if len(set(variables)) != len(variables):
        raise ValueError(f"channel names collide as identifiers: {channels}")
    unpack = ", ".join(variables) + " = " + ", ".join(
        f"inputs[:,{index}]" for index in range(len(variables))
    )
    argument_docs = "\n".join(
        f"        {variable}: A numpy array of observations of {name}."
        for variable, name in zip(variables, channels, strict=True)
    )
    # Named after the derivative it computes, as upstream names its `dv`.
    derivative = variables[channels.index(target)]
    specification = SPECIFICATION_TEMPLATE.format(
        description=description.strip(),
        max_params=max_params,
        unpack=unpack,
        arguments=", ".join(variables),
        signature=", ".join(f"{variable}: np.ndarray" for variable in variables),
        target=target,
        argument_docs=argument_docs,
        initial="d" + derivative,
        initial_expression=" + ".join(
            [
                f"params[{index}] * {variable}"
                for index, variable in enumerate(variables)
            ]
            + [f"params[{len(variables)}]"]
        ),
    )
    return specification, dict(zip(variables, channels, strict=True))
