"""Convert an LLM-SR program into the expression grammar the evaluator parses.

LLM-SR does program synthesis: what it evolves is the body of a Python
function, not a symbolic expression. A body may legitimately contain
assignments, numpy calls, loops or conditionals, while the frozen evaluator
parses a single restricted arithmetic expression over public channels.

This converts the subset that has an expression equivalent and refuses the
rest by name, so the cost of our restricted grammar is a counted number rather
than an impression. Refusing is not a judgement about the method: a loop over
timesteps is a perfectly good discovery that our evaluator cannot score.

Nothing here executes the program. The body is parsed, its assignments are
inlined by substitution, and the result is handed to the restricted parser.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY

#: numpy spellings that have an approved counterpart in our grammar.
NUMPY_EQUIVALENTS = {
    "abs": "abs",
    "absolute": "abs",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "tanh": "tanh",
    "maximum": "max",
    "minimum": "min",
}

#: Python operators the grammar accepts, mirroring the restricted parser.
_BINARY = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**",
}


class InexpressibleProgram(ValueError):
    """A synthesized program has no equivalent in the restricted grammar."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}{f': {detail}' if detail else ''}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ProgramConversion:
    """The converted expression plus what had to be assumed to get there."""

    expression: str
    used_parameters: tuple[int, ...]
    used_inputs: tuple[str, ...]


def _format_number(value: float) -> str:
    """Render a fitted coefficient without losing the value to rounding."""
    return repr(float(value))


class _Inliner(ast.NodeTransformer):
    """Replace local names by their assigned expressions, and params by values."""

    def __init__(
        self,
        bindings: dict[str, ast.expr],
        parameters: dict[int, float],
        inputs: dict[str, str],
    ) -> None:
        self.bindings = bindings
        self.parameters = parameters
        self.inputs = inputs
        self.used_parameters: set[int] = set()
        self.used_inputs: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """``params[3]`` becomes the fitted number it stood for."""
        # Deliberately not generic_visit first: that would descend into the
        # `params` name, which is not a channel and would be refused.
        if not isinstance(node.value, ast.Name):
            raise InexpressibleProgram("indexes a computed value",
                                       ast.unparse(node))
        if node.value.id not in {"params", "parameters"}:
            raise InexpressibleProgram("indexes something other than params",
                                       ast.unparse(node))
        index = node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
            raise InexpressibleProgram("parameter index is not a literal",
                                       ast.unparse(node))
        if index.value not in self.parameters:
            raise InexpressibleProgram(
                "uses a parameter the fit did not produce", ast.unparse(node)
            )
        self.used_parameters.add(index.value)
        return ast.copy_location(
            ast.Constant(value=float(self.parameters[index.value])), node
        )

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Inline a local assignment, or rename an input to its channel."""
        if node.id in self.bindings:
            return self.visit(ast.copy_location(self.bindings[node.id], node))
        if node.id in self.inputs:
            self.used_inputs.add(node.id)
            return ast.copy_location(ast.Name(id=self.inputs[node.id],
                                              ctx=ast.Load()), node)
        raise InexpressibleProgram("unknown name", node.id)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Keep only calls with an approved counterpart."""
        name = _called_name(node.func)
        if name is None:
            raise InexpressibleProgram("call is not a plain function",
                                       ast.unparse(node))
        approved = NUMPY_EQUIVALENTS.get(name, name)
        if approved not in APPROVED_FUNCTION_ARITY:
            raise InexpressibleProgram("function is not approved", name)
        if node.keywords:
            raise InexpressibleProgram("call uses keyword arguments", name)
        arguments = [self.visit(item) for item in node.args]
        return ast.copy_location(
            ast.Call(func=ast.Name(id=approved, ctx=ast.Load()),
                     args=arguments, keywords=[]),
            node,
        )


def _called_name(func: ast.expr) -> str | None:
    """``np.exp`` and ``exp`` both name ``exp``; anything else names nothing."""
    if isinstance(func, ast.Name):
        return func.id
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in {"np", "numpy", "math"}
    ):
        return func.attr
    return None


def _check_supported(tree: ast.AST) -> None:
    """Refuse constructs with no expression equivalent, naming the construct."""
    unsupported = {
        ast.For: "a loop", ast.While: "a loop", ast.If: "a branch",
        ast.IfExp: "a conditional expression", ast.ListComp: "a comprehension",
        ast.GeneratorExp: "a comprehension", ast.Lambda: "a lambda",
        ast.Compare: "a comparison", ast.BoolOp: "a boolean operator",
        ast.Attribute: "an attribute access", ast.Try: "exception handling",
    }
    # `np.tanh` is an attribute, but a supported one when it is what a call
    # names; only attributes used as values are rejected here.
    resolved_call_targets = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node.func) is not None
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) in resolved_call_targets:
            continue
        for kind, description in unsupported.items():
            if isinstance(node, kind):
                raise InexpressibleProgram(description, ast.unparse(node)[:80])


def convert_program(
    body: str,
    inputs: dict[str, str],
    parameters: dict[int, float],
) -> ProgramConversion:
    """Turn one evolved function body into a restricted-grammar expression.

    ``inputs`` maps the synthesized function's argument names to our public
    channel names; ``parameters`` maps ``params`` indices to fitted values.
    """
    try:
        module = ast.parse(body.strip())
    except SyntaxError as exc:
        raise InexpressibleProgram("body does not parse", str(exc)) from exc

    bindings: dict[str, ast.expr] = {}
    returned: ast.expr | None = None
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise InexpressibleProgram("assignment is not to a single name",
                                           ast.unparse(statement)[:80])
            bindings[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Return):
            if statement.value is None:
                raise InexpressibleProgram("returns nothing")
            returned = statement.value
            break
        elif isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue  # a docstring
        else:
            raise InexpressibleProgram("statement has no expression equivalent",
                                       ast.unparse(statement)[:80])
    if returned is None:
        raise InexpressibleProgram("body never returns a value")

    _check_supported(ast.Expression(body=returned))
    for value in bindings.values():
        _check_supported(ast.Expression(body=value))

    inliner = _Inliner(bindings, parameters, inputs)
    converted = inliner.visit(ast.Expression(body=returned))
    expression = ast.unparse(ast.fix_missing_locations(converted))
    return ProgramConversion(
        expression=expression,
        used_parameters=tuple(sorted(inliner.used_parameters)),
        used_inputs=tuple(sorted(inliner.used_inputs)),
    )
