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
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

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
        parameters: dict[int, float] | None,
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
        if index.value >= MAX_NPARAMS:
            raise InexpressibleProgram(
                "parameter index is beyond the declared vector", ast.unparse(node)
            )
        self.used_parameters.add(index.value)
        if self.parameters is None:
            # Symbolic mode: the coefficient is still to be fitted.
            return ast.copy_location(
                ast.Name(id=parameter_symbol(index.value), ctx=ast.Load()), node
            )
        if index.value not in self.parameters:
            raise InexpressibleProgram(
                "uses a parameter the fit did not produce", ast.unparse(node)
            )
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
    parameters: dict[int, float] | None = None,
) -> ProgramConversion:
    """Turn one evolved function body into a restricted-grammar expression.

    ``inputs`` maps the synthesized function's argument names to our public
    channel names. ``parameters`` maps ``params`` indices to fitted values;
    passing ``None`` leaves each coefficient as a symbol so it can be fitted
    first, which is how a model is recovered before its values are known.
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


#: Their specification template fixes the parameter vector length.
MAX_NPARAMS = 10

#: One node per operator, evaluated over numpy arrays. Defined here rather
#: than reached for with eval so that recovering a model from LLM-SR never
#: executes proposer-written code, whatever their own evaluator does.
_NUMPY_FUNCTIONS = {
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "max": np.maximum,
    "min": np.minimum,
    "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "softplus": lambda value: np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0),
}


def parameter_symbol(index: int) -> str:
    """The name a fitted coefficient carries once params[i] is symbolic."""
    return f"p_{index}"


def evaluate_expression(
    expression: str, bindings: dict[str, Any]
) -> NDArray[np.float64]:
    """Evaluate a restricted expression over numpy arrays, without exec.

    Only the node types the grammar approves are handled; anything else raises,
    so a program that slipped through conversion cannot execute here either.
    """
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in bindings:
                raise InexpressibleProgram("unbound name", node.id)
            return bindings[node.id]
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise InexpressibleProgram("unary operator", ast.unparse(node))
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            for kind, apply in (
                (ast.Add, np.add), (ast.Sub, np.subtract),
                (ast.Mult, np.multiply), (ast.Div, np.divide),
                (ast.Pow, np.power),
            ):
                if isinstance(node.op, kind):
                    return apply(left, right)
            raise InexpressibleProgram("binary operator", ast.unparse(node))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _NUMPY_FUNCTIONS.get(node.func.id)
            if function is None:
                raise InexpressibleProgram("function is not approved", node.func.id)
            return function(*[walk(item) for item in node.args])
        raise InexpressibleProgram("node has no numeric meaning", ast.unparse(node))

    with np.errstate(all="ignore"):
        return np.asarray(walk(tree.body), dtype=float)


def refit_parameters(
    expression: str,
    channels: dict[str, NDArray[np.float64]],
    target: NDArray[np.float64],
    used: tuple[int, ...],
) -> dict[int, float] | None:
    """Re-run the coefficient fit their specification performs and discards.

    Their ``evaluate`` optimises the parameters with BFGS from an all-ones
    start and then returns only the loss, so the values that made the score
    are not saved. This repeats that fit, on train data, to recover them.
    """
    if not used:
        return {}
    names = [parameter_symbol(index) for index in used]

    def loss(values: NDArray[np.float64]) -> float:
        bindings = {**channels, **dict(zip(names, values, strict=True))}
        try:
            predicted = evaluate_expression(expression, bindings)
        except InexpressibleProgram:
            return float("inf")
        residual = np.asarray(predicted, dtype=float) - target
        if not np.isfinite(residual).all():
            return float("inf")
        return float(np.mean(residual**2))

    result = minimize(loss, np.ones(len(used)), method="BFGS", tol=1e-6)
    if not np.isfinite(result.fun):
        return None
    return {index: float(value) for index, value in zip(used, result.x, strict=True)}


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


def equation_body(function_source: str) -> str:
    """The body of the evolved function, without its signature or docstring."""
    try:
        module = ast.parse(textwrap.dedent(function_source))
    except SyntaxError as exc:
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
    first = variables[0]
    specification = SPECIFICATION_TEMPLATE.format(
        description=description.strip(),
        max_params=max_params,
        unpack=unpack,
        arguments=", ".join(variables),
        signature=", ".join(f"{variable}: np.ndarray" for variable in variables),
        target=target,
        argument_docs=argument_docs,
        initial="d" + first,
        initial_expression=" + ".join(
            [
                f"params[{index}] * {variable}"
                for index, variable in enumerate(variables)
            ]
            + [f"params[{len(variables)}]"]
        ),
    )
    return specification, dict(zip(variables, channels, strict=True))
