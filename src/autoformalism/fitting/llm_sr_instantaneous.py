"""Isolated scalar interpreter for fixed current-value baseline expressions.

Real powers, clip and Heaviside do not extend the proposer grammar. Native
source is never executed; nonfinite/domain errors fail the replay.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from autoformalism.expressions import CompiledModel, RuntimeExpressionError
from autoformalism.expressions.parser import ParsedExpression
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.schemas import CandidateModel

ARITY = {
    "exp": (1, 1),
    "tanh": (1, 1),
    "abs": (1, 1),
    "min": (2, 64),
    "max": (2, 64),
    "clip": (3, 3),
    "heaviside": (2, 2),
}
NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def parse(source: str, symbols: set[str]) -> ParsedExpression:
    """Reject everything except bounded scalar arithmetic and listed calls."""
    if len(source) > 4096:
        raise ValueError("current-value expression exceeds length bound")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("invalid current-value expression") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 512:
        raise ValueError("current-value expression exceeds node bound")
    pending = [(tree, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 64:
            raise ValueError("current-value expression exceeds depth bound")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    functions = set()
    for node in nodes:
        if not isinstance(node, NODES):
            raise ValueError(f"unsupported current-value node: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (
            type(node.value) not in (int, float)
            or abs(node.value) > 1e12
            or not math.isfinite(node.value)
        ):
            raise ValueError("invalid current-value numeric literal")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ARITY:
                raise ValueError("unsupported current-value function")
            lo, hi = ARITY[node.func.id]
            if node.keywords or not lo <= len(node.args) <= hi:
                raise ValueError("invalid current-value function arguments")
            functions.add(id(node.func))
    names = {n.id for n in nodes if isinstance(n, ast.Name) and id(n) not in functions}
    if names - symbols:
        raise ValueError(f"unknown current-value symbols: {sorted(names - symbols)}")
    return ParsedExpression(source, tree, frozenset(names))


def evaluate(node: ast.AST, environment: dict[str, float]) -> float:
    """Interpret validated scalar syntax with unmodified division/power domains."""

    def walk(item):
        value = float(raw(item))
        if not math.isfinite(value):
            raise ValueError("nonfinite intermediate value")
        return value

    def raw(item):
        if isinstance(item, ast.Expression):
            return walk(item.body)
        if isinstance(item, ast.Constant):
            return float(item.value)
        if isinstance(item, ast.Name):
            return float(environment[item.id])
        if isinstance(item, ast.UnaryOp):
            value = walk(item.operand)
            return -value if isinstance(item.op, ast.USub) else value
        if isinstance(item, ast.BinOp):
            left, right = walk(item.left), walk(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            if isinstance(item.op, ast.Div):
                return left / right
            if isinstance(item.op, ast.Pow):
                return math.pow(left, right)
        if isinstance(item, ast.Call):
            values = [walk(arg) for arg in item.args]
            name = item.func.id
            if name == "clip":
                return min(max(values[0], values[1]), values[2])
            if name == "heaviside":
                return values[1] if values[0] == 0 else float(values[0] > 0)
            functions = {
                "exp": math.exp,
                "tanh": math.tanh,
                "abs": abs,
                "min": min,
                "max": max,
            }
            if name in functions:
                return functions[name](*values)
        raise ValueError("unvalidated current-value syntax")

    try:
        value = float(walk(node))
    except (ArithmeticError, ValueError, KeyError) as error:
        raise RuntimeExpressionError(
            f"current-value expression domain failure: {error}"
        ) from error
    if not math.isfinite(value):
        raise RuntimeExpressionError("current-value expression is nonfinite")
    return value


class CurrentValueModel(CompiledModel):
    """Standard forcing/integration contract with a private RHS interpreter."""

    replay_interpretation = (
        "Reviewed current-value rewrite at supplied coefficients; no history "
        "filter approximation. Native integration scores do not transfer."
    )

    def _environment(self, time, state, parameters, forcing):
        base = CompiledModel(replace(self.validated, process_order=()))
        environment = base._environment(time, state, parameters, forcing)
        for name in self.validated.process_order:
            environment[name] = evaluate(
                self.validated.process_expressions[name].tree, environment
            )
        return environment

    def rhs(self, time, state, parameters, forcing):
        environment = self._environment(time, state, parameters, forcing)
        return np.asarray(
            [
                evaluate(self.validated.equation_expressions[name].tree, environment)
                for name in self.state_names
            ]
        )

    def evaluate_expression(self, expression, time, state, parameters, forcing):
        return evaluate(
            expression.tree, self._environment(time, state, parameters, forcing)
        )


def compile_current_value(item, context) -> CurrentValueModel:
    """Compile a history-free fixed expression without changing global registries."""
    if item.hidden_states or context.lagged_targets:
        raise ValueError(
            "current-value controls forbid hidden states and target forcing"
        )
    names = set(item.parameters) | set(item.targets) | set(item.processes)
    if names & context.forcing_channels or names & set(ARITY):
        raise ValueError("ambiguous current-value symbol declaration")
    available = names | context.forcing_channels | {context.time_symbol}
    processes = {
        name: parse(source, available) for name, source in item.processes.items()
    }
    equations = {
        name: parse(source, available) for name, source in item.targets.items()
    }
    order = []
    pending = dict(processes)
    while pending:
        ready = [
            name
            for name, expression in pending.items()
            if not (expression.symbols & set(pending))
        ]
        if not ready:
            raise ValueError("cyclic current-value processes")
        order.extend(ready)
        for name in ready:
            del pending[name]
    # Existing validation checks declarations/initializers. Replace arithmetic
    # placeholders only with the separately validated restricted ASTs above.
    placeholder = " + ".join(["0", *(f"0*{name}" for name in item.parameters)])
    base_item = item.model_copy(
        update={"processes": {}, "targets": dict.fromkeys(item.targets, placeholder)}
    )
    base = control.compile_transcription(base_item, context)
    candidate = CandidateModel.model_validate(
        base.validated.candidate.model_dump(mode="json")
        | {
            "processes": [
                {"name": name, "expression": source}
                for name, source in item.processes.items()
            ],
            "state_equations": [
                {"state": name, "rhs": source} for name, source in item.targets.items()
            ],
        }
    )
    used = set().union(*(e.symbols for e in (*processes.values(), *equations.values())))
    return CurrentValueModel(
        replace(
            base.validated,
            candidate=candidate,
            process_expressions=MappingProxyType(processes),
            equation_expressions=MappingProxyType(equations),
            process_order=tuple(order),
            forcing_symbols=frozenset(used & context.forcing_channels),
        )
    )
