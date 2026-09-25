"""Bind the pinned LLM-ODE checkout to Phase-B data.

Everything the method does -- the island search, its prompt instructions, its
BFGS coefficient fitting, its Pareto selection -- comes from upstream. This
module supplies data, appends the declared task specification, and converts the
selected system into the restricted grammar the frozen evaluator parses.

Upstream offers `sin` to its proposer, and the evaluator now approves it: more
than half the Pareto frontier was being discarded for using an operator the
method was explicitly invited to use, which measured our grammar rather than
the method. A selected equation can still fall outside the grammar for other
reasons, and is then recorded with the offending operator named.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY

#: Upstream's SYSTEM_TEMPLATE offers +, -, *, **, /, sin, log, exp and abs.
#: All of them are now in our approved set, so nothing upstream advertises is
#: inexpressible here. Detection is still computed from the parsed equation
#: rather than from this tuple, so a future divergence is caught by measurement
#: and not by this comment.
UNAPPROVED_UPSTREAM_FUNCTIONS: tuple[str, ...] = ()

#: SymPy prints some approved operators with its own spelling: an equation
#: containing ``abs`` comes back as ``Abs``. Renaming these before the grammar
#: check keeps an approved operator from being reported as inexpressible.
SYMPY_PRINTED_ALIASES = {"Abs": "abs", "Max": "max", "Min": "min"}


class InexpressibleEquation(ValueError):
    """A selected equation lies outside the grammar the evaluator parses."""

    def __init__(self, equation: str, functions: tuple[str, ...]) -> None:
        super().__init__(
            f"equation uses {', '.join(functions)}, which the restricted "
            f"grammar does not approve: {equation}"
        )
        self.equation = equation
        self.functions = functions


@dataclass(frozen=True)
class UpstreamModules:
    """The pinned checkout's entry points, imported once."""

    equation_searcher: type
    system: type
    llm: type
    generate_prompt: object


#: Upstream pins ``requires-python = "==3.13.5"`` and calls
#: ``str.replace(..., count=1)``, which only accepts that keyword from 3.13.
#: On an older interpreter every program construction raises TypeError,
#: upstream logs it as a warning and continues, and the islands stay empty for
#: the whole run: a silent two-hour job that discovers nothing.
MINIMUM_UPSTREAM_PYTHON = (3, 13)


def load_upstream(root: Path) -> UpstreamModules:
    """Import the vendored checkout without installing it."""
    if sys.version_info[:2] < MINIMUM_UPSTREAM_PYTHON:
        running = ".".join(str(part) for part in sys.version_info[:3])
        wanted = ".".join(str(part) for part in MINIMUM_UPSTREAM_PYTHON)
        raise RuntimeError(
            f"LLM-ODE requires Python >= {wanted} and this is {running}. "
            "Its programs are built with str.replace(count=1), which older "
            "interpreters reject; the search would run to completion having "
            "created no candidates. Use an interpreter matching the pinned "
            "checkout rather than editing its source."
        )
    resolved = root.expanduser().resolve()
    if not (resolved / "llmode" / "llmode.py").is_file():
        raise ValueError(f"not an LLM-ODE checkout: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from llmode.llm import Llm, generate_prompt
    from llmode.llmode import LlmOdeEquation
    from llmode.system import System

    return UpstreamModules(
        equation_searcher=LlmOdeEquation,
        system=System,
        llm=Llm,
        generate_prompt=generate_prompt,
    )


def unapproved_functions(equation: str) -> tuple[str, ...]:
    """Name every called function the restricted grammar does not approve."""
    try:
        tree = ast.parse(equation, mode="eval")
    except SyntaxError as exc:
        raise InexpressibleEquation(equation, ("unparsable",)) from exc
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return tuple(sorted(called - set(APPROVED_FUNCTION_ARITY)))


def substitute_channels(equation: str, channels: tuple[str, ...]) -> str:
    """Rename upstream's positional variables to our public channel names."""
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(channels):
            raise ValueError(f"equation references x_{index} beyond the channels")
        return channels[index]

    return re.sub(r"\bx_(\d+)\b", replace, equation)


def normalize_printed_functions(equation: str) -> str:
    """Rewrite SymPy's printed spelling of operators our grammar approves."""
    for printed, approved in SYMPY_PRINTED_ALIASES.items():
        equation = re.sub(rf"\b{printed}\(", f"{approved}(", equation)
    return equation


def to_state_equations(
    selected: tuple[str, ...],
    channels: tuple[str, ...],
    targets: tuple[str, ...],
) -> dict[str, str]:
    """Express the selected system in our grammar, or say why it cannot be."""
    if len(selected) != len(targets):
        raise ValueError("expected one selected equation per searched target")
    equations: dict[str, str] = {}
    for target, equation in zip(targets, selected, strict=True):
        renamed = normalize_printed_functions(
            substitute_channels(str(equation), channels)
        )
        unapproved = unapproved_functions(renamed)
        if unapproved:
            raise InexpressibleEquation(renamed, unapproved)
        equations[target] = renamed
    return equations


def specification_prompt(
    base_prompt: object, specification: str
) -> list[dict[str, str]]:
    """Append the declared specification to upstream's own prompt.

    Upstream's instructions, in-context examples and output format are left
    exactly as they are; the specification is added as trailing content.
    """
    messages = [dict(item) for item in base_prompt]  # type: ignore[arg-type]
    if not messages:
        raise ValueError("upstream produced no prompt to extend")
    messages[-1]["content"] = messages[-1]["content"] + specification
    return messages
