"""Converting an LLM-SR program into the grammar the evaluator can score.

LLM-SR evolves the body of a Python function, not an expression. Some bodies
have an exact expression equivalent and some genuinely do not; a loop over
timesteps is a real discovery our evaluator cannot represent. What matters is
that the second kind is refused by name and counted, never silently dropped or
approximated.
"""

from __future__ import annotations

import pytest

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal.llm_sr_upstream import (
    InexpressibleProgram,
    convert_program,
)

PARAMS = {index: float(index + 1) / 2 for index in range(10)}
INPUTS = {"x": "G", "v": "I"}


def _converted(body: str) -> str:
    """Convert, and require the result to survive the restricted parser."""
    result = convert_program(body, INPUTS, PARAMS)
    RestrictedParser().parse(result.expression, location="test")
    return result.expression


def test_the_published_skeleton_converts_with_its_fitted_values() -> None:
    """This is the shape every LLM-SR specification starts from."""
    assert _converted(
        "dv = params[0] * x + params[1] * v + params[2]\nreturn dv"
    ) == "0.5 * G + 1.0 * I + 1.5"


def test_intermediate_assignments_are_inlined() -> None:
    """A body that names subexpressions is still one expression underneath."""
    assert _converted(
        "a = np.exp(params[0] * x)\nb = params[1] / (1.0 + a)\nreturn b"
    ) == "1.0 / (1.0 + exp(0.5 * G))"


def test_numpy_spellings_map_onto_approved_functions() -> None:
    assert _converted('"""doc."""\nreturn np.maximum(params[0] * x, params[1])') == (
        "max(0.5 * G, 1.0)"
    )
    assert _converted("return np.absolute(v) + np.sqrt(params[3] * x)") == (
        "abs(I) + sqrt(2.0 * G)"
    )


def test_which_parameters_and_channels_were_used_is_reported() -> None:
    """The caller needs to know what the model actually depends on."""
    result = convert_program("return params[2] * v", INPUTS, PARAMS)
    assert result.used_parameters == (2,)
    assert result.used_inputs == ("v",)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("out = 0\nfor i in range(2):\n    out = out + x\nreturn out",
         "no expression equivalent"),
        ("if x > 0:\n    return x\nreturn v", "no expression equivalent"),
        ("return np.sin(params[0] * x)", "not approved"),
        ("return params[0] * z", "unknown name"),
        ("return x.T * params[0]", "attribute access"),
        ("return x.mean() * params[0]", "attribute access"),
        ("return params[int(v[0])] * x", "not a literal"),
        ("return [params[0] * y for y in x]", "comprehension"),
        ("return params[0] * x if x > 0 else params[1]", "conditional"),
        ("return params[99] * x", "did not produce"),
        ("x = (", "does not parse"),
        ("pass", "no expression equivalent"),
        ("a = params[0] * x", "never returns"),
    ],
)
def test_a_program_outside_the_grammar_is_refused_by_name(
    body: str, expected: str
) -> None:
    with pytest.raises(InexpressibleProgram, match=expected):
        convert_program(body, INPUTS, PARAMS)


def test_nothing_is_executed_during_conversion(monkeypatch) -> None:
    """The point of parsing rather than running is that nothing runs.

    LLM-SR's own evaluator exec()s these bodies; our conversion must not, or
    the frozen evaluation would be executing proposer-generated code too.
    """
    import builtins

    # `compile` is excluded deliberately: ast.parse uses it with PyCF_ONLY_AST
    # to build a tree, which runs none of the parsed code.
    for name in ("eval", "exec"):
        monkeypatch.setattr(
            builtins, name,
            lambda *a, _n=name, **k: pytest.fail(f"{_n} called during conversion"),
        )
    assert _converted("return params[0] * x + params[1]")
