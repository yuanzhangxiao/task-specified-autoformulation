"""Restricted expression parser and adversarial syntax tests."""

from __future__ import annotations

import pytest

from autoformalism.expressions import ModelValidationError, RestrictedParser


def _codes(error: ModelValidationError) -> set[str]:
    return {item.code for item in error.diagnostics}


def test_parser_accepts_approved_arithmetic_and_functions() -> None:
    parsed = RestrictedParser().parse(
        (
            "exp(x) + log(k) - tanh(y) + sqrt(abs(z)) + "
            "min(x, y) + max(y, z) + sigmoid(x) + softplus(y) + x**2"
        ),
        location="test",
    )

    assert parsed.symbols == frozenset({"x", "y", "z", "k"})


def test_parser_normalizes_safe_time_indexed_symbol_notation() -> None:
    parsed = RestrictedParser().parse(
        "EGP(t) - Uii(t) + Gp(t)",
        location="notation",
    )

    assert parsed.symbols == frozenset({"EGP", "Uii", "Gp"})


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('id')",
        "open('/tmp/pwned', 'w')",
        "(1).__class__.__mro__",
        "x.__class__",
        "[item for item in values]",
        "{item: item for item in values}",
        "(lambda value: value)(x)",
        "x if flag else y",
        "x < y",
        "x and y",
        "values[0]",
        "[x, y]",
        "{x, y}",
        "{'x': x}",
        "f'{x}'",
        "(x := 1)",
        "sum(x)",
        "unknown(other_time)",
        "round(x)",
        "exp + x",
        "min(x)",
        "max(x, y, *values)",
        "exp(x, base=2)",
        "x // y",
        "x % y",
        "x @ y",
        "~x",
        "not x",
        "x ^ y",
        "x << 2",
        "x ** y",
        "x ** 17",
        "'text'",
        "True",
        "None",
    ],
)
def test_parser_rejects_unsupported_or_adversarial_syntax(source: str) -> None:
    with pytest.raises(ModelValidationError) as caught:
        RestrictedParser().parse(source, location="adversarial")

    assert _codes(caught.value) & {
        "UNSUPPORTED_SYNTAX",
        "UNSUPPORTED_CALL",
        "UNSUPPORTED_FUNCTION",
        "INVALID_FUNCTION_ARITY",
        "UNSUPPORTED_OPERATOR",
        "UNSUPPORTED_POWER",
        "UNSUPPORTED_LITERAL",
        "FUNCTION_AS_VALUE",
    }


def test_parser_enforces_resource_limits() -> None:
    with pytest.raises(ModelValidationError) as too_long:
        RestrictedParser(max_length=5).parse("x + y + z", location="limit")
    assert _codes(too_long.value) == {"EXPRESSION_TOO_LONG"}

    with pytest.raises(ModelValidationError) as too_complex:
        RestrictedParser(max_nodes=5).parse("x + y + z", location="limit")
    assert "EXPRESSION_TOO_COMPLEX" in _codes(too_complex.value)

    with pytest.raises(ModelValidationError) as too_deep:
        RestrictedParser(max_depth=3).parse("(((x + 1) + 2) + 3)", location="limit")
    assert "EXPRESSION_TOO_DEEP" in _codes(too_deep.value)


def test_parser_rejects_invalid_numeric_literals() -> None:
    with pytest.raises(ModelValidationError) as caught:
        RestrictedParser(max_literal_magnitude=100.0).parse(
            "1e200 + x",
            location="literal",
        )

    assert "INVALID_NUMERIC_LITERAL" in _codes(caught.value)


def test_sine_is_approved_and_evaluates() -> None:
    """Approved after more than half of LLM-ODE's frontier was discarded for it.

    Their prompt invites the proposer to use `sin`, so refusing it measured our
    grammar rather than their method. The benchmark's own mechanisms contain no
    trigonometry -- every sine in the generator builds an exogenous driving
    waveform, supplied to every method as an observed channel -- so admitting
    it neither helps nor hinders a correct model.
    """
    import math

    from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY

    assert APPROVED_FUNCTION_ARITY["sin"] == (1, 1)
    parsed = RestrictedParser().parse("sin(G) + 2.0 * I", location="test")
    assert parsed is not None

    from autoformalism.expressions.compiler import _call_function

    assert _call_function("sin", [math.pi / 2.0]) == pytest.approx(1.0)


def test_the_sine_interval_accounts_for_extrema_inside_it() -> None:
    """Sine is the only approved function that is not monotonic.

    Bounding it by its endpoints would be unsound: sin over [0, pi] reaches 1
    at pi/2, which neither endpoint shows.
    """
    import math

    from autoformalism.expressions.intervals import Interval, _sine_interval

    assert _sine_interval(Interval(0.0, math.pi)) == Interval(0.0, 1.0)
    assert _sine_interval(Interval(0.0, 100.0)) == Interval(-1.0, 1.0)
    # a monotone stretch keeps its endpoints
    narrow = _sine_interval(Interval(0.0, math.pi / 4.0))
    assert narrow.lower == pytest.approx(0.0)
    assert narrow.upper == pytest.approx(math.sin(math.pi / 4.0))
    # a minimum inside is found too
    assert _sine_interval(Interval(math.pi, 2.0 * math.pi)).lower == -1.0
