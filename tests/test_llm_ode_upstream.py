"""Converting LLM-ODE's selected system into the grammar we can evaluate.

Upstream offers its proposer `sin`, which our approved functions do not
include. An equation using it is not a failure of the method but a limit of
our evaluator, so it is refused with the operator named rather than dropped.
"""

from __future__ import annotations

import pytest

from autoformalism.rebuttal.llm_ode_upstream import (
    InexpressibleEquation,
    specification_prompt,
    substitute_channels,
    to_state_equations,
    unapproved_functions,
)


def test_positional_variables_become_public_channel_names() -> None:
    assert substitute_channels("x_0 * 2.0 - x_1", ("G", "I")) == "G * 2.0 - I"
    # a bare prefix must not be rewritten
    channels = tuple(f"c{i}" for i in range(11))
    assert substitute_channels("exp(x_0) + x_10", channels) == "exp(c0) + c10"


def test_a_variable_beyond_the_channels_is_refused() -> None:
    with pytest.raises(ValueError, match="beyond the channels"):
        substitute_channels("x_0 + x_5", ("G", "I"))


def test_every_operator_upstream_advertises_is_now_approved() -> None:
    """More than half their frontier was discarded for using `sin`.

    Their prompt invites the proposer to use it, so rejecting it measured our
    grammar rather than the method. It is approved now; `cos` is not, because
    nothing upstream offers it.
    """
    assert unapproved_functions("exp(G) + log(I) + abs(G) + tanh(I)") == ()
    assert unapproved_functions("sin(G) * 2.0") == ()
    assert unapproved_functions("sin(G) + cos(I)") == ("cos",)


def test_a_selected_system_is_expressed_per_target() -> None:
    equations = to_state_equations(
        ("-0.5 * x_0 + x_1", "exp(x_0) - 2.0"), ("G", "I"), ("G", "I")
    )
    assert equations == {"G": "-0.5 * G + I", "I": "exp(G) - 2.0"}


def test_a_sine_equation_is_now_expressed_rather_than_refused() -> None:
    equations = to_state_equations(("sin(x_0) * x_1",), ("G", "I"), ("G",))
    assert equations == {"G": "sin(G) * I"}


def test_an_equation_outside_the_grammar_still_names_its_operator() -> None:
    """The mechanism stays, so a future divergence is measured not assumed."""
    with pytest.raises(InexpressibleEquation) as caught:
        to_state_equations(("cos(x_0) * x_1",), ("G", "I"), ("G",))
    assert caught.value.functions == ("cos",)
    assert "cos(G)" in caught.value.equation


def test_one_equation_is_required_per_searched_target() -> None:
    with pytest.raises(ValueError, match="one selected equation per"):
        to_state_equations(("x_0",), ("G", "I"), ("G", "I"))


def test_the_specification_is_appended_to_upstreams_own_prompt() -> None:
    """Their instructions and format are untouched; ours trails them."""
    base = [
        {"role": "system", "content": "Available operators: +, -, *"},
        {"role": "user", "content": "Here are elite equations."},
    ]
    extended = specification_prompt(base, "\n\nTask specification (public):\nX.")
    assert extended[0] == base[0]
    assert extended[1]["content"].startswith("Here are elite equations.")
    assert extended[1]["content"].endswith("Task specification (public):\nX.")
    # the original is not mutated
    assert base[1]["content"] == "Here are elite equations."


def test_sympy_printed_abs_is_not_reported_as_inexpressible() -> None:
    """Upstream offers `abs`; SymPy prints it `Abs`.

    Equation.to_string() returns SymPy's printer output, so an equation the
    proposer wrote as abs(x_0) arrives here as Abs(x_0). Our grammar approves
    abs, so treating the printed spelling as unapproved would report a model
    inexpressible over a naming difference and inflate the cost we attribute
    to the restricted grammar.
    """
    equations = to_state_equations(("-c_0 * Abs(x_1) + x_0",), ("G", "I"), ("G",))
    assert equations == {"G": "-c_0 * abs(I) + G"}


def test_nothing_upstream_advertises_is_now_a_gap() -> None:
    """Their template offers +, -, *, **, /, sin, log, exp, abs -- all approved."""
    from autoformalism.rebuttal.llm_ode_upstream import (
        UNAPPROVED_UPSTREAM_FUNCTIONS,
        normalize_printed_functions,
    )

    assert UNAPPROVED_UPSTREAM_FUNCTIONS == ()
    for advertised in ("exp", "log", "abs", "sin"):
        rendered = normalize_printed_functions(f"{advertised.capitalize()}(G)")
        assert unapproved_functions(f"{advertised}(G)") == ()
        assert rendered.startswith(advertised) or rendered.startswith(
            advertised.capitalize()
        )
