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
    assert substitute_channels("x0 * 2.0 - x1", ("G", "I")) == "G * 2.0 - I"
    # a bare prefix must not be rewritten
    assert substitute_channels("exp(x0) + x10", tuple(f"c{i}" for i in range(11))) == (
        "exp(c0) + c10"
    )


def test_a_variable_beyond_the_channels_is_refused() -> None:
    with pytest.raises(ValueError, match="beyond the channels"):
        substitute_channels("x0 + x5", ("G", "I"))


def test_the_approved_functions_pass_and_sine_does_not() -> None:
    assert unapproved_functions("exp(G) + log(I) + abs(G) + tanh(I)") == ()
    assert unapproved_functions("sin(G) * 2.0") == ("sin",)
    assert unapproved_functions("sin(G) + cos(I)") == ("cos", "sin")


def test_a_selected_system_is_expressed_per_target() -> None:
    equations = to_state_equations(
        ("-0.5 * x0 + x1", "exp(x0) - 2.0"), ("G", "I"), ("G", "I")
    )
    assert equations == {"G": "-0.5 * G + I", "I": "exp(G) - 2.0"}


def test_an_inexpressible_equation_names_its_operator() -> None:
    """So the cost of our grammar can be reported rather than estimated."""
    with pytest.raises(InexpressibleEquation) as caught:
        to_state_equations(("sin(x0) * x1",), ("G", "I"), ("G",))
    assert caught.value.functions == ("sin",)
    assert "sin(G)" in caught.value.equation


def test_one_equation_is_required_per_searched_target() -> None:
    with pytest.raises(ValueError, match="one selected equation per"):
        to_state_equations(("x0",), ("G", "I"), ("G", "I"))


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
