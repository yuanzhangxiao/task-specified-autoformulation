"""Topology owns the outer sign; internal scientific arithmetic stays intact."""

import ast

import pytest

from autoformalism.expressions import ModelValidationError, RestrictedParser
from autoformalism.schemas.staged import InteractionPolarity
from autoformalism.schemas.staged_functions import InteractionFunctionReply
from autoformalism.staged_functions import (
    normalize_outer_weight_reply,
    normalize_topology_owned_sign,
)


def tree(expression):
    return ast.dump(RestrictedParser().parse(expression, location="test").tree)


@pytest.mark.parametrize("sign", ["positive", "negative"])
@pytest.mark.parametrize(
    "source, expected, count",
    [
        ("-x**2", "x**2", 1),
        ("-(x**2)/tau", "x**2/tau", 1),
        ("x**2/(-tau)", "x**2/tau", 1),
        ("(-k)*(-x)", "k*x", 2),
        ("-(-k*x)", "k*x", 2),
        ("-2*x", "2*x", 1),
        ("-(x-y)/tau", "(x-y)/tau", 1),
        ("-k*exp(-x/tau)", "k*exp(-x/tau)", 1),
        ("-(x*(-y))", "x*y", 2),
    ],
)
def test_explicit_outer_factors_are_normalized_and_recorded(
    sign, source, expected, count
):
    raw = InteractionFunctionReply(expression=source, parameters=())
    reply, records = normalize_topology_owned_sign(raw, outer_weight_sign=sign)
    assert tree(reply.expression) == tree(expected)
    assert raw.expression == source
    assert records[0].original_expression == source
    assert records[0].normalized_expression == reply.expression
    assert records[0].outer_weight_sign == sign
    assert records[0].removed_negative_factors == count
    assert normalize_topology_owned_sign(reply, outer_weight_sign=sign) == (reply, ())


@pytest.mark.parametrize(
    "expression",
    [
        "x-y",
        "-x+y",
        "k*(x-y)",
        "exp(-x/tau)",
        "(-x)**2",
        "x**(-2)",
        "tanh(-x)",
        "x/(-tau+x)",
        "-x**2+y**2",
    ],
)
def test_internal_signs_are_not_outer_signs(expression):
    raw = InteractionFunctionReply(expression=expression, parameters=())
    assert normalize_topology_owned_sign(raw, outer_weight_sign="negative") == (raw, ())


def test_unrestricted_function_retains_its_outer_minus():
    raw = InteractionFunctionReply(expression="-x**2", parameters=())
    assert normalize_topology_owned_sign(raw, outer_weight_sign="unrestricted") == (
        raw,
        (),
    )


def test_strict_legacy_policy_still_rejects_but_prepared_reply_binds_sign_once():
    raw = InteractionFunctionReply(
        expression="-k*x**2", parameters=({"name": "k", "role": "coefficient"},)
    )
    with pytest.raises(ValueError, match="DUPLICATED_WHOLE_OUTER_SIGN"):
        normalize_outer_weight_reply(raw, InteractionPolarity.NEGATIVE)
    prepared, _ = normalize_topology_owned_sign(raw, outer_weight_sign="negative")
    normalized, _ = normalize_outer_weight_reply(prepared, InteractionPolarity.NEGATIVE)
    assert normalized.parameters[0].role.value == "nonnegative_coefficient"
    assert tree(normalized.expression) == tree("k*x**2")


@pytest.mark.parametrize("expression", ["__import__('os')", "x = -x**2", "x*****x"])
def test_normalization_never_bypasses_the_restricted_parser(expression):
    with pytest.raises((ModelValidationError, ValueError)):
        normalize_topology_owned_sign(
            InteractionFunctionReply(expression=expression, parameters=()),
            outer_weight_sign="negative",
        )
