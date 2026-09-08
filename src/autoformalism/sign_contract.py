"""Conservative structural analysis for staged outer-weight sign decisions."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from autoformalism.schemas.candidate import ParameterRole
from autoformalism.schemas.staged import InteractionPolarity

_SIGNED_ROLES = {ParameterRole.COEFFICIENT, ParameterRole.OFFSET}
_OUTER_WEIGHT_ROLES = {*_SIGNED_ROLES, ParameterRole.NONNEGATIVE_COEFFICIENT}
_FIXED_SIGNS = {InteractionPolarity.POSITIVE, InteractionPolarity.NEGATIVE}


@dataclass(frozen=True)
class OuterWeightAnalysis:
    """A deliberately narrow conclusion about one complete term expression."""

    identified_parameter: str | None = None
    diagnostic_code: str | None = None
    diagnostic_message: str | None = None


def analyze_outer_weight(
    tree: ast.Expression,
    parameter_roles: Mapping[str, ParameterRole],
    polarity: InteractionPolarity,
) -> OuterWeightAnalysis:
    """Identify a whole-term signed gain without interpreting internal algebra.

    Only a direct numerator factor (or a parameter-only expression) is eligible.
    Parameters inside sums, differences, calls, powers, or source expressions are
    intentionally ignored. This keeps thresholds and internal signed differences
    real-valued unless their own semantic role requires positivity.
    """
    if is_legacy_outer_sign(polarity):
        return OuterWeightAnalysis()

    numerator, denominator = _multiplicative_shell(tree.body)
    negative_count = sum(
        _direct_negative_syntax_count(item) for item in (*numerator, *denominator)
    )
    if polarity in _FIXED_SIGNS and negative_count:
        return OuterWeightAnalysis(
            diagnostic_code="DUPLICATED_WHOLE_OUTER_SIGN",
            diagnostic_message=(
                "the function contains a clear whole-expression negative factor "
                "even though the topology already owns the fixed outer sign"
            ),
        )

    if polarity is InteractionPolarity.UNRESTRICTED:
        candidates = sorted(
            {
                name
                for item in numerator
                if (name := _direct_parameter_name(item)) is not None
                and parameter_roles.get(name) in _OUTER_WEIGHT_ROLES
            }
        )
        if len(candidates) > 1:
            return OuterWeightAnalysis(
                diagnostic_code="AMBIGUOUS_OUTER_WEIGHT_PARAMETERS",
                diagnostic_message=(
                    "multiple direct factors could own the unrestricted outer "
                    f"weight: {candidates}"
                ),
            )
        return OuterWeightAnalysis(
            identified_parameter=candidates[0] if candidates else None
        )

    denominator_parameters = sorted(
        name
        for item in denominator
        if (name := _direct_parameter_name(item)) is not None
        and parameter_roles.get(name) in _SIGNED_ROLES
    )
    if denominator_parameters:
        return OuterWeightAnalysis(
            diagnostic_code="SIGNED_PARAMETER_CONTROLS_OUTER_SIGN_IN_DENOMINATOR",
            diagnostic_message=(
                "a real-valued parameter in the multiplicative denominator can "
                f"reverse the fixed outer sign: {denominator_parameters}"
            ),
        )

    candidates = sorted(
        {
            name
            for item in numerator
            if (name := _direct_parameter_name(item)) is not None
            and parameter_roles.get(name) in _SIGNED_ROLES
        }
    )
    if len(candidates) > 1:
        return OuterWeightAnalysis(
            diagnostic_code="AMBIGUOUS_OUTER_WEIGHT_PARAMETERS",
            diagnostic_message=(
                "multiple real-valued direct factors could own the fixed outer "
                f"weight sign: {candidates}"
            ),
        )
    return OuterWeightAnalysis(
        identified_parameter=candidates[0] if candidates else None
    )


def is_fixed_outer_sign(polarity: InteractionPolarity) -> bool:
    """Return whether a new topology contract fixes the outer weight sign."""
    return polarity in _FIXED_SIGNS


def is_subtractive_outer_sign(polarity: InteractionPolarity) -> bool:
    """Return whether assembly applies exactly one outer subtraction."""
    return polarity in {
        InteractionPolarity.SUBTRACTIVE,
        InteractionPolarity.NEGATIVE,
    }


def is_legacy_outer_sign(polarity: InteractionPolarity) -> bool:
    """Return whether a polarity came from a frozen add/subtract artifact."""
    return polarity in {
        InteractionPolarity.ADDITIVE,
        InteractionPolarity.SUBTRACTIVE,
    }


def _multiplicative_shell(node: ast.expr) -> tuple[list[ast.expr], list[ast.expr]]:
    """Return numerator/denominator factors without entering internal laws."""
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        node = node.operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_num, left_den = _multiplicative_shell(node.left)
        right_num, right_den = _multiplicative_shell(node.right)
        return left_num + right_num, left_den + right_den
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_num, left_den = _multiplicative_shell(node.left)
        right_num, right_den = _multiplicative_shell(node.right)
        return left_num + right_den, left_den + right_num
    return [node], []


def _direct_parameter_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        node = node.operand
    return node.id if isinstance(node, ast.Name) else None


def _direct_negative_syntax_count(node: ast.expr) -> int:
    count = 0
    while isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            count += 1
        elif not isinstance(node.op, ast.UAdd):
            return count
        node = node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and node.value < 0
    ):
        count += 1
    return count
