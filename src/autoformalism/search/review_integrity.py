"""Prospective parameter and validation-selection policies for review v8."""

import math

PARAMETER_POLICY = "reject-conflicting-existing-role-1"
SELECTION_POLICY = "validation-tolerance-then-terms-1"
ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 1e-6


def selection_policy() -> dict:
    """Return the fixed policy recorded in the immutable continuation plan."""
    return {
        "policy": SELECTION_POLICY,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "tolerance_rule": "atol + rtol * max(abs(incumbent), abs(trial))",
        "complexity": "state_and_process_outer_additive_terms",
        "within_tolerance": "fewer_terms_then_incumbent",
    }


def selection_decision(
    incumbent: tuple[float, float], trial: tuple[float, float]
) -> dict:
    """Compare eligible validation scores; invalid keys cannot win a comparison.

    Keys are (validation NMSE, outer additive term count). The finite band is an
    engineering decision tolerance, not a statistical or scientific certificate.
    A simpler model may have slightly higher NMSE within that band.
    """

    def admissible(key):
        return all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in key
        )

    old_ok, new_ok = admissible(incumbent), admissible(trial)
    old, new = incumbent[0], trial[0]
    band, improvement = None, None
    if not new_ok:
        choose, reason = False, "trial_ineligible"
    elif not old_ok:
        choose, reason = True, "no_eligible_incumbent"
    else:
        band = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(abs(old), abs(new))
        improvement = old - new
        if abs(improvement) <= band:
            choose = trial[1] < incumbent[1]
            reason = (
                "within_tolerance_fewer_terms"
                if choose
                else "within_tolerance_keep_incumbent"
            )
        else:
            choose = improvement > 0
            reason = "validation_improved" if choose else "validation_worse"
    return {
        **selection_policy(),
        "incumbent_validation_nmse": old if old_ok else None,
        "trial_validation_nmse": new if new_ok else None,
        "incumbent_terms": incumbent[1] if old_ok else None,
        "trial_terms": trial[1] if new_ok else None,
        "validation_improvement": improvement,
        "comparison_tolerance": band,
        "choose_trial": choose,
        "reason": reason,
    }
