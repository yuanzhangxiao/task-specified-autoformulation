"""One bounded lazy continuation with evidence-gated endpoint audit escalation.

Search precision can establish remaining descent or allocate a strict audit. It
cannot certify stationarity. This opt-in layer leaves the earlier 600-second
polishing protocol and all derivative acceptance thresholds unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from numbers import Real

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_polishing as polishing

PROTOCOL = "llm-sr-detention-staged-polishing-numerics-1"
Mode = controls.Mode
Policy = chart.Policy


def fit(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    mode: Mode = "original",
    seconds: float = 1800,
    policy: Policy | None = None,
    checkpoint: Callable[[dict], None] | None = None,
    active_names: Sequence[str] | None = None,
) -> dict:
    """Run one uninterrupted at-most-1800-second segment from exact saved arrays.

    The caller owns the durable nonrenewable allowance, independent verification
    and later endpoint audits. Every start retains its own parameters and cache.
    The existing control preserves finite improvements before derivative failure.
    """
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, Real)
        or not isfinite(seconds)
        or not 0 < seconds <= 1800
    ):
        raise ValueError(
            "staged polishing allowance must be positive and at most 1800 seconds"
        )
    eligible = controls.select_active(
        item,
        context,
        training,
        parameters,
        mode=mode,
        qualification=qualification,
        policy=policy,
    )
    binding = {
        "staged_polishing_protocol": PROTOCOL,
        "maximum_segments": 1,
        "allowance_seconds": float(seconds),
        "historical_allowances_renewed": False,
    }

    def save(value: dict) -> None:
        if checkpoint:
            checkpoint(value | binding)

    return (
        controls.fit_control(
            item,
            context,
            training,
            parameters,
            mode=mode,
            lazy_jacobian=True,
            active_names=eligible if active_names is None else active_names,
            qualification=qualification,
            seconds=seconds,
            policy=policy,
            checkpoint=save,
            reuse_initial_qualification=True,
        )
        | binding
    )


def audit_decision(
    item,
    context,
    training,
    parameters: Mapping[str, float],
    *,
    qualification: dict,
    verification: Mapping,
    mode: Mode = "original",
    policy: Policy | None = None,
) -> dict:
    """Authenticate an exact endpoint and decide whether to allocate strict checks.

    This function performs no numerical work and mutates no evidence. Escalation
    needs all chart coordinates to qualify for search with no failed/missing
    rows, and every feasible empirical gradient interval to admit the existing
    stationarity tolerance. The search-error bounds already cap uncertainty;
    no new near-stationarity multiplier or acceptance threshold is introduced.

    Inconclusive local-zero rows may allocate strict checks but cannot certify
    stationarity. Only the unchanged strict assessment can do that. Independent
    verification receipts must be authenticated by the calling campaign.
    """
    try:
        assessment = polishing.assess_endpoint(
            item,
            context,
            training,
            parameters,
            qualification=qualification,
            verification=verification,
            mode=mode,
            policy=policy,
        )
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed staged endpoint evidence") from error
    evidence = assessment["coordinate_evidence"]
    covered = (
        qualification["complete"] is True
        and not qualification["missing_indices"]
        and len(evidence) == len(qualification["coordinate_names"])
    )
    lower_bound = (
        max(row["lower_empirical_descent_magnitude"] for row in evidence)
        if covered
        else None
    )
    result = assessment | {
        "protocol": PROTOCOL,
        "endpoint_assessment_protocol": assessment["protocol"],
        "decision": "unresolved",
        "strict_audit_required": False,
        "strict_audit_reason": "endpoint evidence does not justify strict escalation",
        "full_chart_search_qualified": qualification["search_qualified"] is True,
        "full_chart_evidence_available": covered,
        "maximum_lower_empirical_descent_magnitude": lower_bound,
        "near_stationarity_plausible": False,
        "escalation_scope": (
            "Allocate strict checks only; search precision and inconclusive zero "
            "columns never certify stationarity or parameter information."
        ),
    }
    if assessment["status"] == "reliable_descent_remains":
        return result | {
            "decision": "reliable_descent_remains",
            "strict_audit_reason": "reliable feasible descent remains",
        }
    if assessment["qualified_full_chart_stationarity"]:
        return result | {
            "decision": "strict_stationarity_already_qualified",
            "strict_audit_reason": "existing strict evidence already qualifies",
            "near_stationarity_plausible": True,
        }
    if not evidence:
        return result | {
            "decision": "unavailable",
            "strict_audit_reason": assessment["reason"],
        }
    if not covered:
        return result | {"strict_audit_reason": "full-chart evidence is incomplete"}
    if qualification["failed_indices"]:
        return result | {"strict_audit_reason": "a coordinate derivative check failed"}
    if not qualification["search_qualified"]:
        return result | {
            "strict_audit_reason": "full-chart search precision is not qualified"
        }
    if lower_bound > assessment["gradient_tolerance"]:
        return result | {
            "strict_audit_reason": (
                "an inconclusive feasible gradient interval excludes stationarity"
            )
        }
    return result | {
        "decision": "strict_audit_required",
        "strict_audit_required": True,
        "near_stationarity_plausible": True,
        "strict_audit_reason": (
            "complete search-qualified feasible intervals admit the unchanged "
            "stationarity tolerance; strict evidence is required"
        ),
    }
