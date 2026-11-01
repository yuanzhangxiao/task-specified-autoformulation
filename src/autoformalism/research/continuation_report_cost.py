"""Canonical derived timings and narrowly bounded historical reconciliation.

Only addition of authenticated, nonnegative timing operands is reconciled.
Parameters, scores, statuses, call counts, unknown totals and receipt identities
remain exact. Original operation receipts and historical terminals are immutable.
"""

from __future__ import annotations

import math
from copy import deepcopy

from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal

PROTOCOL = "continuation-derived-time-reconciliation-1"


def _number(value) -> float:
    """Accept only finite, nonnegative timing quantities, never booleans."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timing operand must be finite and nonnegative")
    return float(value)


def _time_inputs(operations: dict) -> dict[str, list[float] | None]:
    """Mirror historical operand selection while flattening nested timing sums."""
    ordered = [operations[name] for name in sorted(operations)]
    inputs = {
        "budget_charge_seconds": [
            _number(op["budget_charge_seconds"]) for op in ordered
        ]
    }
    for field in ("wall_seconds", "cpu_seconds"):
        values = [
            None if op.get(field) is None else _number(op[field]) for op in ordered
        ]
        observed = []
        for op, value in zip(ordered, values, strict=True):
            progress = op.get("observed_progress") or {}
            timing = progress.get("overall_timing") or progress.get("timing") or {}
            observed.append(
                value
                if value is not None
                else _number(timing.get("elapsed_" + field, 0))
            )
        inputs[field + "_observed"] = observed
        inputs[field + "_total"] = (
            values if all(value is not None for value in values) else None
        )
    for field in (
        "callback_wall_seconds",
        "callback_cpu_seconds",
        "numerical_active_wall_seconds",
    ):
        values = []
        for op in ordered:
            value = op.get("value") or op.get("observed_progress") or {}
            values.extend(
                _number((value.get(name) or {}).get(field, 0))
                for name in ("search_timing", "verification_timing", "timing")
                if name != "timing" or "verification_timing" not in value
            )
        inputs[field + "_observed"] = values
        inputs[field + "_total"] = (
            values if all(op["status"] == "complete" for op in ordered) else None
        )
    return inputs


def canonical_cost(operations: dict) -> dict:
    """Sum derived timings reproducibly without altering historical receipt code."""
    inputs = _time_inputs(operations)
    ordered = {name: operations[name] for name in sorted(operations)}
    result = journal._cost(ordered)
    for field, values in inputs.items():
        result[field] = None if values is None else _number(math.fsum(values))
    return result


def _addition_bound(values: list[float], operation_count: int) -> float:
    """Bound naive nested addition versus a faithfully rounded sum.

    Gamma_k * sum(abs(x)) bounds binary64 rounding along any historical inner/
    outer addition path. Counting all leaves plus all operation accumulations
    safely exceeds that path length. One ulp covers the canonical sum's own
    rounding; k minimum-subnormal units cover possible underflow. No fitting or
    empirical relative-error tolerance enters this bound.
    """
    k = len(values) + operation_count + 1
    ku = k * 2.0**-53
    if ku >= 0.01:
        raise ValueError("timing summation is too large for bounded reconciliation")
    total = _number(math.fsum(values))
    if total == 0:
        return 0.0
    bound = ku / (1.0 - ku) * total + math.ulp(total) + k * math.ulp(0.0)
    if not math.isfinite(bound):
        raise ValueError("timing reconciliation bound is not finite")
    return math.nextafter(bound, math.inf)


def reconcile(saved: dict, recomputed: dict, operations: dict) -> dict:
    """Audit exact identity or derived-time roundoff; reject every other mismatch.

    The caller must already authenticate each operation's seal, reservation and
    progress. This additional boundary binds both terminal views to those exact
    operation values, recomputes canonical costs, and never modifies either view.
    """
    expected_receipts = {
        name: public.content_sha256(value) for name, value in operations.items()
    }
    if (
        saved.get("operation_receipts") != expected_receipts
        or recomputed.get("operation_receipts") != expected_receipts
    ):
        raise ValueError("terminal operation receipts differ")
    canonical = canonical_cost(operations)
    if public.content_sha256(recomputed.get("cost")) != public.content_sha256(
        canonical
    ):
        raise ValueError("recomputed timing costs are not canonical")
    inputs = _time_inputs(operations)
    left, right = deepcopy(saved), deepcopy(recomputed)
    if (
        not isinstance(left.get("cost"), dict)
        or left["cost"].keys() != right["cost"].keys()
    ):
        raise ValueError("terminal cost fields differ")
    differences = []
    for field, values in inputs.items():
        old, new = left["cost"][field], right["cost"][field]
        if old != new:
            if values is None or old is None or new is None:
                raise ValueError("terminal unknown timing total differs")
            old, new = _number(old), _number(new)
            delta = abs(old - new)
            bound = _addition_bound(values, len(operations))
            if delta > bound:
                raise ValueError(f"terminal timing differs beyond roundoff: {field}")
            differences.append(
                {
                    "path": "cost." + field,
                    "saved": old,
                    "canonical": new,
                    "absolute_difference": delta,
                    "addition_roundoff_bound": bound,
                    "operand_count": len(values),
                }
            )
        # Normalize only known derived timing fields after the bound is checked.
        # Explicit validation also rejects booleans equal to numeric zero or one.
        if values is not None:
            _number(old)
        left["cost"][field] = right["cost"][field]
    if public.content_sha256(left) != public.content_sha256(right):
        raise ValueError("terminal differs outside derived timing roundoff")
    return {
        "protocol": PROTOCOL,
        "status": "derived_time_roundoff_only" if differences else "exact",
        "saved_terminal_sha256": public.content_sha256(saved),
        "canonical_terminal_sha256": public.content_sha256(recomputed),
        "operation_receipts_sha256": public.content_sha256(expected_receipts),
        "timing_operands_sha256": public.content_sha256(inputs),
        "differences": differences,
        "original_terminal_modified": False,
        "scientific_fields_exact": True,
    }
