"""Explicit failure-aware, case-balanced statistics for the bounded pilot."""

import math
import statistics as stats


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm]
        if any(r["assessment_status"] == "pending" for r in selected):
            result[arm] = {"status": "pending"}
            continue
        errors, confirmed, possible = [], [], []
        for name in sorted({r["benchmark_id"] for r in selected}):
            case = [r for r in selected if r["benchmark_id"] == name]
            errors.append(
                stats.median(
                    [
                        r["validation_nmse"]
                        if r["status"] == "complete"
                        and r["validation_nmse"] is not None
                        and math.isfinite(r["validation_nmse"])
                        else math.inf
                        for r in case
                    ]
                )
            )
            confirmed.append(stats.mean(r["confirmed"] for r in case))
            possible.append(stats.mean(r["possible"] for r in case))
        median = stats.median(errors)
        mad = (
            stats.median(abs(x - median) for x in errors)
            if math.isfinite(median)
            else None
        )
        result[arm] = {
            "status": "complete",
            "cases": len(errors),
            "nmse_median": median if math.isfinite(median) else "+infinity",
            "nmse_mad": mad if mad is None or math.isfinite(mad) else "+infinity",
            "compliance_mean": stats.mean(confirmed),
            "compliance_sample_sd": stats.stdev(confirmed)
            if len(confirmed) > 1
            else None,
            "possible_mean": stats.mean(possible),
        }
    return result
