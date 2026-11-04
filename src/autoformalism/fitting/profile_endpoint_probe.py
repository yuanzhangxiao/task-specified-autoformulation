"""Bound-aware small-step probes of a conditionally profiled training objective.

These are finite, local descent diagnostics, not a stationarity certificate.
An observed downhill direction can refute a stopping interpretation; the absence
of one cannot prove joint stationarity, identifiability or equation inadequacy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "profile-endpoint-local-probe-1"


def _chart(oracle, parameters, outer, fixed, chart):
    """Refreeze physical units at this endpoint using the search's scaling rule."""
    if set(fixed) - {oracle.names[i] for i in outer}:
        raise ValueError("fixed coordinates must belong to the outer problem")
    if any(parameters.get(name) != value for name, value in fixed.items()):
        raise ValueError("endpoint differs from its fixed coordinates")
    initial = oracle.vector(parameters)
    start = initial.copy() if chart is None else chart.encode(parameters)
    names = oracle.names if chart is None else chart.names
    lower = oracle.lower if chart is None else chart.lower
    upper = oracle.upper if chart is None else chart.upper
    units = oracle.units if chart is None else chart.units(parameters)
    if chart is not None:
        chart.validate_use(oracle, fixed)
    units = np.maximum(np.abs(start), units)
    if not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("positive finite frozen probe units required")
    logarithmic = {b.tau for b in getattr(oracle.reduction, "blocks", ())}
    if chart is not None:
        logarithmic.add(chart.rate)
    coordinates = []
    for index in outer:
        if oracle.names[index] in fixed or lower[index] == upper[index]:
            continue
        log = names[index] in logarithmic
        if log and start[index] <= 0:
            raise ValueError("logarithmic probes require a positive endpoint")
        if log:
            lo = np.log(lower[index] / start[index]) if lower[index] > 0 else -np.inf
            hi = (
                np.log(upper[index] / start[index])
                if np.isfinite(upper[index])
                else np.inf
            )
        else:
            lo = (lower[index] - start[index]) / units[index]
            hi = (upper[index] - start[index]) / units[index]
        coordinates.append((index, names[index], log, float(lo), float(hi)))

    def point(index, log, step):
        trial = start.copy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            trial[index] = (
                np.exp(np.log(start[index]) + step)
                if log
                else start[index] + step * units[index]
            )
        if not np.isfinite(trial).all() or (log and trial[index] <= 0):
            raise NumericalInvalidTrial("nonrepresentable local probe")
        if trial[index] == start[index]:
            raise NumericalInvalidTrial("local probe rounded to the endpoint")
        vector = trial if chart is None else chart.decode(trial)
        values = oracle.parameters(vector)
        oracle.vector(values)
        return values

    return coordinates, units, point


def profile_endpoint_probe(
    oracle,
    parameters: Mapping[str, float],
    seconds: float,
    checkpoint: Callable[[dict], None],
    *,
    profiler_factory: Callable,
    fixed: Mapping[str, float] | None = None,
    coordinate_chart=None,
) -> dict:
    """Probe two small feasible steps per direction under a nonrenewable budget.

    Every point re-solves the same bounded inner problem. No frozen or profiled
    coordinate is tested as an outer variable. The resolution threshold is a
    conservative reporting convention, not a rigorous integration-error bound.
    The caller journals this operation; it must not auto-renew interrupted work.
    """
    budget = execution._budget(seconds, min(seconds + 120, 3600))
    result = {
        "protocol": PROTOCOL,
        "parameters_sha256": content_sha256(parameters),
        "status": "not_started",
        "complete": False,
        "calls_attempted": 0,
        "calls_started": 0,
        "calls_completed": 0,
        "calls_failed": 0,
        "coordinates": [],
        "baseline": None,
        "best_probe": None,
        "stationarity_certified": False,
        "parameter_confidence_certified": False,
        "global_minimum_certified": False,
        "independent_original_verification_required": True,
        "scope": (
            "small feasible coordinate directions in the profiled training objective"
        ),
        "rule": {
            "units": "frozen at diagnostic endpoint using the search scaling rule",
            "maximum_scaled_step": 0.001,
            "second_step_ratio": 0.5,
            "loss_resolution": "max(1e-12, 1e-8*abs(baseline_loss))",
            "material_drop": "greater than 10 times loss resolution at both steps",
            "slope_consistency": (
                "20% relative or four resolution units per smaller step"
            ),
            "absence_of_descent_is_not_stationarity": True,
        },
    }
    interrupted = None

    def save():
        budget.callback(lambda: checkpoint(deepcopy(result)))

    def evaluate(profiler, values):
        result["calls_attempted"] += 1

        def call(deadline):
            result["calls_started"] += 1
            try:
                endpoint = profiler.profile(values, deadline)
                loss = endpoint["training_nmse"]
                if not np.isfinite(loss) or loss < 0:
                    raise NumericalInvalidTrial("invalid profiled loss")
            except BaseException:
                result["calls_failed"] += 1
                raise
            result["calls_completed"] += 1
            return endpoint

        endpoint = budget.numerical(call)
        budget.check()
        loss = endpoint["training_nmse"]
        if result["best_probe"] is None or loss < result["best_probe"]["training_nmse"]:
            result["best_probe"] = deepcopy(endpoint)
            save()
        return endpoint

    save()
    try:
        profiler = budget.numerical(lambda _: profiler_factory(oracle))
        coordinates, units, point = _chart(
            oracle, parameters, profiler.outer, dict(fixed or {}), coordinate_chart
        )
        result["profile_certificate_sha256"] = profiler.audit["sha256"]
        result["fixed_coordinates"] = dict(fixed or {})
        result["inner_coordinates"] = [oracle.names[i] for i in profiler.inner]
        result["planned_calls"] = 1 + sum(
            2 * ((lo < 0) + (hi > 0)) for _, _, _, lo, hi in coordinates
        )
        baseline = evaluate(profiler, parameters)
        result["baseline"] = baseline
        f0 = baseline["training_nmse"]
        resolution = max(1e-12, 1e-8 * abs(f0))
        result["loss_resolution"] = resolution
        save()
        for index, name, log, lo, hi in coordinates:
            row = {
                "name": name,
                "unit": float(units[index]),
                "coordinate": "log_ratio" if log else "scaled_physical_difference",
                "directions": [],
            }
            result["coordinates"].append(row)
            for sign, available in ((-1, -lo), (1, hi)):
                direction = {"sign": sign, "samples": [], "status": "bound_blocked"}
                row["directions"].append(direction)
                if available <= 0:
                    continue
                step = min(0.001, available)
                direction["status"] = "incomplete"
                try:
                    for distance in (step, step / 2):
                        trial = point(index, log, sign * distance)
                        endpoint = evaluate(profiler, trial)
                        loss = endpoint["training_nmse"]
                        direction["samples"].append(
                            {
                                "distance": distance,
                                "training_nmse": loss,
                                "outward_slope": (loss - f0) / distance,
                            }
                        )
                        save()
                except (NumericalInvalidTrial, ValueError, FloatingPointError) as error:
                    direction.update(
                        status="numerically_unresolved", error=str(error)[:500]
                    )
                    save()
                    continue
                large, small = direction["samples"]
                slopes = [large["outward_slope"], small["outward_slope"]]
                tolerance = max(
                    0.2 * max(abs(v) for v in slopes), 4 * resolution / (step / 2)
                )
                consistent = abs(slopes[0] - slopes[1]) <= tolerance
                descent = all(
                    sample["training_nmse"] < f0 - 10 * resolution
                    for sample in direction["samples"]
                )
                direction.update(
                    two_scale_slopes_consistent=consistent,
                    material_decrease_at_both_steps=descent,
                    status=(
                        "descent_detected"
                        if descent and consistent
                        else "step_sensitive"
                        if not consistent
                        else "no_resolved_descent"
                    ),
                )
                save()
        result["complete"] = True
        statuses = [
            d["status"] for row in result["coordinates"] for d in row["directions"]
        ]
        result["status"] = (
            "feasible_descent_detected"
            if "descent_detected" in statuses
            else "numerically_unresolved"
            if "numerically_unresolved" in statuses
            else "step_resolution_limited"
            if "step_sensitive" in statuses
            else "no_resolved_coordinate_descent"
        )
    except Exception as error:
        result.update(
            status="incomplete", error=f"{type(error).__name__}: {error}"[:500]
        )
    except BaseException as error:
        interrupted = error
        result.update(status="interrupted", error=type(error).__name__)
    output = continuation._finish(
        result, budget, checkpoint, "endpoint_probe", interrupted
    )
    if output["status"] in {"incomplete", "interrupted"}:
        output["complete"] = False
    return output
