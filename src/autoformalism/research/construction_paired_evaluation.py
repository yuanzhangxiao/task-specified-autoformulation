"""Bounded prefix inference and forecasts for frozen affine construction fits.

Only a typed observation prefix reaches root inference. Forecasts receive a
target-free schedule and preserve every shared training parameter. Certified
affine propagation is checked against the original equations under the e7067f3
Radau tolerance ladder; failed checks never publish usable predictions.
"""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from time import monotonic, process_time

import numpy as np

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import recovery_numerics
from autoformalism.fitting.public_fitting import content_sha256


def _invalidate(result, error):
    """Never retain a success flag after incomplete verification or persistence."""
    result.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    verification._invalidate(result, error)


def _save(result, budget, save, timing_key):
    if budget.callback_failures:
        raise RuntimeError("checkpoint already failed")
    result[timing_key] = budget.snapshot()
    snapshot = deepcopy(result)
    budget.callback(lambda: save(snapshot))


def _ladder(result, evaluate, seconds, save, *, objective, accept):
    """One finite allowance covers affine and increasingly strict Radau checks."""
    budget = execution._budget(seconds, seconds + 60)
    result.update(
        verification_protocol="paired-affine-radau-ladder-1",
        verification_calls_attempted=0,
        verification_calls_started=0,
        verification_calls_completed=0,
        verification_calls_failed=0,
        replays=[],
    )
    interrupted = None

    def replay(tolerance):
        item = {"method": "certified_affine" if tolerance is None else "original_Radau"}
        if tolerance is not None:
            item.update(rtol=tolerance[0], atol=tolerance[1])
        item["status"] = "reserved"
        result["replays"].append(item)
        result["verification_calls_attempted"] += 1
        _save(result, budget, save, "verification_timing")

        def run(deadline):
            result["verification_calls_started"] += 1
            item["status"] = "started"
            try:
                values = np.asarray(evaluate(tolerance, deadline), dtype=float)
                if not values.size or not np.isfinite(values).all():
                    raise ValueError("nonfinite or empty verification values")
            except BaseException as error:
                result["verification_calls_failed"] += 1
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            result["verification_calls_completed"] += 1
            item["status"] = "complete"
            return values

        values = budget.numerical(run)
        budget.check()
        return values

    try:
        reference = replay(None)
        for tolerance in continuation._ENDPOINT_TOLERANCE_LADDER:
            alternate = replay(tolerance)
            if alternate.shape != reference.shape:
                raise ValueError("verification output shapes differ")
            gap = float(np.max(np.abs(reference - alternate), initial=0))
            passed = gap <= 1e-7
            result["solver_difference"] = gap
            if objective:
                loss, other = [float(np.mean(v * v)) for v in (reference, alternate)]
                if not np.isfinite([loss, other]).all():
                    raise ValueError("nonfinite prefix loss")
                result.update(
                    normalized_mse=loss, solver_loss_discrepancy=abs(loss - other)
                )
                passed &= abs(loss - other) <= max(1e-12, 1e-8 * max(loss, other))
            result["replays"][-1].update(
                solver_difference=gap, original_equations_verified=passed
            )
            if passed:
                accept(reference)
                break
        else:
            result["verification_error"] = (
                "affine and original Radau predictions disagree"
            )
        budget.check()
    except Exception as error:
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    with suppress(TimeoutError):
        budget.check()
    if interrupted is None and not budget.callback_failures:
        try:
            budget.check(numerical=False)
            _save(result, budget, save, "verification_timing")
            budget.check()
        except Exception as error:
            _invalidate(result, error)
        except BaseException as error:
            interrupted = error
            _invalidate(result, error)
    result["verification_timing"] = budget.snapshot()
    if interrupted is not None:
        interrupted.execution_receipt = deepcopy(result)
        raise interrupted
    return result


def infer(
    problem, fit, prefix, numerical, seconds, verification_seconds, maximum_calls, save
):
    """Fit only trajectory roots from a prefix; shared parameters remain frozen."""
    if not isinstance(prefix, kernel.PrefixView):
        raise TypeError("typed prefix-only view required")
    if type(maximum_calls) is not int or not 1 <= maximum_calls <= 10000:
        raise ValueError("finite positive root search call cap required")
    wall, cpu = monotonic(), process_time()
    budget = execution._budget(seconds, seconds + 120)
    result = {
        "protocol": "prefix-inference-1",
        "execution_protocol": "paired-affine-prefix-1",
        "status": "complete",
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit),
        "prefix_sha256": content_sha256(prefix),
        "boundary_sha256": content_sha256(kernel._boundary(prefix)),
        "prefix_end_time": prefix.time[-1],
        "prefix_forcing_sha256": content_sha256(
            kernel._prefix_forcing(prefix, prefix.time[-1])
        ),
        "cutoff": prefix.cutoff,
        "shared_parameters": deepcopy(fit["shared_parameters"]),
        "training_scales": deepcopy(fit["training_scales"]),
        "policy": numerical.model_dump(mode="json"),
        "execution_policy": {
            "root_search_seconds": seconds,
            "root_search_overall_seconds": seconds + 120,
            "verification_seconds": verification_seconds,
            "verification_overall_seconds": verification_seconds + 60,
            "maximum_calls": maximum_calls,
            "reject_invalid_trials": True,
            "root_start": "declared",
        },
        "fit_success": False,
        "root_optimization_convergence_assessed": False,
        "parameters": None,
        "roots": None,
        "search": {"best": None, "calls": 0, "stop_reason": "not_started"},
        "search_calls_attempted": 0,
        "search_calls_started": 0,
        "search_calls_completed": 0,
        "search_calls_failed": 0,
    }
    oracle, interrupted, timed = None, None, None

    def save_search(partial):
        result["search"] = deepcopy(partial)
        result.update(
            search_calls_attempted=partial.get("calls", 0),
            search_calls_started=timed.started,
            search_calls_completed=timed.completed,
            search_calls_failed=timed.failed,
        )
        _save(result, budget, save, "search_timing")

    try:
        oracle = budget.numerical(
            lambda deadline: affine.PrefixOracle(problem, fit, prefix, numerical)
        )
        budget.check()
        result.update(
            root_start=deepcopy(oracle.start),
            affine_audit=deepcopy(oracle.affine_audit),
        )
        timed = continuation._TimedOracle(oracle, budget, reject_invalid_trials=True)
        if oracle.names:
            search = recovery_numerics.search(
                timed,
                oracle.start,
                budget.wall + seconds + 120,
                save_search,
                calls=maximum_calls,
                target=1e-12,
                telemetry=True,
                record_steps=True,
                optimizer_scaling="guarded_relative",
                objective_multiplier=1.0,
                reject_invalid_trials=True,
            )
        else:
            search = {
                "best": {"parameters": {}},
                "calls": 0,
                "stop_reason": "no_free_coordinates",
            }
        save_search(search)
    except Exception as error:
        result["search_error"] = f"{type(error).__name__}: {error}"
        result["search"]["stop_reason"] = (
            "budget_limited" if isinstance(error, TimeoutError) else "numerical_failure"
        )
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    result["search_timing"] = budget.snapshot()
    if timed is not None:
        result.update(
            search_calls_started=timed.started,
            search_calls_completed=timed.completed,
            search_calls_failed=timed.failed,
        )
    if interrupted is not None:
        result["overall_timing"] = execution._overall(wall, cpu)
        interrupted.execution_receipt = deepcopy(result)
        raise interrupted
    best = result["search"].get("best")
    if best is not None and oracle is not None and not budget.callback_failures:
        parameters = deepcopy(best["parameters"])
        vector = oracle.vector(parameters)

        def evaluate(tolerance, deadline):
            if tolerance is None:
                return oracle.evaluate(vector, deadline, jacobian=False)[0]
            policy = numerical.model_copy(
                update={"rtol": tolerance[0], "atol": tolerance[1]}
            )
            reference = kernel.PrefixOracle(problem, fit, prefix, policy)
            return reference.evaluate(vector, deadline, jacobian=False, method="Radau")[
                0
            ]

        def accept(reference):
            del reference
            result.update(
                fit_success=True,
                roots=parameters,
                parameters=parameters,
                realized_initial_states=kernel._realized_initials(oracle, parameters),
            )

        _ladder(
            result, evaluate, verification_seconds, save, objective=True, accept=accept
        )
    elif budget.callback_failures:
        _invalidate(
            result, RuntimeError("prefix checkpoint failed; verification not started")
        )
    result["overall_timing"] = execution._overall(wall, cpu)
    return result


def forecast(problem, fit, inference, schedule, numerical, seconds, save):
    """Verify a free suffix forecast using no target observations after time zero."""
    if not isinstance(schedule, kernel.ForecastSchedule):
        raise TypeError("typed target-free forecast schedule required")
    kernel._validate_fit(problem, fit)
    if not inference.get("fit_success") or inference.get(
        "fit_sha256"
    ) != content_sha256(fit):
        raise ValueError("verified inference from this frozen fit required")
    if inference.get("problem_sha256") != content_sha256(problem) or inference.get(
        "boundary_sha256"
    ) != content_sha256(kernel._boundary(schedule)):
        raise ValueError("forecast initial boundary or policy differs from inference")
    if (
        inference.get("prefix_end_time") not in schedule.time
        or inference["prefix_end_time"] > schedule.cutoff
    ):
        raise ValueError("forecast prefix ends outside the permitted window")
    if inference.get("cutoff") != schedule.cutoff or inference.get(
        "prefix_forcing_sha256"
    ) != content_sha256(kernel._prefix_forcing(schedule, inference["prefix_end_time"])):
        raise ValueError("forecast prefix time/input history differs from inference")
    if (
        inference.get("shared_parameters") != fit["shared_parameters"]
        or inference.get("training_scales") != fit["training_scales"]
    ):
        raise ValueError("inference changed frozen shared parameters or scales")
    wall, cpu = monotonic(), process_time()
    result = {
        "protocol": "prefix-forecast-1",
        "execution_protocol": "paired-affine-forecast-1",
        "execution_policy": {
            "verification_seconds": seconds,
            "verification_overall_seconds": seconds + 60,
        },
        "status": "complete",
        "trajectory_id": schedule.trajectory_id,
        "time": list(schedule.time),
        "cutoff": schedule.cutoff,
        "forecast_success": False,
        "predictions": {},
        "states": {},
        "inference_sha256": content_sha256(inference),
        "schedule_sha256": content_sha256(schedule),
        "shared_parameters": deepcopy(fit["shared_parameters"]),
        "training_scales": deepcopy(fit["training_scales"]),
    }
    context = {}

    def evaluate(tolerance, deadline):
        if not context:
            system, _, audit, settings, variables = kernel._compile(problem, numerical)
            kernel._validate_future(system, schedule)
            row = kernel._trajectory(schedule)
            kernel._validate_row(system, row)
            if set(inference["roots"]) != set(audit["trajectory_parameter_names"]):
                raise ValueError("inferred root coordinates differ")
            roots = set(audit["trajectory_parameter_names"])
            if set(fit["shared_parameters"]) != set(system.names) - roots:
                raise ValueError("frozen shared parameter names differ")
            vector = kernel._physical_vector(
                system,
                variables,
                dict(fit["shared_parameters"]) | dict(inference["roots"]),
            )
            if set(fit["training_scales"]) != set(system.channels):
                raise ValueError("frozen scale channels differ")
            scales = np.asarray(
                [fit["training_scales"][name] for name in system.channels]
            )
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("invalid frozen training scales")
            propagation = affine._Propagation(system, ())
            context.update(
                system=system,
                row=row,
                vector=vector,
                settings=settings,
                scales=scales,
                propagation=propagation,
            )
            result["affine_audit"] = deepcopy(propagation.audit)
        affine._check_deadline(deadline)
        if tolerance is None:
            values, _ = context["propagation"].rollout(
                context["row"], context["vector"], deadline, jacobian=False, cache={}
            )
        else:
            settings = context["settings"].model_copy(
                update={
                    "integration_method": "Radau",
                    "relative_tolerance": tolerance[0],
                    "absolute_tolerance": tolerance[1],
                }
            )
            values, _, _, _ = kernel.symbolic_rollout(
                context["system"],
                context["row"],
                context["vector"],
                settings,
                deadline,
                sensitivities=False,
            )
        return values / context["scales"]

    def accept(reference):
        values = reference * context["scales"]
        result.update(
            forecast_success=True,
            predictions={
                name: values[:, index].tolist()
                for index, name in enumerate(context["system"].channels)
            },
        )

    _ladder(result, evaluate, seconds, save, objective=False, accept=accept)
    result["overall_timing"] = execution._overall(wall, cpu)
    return result
