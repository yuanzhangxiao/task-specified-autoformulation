"""Opt-in affine prefix recovery with frozen fits and implicit ODE verification.

Affine dynamics alone do not imply affine root residuals: causal initialization
and every composite readout are certified separately before bounded linear least
squares. All other supported affine systems use the existing bounded root search.
Stage allowances are cooperative and exclude synchronous checkpoint time only.
The caller owns durable operation journals and must not automatically retry an
interrupted reservation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import copy, deepcopy
from time import monotonic, process_time

import casadi as ca
import numpy as np
from scipy.optimize import lsq_linear

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import recovery_numerics
from autoformalism.fitting.affine_precision import AffinePrecisionPolicy
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.prefix_initialization import PrefixFitProblem

PROTOCOL = "portfolio-affine-prefix-recovery-1"


def _failure(result: dict, error: BaseException, stage: str) -> None:
    """Keep measured solver context without turning numerical errors into proofs."""
    result.setdefault("failures", []).append(
        {"stage": stage, "type": type(error).__name__, "message": str(error)}
        | (
            {"diagnostic": continuation._bounded_diagnostic(error.diagnostic)}
            if hasattr(error, "diagnostic")
            else {}
        )
    )
    verification._invalidate(result, error)


def _save(result: dict, budget: verification._Budget, checkpoint) -> None:
    """Persist a measured lower-bound receipt; callback failures stop the stage."""
    if checkpoint is not None:
        result["timing"] = budget.snapshot()
        snapshot = deepcopy(result)
        budget.callback(lambda: checkpoint(snapshot))


def _checks(result, budget, checkpoint, evaluate: Callable, qualify: Callable) -> dict:
    """Compare affine propagation with one independently integrated original ODE."""
    interrupted = None
    try:
        solutions = []
        for method in ("affine", "Radau"):
            budget.check()
            result["verification_calls_attempted"] += 1
            _save(result, budget, checkpoint)
            budget.check()

            def call(deadline, method=method):
                result["verification_calls_started"] += 1
                try:
                    value = evaluate(method, deadline)
                except BaseException as error:
                    result["verification_calls_failed"] += 1
                    _failure(result, error, method)
                    raise
                result["verification_calls_completed"] += 1
                return value

            value = budget.numerical(call)
            budget.check()
            solutions.append(value)
        qualify(solutions)
        budget.check()
    except Exception as error:
        _failure(result, error, "verification")
    except BaseException as error:
        interrupted = error
        _failure(result, error, "verification")
    finally:
        with suppress(TimeoutError):
            budget.check()
        if interrupted is None and not budget.callback_failures:
            try:
                budget.check(numerical=False)
                _save(result, budget, checkpoint)
                budget.check()
            except Exception as error:
                _failure(result, error, "verification_checkpoint")
            except BaseException as error:
                interrupted = error
                _failure(result, error, "verification_checkpoint")
        result["timing"] = budget.snapshot()
    if interrupted is not None:
        interrupted.verification_receipt = deepcopy(result)
        raise interrupted
    return result


def _root_certificate(oracle: affine.PrefixOracle) -> dict:
    """Prove sufficient symbolic conditions for an affine residual in all roots."""
    system = oracle.system
    roots = ca.SX.sym("prefix_roots", len(oracle.names))
    theta = ca.vertcat(
        *[
            roots[oracle.names.index(name)]
            if name in oracle.names
            else oracle.frozen[name]
            for name in system.names
        ]
    )
    initial = system.initial_symbolic(oracle.rows[0], theta)
    initial_jacobian = ca.jacobian(initial, roots)
    state = ca.SX.sym("prefix_state", system.state_count)
    time = ca.SX.sym("prefix_time")
    inputs = ca.SX.sym("prefix_inputs", len(system.inputs))
    observation = system.observe(time, state, theta, inputs)
    coordinates = ca.vertcat(state, roots)
    observation_jacobian = ca.jacobian(observation, coordinates)
    conditions = {
        "generator_independent_of_roots": not ca.depends_on(
            oracle._propagation.generator(theta), roots
        ),
        "initial_state_affine_in_roots": not ca.depends_on(initial_jacobian, roots),
        "readouts_jointly_affine_in_state_and_roots": not ca.depends_on(
            observation_jacobian, coordinates
        ),
    }
    return {
        "protocol": "prefix-affine-root-residual-certificate-1",
        "certified": all(conditions.values()),
        "conditions": conditions,
        "root_order": list(oracle.names),
        "scope": "sufficient symbolic certificate at frozen shared coefficients",
    }


class _RecordedOracle(execution._SearchOracle):
    """Retain structured failures even when the inherited optimizer catches them."""

    def evaluate(self, vector, deadline):
        try:
            return super().evaluate(vector, deadline)
        except BaseException as error:
            _failure(self.result, error, "search_rollout")
            raise


def _linear_search(oracle, measured, budget, numerical, save) -> dict:
    """Solve the certified residual in scaled root coordinates with box bounds."""
    result = {
        "best": None,
        "calls": 0,
        "completed_calls": 0,
        "stop_reason": "not_started",
        "budget_exhausted": False,
        "method": "bounded_linear_least_squares",
        "linear_solves_started": 0,
        "linear_solves_completed": 0,
    }
    start = oracle.vector(oracle.start)

    def evaluate(vector):
        result["calls"] += 1
        save(result)
        residual, derivative = measured.evaluate(vector, None)
        result["completed_calls"] += 1
        value = float(np.mean(residual**2))
        if result["best"] is None or value < result["best"]["training_nmse"]:
            result["best"] = {
                "parameters": oracle.parameters(vector),
                "training_nmse": value,
            }
        save(result)
        return residual, derivative

    try:
        residual, derivative = evaluate(start)
        if numerical.maximum_calls < 2:
            result.update(stop_reason="call_limit", budget_exhausted=True)
        else:
            units = oracle.units
            lower, upper = (
                (oracle.lower - start) / units,
                (oracle.upper - start) / units,
            )
            free = lower < upper
            step = np.zeros(len(start))
            budget.check()

            def solve(deadline):
                # scipy's bounded solve has no interrupt callback; both the
                # iteration cap and a checked inclusive deadline apply.
                del deadline
                result["linear_solves_started"] += 1
                solved = (
                    lsq_linear(
                        derivative[:, free] * units[free],
                        -residual,
                        bounds=(lower[free], upper[free]),
                        tol=1e-12,
                        max_iter=numerical.maximum_calls,
                    )
                    if np.any(free)
                    else None
                )
                result["linear_solves_completed"] += 1
                return solved

            solved = budget.numerical(solve)
            budget.check()
            if solved is not None:
                step[free] = solved.x
                result["linear_solver_status"] = int(solved.status)
                result["linear_solver_iterations"] = int(solved.nit)
                result["linear_solver_success"] = bool(solved.success)
            events = []
            point = recovery_numerics.restore(
                start, units, step, oracle.lower, oracle.upper, oracle.names, events
            )
            result["domain_events"] = events
            evaluate(point)
            result.update(
                stop_reason="linear_solve_completed"
                if solved is None or solved.success
                else "linear_iteration_limit",
                budget_exhausted=bool(solved is not None and not solved.success),
            )
    except TimeoutError:
        result.update(stop_reason="budget_limited", budget_exhausted=True)
    return result


def infer_prefix(
    problem: PrefixFitProblem,
    fit_result: dict,
    prefix: kernel.PrefixView,
    numerical_policy: kernel.NumericalPolicy,
    execution_policy: execution.SearchExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
    *,
    root_guesses: Mapping[str, float] | None = None,
    precision_policy: AffinePrecisionPolicy | None = None,
) -> dict:
    """Infer fresh prefix roots, freezing shared coefficients and training scales."""
    if not isinstance(execution_policy, execution.SearchExecutionPolicy):
        raise TypeError("typed search execution policy required")
    wall, cpu = monotonic(), process_time()
    budget = execution._budget(
        numerical_policy.seconds, execution_policy.search_overall_seconds
    )
    oracle = affine.PrefixOracle(
        problem,
        fit_result,
        prefix,
        numerical_policy,
        root_guesses=root_guesses,
        precision_policy=precision_policy or AffinePrecisionPolicy(),
    )
    result = verification._base_result(
        verification.ExecutionPolicy(
            numerical_seconds=numerical_policy.verification_seconds,
            overall_seconds=execution_policy.verification_overall_seconds,
        )
    ) | {
        "protocol": "prefix-inference-1",
        "recovery_protocol": PROTOCOL,
        "execution_policy": execution_policy.model_dump(mode="json"),
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "boundary_sha256": content_sha256(kernel._boundary(prefix)),
        "prefix_end_time": prefix.time[-1],
        "prefix_forcing_sha256": content_sha256(
            kernel._prefix_forcing(prefix, prefix.time[-1])
        ),
        "cutoff": prefix.cutoff,
        "shared_parameters": dict(oracle.frozen),
        "training_scales": dict(oracle.scales),
        "root_start": dict(oracle.start),
        "policy": numerical_policy.model_dump(mode="json"),
        "fit_success": False,
        "parameters": None,
        "roots": None,
        "realized_initial_states": None,
        "no_search_performed": False,
        "search_calls_attempted": 0,
        "search_calls_started": 0,
        "search_calls_completed": 0,
        "search_calls_failed": 0,
        "verification_timing": None,
        "failures": [],
        "verification_methods": ["affine", "Radau"],
        "affine_audit": oracle.affine_audit,
        "root_residual_certificate": _root_certificate(oracle),
        "search": {
            "best": None,
            "calls": 0,
            "completed_calls": 0,
            "stop_reason": "not_started",
            "budget_exhausted": False,
        },
    }
    verification_budget = None
    interrupted = None

    def save(search):
        # Keep live counters if an interrupt occurs between durable callbacks;
        # _save makes the isolated copy actually sent to the caller.
        result["search"] = search
        result["search_calls_attempted"] = search["calls"]
        result["search_timing"] = budget.snapshot()
        _save(result, budget, checkpoint)

    try:
        budget.check()
        measured = _RecordedOracle(oracle, budget, result)
        if not oracle.names:
            result["search"].update(
                best={"parameters": {}, "training_nmse": None},
                stop_reason="no_free_coordinates",
            )
        elif result["root_residual_certificate"]["certified"]:
            result["search"] = _linear_search(
                oracle, measured, budget, numerical_policy, save
            )
        else:
            result["search"] = recovery_numerics.search(
                measured,
                oracle.start,
                budget.wall + execution_policy.search_overall_seconds,
                save,
                calls=numerical_policy.maximum_calls,
                target=-1.0,
                telemetry=True,
                optimizer_scaling="guarded_relative",
                objective_multiplier=1.0,
            )
            result["search"].setdefault(
                "budget_exhausted", result["search"]["stop_reason"] == "budget_limited"
            )
            result["search"]["method"] = "bounded_nonlinear_affine_propagation"
        result["search_calls_attempted"] = result["search"]["calls"]
        with suppress(TimeoutError):
            budget.check()
        budget.check(numerical=False)
        result["search_timing"] = budget.snapshot()
        best = result["search"]["best"]
        if best is not None:
            verification_budget = execution._budget(
                numerical_policy.verification_seconds,
                execution_policy.verification_overall_seconds,
            )
            parameters = deepcopy(best["parameters"])
            vector = oracle.vector(parameters)

            def evaluate(method, deadline):
                return oracle.evaluate(
                    vector,
                    deadline,
                    jacobian=method == "affine",
                    method=None if method == "affine" else method,
                )

            def qualify(solutions):
                residual, derivative = solutions[0]
                difference = float(
                    np.max(np.abs(residual - solutions[1][0]), initial=0.0)
                )
                verified = difference <= numerical_policy.solver_tolerance
                result.update(
                    fit_success=verified,
                    parameters=parameters if verified else None,
                    roots=deepcopy(parameters) if verified else None,
                    normalized_mse=float(np.mean(residual**2)),
                    solver_difference=difference,
                    rank_diagnostic=kernel._rank(derivative, oracle.units),
                    realized_initial_states=kernel._realized_initials(
                        oracle, parameters if verified else None
                    ),
                )
                if not verified:
                    result["verification_error"] = (
                        "affine/original prefix integrators disagree"
                    )
                else:
                    result.pop("verification_error", None)

            _checks(result, verification_budget, checkpoint, evaluate, qualify)
    except Exception as error:
        _failure(result, error, "prefix_inference")
    except BaseException as error:
        interrupted = error
        _failure(result, error, "prefix_inference")
    finally:
        execution._counts(result)
        if verification_budget is None:
            with suppress(TimeoutError):
                budget.check()
            if interrupted is None and not budget.callback_failures:
                try:
                    budget.check(numerical=False)
                    _save(result, budget, checkpoint)
                except Exception as error:
                    _failure(result, error, "search_checkpoint")
                except BaseException as error:
                    interrupted = error
                    _failure(result, error, "search_checkpoint")
            result["search_timing"] = budget.snapshot()
        else:
            result["verification_timing"] = verification_budget.snapshot()
        result["overall_timing"] = execution._overall(wall, cpu)
    if interrupted is not None:
        interrupted.execution_receipt = deepcopy(result)
        interrupted.verification_receipt = deepcopy(result)
        raise interrupted
    return result


def forecast(
    problem: PrefixFitProblem,
    fit_result: dict,
    inference: dict,
    schedule: kernel.ForecastSchedule,
    numerical_policy: kernel.NumericalPolicy,
    execution_policy: verification.ExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
    *,
    precision_policy: AffinePrecisionPolicy | None = None,
) -> dict:
    """Forecast a target-free schedule with affine/original-equation agreement."""
    budget = verification._Budget(execution_policy)
    if not isinstance(schedule, kernel.ForecastSchedule):
        raise TypeError("typed target-free forecast schedule required")
    kernel._validate_fit(problem, fit_result)
    if not inference.get("fit_success") or inference.get(
        "fit_sha256"
    ) != content_sha256(fit_result):
        raise ValueError("verified inference from this frozen fit required")
    expected = {
        "problem_sha256": content_sha256(problem),
        "boundary_sha256": content_sha256(kernel._boundary(schedule)),
        "policy": numerical_policy.model_dump(mode="json"),
        "shared_parameters": fit_result["shared_parameters"],
        "training_scales": fit_result["training_scales"],
    }
    if any(inference.get(name) != value for name, value in expected.items()):
        raise ValueError("forecast initial boundary, frozen values or policy differ")
    if inference.get("prefix_forcing_sha256") != content_sha256(
        kernel._prefix_forcing(schedule, inference["prefix_end_time"])
    ):
        raise ValueError("forecast prefix time/input history differs from inference")
    system, _, audit, settings, variables = kernel._compile(problem, numerical_policy)
    kernel._validate_future(system, schedule)
    row = kernel._trajectory(schedule)
    kernel._validate_row(system, row)
    if set(inference["roots"]) != set(audit["trajectory_parameter_names"]):
        raise ValueError("inferred root coordinates differ")
    vector = kernel._physical_vector(
        system,
        variables,
        dict(fit_result["shared_parameters"]) | dict(inference["roots"]),
    )
    # Add states to the readout of an isolated compiled-system copy. The dynamics
    # and original independent verification system remain unchanged.
    combined = copy(system)
    time, state = ca.SX.sym("time"), ca.SX.sym("state", system.state_count)
    theta, inputs = (
        ca.SX.sym("theta", len(system.names)),
        ca.SX.sym("input", len(system.inputs)),
    )
    combined.observe = ca.Function(
        "forecast_outputs_and_states",
        [time, state, theta, inputs],
        [ca.vertcat(system.observe(time, state, theta, inputs), state)],
    )
    propagation = affine._Propagation(
        combined, (), precision_policy or AffinePrecisionPolicy()
    )
    result = verification._base_result(execution_policy) | {
        "protocol": "prefix-forecast-1",
        "recovery_protocol": PROTOCOL,
        "trajectory_id": row.trajectory_id,
        "time": row.time.tolist(),
        "cutoff": schedule.cutoff,
        "forecast_success": False,
        "predictions": {},
        "states": {},
        "solver_counters": {},
        "inference_sha256": content_sha256(inference),
        "schedule_sha256": content_sha256(schedule),
        "failures": [],
        "verification_methods": ["affine", "Radau"],
        "affine_audit": propagation.audit,
    }

    def evaluate(method, deadline):
        if method == "affine":
            values, _ = propagation.rollout(
                row, vector, deadline, jacobian=False, cache={}
            )
            count = len(system.channels)
            return values[:, :count], values[:, count:]
        values, _, states, counters = kernel.symbolic_rollout(
            system,
            row,
            vector,
            settings.model_copy(update={"integration_method": method}),
            deadline,
        )
        result["solver_counters"][method] = counters
        return values, states

    def qualify(solutions):
        scales = np.asarray(
            [fit_result["training_scales"][name] for name in system.channels]
        )
        difference = float(np.max(np.abs((solutions[0][0] - solutions[1][0]) / scales)))
        result["solver_difference"] = difference
        if difference <= numerical_policy.solver_tolerance:
            values, states = solutions[0]
            result.update(
                forecast_success=True,
                predictions={
                    name: values[:, i].tolist()
                    for i, name in enumerate(system.channels)
                },
                states={
                    name: states[:, i].tolist()
                    for i, name in enumerate(system.model.state_names)
                },
            )
        else:
            result["verification_error"] = (
                "affine/original forecast integrators disagree"
            )

    return _checks(result, budget, checkpoint, evaluate, qualify)
