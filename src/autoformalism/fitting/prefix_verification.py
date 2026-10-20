"""Verification-only recovery of frozen prefix candidates and free forecasts.

The M44 numerical tolerances and independent integrators are unchanged. Only
synchronous checkpoint time is excluded from the numerical allowance; it remains
fully charged to a separate overall wall allowance. Deadlines are cooperative:
an already blocking callback cannot be preempted, but expiry prevents more work.
Callers own durable journals and must not automatically retry interrupted work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from time import monotonic, process_time
from typing import Literal, TypeVar

import numpy as np
from pydantic import Field

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem

T = TypeVar("T")


class ExecutionPolicy(StrictSchema):
    """Finite numerical-active and inclusive wall allowances, frozen separately."""

    protocol: Literal["prefix-verification-execution-1"] = (
        "prefix-verification-execution-1"
    )
    numerical_seconds: FiniteFloat = Field(default=5, gt=0, le=300)
    overall_seconds: FiniteFloat = Field(default=60, gt=0, le=3600)


class _CheckpointFailure(RuntimeError):
    """A failed durable callback prohibits further numerical work."""


class _Budget:
    """Disjoint timers with checks after callbacks, numerical work and sealing."""

    def __init__(self, policy: ExecutionPolicy):
        if not isinstance(policy, ExecutionPolicy):
            raise TypeError("typed verification execution policy required")
        self.policy = policy
        self.wall, self.cpu = monotonic(), process_time()
        self.callback_wall = self.callback_cpu = 0.0
        self.numerical_wall = self.numerical_cpu = 0.0
        self.callback_calls = self.callback_failures = 0
        self.timeout_reason: str | None = None

    def check(self, *, numerical: bool = True) -> None:
        """Include all elapsed wall time except callbacks in numerical use."""
        elapsed = monotonic() - self.wall
        if elapsed >= self.policy.overall_seconds:
            self.timeout_reason = "overall_wall_limit"
        elif (
            numerical and elapsed - self.callback_wall >= self.policy.numerical_seconds
        ):
            self.timeout_reason = "numerical_active_wall_limit"
        else:
            return
        raise TimeoutError(f"prefix verification {self.timeout_reason} reached")

    def deadline(self) -> float:
        """Recompute a finite absolute integrator deadline after each callback."""
        self.check()
        now = monotonic()
        remaining = self.policy.numerical_seconds - (
            now - self.wall - self.callback_wall
        )
        return min(self.wall + self.policy.overall_seconds, now + remaining)

    def callback(self, function: Callable[[], None]) -> None:
        self.check(numerical=False)
        wall, cpu = monotonic(), process_time()
        self.callback_calls += 1
        try:
            function()
        except BaseException as error:
            self.callback_failures += 1
            if isinstance(error, Exception):
                raise _CheckpointFailure(
                    f"prefix verification checkpoint failed: {error}"
                ) from error
            raise
        finally:
            self.callback_wall += monotonic() - wall
            self.callback_cpu += process_time() - cpu
            # Keep expiry evidence even when the callback raises or interrupts;
            # preserve that original exception rather than replacing it here.
            with suppress(TimeoutError):
                self.check(numerical=False)
        self.check(numerical=False)

    def numerical(self, function: Callable[[float], T]) -> T:
        deadline = self.deadline()
        wall, cpu = monotonic(), process_time()
        try:
            return function(deadline)
        finally:
            self.numerical_wall += monotonic() - wall
            self.numerical_cpu += process_time() - cpu

    def snapshot(self) -> dict:
        elapsed, cpu = monotonic() - self.wall, process_time() - self.cpu
        return {
            "protocol": "prefix-verification-timing-1",
            "elapsed_wall_seconds": elapsed,
            "elapsed_cpu_seconds": cpu,
            "callback_wall_seconds": self.callback_wall,
            "callback_cpu_seconds": self.callback_cpu,
            "numerical_active_wall_seconds": max(0.0, elapsed - self.callback_wall),
            "numerical_wall_seconds": self.numerical_wall,
            "numerical_cpu_seconds": self.numerical_cpu,
            "callback_calls": self.callback_calls,
            "callback_failures": self.callback_failures,
            "timeout_reason": self.timeout_reason,
            "scope": (
                "Process CPU and monotonic wall include setup, failures and final "
                "callbacks. Only measured synchronous callbacks are excluded from "
                "numerical-active wall; descheduling remains charged. Numerical "
                "timers cover oracle calls, including residual assembly. Progress "
                "snapshots exclude their currently executing callback. Blocking "
                "calls may return after the cooperative overall deadline."
            ),
        }


def _base_result(execution: ExecutionPolicy) -> dict:
    return {
        "calls": 0,
        "completed_calls": 0,
        "no_search_performed": True,
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
        "execution_policy": execution.model_dump(mode="json"),
    }


def _invalidate(result: dict, error: BaseException) -> None:
    """Never preserve a qualified result after a deadline or callback failure."""
    result["verification_error"] = f"{type(error).__name__}: {error}"
    if "fit_success" in result:
        result.update(
            fit_success=False, parameters=None, roots=None, realized_initial_states=None
        )
    if "forecast_success" in result:
        result.update(forecast_success=False, predictions={}, states={})


def _run(
    result: dict,
    budget: _Budget,
    checkpoint: kernel.Checkpoint | None,
    evaluate: Callable[[str, float], T],
    qualify: Callable[[list[T]], None],
) -> dict:
    """Run exactly two checks and preserve terminal timing even on interruption."""
    interrupted: BaseException | None = None

    def save() -> None:
        if checkpoint is not None:
            result["timing"] = budget.snapshot()
            snapshot = deepcopy(result)
            budget.callback(lambda: checkpoint(snapshot))

    try:
        solutions = []
        for method in ("DOP853", "Radau"):
            budget.check()
            result["verification_calls_attempted"] += 1
            save()
            budget.check()

            def call(deadline: float, method: str = method) -> T:
                # A reservation is not a call until its checkpoint and deadline
                # checks return and the numerical function actually starts.
                result["verification_calls_started"] += 1
                try:
                    return evaluate(method, deadline)
                except BaseException:
                    result["verification_calls_failed"] += 1
                    raise

            solution = budget.numerical(call)
            result["verification_calls_completed"] += 1
            budget.check()
            solutions.append(solution)
        qualify(solutions)
        budget.check()
    except Exception as error:
        if isinstance(error, TimeoutError) and budget.timeout_reason is None:
            with suppress(TimeoutError):
                budget.check()
            if budget.timeout_reason is None:
                budget.timeout_reason = "integration_timeout"
        _invalidate(result, error)
    except BaseException as error:
        interrupted = error
        _invalidate(result, error)
    finally:
        # Failed callbacks and interrupted numerical calls also consume their
        # full elapsed interval, even when the original exception is not a timeout.
        with suppress(TimeoutError):
            budget.check()
        # Do not retry a failed callback or start another after overall expiry.
        if interrupted is None and not budget.callback_failures:
            try:
                budget.check(numerical=False)
                save()
                budget.check()
            except Exception as error:
                _invalidate(result, error)
            except BaseException as error:
                interrupted = error
                _invalidate(result, error)
        result["timing"] = budget.snapshot()
        if interrupted is not None:
            # The owning journal can persist the fully measured receipt while
            # preserving the interrupt, rather than treating it as a retry.
            interrupted.verification_receipt = deepcopy(result)
    if interrupted is not None:
        raise interrupted
    return result


def _saved_candidate(
    problem: PrefixFitProblem,
    fit_result: dict,
    prefix: kernel.PrefixView,
    old: dict,
    policy: kernel.NumericalPolicy,
) -> tuple[kernel.PrefixOracle, np.ndarray]:
    """Bind the saved failed inference to the exact permitted prefix and fit."""
    if not isinstance(prefix, kernel.PrefixView):
        raise TypeError("typed prefix-only view required")
    if (
        old.get("protocol") != "prefix-inference-1"
        or old.get("fit_success") is not False
    ):
        raise ValueError("saved failed prefix inference required")
    expected = {
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit_result),
        "prefix_sha256": content_sha256(prefix),
        "boundary_sha256": content_sha256(kernel._boundary(prefix)),
        "prefix_end_time": prefix.time[-1],
        "prefix_forcing_sha256": content_sha256(
            kernel._prefix_forcing(prefix, prefix.time[-1])
        ),
        "cutoff": prefix.cutoff,
        "shared_parameters": fit_result["shared_parameters"],
        "training_scales": fit_result["training_scales"],
        "policy": policy.model_dump(mode="json"),
    }
    if any(old.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "saved inference identity, frozen values or numerical policy differ"
        )
    if any(old.get(name) is not None for name in ("parameters", "roots")):
        raise ValueError("failed inference cannot supply verified parameters")
    if not isinstance(old.get("root_start"), Mapping):
        raise ValueError("saved inference root_start required")
    oracle = kernel.PrefixOracle(
        problem, fit_result, prefix, policy, root_guesses=old["root_start"]
    )
    search = old.get("search")
    best = search.get("best") if isinstance(search, Mapping) else None
    if not isinstance(best, Mapping) or not isinstance(best.get("parameters"), Mapping):
        raise ValueError("saved search best candidate required")
    loss = best.get("training_nmse")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (float, int))
        or not np.isfinite(loss)
        or loss < 0
    ):
        raise ValueError("saved search best objective must be finite and nonnegative")
    return oracle, oracle.vector(best["parameters"])


def verify_inference(
    problem: PrefixFitProblem,
    fit_result: dict,
    prefix: kernel.PrefixView,
    old_inference: dict,
    numerical_policy: kernel.NumericalPolicy,
    execution_policy: ExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
) -> dict:
    """Verify exactly the saved failed candidate without search or suffix access."""
    budget = _Budget(execution_policy)
    oracle, vector = _saved_candidate(
        problem, fit_result, prefix, old_inference, numerical_policy
    )
    parameters = deepcopy(old_inference["search"]["best"]["parameters"])
    result = _base_result(execution_policy) | {
        "protocol": "prefix-inference-1",
        "supplement_protocol": "prefix-inference-verification-supplement-1",
        "original_inference_sha256": content_sha256(old_inference),
        "search": deepcopy(old_inference["search"]),
        "historical_search_reused": True,
        "fit_success": False,
        "parameters": None,
        "roots": None,
        "realized_initial_states": None,
        **{
            name: deepcopy(old_inference[name])
            for name in (
                "problem_sha256",
                "fit_sha256",
                "prefix_sha256",
                "boundary_sha256",
                "prefix_end_time",
                "prefix_forcing_sha256",
                "cutoff",
                "shared_parameters",
                "training_scales",
                "policy",
                "root_start",
            )
        },
    }

    def evaluate(method: str, deadline: float) -> tuple[np.ndarray, np.ndarray | None]:
        return oracle.evaluate(
            vector, deadline, jacobian=method == "DOP853", method=method
        )

    def qualify(solutions: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
        residual, derivative = solutions[0]
        difference = float(np.max(np.abs(residual - solutions[1][0]), initial=0.0))
        verified = difference <= numerical_policy.solver_tolerance
        result.update(
            fit_success=verified,
            parameters=deepcopy(parameters) if verified else None,
            roots=deepcopy(parameters) if verified else None,
            normalized_mse=float(np.mean(residual**2)),
            solver_difference=difference,
            rank_diagnostic=kernel._rank(derivative, oracle.units),
            realized_initial_states=kernel._realized_initials(
                oracle, parameters if verified else None
            ),
        )
        if not verified:
            result["verification_error"] = "independent prefix integrators disagree"

    return _run(result, budget, checkpoint, evaluate, qualify)


def forecast(
    problem: PrefixFitProblem,
    fit_result: dict,
    inference: dict,
    schedule: kernel.ForecastSchedule,
    numerical_policy: kernel.NumericalPolicy,
    execution_policy: ExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
) -> dict:
    """Repeat the kernel's target-free forecast with separately measured callbacks."""
    budget = _Budget(execution_policy)
    if not isinstance(schedule, kernel.ForecastSchedule):
        raise TypeError("typed target-free forecast schedule required")
    kernel._validate_fit(problem, fit_result)
    if not inference.get("fit_success") or inference.get(
        "fit_sha256"
    ) != content_sha256(fit_result):
        raise ValueError("verified inference from this frozen fit required")
    if (
        inference.get("problem_sha256") != content_sha256(problem)
        or inference.get("boundary_sha256")
        != content_sha256(kernel._boundary(schedule))
        or inference.get("policy") != numerical_policy.model_dump(mode="json")
        or inference.get("shared_parameters") != fit_result["shared_parameters"]
        or inference.get("training_scales") != fit_result["training_scales"]
    ):
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
    parameters = dict(fit_result["shared_parameters"]) | dict(inference["roots"])
    vector = kernel._physical_vector(system, variables, parameters)
    result = _base_result(execution_policy) | {
        "protocol": "prefix-forecast-1",
        "supplement_protocol": "prefix-forecast-verification-supplement-1",
        "trajectory_id": row.trajectory_id,
        "time": row.time.tolist(),
        "cutoff": schedule.cutoff,
        "forecast_success": False,
        "predictions": {},
        "states": {},
        "solver_counters": {},
        "inference_sha256": content_sha256(inference),
        "schedule_sha256": content_sha256(schedule),
    }

    def evaluate(method: str, deadline: float) -> tuple[np.ndarray, np.ndarray]:
        values, _, states, counters = kernel.symbolic_rollout(
            system,
            row,
            vector,
            settings.model_copy(update={"integration_method": method}),
            deadline,
        )
        result["solver_counters"][method] = counters
        return values, states

    def qualify(solutions: list[tuple[np.ndarray, np.ndarray]]) -> None:
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
                    name: values[:, index].tolist()
                    for index, name in enumerate(system.channels)
                },
                states={
                    name: states[:, index].tolist()
                    for index, name in enumerate(system.model.state_names)
                },
            )
        else:
            result["verification_error"] = "independent forecast integrators disagree"

    return _run(result, budget, checkpoint, evaluate, qualify)
