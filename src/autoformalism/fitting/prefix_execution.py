"""Opt-in prefix search with separately bounded search and verification time.

Historical numerical kernels are unchanged. Only synchronous checkpoint time is
excluded from each stage's numerical-active allowance. Each stage also has an
inclusive, cooperative wall deadline; blocking callbacks cannot be preempted.
The caller owns durable operation journals and must not retry interrupted work
automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import recovery_numerics
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit


class SearchExecutionPolicy(StrictSchema):
    """Independent inclusive wall caps; numerical allowances remain unchanged."""

    protocol: Literal["prefix-search-execution-1"] = "prefix-search-execution-1"
    search_overall_seconds: FiniteFloat = Field(default=120, gt=0, le=3600)
    verification_overall_seconds: FiniteFloat = Field(default=60, gt=0, le=3600)


class _StagePolicy(verification.ExecutionPolicy):
    """Permit the full existing search allowance without changing M45 defaults."""

    numerical_seconds: FiniteFloat = Field(default=30, gt=0, le=3600)


def _budget(seconds: float, overall: float) -> verification._Budget:
    """Create a stage timer without changing the historical numerical policy."""
    return verification._Budget(
        _StagePolicy(numerical_seconds=seconds, overall_seconds=overall)
    )


def _overall(wall: float, cpu: float) -> dict:
    """Measure both stages, including setup and failed synchronous persistence."""
    return {
        "elapsed_wall_seconds": monotonic() - wall,
        "elapsed_cpu_seconds": process_time() - cpu,
        "scope": (
            "Includes oracle construction, search, verification and synchronous "
            "checkpoints. Progress receipts exclude the current callback and "
            "therefore contain lower bounds; terminal returned receipts include it."
        ),
    }


def _counts(result: dict) -> None:
    """Distinguish started numerical work from historical search reservations."""
    result["calls"] = result["search_calls_started"]
    result["completed_calls"] = result["search_calls_completed"]


class _SearchOracle:
    """Delegate model semantics while measuring only actually started calls."""

    def __init__(self, oracle, budget: verification._Budget, result: dict):
        self.oracle, self.budget, self.result = oracle, budget, result

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def evaluate(self, vector, deadline):
        """Replace the optimizer's wall deadline with the unspent active budget."""
        del deadline

        def call(active_deadline):
            self.result["search_calls_started"] += 1
            _counts(self.result)
            try:
                output = self.oracle.evaluate(vector, active_deadline)
            except BaseException:
                self.result["search_calls_failed"] += 1
                raise
            self.result["search_calls_completed"] += 1
            _counts(self.result)
            return output

        output = self.budget.numerical(call)
        # A solver that returns after expiry cannot introduce a new accepted best.
        self.budget.check()
        return output


def _enrich(result: dict, oracle, *, training: bool) -> None:
    """Keep successful and failed receipts compatible with the M43 boundary."""
    parameters = result["parameters"]
    result["realized_initial_states"] = kernel._realized_initials(oracle, parameters)
    if training:
        result["shared_parameters"] = (
            {name: parameters[name] for name in oracle.shared}
            if parameters is not None
            else None
        )
        result["training_roots"] = (
            {
                row.trajectory_id: {
                    name: parameters[f"trajectory:{row.trajectory_id}:{name}"]
                    for name in oracle.roots
                }
                for row in oracle.rows
            }
            if parameters is not None
            else None
        )
    else:
        result["roots"] = deepcopy(parameters)


def _solve(
    oracle,
    numerical: kernel.NumericalPolicy,
    execution: SearchExecutionPolicy,
    checkpoint: kernel.Checkpoint | None,
    result: dict,
    search_budget: verification._Budget,
    wall: float,
    cpu: float,
    *,
    training: bool,
) -> dict:
    """Search once, then independently verify the retained endpoint once."""
    result.update(
        execution_policy=execution.model_dump(mode="json"),
        execution_protocol="prefix-search-execution-1",
        fit_success=False,
        parameters=None,
        search={
            "best": None,
            "calls": 0,
            "completed_calls": 0,
            "stop_reason": "not_started",
        },
        search_start=deepcopy(oracle.start),
        search_calls_attempted=0,
        search_calls_started=0,
        search_calls_completed=0,
        search_calls_failed=0,
        verification_calls_attempted=0,
        verification_calls_started=0,
        verification_calls_completed=0,
        verification_calls_failed=0,
        verification_timing=None,
        count_scope=(
            "search.calls and search_calls_attempted count optimizer reservations; "
            "calls and search_calls_started count numerical oracle entries. "
            "Completed calls include returned calls later rejected for expiry."
        ),
    )
    _enrich(result, oracle, training=training)
    _counts(result)
    verification_budget = None
    interrupted = None

    def save_search(partial: dict) -> None:
        result["search"] = deepcopy(partial)
        result["search_calls_attempted"] = partial["calls"]
        _counts(result)
        if search_budget.callback_failures:
            raise verification._CheckpointFailure(
                result.get("search_error", "search checkpoint already failed")
            )
        search_budget.check(numerical=False)
        if checkpoint is not None:
            result["search_timing"] = search_budget.snapshot()
            result["overall_timing"] = _overall(wall, cpu)
            snapshot = deepcopy(result)
            try:
                search_budget.callback(lambda: checkpoint(snapshot))
            except Exception as error:
                result["search_error"] = f"{type(error).__name__}: {error}"
                raise

    def save_verification(snapshot: dict) -> None:
        snapshot["verification_timing"] = snapshot["timing"]
        snapshot["overall_timing"] = _overall(wall, cpu)
        if checkpoint is not None:
            checkpoint(snapshot)

    try:
        search_budget.check()
        if oracle.names:
            result["search"] = recovery_numerics.search(
                _SearchOracle(oracle, search_budget, result),
                oracle.start,
                search_budget.wall + execution.search_overall_seconds,
                save_search,
                calls=numerical.maximum_calls,
                target=-1.0,
                telemetry=True,
                optimizer_scaling="guarded_relative",
                objective_multiplier=1.0,
            )
        else:
            result["search"] = {
                "best": {"parameters": {}, "training_nmse": None},
                "calls": 0,
                "completed_calls": 0,
                "stop_reason": "no_free_coordinates",
            }
        result["search_calls_attempted"] = result["search"]["calls"]
        # Exhausting numerical search time may still leave a valid best to check
        # with the independently declared verification allowance.
        with suppress(TimeoutError):
            search_budget.check()
        search_budget.check(numerical=False)
        result["search_timing"] = search_budget.snapshot()
        if result["search"]["best"] is not None:
            verification_budget = _budget(
                numerical.verification_seconds, execution.verification_overall_seconds
            )
            parameters = deepcopy(result["search"]["best"]["parameters"])
            vector = oracle.vector(parameters)

            def evaluate(method: str, deadline: float):
                return oracle.evaluate(
                    vector, deadline, jacobian=method == "DOP853", method=method
                )

            def qualify(solutions: list) -> None:
                residual, derivative = solutions[0]
                difference = float(
                    np.max(np.abs(residual - solutions[1][0]), initial=0.0)
                )
                verified = difference <= numerical.solver_tolerance
                result.update(
                    fit_success=verified,
                    parameters=parameters if verified else None,
                    normalized_mse=float(np.mean(residual**2)),
                    solver_difference=difference,
                    rank_diagnostic=kernel._rank(derivative, oracle.units),
                )
                _enrich(result, oracle, training=training)
                if not verified:
                    result["verification_error"] = (
                        "independent prefix integrators disagree"
                    )

            verification._run(
                result,
                verification_budget,
                save_verification if checkpoint is not None else None,
                evaluate,
                qualify,
            )
            verification_budget.check()
    except Exception as error:
        verification._invalidate(result, error)
        if verification_budget is None:
            result.setdefault("search_error", f"{type(error).__name__}: {error}")
    except BaseException as error:
        interrupted = error
        verification._invalidate(result, error)
    finally:
        _counts(result)
        if not result["fit_success"]:
            _enrich(result, oracle, training=training)
        if verification_budget is None:
            with suppress(TimeoutError):
                search_budget.check()
            result["search_timing"] = search_budget.snapshot()
        else:
            result["verification_timing"] = verification_budget.snapshot()
        result["overall_timing"] = _overall(wall, cpu)
        if interrupted is not None:
            interrupted.execution_receipt = deepcopy(result)
            interrupted.verification_receipt = deepcopy(result)
    if interrupted is not None:
        raise interrupted
    return result


def fit_training(
    problem: PrefixFitProblem,
    training: PublicSplit,
    numerical: kernel.NumericalPolicy,
    execution: SearchExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
) -> dict:
    """Fit joint training coordinates with separate persistence-aware stages."""
    if not isinstance(execution, SearchExecutionPolicy):
        raise TypeError("typed search execution policy required")
    wall, cpu = monotonic(), process_time()
    budget = _budget(numerical.seconds, execution.search_overall_seconds)
    oracle = kernel.TrainingOracle(problem, training, numerical)
    result = {
        "protocol": "prefix-training-fit-1",
        "problem_sha256": content_sha256(problem),
        "training_sha256": content_sha256(training),
        "training_scales": dict(oracle.scales),
        "initialization_audit": deepcopy(oracle.audit),
        "policy": numerical.model_dump(mode="json"),
        "selection_data": "train_only",
    }
    return _solve(
        oracle,
        numerical,
        execution,
        checkpoint,
        result,
        budget,
        wall,
        cpu,
        training=True,
    )


def infer_prefix(
    problem: PrefixFitProblem,
    fit: dict,
    prefix: kernel.PrefixView,
    numerical: kernel.NumericalPolicy,
    execution: SearchExecutionPolicy,
    checkpoint: kernel.Checkpoint | None = None,
    *,
    root_guesses: Mapping[str, float] | None = None,
) -> dict:
    """Infer roots only, preserving the frozen fit and recording every start."""
    if not isinstance(execution, SearchExecutionPolicy):
        raise TypeError("typed search execution policy required")
    wall, cpu = monotonic(), process_time()
    budget = _budget(numerical.seconds, execution.search_overall_seconds)
    oracle = kernel.PrefixOracle(
        problem, fit, prefix, numerical, root_guesses=root_guesses
    )
    result = {
        "protocol": "prefix-inference-1",
        "problem_sha256": content_sha256(problem),
        "fit_sha256": content_sha256(fit),
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
        "policy": numerical.model_dump(mode="json"),
    }
    return _solve(
        oracle,
        numerical,
        execution,
        checkpoint,
        result,
        budget,
        wall,
        cpu,
        training=False,
    )
