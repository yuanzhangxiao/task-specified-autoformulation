"""Residual-only controls on an already certified nonlinear coefficient chart.

These controls do not bypass derivative gates for gradient optimizers. They use
no parameter derivatives and make no stationarity, parameter-confidence or global
optimality claim. The caller owns separate, nonrenewable operation reservations
and independently verifies any proposed improvement in the original equations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_nonlinear_reduction as reduction
from autoformalism.fitting.compact_search_checkpoint import (
    SearchCheckpointPolicy,
    SearchProgress,
)
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "portfolio-certified-residual-search-1"
REPEAT_TOLERANCE = 1e-12
MAX_FAILURE_EXAMPLES = 8
Checkpoint = Callable[[dict], None]


class _CallLimit(TimeoutError):
    """The declared number of objective requests has been consumed."""


def _claims() -> dict:
    return {
        "protocol": PROTOCOL,
        "derivatives_used": False,
        "gradient_qualified": False,
        "parameter_information_qualified": False,
        "convergence_certified": False,
        "global_minimum_certified": False,
        "independent_original_verification_required": True,
    }


def _certificate(original, oracle, parameters: Mapping[str, float]) -> tuple:
    """Require the existing executable proof and validate this exact lifted point."""
    if not isinstance(oracle, portfolio.PortfolioOracle) or not isinstance(
        oracle.oracle, reduction.ReducedOracle
    ):
        raise ValueError("certified nonlinear reduced oracle required")
    if original is not oracle.original:
        raise ValueError("original oracle differs from certified chart")
    receipt = oracle.reduction.receipt
    proof = receipt.get("proof", {})
    if (
        oracle.portfolio_audit.get("protocol") != reduction.PROTOCOL
        or oracle.portfolio_audit.get("reduction") != receipt
        or proof.get("protocol") != reduction.PROTOCOL
        or not proof.get("zero_product_branch_preserved")
        or not receipt.get("original_model_unchanged")
        or receipt.get("sha256")
        != content_sha256({k: v for k, v in receipt.items() if k != "sha256"})
    ):
        raise ValueError("nonlinear chart proof or audit differs")
    point = oracle.vector(parameters)
    lifted = oracle.reduction.lift(point)
    original.vector(original.parameters(lifted))
    if not np.allclose(oracle.reduction.reduce(lifted), point, rtol=1e-13, atol=1e-14):
        raise ValueError("nonlinear chart lift does not round-trip")
    return (
        point,
        lifted,
        {
            "chart_sha256": receipt["sha256"],
            "portfolio_audit_sha256": content_sha256(oracle.portfolio_audit),
            "parameters_sha256": content_sha256(dict(parameters)),
            "original_parameters_sha256": content_sha256(original.parameters(lifted)),
            "original_domain_checked": True,
            "zero_product_branch_preserved": True,
            "source_equations_unchanged": True,
        },
    )


def _residual(value, expected: int | None = None) -> tuple[np.ndarray, float]:
    residual = np.asarray(value, float)
    if (
        residual.ndim != 1
        or not residual.size
        or (expected is not None and residual.size != expected)
    ):
        raise ValueError("residual shape changed or is invalid")
    with np.errstate(over="ignore", invalid="ignore"):
        loss = float(np.mean(residual**2))
    if not np.isfinite(residual).all() or not np.isfinite(loss):
        raise NumericalInvalidTrial("nonfinite residual-only objective")
    return residual, loss


def _account(result: dict, oracle, before: dict) -> None:
    result["portfolio_execution"] = oracle.execution_receipt()
    result["stage_backend_calls"] = {
        key: value - before[key] for key, value in oracle.counts.items()
    }


def qualify_residual(
    original, oracle, parameters, seconds: float, checkpoint: Checkpoint
) -> dict:
    """Repeat fresh BDF residuals, then replay the original lift with Radau.

    This qualifies only residual-based search at the supplied seed. Neither cached
    repeatability nor a finite prediction discrepancy certifies derivatives.
    """
    if isinstance(seconds, bool) or not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("residual qualification requires <=600 positive seconds")
    budget = execution._budget(seconds, seconds + 60)
    result = _claims() | {
        "passed": False,
        "no_search_performed": True,
        "original_equations_verified": False,
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
        "replays": [],
        "repeatability_tolerance": REPEAT_TOLERANCE,
        "original_prediction_tolerance": portfolio.PAIRING_TOLERANCE,
        "count_scope": "Three fresh state-only solves; repeated BDF bypasses cache.",
    }
    before = deepcopy(oracle.counts)
    interrupted = None

    def save():
        _account(result, oracle, before)
        budget.callback(
            lambda: checkpoint(deepcopy(result) | {"timing": budget.snapshot()})
        )

    try:
        point, lifted, certificate = _certificate(original, oracle, parameters)
        result["certificate"] = certificate
        arrays = []
        for label, method in (
            ("canonical", "BDF"),
            ("repeat", "BDF"),
            ("original", "Radau"),
        ):
            result["verification_calls_attempted"] += 1
            save()

            def evaluate(deadline, label=label, method=method):
                result["verification_calls_started"] += 1
                try:
                    value = (
                        original.evaluate(
                            lifted, deadline, jacobian=False, method=method
                        )
                        if label == "original"
                        else oracle._call(
                            point, deadline, jacobian=False, method=method
                        )
                    )
                except BaseException:
                    result["verification_calls_failed"] += 1
                    raise
                result["verification_calls_completed"] += 1
                return value[0]

            raw = budget.numerical(evaluate)
            budget.check()
            residual, loss = _residual(raw, len(arrays[0]) if arrays else None)
            arrays.append(residual)
            result["replays"].append(
                {
                    "label": label,
                    "method": method,
                    "training_nmse": loss,
                    "residual_sha256": content_sha256(residual.tolist()),
                }
            )
            save()
        repeat_gap = float(np.max(np.abs(arrays[0] - arrays[1])))
        original_gap = float(np.max(np.abs(arrays[0] - arrays[2])))
        result.update(
            training_nmse=result["replays"][0]["training_nmse"],
            repeat_prediction_discrepancy=repeat_gap,
            original_Radau_prediction_discrepancy=original_gap,
            original_equations_verified=original_gap <= portfolio.PAIRING_TOLERANCE,
            passed=repeat_gap <= REPEAT_TOLERANCE
            and original_gap <= portfolio.PAIRING_TOLERANCE,
        )
    except Exception as error:
        continuation._invalidate(
            result, error, "qualification", preserve_diagnostic=True
        )
    except BaseException as error:
        interrupted = error
        continuation._invalidate(
            result, error, "qualification", preserve_diagnostic=True
        )
    _account(result, oracle, before)
    return continuation._finish(
        result, budget, checkpoint, "qualification", interrupted
    )


def search(
    oracle,
    parameters,
    seconds: float,
    maximum_calls: int,
    checkpoint: Checkpoint,
    *,
    fixed: Mapping[str, float] | None = None,
    checkpoint_policy: SearchCheckpointPolicy | None = None,
) -> dict:
    """One bounded Powell control, with a balanced frozen-seed coordinate sweep.

    The call limit counts every objective request, including invalid chart trials and
    oracle cache hits. Actual backend entries are reported separately. No artificial
    physical bounds, finite replacements for infinities or renewed allowances occur.
    Explicitly held coordinates are overlaid on the seed and excluded from every
    search direction. Their original domains and the complete lifted point remain
    subject to the existing certificate; a conditional search is not a profile
    minimum or a parameter-confidence statement.
    """
    if (
        isinstance(seconds, bool)
        or not np.isfinite(seconds)
        or not 0 < seconds <= 1800
        or type(maximum_calls) is not int
        or not 1 <= maximum_calls <= 2000
    ):
        raise ValueError("residual search requires <=1800 seconds and <=2000 calls")
    if checkpoint_policy is not None and not isinstance(
        checkpoint_policy, SearchCheckpointPolicy
    ):
        raise TypeError("typed SearchCheckpointPolicy required")
    if fixed is not None and not isinstance(fixed, Mapping):
        raise TypeError("fixed coordinates must be a mapping")
    budget = execution._budget(seconds, seconds + 120)
    result = _claims() | {
        "optimizer": "Powell over fixed seed-scaled certified coordinates",
        "best": None,
        "last_accepted": None,
        "accepted_history": [],
        "search_calls_attempted": 0,
        "search_calls_started": 0,
        "search_calls_completed": 0,
        "search_calls_failed": 0,
        "invalid_trials": 0,
        "invalid_trial_histogram": {},
        "invalid_trial_examples": [],
        "initial_sweep_completed": False,
        "coordinates_varied": [],
        "initial_sweep_coordinates_completed": [],
        "budget_exhausted": False,
        "stop_reason": "not_started",
        "count_scope": (
            "attempted counts objective requests; started/completed/failed count "
            "oracle "
            "entries including cache hits. Returned invalid residuals are completed "
            "calls but invalid trials. Backend deltas exclude cache hits."
        ),
    }
    before = deepcopy(oracle.counts)
    interrupted = None
    completed = {}
    progress = (
        SearchProgress(checkpoint_policy, checkpoint)
        if checkpoint_policy is not None
        else None
    )
    if progress is not None:
        progress.attach(result)

    def save(*, force=False, event="progress"):
        _account(result, oracle, before)
        if progress is not None:
            progress.save(result, budget, force=force, event=event)
            return
        budget.callback(
            lambda: checkpoint(deepcopy(result) | {"timing": budget.snapshot()})
        )

    def invalid(error, point):
        result["invalid_trials"] += 1
        text = f"{type(error).__name__}: {error}"[:300]
        histogram = result["invalid_trial_histogram"]
        key = (
            text
            if text in histogram or len(histogram) < MAX_FAILURE_EXAMPLES
            else "other"
        )
        histogram[key] = histogram.get(key, 0) + 1
        if len(result["invalid_trial_examples"]) < MAX_FAILURE_EXAMPLES:
            result["invalid_trial_examples"].append(
                {
                    "error": text,
                    "parameters": oracle.parameters(point)
                    if point is not None
                    else None,
                    "diagnostic": continuation._bounded_diagnostic(
                        getattr(error, "diagnostic", {})
                    ),
                }
            )
        save()
        return float("inf")

    try:
        held = dict(fixed or {})
        if set(held) - set(oracle.names):
            raise ValueError("fixed coordinates must exist in the certified chart")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            for value in held.values()
        ):
            raise ValueError("fixed coordinates require finite numeric values")
        held = {name: float(value) for name, value in held.items()}
        start = dict(parameters) | held
        initial, _, certificate = _certificate(oracle.original, oracle, start)
        free = np.asarray(
            [
                i
                for i, name in enumerate(oracle.names)
                if name not in held and oracle.lower[i] < oracle.upper[i]
            ],
            int,
        )
        units = np.maximum(np.abs(initial), np.asarray(oracle.units, float))[free]
        if not np.isfinite(units).all() or np.any(units <= 0):
            raise ValueError("finite positive seed scales required")
        lower = (oracle.lower[free] - initial[free]) / units
        upper = (oracle.upper[free] - initial[free]) / units
        bounds = list(zip(lower, upper, strict=True))
        result.update(
            certificate=certificate,
            optimizer_start=start,
            fixed_coordinates=held,
            free_coordinates=[oracle.names[i] for i in free],
            coordinate_chart={
                "rule": "theta=seed+q*max(abs(seed),declared_unit)",
                "units": units.tolist(),
                "artificial_bounds_added": False,
            },
        )
        best_q = np.zeros(len(free))
        residual_size = None

        def point_for(q):
            q = np.asarray(q, float)
            if (
                q.shape != free.shape
                or not np.isfinite(q).all()
                or np.any(q < lower)
                or np.any(q > upper)
            ):
                raise NumericalInvalidTrial("trial outside declared scaled domain")
            point = initial.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                point[free] = initial[free] + q * units
            if not np.isfinite(point).all():
                raise NumericalInvalidTrial("nonrepresentable affine chart trial")
            # Exact original bounds and seed, correcting only affine roundoff.
            point[free] = np.clip(point[free], oracle.lower[free], oracle.upper[free])
            point[free[q == 0]] = initial[free[q == 0]]
            oracle.vector(oracle.parameters(point))
            lifted = oracle.reduction.lift(point)
            oracle.original.vector(oracle.original.parameters(lifted))
            return point

        def objective(q):
            nonlocal best_q, residual_size
            budget.check()
            if result["search_calls_attempted"] >= maximum_calls:
                raise _CallLimit("residual objective call allowance exhausted")
            result["search_calls_attempted"] += 1
            point = None
            try:
                point = point_for(q)
            except (NumericalInvalidTrial, ArithmeticError) as error:
                return invalid(error, None)
            varied = {oracle.names[i] for i in free if point[i] != initial[i]}
            result["coordinates_varied"] = sorted(
                set(result["coordinates_varied"]) | varied
            )
            save()

            def evaluate(deadline):
                result["search_calls_started"] += 1
                try:
                    value = oracle.evaluate(point, deadline, jacobian=False)
                except BaseException:
                    result["search_calls_failed"] += 1
                    raise
                result["search_calls_completed"] += 1
                return value[0]

            try:
                raw = budget.numerical(evaluate)
                budget.check()  # Late returns cannot become a new best.
                residual, loss = _residual(raw, residual_size)
            except (NumericalInvalidTrial, ArithmeticError) as error:
                return invalid(error, point)
            residual_size = len(residual)
            endpoint = {
                "parameters": oracle.parameters(point),
                "training_nmse": loss,
                "reduced_vector": point.tolist(),
            }
            completed[point.tobytes()] = endpoint
            improved = (
                result["best"] is None or loss < result["best"]["training_nmse"]
            )
            if improved:
                result["best"] = deepcopy(endpoint)
                best_q = np.asarray(q, float).copy()
            save(force=improved, event="incumbent" if improved else "progress")
            return loss

        def accepted(q, phase="powell_iteration"):
            point = point_for(q)
            endpoint = completed.get(point.tobytes())
            if endpoint is None:
                raise RuntimeError(
                    "accepted Powell endpoint lacks a finite evaluated receipt"
                )
            result["last_accepted"] = deepcopy(endpoint)
            result["accepted_history"].append(
                deepcopy(endpoint)
                | {
                    "phase": phase,
                    "scope": "derivative-free accepted point; no stationarity claim",
                }
            )
            save(force=True, event="accepted_iteration")

        zero = np.zeros(len(free))
        if not np.isfinite(objective(zero)):
            raise ValueError("residual-search seed is unavailable")
        accepted(zero, "seed")
        for j, (lo, hi) in enumerate(bounds):
            for direction in (-1.0, 1.0):
                q = zero.copy()
                q[j] = np.clip(direction, lo, hi)
                if q[j] != 0:
                    objective(q)
            result["initial_sweep_coordinates_completed"].append(oracle.names[free[j]])
            save()
        result["initial_sweep_completed"] = True
        save(force=True, event="initial_sweep_complete")
        if len(free):
            remaining = maximum_calls - result["search_calls_attempted"]
            if remaining <= 0:
                raise _CallLimit("residual objective call allowance exhausted")
            fit = minimize(
                objective,
                best_q,
                method="Powell",
                bounds=bounds,
                callback=accepted,
                options={"maxfev": remaining, "xtol": 1e-8, "ftol": 1e-10},
            )
            exhausted = result["search_calls_attempted"] >= maximum_calls
            result.update(
                stop_reason="call_limit" if exhausted else "optimizer_stopped",
                optimizer_success=bool(fit.success),
                optimizer_message=str(fit.message),
                optimizer_iterations=int(fit.nit),
                budget_exhausted=exhausted,
            )
        else:
            result.update(stop_reason="no_free_coordinates", optimizer_success=True)
    except Exception as error:
        continuation._invalidate(result, error, "search", preserve_diagnostic=True)
        if isinstance(error, _CallLimit):
            result["stop_reason"] = "call_limit"
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "search", preserve_diagnostic=True)
        result["stop_reason"] = "interrupted"
    _account(result, oracle, before)
    return continuation._finish(
        result,
        budget,
        progress.final if progress is not None else checkpoint,
        "search",
        interrupted,
    )
