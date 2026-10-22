"""Opt-in unchanged-model continuation with affine execution and explicit evidence.

The campaign owns immutable starts and operation journals. No model, domain,
initializer or data split is altered here. Search and diagnostic budgets are
separate, and a failed new point never displaces a verified incumbent.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from autoformalism.fitting import (
    poor_fit,
    prefix_affine,
    prefix_execution,
    recovery_numerics,
)
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import public_fitting as public


def make_oracle(problem, training, numerical, arm):
    """Certify the original model; reduce only supported exact gain products."""
    original = prefix_affine.TrainingOracle(problem, training, numerical)
    if arm == "affine_original":
        return original, original
    if arm == "affine_reduced":
        return original, gains.ReducedTrainingOracle(original)
    raise ValueError("unknown continuation arm")


def to_effective(original, oracle, parameters):
    """Map a complete original shared/root vector into the arm's coordinates."""
    vector = original.vector(parameters)
    if oracle is not original:
        vector = oracle.reduction.reduce(vector)
    return oracle.parameters(vector)


def to_original(original, oracle, parameters):
    """Return an admissible original-domain representative, never a changed ODE."""
    vector = oracle.vector(parameters)
    if oracle is not original:
        vector = oracle.reduction.lift(vector)
    return original.parameters(vector)


def predetermined_starts(oracle, retained):
    """Three fixed original-coordinate starts, independent of error or held-out data."""
    saved = oracle.parameters(oracle.vector(retained))
    declared = deepcopy(oracle.start)
    vector = oracle.vector(declared)
    rng = np.random.default_rng(1729)
    trial = vector + rng.uniform(-0.5, 0.5, len(vector)) * oracle.units
    # Respect the existing domains, without adding a biological sign constraint.
    trial = np.clip(trial, oracle.lower, oracle.upper)
    perturbed = oracle.parameters(oracle.vector(oracle.parameters(trial)))
    return [
        {"name": "inherited", "parameters": saved},
        {"name": "declared", "parameters": declared},
        {"name": "perturbed_declared", "parameters": perturbed},
    ]


class _TimedOracle:
    """Track actual numerical calls while excluding measured checkpoint writes."""

    def __init__(self, oracle, budget):
        self.oracle, self.budget = oracle, budget
        self.started = self.completed = self.failed = 0

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def evaluate(self, vector, deadline):
        del deadline

        def run(active_deadline):
            self.started += 1
            try:
                result = self.oracle.evaluate(vector, active_deadline)
            except BaseException:
                self.failed += 1
                raise
            self.completed += 1
            return result

        value = self.budget.numerical(run)
        self.budget.check()
        return value


def _invalidate(result, error, stage):
    """Keep useful candidates but revoke qualification after incomplete work."""
    result.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    if stage == "search":
        result["stop_reason"] = (
            "budget_limited" if isinstance(error, TimeoutError) else "numerical_failure"
        )
        result["budget_exhausted"] = isinstance(error, TimeoutError)
    elif stage == "verification":
        result["original_equations_verified"] = False
    elif stage == "qualification":
        result["passed"] = False
    else:
        result["assessment"] = None


def _finish(result, budget, checkpoint, stage, interrupted=None):
    """Measure the final write once, then check both inclusive and active limits.

    A persisted progress snapshot necessarily predates its own write. The returned
    receipt includes that write and is authoritative for the owning stage journal.
    Never retry a failed callback or start another after inclusive expiry.
    """
    result.setdefault("status", "complete")
    try:
        budget.check()
    except TimeoutError as error:
        _invalidate(result, error, stage)
    if interrupted is None and not budget.callback_failures:
        try:
            budget.check(numerical=False)
            result["timing"] = budget.snapshot()
            snapshot = deepcopy(result)
            budget.callback(lambda: checkpoint(snapshot))
            budget.check()
        except Exception as error:
            _invalidate(result, error, stage)
        except BaseException as error:
            interrupted = error
            _invalidate(result, error, stage)
    result["timing"] = budget.snapshot()
    if interrupted is not None:
        interrupted.execution_receipt = deepcopy(result)
        raise interrupted
    return result


def search(oracle, start, seconds, maximum_calls, checkpoint):
    """One bounded TRF attempt with evaluation and accepted-step receipts."""
    budget = prefix_execution._budget(seconds, seconds + 120)
    timed = _TimedOracle(oracle, budget)
    last = {"best": None, "calls": 0, "stop_reason": "not_started"}

    def save(partial):
        nonlocal last
        last = deepcopy(partial)
        if budget.callback_failures:
            raise RuntimeError("continuation checkpoint already failed")
        budget.check(numerical=False)
        budget.callback(
            lambda: checkpoint(
                partial
                | {
                    "timing": budget.snapshot(),
                    "calls_started": timed.started,
                    "search_calls_attempted": partial.get("calls", timed.started),
                    "search_calls_started": timed.started,
                    "search_calls_completed": timed.completed,
                    "search_calls_failed": timed.failed,
                }
            )
        )

    interrupted = None
    try:
        result = recovery_numerics.search(
            timed,
            start,
            budget.wall + seconds + 120,
            save,
            calls=maximum_calls,
            target=1e-12,
            telemetry=True,
            record_steps=True,
            optimizer_scaling="guarded_relative",
            objective_multiplier=1.0,
            poor_fit=poor_fit.Policy(),
        )
    except Exception as error:
        result = last
        _invalidate(result, error, "search")
    except BaseException as error:
        interrupted, result = error, last
        _invalidate(result, error, "search")
        result["stop_reason"] = "interrupted"
        result["budget_exhausted"] = True
    result.update(
        timing=budget.snapshot(),
        calls_started=timed.started,
        calls_completed=timed.completed,
        calls_failed=timed.failed,
        optimizer_start=deepcopy(start),
        global_minimum_certified=False,
        search_calls_attempted=result["calls"],
        search_calls_started=timed.started,
        search_calls_completed=timed.completed,
        search_calls_failed=timed.failed,
    )
    return _finish(result, budget, checkpoint, "search", interrupted)


def verify_endpoint(original, oracle, parameters, seconds, checkpoint):
    """Compare the exact affine rollout with original-model Radau, output-only."""
    budget = prefix_execution._budget(seconds, seconds + 60)
    vector = oracle.vector(parameters)
    lifted = to_original(original, oracle, parameters)
    result = {
        "parameters": lifted,
        "effective_parameters": deepcopy(parameters),
        "original_equations_verified": False,
        "calls_started": 0,
        "calls_completed": 0,
        "methods": ["certified_affine", "original_Radau"],
    }
    residuals = []
    interrupted = None
    try:
        for source, point, method in (
            (oracle, vector, None),
            (original, original.vector(lifted), "Radau"),
        ):
            budget.callback(lambda: checkpoint(result | {"timing": budget.snapshot()}))

            def evaluate(deadline, s=source, p=point, m=method):
                result["calls_started"] += 1
                value = s.evaluate(p, deadline, jacobian=False, method=m)
                result["calls_completed"] += 1
                return value

            r, _ = budget.numerical(evaluate)
            budget.check()
            if not np.isfinite(r).all():
                raise ValueError("nonfinite verified residual")
            residuals.append(r)
        loss, alternate = [float(np.mean(r * r)) for r in residuals]
        gap = float(np.max(np.abs(residuals[0] - residuals[1]), initial=0))
        result.update(
            reduced_vector=vector.tolist(),
            training_nmse=loss,
            alternate_training_nmse=alternate,
            solver_loss_discrepancy=abs(loss - alternate),
            solver_prediction_discrepancy=gap,
            original_equations_verified=(
                gap <= 1e-7
                and abs(loss - alternate) <= max(1e-12, 1e-8 * max(loss, alternate))
            ),
        )
        if result["original_equations_verified"]:
            columns = residuals[0].reshape(-1, len(original.system.channels))
            result["per_target_normalized_mse"] = dict(
                zip(
                    original.system.channels,
                    np.mean(columns**2, axis=0).tolist(),
                    strict=True,
                )
            )
    except Exception as error:
        _invalidate(result, error, "verification")
    except BaseException as error:
        interrupted = error
        _invalidate(result, error, "verification")
    result["timing"] = budget.snapshot()
    result["verification_calls_attempted"] = result["calls_started"]
    result["verification_calls_started"] = result["calls_started"]
    result["verification_calls_completed"] = result["calls_completed"]
    result["verification_calls_failed"] = (
        result["calls_started"] - result["calls_completed"]
    )
    return _finish(result, budget, checkpoint, "verification", interrupted)


def qualification(original, oracle, retained, seconds, checkpoint):
    """Check the lift at the inherited point and audit actual optimizer derivatives."""
    start = to_effective(original, oracle, retained)
    vector = oracle.vector(start)
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {
        "affine_audit": deepcopy(original.affine_audit),
        "original_coordinates": len(original.names),
        "optimizer_coordinates": len(oracle.names),
        "gain_reduction": deepcopy(oracle.reduction.receipt)
        if oracle is not original
        else None,
        "passed": False,
    }
    interrupted = None
    try:
        before, _ = budget.numerical(
            lambda deadline: original.evaluate(
                original.vector(retained), deadline, jacobian=False
            )
        )
        after, _ = budget.numerical(
            lambda deadline: oracle.evaluate(vector, deadline, jacobian=False)
        )
        gap = float(np.max(np.abs(before - after), initial=0))
        result["lift_prediction_discrepancy"] = gap
        check = budget.numerical(
            lambda deadline: adequacy.derivative_check(
                oracle.evaluate,
                vector,
                oracle.lower,
                oracle.upper,
                deadline,
                coordinate_units=oracle.units,
                residual_only=lambda x, d: oracle.evaluate(x, d, jacobian=False)[0],
            )
        )
        result["derivative_check"] = public._jsonable(check)
        budget.check()
        result["passed"] = gap <= 1e-7 and check.status == "passed"
    except Exception as error:
        _invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        _invalidate(result, error, "qualification")
    return _finish(result, budget, checkpoint, "qualification", interrupted)


def diagnose(oracle, parameters, verification, search_result, seconds, checkpoint):
    """Assess the exact retained endpoint; a weak rank is distinct from convergence."""
    budget = prefix_execution._budget(seconds, seconds + 60)
    result, interrupted = {"assessment": None}, None
    try:
        vector = oracle.vector(parameters)
        derivative = budget.numerical(
            lambda deadline: adequacy.derivative_check(
                oracle.evaluate,
                vector,
                oracle.lower,
                oracle.upper,
                deadline,
                coordinate_units=oracle.units,
                residual_only=lambda x, d: oracle.evaluate(x, d, jacobian=False)[0],
            )
        )
        result["derivative_check"] = public._jsonable(derivative)
        residual, jacobian = budget.numerical(
            lambda deadline: oracle.evaluate(vector, deadline)
        )
        stop = search_result.get("stop_reason")
        history = [
            {
                key: step[key]
                for key in ("training_nmse", "coordinate_step_norm", "evaluation_count")
            }
            for step in search_result.get("accepted_steps", [])
        ]
        assessment = adequacy.assess(
            vector,
            residual,
            jacobian,
            oracle.lower,
            oracle.upper,
            verification={
                key: verification[key]
                for key in adequacy.OriginalVerification.model_fields
            },
            derivative=derivative,
            stop_reason=stop,
            budget_exhausted=bool(search_result.get("budget_exhausted"))
            or stop in {"budget_limited", "numerical_failure", "interrupted"},
            history=history,
            coordinate_units=oracle.units,
        )
        budget.check()
        result["assessment"] = public._jsonable(assessment)
    except Exception as error:
        _invalidate(result, error, "diagnosis")
    except BaseException as error:
        interrupted = error
        _invalidate(result, error, "diagnosis")
    return _finish(result, budget, checkpoint, "diagnosis", interrupted)
