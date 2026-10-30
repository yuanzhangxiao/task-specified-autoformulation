"""Opt-in mixed affine/nonlinear fitting with certified coordinate reductions.

The caller owns frozen inputs, stage journals and nonrenewable allowances. This
adapter preserves the original ODE, initial graph, domains and supplied forcing.
Residual-only solves define the objective; sensitivity solves must agree with
that baseline before their derivatives can be used or independently checked.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import deepcopy
from time import monotonic

import numpy as np

from autoformalism.fitting import (
    poor_fit,
    prefix_affine,
    prefix_execution,
    recovery_numerics,
)
from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.prefix_structural_reduction import StructuralTrainingOracle
from autoformalism.fitting.sensitivity_probe import SymbolicRolloutFailure
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "scoped-prefix-portfolio-adapter-1"
PAIRING_TOLERANCE = 1e-7
CACHE_POINTS = 8
Checkpoint = Callable[[dict], None]


def policy_receipt() -> dict:
    """Expose numerical routing, objective pairing and exact reduction policies."""
    return {
        "protocol": PROTOCOL,
        "routing": "certified_affine_else_original_ODE",
        "affine_precision": recovery.policy_receipt()["precision_policies"]["guarded"],
        "reductions": recovery.policy_receipt(structural_reduction=True),
        "canonical_residual": "output_only_original_model_solve",
        "pairing_max_normalized_residual_discrepancy": PAIRING_TOLERANCE,
        "pairing_is_derivative_accuracy_certificate": False,
        "cache_points": CACHE_POINTS,
        "cache_identity": "model_data_chart_vector_method_solver_settings",
        "lazy_jacobian": True,
        "optimizer_scaling": "guarded_relative",
        "derivative_policy": _policy().model_dump(mode="json"),
        "original_parameter_identifiability_certified": False,
    }


def _policy() -> adequacy.Policy:
    return adequacy.Policy(
        precision_aware=True,
        local_adaptive=True,
        local_bounds_aware=True,
        diagnostic_precision=True,
    )


def _check(deadline: float | None) -> None:
    if deadline is None or not np.isfinite(deadline):
        raise ValueError("finite portfolio deadline required")
    if monotonic() >= deadline:
        raise TimeoutError("portfolio numerical allowance exhausted")


class PortfolioOracle:
    """Pair independent sensitivities with a bounded canonical-residual cache.

    The observed pairing discrepancy is an empirical consistency check, not an
    error bound on derivatives. Qualification still tests every active column
    against finite differences of the canonical residual. Cache hits and all
    underlying rollout calls are reported separately.
    """

    def __init__(self, oracle, audit: dict):
        self.oracle = oracle
        self.portfolio_audit = deepcopy(audit)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._identity = public.content_sha256(audit)
        self.counts = {
            "residual_solves_started": 0,
            "residual_solves_completed": 0,
            "jacobian_solves_started": 0,
            "jacobian_solves_completed": 0,
            "cache_hits": 0,
            "pairing_checks": 0,
            "pairing_failures": 0,
        }
        self.maximum_pairing_discrepancy = 0.0

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def execution_receipt(self) -> dict:
        return self.counts | {
            "maximum_pairing_discrepancy": self.maximum_pairing_discrepancy,
            "cached_points": len(self._cache),
            "scope": "cumulative_since_oracle_construction",
        }

    def _call(self, vector, deadline, *, jacobian, method):
        _check(deadline)
        kind = "jacobian" if jacobian else "residual"
        self.counts[f"{kind}_solves_started"] += 1
        try:
            r, j = self.oracle.evaluate(
                vector, deadline, jacobian=jacobian, method=method
            )
        except SymbolicRolloutFailure as error:
            failure = recovery_numerics.NumericalInvalidTrial(str(error))
            failure.diagnostic = error.diagnostic
            raise failure from error
        except ValueError as error:
            if str(error) == "symbolic output or sensitivity is nonfinite":
                raise recovery_numerics.NumericalInvalidTrial(str(error)) from error
            raise
        _check(deadline)
        r = np.asarray(r, dtype=float)
        if r.ndim != 1 or not r.size:
            raise ValueError("invalid portfolio residual shape")
        if not np.isfinite(r).all():
            raise recovery_numerics.NumericalInvalidTrial("nonfinite residual")
        if jacobian:
            j = np.asarray(j, dtype=float)
            if j.shape != (r.size, len(self.names)):
                raise ValueError("invalid portfolio Jacobian shape")
            if not np.isfinite(j).all():
                raise recovery_numerics.NumericalInvalidTrial("nonfinite Jacobian")
        self.counts[f"{kind}_solves_completed"] += 1
        return r, j

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        """Preserve a single objective even when augmented integration differs."""
        _check(deadline)
        vector = self.vector(self.parameters(vector))
        key = (
            self._identity,
            vector.tobytes(),
            method,
            public.content_sha256(self.original.settings),
        )
        if key in self._cache:
            self.counts["cache_hits"] += 1
            baseline = self._cache.pop(key)
        else:
            baseline, _ = self._call(vector, deadline, jacobian=False, method=method)
        self._cache[key] = baseline.copy()
        while len(self._cache) > CACHE_POINTS:
            self._cache.popitem(last=False)
        if not jacobian:
            return baseline.copy(), None
        r, j = self._call(vector, deadline, jacobian=True, method=method)
        discrepancy = float(np.max(np.abs(r - baseline), initial=0))
        self.counts["pairing_checks"] += 1
        self.maximum_pairing_discrepancy = max(
            discrepancy, self.maximum_pairing_discrepancy
        )
        if discrepancy > PAIRING_TOLERANCE:
            self.counts["pairing_failures"] += 1
            raise recovery_numerics.NumericalInvalidTrial(
                "sensitivity/output-only residual discrepancy exceeds pairing policy"
            )
        return baseline.copy(), j


def make_oracle(
    problem: PrefixFitProblem,
    training: PublicSplit,
    numerical: kernel.NumericalPolicy,
):
    """Route only by an affine certificate and retain exact reduction receipts."""
    original = kernel.TrainingOracle(problem, training, numerical)
    try:
        prefix_affine._Propagation(
            original.system, tuple(range(len(original.system.names)))
        )
    except ValueError as error:
        if str(error) != "requires autonomous jointly affine state/input dynamics":
            raise
        route = "original_ODE"
    else:
        route = "certified_affine"
        original = prefix_affine.TrainingOracle(
            problem,
            training,
            numerical,
            precision_policy=recovery.precision_policy("guarded"),
        )
    active = recovery.ActiveTrainingOracle(original, original.start, "guarded")
    reduced = StructuralTrainingOracle(active, original.start)
    audit = {
        "protocol": PROTOCOL,
        "route": route,
        "problem_sha256": public.content_sha256(problem),
        "training_sha256": public.content_sha256(training),
        "numerical_policy": numerical.model_dump(mode="json"),
        "forcing_symbols": list(original.system.inputs),
        "original_coordinates": len(original.names),
        "optimizer_coordinates": len(reduced.names),
        "reduction": reduced.reduction.receipt,
        "affine_certificate": getattr(original, "affine_audit", None),
        "original_equations_unchanged": True,
        "original_parameter_identifiability_certified": False,
    }
    original.portfolio_audit = deepcopy(audit)
    return original, PortfolioOracle(reduced, audit)


class _TimedOracle:
    """Keep every cache miss and sensitivity solve within one active allowance."""

    def __init__(self, oracle, budget):
        self.oracle, self.budget = oracle, budget

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def evaluate(self, vector, deadline, *, jacobian=True):
        del deadline
        result = self.budget.numerical(
            lambda d: self.oracle.evaluate(vector, d, jacobian=jacobian)
        )
        self.budget.check()
        return result


def search(
    oracle: PortfolioOracle,
    start: Mapping[str, float],
    seconds: float,
    maximum_calls: int,
    checkpoint: Checkpoint,
) -> dict:
    """Run one lazy TRF attempt; preserve useful points without claiming success."""
    budget = prefix_execution._budget(seconds, seconds + 120)
    timed = _TimedOracle(oracle, budget)
    last = {"best": None, "calls": 0, "stop_reason": "not_started"}
    before = deepcopy(oracle.counts)

    def save(partial):
        nonlocal last
        last = deepcopy(partial)
        budget.check(numerical=False)
        budget.callback(
            lambda: checkpoint(
                deepcopy(partial)
                | {
                    "timing": budget.snapshot(),
                    "portfolio_execution": oracle.execution_receipt(),
                }
            )
        )

    interrupted = None
    try:
        result = recovery_numerics.search(
            timed,
            dict(start),
            budget.wall + seconds + 120,
            save,
            calls=maximum_calls,
            target=1e-12,
            telemetry=True,
            record_steps=True,
            record_full_steps=True,
            optimizer_scaling="guarded_relative",
            objective_multiplier=1.0,
            poor_fit=poor_fit.Policy(),
            reject_invalid_trials=True,
            lazy_jacobian=True,
        )
    except Exception as error:
        result = last
        continuation._invalidate(result, error, "search")
    except BaseException as error:
        interrupted, result = error, last
        continuation._invalidate(result, error, "search")
        result.update(stop_reason="interrupted", budget_exhausted=True)
    result.update(
        optimizer_start=dict(start),
        portfolio_execution=oracle.execution_receipt(),
        stage_backend_calls={k: v - before[k] for k, v in oracle.counts.items()},
        global_minimum_certified=False,
        search_calls_started=result.get("oracle_calls", 0),
        search_calls_completed=result.get("completed_residual_calls", 0)
        + result.get("completed_jacobian_calls", 0),
        search_calls_failed=result.get("failed_calls", 0)
        + result.get("failed_jacobian_calls", 0),
    )
    return continuation._finish(result, budget, checkpoint, "search", interrupted)


def qualification(
    original, oracle, parameters, seconds, checkpoint: Checkpoint
) -> dict:
    """Test the actual reduced objective locally before permitting search."""
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {"passed": False, "no_search_performed": True}
    interrupted = None
    try:
        point = oracle.vector(parameters)
        lifted = continuation.to_original(original, oracle, parameters)
        reference, _ = budget.numerical(
            lambda d: original.evaluate(original.vector(lifted), d, jacobian=False)
        )
        residual, _ = budget.numerical(
            lambda d: oracle.evaluate(point, d, jacobian=False)
        )
        gap = float(np.max(np.abs(reference - residual), initial=0))
        check = budget.numerical(
            lambda d: recovery._derivative(oracle, point, d, _policy())
        )
        result.update(
            lift_prediction_discrepancy=gap,
            derivative_check=public._jsonable(check),
            passed=bool(np.isfinite(gap) and gap <= 1e-7 and check.search_qualified),
        )
    except Exception as error:
        continuation._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "qualification")
    result.update(
        portfolio_audit=oracle.portfolio_audit,
        portfolio_execution=oracle.execution_receipt(),
    )
    return continuation._finish(
        result, budget, checkpoint, "qualification", interrupted
    )


def verify_endpoint(
    original, oracle, parameters, seconds, checkpoint: Checkpoint
) -> dict:
    """Independently replay the lifted original equations using Radau."""
    methods = [oracle.portfolio_audit["route"], "original_Radau"]

    def save(value):
        checkpoint(
            value
            | {
                "methods": methods,
                "portfolio_execution": oracle.execution_receipt(),
            }
        )

    result = continuation.verify_endpoint(original, oracle, parameters, seconds, save)
    result["methods"] = methods
    result["portfolio_execution"] = oracle.execution_receipt()
    return result


def diagnose(
    original,
    oracle,
    parameters,
    verification,
    search_result,
    seconds,
    checkpoint: Checkpoint,
) -> dict:
    """Separate local adequacy from original-factor identifiability and budgets."""
    if oracle.original is not original:
        raise ValueError("diagnosis oracle identity differs")
    budget = prefix_execution._budget(seconds, seconds + 60)
    result = {"assessment": None, "original_parameter_identifiability_certified": False}
    interrupted = None
    try:
        point = oracle.vector(parameters)
        derivative = budget.numerical(
            lambda d: recovery._derivative(oracle, point, d, _policy())
        )
        result["derivative_check"] = public._jsonable(derivative)
        residual, jacobian = budget.numerical(lambda d: oracle.evaluate(point, d))
        matches = search_result.get("last_accepted_parameters") == dict(parameters)
        history = (
            [
                {
                    key: step[key]
                    for key in adequacy.HistoryPoint.model_fields
                    if key in step
                }
                for step in search_result.get("accepted_steps", [])
            ]
            if matches
            else []
        )
        evidence = (
            {
                key: verification[key]
                for key in adequacy.OriginalVerification.model_fields
            }
            if set(adequacy.OriginalVerification.model_fields) <= verification.keys()
            else None
        )
        assessment = adequacy.assess(
            point,
            residual,
            jacobian,
            oracle.lower,
            oracle.upper,
            verification=evidence,
            derivative=derivative,
            stop_reason=search_result.get("stop_reason") if matches else None,
            budget_exhausted=bool(search_result.get("budget_exhausted", True)),
            history=history,
            coordinate_units=oracle.units,
            policy=_policy(),
        )
        result.update(
            assessment=public._jsonable(assessment),
            endpoint_matches_accepted_history=matches,
            assessment_scope="active reduced training coordinates only",
        )
    except Exception as error:
        continuation._invalidate(result, error, "diagnosis")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "diagnosis")
    result.update(
        portfolio_audit=oracle.portfolio_audit,
        portfolio_execution=oracle.execution_receipt(),
    )
    return continuation._finish(result, budget, checkpoint, "diagnosis", interrupted)
