"""Precision-qualified warm continuation of unchanged saved affine endpoints.

The campaign owns nonrenewable stage reservations and old-incumbent retention.
Only a complete qualification bound to this exact start, chart and precision can
open a shared search. No new starts, equation revisions or tolerance relaxation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.prefix_additive_reduction import AdditiveTrainingOracle

PROTOCOL = "portfolio-certified-additive-polish-1"


def make_oracle(problem, training, numerical, retained, *, precision_mode="guarded"):
    """Compose existing reductions and an exact sum chart around the saved point."""
    original, base = recovery.make_oracle(
        problem,
        training,
        numerical,
        retained,
        precision_mode=precision_mode,
        structural_reduction=True,
    )
    original.vector(retained)
    reduced = AdditiveTrainingOracle(base, retained)
    reduced.incumbent_original_parameters = deepcopy(dict(retained))
    audit = {
        "protocol": PROTOCOL,
        "route": "certified_affine",
        "problem_sha256": public.content_sha256(problem),
        "training_sha256": public.content_sha256(training),
        "source_parameters_sha256": public.content_sha256(retained),
        "numerical_policy": numerical.model_dump(mode="json"),
        "affine_precision_policy": asdict(original.precision_policy),
        "original_coordinates": len(original.names),
        "prior_coordinates": len(base.names),
        "optimizer_coordinates": len(reduced.names),
        "reduction": reduced.reduction.receipt,
        "optimizer_units": reduced.units.tolist(),
        "start_policy": "projected_saved_incumbent_only",
        "original_equations_unchanged": True,
        "original_parameter_identifiability_certified": False,
    }
    original.portfolio_audit = deepcopy(audit)
    return original, portfolio.PortfolioOracle(reduced, audit)


def _identity(oracle, parameters):
    """Bind derivative evidence to the unchanged model, chart, point and settings."""
    oracle.vector(parameters)
    return {
        "protocol": PROTOCOL,
        "portfolio_audit_sha256": public.content_sha256(oracle.portfolio_audit),
        "parameters_sha256": public.content_sha256(parameters),
        "solver_settings_sha256": public.content_sha256(oracle.original.settings),
        "affine_precision_policy": asdict(oracle.original.precision_policy),
        "derivative_policy": portfolio._policy().model_dump(mode="json"),
    }


def qualification(original, oracle, parameters, seconds, checkpoint):
    """Qualify the incumbent lift, original Radau agreement and every derivative."""
    if oracle.original is not original:
        raise ValueError("qualification oracle identity differs")
    if dict(parameters) != oracle.start:
        raise ValueError("qualification requires the projected saved incumbent")
    budget = execution._budget(seconds, seconds + 60)
    result = {
        "protocol": PROTOCOL,
        "qualification_identity": _identity(oracle, parameters),
        "passed": False,
        "no_search_performed": True,
        "original_incumbent_preserved": True,
        "calls_started": 0,
        "calls_completed": 0,
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
    }
    before = deepcopy(oracle.counts)
    interrupted = None

    def call(function):
        result["verification_calls_attempted"] += 1
        budget.callback(lambda: checkpoint(deepcopy(result)))

        def evaluate(deadline):
            result["calls_started"] += 1
            result["verification_calls_started"] += 1
            value = function(deadline)
            result["calls_completed"] += 1
            result["verification_calls_completed"] += 1
            return value

        value = budget.numerical(evaluate)
        budget.check()
        return value

    try:
        point = oracle.vector(parameters)
        lifted = continuation.to_original(original, oracle, parameters)
        source, _ = call(
            lambda deadline: original.evaluate(
                original.vector(oracle.incumbent_original_parameters),
                deadline,
                jacobian=False,
            )
        )
        residual, _ = call(
            lambda deadline: oracle.evaluate(point, deadline, jacobian=False)
        )
        reference, _ = call(
            lambda deadline: original.evaluate(
                original.vector(lifted),
                deadline,
                jacobian=False,
                method="Radau",
            )
        )
        source_gap = float(np.max(np.abs(source - residual), initial=0.0))
        ode_gap = float(np.max(np.abs(reference - residual), initial=0.0))
        loss, other_loss = float(np.mean(residual**2)), float(np.mean(reference**2))
        agreement = bool(
            np.isfinite([source_gap, ode_gap, loss, other_loss]).all()
            and source_gap <= portfolio.PAIRING_TOLERANCE
            and ode_gap <= portfolio.PAIRING_TOLERANCE
            and abs(loss - other_loss) <= max(1e-12, 1e-8 * max(loss, other_loss))
        )
        result.update(
            source_lift_prediction_discrepancy=source_gap,
            original_Radau_prediction_discrepancy=ode_gap,
            solver_loss_discrepancy=abs(loss - other_loss),
            training_nmse=loss,
            alternate_training_nmse=other_loss,
            original_equations_verified=agreement,
        )
        if agreement:
            derivative = call(
                lambda deadline: recovery._derivative(
                    oracle,
                    point,
                    deadline,
                    portfolio._policy(),
                )
            )
            result["derivative_check"] = public._jsonable(derivative)
            result["passed"] = bool(
                derivative.error is None and derivative.search_qualified
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
    result.update(
        verification_calls_failed=(
            result["verification_calls_started"]
            - result["verification_calls_completed"]
        ),
        stage_backend_calls={
            key: value - before[key] for key, value in oracle.counts.items()
        },
        portfolio_execution=oracle.execution_receipt(),
    )
    return continuation._finish(
        result,
        budget,
        checkpoint,
        "qualification",
        interrupted,
        preserve_diagnostic=True,
    )


def search(
    oracle, start, seconds, maximum_calls, checkpoint, *, qualification: Mapping
):
    """Start one bounded continuation only from matching complete precision evidence."""
    try:
        derivative = adequacy.DerivativeCheck.model_validate(
            qualification.get("derivative_check")
        )
        required = tuple(int(i) for i in np.flatnonzero(oracle.lower < oracle.upper))
        derivative_matches = (
            derivative.error is None
            and derivative.search_qualified
            and derivative.checked_indices == derivative.required_indices == required
            and derivative.reduced_vector == tuple(oracle.vector(start))
            and derivative.coordinate_units == tuple(oracle.units)
            and derivative.policy == portfolio._policy()
            and derivative.lower_bounds_sha256 == adequacy._digest(oracle.lower)
            and derivative.upper_bounds_sha256 == adequacy._digest(oracle.upper)
        )
    except (TypeError, ValueError):
        derivative_matches = False
    if (
        qualification.get("status") != "complete"
        or qualification.get("passed") is not True
        or qualification.get("original_equations_verified") is not True
        or qualification.get("qualification_identity") != _identity(oracle, start)
        or dict(start) != oracle.start
        or not derivative_matches
    ):
        raise ValueError("complete matching incumbent precision qualification required")
    result = portfolio.search(oracle, start, seconds, maximum_calls, checkpoint)
    result["qualification_sha256"] = public.content_sha256(qualification)
    result["original_incumbent_preserved"] = True
    return result


def verify_endpoint(original, oracle, parameters, seconds, checkpoint):
    """Replay a proposed retained point through unchanged original equations."""
    return portfolio.verify_endpoint(original, oracle, parameters, seconds, checkpoint)


def diagnose(
    original, oracle, parameters, verification, search_result, seconds, checkpoint
):
    """Retain strict convergence gates and only the endpoint's own accepted history."""
    return portfolio.diagnose(
        original, oracle, parameters, verification, search_result, seconds, checkpoint
    )
