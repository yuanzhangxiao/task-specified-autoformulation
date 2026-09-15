"""Unchanged sensitivity refinement from one complete retained parameter vector."""

from __future__ import annotations

import math
from pathlib import Path
from time import monotonic

from scipy.optimize import least_squares

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
from autoformalism.fitting.fitter import evaluate_fitted_candidate
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.stagnation import instrumented_fit
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def _score(model, split, parameters, scale, settings) -> dict:
    """Production rollout with frozen learned causal maps, never local refitting."""
    initials, metrics = evaluate_fitted_candidate(
        model,
        split,
        global_parameters=parameters,
        global_initial_conditions={},
        target_scales={"v01": scale},
        config=settings,
        fit_trajectory_initial_conditions=False,
    )
    return {
        "metrics": public._jsonable(metrics),
        "local_initials": public._jsonable(initials),
    }


def run_extension(frozen: dict, directory: Path) -> dict:
    """Warm-start once; compare only training objectives and retain the incumbent."""
    setup_started = monotonic()
    parent = frozen["parent"]
    parent_result = parent["result"]["result"]
    request = PublicFitRequest.model_validate(parent["freeze"]["request"])
    training = public.unpack_split(
        PublicSplit.model_validate(parent["freeze"]["training"])
    )
    model, _, _ = public._lower(request)
    if tuple(request.context.targets) != ("v01",):
        raise ValueError("continuation pilot supports only target v01")
    # This milestone continues a smooth sensitivity stage only. It does not
    # silently alter the existing piecewise polling/refinement policy.
    system = SymbolicODE(model, allow_piecewise=False)
    config = CollocationSensitivityConfig.model_validate(parent["freeze"]["settings"])
    settings = config.fit_config()
    scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("training scale must be positive and finite")
    start = dict(parent_result["parameters"])
    budget = EvaluationBudget(
        float("inf"), frozen["selection"]["additional_residual_calls"]
    )
    oracle = GuardedOracle(
        system,
        training,
        scale,
        settings,
        directory / "calls",
        None,
        sensitivities=True,
        budget=budget,
        point_seconds=max(30.0, config.recovery_probe_seconds),
    )
    vector = oracle.vector(start)  # Exact full layout, including learned initials.
    setup_seconds = monotonic() - setup_started
    started = monotonic()
    budget.deadline = started + frozen["selection"]["additional_seconds"]
    report, failure, initial_check = {}, None, None

    def optimizer(fun, x, **kwargs):
        kwargs.update(jac=oracle.jacobian, ftol=config.least_squares_ftol)
        return least_squares(fun, x, **kwargs)

    try:
        residual = oracle(vector)
        first_cost = float(0.5 * residual @ residual)
        old_cost = frozen["gate"]["parent_training_cost"]
        agrees = math.isclose(first_cost, old_cost, rel_tol=1e-5, abs_tol=1e-8)
        initial_check = {
            "parameters": start,
            "cost": first_cost,
            "parent_cost": old_cost,
            "agrees": agrees,
            "charged_to_extension_budget": True,
        }
        public._write(directory / "start_check.json", initial_check)
        if not agrees:
            raise ValueError("retained-parent objective differs under current runtime")
        report = instrumented_fit(
            oracle,
            start,
            diff_step=None,
            max_nfev=max(1, budget.maximum - budget.calls),
            settings=settings,
            optimizer=optimizer,
        )
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    numerical_seconds = monotonic() - started
    exhausted = (
        budget.calls >= budget.maximum
        or monotonic() >= budget.deadline
        or report.get("optimizer_status") in (-2, 0)
    )
    evidence = {
        "method": "warm_started_sensitivity",
        "initial_parameters": start,
        "optimizer_state_restored": False,
        "collocation_reruns": 0,
        "additional_starts": 0,
        "setup_seconds": setup_seconds,
        "numerical_seconds": numerical_seconds,
        "actual_residual_calls": budget.calls,
        "valid_residual_evaluations": oracle.valid_calls,
        "best_evaluated": oracle.best,
        "optimizer": report,
        "start_check": initial_check,
        "failure": failure,
        "failure_evidence": oracle.failure_evidence,
        "integration_failures": oracle.failures.count,
        "budget_exhausted": exhausted,
    }
    # Publish the useful numerical evidence before potentially expensive scoring.
    public._write(directory / "extension_checkpoint.json", evidence)
    selected, parameters = "parent", start
    train_metric, val_metric = parent_result["training"], parent_result["validation"]
    scoring, scoring_error = {}, None
    scoring_started = monotonic()
    best = oracle.best
    if (
        initial_check
        and initial_check["agrees"]
        and best
        and best["parameters"] != start
        and best["cost"] < frozen["gate"]["parent_training_cost"]
    ):
        try:
            scoring["training"] = _score(
                model, training, best["parameters"], scale, settings
            )
            metric = public._metrics(
                scoring["training"]["metrics"], request.context.targets
            )
            if not metric.available:
                scoring_error = "training: no complete finite production rollout"
            # Retain the historical incumbent on worse or unavailable production
            # training scores. Validation values never participate in this choice.
            if (
                metric.available
                and metric.normalized_mse < train_metric["normalized_mse"]
            ):
                selected, parameters = "extension", best["parameters"]
                train_metric = metric.model_dump(mode="json")
        except Exception as error:
            scoring_error = f"training: {type(error).__name__}: {error}"
    public._write(
        directory / "training_selection.json",
        {
            "selected": selected,
            "parameters": parameters,
            "training": train_metric,
            "training_only": True,
        },
    )
    if selected == "extension":
        # Open validation only after the final training selection. Its failure
        # is visible; it cannot cause selection of a different parameter vector.
        val_metric = public.PublicFitMetrics().model_dump(mode="json")
        try:
            validation = public.unpack_split(
                PublicSplit.model_validate(parent["freeze"]["validation"])
            )
            scoring["validation"] = _score(
                model, validation, parameters, scale, settings
            )
            val_metric = public._metrics(
                scoring["validation"]["metrics"],
                request.context.targets,
            ).model_dump(mode="json")
        except Exception as error:
            scoring_error = f"validation: {type(error).__name__}: {error}"
    native = None
    if parameters == report.get("native_optimizer_parameters"):
        native = report.get("optimizer_native_success")
    numerical_failure = (
        failure is not None or scoring_error is not None or not val_metric["available"]
    )
    feedback = (
        "budget_limited_unresolved"
        if exhausted
        and (failure is None or failure["type"] == "TimeoutError")
        and scoring_error is None
        else "numerical_failure_unresolved"
        if numerical_failure
        else "local_optimizer_stopped"
    )
    parent_calls = parent_result["actual_residual_calls"]
    fields = {
        "status": "extension_failed" if numerical_failure else "complete",
        "selected": selected,
        "parameters": parameters,
        "training": train_metric,
        "validation": val_metric,
        "extension_budget_exhausted": exhausted,
        "extension_residual_calls": budget.calls,
        "cumulative_residual_calls": None
        if parent_calls is None
        else parent_calls + budget.calls,
        "extension_numerical_seconds": numerical_seconds,
        "native_optimizer_converged": native,
        "feedback_status": feedback,
        "message": (
            "One authorized warm-start window ended; selected by training score. "
            "Report fit quality separately; no structural impossibility "
            "or automatic follow-up."
        ),
    }
    return public._jsonable(
        {
            **evidence,
            "scoring": scoring,
            "scoring_error": scoring_error,
            "scoring_seconds": monotonic() - scoring_started,
            "result_fields": fields,
        }
    )
