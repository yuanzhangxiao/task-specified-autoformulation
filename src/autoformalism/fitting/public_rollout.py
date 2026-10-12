"""Opt-in training-only bridge from public construction requests to M28 fitting."""

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    RuntimeExpressionError,
)
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import fitter, nonlinear_rollout, recovery_numerics
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.rollout_adapter import AdapterPolicy
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest


class Policy(StrictSchema):
    """Default guarded scaling; piecewise derivatives require explicit consent."""

    protocol: Literal["public-rollout-policy-1"] = "public-rollout-policy-1"
    allow_piecewise: bool = False
    adapter: AdapterPolicy = Field(
        default_factory=lambda: AdapterPolicy(strategy="guarded_only")
    )


class Inputs(StrictSchema):
    """No validation, private reference, per-trajectory fit or hidden-state data."""

    request: PublicFitRequest
    training: checks.TrainingOnlySplit
    incumbent: dict[str, float] | None = None


class Result(StrictSchema):
    """Capability failure is distinct from fit quality and scientific validity."""

    protocol: Literal["public-rollout-result-1"] = "public-rollout-result-1"
    identity: dict
    status: Literal["complete", "retained_unverified", "setup_unavailable"]
    preparation: dict
    fit: adapter.FitResult | None
    validation_used_for_selection: Literal[False] = False
    reference_values_used: Literal[False] = False
    test_data_opened: Literal[False] = False


def prepare(inputs: Inputs, policy: Policy) -> dict:
    """Lower exactly once, fill omitted guesses, and freeze domain-only restarts.

    Missing guesses use the midpoint of the existing runtime start interval,
    intersected with the declared domain. Supplied guesses are never clipped.
    Coordinate construction uses training observations and public initial maps.
    No rollout is performed and no new physical constraint is inferred here.
    """
    request, training = inputs.request, inputs.training
    for row in training.rows:
        for role in ("targets", "auxiliaries", "external_inputs", "fixed_covariates"):
            if set(getattr(row, role)) != set(getattr(request.context, role)):
                raise ValueError(f"train/{row.trajectory_id}: {role} differ")
    model, guesses, initialization = public._lower(request)
    if not model.parameter_names:
        raise ValueError(
            "fixed models without optimization variables need evaluation only"
        )
    settings = FitConfig()
    defaults = {}
    for spec in model.validated.candidate.parameters:
        variable = fitter._parameter_variable(model, spec, settings)
        if spec.name not in guesses:
            defaults[spec.name] = (variable.start_lower + variable.start_upper) / 2
    start = defaults | guesses
    train = public.unpack_split(training)
    schema = (
        checks.PiecewiseTrainingProblem
        if policy.allow_piecewise
        else checks.TrainingProblem
    )
    problem = schema(
        request=request,
        training=training,
        start=start,
        incumbent=inputs.incumbent if inputs.incumbent is not None else start,
        coordinates=training_coordinates(model, train, start),
    )
    options = {"reduce_inactive": True} if policy.adapter.search.poor_fit else {}
    oracle, profile, routing = nonlinear_rollout.build(problem, **options)
    starts = [
        start,
        *recovery_numerics.diverse_starts(
            oracle, problem, policy.adapter.search.restart_count, request.random_seed
        ),
    ]
    adapter.recovery.validate_starts(
        oracle, profile, problem, starts, allow_incumbent=True, **options
    )
    return {
        "problem": problem.model_dump(mode="json"),
        "starts": starts,
        "routing": routing,
        "defaulted_guesses": defaults,
        "default_policy": "runtime-start-interval-midpoint-1",
        "initialization_audit": initialization,
        "lowered_candidate_sha256": public.content_sha256(
            model.validated.candidate.model_dump(mode="json")
        ),
        "parameter_count": len(start),
        "target_count": len(request.context.targets),
        "state_count": len(model.state_names),
        "initial_parameter_names": sorted(
            set(start) - {p.name for p in request.base_candidate.parameters}
        ),
    }


def run(inputs: Inputs, policy: Policy, folder: Path) -> Result:
    """Freeze identity and setup before bounded search; exact resume is idempotent."""
    if not isinstance(inputs, Inputs):
        raise TypeError("requires typed public training-only Inputs")
    identity = {
        "inputs_sha256": public.content_sha256(inputs.model_dump(mode="json")),
        "policy": policy.model_dump(mode="json"),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
    }
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        if (folder / "result.json").exists():
            return completed(folder, identity)

        def setup(end, save):
            try:
                value = prepare(inputs, policy)
            except (ValueError, ModelValidationError, RuntimeExpressionError) as error:
                return {
                    "supported": False,
                    "error": str(error),
                    "type": type(error).__name__,
                }
            if monotonic() >= end:
                raise TimeoutError("public rollout preparation allowance exhausted")
            return {"supported": True, **value}

        preparation = checks.operation(folder / "preparation", identity, 180, setup)
        value = preparation.get("value") or {}
        fitted = None
        if value.get("supported"):
            schema = (
                checks.PiecewiseTrainingProblem
                if policy.allow_piecewise
                else checks.TrainingProblem
            )
            fitted = adapter.run(
                schema.model_validate(value["problem"]),
                value["starts"],
                policy.adapter,
                folder / "fit",
            )
        result = Result(
            identity=identity,
            status=fitted.status if fitted else "setup_unavailable",
            preparation=preparation,
            fit=fitted,
        )
        seal(folder / "result.json", result.model_dump(mode="json"))
        return result


def completed(folder: Path, identity: dict | None = None) -> Result:
    """Verify terminal handoff, preparation and backend before reuse or reporting."""
    saved = Result.model_validate(read_seal(folder / "result.json"))
    frozen = read_seal(folder / "identity.json")
    if saved.identity != frozen or (identity is not None and identity != frozen):
        raise ValueError("public rollout identity differs")
    if saved.preparation != read_seal(folder / "preparation/result.json"):
        raise ValueError("public rollout preparation differs")
    if saved.fit is not None:
        fit = adapter.FitResult.model_validate(read_seal(folder / "fit/result.json"))
        if fit != saved.fit or fit.backend_sha256 != public.content_sha256(
            read_seal(folder / "fit/backend.json")
        ):
            raise ValueError("public rollout backend differs")
        for point in (fit.parameters or {},):
            if not np.isfinite(list(point.values())).all():
                raise ValueError("public rollout retained nonfinite parameters")
    return saved
