"""M31: candidate-bound training evidence; no scientific specification verdict.

This opt-in adapter observes a sealed fit. It cannot fit parameters, revise an
equation, read validation, or promote a local optimum to a family certificate.
"""

import ast
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting import scaling_recovery as recovery
from autoformalism.fitting.affine_propagation import numerically_verified
from autoformalism.fitting.fitting_assessment import assess
from autoformalism.fitting.output_activity import certificate
from autoformalism.fitting.sensitivity_contract import certify_expressions
from autoformalism.schemas.base import StrictSchema


class Policy(StrictSchema):
    """One diagnostic allowance, separate from the historical fitting budget."""

    check_seconds: float = Field(default=180, ge=1, le=600)


class Binding(StrictSchema):
    """Hashes bind all evidence to one retained vector and training problem."""

    problem_sha256: str
    request_sha256: str
    candidate_sha256: str
    training_sha256: str
    parameters_sha256: str | None
    fit_sha256: str
    backend_sha256: str
    fitting_policy_sha256: str
    feedback_policy_sha256: str
    diagnostic_allowance_seconds: float
    source_sha256: str
    runtime_sha256: str


class Dependency(StrictSchema):
    """Syntactic output reachability is not required-mechanism certification."""

    target: str
    states: list[str]
    parameters: list[str]
    external_inputs: list[str]
    auxiliaries: list[str]
    fixed_covariates: list[str]
    initial_parameters: list[str]
    initial_data_sources: list[str]
    scope: Literal["syntactic_output_dependency"] = "syntactic_output_dependency"


class Residual(StrictSchema):
    """Compact per-target summaries, without individual samples or causal claims."""

    target: str
    sample_count: int
    normalization_scale: float
    primary_nmse: float
    alternate_nmse: float
    mean_normalized_residual: float
    worst_trajectory: str
    worst_trajectory_nmse: float
    maximum_absolute_normalized_residual: float
    maximum_residual_trajectory: str
    maximum_residual_time: float
    residual_convention: Literal["prediction_minus_observation"] = (
        "prediction_minus_observation"
    )


class LocalSensitivity(StrictSchema):
    """Active coefficients and initializer parameters jointly, in fixed units."""

    parameters: list[str]
    units: list[float]
    singular_values: list[float]
    rank: int
    parameter_count: int
    condition_number: float | None
    column_norms: list[float]
    weak_direction: dict[str, float]
    initials_included: Literal[True]
    global_identifiability_certified: Literal[False] = False


class Diagnostics(StrictSchema):
    """Absent diagnostics never erase an incumbent or fabricate a residual."""

    status: Literal["complete", "unavailable", "interrupted", "skipped"]
    artifact_sha256: str | None = None
    solver_agreement: bool | None = None
    strict_sensitivity_verification: bool | None = None
    retained_metrics_reproduced: bool | None = None
    replay_training_nmse: float | None = None
    replay_alternate_training_nmse: float | None = None
    replay_worst_trajectory_nmse: float | None = None
    replay_solver_loss_discrepancy: float | None = None
    replay_solver_prediction_discrepancy: float | None = None
    residuals: list[Residual] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    active_sensitivity: LocalSensitivity | None = None
    sensitivity_status: Literal[
        "available", "unavailable", "not_applicable", "not_verified"
    ] = "not_verified"
    derivative_scope: Literal[
        "smooth_local", "piecewise_branch_convention", "not_checked"
    ] = "not_checked"
    inactive_parameters: list[str] = Field(default_factory=list)
    disconnected_states: list[str] = Field(default_factory=list)


class DiagnosticCost(StrictSchema):
    wall_seconds: float | None
    cpu_seconds: float | None
    budget_charge_seconds: float
    calls: int | None
    accounting_complete: bool
    optimization_calls: Literal[0] = 0


class SearchWitness(StrictSchema):
    """A particular saved operation, never a solver's unbound status message."""

    kind: Literal["restricted_output_optimum", "local_stationarity", "improvement"]
    operation: str
    operation_sha256: str
    parameters_sha256: str
    scope: Literal["fixed_candidate_output", "retained_point_local_search"]
    specification_incompatibility_certified: Literal[False] = False


class FeedbackPacket(StrictSchema):
    protocol: Literal["fitting-feedback-1"] = "fitting-feedback-1"
    identity: Binding
    diagnosis: Literal[
        "accurate_local_evidence",
        "accurate_ambiguity",
        "accurate_unresolved",
        "solved_restricted_problem_poor_fit",
        "stationary_poor_fit",
        "search_improving",
        "search_incomplete",
        "rollout_unverified",
    ]
    recommended_action: Literal[
        "retain_with_local_evidence",
        "assess_parameter_uncertainty",
        "review_equations",
        "review_equations_or_new_starts",
        "continue_fitting",
        "diversify_or_extend_search",
        "verify_numerics",
    ]
    fit_quality: str
    historical_fit_quality: str
    parameter_information: str
    training_nmse: float | None
    target_nmse: float
    worst_trajectory_target_nmse: float
    search_witness: SearchWitness | None
    diagnostics: Diagnostics
    cost: DiagnosticCost
    historical_fit_cost: adapter.FitCost
    specification_verdict: Literal["not_assessed"] = "not_assessed"
    automatic_specification_revision_allowed: Literal[False] = False
    confidence_probability: None = None
    profiles_minimized: Literal[False] = False
    scientific_correctness_certified: Literal[False] = False
    validation_used: Literal[False] = False
    test_data_opened: Literal[False] = False
    parameters_changed: Literal[False] = False
    limitations: list[str]


def validate_evidence(problem, fit, backend):
    """Reject cross-model/point/policy substitutions before any diagnostic work."""
    if not isinstance(problem, checks.TrainingProblem):
        raise TypeError("requires a typed training-only problem")
    if not isinstance(fit, adapter.FitResult):
        raise TypeError("requires a typed FitResult")
    digest = public.content_sha256(problem.model_dump(mode="json"))
    if (
        fit.identity["problem_sha256"] != digest
        or backend["identity"]["problem_sha256"] != digest
        or fit.backend_sha256 != public.content_sha256(backend)
        or adapter.summarize(backend, fit.identity) != fit
        or backend["identity"]["policy"] != fit.identity["policy"]["search"]
        or public.content_sha256(backend["identity"]["starts"])
        != fit.identity["starts_sha256"]
    ):
        raise ValueError("feedback problem/fit/backend identity differs")
    if (backend.get("selected") or {}).get("parameters") != fit.parameters:
        raise ValueError("feedback retained parameter vector differs")


def dependencies(oracle):
    """Expose paths, including initial-map dependencies, without inferring claims."""
    system = oracle.system
    rhs, outputs, audit = certify_expressions(
        system.model, allow_piecewise=system.allow_piecewise
    )
    states, parameters = set(system.model.state_names), set(system.names)

    def names(expression):
        return {n.id for n in ast.walk(expression.tree) if isinstance(n, ast.Name)}

    rows = []
    for target, output in outputs.items():
        reached, used = set(), names(output)
        while new := (used & states) - reached:
            reached |= new
            used |= set().union(*(names(rhs[n]) for n in new))
        initial_symbols = set()
        direct = system.model.direct_state_observation_channels
        for state in reached & set(direct):
            initial_symbols.add(direct[state])
        for state in reached - set(direct):
            if state in audit.initial_expressions:
                initial_symbols |= names(audit.initial_expressions[state])
        initial = initial_symbols & parameters
        rows.append(
            Dependency(
                target=target,
                states=sorted(reached),
                parameters=sorted((used & parameters) | initial),
                external_inputs=sorted(
                    used & set(system.model.validated.context.external_inputs)
                ),
                auxiliaries=sorted(
                    used & set(system.model.validated.context.auxiliaries)
                ),
                fixed_covariates=sorted(
                    used & set(system.model.validated.context.fixed_covariates)
                ),
                initial_parameters=sorted(initial),
                initial_data_sources=sorted(initial_symbols - parameters),
            ).model_dump(mode="json")
        )
    return rows


def residual_summary(oracle, a, b):
    """Use the fitter's normalization and exact sample/channel ordering."""
    channels = oracle.system.channels
    sizes = [len(r.time) * len(channels) for r in oracle.train.trajectories]
    if len(a) != sum(sizes) or len(b) != sum(sizes):
        raise ValueError("feedback residual layout differs")
    first = np.split(a, np.cumsum(sizes)[:-1])
    second = np.split(b, np.cumsum(sizes)[:-1])
    result = []
    for k, target in enumerate(channels):
        values = [r.reshape(-1, len(channels))[:, k] for r in first]
        other = [r.reshape(-1, len(channels))[:, k] for r in second]
        worst = int(
            np.argmax(
                [
                    max(np.mean(x * x), np.mean(y * y))
                    for x, y in zip(values, other, strict=True)
                ]
            )
        )
        peak_row = int(np.argmax([np.max(np.abs(v)) for v in values]))
        peak = int(np.argmax(np.abs(values[peak_row])))
        merged = np.concatenate(values)
        result.append(
            Residual(
                target=target,
                sample_count=len(merged),
                normalization_scale=oracle.scales[target],
                primary_nmse=float(np.mean(merged**2)),
                alternate_nmse=float(np.mean(np.concatenate(other) ** 2)),
                mean_normalized_residual=float(np.mean(merged)),
                worst_trajectory=oracle.train.trajectories[worst].trajectory_id,
                worst_trajectory_nmse=float(
                    max(np.mean(values[worst] ** 2), np.mean(other[worst] ** 2))
                ),
                maximum_absolute_normalized_residual=float(abs(values[peak_row][peak])),
                maximum_residual_trajectory=oracle.train.trajectories[
                    peak_row
                ].trajectory_id,
                maximum_residual_time=float(
                    oracle.train.trajectories[peak_row].time[peak]
                ),
            ).model_dump(mode="json")
        )
    return result


def diagnose(problem, parameters, selected, deadline, checkpoint):
    """Three fixed-vector calls: two independent outputs, one joint Jacobian."""
    oracle = checks.Rollouts(
        problem,
        checks.ConfidencePolicy(
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
        ),
    )
    vector = oracle.vector(parameters)
    activity = certificate(oracle.system)
    checkpoint({"calls": 1})
    a, _ = oracle.evaluate(vector, deadline, jacobian=False, method="DOP853")
    checkpoint({"calls": 2})
    b, _ = oracle.evaluate(vector, deadline, jacobian=False, method="Radau")
    point = {
        "parameters": parameters,
        "training_nmse": float(np.mean(a * a)),
        "alternate_training_nmse": float(np.mean(b * b)),
        "solver_loss_discrepancy": float(abs(np.mean(a * a) - np.mean(b * b))),
        "solver_prediction_discrepancy": float(np.max(np.abs(a - b))),
    }
    strict = numerical.usable(point) and numerically_verified(point, 1e-12)
    reproduced = max(
        abs(point[k] - selected[k])
        for k in ("training_nmse", "alternate_training_nmse")
    ) <= max(
        1e-10, 1e-7 * max(point["training_nmse"], point["alternate_training_nmse"])
    )
    active = [
        i for i, n in enumerate(oracle.names) if n in activity["active_parameters"]
    ]
    sensitivity = None
    sensitivity_status = "not_applicable" if not active else "not_verified"
    if strict and reproduced and active:
        checkpoint({"calls": 3})
        sensitivity_status = "unavailable"
        try:
            r, j = oracle.evaluate(vector, deadline)
            if np.max(np.abs(r - a)) <= 1e-7:
                sensitivity = checks.sensitivity(
                    j[:, active],
                    oracle.units[active],
                    tuple(oracle.names[i] for i in active),
                )
                sensitivity_status = "available"
        except (TimeoutError, ValueError, RuntimeError, ArithmeticError):
            # A failed derivative check must not erase the two completed rollouts.
            pass
    return {
        "solver_agreement": numerical.usable(point),
        "strict_sensitivity_verification": strict,
        "retained_metrics_reproduced": reproduced,
        "replay_training_nmse": point["training_nmse"],
        "replay_alternate_training_nmse": point["alternate_training_nmse"],
        "replay_worst_trajectory_nmse": max(
            *nonlinear_rollout.trajectory_losses(oracle, a),
            *nonlinear_rollout.trajectory_losses(oracle, b),
        ),
        "replay_solver_loss_discrepancy": point["solver_loss_discrepancy"],
        "replay_solver_prediction_discrepancy": point["solver_prediction_discrepancy"],
        "residuals": residual_summary(oracle, a, b),
        "dependencies": dependencies(oracle),
        "active_sensitivity": sensitivity,
        "sensitivity_status": sensitivity_status,
        "derivative_scope": "piecewise_branch_convention"
        if oracle.system.has_piecewise
        else "smooth_local",
        "inactive_parameters": activity["inactive_parameters"],
        "disconnected_states": activity["disconnected_states"],
    }


def search_witness(backend, parameters):
    """Require saved evidence at the retained vector, not merely another branch."""
    if parameters is None:
        return None
    digest = public.content_sha256(parameters)
    route = backend["routing"]
    exact = bool(route.get("certificate", {}).get("fully_profiled_output")) or (
        bool(route.get("output_activity"))
        and not route["output_activity"]["active_parameters"]
    )
    found = []
    for operation in backend["operations"]:
        value = operation.get("value") or {}
        point = value.get("best") or {}
        if operation["status"] != "complete" or point.get("parameters") != parameters:
            continue
        loss = point.get("training_nmse")
        selected = backend["selected"]
        if loss is None or max(
            abs(loss - selected[k])
            for k in ("training_nmse", "alternate_training_nmse")
        ) > max(1e-10, 1e-7 * abs(loss)):
            continue
        reason = value.get("stop_reason")
        kind = None
        if reason == "output_problem_solved" and exact:
            kind = "restricted_output_optimum"
        elif (
            reason == "stationary_poor_fit"
            and value.get("stationarity", {}).get("parameters_sha256") == digest
        ):
            kind = "local_stationarity"
        elif (
            recovery.continuation_reason(
                value, recovery.Policy.model_validate(backend["identity"]["policy"])
            )
            == "improving"
        ):
            kind = "improvement"
        if kind:
            found.append(
                SearchWitness(
                    kind=kind,
                    operation=operation["operation"],
                    operation_sha256=public.content_sha256(operation),
                    parameters_sha256=digest,
                    scope="fixed_candidate_output"
                    if kind == "restricted_output_optimum"
                    else "retained_point_local_search",
                )
            )
    priority = {
        "restricted_output_optimum": 0,
        "local_stationarity": 1,
        "improvement": 2,
    }
    return min(found, key=lambda w: priority[w.kind]) if found else None


def classify(fit, backend, diagnostics, *, derivative_qualified=False):
    """A deterministic action contract, conditional on numerical evidence only."""
    selected = backend["selected"]
    information = "unavailable"
    sensitivity = diagnostics.active_sensitivity
    if (
        diagnostics.strict_sensitivity_verification
        and sensitivity
        and (diagnostics.derivative_scope == "smooth_local" or derivative_qualified)
    ):
        alternatives = [
            o["value"]
            for o in backend["operations"]
            if o.get("value") and "alternate_training_nmse" in o["value"]
        ]
        information = assess(
            selected,
            sensitivity.model_dump(),
            alternatives,
            [],
            fit.assessment.target_nmse,
        )["parameter_information"]
    witness = search_witness(backend, fit.parameters)
    if not numerical.usable(selected) or (
        diagnostics.status == "complete"
        and (
            not diagnostics.solver_agreement
            or not diagnostics.retained_metrics_reproduced
        )
    ):
        return "rollout_unverified", "verify_numerics", information, None
    accurate = fit.verification.target_reached
    if diagnostics.status == "complete":
        # Reproduction has a looser tolerance than accuracy. Check both numerical
        # objectives anew; do not inherit a historical target_reached flag.
        point = {
            "solver_loss_discrepancy": diagnostics.replay_solver_loss_discrepancy,
            "solver_prediction_discrepancy": (
                diagnostics.replay_solver_prediction_discrepancy
            ),
        }
        accurate = bool(
            numerically_verified(point, fit.assessment.target_nmse)
            and max(
                diagnostics.replay_training_nmse,
                diagnostics.replay_alternate_training_nmse,
            )
            <= fit.assessment.target_nmse
            and diagnostics.replay_worst_trajectory_nmse
            <= backend["identity"]["policy"]["trajectory_nmse"]
        )
    if accurate:
        if information == "local_sensitivity_resolved_profiles_not_tested":
            return (
                "accurate_local_evidence",
                "retain_with_local_evidence",
                information,
                witness,
            )
        diagnosis = (
            "accurate_ambiguity"
            if information != "unavailable"
            else "accurate_unresolved"
        )
        return diagnosis, "assess_parameter_uncertainty", information, witness
    if witness:
        diagnosis, action = {
            "restricted_output_optimum": (
                "solved_restricted_problem_poor_fit",
                "review_equations",
            ),
            "local_stationarity": (
                "stationary_poor_fit",
                "review_equations_or_new_starts",
            ),
            "improvement": ("search_improving", "continue_fitting"),
        }[witness.kind]
        return diagnosis, action, information, witness
    return "search_incomplete", "diversify_or_extend_search", information, None


def completed(folder: Path, *, runtime: bool = False) -> FeedbackPacket:
    """Read terminal evidence without rerunning diagnostics or renewing budgets."""
    saved = FeedbackPacket.model_validate(read_seal(folder / "result.json"))
    if saved.identity.model_dump(mode="json") != read_seal(folder / "identity.json"):
        raise ValueError("feedback resume identity differs")
    if runtime and (
        saved.identity.source_sha256 != public._source_identity()
        or saved.identity.runtime_sha256 != public.content_sha256(public._runtime())
    ):
        raise ValueError("feedback runtime/source differs")
    if saved.diagnostics.artifact_sha256 is not None:
        operation = read_seal(folder / "diagnostics/result.json")
        if (
            saved.diagnostics.artifact_sha256 != public.content_sha256(operation)
            or operation["identity"]
            != saved.identity.model_dump(mode="json")
            | {"allowance_seconds": saved.identity.diagnostic_allowance_seconds}
            or saved.diagnostics
            != Diagnostics(
                status=operation["status"],
                artifact_sha256=public.content_sha256(operation),
                **(operation.get("value") or {}),
            )
            or any(
                getattr(saved.cost, k) != operation[k]
                for k in (
                    "wall_seconds",
                    "cpu_seconds",
                    "budget_charge_seconds",
                    "calls",
                    "accounting_complete",
                )
            )
        ):
            raise ValueError("feedback diagnostics differ")
    return saved


def run(
    problem: checks.TrainingProblem,
    fit: adapter.FitResult,
    backend: dict,
    folder: Path,
    policy: Policy | None = None,
) -> FeedbackPacket:
    """Resume sealed diagnostics; no optimization, mutation, or scientific verdict."""
    policy = policy or Policy()
    validate_evidence(problem, fit, backend)
    binding = Binding(
        problem_sha256=public.content_sha256(problem.model_dump(mode="json")),
        request_sha256=public.content_sha256(problem.request.model_dump(mode="json")),
        candidate_sha256=public.content_sha256(
            public._lower(problem.request)[0].validated.candidate.model_dump(
                mode="json"
            )
        ),
        training_sha256=public.content_sha256(problem.training.model_dump(mode="json")),
        parameters_sha256=public.content_sha256(fit.parameters)
        if fit.parameters is not None
        else None,
        fit_sha256=public.content_sha256(fit.model_dump(mode="json")),
        backend_sha256=fit.backend_sha256,
        fitting_policy_sha256=public.content_sha256(fit.identity["policy"]),
        feedback_policy_sha256=public.content_sha256(policy.model_dump(mode="json")),
        diagnostic_allowance_seconds=policy.check_seconds,
        source_sha256=public._source_identity(),
        runtime_sha256=public.content_sha256(public._runtime()),
    )
    identity = binding.model_dump(mode="json")
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        path = folder / "result.json"
        if path.exists():
            saved = completed(folder)
            if saved.identity != binding:
                raise ValueError("feedback resume identity differs")
            return saved
        if fit.parameters is None:
            diagnostics = Diagnostics(status="skipped")
            cost = DiagnosticCost(
                wall_seconds=0,
                cpu_seconds=0,
                budget_charge_seconds=0,
                calls=0,
                accounting_complete=True,
            )
        else:
            operation = checks.operation(
                folder / "diagnostics",
                identity,
                policy.check_seconds,
                lambda end, save: diagnose(
                    problem, fit.parameters, backend["selected"], end, save
                ),
            )
            diagnostics = Diagnostics(
                status=operation["status"],
                artifact_sha256=public.content_sha256(operation),
                **(operation.get("value") or {}),
            )
            cost = DiagnosticCost(
                **{
                    k: operation[k]
                    for k in (
                        "wall_seconds",
                        "cpu_seconds",
                        "budget_charge_seconds",
                        "calls",
                        "accounting_complete",
                    )
                }
            )
        diagnosis, action, information, witness = classify(fit, backend, diagnostics)
        packet = FeedbackPacket(
            identity=binding,
            diagnosis=diagnosis,
            recommended_action=action,
            fit_quality="numerical_target_reached"
            if diagnosis.startswith("accurate_")
            else "unverified"
            if diagnosis == "rollout_unverified"
            else "above_numerical_target",
            historical_fit_quality=fit.assessment.fit_quality,
            parameter_information=information,
            training_nmse=max(
                diagnostics.replay_training_nmse,
                diagnostics.replay_alternate_training_nmse,
            )
            if diagnostics.status == "complete"
            else fit.assessment.training_nmse,
            target_nmse=fit.assessment.target_nmse,
            worst_trajectory_target_nmse=backend["identity"]["policy"][
                "trajectory_nmse"
            ],
            search_witness=witness,
            diagnostics=diagnostics,
            cost=cost,
            historical_fit_cost=fit.cost,
            limitations=[
                "Numerical targets apply to noiseless diagnostics, not a noise "
                "model or scientific adequacy.",
                "Poor fit of one equation does not falsify its specification "
                "or every compatible model.",
                "Residual location and syntactic reachability do not identify "
                "a faulty mechanism.",
                "Local sensitivity and finite alternatives do not certify "
                "unique coefficients or global identifiability.",
                "No full profile minimization, calibrated confidence, "
                "or automatic equation/specification edit.",
            ],
        )
        seal(path, packet.model_dump(mode="json"))
        return packet
