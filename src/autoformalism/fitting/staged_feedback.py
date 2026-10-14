"""M32: durable fixed-vector diagnostics and empirical piecewise qualification.

Only typed training data enter this adapter. Independent stages have immutable
identities and separate allowances; interrupted work is never silently retried.
"""

from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import fitting_feedback as legacy
from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting import piecewise_assessment as piecewise
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import rollout_adapter as adapter
from autoformalism.fitting.affine_propagation import numerically_verified
from autoformalism.fitting.output_activity import certificate
from autoformalism.fitting.piecewise_assessment import PiecewiseAssessmentPolicy
from autoformalism.fitting.sensitivity_probe import symbolic_rollout
from autoformalism.schemas.base import StrictSchema


class Policy(StrictSchema):
    """Per-stage ceilings plus a shared perturbation allowance; no fitting work."""

    primary_seconds: float = Field(default=120, ge=1, le=1800)
    alternate_seconds: float = Field(default=360, ge=1, le=1800)
    sensitivity_seconds: float = Field(default=180, ge=1, le=1800)
    perturbation_seconds: float = Field(default=120, ge=1, le=1800)
    perturbation_total_seconds: float = Field(default=1800, ge=1, le=7200)
    relative_tolerance: float = Field(default=1e-12, ge=1e-13, le=1e-10)
    absolute_tolerance: float = Field(default=1e-14, ge=1e-15, le=1e-12)
    piecewise: PiecewiseAssessmentPolicy = Field(
        default_factory=lambda: PiecewiseAssessmentPolicy(include_weak_direction=True)
    )

    @property
    def total_seconds(self):
        return (
            self.primary_seconds
            + self.alternate_seconds
            + self.sensitivity_seconds
            + self.perturbation_total_seconds
        )


class StageRecord(StrictSchema):
    key: str
    status: Literal["complete", "unavailable", "interrupted"]
    artifact_sha256: str
    parameters_sha256: str
    solver: Literal["DOP853", "Radau"]
    sensitivities: bool
    allowance_seconds: float
    relative_tolerance: float
    absolute_tolerance: float
    wall_seconds: float | None
    cpu_seconds: float | None
    budget_charge_seconds: float
    calls_started: int | None
    calls_completed: int | None
    accounting_complete: bool
    error: str | None = None
    progress: dict = Field(default_factory=dict)


class Diagnostics(legacy.Diagnostics):
    derivative_scope: Literal[
        "smooth_local",
        "piecewise_branch_convention",
        "not_checked",
        "piecewise_numerically_supported",
    ] = "not_checked"
    output_status: Literal["verified", "unverified", "unavailable", "skipped"]
    reason_codes: list[str] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    piecewise_assessment: piecewise.PiecewiseAssessmentResult | None = None


class DiagnosticCost(legacy.DiagnosticCost):
    calls_completed: int | None
    stages_completed: int
    stages_unavailable: int
    bookkeeping_excluded: Literal[True] = True


class FeedbackPacket(legacy.FeedbackPacket):
    protocol: Literal["fitting-feedback-2"] = "fitting-feedback-2"
    diagnostics: Diagnostics
    cost: DiagnosticCost
    historical_search_witness: legacy.SearchWitness | None = None


def training_problem(raw):
    kind = (
        checks.PiecewiseTrainingProblem
        if raw.get("allow_piecewise")
        else checks.TrainingProblem
    )
    return kind.model_validate(raw)


def _metadata(oracle, vector, states, policy):
    """Save lightweight assembly inputs; symbolic setup stays inside stage time."""
    branch = None
    if oracle.system.has_piecewise:
        try:
            branch = piecewise.sampled_branch_summary(
                oracle, vector, states, policy.piecewise
            )
        except (ValueError, RuntimeError, ArithmeticError) as error:
            branch = piecewise.BranchSummary(status="unavailable", error=str(error))
    return {
        "names": list(oracle.names),
        "units": oracle.units.tolist(),
        "lower": [float(x) if np.isfinite(x) else None for x in oracle.lower],
        "upper": [float(x) if np.isfinite(x) else None for x in oracle.upper],
        "channels": list(oracle.system.channels),
        "scales": oracle.scales,
        "initial_parameters": list(oracle.system.initial_parameter_names),
        "has_piecewise": oracle.system.has_piecewise,
        "activity": certificate(oracle.system),
        "dependencies": legacy.dependencies(oracle),
        "branch_summary": branch.model_dump(mode="json") if branch else None,
    }


def _proxy(problem, metadata):
    return SimpleNamespace(
        names=tuple(metadata["names"]),
        units=np.array(metadata["units"]),
        lower=np.array([-np.inf if x is None else x for x in metadata["lower"]]),
        upper=np.array([np.inf if x is None else x for x in metadata["upper"]]),
        train=public.unpack_split(problem.training),
        scales=metadata["scales"],
        system=SimpleNamespace(
            channels=metadata["channels"],
            initial_parameter_names=metadata["initial_parameters"],
        ),
    )


def evaluate_stage(
    oracle, vector, deadline, checkpoint, method, sensitivities, *, metadata_policy=None
):
    """One complete residual evaluation, with explicit progress through trajectories."""
    oracle.vector(oracle.parameters(vector))
    settings = oracle.settings.model_copy(update={"integration_method": method})
    scales = np.array([oracle.scales[c] for c in oracle.system.channels])
    residuals, jacobians, states, counters = [], [], {}, {}
    checkpoint({"calls": 1, "calls_completed": 0, "phase": "integration"})
    for index, row in enumerate(oracle.train.trajectories):
        checkpoint({"trajectory": row.trajectory_id, "trajectories_completed": index})
        try:
            y, j, state, count = symbolic_rollout(
                oracle.system,
                row,
                vector,
                settings,
                deadline,
                sensitivities=sensitivities,
            )
        except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
            checkpoint(
                {
                    "failure_kind": "timeout"
                    if isinstance(error, TimeoutError)
                    else "integration_failure",
                    "failure_type": type(error).__name__,
                }
            )
            raise
        observed = np.column_stack([row.targets[c] for c in oracle.system.channels])
        residuals.append(((y - observed) / scales).ravel())
        states[row.trajectory_id], counters[row.trajectory_id] = state, count
        if sensitivities:
            jacobians.append((j / scales[None, :, None]).reshape(-1, len(vector)))
    residual = np.concatenate(residuals)
    jacobian = np.vstack(jacobians) if sensitivities else None
    if not np.isfinite(residual).all() or (
        jacobian is not None and not np.isfinite(jacobian).all()
    ):
        raise ValueError("nonfinite diagnostic output")
    checkpoint(
        {
            "calls_completed": 1,
            "trajectories_completed": len(residuals),
            "phase": "summarizing",
        }
    )
    value = {
        "residual": residual.tolist(),
        "jacobian": jacobian.tolist() if jacobian is not None else None,
        "trajectory_counters": counters,
    }
    if metadata_policy is not None:
        value["metadata"] = _metadata(oracle, vector, states, metadata_policy)
        value["sampled_states"] = {
            name: values.tolist() for name, values in states.items()
        }
    # A completed integration is still useful if bookkeeping crosses the
    # cooperative deadline. Charge actual time and expose the overrun; do not
    # discard a finished output because its final serialization ran late.
    value["deadline_overrun"] = monotonic() > deadline
    return value


def _record(key, operation):
    identity = operation["identity"]
    progress = operation.get("progress", {})
    return StageRecord(
        key=key,
        status=operation["status"],
        artifact_sha256=public.content_sha256(operation),
        parameters_sha256=identity["point_sha256"],
        solver=identity["solver"],
        sensitivities=identity["sensitivities"],
        allowance_seconds=identity["allowance_seconds"],
        relative_tolerance=identity["relative_tolerance"],
        absolute_tolerance=identity["absolute_tolerance"],
        wall_seconds=operation["wall_seconds"],
        cpu_seconds=operation["cpu_seconds"],
        budget_charge_seconds=operation["budget_charge_seconds"],
        calls_started=operation["calls"],
        calls_completed=None
        if not operation["accounting_complete"]
        else progress.get("calls_completed", 0),
        accounting_complete=operation["accounting_complete"],
        error=operation.get("error"),
        progress=progress,
    )


def _cost(records):
    def total(field):
        values = [getattr(r, field) for r in records]
        return None if any(x is None for x in values) else sum(values)

    return DiagnosticCost(
        wall_seconds=total("wall_seconds"),
        cpu_seconds=total("cpu_seconds"),
        budget_charge_seconds=total("budget_charge_seconds"),
        calls=total("calls_started"),
        calls_completed=total("calls_completed"),
        accounting_complete=all(r.accounting_complete for r in records),
        stages_completed=sum(r.status == "complete" for r in records),
        stages_unavailable=sum(r.status != "complete" for r in records),
    )


def _value(operations, key):
    row = operations.get(key, {})
    return row.get("value") if row.get("status") == "complete" else None


def _base_diagnostics(problem, selected, operations):
    records = [_record(k, op) for k, op in operations.items()]
    diagnostic = SimpleNamespace(
        **Diagnostics(
            status="unavailable", output_status="unavailable", stages=records
        ).__dict__
    )
    for r in records:
        if r.status != "complete":
            reason = (
                "INTERRUPTED"
                if r.status == "interrupted"
                else (
                    "TIMEOUT"
                    if r.progress.get("failure_kind") == "timeout"
                    or "allowance" in (r.error or "")
                    else "UNAVAILABLE"
                )
            )
            prefix = "PERTURBATION" if r.key.startswith("perturb/") else r.key.upper()
            diagnostic.reason_codes.append(prefix + "_" + reason)
    a, b = _value(operations, "primary"), _value(operations, "alternate")
    source = a or b
    if source:
        metadata = source["metadata"]
        diagnostic.dependencies = [
            legacy.Dependency.model_validate(d) for d in metadata["dependencies"]
        ]
        diagnostic.inactive_parameters = metadata["activity"]["inactive_parameters"]
        diagnostic.disconnected_states = metadata["activity"]["disconnected_states"]
        diagnostic.derivative_scope = (
            "piecewise_branch_convention"
            if metadata["has_piecewise"]
            else "smooth_local"
        )
        # Retain a completed primary value even when the independent replay fails.
        if a:
            diagnostic.replay_training_nmse = float(
                np.mean(np.array(a["residual"]) ** 2)
            )
        if b:
            diagnostic.replay_alternate_training_nmse = float(
                np.mean(np.array(b["residual"]) ** 2)
            )
    if not (a and b):
        if source:
            diagnostic.replay_worst_trajectory_nmse = max(
                nonlinear_rollout.trajectory_losses(
                    _proxy(problem, source["metadata"]),
                    np.array(source["residual"]),
                )
            )
        available = {
            "training_nmse": diagnostic.replay_training_nmse,
            "alternate_training_nmse": diagnostic.replay_alternate_training_nmse,
        }
        if any(
            value is not None
            and abs(value - selected[key]) > max(1e-10, 1e-7 * abs(value))
            for key, value in available.items()
        ):
            diagnostic.output_status = "unverified"
            diagnostic.retained_metrics_reproduced = False
            diagnostic.reason_codes.append("PARTIAL_REPLAY_METRICS_DIFFER")
        return Diagnostics.model_validate(vars(diagnostic))
    oracle = _proxy(problem, a["metadata"])
    ra, rb = np.array(a["residual"]), np.array(b["residual"])
    point = {
        "parameters": selected["parameters"],
        "training_nmse": float(np.mean(ra**2)),
        "alternate_training_nmse": float(np.mean(rb**2)),
        "solver_loss_discrepancy": float(abs(np.mean(ra**2) - np.mean(rb**2))),
        "solver_prediction_discrepancy": float(np.max(np.abs(ra - rb))),
    }
    reproduced = max(
        abs(point[k] - selected[k])
        for k in ("training_nmse", "alternate_training_nmse")
    ) <= max(
        1e-10, 1e-7 * max(point["training_nmse"], point["alternate_training_nmse"])
    )
    diagnostic.status = "complete"
    diagnostic.solver_agreement = numerical.usable(point)
    diagnostic.strict_sensitivity_verification = numerical.usable(
        point
    ) and numerically_verified(point, 1e-12)
    diagnostic.retained_metrics_reproduced = reproduced
    diagnostic.output_status = (
        "verified" if diagnostic.solver_agreement and reproduced else "unverified"
    )
    diagnostic.replay_worst_trajectory_nmse = max(
        *nonlinear_rollout.trajectory_losses(oracle, ra),
        *nonlinear_rollout.trajectory_losses(oracle, rb),
    )
    diagnostic.replay_solver_loss_discrepancy = point["solver_loss_discrepancy"]
    diagnostic.replay_solver_prediction_discrepancy = point[
        "solver_prediction_discrepancy"
    ]
    diagnostic.residuals = [
        legacy.Residual.model_validate(r)
        for r in legacy.residual_summary(oracle, ra, rb)
    ]
    if not diagnostic.strict_sensitivity_verification or not reproduced:
        diagnostic.reason_codes.append("STRICT_REPLAY_NOT_VERIFIED")
        return Diagnostics.model_validate(vars(diagnostic))
    active = a["metadata"]["activity"]["active_parameters"]
    j = _value(operations, "sensitivity")
    diagnostic.sensitivity_status = "not_applicable" if not active else "unavailable"
    if j:
        if np.max(np.abs(np.array(j["residual"]) - ra)) <= 1e-7:
            indices = [i for i, n in enumerate(oracle.names) if n in active]
            diagnostic.active_sensitivity = legacy.LocalSensitivity.model_validate(
                checks.sensitivity(
                    np.array(j["jacobian"])[:, indices],
                    oracle.units[indices],
                    tuple(oracle.names[i] for i in indices),
                )
            )
            diagnostic.sensitivity_status = "available"
        else:
            diagnostic.reason_codes.append("SENSITIVITY_BASE_ROLLOUT_DIFFERS")
    return Diagnostics.model_validate(vars(diagnostic))


def _assessment(
    problem, selected, policy, operations, diagnostics, *, perturbation_plan=None
):
    a, b, j = (_value(operations, k) for k in ("primary", "alternate", "sensitivity"))
    if not (
        a
        and b
        and j
        and diagnostics.active_sensitivity
        and a["metadata"]["has_piecewise"]
    ):
        return None
    oracle = _proxy(problem, a["metadata"])
    vector = np.array([selected["parameters"][n] for n in oracle.names])
    active, jacobian = (
        a["metadata"]["activity"]["active_parameters"],
        np.array(j["jacobian"]),
    )
    plan = (
        piecewise.perturbation_plan(
            oracle, vector, active, policy.piecewise, jacobian=jacobian
        )
        if perturbation_plan is None
        else perturbation_plan
    )
    evaluations = {}
    for item in plan:
        primary, alternate = (
            _value(operations, f"perturb/{item.key}/{solver}")
            for solver in ("primary", "alternate")
        )
        evaluations[item.key] = piecewise.PerturbationEvaluation(
            status="complete" if primary and alternate else "unavailable",
            residual=primary["residual"] if primary else None,
            alternate_residual=alternate["residual"] if alternate else None,
            error=item.reason if not item.in_domain else None,
        )
    return piecewise.assess_derivatives(
        oracle,
        vector,
        active,
        policy.piecewise,
        a["residual"],
        jacobian,
        evaluations,
        base_alternate_residual=b["residual"],
        branch_summary=piecewise.BranchSummary.model_validate(
            a["metadata"]["branch_summary"]
        ),
        frozen_plan=plan,
    )


def _assemble(
    problem, fit, backend, policy, binding, operations, *, perturbation_plan=None
):
    if fit.parameters is None:
        diagnostics = SimpleNamespace(
            **Diagnostics(
                status="skipped",
                output_status="skipped",
                reason_codes=["NO_RETAINED_ENDPOINT"],
            ).__dict__
        )
    else:
        diagnostics = SimpleNamespace(
            **_base_diagnostics(problem, backend["selected"], operations).__dict__
        )
        result = _assessment(
            problem,
            backend["selected"],
            policy,
            operations,
            diagnostics,
            perturbation_plan=perturbation_plan,
        )
        diagnostics.piecewise_assessment = result
        if result:
            diagnostics.reason_codes.append("PIECEWISE_" + result.status.upper())
            if result.status == "supported":
                diagnostics.derivative_scope = "piecewise_numerically_supported"
        if (
            sum(
                o["budget_charge_seconds"]
                for k, o in operations.items()
                if k.startswith("perturb/")
            )
            >= policy.perturbation_total_seconds
        ):
            diagnostics.reason_codes.append("PERTURBATION_BUDGET_EXHAUSTED")
        diagnostics.reason_codes = sorted(set(diagnostics.reason_codes))
    available_losses = [
        value
        for value in (
            diagnostics.replay_training_nmse,
            diagnostics.replay_alternate_training_nmse,
        )
        if value is not None
    ]
    if (
        diagnostics.status == "unavailable"
        and fit.verification.target_reached
        and available_losses
        and (
            max(available_losses) > fit.assessment.target_nmse
            or diagnostics.replay_worst_trajectory_nmse
            > backend["identity"]["policy"]["trajectory_nmse"]
        )
    ):
        diagnostics.output_status = "unverified"
        diagnostics.reason_codes.append("PARTIAL_REPLAY_ACCURACY_CONTRADICTION")
    diagnostics.artifact_sha256 = public.content_sha256(operations)
    diagnostics = Diagnostics.model_validate(vars(diagnostics))
    qualified = bool(
        diagnostics.piecewise_assessment
        and diagnostics.piecewise_assessment.status == "supported"
    )
    diagnosis, action, information, witness = legacy.classify(
        fit, backend, diagnostics, derivative_qualified=qualified
    )
    historical_witness = witness
    if (
        witness is not None
        and witness.kind == "local_stationarity"
        and not (
            diagnostics.strict_sensitivity_verification
            and diagnostics.retained_metrics_reproduced
            and diagnostics.sensitivity_status == "available"
            and (diagnostics.derivative_scope == "smooth_local" or qualified)
        )
    ):
        diagnosis, action, witness = (
            "search_incomplete",
            "diversify_or_extend_search",
            None,
        )
        diagnostics = diagnostics.model_copy(
            update={
                "reason_codes": sorted(
                    set(diagnostics.reason_codes)
                    | {"HISTORICAL_STATIONARITY_UNQUALIFIED"}
                )
            }
        )
    if diagnostics.output_status == "unverified":
        diagnosis, action, information, witness = (
            "rollout_unverified",
            "verify_numerics",
            "unavailable",
            None,
        )
    return FeedbackPacket(
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
        training_nmse=max(available_losses)
        if available_losses
        else fit.assessment.training_nmse,
        target_nmse=fit.assessment.target_nmse,
        worst_trajectory_target_nmse=backend["identity"]["policy"]["trajectory_nmse"],
        search_witness=witness,
        historical_search_witness=historical_witness,
        diagnostics=diagnostics,
        cost=_cost(diagnostics.stages),
        historical_fit_cost=fit.cost,
        limitations=[
            "Piecewise support is finite local numerical evidence, not classical "
            "differentiability, global identifiability or calibrated confidence.",
            "All structurally active coefficient and initializer axes plus the "
            "declared weak joint direction are checked; other directions and "
            "scales remain untested.",
            "Branch margins are sampled, not certified between observations; "
            "solver disagreement is an empirical noise floor, not an error bound.",
            "No full profile minimization, scientific verdict, automatic equation "
            "edit, or validation/test access.",
            "Diagnostic timing includes symbolic setup and integration; lightweight "
            "artifact assembly and verification are excluded. Historical fit "
            "cost is separate.",
        ],
    )


def _validate_operations(
    problem, fit, backend, policy, binding, operations, *, perturbation_plan=None
):
    """Derive every stage identity and allowance from the frozen diagnostic plan."""
    expected_keys = []

    def expect(key, parameters, method, sensitivities, allowance):
        expected_keys.append(key)
        if key not in operations:
            raise ValueError("diagnostic stage missing from plan")
        expected = binding.model_dump(mode="json") | {
            "stage": key,
            "point_sha256": public.content_sha256(parameters),
            "solver": method,
            "sensitivities": sensitivities,
            "relative_tolerance": policy.relative_tolerance,
            "absolute_tolerance": policy.absolute_tolerance,
            "allowance_seconds": allowance,
        }
        if operations[key]["identity"] != expected:
            raise ValueError("diagnostic stage metadata differs from frozen plan")
        operation = operations[key]
        charge = operation["budget_charge_seconds"]
        if not np.isfinite(charge) or charge < 0:
            raise ValueError("invalid diagnostic budget charge")
        if operation.get("budget_restarted") is not False:
            raise ValueError("diagnostic stage budget was restarted")
        if operation["status"] == "interrupted":
            if (
                operation["value"] is not None
                or operation["accounting_complete"] is not False
                or any(
                    operation[field] is not None
                    for field in ("wall_seconds", "cpu_seconds", "calls")
                )
                or charge != allowance
            ):
                raise ValueError("interrupted diagnostic accounting differs")
        else:
            if (
                operation["status"] not in {"complete", "unavailable"}
                or operation["accounting_complete"] is not True
                or operation["calls"] not in (0, 1)
                or any(
                    not isinstance(operation[field], (int, float))
                    or not np.isfinite(operation[field])
                    or operation[field] < 0
                    for field in ("wall_seconds", "cpu_seconds")
                )
                or charge != operation["wall_seconds"]
            ):
                raise ValueError("diagnostic stage accounting differs")
            if operation["status"] == "complete":
                value = operation["value"]
                if (
                    not isinstance(value, dict)
                    or operation["calls"] != 1
                    or operation["progress"].get("calls_completed") != 1
                ):
                    raise ValueError(
                        "complete diagnostic stage lacks completed evidence"
                    )
                residual = np.asarray(value["residual"], dtype=float)
                if (
                    residual.ndim != 1
                    or not len(residual)
                    or not np.isfinite(residual).all()
                ):
                    raise ValueError("invalid saved diagnostic residual")
                if sensitivities:
                    jacobian = np.asarray(value["jacobian"], dtype=float)
                    if (
                        jacobian.shape != (len(residual), len(parameters))
                        or not np.isfinite(jacobian).all()
                    ):
                        raise ValueError("invalid saved diagnostic Jacobian")
                elif value["jacobian"] is not None:
                    raise ValueError(
                        "output-only diagnostic unexpectedly contains Jacobian"
                    )
            elif operation["value"] is not None or not operation.get("error"):
                raise ValueError(
                    "unavailable diagnostic stage contains completed evidence"
                )

    if fit.parameters is not None:
        expect("primary", fit.parameters, "DOP853", False, policy.primary_seconds)
        expect("alternate", fit.parameters, "Radau", False, policy.alternate_seconds)
        base = _base_diagnostics(problem, backend["selected"], operations)
        a = _value(operations, "primary")
        if (
            a
            and base.strict_sensitivity_verification
            and base.retained_metrics_reproduced
            and a["metadata"]["activity"]["active_parameters"]
        ):
            expect(
                "sensitivity",
                fit.parameters,
                "DOP853",
                True,
                policy.sensitivity_seconds,
            )
            if base.active_sensitivity and a["metadata"]["has_piecewise"]:
                oracle = _proxy(problem, a["metadata"])
                vector = np.array([fit.parameters[n] for n in oracle.names])
                plan = (
                    piecewise.perturbation_plan(
                        oracle,
                        vector,
                        a["metadata"]["activity"]["active_parameters"],
                        policy.piecewise,
                        jacobian=np.array(
                            _value(operations, "sensitivity")["jacobian"]
                        ),
                    )
                    if perturbation_plan is None
                    else perturbation_plan
                )
                charged = 0.0
                for item in plan:
                    if not item.in_domain:
                        continue
                    for suffix, solver in (
                        ("primary", "DOP853"),
                        ("alternate", "Radau"),
                    ):
                        remaining = policy.perturbation_total_seconds - charged
                        if remaining <= 0:
                            break
                        key = f"perturb/{item.key}/{suffix}"
                        expect(
                            key,
                            dict(zip(oracle.names, item.vector, strict=True)),
                            solver,
                            False,
                            min(policy.perturbation_seconds, remaining),
                        )
                        charged += operations[key]["budget_charge_seconds"]
    if list(operations) != expected_keys:
        raise ValueError("diagnostic stages differ from frozen plan")


def completed(folder: Path, *, runtime=False):
    """Reassemble the packet from sealed stages without rerunning integrations."""
    from autoformalism.fitting.portable_feedback import load_snapshot

    return load_snapshot(folder, runtime=runtime).packet


def _binding(problem, fit, policy, *, source=None, runtime=None):
    return legacy.Binding(
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
        diagnostic_allowance_seconds=policy.total_seconds,
        source_sha256=source or public._source_identity(),
        runtime_sha256=runtime or public.content_sha256(public._runtime()),
    )


def run(
    problem, fit: adapter.FitResult, backend, folder: Path, policy: Policy | None = None
):
    """Durably evaluate an unchanged fit; each interrupted stage keeps its charge."""
    policy = policy or Policy()
    legacy.validate_evidence(problem, fit, backend)
    binding = _binding(problem, fit, policy)
    identity = binding.model_dump(mode="json")
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        seal(
            folder / "context.json",
            {
                "problem": problem.model_dump(mode="json"),
                "fit": fit.model_dump(mode="json"),
                "backend": backend,
                "policy": policy.model_dump(mode="json"),
            },
        )
        if (folder / "result.json").exists():
            return completed(folder)
        from autoformalism.fitting.portable_feedback import (
            freeze_plan,
            save_environment,
        )

        save_environment(folder)
        operations, oracle, plan = {}, None, ()

        def stage(key, parameters, method, sensitivities, seconds, metadata=False):
            def evaluate(deadline, checkpoint):
                nonlocal oracle
                checkpoint(
                    {"calls": 0, "calls_completed": 0, "phase": "symbolic_setup"}
                )
                if oracle is None:
                    oracle = checks.Rollouts(
                        problem,
                        checks.ConfidencePolicy(
                            relative_tolerance=policy.relative_tolerance,
                            absolute_tolerance=policy.absolute_tolerance,
                        ),
                    )
                return evaluate_stage(
                    oracle,
                    oracle.vector(parameters),
                    deadline,
                    checkpoint,
                    method,
                    sensitivities,
                    metadata_policy=policy if metadata else None,
                )

            stage_identity = identity | {
                "stage": key,
                "point_sha256": public.content_sha256(parameters),
                "solver": method,
                "sensitivities": sensitivities,
                "relative_tolerance": policy.relative_tolerance,
                "absolute_tolerance": policy.absolute_tolerance,
            }
            path = folder / "stages" / key
            save_environment(path)
            op = checks.operation(path, stage_identity, seconds, evaluate)
            # Keep the progress snapshot in a second sealed envelope; raw operation
            # remains immutable under the common operation-resume contract.
            envelope = path / "record.json"
            if envelope.exists():
                op = read_seal(envelope)
                if {k: v for k, v in op.items() if k != "progress"} != read_seal(
                    path / "result.json"
                ):
                    raise ValueError("stage progress envelope differs")
            else:
                progress = (
                    public._read(path / "progress.json")
                    if (path / "progress.json").exists()
                    else {}
                )
                op = op | {"progress": progress}
                seal(envelope, op)
            operations[key] = op

        if fit.parameters is not None:
            stage(
                "primary", fit.parameters, "DOP853", False, policy.primary_seconds, True
            )
            stage(
                "alternate",
                fit.parameters,
                "Radau",
                False,
                policy.alternate_seconds,
                True,
            )
            base = _base_diagnostics(problem, backend["selected"], operations)
            a = _value(operations, "primary")
            if (
                a
                and base.strict_sensitivity_verification
                and base.retained_metrics_reproduced
                and a["metadata"]["activity"]["active_parameters"]
            ):
                stage(
                    "sensitivity",
                    fit.parameters,
                    "DOP853",
                    True,
                    policy.sensitivity_seconds,
                )
                base = _base_diagnostics(problem, backend["selected"], operations)
                if base.active_sensitivity and a["metadata"]["has_piecewise"]:
                    proxy = _proxy(problem, a["metadata"])
                    vector = np.array([fit.parameters[n] for n in proxy.names])
                    plan = freeze_plan(
                        folder,
                        binding,
                        proxy,
                        vector,
                        a["metadata"]["activity"]["active_parameters"],
                        policy.piecewise,
                        np.array(_value(operations, "sensitivity")["jacobian"]),
                    )
                    for item in plan:
                        if not item.in_domain:
                            continue
                        for suffix, method in (
                            ("primary", "DOP853"),
                            ("alternate", "Radau"),
                        ):
                            charged = sum(
                                op["budget_charge_seconds"]
                                for key, op in operations.items()
                                if key.startswith("perturb/")
                            )
                            remaining = policy.perturbation_total_seconds - charged
                            if remaining <= 0:
                                break
                            stage(
                                f"perturb/{item.key}/{suffix}",
                                dict(zip(proxy.names, item.vector, strict=True)),
                                method,
                                False,
                                min(policy.perturbation_seconds, remaining),
                            )
        packet = _assemble(
            problem,
            fit,
            backend,
            policy,
            binding,
            operations,
            perturbation_plan=plan,
        )
        seal(
            folder / "stages.json",
            [[key, public.content_sha256(op)] for key, op in operations.items()],
        )
        seal(folder / "result.json", packet.model_dump(mode="json"))
        return packet
