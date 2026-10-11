"""Opt-in common training-only interface for qualified rollout fitting."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import scaling_recovery as recovery
from autoformalism.fitting.fitting_assessment import FittingAssessment
from autoformalism.schemas.base import StrictSchema


class AdapterPolicy(StrictSchema):
    """One shared search allowance; alternative scaling cannot enlarge it."""

    protocol: Literal["rollout-fitting-adapter-1"] = "rollout-fitting-adapter-1"
    strategy: Literal["guarded_only", "guarded_portfolio"] = "guarded_portfolio"
    search: recovery.Policy = Field(default_factory=recovery.Policy)


class TrainingVerification(StrictSchema):
    """Numerical evidence is distinct from optimization and parameter certainty."""

    target_reached: bool
    primary_nmse: float | None
    alternate_nmse: float | None
    worst_trajectory_nmse: float | None
    maximum_normalized_solver_difference: float | None
    tighter_check_used: bool


class FitCost(StrictSchema):
    """Unknown usage stays unknown; interrupted operations retain their charge."""

    blocks_used: int
    maximum_search_calls: int
    search_calls_started: int | None
    search_calls_completed: int | None
    search_budget_charge_seconds: float
    wall_seconds: float
    cpu_seconds: float
    budget_charge_seconds: float
    accounting_complete: bool


class FitResult(StrictSchema):
    """Usable handoff for construction, with no hidden coefficient oracle."""

    protocol: Literal["rollout-fit-result-1"] = "rollout-fit-result-1"
    identity: dict
    status: Literal["complete", "retained_unverified"]
    parameters: dict[str, float] | None
    selected_origin: str | None
    routing: dict
    verification: TrainingVerification
    assessment: FittingAssessment
    cost: FitCost
    backend_sha256: str
    validation_used_for_selection: Literal[False] = False
    reference_values_used: Literal[False] = False
    test_data_opened: Literal[False] = False


def summarize(backend: dict, identity: dict) -> FitResult:
    """Expose independent quality axes rather than one success flag."""
    selected = backend["selected"] or {}
    return FitResult(
        identity=identity,
        status=backend["status"],
        parameters=backend["retained_parameters"],
        selected_origin=selected.get("origin"),
        routing=backend["routing"],
        assessment=FittingAssessment.model_validate(backend["assessment"]),
        verification=TrainingVerification(
            target_reached=backend["accuracy_verified"],
            primary_nmse=selected.get("training_nmse"),
            alternate_nmse=selected.get("alternate_training_nmse"),
            worst_trajectory_nmse=selected.get("maximum_trajectory_nmse"),
            maximum_normalized_solver_difference=selected.get(
                "solver_prediction_discrepancy"
            ),
            tighter_check_used=selected.get("tight_verification", False),
        ),
        cost=FitCost(
            blocks_used=backend["blocks_used"],
            maximum_search_calls=backend["maximum_search_calls"],
            search_calls_started=backend["search_calls_started"],
            search_calls_completed=backend["search_calls_completed"],
            search_budget_charge_seconds=backend["search_budget_charge_seconds"],
            wall_seconds=backend["additional_wall_seconds"],
            cpu_seconds=backend["additional_cpu_seconds"],
            budget_charge_seconds=backend["additional_budget_charge_seconds"],
            accounting_complete=backend["cost_complete"]
            and backend["search_call_accounting_complete"],
        ),
        backend_sha256=public.content_sha256(backend),
    )


def run(
    problem: checks.TrainingProblem,
    starts: list[dict[str, float]],
    policy: AdapterPolicy,
    folder: Path,
) -> FitResult:
    """Fit explicit shared parameters/initializers using training observations only.

    Caller supplies a frozen, domain-only restart list beginning at problem.start.
    A distinct supplied incumbent is independently verified before being retained.
    Unsupported equations/domains raise validation errors rather than gaining an
    invented residual penalty. Piecewise derivatives require explicit opt-in.
    """
    if not isinstance(problem, checks.TrainingProblem):
        raise TypeError("adapter requires a typed training-only problem")
    identity = {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
        "starts_sha256": public.content_sha256(starts),
        "policy": policy.model_dump(mode="json"),
    }
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        result_path, backend_path = folder / "result.json", folder / "backend.json"
        if result_path.exists():
            saved = FitResult.model_validate(read_seal(result_path))
            if (
                saved.identity != identity
                or saved.backend_sha256
                != public.content_sha256(read_seal(backend_path))
            ):
                raise ValueError("adapter result/identity differs")
            return saved
        backend = recovery.run(
            problem,
            starts,
            policy.search,
            "adapter_" + policy.strategy,
            folder / "search",
        )
        seal(backend_path, backend)
        result = summarize(backend, identity)
        seal(result_path, result.model_dump(mode="json"))
        return result
