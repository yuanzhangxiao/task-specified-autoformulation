"""Private Phase-B mechanism-response subspace evaluation after model freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.benchmarks.phase_b_gates import mechanism_gate_definition
from autoformalism.benchmarks.phase_b_generation import (
    Family,
    PrivateTrajectory,
    phase_b_protocols,
    simulate_phase_b,
)
from autoformalism.benchmarks.phase_b_public import (
    PhaseBPublicSpec,
    phase_b_public_spec,
    render_phase_b_prompts,
)
from autoformalism.data import DatasetSplit, SplitName
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.final_evaluation import (
    FrozenEvaluationSubject,
    HiddenMechanismEndpoint,
)
from autoformalism.rebuttal.hidden import HiddenSubspaceMetric, hidden_subspace_nmse
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
    mechanism_claim_components,
)
from autoformalism.schemas import CandidateModel

HiddenMode = Literal[
    "mechanism_response_equivalence",
    "mechanism_sensitivity_subspace",
    "not_applicable",
]

NOMINAL_PAIR_RELATIVE_TOLERANCE = 1e-7
NOMINAL_PAIR_ABSOLUTE_TOLERANCE = 1e-8


class PhaseBHiddenSubspaceContract(BaseModel):
    """Frozen private definition of one representation-invariant endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-hidden-subspace-contract-1"] = (
        "phase-b-hidden-subspace-contract-1"
    )
    benchmark_id: str
    family: Family
    task: str
    tier: Literal["easy", "hard"]
    dynamics: Literal["canonical", "perturbed"]
    mode: HiddenMode
    private_mechanism_directions: tuple[str, ...]
    claimed_dimension: int = Field(ge=0)
    target_sources: dict[str, str]
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensitivity_fraction: float = Field(default=1e-3, gt=0.0, le=0.01)

    @model_validator(mode="after")
    def dimension_matches_mode(self) -> PhaseBHiddenSubspaceContract:
        if self.mode == "not_applicable":
            if self.claimed_dimension or self.private_mechanism_directions:
                raise ValueError("inapplicable hidden contract must have no subspace")
        elif not 0 < self.claimed_dimension <= len(
            self.private_mechanism_directions
        ):
            raise ValueError("hidden claimed dimension is unsupported")
        if not self.target_sources:
            raise ValueError("hidden contract requires public target mappings")
        return self

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class PhaseBHiddenSubspaceOutcome(BaseModel):
    """Checkpointable result for one frozen candidate and private contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-hidden-subspace-outcome-1"] = (
        "phase-b-hidden-subspace-outcome-1"
    )
    subject_id: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["available", "unrecovered", "not_applicable", "failed"]
    candidate_components: tuple[str, ...] = ()
    metric: HiddenSubspaceMetric | None = None
    error_type: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def fields_match_status(self) -> PhaseBHiddenSubspaceOutcome:
        if self.status == "available":
            if self.metric is None or not self.metric.recovered:
                raise ValueError("available hidden outcome requires recovery")
        elif self.status == "unrecovered":
            if self.metric is None or self.metric.recovered:
                raise ValueError("unrecovered outcome requires an unrecovered metric")
        elif self.metric is not None:
            raise ValueError(f"{self.status} hidden outcome cannot carry a metric")
        if self.status == "failed":
            if not self.error_type or not self.error:
                raise ValueError("failed hidden outcome requires diagnostics")
        elif self.error_type is not None or self.error is not None:
            raise ValueError("nonfailed hidden outcome cannot carry diagnostics")
        return self


def phase_b_hidden_subspace_contract(
    benchmark_id: str,
    *,
    data_root: Path = Path("data_raw"),
) -> PhaseBHiddenSubspaceContract:
    """Construct the prespecified private contract for a registered cell."""
    public = _public_spec_by_id(benchmark_id, data_root=data_root)
    prompt, _ = render_phase_b_prompts(public)
    target_sources = {
        item.public_name: item.private_source
        for item in public.channels
        if item.role == "target"
    }
    if public.family == "dalla_man" and public.task == "T4":
        mode: HiddenMode = "not_applicable"
        directions: tuple[str, ...] = ()
        dimension = 0
    else:
        definition = mechanism_gate_definition(
            public.family,
            public.tier,
            task=public.task if public.family == "dalla_man" else None,
            data_root=data_root,
        )
        mode = (
            "mechanism_response_equivalence"
            if public.family == "dalla_man" and public.task == "T1"
            else "mechanism_sensitivity_subspace"
        )
        directions = definition.mechanisms
        dimension = definition.claimed_dimension
    return PhaseBHiddenSubspaceContract(
        benchmark_id=benchmark_id,
        family=public.family,
        task=public.task,
        tier=public.tier,
        dynamics=public.dynamics,
        mode=mode,
        private_mechanism_directions=directions,
        claimed_dimension=dimension,
        target_sources=target_sources,
        public_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def evaluate_phase_b_hidden_subspace(
    subject: FrozenEvaluationSubject,
    *,
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    public_mechanism_spec: MechanismEvaluationSpec,
    contract: PhaseBHiddenSubspaceContract,
    private_data_root: Path = Path("data_raw"),
    maximum_trajectory_wall_time_seconds: float = 300.0,
    reference_directions: tuple[
        NDArray[np.float64], NDArray[np.float64]
    ]
    | None = None,
) -> tuple[FrozenEvaluationSubject, PhaseBHiddenSubspaceOutcome]:
    """Evaluate a frozen candidate against private mechanism response directions."""
    _validate_inputs(subject, training_split, test_split, contract)
    if contract.mode == "not_applicable":
        endpoint = HiddenMechanismEndpoint(
            mechanism_id="claimed_mechanism_response_subspace",
            status="not_applicable",
            recovered=False,
            message="the frozen benchmark evaluates flux compatibility instead",
        )
        updated = _append_endpoint(subject, endpoint)
        return updated, _outcome(
            subject,
            contract,
            status="not_applicable",
        )
    if subject.parameterization.status not in {"available", "not_required"}:
        raise ValueError(
            "hidden subspace evaluation requires replay-complete parameterization"
        )
    public_evaluation = evaluate_mechanisms(subject.candidate, public_mechanism_spec)
    structurally_recovered = (
        public_evaluation.mechanism_compliance_complete
        and public_evaluation.mechanism_compliance == 1.0
    )
    claims = mechanism_claim_components(subject.candidate, public_mechanism_spec)
    components = tuple(sorted(set().union(*map(set, claims.values()))))
    settings = FitConfig(
        integration_backend="solve_ivp",
        allow_derivative_regression=False,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        maximum_wall_time_seconds=maximum_trajectory_wall_time_seconds,
    )
    scales = subject.target_prediction.normalization_scales
    train_candidate = _candidate_directions(
        subject,
        training_split,
        components,
        contract.sensitivity_fraction,
        scales,
        settings,
    )
    test_candidate = _candidate_directions(
        subject,
        test_split,
        components,
        contract.sensitivity_fraction,
        scales,
        settings,
    )
    train_reference, test_reference = reference_directions or (
        phase_b_reference_directions(
            training_split=training_split,
            test_split=test_split,
            contract=contract,
            normalization_scales=scales,
            private_data_root=private_data_root,
        )
    )
    metric = hidden_subspace_nmse(
        train_candidate,
        train_reference,
        test_candidate,
        test_reference,
        claimed_dimension=contract.claimed_dimension,
        structurally_recovered=structurally_recovered,
    )
    endpoint = HiddenMechanismEndpoint(
        mechanism_id="claimed_mechanism_response_subspace",
        status="available" if metric.recovered else "missing",
        recovered=metric.recovered,
        aligned_test_nmse=metric.aligned_test_nmse,
        message=(
            "training-aligned mechanism-response subspace scored on sealed test"
            if metric.recovered
            else (
                "public mechanism compliance or candidate sensitivity rank was "
                "insufficient for recovery"
            )
        ),
    )
    updated = _append_endpoint(subject, endpoint)
    return updated, _outcome(
        subject,
        contract,
        status="available" if metric.recovered else "unrecovered",
        components=components,
        metric=metric,
    )


def phase_b_reference_directions(
    *,
    training_split: DatasetSplit,
    test_split: DatasetSplit,
    contract: PhaseBHiddenSubspaceContract,
    normalization_scales: dict[str, float],
    private_data_root: Path = Path("data_raw"),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate frozen private train/test mechanism-response direction matrices."""
    if contract.mode == "not_applicable":
        raise ValueError("inapplicable hidden contract has no reference directions")
    if training_split.name is not SplitName.TRAIN:
        raise ValueError("reference alignment requires the training split")
    if test_split.name is not SplitName.TEST:
        raise ValueError("reference scoring requires the test split")
    if set(normalization_scales) != set(contract.target_sources):
        raise ValueError("reference normalization scales differ from public targets")
    return (
        _private_directions(
            training_split,
            contract,
            normalization_scales,
            private_data_root=private_data_root,
        ),
        _private_directions(
            test_split,
            contract,
            normalization_scales,
            private_data_root=private_data_root,
        ),
    )


def failed_hidden_subspace_result(
    subject: FrozenEvaluationSubject,
    contract: PhaseBHiddenSubspaceContract,
    error: Exception,
) -> tuple[FrozenEvaluationSubject, PhaseBHiddenSubspaceOutcome]:
    """Retain a private evaluation failure without manufacturing a score."""
    endpoint = HiddenMechanismEndpoint(
        mechanism_id="claimed_mechanism_response_subspace",
        status="failed",
        recovered=False,
        message=f"{type(error).__name__}: {error}",
    )
    return _append_endpoint(subject, endpoint), _outcome(
        subject,
        contract,
        status="failed",
        error=error,
    )


def _candidate_directions(
    subject: FrozenEvaluationSubject,
    split: DatasetSplit,
    components: tuple[str, ...],
    fraction: float,
    scales: dict[str, float],
    settings: FitConfig,
) -> NDArray[np.float64]:
    nominal = _candidate_predictions(subject, split, subject.candidate, settings)
    columns = []
    for component in components:
        perturbed = _scaled_candidate_component(subject.candidate, component, fraction)
        values = _candidate_predictions(subject, split, perturbed, settings)
        columns.append((values - nominal) / fraction)
    rows = _expected_sample_count(split, tuple(scales))
    if not columns:
        return np.empty((rows, 0), dtype=float)
    return np.column_stack(columns)


def _candidate_predictions(
    subject: FrozenEvaluationSubject,
    split: DatasetSplit,
    candidate: CandidateModel,
    settings: FitConfig,
) -> NDArray[np.float64]:
    compiled = compile_candidate(candidate, subject.validation_context)
    output: list[NDArray[np.float64]] = []
    for trajectory in split.trajectories:
        deadline = monotonic() + (settings.maximum_wall_time_seconds or 300.0)
        simulation = simulate_trajectory(
            compiled,
            trajectory,
            subject.parameterization.global_parameters,
            subject.parameterization.global_initial_conditions,
            settings,
            deadline=deadline,
            reset_observed_states=False,
        )
        if not simulation.success:
            raise RuntimeError(
                f"candidate sensitivity rollout failed on {trajectory.trajectory_id}: "
                f"{simulation.message}"
            )
        matrix = np.column_stack(
            [
                simulation.predictions[target]
                / subject.target_prediction.normalization_scales[target]
                for target in subject.validation_context.targets
            ]
        )
        output.append(matrix.ravel())
    return np.concatenate(output)


def _private_directions(
    split: DatasetSplit,
    contract: PhaseBHiddenSubspaceContract,
    scales: dict[str, float],
    *,
    private_data_root: Path,
) -> NDArray[np.float64]:
    protocols = tuple(
        item
        for item in phase_b_protocols(
            contract.family,
            task=contract.task if contract.family == "dalla_man" else None,
        )
        if item.split == split.name.value
    )
    if len(protocols) != len(split.trajectories):
        raise ValueError("private/public trajectory counts differ")
    nominal = tuple(
        simulate_phase_b(
            protocol,
            dynamics=contract.dynamics,
            data_root=private_data_root,
        )
        for protocol in protocols
    )
    _validate_private_matches_public(split, nominal, contract)
    columns = []
    for mechanism in contract.private_mechanism_directions:
        shifted = tuple(
            simulate_phase_b(
                protocol,
                dynamics=contract.dynamics,
                data_root=private_data_root,
                private_mechanism_scales={
                    mechanism: 1.0 + contract.sensitivity_fraction
                },
            )
            for protocol in protocols
        )
        values = []
        for left, right in zip(shifted, nominal, strict=True):
            matrix = np.column_stack(
                [
                    (_private_values(left, source) - _private_values(right, source))
                    / scales[public_name]
                    for public_name, source in contract.target_sources.items()
                ]
            )
            values.append(matrix.ravel() / contract.sensitivity_fraction)
        columns.append(np.concatenate(values))
    return np.column_stack(columns)


def _validate_private_matches_public(
    split: DatasetSplit,
    private: tuple[PrivateTrajectory, ...],
    contract: PhaseBHiddenSubspaceContract,
) -> None:
    for public, reference in zip(split.trajectories, private, strict=True):
        if not np.allclose(public.time, reference.time, rtol=0.0, atol=1e-10):
            raise ValueError("private/public time grids differ")
        for public_name, source in contract.target_sources.items():
            observed = public.targets[public_name]
            expected = _private_values(reference, source)
            difference = np.abs(observed - expected)
            tolerance = (
                NOMINAL_PAIR_ABSOLUTE_TOLERANCE
                + NOMINAL_PAIR_RELATIVE_TOLERANCE * np.abs(expected)
            )
            if not np.all(difference <= tolerance):
                maximum_index = int(np.argmax(difference - tolerance))
                raise ValueError(
                    "private/public nominal target differs beyond the frozen "
                    f"numeric reproducibility tolerance: target={public_name}, "
                    f"trajectory={public.trajectory_id}, index={maximum_index}, "
                    f"max_abs={float(np.max(difference)):.12g}, "
                    f"allowed_at_index={float(tolerance[maximum_index]):.12g}, "
                    f"rtol={NOMINAL_PAIR_RELATIVE_TOLERANCE:.1e}, "
                    f"atol={NOMINAL_PAIR_ABSOLUTE_TOLERANCE:.1e}"
                )


def _private_values(
    trajectory: PrivateTrajectory,
    source: str,
) -> NDArray[np.float64]:
    if source in trajectory.state_names:
        return trajectory.states[:, trajectory.state_names.index(source)]
    if source in trajectory.derived:
        return trajectory.derived[source]
    raise ValueError(f"private target source is unavailable: {source}")


def _scaled_candidate_component(
    candidate: CandidateModel,
    component: str,
    fraction: float,
) -> CandidateModel:
    payload = candidate.model_dump(mode="json")
    factor = 1.0 + fraction
    changed = False
    for process in payload["processes"]:
        if process["name"] == component:
            process["expression"] = f"({factor:.17g}) * ({process['expression']})"
            changed = True
    for equation in payload["state_equations"]:
        if equation["state"] == component:
            equation["rhs"] = f"({factor:.17g}) * ({equation['rhs']})"
            changed = True
    if not changed:
        raise ValueError(f"tagged mechanism component is unavailable: {component}")
    payload["candidate_id"] = _derived_candidate_id(candidate.candidate_id, component)
    return CandidateModel.model_validate(payload)


def _derived_candidate_id(candidate_id: str, component: str) -> str:
    digest = hashlib.sha256(f"{candidate_id}:{component}".encode()).hexdigest()[:12]
    prefix = candidate_id[:40].rstrip("_") or "candidate"
    return f"{prefix}_sensitivity_{digest}"


def _append_endpoint(
    subject: FrozenEvaluationSubject,
    endpoint: HiddenMechanismEndpoint,
) -> FrozenEvaluationSubject:
    if any(
        item.mechanism_id == endpoint.mechanism_id
        for item in subject.hidden_mechanisms
    ):
        raise ValueError(f"hidden endpoint already exists: {endpoint.mechanism_id}")
    payload = subject.model_dump(mode="json")
    payload["private_metrics_opened_after_freeze"] = True
    payload["hidden_mechanisms"] = [
        *(item.model_dump(mode="json") for item in subject.hidden_mechanisms),
        endpoint.model_dump(mode="json"),
    ]
    return FrozenEvaluationSubject.model_validate(payload)


def _outcome(
    subject: FrozenEvaluationSubject,
    contract: PhaseBHiddenSubspaceContract,
    *,
    status: Literal["available", "unrecovered", "not_applicable", "failed"],
    components: tuple[str, ...] = (),
    metric: HiddenSubspaceMetric | None = None,
    error: Exception | None = None,
) -> PhaseBHiddenSubspaceOutcome:
    return PhaseBHiddenSubspaceOutcome(
        subject_id=subject.subject_id,
        candidate_sha256=subject.source_provenance.candidate_sha256,
        contract_sha256=contract.sha256,
        status=status,
        candidate_components=components,
        metric=metric,
        error_type=None if error is None else type(error).__name__,
        error=None if error is None else str(error),
    )


def _validate_inputs(
    subject: FrozenEvaluationSubject,
    training: DatasetSplit,
    test: DatasetSplit,
    contract: PhaseBHiddenSubspaceContract,
) -> None:
    if training.name is not SplitName.TRAIN or test.name is not SplitName.TEST:
        raise ValueError("hidden subspace requires training and test splits")
    if subject.benchmark_id != contract.benchmark_id or subject.tier != contract.tier:
        raise ValueError("hidden contract does not match frozen subject")
    if not subject.private_metrics_opened_after_freeze:
        raise ValueError("target post-freeze evaluation must precede hidden evaluation")
    if subject.target_prediction.status != "available":
        raise ValueError("hidden evaluation requires successful target replay")
    if subject.target_prediction.evaluation_protocol != "unseen_condition_free_rollout":
        raise ValueError("hidden evaluation requires the common free-rollout protocol")
    if set(subject.target_prediction.normalization_scales) != set(
        subject.validation_context.targets
    ):
        raise ValueError("hidden evaluation requires frozen training target scales")


def _expected_sample_count(split: DatasetSplit, targets: tuple[str, ...]) -> int:
    return sum(len(item.time) * len(targets) for item in split.trajectories)


def _public_spec_by_id(benchmark_id: str, *, data_root: Path) -> PhaseBPublicSpec:
    for family, tasks, dynamics, variants in (
        (
            "dalla_man",
            ("T1", "T2", "T3", "T4"),
            ("canonical", "perturbed"),
            ("named", "obfuscated"),
        ),
        ("cstr", (None,), ("canonical",), ("named", "obfuscated")),
        ("alien_device", (None,), ("canonical",), ("functional", "opaque")),
    ):
        for task in tasks:
            for dynamic in dynamics:
                for variant in variants:
                    for tier in ("easy", "hard"):
                        spec = phase_b_public_spec(
                            family,  # type: ignore[arg-type]
                            tier,  # type: ignore[arg-type]
                            variant,  # type: ignore[arg-type]
                            task=task,
                            dynamics=dynamic,  # type: ignore[arg-type]
                            data_root=data_root,
                        )
                        if spec.benchmark_id == benchmark_id:
                            return spec
    raise ValueError(f"unknown Phase-B benchmark identifier: {benchmark_id}")
