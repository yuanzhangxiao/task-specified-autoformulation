"""Typed, non-scalar final evaluation for frozen discovered models."""

from __future__ import annotations

import ast
import math
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.expressions import CandidateValidator, ValidationContext
from autoformalism.expressions.diagnostics import (
    ModelValidationError,
    ValidationDiagnostic,
)
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluation,
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.candidate import ParameterScope

EndpointStatus = Literal["available", "not_applicable", "missing", "failed"]


class RuntimeValidityEndpoint(BaseModel):
    """Public deterministic static-validation outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def failure_list_matches_status(self) -> RuntimeValidityEndpoint:
        """Require invalid candidates to retain at least one reason."""
        if self.valid and self.failures:
            raise ValueError("a valid candidate cannot have runtime failures")
        if not self.valid and not self.failures:
            raise ValueError("an invalid candidate requires a runtime failure")
        return self


def certify_runtime_validity(
    candidate: CandidateModel,
    context: ValidationContext,
) -> RuntimeValidityEndpoint:
    """Run the deterministic public candidate validator and retain diagnostics."""
    try:
        validated = CandidateValidator().validate(candidate, context)
    except ModelValidationError as exc:
        return RuntimeValidityEndpoint(
            valid=False,
            failures=tuple(_diagnostic_text(item) for item in exc.diagnostics),
        )
    return RuntimeValidityEndpoint(
        valid=True,
        warnings=tuple(_diagnostic_text(item) for item in validated.warnings),
    )


class TargetPredictionEndpoint(BaseModel):
    """Private held-out target behavior, opened only after selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EndpointStatus
    evaluation_protocol: Literal[
        "legacy_unspecified", "unseen_condition_free_rollout"
    ] = "legacy_unspecified"
    normalized_mse: float | None = Field(default=None, ge=0.0)
    per_target_normalized_mse: dict[str, float] = Field(default_factory=dict)
    normalization_scales: dict[str, float] = Field(default_factory=dict)
    trajectory_count: int = Field(default=0, ge=0)
    successful_trajectory_count: int = Field(default=0, ge=0)
    failed_trajectories: tuple[str, ...] = ()
    message: str | None = None

    @model_validator(mode="after")
    def values_match_status(self) -> TargetPredictionEndpoint:
        """Prevent absent or failed endpoints from carrying test scores."""
        if self.status == "available" and self.normalized_mse is None:
            raise ValueError("available target prediction requires normalized_mse")
        if self.status != "available" and (
            self.normalized_mse is not None or self.per_target_normalized_mse
        ):
            raise ValueError("unavailable target prediction cannot carry scores")
        if any(value < 0.0 for value in self.per_target_normalized_mse.values()):
            raise ValueError("per-target normalized MSE must be nonnegative")
        if any(value <= 0.0 for value in self.normalization_scales.values()):
            raise ValueError("normalization scales must be positive")
        if self.successful_trajectory_count > self.trajectory_count:
            raise ValueError("successful trajectories cannot exceed requested count")
        if self.evaluation_protocol == "unseen_condition_free_rollout":
            if not self.trajectory_count:
                raise ValueError("free-rollout evaluation requires trajectories")
            if self.status == "available" and (
                self.successful_trajectory_count != self.trajectory_count
                or self.failed_trajectories
            ):
                raise ValueError("available free rollout requires complete coverage")
        return self


class HiddenMechanismEndpoint(BaseModel):
    """Private hidden-mechanism recovery and conditional aligned error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism_id: str
    status: EndpointStatus
    recovered: bool
    aligned_test_nmse: float | None = Field(default=None, ge=0.0)
    message: str | None = None

    @model_validator(mode="after")
    def conditional_error_matches_recovery(self) -> HiddenMechanismEndpoint:
        """Report hidden NMSE only for recovered, successfully evaluated units."""
        if self.status == "available":
            if not self.recovered or self.aligned_test_nmse is None:
                raise ValueError(
                    "available hidden endpoint requires recovery and aligned NMSE"
                )
        elif self.aligned_test_nmse is not None:
            raise ValueError("unavailable hidden endpoint cannot carry aligned NMSE")
        return self


class InterventionEndpoint(BaseModel):
    """One private intervention or distribution-shift evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    status: EndpointStatus
    target_nmse: float | None = Field(default=None, ge=0.0)
    response_direction_correct: bool | None = None
    response_shape_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    peak_timing_error_fraction: float | None = Field(default=None, ge=0.0)
    message: str | None = None

    @model_validator(mode="after")
    def values_match_status(self) -> InterventionEndpoint:
        """Keep failed or absent intervention results score-free."""
        values = (
            self.target_nmse,
            self.response_direction_correct,
            self.response_shape_correlation,
            self.peak_timing_error_fraction,
        )
        if self.status == "available" and self.target_nmse is None:
            raise ValueError("available intervention requires target_nmse")
        if self.status != "available" and any(value is not None for value in values):
            raise ValueError("unavailable intervention cannot carry scores")
        return self


class QualitativeLLMEndpoint(BaseModel):
    """Optional qualitative assessment, kept separate from deterministic metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EndpointStatus
    protocol: str
    requested_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    pass_count: int = Field(default=0, ge=0)
    fail_count: int = Field(default=0, ge=0)
    indeterminate_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> QualitativeLLMEndpoint:
        """Retain provider coverage without inventing missing judgments."""
        if self.successful_calls > self.requested_calls:
            raise ValueError("successful LLM calls cannot exceed requested calls")
        if self.status == "available" and not self.successful_calls:
            raise ValueError("available LLM endpoint requires a successful call")
        return self


class SourceArtifactProvenance(BaseModel):
    """Content-addressed origin of one method-specific frozen model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: Literal[
        "autoformalism_summary",
        "raw_data_agent_run",
        "sindy_result",
        "pysr_result",
        "d3_result",
        "direct_candidate",
    ]
    request_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    auxiliary_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def auxiliary_hashes_are_valid(self) -> SourceArtifactProvenance:
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.auxiliary_sha256.values()
        ):
            raise ValueError("auxiliary artifact hashes must be lowercase SHA-256")
        return self


class FrozenParameterization(BaseModel):
    """Fitted scalar values needed to replay one frozen candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "partial", "not_required", "missing"]
    global_parameters: dict[str, float] = Field(default_factory=dict)
    global_initial_conditions: dict[str, float] = Field(default_factory=dict)
    message: str | None = None

    @model_validator(mode="after")
    def values_are_finite(self) -> FrozenParameterization:
        values = (
            *self.global_parameters.values(),
            *self.global_initial_conditions.values(),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frozen fitted values must be finite")
        if self.status in {"missing", "not_required"} and values:
            raise ValueError(f"{self.status} parameterization cannot carry values")
        if self.status == "partial" and not values:
            raise ValueError("partial parameterization requires at least one value")
        return self


class FrozenEvaluationSubject(BaseModel):
    """Method-independent frozen model plus already-computed private endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["frozen-evaluation-subject-1"] = (
        "frozen-evaluation-subject-1"
    )
    subject_id: str
    method: str
    benchmark_id: str
    tier: str
    repetition: int = Field(ge=0)
    selection_frozen: Literal[True] = True
    private_metrics_opened_after_freeze: bool
    public_mechanism_applicable: bool = True
    source_provenance: SourceArtifactProvenance
    candidate: CandidateModel
    parameterization: FrozenParameterization
    validation_context: ValidationContext
    target_prediction: TargetPredictionEndpoint
    hidden_mechanisms: tuple[HiddenMechanismEndpoint, ...] = ()
    interventions: tuple[InterventionEndpoint, ...] = ()
    qualitative_llm: QualitativeLLMEndpoint | None = None

    @model_validator(mode="after")
    def private_metrics_respect_freeze(self) -> FrozenEvaluationSubject:
        """Reject private results that were opened before the model was frozen."""
        private_available = self.target_prediction.status == "available" or any(
            item.status == "available"
            for item in (*self.hidden_mechanisms, *self.interventions)
        )
        if private_available and not self.private_metrics_opened_after_freeze:
            raise ValueError("private metrics require a post-freeze evaluation")
        if self.target_prediction.status == "available":
            expected_targets = set(self.validation_context.targets)
            actual_targets = set(self.target_prediction.per_target_normalized_mse)
            if actual_targets != expected_targets:
                raise ValueError(
                    "per-target test metrics differ from public targets; "
                    f"missing={sorted(expected_targets - actual_targets)}, "
                    f"extra={sorted(actual_targets - expected_targets)}"
                )
            if (
                self.target_prediction.evaluation_protocol
                == "unseen_condition_free_rollout"
                and set(self.target_prediction.normalization_scales) != expected_targets
            ):
                raise ValueError(
                    "free-rollout normalization scales differ from public targets"
                )
        mechanism_ids = [item.mechanism_id for item in self.hidden_mechanisms]
        if len(mechanism_ids) != len(set(mechanism_ids)):
            raise ValueError("hidden mechanism identifiers must be unique")
        case_ids = [item.case_id for item in self.interventions]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("intervention case identifiers must be unique")
        self._validate_parameterization()
        return self

    def _validate_parameterization(self) -> None:
        expected_parameters = {
            item.name: item
            for item in self.candidate.parameters
            if item.scope is ParameterScope.GLOBAL
        }
        expected_initials = {
            item.state: item
            for item in self.candidate.initial_conditions
            if item.scope is ParameterScope.GLOBAL
            and item.initialization_range is not None
        }
        trajectory_specific = any(
            item.scope is ParameterScope.TRAJECTORY_SPECIFIC
            for item in (
                *self.candidate.parameters,
                *self.candidate.initial_conditions,
            )
        )
        actual_parameters = set(self.parameterization.global_parameters)
        actual_initials = set(self.parameterization.global_initial_conditions)
        extra_parameters = actual_parameters - set(expected_parameters)
        extra_initials = actual_initials - set(expected_initials)
        if extra_parameters or extra_initials:
            raise ValueError(
                "frozen fitted values contain undeclared scalars; "
                f"extra_parameters={sorted(extra_parameters)}, "
                f"extra_initials={sorted(extra_initials)}"
            )
        for name, value in self.parameterization.global_parameters.items():
            bounds = expected_parameters[name].bounds
            if bounds is not None and not bounds.lower <= value <= bounds.upper:
                raise ValueError(f"frozen parameter {name} lies outside bounds")
        for state, value in self.parameterization.global_initial_conditions.items():
            bounds = expected_initials[state].initialization_range
            assert bounds is not None
            if not bounds.lower <= value <= bounds.upper:
                raise ValueError(f"frozen initial condition {state} lies outside range")
        complete = (
            actual_parameters == set(expected_parameters)
            and actual_initials == set(expected_initials)
            and not trajectory_specific
        )
        required = bool(expected_parameters or expected_initials or trajectory_specific)
        status = self.parameterization.status
        if status == "available" and not complete:
            raise ValueError("available parameterization is not replay-complete")
        if status == "not_required" and required:
            raise ValueError("nontrivial candidate cannot use not_required values")
        if status == "not_required" and not complete:
            raise ValueError("not_required parameterization is inconsistent")
        if status == "missing" and (
            actual_parameters or actual_initials or not required
        ):
            raise ValueError("missing parameterization status is inconsistent")
        if status == "partial" and complete:
            raise ValueError("complete fitted values cannot be marked partial")


class ComplexityEndpoint(BaseModel):
    """Deterministic model-size measures after the frozen pruning policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_count: int = Field(ge=0)
    latent_state_count: int = Field(ge=0)
    process_count: int = Field(ge=0)
    parameter_count: int = Field(ge=0)
    additive_term_count: int = Field(ge=0)


class PublicMechanismEndpoint(BaseModel):
    """Public deterministic task-mechanism compliance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "available",
        "not_applicable",
        "missing",
        "invalid_runtime",
    ]
    specification_source: Literal["legacy", "public_prompt"] | None = None
    public_prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluation: MechanismEvaluation | None = None

    @model_validator(mode="after")
    def evaluation_matches_status(self) -> PublicMechanismEndpoint:
        """Make absent public contracts explicit rather than silently scoring zero."""
        if (self.status == "available") != (self.evaluation is not None):
            raise ValueError("public mechanism status and evaluation disagree")
        if self.status != "available" and (
            self.specification_source is not None
            or self.public_prompt_sha256 is not None
        ):
            raise ValueError("unavailable public mechanism cannot carry provenance")
        if self.status == "available" and self.specification_source is None:
            raise ValueError("available public mechanism requires specification source")
        if (
            self.specification_source == "public_prompt"
            and self.public_prompt_sha256 is None
        ):
            raise ValueError("public-prompt mechanism endpoint requires prompt SHA-256")
        return self


class FinalEvaluationRecord(BaseModel):
    """Separate final endpoints for one frozen model; deliberately no total score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-final-evaluation-1"] = "phase-b-final-evaluation-1"
    subject_id: str
    method: str
    benchmark_id: str
    tier: str
    repetition: int
    source_provenance: SourceArtifactProvenance
    parameterization: FrozenParameterization
    runtime: RuntimeValidityEndpoint
    public_mechanism: PublicMechanismEndpoint
    target_prediction: TargetPredictionEndpoint
    hidden_mechanisms: tuple[HiddenMechanismEndpoint, ...]
    interventions: tuple[InterventionEndpoint, ...]
    complexity: ComplexityEndpoint
    qualitative_llm: QualitativeLLMEndpoint | None


def evaluate_frozen_subject(
    subject: FrozenEvaluationSubject,
    mechanism_spec: MechanismEvaluationSpec | None,
    *,
    mechanism_spec_not_applicable: bool = False,
) -> FinalEvaluationRecord:
    """Compute public compliance and complexity while retaining private endpoints."""
    runtime = certify_runtime_validity(
        subject.candidate,
        subject.validation_context,
    )
    if not runtime.valid:
        if subject.target_prediction.status == "available" or any(
            endpoint.status == "available"
            for endpoint in (*subject.hidden_mechanisms, *subject.interventions)
        ):
            raise ValueError(
                "runtime-invalid candidate cannot carry available private metrics"
            )
        public = PublicMechanismEndpoint(status="invalid_runtime")
    elif mechanism_spec is not None:
        if mechanism_spec_not_applicable:
            raise ValueError("not-applicable subject cannot have a mechanism spec")
        if (mechanism_spec.benchmark_id, mechanism_spec.tier) != (
            subject.benchmark_id,
            subject.tier,
        ):
            raise ValueError("mechanism specification does not match subject")
        public = PublicMechanismEndpoint(
            status="available",
            specification_source=mechanism_spec.source,
            public_prompt_sha256=mechanism_spec.public_prompt_sha256,
            evaluation=evaluate_mechanisms(subject.candidate, mechanism_spec),
        )
    else:
        public = PublicMechanismEndpoint(
            status="not_applicable" if mechanism_spec_not_applicable else "missing"
        )
    candidate = subject.candidate
    complexity = ComplexityEndpoint(
        state_count=len(candidate.states),
        latent_state_count=sum(
            state.kind.value == "latent" for state in candidate.states
        ),
        process_count=len(candidate.processes),
        parameter_count=len(candidate.parameters),
        additive_term_count=sum(
            _additive_term_count(equation.rhs) for equation in candidate.state_equations
        ),
    )
    return FinalEvaluationRecord(
        subject_id=subject.subject_id,
        method=subject.method,
        benchmark_id=subject.benchmark_id,
        tier=subject.tier,
        repetition=subject.repetition,
        source_provenance=subject.source_provenance,
        parameterization=subject.parameterization,
        runtime=runtime,
        public_mechanism=public,
        target_prediction=subject.target_prediction,
        hidden_mechanisms=subject.hidden_mechanisms,
        interventions=subject.interventions,
        complexity=complexity,
        qualitative_llm=subject.qualitative_llm,
    )


def evaluation_summary(records: tuple[FinalEvaluationRecord, ...]) -> dict[str, object]:
    """Return endpoint-wise coverage and means without constructing a total score."""
    target = [
        item.target_prediction.normalized_mse
        for item in records
        if item.target_prediction.status == "available"
        and item.target_prediction.normalized_mse is not None
    ]
    graph_compliance = [
        item.public_mechanism.evaluation.graph_mechanism_compliance
        for item in records
        if item.public_mechanism.evaluation is not None
    ]
    graph_complete = [
        item.public_mechanism.evaluation.graph_mechanism_compliance_complete
        for item in records
        if item.public_mechanism.evaluation is not None
    ]
    annotation_compliance = [
        item.public_mechanism.evaluation.mechanism_annotation_compliance
        for item in records
        if item.public_mechanism.evaluation is not None
    ]
    annotation_complete = [
        item.public_mechanism.evaluation.mechanism_annotation_compliance_complete
        for item in records
        if item.public_mechanism.evaluation is not None
    ]
    hidden = [
        endpoint.aligned_test_nmse
        for item in records
        for endpoint in item.hidden_mechanisms
        if endpoint.status == "available" and endpoint.aligned_test_nmse is not None
    ]
    hidden_required = [
        endpoint for item in records for endpoint in item.hidden_mechanisms
    ]
    interventions = [
        endpoint.target_nmse
        for item in records
        for endpoint in item.interventions
        if endpoint.status == "available" and endpoint.target_nmse is not None
    ]
    llm_requested = sum(
        item.qualitative_llm.requested_calls
        for item in records
        if item.qualitative_llm is not None
    )
    llm_successful = sum(
        item.qualitative_llm.successful_calls
        for item in records
        if item.qualitative_llm is not None
    )
    target_trajectory_count = sum(
        item.target_prediction.trajectory_count for item in records
    )
    successful_target_trajectories = sum(
        item.target_prediction.successful_trajectory_count for item in records
    )
    return {
        "schema_version": "phase-b-final-evaluation-summary-1",
        "record_count": len(records),
        "runtime_valid_rate": _rate(
            sum(item.runtime.valid for item in records), records
        ),
        "replay_complete_rate": _rate(
            sum(
                item.parameterization.status in {"available", "not_required"}
                for item in records
            ),
            records,
        ),
        "parameterization_status_counts": {
            status: sum(item.parameterization.status == status for item in records)
            for status in ("available", "partial", "not_required", "missing")
        },
        "target_prediction_coverage": _rate(len(target), records),
        "target_trajectory_success_rate": (
            successful_target_trajectories / target_trajectory_count
            if target_trajectory_count
            else None
        ),
        "mean_target_test_nmse": _mean(target),
        "public_mechanism_coverage": _rate(len(graph_compliance), records),
        "public_graph_mechanism_coverage": _rate(len(graph_compliance), records),
        "mean_public_graph_mechanism_compliance": _mean(graph_compliance),
        "public_graph_mechanism_complete_assessment_rate": _mean(
            [float(value) for value in graph_complete]
        ),
        "public_mechanism_annotation_coverage": _rate(
            len(annotation_compliance), records
        ),
        "mean_public_mechanism_annotation_compliance": _mean(
            annotation_compliance
        ),
        "public_mechanism_annotation_complete_assessment_rate": _mean(
            [float(value) for value in annotation_complete]
        ),
        # Backward-compatible aggregate name: graph compliance is the primary
        # scientific endpoint, not a blend with annotation quality.
        "mean_public_mechanism_compliance": _mean(graph_compliance),
        "hidden_mechanism_recovery_rate": (
            sum(item.recovered for item in hidden_required) / len(hidden_required)
            if hidden_required
            else None
        ),
        "mean_hidden_nmse_conditional_on_recovery": _mean(hidden),
        "intervention_endpoint_count": sum(len(item.interventions) for item in records),
        "mean_intervention_target_nmse": _mean(interventions),
        "qualitative_llm_requested_calls": llm_requested,
        "qualitative_llm_successful_calls": llm_successful,
        "qualitative_llm_response_rate": (
            llm_successful / llm_requested if llm_requested else None
        ),
    }


def _additive_term_count(source: str) -> int:
    node = ast.parse(source, mode="eval").body
    return _count_additive_node(node)


def _count_additive_node(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _count_additive_node(node.left) + _count_additive_node(node.right)
    return 1


def _rate(numerator: int, records: tuple[FinalEvaluationRecord, ...]) -> float | None:
    return numerator / len(records) if records else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _diagnostic_text(diagnostic: ValidationDiagnostic) -> str:
    return f"{diagnostic.code} at {diagnostic.location}: {diagnostic.message}"
