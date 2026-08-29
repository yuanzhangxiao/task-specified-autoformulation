"""Post-selection analyses used by the rebuttal experiments."""

from autoformalism.rebuttal.artifacts import CandidateArtifact, index_artifacts
from autoformalism.rebuttal.benchmark_audit import (
    ExcitationMetrics,
    ResponsePhaseMetrics,
    ShortcutMetrics,
    audit_excitation,
    audit_response_phases,
    audit_shortcuts,
    downsample_split,
)
from autoformalism.rebuttal.final_evaluation import (
    FinalEvaluationRecord,
    FrozenEvaluationSubject,
    FrozenParameterization,
    HiddenMechanismEndpoint,
    InterventionEndpoint,
    QualitativeLLMEndpoint,
    RuntimeValidityEndpoint,
    SourceArtifactProvenance,
    TargetPredictionEndpoint,
    certify_runtime_validity,
    evaluate_frozen_subject,
    evaluation_summary,
)
from autoformalism.rebuttal.hidden import (
    HiddenSubspaceMetric,
    hidden_subspace_nmse,
)
from autoformalism.rebuttal.intervention_evaluation import (
    FrozenModel,
    InterventionEvaluation,
    evaluate_frozen_model,
    load_frozen_model,
)
from autoformalism.rebuttal.interventions import (
    InterventionCase,
    InterventionSuite,
    ReferenceTrajectory,
    load_intervention_suite,
    simulate_reference,
)
from autoformalism.rebuttal.llm_assets import (
    LLMCacheAudit,
    LLMCacheRecord,
    LLMCacheResolution,
    audit_llm_caches,
    resolve_llm_caches,
)
from autoformalism.rebuttal.mechanisms import (
    MechanismComplianceResult,
    MechanismEvaluation,
    MechanismEvaluationSpec,
    evaluate_mechanisms,
    mechanism_claim_components,
)
from autoformalism.rebuttal.objectives import (
    FrozenSelectionResult,
    ObjectiveComparison,
    compare_ratio_and_weighted_sum,
    select_frozen_candidate,
)
from autoformalism.rebuttal.observability import (
    ObservabilityResult,
    ParameterSensitivityResult,
    empirical_dalla_observability,
    empirical_dalla_parameter_sensitivity,
)
from autoformalism.rebuttal.postfreeze_evaluation import (
    PostFreezeEvaluationOutcome,
    evaluate_subject_on_test,
    outcome_for_subject,
)
from autoformalism.rebuttal.statistics import (
    PairedLogComparison,
    holm_adjust,
    paired_log_comparison,
    wilson_interval,
)

__all__ = [
    "CandidateArtifact",
    "ExcitationMetrics",
    "FinalEvaluationRecord",
    "FrozenEvaluationSubject",
    "FrozenModel",
    "FrozenParameterization",
    "FrozenSelectionResult",
    "HiddenMechanismEndpoint",
    "HiddenSubspaceMetric",
    "InterventionCase",
    "InterventionEndpoint",
    "InterventionEvaluation",
    "InterventionSuite",
    "LLMCacheAudit",
    "LLMCacheRecord",
    "LLMCacheResolution",
    "MechanismComplianceResult",
    "MechanismEvaluation",
    "MechanismEvaluationSpec",
    "ObjectiveComparison",
    "ObservabilityResult",
    "PairedLogComparison",
    "ParameterSensitivityResult",
    "PostFreezeEvaluationOutcome",
    "QualitativeLLMEndpoint",
    "ReferenceTrajectory",
    "ResponsePhaseMetrics",
    "RuntimeValidityEndpoint",
    "ShortcutMetrics",
    "SourceArtifactProvenance",
    "TargetPredictionEndpoint",
    "audit_excitation",
    "audit_llm_caches",
    "audit_response_phases",
    "audit_shortcuts",
    "certify_runtime_validity",
    "compare_ratio_and_weighted_sum",
    "downsample_split",
    "empirical_dalla_observability",
    "empirical_dalla_parameter_sensitivity",
    "evaluate_frozen_model",
    "evaluate_frozen_subject",
    "evaluate_mechanisms",
    "evaluate_subject_on_test",
    "evaluation_summary",
    "hidden_subspace_nmse",
    "holm_adjust",
    "index_artifacts",
    "load_frozen_model",
    "load_intervention_suite",
    "mechanism_claim_components",
    "outcome_for_subject",
    "paired_log_comparison",
    "resolve_llm_caches",
    "select_frozen_candidate",
    "simulate_reference",
    "wilson_interval",
]
