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
    MechanismEvaluation,
    MechanismEvaluationSpec,
    evaluate_mechanisms,
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
from autoformalism.rebuttal.statistics import (
    PairedLogComparison,
    holm_adjust,
    paired_log_comparison,
    wilson_interval,
)

__all__ = [
    "CandidateArtifact",
    "ExcitationMetrics",
    "FrozenModel",
    "FrozenSelectionResult",
    "InterventionCase",
    "InterventionEvaluation",
    "InterventionSuite",
    "LLMCacheAudit",
    "LLMCacheRecord",
    "LLMCacheResolution",
    "MechanismEvaluation",
    "MechanismEvaluationSpec",
    "ObjectiveComparison",
    "ObservabilityResult",
    "PairedLogComparison",
    "ParameterSensitivityResult",
    "ReferenceTrajectory",
    "ResponsePhaseMetrics",
    "ShortcutMetrics",
    "audit_excitation",
    "audit_llm_caches",
    "audit_response_phases",
    "audit_shortcuts",
    "compare_ratio_and_weighted_sum",
    "downsample_split",
    "empirical_dalla_observability",
    "empirical_dalla_parameter_sensitivity",
    "evaluate_frozen_model",
    "evaluate_mechanisms",
    "holm_adjust",
    "index_artifacts",
    "load_frozen_model",
    "load_intervention_suite",
    "paired_log_comparison",
    "resolve_llm_caches",
    "select_frozen_candidate",
    "simulate_reference",
    "wilson_interval",
]
