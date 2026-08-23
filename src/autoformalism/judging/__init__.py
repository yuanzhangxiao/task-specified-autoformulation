"""Scientific-judge fact extraction and deterministic aggregation."""

from autoformalism.judging.hybrid import (
    ATOMIC_EVIDENCE_SCHEMA_VERSION,
    STRUCTURAL_FACTS_SCHEMA_VERSION,
    AtomicEvidencePlan,
    HybridScoringConfig,
    atomic_candidate_context,
    atomic_findings_payload,
    atomic_role_compatibility_assessments,
    build_atomic_evidence_plan,
    build_group_registry,
    candidate_claims,
    deterministic_pair_assessments,
    extract_public_requirements,
    merge_atomic_assessments,
    score_hybrid_pair,
    semantic_absolute_units,
    structural_facts,
)

__all__ = [
    "ATOMIC_EVIDENCE_SCHEMA_VERSION",
    "STRUCTURAL_FACTS_SCHEMA_VERSION",
    "AtomicEvidencePlan",
    "HybridScoringConfig",
    "atomic_candidate_context",
    "atomic_findings_payload",
    "atomic_role_compatibility_assessments",
    "build_atomic_evidence_plan",
    "build_group_registry",
    "candidate_claims",
    "deterministic_pair_assessments",
    "extract_public_requirements",
    "merge_atomic_assessments",
    "score_hybrid_pair",
    "semantic_absolute_units",
    "structural_facts",
]
