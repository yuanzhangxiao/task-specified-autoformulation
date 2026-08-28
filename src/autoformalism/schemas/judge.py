"""Strict structured judge-result schema."""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from autoformalism.schemas.base import (
    Identifier,
    NonEmptyText,
    StrictSchema,
    UnitInterval,
)


class ActionPriority(str, Enum):
    """Priority assigned to a requested candidate edit."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class HardRedFlag(StrictSchema):
    """Blocking or severe specification violation."""

    code: Identifier
    description: NonEmptyText = "unspecified"
    evidence: NonEmptyText


class ActionableEdit(StrictSchema):
    """Concrete edit the proposer can apply in a later milestone."""

    target: Identifier
    instruction: NonEmptyText
    priority: ActionPriority


class CategoryScore(StrictSchema):
    """Numeric category assessment with an optional concise justification."""

    score: UnitInterval
    justification: NonEmptyText = "unspecified"


class CategoryScores(StrictSchema):
    """Fixed task-compliance rubric shared by every benchmark judge prompt.

    A fixed object is intentional: OpenAI Structured Outputs rejects the
    ``propertyNames`` keyword that Pydantic emits for identifier-constrained
    dictionary keys.  Explicit fields also prevent providers from inventing,
    omitting, or renaming scoring categories.
    """

    task_output_coverage: CategoryScore
    mechanism_state_adequacy: CategoryScore
    mathematical_completeness: CategoryScore
    data_causal_consistency: CategoryScore
    constraint_compliance: CategoryScore
    parsimony_interpretability: CategoryScore

    def __getitem__(self, name: str) -> CategoryScore:
        """Provide read-only mapping-style access for existing consumers."""
        value = getattr(self, name)
        if not isinstance(value, CategoryScore):  # pragma: no cover - schema invariant
            raise KeyError(name)
        return value


class JudgeResult(StrictSchema):
    """Structured task-compliance assessment independent of fit metrics."""

    schema_version: Literal["1"] = "1"
    hard_red_flags: tuple[HardRedFlag, ...] = Field(default=(), max_length=64)
    category_scores: CategoryScores
    aggregate_score: UnitInterval
    missing_requirements: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=128
    )
    actionable_edits: tuple[ActionableEdit, ...] = Field(
        default=(), max_length=128
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_numeric_scores(cls, value: object) -> object:
        """Accept cached/mock V1 numeric scores while emitting the richer schema."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        scores = payload.get("category_scores")
        if isinstance(scores, dict):
            payload["category_scores"] = {
                name: ({"score": score} if isinstance(score, (int, float)) else score)
                for name, score in scores.items()
            }
        return payload

    @property
    def numeric_category_scores(self) -> dict[str, float]:
        """Return the score-only mapping used by ranking and proposer feedback."""
        return {
            name: item.score
            for name, item in self.category_scores.__dict__.items()
        }


class ScientificCategoryScores(StrictSchema):
    """Prospective v2 categories restricted to scientific semantics."""

    mechanistic_coherence: CategoryScore
    source_sink_balance_semantics: CategoryScore
    dynamic_plausibility: CategoryScore
    mechanism_coupling_task_sufficiency: CategoryScore
    nonredundancy_accounting: CategoryScore
    latent_state_complexity_justification: CategoryScore


SCIENTIFIC_CATEGORY_WEIGHTS: dict[str, float] = {
    "mechanistic_coherence": 0.20,
    "source_sink_balance_semantics": 0.20,
    "dynamic_plausibility": 0.20,
    "mechanism_coupling_task_sufficiency": 0.20,
    "nonredundancy_accounting": 0.10,
    "latent_state_complexity_justification": 0.10,
}


class ScientificJudgeResult(StrictSchema):
    """Scientific-only v2 judge assessment independent of fit metrics."""

    schema_version: Literal["2"] = "2"
    hard_red_flags: tuple[HardRedFlag, ...] = Field(default=(), max_length=64)
    category_scores: ScientificCategoryScores
    missing_requirements: tuple[NonEmptyText, ...] = Field(
        default=(), max_length=128
    )
    actionable_edits: tuple[ActionableEdit, ...] = Field(
        default=(), max_length=128
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_numeric_scores(cls, value: object) -> object:
        """Accept concise numeric scores in trusted mocks and fixtures."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        scores = payload.get("category_scores")
        if isinstance(scores, dict):
            payload["category_scores"] = {
                name: ({"score": score} if isinstance(score, (int, float)) else score)
                for name, score in scores.items()
            }
        return payload

    @property
    def aggregate_score(self) -> float:
        """Return the runtime-owned weighted aggregate of category scores."""
        return sum(
            SCIENTIFIC_CATEGORY_WEIGHTS[name] * score
            for name, score in self.numeric_category_scores.items()
        )

    @property
    def numeric_category_scores(self) -> dict[str, float]:
        """Return the score-only mapping used by search feedback."""
        return {
            name: item.score
            for name, item in self.category_scores.__dict__.items()
        }


class ComparativeVerdict(str, Enum):
    """One blinded comparative answer with explicit abstention options."""

    CANDIDATE_A = "candidate_a"
    CANDIDATE_B = "candidate_b"
    TIE = "tie"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class ComparativeAnswer(StrictSchema):
    """Verdict and candidate-grounded evidence for one atomic question."""

    verdict: ComparativeVerdict
    evidence: NonEmptyText


class AtomicComparativeAnswers(StrictSchema):
    """Fixed atomic scientific questions used only by judge calibration."""

    claimed_mechanisms_represented: ComparativeAnswer
    task_inputs_connected_to_targets: ComparativeAnswer
    claimed_processes_connected_to_balances: ComparativeAnswer
    source_terms_have_consistent_signs: ComparativeAnswer
    sink_terms_have_consistent_signs: ComparativeAnswer
    fluxes_not_duplicated: ComparativeAnswer
    components_not_disconnected: ComparativeAnswer
    mechanisms_not_conflicting: ComparativeAnswer
    latent_states_have_incoming_pathways: ComparativeAnswer
    latent_states_have_outgoing_influence: ComparativeAnswer
    latent_accumulators_have_relaxation_or_justification: ComparativeAnswer
    claimed_decay_opposes_accumulated_quantity: ComparativeAnswer
    claimed_delay_has_drive_and_relaxation: ComparativeAnswer
    claimed_saturation_is_structurally_bounded: ComparativeAnswer


class ComparativeJudgeResult(StrictSchema):
    """Blinded pairwise scientific assessment for calibration experiments."""

    schema_version: Literal["comparative-1"] = "comparative-1"
    answers: AtomicComparativeAnswers

    @property
    def numeric_preference(self) -> float | None:
        """Average atomic A preference, excluding indeterminate answers."""
        values = [
            {
                ComparativeVerdict.CANDIDATE_A: 1.0,
                ComparativeVerdict.CANDIDATE_B: 0.0,
                ComparativeVerdict.TIE: 0.5,
            }.get(answer.verdict)
            for answer in self.answers.__dict__.values()
        ]
        determined = [value for value in values if value is not None]
        return sum(determined) / len(determined) if determined else None

    @property
    def indeterminate_rate(self) -> float:
        """Fraction of atomic questions on which the judge abstained."""
        answers = tuple(self.answers.__dict__.values())
        return sum(
            answer.verdict is ComparativeVerdict.INDETERMINATE
            for answer in answers
        ) / len(answers)

    @property
    def not_applicable_rate(self) -> float:
        """Fraction of atomic questions irrelevant to both candidates."""
        answers = tuple(self.answers.__dict__.values())
        return sum(
            answer.verdict is ComparativeVerdict.NOT_APPLICABLE
            for answer in answers
        ) / len(answers)


class AbsoluteVerdict(str, Enum):
    """Candidate-specific truth value for one atomic scientific predicate."""

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class AbsoluteCriterion(str, Enum):
    """Stable atomic predicates used by the hybrid calibration protocol."""

    REQUIRED_MECHANISM_REPRESENTED = "required_mechanism_represented"
    REQUIRED_MECHANISM_CONNECTED = "required_mechanism_connected"
    TASK_INPUTS_REACH_TARGETS = "task_inputs_reach_targets"
    CLAIMED_COMPONENTS_REACH_TARGETS = "claimed_components_reach_targets"
    SOURCE_ROLES_CONSISTENT = "source_roles_consistent"
    SINK_ROLES_CONSISTENT = "sink_roles_consistent"
    SEMANTIC_FLUXES_NOT_DUPLICATED = "semantic_fluxes_not_duplicated"
    MECHANISM_CLAIMS_NOT_CONFLICTING = "mechanism_claims_not_conflicting"
    LATENT_STATES_HAVE_INCOMING_PATHWAYS = (
        "latent_states_have_incoming_pathways"
    )
    LATENT_STATES_REACH_TARGETS = "latent_states_reach_targets"
    LATENT_ACCUMULATORS_JUSTIFIED = "latent_accumulators_justified"
    CLAIMED_DELAYS_MEANINGFUL = "claimed_delays_meaningful"
    CLAIMED_SATURATIONS_APPROPRIATE = "claimed_saturations_appropriate"
    PROPOSER_CLAIMS_SUPPORTED = "proposer_claims_supported"
    TARGET_MAPPING_SEMANTICALLY_CONSISTENT = (
        "target_mapping_semantically_consistent"
    )
    INITIALIZATION_SEMANTICALLY_CONSISTENT = (
        "initialization_semantically_consistent"
    )


class CandidateAbsoluteAssessment(StrictSchema):
    """One absolute verdict with candidate-grounded evidence."""

    verdict: AbsoluteVerdict
    evidence: NonEmptyText


class PairedAbsoluteAssessment(StrictSchema):
    """The same absolute predicate evaluated independently for A and B."""

    criterion: AbsoluteCriterion
    subject_id: Identifier
    candidate_a: CandidateAbsoluteAssessment
    candidate_b: CandidateAbsoluteAssessment


class RelativeCriterion(str, Enum):
    """Irreducibly comparative scientific criteria."""

    PARSIMONY_WHILE_TASK_SUFFICIENT = "parsimony_while_task_sufficient"
    FEWER_UNSUPPORTED_ASSUMPTIONS = "fewer_unsupported_assumptions"
    MECHANISTIC_INTERPRETABILITY = "mechanistic_interpretability"


class RelativeVerdict(str, Enum):
    """Blinded pairwise verdict without an absolute truth interpretation."""

    CANDIDATE_A = "candidate_a"
    CANDIDATE_B = "candidate_b"
    TIE = "tie"
    INDETERMINATE = "indeterminate"


class RelativeAssessment(StrictSchema):
    """One direct comparative answer retained separately from absolute facts."""

    criterion: RelativeCriterion
    verdict: RelativeVerdict
    evidence: NonEmptyText


class ExpectedContributionDirection(str, Enum):
    """Scientific direction inferred without seeing an occurrence's outer sign."""

    POSITIVE = "positive_contribution"
    NEGATIVE = "negative_contribution"
    CONTEXT_DEPENDENT = "context_dependent"
    INSUFFICIENT_PUBLIC_INFORMATION = "insufficient_public_information"


class AtomicSignedOccurrenceAssessment(StrictSchema):
    """Expected scientific direction for one sign-blinded additive occurrence."""

    occurrence_id: Identifier
    expected_direction: ExpectedContributionDirection
    evidence: NonEmptyText


class RepeatedContributionRelation(str, Enum):
    """Scientific relationship between two certified repeated expressions."""

    SAME_PHYSICAL_CONTRIBUTION = "same_physical_contribution"
    DISTINCT_CONTRIBUTIONS = "distinct_contributions"
    INSUFFICIENT_PUBLIC_INFORMATION = "insufficient_public_information"


class AtomicRepeatedContributionAssessment(StrictSchema):
    """Scientific interpretation of one exact-repeat pair."""

    repeat_pair_id: Identifier
    relation: RepeatedContributionRelation
    evidence: NonEmptyText


class AtomicJudgeResult(StrictSchema):
    """Sign-blinded occurrence judgments made before pairwise comparison."""

    schema_version: Literal["atomic-judge-1"] = "atomic-judge-1"
    signed_occurrence_assessments: tuple[
        AtomicSignedOccurrenceAssessment, ...
    ] = Field(max_length=256)
    repeated_contribution_assessments: tuple[
        AtomicRepeatedContributionAssessment, ...
    ] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_identifiers(self) -> AtomicJudgeResult:
        """Reject repeated occurrence or repeat-pair identifiers."""
        occurrence_ids = [
            item.occurrence_id for item in self.signed_occurrence_assessments
        ]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("signed occurrence identifiers must be unique")
        repeat_ids = [
            item.repeat_pair_id
            for item in self.repeated_contribution_assessments
        ]
        if len(repeat_ids) != len(set(repeat_ids)):
            raise ValueError("repeated contribution identifiers must be unique")
        return self

    def validate_expected_units(
        self,
        *,
        occurrence_ids: set[str],
        repeat_pair_ids: set[str],
    ) -> None:
        """Require exactly the runtime-requested sign-blinded atomic units."""
        actual_occurrences = {
            item.occurrence_id for item in self.signed_occurrence_assessments
        }
        actual_repeats = {
            item.repeat_pair_id
            for item in self.repeated_contribution_assessments
        }
        if actual_occurrences != occurrence_ids or actual_repeats != repeat_pair_ids:
            raise ValueError(
                "atomic assessment units differ; "
                f"missing_occurrences={sorted(occurrence_ids - actual_occurrences)}, "
                f"extra_occurrences={sorted(actual_occurrences - occurrence_ids)}, "
                f"missing_repeats={sorted(repeat_pair_ids - actual_repeats)}, "
                f"extra_repeats={sorted(actual_repeats - repeat_pair_ids)}"
            )

    def fill_missing_units_with_insufficient_information(
        self,
        *,
        occurrence_ids: set[str],
        repeat_pair_ids: set[str],
    ) -> tuple[AtomicJudgeResult, tuple[str, ...], tuple[str, ...]]:
        """Fill only missing runtime-owned units with neutral assessments.

        Extra or otherwise unexpected identifiers are never repaired because
        they can indicate that the provider answered a different question.
        """
        actual_occurrences = {
            item.occurrence_id for item in self.signed_occurrence_assessments
        }
        actual_repeats = {
            item.repeat_pair_id
            for item in self.repeated_contribution_assessments
        }
        if actual_occurrences - occurrence_ids or actual_repeats - repeat_pair_ids:
            return self, (), ()
        missing_occurrences = tuple(sorted(occurrence_ids - actual_occurrences))
        missing_repeats = tuple(sorted(repeat_pair_ids - actual_repeats))
        if not missing_occurrences and not missing_repeats:
            return self, (), ()
        repaired = AtomicJudgeResult(
            signed_occurrence_assessments=(
                *self.signed_occurrence_assessments,
                *(
                    AtomicSignedOccurrenceAssessment(
                        occurrence_id=occurrence_id,
                        expected_direction=(
                            ExpectedContributionDirection.INSUFFICIENT_PUBLIC_INFORMATION
                        ),
                        evidence=(
                            "Provider omitted this runtime-requested unit after "
                            "bounded retries; retained as insufficient public "
                            "information without inferring a scientific answer."
                        ),
                    )
                    for occurrence_id in missing_occurrences
                ),
            ),
            repeated_contribution_assessments=(
                *self.repeated_contribution_assessments,
                *(
                    AtomicRepeatedContributionAssessment(
                        repeat_pair_id=repeat_pair_id,
                        relation=(
                            RepeatedContributionRelation.INSUFFICIENT_PUBLIC_INFORMATION
                        ),
                        evidence=(
                            "Provider omitted this runtime-requested unit after "
                            "bounded retries; retained as insufficient public "
                            "information without inferring a scientific answer."
                        ),
                    )
                    for repeat_pair_id in missing_repeats
                ),
            ),
        )
        repaired.validate_expected_units(
            occurrence_ids=occurrence_ids,
            repeat_pair_ids=repeat_pair_ids,
        )
        return repaired, missing_occurrences, missing_repeats


class TargetCompletenessAssessment(StrictSchema):
    """One candidate-specific verdict for one public target channel."""

    target_id: Identifier
    verdict: AbsoluteVerdict
    evidence: NonEmptyText


class TargetCompletenessJudgeResult(StrictSchema):
    """Absolute target-completeness answers for exactly one candidate.

    This contract intentionally contains no second candidate, comparative
    verdict, score, or overall winner. The runtime owns the requested target
    identifiers and validates that the provider answered each exactly once.
    """

    schema_version: Literal["target-completeness-judge-1"] = (
        "target-completeness-judge-1"
    )
    target_assessments: tuple[TargetCompletenessAssessment, ...] = Field(
        max_length=64
    )

    @model_validator(mode="after")
    def unique_target_identifiers(self) -> TargetCompletenessJudgeResult:
        """Reject repeated target identifiers."""
        target_ids = [item.target_id for item in self.target_assessments]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target completeness identifiers must be unique")
        return self

    def validate_expected_targets(self, target_ids: set[str]) -> None:
        """Require exactly the runtime-requested public target identifiers."""
        actual = {item.target_id for item in self.target_assessments}
        if actual != target_ids:
            raise ValueError(
                "target completeness units differ; "
                f"missing_targets={sorted(target_ids - actual)}, "
                f"extra_targets={sorted(actual - target_ids)}"
            )

    @property
    def overall_verdict(self) -> AbsoluteVerdict:
        """Return the conjunctive verdict over all requested targets."""
        verdicts = {item.verdict for item in self.target_assessments}
        if AbsoluteVerdict.FAIL in verdicts:
            return AbsoluteVerdict.FAIL
        if verdicts == {AbsoluteVerdict.PASS}:
            return AbsoluteVerdict.PASS
        return AbsoluteVerdict.INDETERMINATE


class HybridJudgeResult(StrictSchema):
    """Paired absolute assessments plus a separate comparative residual."""

    schema_version: Literal["hybrid-1"] = "hybrid-1"
    absolute_assessments: tuple[PairedAbsoluteAssessment, ...] = Field(
        min_length=1, max_length=256
    )
    comparative_assessments: tuple[RelativeAssessment, ...] = Field(
        min_length=len(RelativeCriterion), max_length=len(RelativeCriterion)
    )

    @model_validator(mode="after")
    def unique_and_complete(self) -> HybridJudgeResult:
        """Reject duplicate atomic units and missing comparative criteria."""
        absolute_keys = [
            (item.criterion, item.subject_id)
            for item in self.absolute_assessments
        ]
        if len(absolute_keys) != len(set(absolute_keys)):
            raise ValueError("absolute assessment keys must be unique")
        relative = [item.criterion for item in self.comparative_assessments]
        if len(relative) != len(set(relative)):
            raise ValueError("comparative criteria must be unique")
        if set(relative) != set(RelativeCriterion):
            raise ValueError("every comparative criterion is required")
        return self

    def validate_expected_absolute_units(
        self,
        expected: set[tuple[AbsoluteCriterion, str]],
    ) -> None:
        """Require the provider to return exactly the runtime-requested units."""
        actual = {
            (item.criterion, item.subject_id)
            for item in self.absolute_assessments
        }
        if actual != expected:
            missing = sorted(
                f"{criterion.value}:{subject}"
                for criterion, subject in expected - actual
            )
            extra = sorted(
                f"{criterion.value}:{subject}"
                for criterion, subject in actual - expected
            )
            raise ValueError(
                f"absolute assessment units differ; missing={missing}, extra={extra}"
            )

    def discard_redundant_absolute_units(
        self,
        *,
        expected: set[tuple[AbsoluteCriterion, str]],
        redundant: set[tuple[AbsoluteCriterion, str]],
    ) -> tuple[HybridJudgeResult, tuple[tuple[AbsoluteCriterion, str], ...]]:
        """Remove only a complete, explicitly whitelisted set of extra units."""
        actual = {
            (item.criterion, item.subject_id)
            for item in self.absolute_assessments
        }
        missing = expected - actual
        extra = actual - expected
        if missing or not redundant or extra != redundant:
            return self, ()
        retained = tuple(
            item
            for item in self.absolute_assessments
            if (item.criterion, item.subject_id) not in extra
        )
        payload = self.model_dump(mode="json")
        payload["absolute_assessments"] = [
            item.model_dump(mode="json") for item in retained
        ]
        normalized = HybridJudgeResult.model_validate(payload)
        normalized.validate_expected_absolute_units(expected)
        removed = tuple(
            sorted(extra, key=lambda item: (item[0].value, item[1]))
        )
        return normalized, removed

    @property
    def numeric_relative_preference(self) -> float | None:
        """Return mean A preference for determined comparative assessments."""
        values = [
            {
                RelativeVerdict.CANDIDATE_A: 1.0,
                RelativeVerdict.CANDIDATE_B: 0.0,
                RelativeVerdict.TIE: 0.5,
            }.get(item.verdict)
            for item in self.comparative_assessments
        ]
        determined = [value for value in values if value is not None]
        return sum(determined) / len(determined) if determined else None

    @property
    def numeric_relative_preference_fixed_denominator(self) -> float:
        """Return mean A preference with indeterminate answers scored neutrally.

        Every schema-required comparative question contributes to the
        denominator. Candidate-A, candidate-B, tie, and indeterminate verdicts
        contribute 1, 0, 0.5, and 0.5 respectively. In signed preference space
        this is equivalent to averaging +1, -1, 0, and 0 over the fixed
        question set.
        """
        values = [
            {
                RelativeVerdict.CANDIDATE_A: 1.0,
                RelativeVerdict.CANDIDATE_B: 0.0,
                RelativeVerdict.TIE: 0.5,
                RelativeVerdict.INDETERMINATE: 0.5,
            }[item.verdict]
            for item in self.comparative_assessments
        ]
        return sum(values) / len(values)


JudgeAssessment: TypeAlias = JudgeResult | ScientificJudgeResult


def parse_judge_assessment(value: object) -> JudgeAssessment:
    """Load a historical v1 or prospective v2 judge checkpoint."""
    if isinstance(value, ScientificJudgeResult | JudgeResult):
        return value
    if isinstance(value, dict) and value.get("schema_version") == "2":
        return ScientificJudgeResult.model_validate(value)
    return JudgeResult.model_validate(value)
