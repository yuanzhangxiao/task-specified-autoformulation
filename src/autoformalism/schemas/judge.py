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


JudgeAssessment: TypeAlias = JudgeResult | ScientificJudgeResult


def parse_judge_assessment(value: object) -> JudgeAssessment:
    """Load a historical v1 or prospective v2 judge checkpoint."""
    if isinstance(value, ScientificJudgeResult | JudgeResult):
        return value
    if isinstance(value, dict) and value.get("schema_version") == "2":
        return ScientificJudgeResult.model_validate(value)
    return JudgeResult.model_validate(value)
