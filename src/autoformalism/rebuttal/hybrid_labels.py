"""Certified question-level labels for hybrid judge calibration."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.schemas import (
    AbsoluteCriterion,
    AbsoluteVerdict,
    RelativeCriterion,
)
from autoformalism.schemas.base import Identifier, NonEmptyText, StrictSchema


class ExpectedVerdict(str, Enum):
    """Gold absolute verdict, with an explicit unscored state."""

    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"
    UNLABELED = "unlabeled"


class ExpectedPairPreference(str, Enum):
    """Gold pairwise outcome, including an explicit unscored state."""

    BASELINE = "baseline"
    MUTATED = "mutated"
    TIE = "tie"
    INDETERMINATE = "indeterminate"
    UNLABELED = "unlabeled"


class ExpectedAbsoluteLabel(StrictSchema):
    """Question-level truth for both members of one calibration pair."""

    criterion: AbsoluteCriterion
    subject_id: Identifier
    baseline: ExpectedVerdict
    mutated: ExpectedVerdict
    rationale: NonEmptyText
    label_source: NonEmptyText


class ExpectedComparativeLabel(StrictSchema):
    """Expected outcome for one irreducibly comparative question."""

    criterion: RelativeCriterion
    preference: ExpectedPairPreference
    rationale: NonEmptyText
    label_source: NonEmptyText


class HybridCalibrationLabels(StrictSchema):
    """Frozen item-level and overall truth for one blinded pair."""

    schema_version: Literal["hybrid-labels-2"] = "hybrid-labels-2"
    pair_id: Identifier
    overall_preference: ExpectedPairPreference
    absolute_labels: tuple[ExpectedAbsoluteLabel, ...] = Field(max_length=256)
    comparative_labels: tuple[ExpectedComparativeLabel, ...] = Field(
        max_length=len(RelativeCriterion)
    )

    @model_validator(mode="after")
    def unique_labels(self) -> HybridCalibrationLabels:
        """Reject ambiguous duplicate gold labels."""
        absolute = [
            (item.criterion, item.subject_id) for item in self.absolute_labels
        ]
        if len(absolute) != len(set(absolute)):
            raise ValueError("absolute gold-label keys must be unique")
        comparative = [item.criterion for item in self.comparative_labels]
        if len(comparative) != len(set(comparative)):
            raise ValueError("comparative gold-label criteria must be unique")
        return self


def expected_from_runtime(verdict: AbsoluteVerdict) -> ExpectedVerdict:
    """Convert one certified runtime verdict to a gold-label value."""
    return ExpectedVerdict(verdict.value)


class MutationAbsoluteExpectation(StrictSchema):
    """One semantic verdict guaranteed by a controlled mutation recipe."""

    criterion: AbsoluteCriterion
    subject_id: Identifier = "candidate"
    baseline: ExpectedVerdict = ExpectedVerdict.UNLABELED
    mutated: ExpectedVerdict
    rationale: NonEmptyText


class MutationComparativeExpectation(StrictSchema):
    """One relative verdict guaranteed by a controlled mutation recipe."""

    criterion: RelativeCriterion
    preference: ExpectedPairPreference
    rationale: NonEmptyText


class MutationLabelContract(StrictSchema):
    """Auditable gold-label consequences of one controlled mutation."""

    mutation_type: Identifier
    overall_preference: ExpectedPairPreference
    absolute: tuple[MutationAbsoluteExpectation, ...] = ()
    comparative: tuple[MutationComparativeExpectation, ...] = ()


_BASELINE = ExpectedPairPreference.BASELINE
_CONTRACTS = {
    contract.mutation_type: contract
    for contract in (
        MutationLabelContract(
            mutation_type="wrong_meal_sink",
            overall_preference=_BASELINE,
            absolute=(
                MutationAbsoluteExpectation(
                    criterion=AbsoluteCriterion.SOURCE_ROLES_CONSISTENT,
                    mutated=ExpectedVerdict.FAIL,
                    rationale=(
                        "The mutation adds meal input as a negative glucose term, "
                        "which contradicts its certified source role."
                    ),
                ),
            ),
            comparative=(
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
                    preference=_BASELINE,
                    rationale=(
                        "Only the mutated candidate adds an unsupported meal sink."
                    ),
                ),
            ),
        ),
        MutationLabelContract(
            mutation_type="duplicated_gp_flux",
            overall_preference=_BASELINE,
            absolute=(
                MutationAbsoluteExpectation(
                    criterion=AbsoluteCriterion.SEMANTIC_FLUXES_NOT_DUPLICATED,
                    mutated=ExpectedVerdict.FAIL,
                    rationale=(
                        "The mutation repeats an existing top-level glucose-balance "
                        "flux without adding a distinct mechanism."
                    ),
                ),
            ),
            comparative=(
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
                    preference=_BASELINE,
                    rationale=(
                        "The mutated expression duplicates an existing flux and "
                        "adds no task-relevant capability."
                    ),
                ),
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
                    preference=_BASELINE,
                    rationale="Only the mutated candidate assumes a duplicate flux.",
                ),
            ),
        ),
        MutationLabelContract(
            mutation_type="unjustified_one_sided_accumulator",
            overall_preference=_BASELINE,
            absolute=(
                MutationAbsoluteExpectation(
                    criterion=AbsoluteCriterion.LATENT_ACCUMULATORS_JUSTIFIED,
                    mutated=ExpectedVerdict.FAIL,
                    rationale=(
                        "The added latent state has an input but no release, decay, "
                        "or other removal term."
                    ),
                ),
            ),
            comparative=(
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
                    preference=_BASELINE,
                    rationale=(
                        "Only the mutated candidate adds a one-sided latent state "
                        "that is unnecessary for the unchanged targets."
                    ),
                ),
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
                    preference=_BASELINE,
                    rationale=(
                        "Only the mutated candidate assumes an unjustified latent "
                        "accumulation mechanism."
                    ),
                ),
            ),
        ),
        MutationLabelContract(
            mutation_type="retained_disconnected_claimed_mechanism",
            overall_preference=_BASELINE,
            comparative=(
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.PARSIMONY_WHILE_TASK_SUFFICIENT,
                    preference=_BASELINE,
                    rationale=(
                        "Only the mutated candidate adds a retained subsystem with "
                        "no path to any requested target."
                    ),
                ),
                MutationComparativeExpectation(
                    criterion=RelativeCriterion.FEWER_UNSUPPORTED_ASSUMPTIONS,
                    preference=_BASELINE,
                    rationale=(
                        "Only the mutated candidate claims a target-disconnected "
                        "mechanism."
                    ),
                ),
            ),
        ),
    )
}


def mutation_label_contract(mutation_type: str) -> MutationLabelContract:
    """Return the explicit contract for a mutation that survives repair."""
    try:
        return _CONTRACTS[mutation_type]
    except KeyError as error:
        raise ValueError(
            f"no certified hybrid-label contract for mutation: {mutation_type}"
        ) from error
