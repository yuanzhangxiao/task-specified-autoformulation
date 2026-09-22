"""Campaign identity for external discovery methods run at a pinned revision.

These methods are not reimplemented. Their own repositories run the search,
and this module records what was actually run, so a result can be reported
against the protocol its authors published rather than against ours.

Three kinds of departure from an upstream default are distinguished, because
they license different claims:

* **compute budget** may be reduced, because a published budget is a resource
  decision. It is recorded next to the published value so a result is never
  compared with a citation obtained at a different budget.
* **prompt information** may be equalised across baselines, because withholding
  the public task specification from one method measures information access
  rather than method quality. Any such change is declared.
* **operators, priors, fitting and selection** may not change at all. That is
  the method, and altering it would make the result someone else's work
  reported under their name.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.rebuttal.final_evaluation_pilot import FinalEvaluationPilotCell

#: Upstream methods integrated by vendoring rather than reimplementation.
VendoredMethod = Literal["llm_sr", "llm_ode"]


class UpstreamRevision(BaseModel):
    """The exact external code that produced a campaign's models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=1)
    #: Left unmodified. Recording it makes a later diff a reviewable claim.
    search_modified: Literal[False] = False


class DeclaredBudget(BaseModel):
    """What this campaign spends, beside what the method publishes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: Literal["llm_samples", "iterations"]
    published_default: int = Field(gt=0)
    declared: int = Field(gt=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def reduction_is_stated(self) -> DeclaredBudget:
        """A reduced budget needs a reason; an increased one is not ours to make."""
        if self.declared > self.published_default:
            raise ValueError(
                "a budget above the published default is not a faithful run"
            )
        return self

    @property
    def is_reduced(self) -> bool:
        """Whether results must be reported as budget-conditioned."""
        return self.declared < self.published_default


class PromptPolicy(BaseModel):
    """Whether the method received the public task specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supplies_public_task_specification: bool
    #: True when the upstream protocol withholds it, so supplying it is an
    #: adaptation that must never be described as the native prompt.
    upstream_withholds_specification: bool
    search_prompt_instructions_modified: Literal[False] = False
    autoformalism_internals_disclosed: Literal[False] = False
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def an_adaptation_is_declared(self) -> PromptPolicy:
        """Departing from the upstream prompt requires saying so."""
        adapted = (
            self.supplies_public_task_specification
            and self.upstream_withholds_specification
        )
        if adapted and "adaptation" not in self.description.lower():
            raise ValueError(
                "supplying a specification the upstream protocol withholds must "
                "be described as an adaptation"
            )
        return self


class VendoredCampaignPlan(BaseModel):
    """Everything that fixes a campaign's identity before its first call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["phase-b-vendored-campaign-plan-1"]
    status: Literal["proposed_pending_review", "frozen_before_calls"]
    method: VendoredMethod
    upstream: UpstreamRevision
    budget: DeclaredBudget
    prompt_policy: PromptPolicy
    #: Upstream's island count. It changes the search and the call volume, so
    #: it belongs to the sealed identity: read from the environment at run time
    #: it would silently alter what a frozen plan means.
    islands: int = Field(default=4, ge=1, le=64)
    provider: Literal["openai", "vllm"]
    model: str = Field(min_length=1)
    cells: tuple[FinalEvaluationPilotCell, ...] = Field(min_length=1)
    repetitions: tuple[int, ...] = Field(min_length=1)
    execution_semantics: Literal["continuous_ode_free_rollout"]
    #: How the method obtains derivatives. Upstream choices are kept, so a
    #: method is not credited or penalised for our estimator; the
    #: difference from the other symbolic baselines is reported instead.
    derivative_provenance: Literal[
        "estimated_numpy_gradient",
        "upstream_findiff_fourth_order",
        "not_applicable",
    ]
    test_data_opened: Literal[False]
    private_reference_opened: Literal[False]

    @model_validator(mode="after")
    def identities_are_unique(self) -> VendoredCampaignPlan:
        """Reject duplicate cells or repetitions."""
        cells = [(item.benchmark_id, item.tier) for item in self.cells]
        if len(cells) != len(set(cells)):
            raise ValueError("benchmark cells must be unique")
        if len(self.repetitions) != len(set(self.repetitions)) or any(
            item < 0 for item in self.repetitions
        ):
            raise ValueError("repetitions must be unique and nonnegative")
        if ":" in self.model:
            raise ValueError("supply a model id without a provider prefix")
        return self

    @property
    def expected_task_count(self) -> int:
        """One task per planned cell and repetition."""
        return len(self.cells) * len(self.repetitions)

    def reporting_qualifications(self) -> tuple[str, ...]:
        """Statements any report of this campaign must carry."""
        notes: list[str] = []
        if self.budget.is_reduced:
            notes.append(
                f"budget-conditioned at {self.budget.declared} "
                f"{self.budget.unit} against a published default of "
                f"{self.budget.published_default}"
            )
        if (
            self.prompt_policy.supplies_public_task_specification
            and self.prompt_policy.upstream_withholds_specification
        ):
            notes.append(
                "prompt adapted: supplied the public task specification, which "
                "the upstream protocol withholds"
            )
        return tuple(notes)
