"""Campaign identity for external methods run at a pinned upstream revision.

These methods are vendored, not reimplemented, so the record of what was run
has to carry the departures from their published protocol. Three are
distinguished: a compute budget may be reduced, prompt information may be
equalised, and the search itself may not change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.vendored_campaign import (
    DeclaredBudget,
    PromptPolicy,
    VendoredCampaignPlan,
)

CONFIGS = {
    "llm_sr": Path("configs/phase_b_llm_sr_campaign_v1.json"),
    "llm_ode": Path("configs/phase_b_llm_ode_campaign_v1.json"),
}


def _plan(name: str) -> VendoredCampaignPlan:
    return VendoredCampaignPlan.model_validate_json(
        CONFIGS[name].read_text(encoding="utf-8")
    )


def test_both_campaigns_cover_the_full_matrix_at_a_pinned_revision() -> None:
    for name in CONFIGS:
        plan = _plan(name)
        assert plan.expected_task_count == 120
        assert plan.upstream.search_modified is False
        assert len(plan.upstream.commit) == 40
        assert plan.test_data_opened is False
        # nothing may run until a budget is confirmed
        assert plan.status == "proposed_pending_review"


def test_a_reduced_budget_must_be_reported_against_the_published_default() -> None:
    plan = _plan("llm_sr")
    assert plan.budget.published_default == 10000
    assert plan.budget.is_reduced
    notes = plan.reporting_qualifications()
    assert any("budget-conditioned at 200" in note for note in notes)
    assert any("10000" in note for note in notes)


def test_a_budget_above_the_published_default_is_not_faithful() -> None:
    with pytest.raises(ValueError, match="not a faithful run"):
        DeclaredBudget(
            unit="llm_samples",
            published_default=100,
            declared=200,
            rationale="more search than the authors ran",
        )


def test_supplying_a_withheld_specification_must_be_called_an_adaptation() -> None:
    """LLM-ODE withholds it by design; equalising is a declared change."""
    with pytest.raises(ValueError, match="must be described as an adaptation"):
        PromptPolicy(
            supplies_public_task_specification=True,
            upstream_withholds_specification=True,
            description="supplied the public task specification",
        )
    allowed = PromptPolicy(
        supplies_public_task_specification=True,
        upstream_withholds_specification=True,
        description="Declared adaptation: equalises task information.",
    )
    assert allowed.search_prompt_instructions_modified is False


def test_llm_ode_records_the_prompt_adaptation_and_llm_sr_does_not() -> None:
    """LLM-SR's own specifications embed a description; LLM-ODE's do not."""
    ode = _plan("llm_ode")
    assert ode.prompt_policy.upstream_withholds_specification is True
    assert any(
        "prompt adapted" in note for note in ode.reporting_qualifications()
    )
    sr = _plan("llm_sr")
    assert sr.prompt_policy.upstream_withholds_specification is False
    assert not any(
        "prompt adapted" in note for note in sr.reporting_qualifications()
    )


def test_the_search_itself_is_not_adjustable_through_the_plan() -> None:
    """Operators, priors, fitting and selection are the method."""
    raw = json.loads(CONFIGS["llm_sr"].read_text(encoding="utf-8"))
    raw["upstream"]["search_modified"] = True
    with pytest.raises(ValueError):
        VendoredCampaignPlan.model_validate(raw)
    raw = json.loads(CONFIGS["llm_ode"].read_text(encoding="utf-8"))
    raw["prompt_policy"]["search_prompt_instructions_modified"] = True
    with pytest.raises(ValueError):
        VendoredCampaignPlan.model_validate(raw)


def test_llm_sr_uses_the_same_derivatives_as_the_other_symbolic_baselines() -> None:
    """It regresses a derivative from features, as SINDy and PySR do here."""
    assert _plan("llm_sr").derivative_provenance == "estimated_numpy_gradient"
