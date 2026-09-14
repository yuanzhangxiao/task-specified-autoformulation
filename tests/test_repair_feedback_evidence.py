"""Recorded failure shapes exercised without provider calls or benchmark mutation."""

import json
from types import SimpleNamespace

import pytest

from autoformalism.fitting.initialization import (
    InitialCausalMap,
    InitialValueGuess,
    LatentInitializationPlan,
    LatentInitializationReply,
)
from autoformalism.judging import build_atomic_evidence_plan
from autoformalism.rebuttal.repair_drafts import RepairActionV2, advance, empty_draft
from autoformalism.rebuttal.repair_evidence import decision_report, numerical_findings
from autoformalism.rebuttal.repair_feedback import initialization_facts
from autoformalism.rebuttal.repair_judge_evidence import named_references, review_status
from autoformalism.rebuttal.repair_transactions import (
    default_initialization,
    request_repair,
)
from autoformalism.schemas import AtomicJudgeResult
from tests.test_repair_comparison import CONTEXT, Client, candidate


def test_collocation_timeout_is_not_missing_initial_state_policy():
    model = candidate()
    plan = default_initialization(model, CONTEXT)
    fit = {
        "status": "complete",
        "initializer": {
            "success": False,
            "message": "initializer wall-clock limit reached",
        },
        "parameters": {"init_m_value": -2.4},
    }
    facts = initialization_facts(model, plan, CONTEXT, fit)
    row = facts["latent_boundaries"][0]
    assert (
        row["state"] == "m" and row["policy_present"] and row["training_fitted_policy"]
    )
    assert row["optimizer_guesses_not_fixed_values"] == {"init_m_value": 0.0}
    assert row["selected_fit_parameter_values"] == {"init_m_value": -2.4}
    assert facts["observed_initial_channels"] == {"y": "v01"}
    assert not facts["validation_initial_parameters_fitted"]
    observations = numerical_findings(model, fit)
    assert observations[0].code == "COLLOCATION_INITIALIZER_UNAVAILABLE"
    assert "not latent-state" in observations[0].observation
    assert any(f.code == "FINITE_ROLLOUT" for f in observations)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "0"])
def test_missing_or_invalid_selected_initial_is_not_a_zero_estimate(value):
    model = candidate()
    facts = initialization_facts(
        model,
        default_initialization(model, CONTEXT),
        CONTEXT,
        {"parameters": {"init_m_value": value}},
    )
    assert facts["latent_boundaries"][0]["selected_fit_parameter_values"] == {}
    assert facts["latent_boundaries"][0]["selected_values_available"] is False


def test_map_parameter_names_come_from_real_lowering():
    model = candidate()
    plan = LatentInitializationPlan(
        rules={
            "m": LatentInitializationReply(
                initial=InitialCausalMap(expression="a*v01", parameters=[{"name": "a"}])
            )
        }
    )
    row = initialization_facts(model, plan, CONTEXT, {"parameters": {"init_m_a": 3.0}})[
        "latent_boundaries"
    ][0]
    assert row["mode"] == "map" and not row["already_shared_training_value"]
    assert row["selected_fit_parameter_values"] == {"init_m_a": 3.0}


def test_repeated_initializer_has_exact_noop_explanation_and_resume(tmp_path):
    model = candidate()
    plan = default_initialization(model, CONTEXT)
    response = {
        "scope": "model",
        "hypothesis": "Missing initials cause a timeout.",
        "initializers": [{"state": "m", "causal_map": None}],
    }
    client = Client([response])
    args = (model, plan, CONTEXT, {}, "Public", client, tmp_path, 1)
    revised, initial, transaction = request_repair(*args)
    assert revised == model and initial == plan
    audit = transaction["audit"]
    assert audit["status"] == "no_change"
    assert audit["initializer_changes"] == []
    effect = audit["actual_effects"][0]
    assert effect["before"] == effect["after"]
    assert effect["reason"] == "shared_training_value_already_enabled"
    assert transaction["action_states"][0]["status"] == "unchanged"
    assert request_repair(*args)[2] == transaction and len(client.requests) == 1
    history = [
        {
            "outcome": "no_change",
            "actual_effects": audit["actual_effects"],
            "no_change_reason": audit["no_change_reason"],
        }
    ]
    next_report = decision_report(model, [], [], [], history)
    assert next_report["recent_actions"][0]["actual_effects"][0] == effect


@pytest.mark.parametrize(
    "raw,reason",
    [
        ({"scope": "no_change"}, "explicit_decline"),
        ({"scope": "model"}, "empty_edit"),
        (
            {
                "scope": "model",
                "equations": [{"component": "f", "expression": "(sigmoid( m ))"}],
            },
            "equivalent_model_and_initialization",
        ),
    ],
)
def test_noop_types_are_separate(raw, reason):
    model = candidate()
    _, _, draft, diagnostics = advance(
        model,
        default_initialization(model, CONTEXT),
        CONTEXT,
        empty_draft(),
        RepairActionV2.model_validate(raw),
    )
    assert not diagnostics and draft["status"] == "no_change"
    assert draft["audit"]["no_change_reason"] == reason


def test_null_map_can_be_a_real_change_from_a_causal_map():
    model = candidate()
    plan = LatentInitializationPlan(
        rules={
            "m": LatentInitializationReply(initial=InitialCausalMap(expression="v01"))
        }
    )
    _, revised, draft, diagnostics = advance(
        model,
        plan,
        CONTEXT,
        empty_draft(),
        RepairActionV2.model_validate(
            {"scope": "model", "initializers": [{"state": "m"}]}
        ),
    )
    assert not diagnostics and draft["status"] == "committed"
    assert isinstance(revised.rules["m"].initial, InitialValueGuess)
    assert draft["audit"]["actual_effects"][0]["status"] == "changed"


def test_effects_preserve_actual_sign_change_regardless_of_hypothesis():
    model = candidate()
    raw = {
        "scope": "model",
        "hypothesis": "Restore a positive contribution.",
        "equations": [{"component": "m", "expression": "-rate*m-gain*u01"}],
    }
    _, _, draft, diagnostics = advance(
        model,
        default_initialization(model, CONTEXT),
        CONTEXT,
        empty_draft(),
        RepairActionV2.model_validate(raw),
    )
    assert not diagnostics and draft["status"] == "committed"
    effect = draft["audit"]["actual_effects"][0]
    assert effect["before"][1] == "-rate*m+gain*u01"
    assert effect["after"][1] == "-rate*m-gain*u01"
    assert json.loads(json.dumps(draft)) == draft


def test_exact_judge_references_resolve_both_orientations_not_unknown_ids():
    model = candidate()
    model = model.model_copy(
        update={
            "state_equations": (
                model.state_equations[0].model_copy(
                    update={"rhs": "-(rate*m)+gain*u01"}
                ),
                model.state_equations[1],
            )
        }
    )
    plan = build_atomic_evidence_plan(model, model)
    items = [
        o
        for o in plan.occurrences
        if o.governed_quantity == "m" and set(o.symbols) == {"rate", "m"}
    ]
    assert len(items) == 2
    evidence = " ".join(o.occurrence_id for o in items) + " occurrence_unknown"
    result = named_references(model, model, evidence)
    assert {r["orientation"] for r in result["named_references"]} == {
        "forward",
        "reverse",
    }
    assert all(
        r["actual_outer_polarity"] == "negative" for r in result["named_references"]
    )
    assert all(r["location"] == "equation:m" for r in result["named_references"])
    assert result["unresolved_reference_ids"] == ["occurrence_unknown"]
    assert "expected_direction" not in json.dumps(result)


def test_parent_only_judge_reference_does_not_blame_current_candidate():
    parent, child = candidate(), candidate()
    child = child.model_copy(
        update={
            "state_equations": (
                child.state_equations[0].model_copy(
                    update={"rhs": "-rate*m+2*gain*u01"}
                ),
                child.state_equations[1],
            )
        }
    )
    plan = build_atomic_evidence_plan(parent, child)
    old = next(
        o
        for o in plan.occurrences
        if o.candidate_side == "candidate_a" and o.unsigned_expression == "gain * u01"
    )
    result = named_references(parent, child, old.occurrence_id)
    assert result["named_references"] == []
    assert result["unresolved_reference_ids"] == [old.occurrence_id]


def test_judge_missing_and_duplicate_units_stay_failures_not_approval():
    payload = {
        "signed_occurrence_assessments": [],
        "repeated_contribution_assessments": [],
    }
    parsed = AtomicJudgeResult.model_validate(payload)
    with pytest.raises(ValueError, match="missing_occurrences") as missing:
        parsed.validate_expected_units(
            occurrence_ids={"occurrence_a"}, repeat_pair_ids=set()
        )
    entry = {
        "occurrence_id": "occurrence_a",
        "expected_direction": "positive_contribution",
        "evidence": "Hypothesis only.",
    }
    with pytest.raises(ValueError, match="must be unique") as duplicate:
        AtomicJudgeResult.model_validate(
            {"signed_occurrence_assessments": [entry, entry]}
        )
    review = {
        "status": "indeterminate",
        "findings": [],
        "comparison": {
            "prior_terminal_failures": [str(missing.value), str(duplicate.value)]
        },
    }
    status = review_status(review)
    assert not status["advice_available"]
    assert status["no_findings_does_not_imply_approval"]
    assert [d["code"] for d in status["diagnostics"]] == [
        "JUDGE_ATOMIC_UNIT_SET_MISMATCH",
        "JUDGE_DUPLICATE_OCCURRENCE_ID",
    ]
    assert review_status({"status": "reviewed", "findings": []})["advice_available"]
    assert review_status(None)["status"] == "not_requested"


@pytest.mark.parametrize("routing_policy", ["legacy_category", "evidence_strength"])
def test_next_live_request_receives_noop_effect_and_current_initial_values(
    tmp_path, monkeypatch, routing_policy
):
    from autoformalism.rebuttal import repair_comparison as campaign

    model = candidate()
    task = {
        "task_id": "synthetic",
        "candidate_path": "parent.json",
        "benchmark_id": "control",
        "tier": "hard",
        "seed": 0,
        "arm": campaign.ARMS[0],
    }
    config = campaign.RepairComparisonConfig(
        judge_revision="a" * 40,
        rounds=2,
        routing_policy=routing_policy,
        protocol="repair-feedback-comparison-4"
        if routing_policy == "evidence_strength"
        else "repair-feedback-comparison-3",
    )
    campaign.atomic_json(tmp_path / "inputs/parent.json", model.model_dump(mode="json"))
    prompt = tmp_path / "inputs/frozen/public/phase_b_v1/control/proposer_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Public task: predict v01 through internal dynamic memory.")
    monkeypatch.setattr(
        campaign, "verify", lambda root: {"config": config.model_dump(mode="json")}
    )
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *a: (SimpleNamespace(train=None, validation=None), CONTEXT),
    )
    monkeypatch.setattr(campaign, "_public_fit_context", lambda data: {})
    fits = []

    def fit(*args, **kwargs):
        fits.append(1)
        return {
            "status": "complete",
            "parameters": {"init_m_value": 2.0},
            "initializer": {
                "success": False,
                "message": "initializer wall-clock limit reached",
            },
            "training": {"normalized_mse": 0.5},
            "validation": {"normalized_mse": 1 / len(fits)},
        }

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", fit)

    class RoundClient(Client):
        def call(self, **kwargs):
            self.replies = (
                [{"scope": "model", "initializers": [{"state": "m"}]}]
                if kwargs["step"].endswith("1")
                else [
                    {
                        "scope": "function",
                        "equations": [{"component": "f", "expression": "tanh(m)"}],
                    }
                ]
            )
            return super().call(**kwargs)

    client = RoundClient([{}])
    result = campaign.run_task(tmp_path, task, client)
    assert [r["outcome"] for r in result["rounds"]] == ["no_change", "committed"]
    assert len(fits) == 2  # baseline + changed candidate, not redundant initializer
    report = client.requests[1]["report"]
    assert (
        report["recent_actions"][0]["actual_effects"][0]["reason"]
        == "shared_training_value_already_enabled"
    )
    assert report["initialization_facts"]["latent_boundaries"][0][
        "selected_fit_parameter_values"
    ] == {"init_m_value": 2.0}
    assert report["scientific_review_status"]["status"] == "not_requested"
    assert ("priority_evidence" in report) == (routing_policy == "evidence_strength")
    if routing_policy == "evidence_strength":
        assert report["numerical_reliability"]["support"]["strength"] == "unresolved"
    assert result["best"]["round"] == 2
    assert campaign.run_task(tmp_path, task, client) == result
    assert len(fits) == 2 and len(client.requests) == 2


def test_summary_separates_unavailable_judge_from_completed_campaign(
    tmp_path, monkeypatch
):
    from autoformalism.rebuttal import repair_comparison as campaign
    from tests.test_repair_comparison import split_plan

    plan = split_plan(tmp_path, monkeypatch)
    for task in plan["tasks"]:
        campaign.atomic_json(
            tmp_path / "results" / task["task_id"] / "state.json",
            {"status": "complete"},
        )
    campaign.atomic_json(
        tmp_path / "reviews/one/review.json", {"status": "reviewed", "findings": []}
    )
    campaign.atomic_json(
        tmp_path / "reviews/two/review.json",
        {
            "status": "indeterminate",
            "findings": [],
            "comparison": {
                "prior_terminal_failures": [
                    "signed occurrence identifiers must be unique"
                ]
            },
            "cost": {"physical_requests": 20},
        },
    )
    summary = campaign.summarize(tmp_path, arm="redesigned_prefit_judge")
    assert summary["status"] == "complete"
    assert summary["scientific_review_status_counts"] == {
        "reviewed": 1,
        "indeterminate": 1,
    }
    missing = next(
        r
        for r in summary["scientific_review_diagnostics"]
        if r["status"] == "indeterminate"
    )
    assert missing["physical_requests"] == 20 and not missing["advice_available"]
    assert (
        campaign.summarize(tmp_path, arm="redesigned_runtime")[
            "scientific_review_status_counts"
        ]
        == {}
    )
