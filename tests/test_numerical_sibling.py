"""Optional numerical hypotheses, atomic changes and physical-call accounting."""

import copy
import json

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import prefit_numerical_sibling as campaign
from autoformalism.rebuttal.prefit_feedback import EpisodeClient
from autoformalism.search import numerical_sibling as review
from scripts.smoke_prefit_numerical_sibling import fixture


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    root = fixture(tmp_path_factory.mktemp("numerical-sibling"))
    campaign.replay(root)
    return campaign.verify(root)


def reply_for(plan, residual, *, action="revise_function", expression=None):
    selected = next(
        s for s in plan["bundle"]["slots"] if s["selected_term"]["lhs"] == "m"
    )
    revision = {
        **selected["accepted_reply"],
        "interaction_id": selected["interaction_id"],
    }
    if expression:
        revision["expression"] = expression
    else:
        revision["expression"] = revision["expression"].replace("v01**2", "tanh(v01)")
    return {
        "action": action,
        "hypothesis": (
            "The measured mismatch motivates a bounded feedback hypothesis; "
            "the fit is unfinished."
        ),
        "evidence_ids": [residual["packet"]["rows"][0]["evidence_id"]],
        "revision": revision if action == "revise_function" else None,
    }


def test_atomic_revision_preserves_required_feedback_and_causal_boundary(context):
    plan, residual = context
    before = copy.deepcopy(plan)
    result = review.apply_revision(
        plan["bundle"], plan["bindings"], residual["packet"], reply_for(plan, residual)
    )
    assert result["outcome"] == "committed"
    final = result["final"]
    assert final["protected_slots_preserved"] == len(plan["bundle"]["slots"]) - 1
    assert final["initialization"]["plan"] == plan["bundle"]["initialization"]["plan"]
    assert not final["diagnosis"]["requirement_gap"]
    assert result["provenance"]["stage"] == "numerical_sibling"
    assert not result["provenance"]["hypothesis_confirmed"]
    assert plan == before


def test_noop_canonical_reply_does_not_create_child(context):
    plan, residual = context
    selected = next(
        s for s in plan["bundle"]["slots"] if s["selected_term"]["lhs"] == "m"
    )
    reply = reply_for(
        plan, residual, expression=selected["accepted_reply"]["expression"]
    )
    result = review.apply_revision(
        plan["bundle"], plan["bindings"], residual["packet"], reply
    )
    assert result["outcome"] == "unchanged_canonical_function"
    assert result["final"] is None


@pytest.mark.parametrize(
    "mutation", ["linear", "source", "evidence", "initializer", "wrong_action"]
)
def test_invalid_revision_cannot_change_parent(context, mutation):
    plan, residual = context
    before = copy.deepcopy(plan)
    raw = reply_for(plan, residual)
    if mutation == "linear":
        raw["revision"]["expression"] = raw["revision"]["expression"].replace(
            "tanh(v01)", "v01"
        )
    elif mutation == "source":
        raw["revision"]["expression"] += "+unlisted_state"
    elif mutation == "evidence":
        raw["evidence_ids"] = ["validation_result"]
    elif mutation == "initializer":
        raw["revision"]["initialization_plan"] = {}
    else:
        raw["action"] = "no_change"
    with pytest.raises((ValueError, ModelValidationError)):
        review.apply_revision(plan["bundle"], plan["bindings"], residual["packet"], raw)
    assert plan == before


def test_payload_uses_exact_retained_parameters_without_raw_fitter(context):
    plan, residual = context
    parameters = residual["seed"]["state"]["parameters"]
    payload = review.payload(
        plan["bundle"], plan["bindings"], residual["packet"], parameters
    )
    assert payload["retained_fitted_parameters"] == parameters
    assert "validation" not in payload and "seed" not in payload
    with pytest.raises(ValueError, match="parameters differ"):
        review.payload(
            plan["bundle"],
            plan["bindings"],
            residual["packet"],
            {**parameters, "unknown": 1},
        )


def client(root, plan, replies, calls):
    def transport(url, body, timeout):
        index = len(calls)
        calls.append(body)
        reply, reason = replies[min(index, len(replies) - 1)]
        return {
            "choices": [
                {"finish_reason": reason, "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return EpisodeClient(
        settings=campaign.SiblingConfig.model_validate(plan["config"]).model_settings,
        base_url="http://unused",
        directory=root / "results/calls",
        namespace=campaign._namespace(plan),
        seed=0,
        transport=transport,
    )


def test_truncated_delivery_is_not_a_mathematical_rejection_and_resume_is_exact(
    context, tmp_path
):
    plan, residual = context
    raw = reply_for(plan, residual, action="no_change")
    calls = []
    proposer = client(tmp_path, plan, [(raw, "length"), (raw, "stop")], calls)
    state = campaign.run_episode(tmp_path, plan, residual, proposer)
    assert state["stop_reason"] == "no_change" and len(calls) == 2
    feedback = state["attempts"][0]["feedback"]
    assert feedback["stage"] == "provider_response"
    assert feedback["code"] == "PROVIDER_RESPONSE_TRUNCATED"
    assert not feedback["mathematical_reply_evaluated"]
    assert campaign.run_episode(tmp_path, plan, residual, proposer) == state
    assert len(calls) == 2


def test_exhaustion_keeps_valid_parent(context, tmp_path):
    plan, residual = context
    raw = reply_for(plan, residual)
    raw["evidence_ids"] = ["not_present"]
    calls = []
    proposer = client(tmp_path, plan, [(raw, "stop")], calls)
    state = campaign.run_episode(tmp_path, plan, residual, proposer)
    assert state["stop_reason"] == "attempts_exhausted"
    assert state["decision"] is None and len(calls) == 3
    assert campaign.run_episode(tmp_path, plan, residual, proposer) == state
    assert len(calls) == 3


def test_report_uses_measured_provider_latency_without_new_calls(context, tmp_path):
    plan, residual = context
    raw = reply_for(plan, residual, action="no_change")
    calls = []
    proposer = client(tmp_path, plan, [(raw, "stop")], calls)
    campaign.run_episode(tmp_path, plan, residual, proposer)
    _, records = campaign._state(tmp_path, plan, residual)
    expected = sum(r["latency_seconds"] for r in records)
    assert expected > 0
    report = campaign._report(tmp_path, plan, residual)
    assert report["provider_seconds"] == pytest.approx(expected)
    assert report["unknown_latency_requests"] == 0
    assert len(calls) == report["physical_requests"] == 1


def test_response_saved_before_state_commit_is_reused(context, tmp_path, monkeypatch):
    plan, residual = context
    raw = reply_for(plan, residual, action="no_change")
    calls = []
    original = campaign._save_state

    def interrupted(*args):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(campaign, "_save_state", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        campaign.run_episode(
            tmp_path, plan, residual, client(tmp_path, plan, [(raw, "stop")], calls)
        )
    monkeypatch.setattr(campaign, "_save_state", original)
    state = campaign.run_episode(
        tmp_path, plan, residual, client(tmp_path, plan, [(raw, "stop")], calls)
    )
    assert state["stop_reason"] == "no_change" and len(calls) == 1


def test_uncertain_delivery_consumes_request_and_next_attempt_only(context, tmp_path):
    plan, residual = context
    raw = reply_for(plan, residual, action="no_change")
    calls = []
    proposer = client(tmp_path, plan, [(raw, "stop")], calls)

    def interrupted(*args):
        raise KeyboardInterrupt

    proposer.transport = interrupted
    with pytest.raises(KeyboardInterrupt):
        campaign.run_episode(tmp_path, plan, residual, proposer)
    resumed = client(tmp_path, plan, [(raw, "stop")], calls)
    state = campaign.run_episode(tmp_path, plan, residual, resumed)
    assert len(state["attempts"]) == 2 and len(calls) == 1
    assert state["attempts"][0]["feedback"]["code"] == "PROVIDER_RESPONSE_UNAVAILABLE"
    assert state["stop_reason"] == "no_change"
    report = campaign._report(tmp_path, plan, residual)
    assert report["physical_requests"] == 2
    assert report["unknown_latency_requests"] == 1
