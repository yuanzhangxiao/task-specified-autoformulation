"""Frozen paired arms, explicit unanswered hypotheses and durable one-call trials."""

import json

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting.public_fitting import content_sha256
from scripts import compare_graph_critic_feedback as critic


def setup_packets(
    root, readiness="conditional_equation_review", cohort="fresh", controls=()
):
    freeze = {
        "fresh_parents": ["one", "two"],
        "transfer_controls": list(controls),
        "critic_policy_sha256": content_sha256(critic.policy()),
    }
    seal(root / "freeze.json", freeze)
    for parent in dict.fromkeys([*freeze["fresh_parents"], *controls]):
        packet = {
            "protocol": "graph-critic-feedback-1",
            "parent_id": parent,
            "cohort": cohort if parent == "two" else "fresh",
            "freeze_sha256": content_sha256(freeze),
            "public_task": "Specify a causal transfer.",
            "equations": {"x": "u - k*x"},
            "evidence": {"window-1": {"observed": 0.6, "predicted": 0.1}},
            "deterministic_feedback": ["The predicted rise is small."],
            "readiness": readiness,
        }
        if parent in controls:
            packet["cohort"] = "transfer_control"
        seal(root / "feedback" / f"{parent}.json", packet)


def response(changes=None):
    advice = {
        "action": "investigate_equation",
        "evidence_ids": ["window-1", "other"],
        "equation_ids": ["x"],
        "hypothesis_or_reason": "The input contribution could be too small.",
        "alternative_explanation": "Excessive clearance could have the same effect.",
        "discriminating_check": "Inspect signed flows across the window.",
        "disconfirming_outcome": "Recovery improves equally under revised clearance.",
    }
    advice.update(changes or {})
    return {
        "model": critic.MODEL,
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(advice)}}
        ],
        "usage": {"total_tokens": 123},
    }


def test_common_inputs_and_one_request_on_resume(tmp_path):
    setup_packets(tmp_path)
    calls = []

    def provider(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        assert packet == read_seal(
            tmp_path / "feedback" / f"{packet['parent_id']}.json"
        )
        calls.append(packet["parent_id"])
        return response()

    prepared = critic.compare(tmp_path, prepare_only=True, transport=provider)
    assert prepared["statuses"] == {"prepared": 2} and calls == []
    complete = critic.compare(tmp_path, transport=provider)
    assert calls == ["one", "two"]
    assert complete["paired_feedback_available"] == 2
    assert critic.compare(tmp_path, transport=provider) == complete
    assert calls == ["one", "two"]
    result = read_seal(tmp_path / "comparison" / "one" / "assisted.json")
    control = read_seal(tmp_path / "comparison" / "one" / "control.json")
    assert {k: result[k] for k in control} == control
    assert result["critic"]["unverified_citations"] == ["other"]
    assert result["critic"]["advice"]["evidence_ids"] == ["window-1"]
    assert result["critic"]["observed_total_tokens"] == 123


@pytest.mark.parametrize(
    ("changes", "readiness"),
    [
        ({"evidence_ids": ["invented"]}, "conditional_equation_review"),
        ({"equation_ids": ["unbound"]}, "conditional_equation_review"),
        ({"invented_metric": 10}, "conditional_equation_review"),
        ({}, "continue_fitting"),
    ],
)
def test_unverified_claims_stay_out_of_feedback(tmp_path, changes, readiness):
    setup_packets(tmp_path, readiness)
    result = critic.compare(tmp_path, transport=lambda *args: response(changes))
    assert result["statuses"] == {"unusable": 2}
    assert all(r["advice"] is None for r in result["rows"])


def test_unavailable_provider_is_logged_without_replacement(tmp_path):
    setup_packets(tmp_path)
    calls = []

    def failure(*args):
        calls.append(1)
        raise TimeoutError("timed out")

    result = critic.compare(tmp_path, transport=failure)
    assert result["statuses"] == {"provider_failure": 2}
    assert critic.compare(tmp_path, transport=failure) == result
    assert len(calls) == 2


def test_complete_roster_checked_before_spending(tmp_path):
    setup_packets(tmp_path, cohort="development")
    with pytest.raises(ValueError, match="identity differs"):
        critic.compare(tmp_path, transport=lambda *_: pytest.fail("unexpected HTTP"))


def test_freeze_protects_prompt_and_schema(tmp_path):
    seal(tmp_path / "freeze.json", {"fresh_parents": ["one"]})
    with pytest.raises(ValueError, match="policy differs"):
        critic.compare(tmp_path)


def test_wrong_served_model_withholds_reply():
    record = {"status": "responded", "raw_response": response() | {"model": "mock"}}
    assert critic._assessment({}, record)["status"] == "unusable"


def test_hosted_request_goes_on_stdin_and_never_in_shell(monkeypatch):
    sentinel = "fake user field; $(not-a-command)"
    body = critic._body({"public_task": sentinel})

    def captured(argv, *, input, capture_output, text, timeout):
        assert argv[-2] == "js2" and argv[-1].endswith(critic.ENDPOINT)
        assert all(sentinel not in arg for arg in argv)
        assert json.loads(input) == body
        assert body["model"] == critic.MODEL
        assert body["cache"] == {"no-cache": True}
        assert capture_output and text and timeout == 180
        return critic.subprocess.CompletedProcess(argv, 0, '{"model":"gpt-oss-120b"}')

    monkeypatch.setattr(critic.subprocess, "run", captured)
    assert critic._js2_post(critic.ENDPOINT, body, 180)["model"] == critic.MODEL
    with pytest.raises(ValueError, match="unexpected hosted"):
        critic._js2_post("https://wrong.invalid", body, 180)


def test_transfer_controls_have_separate_denominator_and_one_call(tmp_path):
    setup_packets(tmp_path, controls=["transfer"])
    seen = []

    def provider(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"])
        seen.append(packet["parent_id"])
        return (
            response({"evidence_ids": ["absent"]})
            if packet["cohort"] == "transfer_control"
            else response()
        )

    result = critic.compare(tmp_path, transport=provider)
    assert result["paired_feedback_available"] == result["expected"] == 2
    assert [r["parent_id"] for r in result["rows"]] == ["one", "two"]
    transfer = result["transfer_controls"]
    assert transfer["expected"] == 1
    assert transfer["statuses"] == {"unusable": 1}
    assert [r["parent_id"] for r in transfer["rows"]] == ["transfer"]
    assert critic.compare(tmp_path, transport=provider) == result
    assert seen == ["one", "two", "transfer"]


def test_control_cannot_overlap_primary_roster(tmp_path):
    setup_packets(tmp_path, controls=["two"])
    with pytest.raises(ValueError, match="overlapping"):
        critic.compare(tmp_path, transport=lambda *_: pytest.fail("unexpected HTTP"))
