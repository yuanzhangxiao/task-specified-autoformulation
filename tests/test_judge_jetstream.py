"""Authenticated transport, real paired orchestration, isolation and bounded resume."""

import io
import json
import os
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autoformalism.llm import jetstream
from autoformalism.llm.exceptions import LLMProviderError
from autoformalism.rebuttal import judge_jetstream as pilot
from autoformalism.rebuttal import judge_sign_recheck as original
from autoformalism.rebuttal import repair_scientific_judge as judge
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.judge import AtomicJudgeResult
from scripts.judge_jetstream import main
from tests.test_judge_sign_delta import exported_plan
from tests.test_judge_sign_recheck import mock_client

TOKEN = "unit-test-placeholder-not-a-real-credential"


def distinct_pair_plan(tmp_path):
    """Make the synthetic pair asymmetric so orientation caching cannot collapse it."""
    source = exported_plan(tmp_path)
    value = sealed_read(source)
    value.pop("artifact_sha256")
    row = value["reviews"][0]
    old_key = row["historical_request_sha256"]
    row["request"]["candidate"]["state_equations"][0]["rhs"] = "-3*Gp"
    key = original.content_hash(row["request"])
    row["historical_request_sha256"] = key
    row["historical_review"]["request_sha256"] = key
    row["changed_occurrences"] = original.evidence_changes(row["request"])
    for placement in value["placements"]:
        if placement["request_sha256"] == old_key:
            placement["request_sha256"] = key
    path = tmp_path / "distinct.json"
    sealed_write(path, value)
    return path


class Endpoint:
    """Serve known schema-valid responses through the real HTTP adapter boundary."""

    def __init__(self, request):
        mock = mock_client(request)
        self.replies = [
            mock._atomic_responses.popleft(),
            mock._hybrid_responses.popleft(),
            mock._atomic_responses.popleft(),
            mock._hybrid_responses.popleft(),
        ]
        self.requests = []
        self.fail_first = False

    def open(self, request, timeout):
        assert request.full_url == jetstream.ENDPOINT
        assert request.get_header("Authorization") == "Bearer " + TOKEN
        assert timeout == 900
        body = json.loads(request.data)
        self.requests.append(body)
        assert body["model"] == jetstream.MODEL
        if self.fail_first and len(self.requests) == 1:
            content = "invalid JSON"
        else:
            content = self.replies.pop(0).model_dump_json()
        value = {
            "model": jetstream.MODEL,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": content,
                        "reasoning_content": "Separate reasoning.",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "echoed_header_for_redaction_test": TOKEN,
        }
        response = io.BytesIO(json.dumps(value).encode())
        response.status = 200
        return response


def smoke(tmp_path):
    """Test both orientations, real schemas, accounting and zero-call resume."""
    source = distinct_pair_plan(tmp_path)
    before = source.read_bytes()
    root = tmp_path / "jetstream"
    plan = pilot.freeze(source, root)
    assert pilot.freeze(source, root) == plan
    assert pilot.verify(root) == plan
    assert plan["execution"]["served_model_revision"] is None
    endpoint = Endpoint(plan["request"])
    with patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint):
        result = pilot.run(root, lambda: TOKEN)
        assert result["status"] == "reviewed", result
        assert (
            pilot.run(root, lambda: pytest.fail("resume must not ask for a key"))
            == result
        )
    assert len(endpoint.requests) == 4
    # Compare the actual prompts against the unchanged local-runner construction.
    baseline = mock_client(plan["request"])
    judge.perform_review(
        plan["request"],
        tmp_path / "baseline",
        "http://unused",
        client_factory=lambda _: baseline,
    )
    for body, call in zip(endpoint.requests, baseline.calls, strict=True):
        assert body["messages"] == [
            {"role": "system", "content": call["system_prompt"]},
            {"role": "user", "content": call["user_prompt"]},
        ]
        assert body["reasoning_effort"] == "low" and body["temperature"] == 0.2
        assert body["max_tokens"] == 6144 and body["seed"] == 12000
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
    report = pilot.report(root)
    assert report["paired_review_completed"]
    assert report["cost"]["physical_requests_started"] == 4
    assert report["cost"]["observed_total_tokens"] == 480
    assert report["cost"]["usage_complete"]
    assert report["schema_attempt_counts"] == {
        "AtomicJudgeResult": 2,
        "HybridJudgeResult": 2,
    }
    assert not report["calibration_established"] and not report["test_data_opened"]
    assert result["review"]["cost"]["physical_requests"] == 4
    assert source.read_bytes() == before
    assert all(TOKEN not in p.read_text() for p in root.rglob("*.json"))
    assert all(TOKEN not in p.read_text() for p in root.rglob("*.jsonl"))
    return {
        k: report[k]
        for k in (
            "status",
            "cost",
            "schema_attempt_counts",
            "paired_review_completed",
            "calibration_established",
        )
    }


def test_complete_pair_unchanged_prompts_cache_accounting_and_no_key_leak(tmp_path):
    smoke(tmp_path)


def test_schema_retry_keeps_raw_usage_and_existing_seed_policy(tmp_path):
    root = tmp_path / "jetstream"
    plan = pilot.freeze(distinct_pair_plan(tmp_path), root)
    endpoint = Endpoint(plan["request"])
    endpoint.fail_first = True
    with patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint):
        pilot.run(root, lambda: TOKEN)
    report = pilot.report(root)
    assert report["paired_review_completed"]
    assert report["cost"]["physical_requests_started"] == 5
    assert report["cost"]["observed_total_tokens"] == 600
    assert [r["seed"] for r in endpoint.requests] == [12000, 12001, 12000, 12000, 12000]


@pytest.mark.parametrize("code", [400, 401, 302])
def test_http_failure_is_logged_without_credentials_or_automatic_new_budget(
    tmp_path, code
):
    root = tmp_path / "jetstream"
    pilot.freeze(exported_plan(tmp_path), root)

    def denied(*args, **kwargs):
        raise urllib.error.HTTPError(
            jetstream.ENDPOINT,
            code,
            "not accepted",
            {},
            io.BytesIO(json.dumps({"echo": TOKEN}).encode()),
        )

    with patch.object(
        jetstream.urllib.request,
        "build_opener",
        return_value=SimpleNamespace(open=denied),
    ):
        result = pilot.run(root, lambda: TOKEN)
        assert result["status"] == "indeterminate"
        assert pilot.run(root, lambda: pytest.fail("must not retry")) == result
    report = pilot.report(root)
    assert report["cost"]["physical_requests_started"] == 2  # Existing fallback seed.
    assert report["cost"]["usage_missing_events"] == 2
    assert not report["paired_review_completed"]
    assert all(TOKEN not in p.read_text() for p in root.rglob("*.json"))


def test_interrupted_attempt_is_reported_and_never_silently_restarted(tmp_path):
    root = tmp_path / "jetstream"
    pilot.freeze(exported_plan(tmp_path), root)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with (
        patch.object(
            jetstream.urllib.request,
            "build_opener",
            return_value=SimpleNamespace(open=interrupted),
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        pilot.run(root, lambda: TOKEN)
    report = pilot.report(root)
    assert report["cost"]["physical_requests_started"] == 1
    assert report["cost"]["usage_missing_events"] == 1
    assert report["calls"][0]["status"] == "started_without_result"
    result = pilot.run(root, lambda: pytest.fail("no credential required on resume"))
    assert result["status"] == "interrupted"
    assert pilot.report(root)["cost"]["physical_requests_started"] == 1


def test_empty_key_does_not_consume_review_or_make_request(tmp_path):
    root = tmp_path / "jetstream"
    pilot.freeze(exported_plan(tmp_path), root)
    with pytest.raises(ValueError, match="key"):
        pilot.run(root, lambda: "")
    assert not (root / "judge/review_started.json").exists()
    assert pilot.report(root)["cost"]["physical_requests_started"] == 0


@pytest.mark.parametrize(
    "field", ["endpoint", "revision", "request", "placements", "cap", "test"]
)
def test_resealed_plan_changes_rejected(tmp_path, field):
    root = tmp_path / "jetstream"
    value = pilot.freeze(exported_plan(tmp_path), root)
    value.pop("artifact_sha256")
    if field == "endpoint":
        value["execution"]["endpoint"] = "https://another.example/api"
    elif field == "revision":
        value["execution"]["served_model_revision"] = "a" * 40
    elif field == "request":
        value["request"]["seed"] += 1
    elif field == "placements":
        value["placements"] = []
    elif field == "cap":
        value["maximum_physical_requests"] += 1
    else:
        value["test_data_opened"] = True
    (root / "plan.json").unlink()
    sealed_write(root / "plan.json", value)
    with pytest.raises(ValueError, match="identity"):
        pilot.verify(root)


def test_only_affected_reviews_may_be_selected_and_source_is_immutable(tmp_path):
    source = exported_plan(tmp_path)
    with pytest.raises(ValueError, match="affected"):
        pilot.freeze(source, tmp_path / "bad", 1)
    with pytest.raises(ValueError, match="separate"):
        pilot.freeze(source, tmp_path)
    plan = pilot.freeze(source, tmp_path / "good", 0)
    assert plan["selection"]["rule"] == "explicit_index"


def test_no_redirect_with_bearer_authentication():
    request = jetstream.urllib.request.Request(jetstream.ENDPOINT)
    with pytest.raises(urllib.error.HTTPError):
        jetstream._NoRedirect().redirect_request(
            request, None, 302, "redirect", {}, "https://another.example"
        )


def test_transport_cap_is_enforced_before_dispatch(tmp_path):
    transport = jetstream.JetstreamTransport(tmp_path, lambda: TOKEN)
    for i in range(jetstream.MAX_CALLS):
        (tmp_path / f"call-{i:03d}").mkdir()
    with pytest.raises(LLMProviderError, match="cap"):
        transport(
            jetstream.BASE_URL + "/v1/chat/completions",
            {"model": "openai/gpt-oss-120b"},
            900,
        )


def test_transport_rejects_unexpected_urls_before_request(tmp_path):
    transport = jetstream.JetstreamTransport(tmp_path, lambda: pytest.fail("no key"))
    with pytest.raises(ValueError, match="endpoint"):
        transport("https://another.example", {}, 900)


def test_pending_report_does_not_claim_complete_usage(tmp_path):
    root = tmp_path / "jetstream"
    pilot.freeze(exported_plan(tmp_path), root)
    report = pilot.report(root)
    assert report["status"] == "pending"
    assert report["cost"]["physical_requests_started"] == 0
    assert not report["cost"]["usage_complete"]


def test_self_pair_uses_existing_orientation_cache(tmp_path):
    root = tmp_path / "jetstream"
    plan = pilot.freeze(exported_plan(tmp_path), root)
    endpoint = Endpoint(plan["request"])
    with patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint):
        result = pilot.run(root, lambda: TOKEN)
    assert result["status"] == "reviewed"
    assert len(endpoint.requests) == 2
    assert result["review"]["cost"]["cache_hit_events"] == 2


def test_json_escaped_credential_is_redacted():
    encoded = '{"echo": "unit-test-placeholder-not-a-real-credent\\u0069al"}'
    assert pilot.json.dumps(jetstream._redact(json.loads(encoded), TOKEN)) == (
        '{"echo": "[REDACTED]"}'
    )


def test_result_identity_cannot_be_reused_from_another_pilot(tmp_path):
    root = tmp_path / "jetstream"
    pilot.freeze(exported_plan(tmp_path), root)
    sealed_write(root / "result.json", {"identity": "different", "status": "reviewed"})
    with pytest.raises(ValueError, match="another"):
        pilot.run(root, lambda: pytest.fail("no key"))
    with pytest.raises(ValueError, match="another"):
        pilot.report(root)


def atomic_body(count=2, repeats=(), suffix=""):
    """Build a synthetic request; no saved scientific response enters the tests."""
    schema = AtomicJudgeResult.model_json_schema()
    schema["properties"]["signed_occurrence_assessments"]["maxItems"] = 32
    plan = {
        "signed_occurrences": [{"occurrence_id": f"occ_{i}"} for i in range(count)],
        "exact_repeat_candidates": [{"repeat_pair_id": name} for name in repeats],
    }
    return {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "atomic_evidence_plan": plan,
                    }
                )
                + suffix,
            }
        ],
        "response_format": {
            "json_schema": {"name": "AtomicJudgeResult", "schema": schema}
        },
    }


@pytest.mark.parametrize(
    "count,repeats", [(0, ()), (2, ()), (2, ("repeat_1",)), (33, ())]
)
def test_wire_schema_binds_ids_and_counts_without_changing_science(count, repeats):
    original_body = atomic_body(count, repeats, "\n\nRepair diagnostic: example")
    before = json.dumps(original_body)
    payload = jetstream.wire_payload(original_body)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert payload["messages"] == original_body["messages"]
    assert json.dumps(original_body) == before
    for field, expected in (
        ("signed_occurrence_assessments", count),
        ("repeated_contribution_assessments", len(repeats)),
    ):
        assert schema["properties"][field]["minItems"] == expected
        assert schema["properties"][field]["maxItems"] == expected
        assert field in schema["required"]
    if count:
        ids = schema["$defs"]["AtomicSignedOccurrenceAssessment"]["properties"][
            "occurrence_id"
        ]
        assert set(ids["enum"]) == {f"occ_{i}" for i in range(count)}
    if repeats:
        ids = schema["$defs"]["AtomicRepeatedContributionAssessment"]["properties"][
            "repeat_pair_id"
        ]
        assert ids["enum"] == list(repeats)
    original_schema = original_body["response_format"]["json_schema"]["schema"]
    for name in ("ExpectedContributionDirection", "RepeatedContributionRelation"):
        assert schema["$defs"][name] == original_schema["$defs"][name]


@pytest.mark.parametrize(
    "mutation", ["missing_plan", "invalid_json", "duplicate", "oversized"]
)
def test_invalid_atomic_plan_cannot_dispatch(tmp_path, mutation):
    body = atomic_body(257 if mutation == "oversized" else 2)
    if mutation == "invalid_json":
        body["messages"][0]["content"] = "not JSON"
    elif mutation == "missing_plan":
        body["messages"][0]["content"] = "{}"
    elif mutation == "duplicate":
        body["messages"][0]["content"] = body["messages"][0]["content"].replace(
            "occ_1", "occ_0"
        )
    transport = jetstream.JetstreamTransport(tmp_path, lambda: pytest.fail("no key"))
    with pytest.raises(LLMProviderError, match="bind"):
        transport(jetstream.BASE_URL + "/v1/chat/completions", body, 900)
    assert not list(tmp_path.glob("call-*"))


def test_provider_ignoring_wire_ids_is_still_rejected_and_progress_is_visible(tmp_path):
    root = tmp_path / "jetstream"
    plan = pilot.freeze(distinct_pair_plan(tmp_path), root)

    class ExtraRepeatEndpoint(Endpoint):
        def open(self, request, timeout):
            correct = self.replies[0]
            response = super().open(request, timeout)
            if len(self.requests) != 1:
                return response
            self.replies.insert(0, correct)
            value = json.loads(response.read())
            content = json.loads(value["choices"][0]["message"]["content"])
            content["repeated_contribution_assessments"] = [
                {
                    "repeat_pair_id": "invented_repeat",
                    "relation": "same_physical_contribution",
                    "evidence": "Invented unit.",
                }
            ]
            value["choices"][0]["message"]["content"] = json.dumps(content)
            response = io.BytesIO(json.dumps(value).encode())
            response.status = 200
            return response

    endpoint = ExtraRepeatEndpoint(plan["request"])
    messages = []
    with patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint):
        result = pilot.run(root, lambda: TOKEN, messages.append)
    assert result["status"] == "reviewed"
    assert len(endpoint.requests) == 5
    assert any("extra_repeats=['invented_repeat']" in m for m in messages)
    assert sum(": accepted," in m for m in messages) == 4
    assert sum(": sending " in m for m in messages) == 5
    assert TOKEN not in "\n".join(messages)
    assert pilot.report(root)["cost"]["physical_requests_started"] == 5


def test_cli_prints_progress_without_credentials(tmp_path, monkeypatch, capsys):
    root = tmp_path / "jetstream"
    plan = pilot.freeze(distinct_pair_plan(tmp_path), root)
    endpoint = Endpoint(plan["request"])
    monkeypatch.setattr("sys.argv", ["judge_jetstream.py", "run", "--root", str(root)])
    with (
        patch.dict(os.environ, {"AF_JETSTREAM_API_KEY": TOKEN}),
        patch.object(jetstream.urllib.request, "build_opener", return_value=endpoint),
    ):
        main()
    output = capsys.readouterr()
    assert "call-000: sending AtomicJudgeResult" in output.err
    assert ": accepted," in output.err
    assert json.loads(output.out)["paired_review_completed"]
    assert TOKEN not in output.err + output.out


def test_progress_output_failure_does_not_abort_saved_calls(tmp_path):
    transport = jetstream.JetstreamTransport(tmp_path, lambda: TOKEN)
    transport._key = TOKEN
    seen = []
    transport._progress = seen.append
    transport.notify(f"echo {TOKEN}")
    assert seen == ["echo [REDACTED]"]

    def closed(message):
        raise BrokenPipeError

    transport._progress = closed
    transport.notify("diagnostic")
