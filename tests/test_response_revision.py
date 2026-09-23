"""Tokenizer limits, caching, delivery classification and protocol isolation."""

import io as stream_io
import json
import subprocess
import sys
import urllib.error

import pytest

from autoformalism.llm.response_revision import (
    PromptPreflightError,
    ResponseRevisionClient,
    diagnostic_transport,
)
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.search import response_revision as revision
from autoformalism.search.review_revision_multi import ScientificRevision
from scripts.review_response import latest_source
from scripts.smoke_review_multi import fixture
from tests.test_response_evidence import response_example


def reply(raw):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}],
        "usage": {"total_tokens": 100},
    }


def client(root, generation, tokenize):
    return ResponseRevisionClient(
        settings=io.StagedModelSettings(max_output_tokens=8192),
        base_url="http://offline",
        directory=root,
        namespace="fixture",
        seed=0,
        transport=generation,
        token_transport=tokenize,
    )


def user():
    bundle, packet, params, evidence = response_example()
    return revision.payload(bundle, packet, params, evidence)


def call(subject, payload=None):
    return subject.call(
        system=revision.SYSTEM_PROMPT,
        user=json.dumps(payload or user()),
        response_model=ScientificRevision,
        step="revision",
        attempt=0,
    )


def test_packing_checks_reserved_output_and_exact_resume(tmp_path):
    generated, checked = [], []

    def tokenize(url, body, timeout):
        assert url.endswith("/tokenize")
        assert body["add_generation_prompt"] is True
        checked.append(body)
        value = json.loads(body["messages"][1]["content"])
        compact = len(value["training_evidence"]["examples"]) == 3
        return {"count": 23999 if compact else 28000, "max_model_len": 32768}

    def generation(url, body, timeout):
        generated.append(body)
        return reply({"hypothesis": "A test hypothesis"})

    subject = client(tmp_path, generation, tokenize)
    result = call(subject)
    assert len(generated) == 1 and len(checked) >= 2
    assert result["prompt_preflight"]["input_limit"] == 24000
    assert result["prompt_preflight"]["input_tokens"] + 8192 + 512 <= 32768
    assert (
        len(
            json.loads(generated[0]["messages"][1]["content"])["training_evidence"][
                "examples"
            ]
        )
        == 3
    )
    assert call(client(tmp_path, generation, tokenize)) == result
    assert len(generated) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1  # tokenizer logs are separate


@pytest.mark.parametrize(
    "response",
    [
        {"count": 40000, "max_model_len": 32768},
        {"count": 2000, "max_model_len": 4096},
        {"count": True, "max_model_len": 32768},
        {},
    ],
)
def test_unfit_or_unverified_token_budget_never_calls_generation(tmp_path, response):
    def never(*_):
        pytest.fail("oversized/unverified prompt sent for generation")

    subject = client(tmp_path, never, lambda *_: response)
    with pytest.raises(PromptPreflightError):
        call(subject)
    assert not subject.records
    assert not list(tmp_path.glob("*.json"))


def test_http_error_body_is_bounded_and_preserved(monkeypatch):
    def error(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://offline",
            400,
            "Bad Request",
            {},
            stream_io.BytesIO(b"context exceeded " + b"x" * 10000),
        )

    monkeypatch.setattr("urllib.request.urlopen", error)
    with pytest.raises(ValueError, match="HTTP 400: context exceeded") as caught:
        diagnostic_transport("http://offline", {}, 10)
    assert len(str(caught.value)) <= 8210


def continued(tmp_path):
    source, root = tmp_path / "source", tmp_path / "continued"
    fixture(source, fitted_only=True)
    plan = continuation.prepare(source, root, 2, 2, protocol=io.RESPONSE_PROTOCOL)
    return source, root, plan


def test_new_protocol_pinned_import_latest_common_and_request_failure_preserve(
    tmp_path, monkeypatch
):
    source, root, plan = continued(tmp_path)
    assert latest_source(source) == 2 and latest_source(root) == 2
    assert plan["config"]["fit_profile"] == "collocation-multi-target-v1"
    assert plan["continuation"]["evidence_policy"] == "training-response-evidence-1"
    task = plan["tasks"][0]
    evidence = response_example()[3]
    monkeypatch.setattr(pipeline, "response_for_selected", lambda *_: evidence)
    calls = []

    def failure(*_):
        calls.append(True)
        raise ValueError("HTTP 400: test failure")

    subject = pipeline._client(
        root, plan, task, 1, "http://offline", lambda: True, failure
    )
    subject.token_transport = lambda *_: {"count": 5000, "max_model_len": 32768}
    proposal = pipeline.propose_one(root, plan, task, 1, subject)
    assert proposal["status"] == "provider_request_failed" and len(calls) == 1
    result = pipeline.fit_one(root, plan, task, 1)
    assert result["selected"] == io.read_round(root, task, 0)["selected"]
    assert result["trial"] is None and result["closed"] is False
    assert pipeline.propose_one(root, plan, task, 1, None) == proposal
    assert pipeline.fit_one(root, plan, task, 1) == result
    finished = subprocess.run(
        [
            sys.executable,
            "scripts/review_deadline.py",
            "finish-round",
            "--root",
            str(root),
            "--round",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        finished.returncode != 0 and "automatic continuation stopped" in finished.stderr
    )


def test_only_displayed_refs_verify_and_bad_equations_get_feedback(tmp_path):
    _, root, plan = continued(tmp_path)
    task = plan["tasks"][0]
    parent = io.read_round(root, task, 0)
    evidence = response_example()[3]
    count = []

    def generation(url, body, timeout):
        shown = json.loads(body["messages"][1]["content"])
        count.append(shown)
        if len(count) == 1:
            return reply(
                {
                    "hypothesis": "An invalid equation",
                    "equations": [{"component": "g", "expression": "unknown*g"}],
                }
            )
        assert "available_evidence_refs" not in shown["retry_feedback"]
        return reply(
            {
                "hypothesis": "Output offset",
                "evidence_refs": ["R001", "E001"],
                "output_mappings": [{"channel": "U", "expression": "disposal+offset"}],
                "new_parameters": [{"name": "offset"}],
            }
        )

    subject = client(
        tmp_path / "calls",
        generation,
        lambda *_: {"count": 5000, "max_model_len": 32768},
    )
    result = revision.propose(plan, task, parent, subject, evidence)
    assert result["status"] == "committed" and len(count) == 2
    audit = result["decision"]["provenance"]["citation_audit"]
    assert len(audit["valid_evidence_ids"]) == 1
    assert audit["unresolved_references"] == ["E001"]


def test_new_protocol_rejects_unfitted_import_without_mutating_old(tmp_path):
    source, root = tmp_path / "source", tmp_path / "continued"
    fixture(source)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    with pytest.raises(ValueError, match="fitted incumbent"):
        continuation.prepare(source, root, 2, 3, protocol=io.RESPONSE_PROTOCOL)
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
