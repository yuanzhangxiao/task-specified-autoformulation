"""Answering LLM-SR's own protocol from an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import urllib.error

import pytest

from autoformalism.rebuttal.llm_sr_shim import (
    DEFAULT_MAX_TOKENS,
    ShimAccounting,
    UpstreamEndpointError,
    complete,
    translate_request,
    translate_response,
)

# What LLM-SR's sampler actually sends: explicit Nones, and no max_new_tokens.
UPSTREAM_PAYLOAD = {
    "prompt": "complete the equation",
    "repeat_prompt": 4,
    "params": {
        "do_sample": True,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "add_special_tokens": False,
        "skip_special_tokens": True,
    },
}


def test_the_published_payload_becomes_one_batched_request() -> None:
    """repeat_prompt is how many skeletons one prompt should yield."""
    request = translate_request(UPSTREAM_PAYLOAD, "openai/gpt-oss-120b")
    assert request["n"] == 4
    assert request["prompt"] == "complete the equation"
    assert request["max_tokens"] == DEFAULT_MAX_TOKENS


def test_explicit_nones_mean_the_served_default_not_a_value() -> None:
    """Their engine reads params.get(name, default) and the keys are present.

    So the argparse defaults of 0.8/30/0.9 never apply upstream either; the
    Nones reach the generator and its own defaults take over. Forwarding a
    literal None would instead be rejected or coerced, changing the search.
    """
    request = translate_request(UPSTREAM_PAYLOAD, "m")
    for name in ("temperature", "top_k", "top_p"):
        assert name not in request


def test_sampling_controls_that_were_set_are_forwarded() -> None:
    payload = {**UPSTREAM_PAYLOAD, "params": {"temperature": 0.8, "top_p": 0.9,
                                              "top_k": 30, "max_new_tokens": 256}}
    request = translate_request(payload, "m")
    assert request["temperature"] == 0.8
    assert request["top_p"] == 0.9
    assert request["top_k"] == 30
    assert request["max_tokens"] == 256


@pytest.mark.parametrize(
    "payload",
    [{}, {"prompt": ""}, {"prompt": "p", "repeat_prompt": 0},
     {"prompt": "p", "repeat_prompt": "four"}],
)
def test_a_malformed_request_is_refused_rather_than_guessed(payload: dict) -> None:
    with pytest.raises(UpstreamEndpointError):
        translate_request(payload, "m")


def test_the_reply_is_the_list_of_completions() -> None:
    assert translate_response(
        {"choices": [{"text": "  a"}, {"text": "b"}]}
    ) == ["  a", "b"]


@pytest.mark.parametrize(
    "payload", [{}, {"choices": []}, {"choices": [{"text": ""}, {"text": ""}]}]
)
def test_an_empty_answer_is_an_error_not_an_empty_sample(payload: dict) -> None:
    """LLM-SR retries forever on exceptions, so silence must not look like data.

    Returning [] here would feed the sampler nothing indefinitely; raising at
    least records the reason where a human can find it.
    """
    with pytest.raises(UpstreamEndpointError):
        translate_response(payload)


def test_a_round_trip_records_what_was_asked(monkeypatch) -> None:
    class _Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps(
                {"choices": [{"text": "return params[0] * x"}] * 4}
            ).encode()

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _Response()
    )
    accounting = ShimAccounting()
    answer = complete(
        UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:8000", model="m",
        accounting=accounting,
    )
    assert answer["content"] == ["return params[0] * x"] * 4
    assert accounting.requests == 1 and accounting.samples == 4
    assert accounting.failures == 0


def test_an_unreachable_endpoint_is_reported_with_its_reason(monkeypatch) -> None:
    """A fault that is not surfaced becomes a job spinning to its walltime."""
    def _raise(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    accounting = ShimAccounting()
    with pytest.raises(UpstreamEndpointError, match="connection refused"):
        complete(UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:1", model="m",
                 accounting=accounting)
    assert accounting.failures == 1
    assert accounting.reasons == {"URLError": 1}
