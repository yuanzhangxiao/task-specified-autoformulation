"""Answering LLM-SR's own protocol from an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import urllib.error
from email.message import Message

import pytest

from autoformalism.rebuttal.llm_call_log import CallLog
from autoformalism.rebuttal.llm_sr_shim import (
    DEFAULT_MAX_TOKENS,
    ShimAccounting,
    UpstreamEndpointError,
    complete,
    join_split_header,
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


def _reply(*contents, reasons=None, usage=None) -> dict:
    """A chat completions reply holding these message contents."""
    reasons = reasons or ["stop"] * len(contents)
    choices = [
        {"index": i, "message": {"role": "assistant", "content": content},
         "finish_reason": reason}
        for i, (content, reason) in enumerate(zip(contents, reasons, strict=True))
    ]
    return {"choices": choices, "usage": usage or {"total_tokens": 9000}}


def test_the_published_payload_becomes_one_batched_chat_request() -> None:
    """Their engine puts the prompt into the chat template as one user turn.

    repeat_prompt is how many skeletons one prompt should yield. Sending the
    prompt as raw completion text instead, as this shim did until 2026-10-05,
    skips the template: gpt-oss then continued it with prose.
    """
    request = translate_request(UPSTREAM_PAYLOAD, "openai/gpt-oss-120b")
    assert request["n"] == 4
    assert request["messages"] == [
        {"role": "user", "content": "complete the equation"}
    ]
    assert "prompt" not in request
    assert request["max_tokens"] == DEFAULT_MAX_TOKENS == 512


def test_a_declared_limit_replaces_the_engine_default_only() -> None:
    """A plan's limit stands in for their engine's 512, never for a sent value."""
    request = translate_request(UPSTREAM_PAYLOAD, "m", max_tokens=4096)
    assert request["max_tokens"] == 4096
    payload = {**UPSTREAM_PAYLOAD, "params": {"max_new_tokens": 256}}
    assert translate_request(payload, "m", max_tokens=4096)["max_tokens"] == 256


def test_a_caching_gateway_is_asked_to_generate_afresh() -> None:
    """Their engine samples anew for every request; a replayed answer would not.

    LiteLLM reads its switch from the body; a Cache-Control header it ignored.
    """
    request = translate_request(UPSTREAM_PAYLOAD, "m", bypass_cache=True)
    assert request["cache"] == {"no-cache": True}
    assert "cache" not in translate_request(UPSTREAM_PAYLOAD, "m")


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


def test_the_reply_is_the_list_of_answers() -> None:
    """Each sample is a message's content: the answer, not the reasoning."""
    reply = _reply("  a", "b")
    reply["choices"][1]["message"]["reasoning_content"] = "we need to think"
    assert translate_response(reply) == ["  a", "b"]


def test_an_answer_cut_off_before_it_began_is_an_empty_sample() -> None:
    """A choice with no content still counts as a sample, as upstream's would."""
    reply = _reply("return params[0]", None, reasons=["stop", "length"])
    assert translate_response(reply) == ["return params[0]", ""]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        _reply("", None, reasons=["length", "length"]),
        {"choices": [{"text": "a raw completion, not a chat reply"}]},
    ],
)
def test_an_empty_answer_is_an_error_not_an_empty_sample(payload: dict) -> None:
    """LLM-SR retries forever on exceptions, so silence must not look like data.

    Returning [] here would feed the sampler nothing indefinitely; raising at
    least records the reason where a human can find it.
    """
    with pytest.raises(UpstreamEndpointError):
        translate_response(payload)


def test_an_empty_answer_says_why_the_model_stopped() -> None:
    with pytest.raises(UpstreamEndpointError, match="length, length"):
        translate_response(_reply("", None, reasons=["length", "length"]))


# A reply the way gpt-oss writes one: prose, a fenced program whose header runs
# over several lines, and prose after it.
SPLIT_REPLY = """Here is an improved version.

```python
import numpy as np

def equation(
    E: np.ndarray,  # endogenous production (mg/kg/min)
    EGP: np.ndarray,
    Uii: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    \"\"\"Improved version of `equation_v0`.\"\"\"
    k_egp = params[0]
    return k_egp * EGP - params[1] * Uii + params[2] * E
```

**Why it works.** The production term is linear.
"""


def _body_as_upstream_reads_it(reply: str) -> list[str]:
    """Upstream's _extract_body: every line after the first starting with def."""
    lines = reply.splitlines()
    start = next(i for i, line in enumerate(lines) if line[:3] == "def")
    return lines[start + 1 :]


def test_a_split_header_is_read_as_one_line() -> None:
    """Upstream takes the rest of a split header as body, and keeps nothing.

    Joined, the body upstream reads starts where the model's body starts, and
    nothing else in the reply changes.
    """
    assert _body_as_upstream_reads_it(SPLIT_REPLY)[0] == (
        "    E: np.ndarray,  # endogenous production (mg/kg/min)"
    )
    joined = join_split_header(SPLIT_REPLY)
    assert (
        "def equation(E: np.ndarray, EGP: np.ndarray, Uii: np.ndarray, "
        "params: np.ndarray) -> np.ndarray:"
    ) in joined.splitlines()
    body = _body_as_upstream_reads_it(joined)
    assert body[0] == '    """Improved version of `equation_v0`."""'
    assert body[2] == "    return k_egp * EGP - params[1] * Uii + params[2] * E"
    before = SPLIT_REPLY.split("def equation(")[0]
    after = SPLIT_REPLY.split(") -> np.ndarray:\n")[1]
    assert joined.startswith(before) and joined.endswith(after)


@pytest.mark.parametrize(
    "reply",
    [
        "def equation(x, params):\n    return params[0] * x\n",
        "    k = params[0]\n    return k * x\n",
        "no program here at all",
        "def equation(\n    x: np.ndarray,\n    params",  # cut off mid-header
        "define the rate first.\n\ndef equation(\n    x,\n    params,\n):\n"
        "    return x\n",
        "def equation(x, params): return params[0] * x\n    y = 1\n",
    ],
    ids=["one-line", "no-header", "prose", "unclosed", "prose-def", "inline-body"],
)
def test_anything_but_a_split_header_reaches_upstream_unchanged(reply: str) -> None:
    """Upstream sees exactly what it would have seen, including its own failures.

    A prose line starting with "def" is the first such line, as upstream reads
    it, so the header after it is not joined either.
    """
    assert join_split_header(reply) == reply


def test_only_the_header_upstream_reads_is_joined() -> None:
    reply = (
        "def equation(\n    x,\n    params,\n):\n    return h(x)\n\n"
        "def h(\n    x,\n):\n    return x\n"
    )
    joined = join_split_header(reply)
    assert joined.startswith("def equation(x, params):\n    return h(x)\n")
    assert joined.endswith("def h(\n    x,\n):\n    return x\n")


def test_headers_are_joined_only_when_declared() -> None:
    reply = _reply(SPLIT_REPLY)
    assert translate_response(reply) == [SPLIT_REPLY]
    assert translate_response(reply, join_headers=True) == [
        join_split_header(SPLIT_REPLY)
    ]


def _answering(monkeypatch, reply: dict, headers: dict | None = None) -> list:
    """Serve `reply` to every request, keeping what was sent."""
    sent: list = []
    received = Message()
    for name, value in (headers or {}).items():
        received[name] = value

    class _Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps(reply).encode()

        headers = received

    def urlopen(request, *args, **kwargs):
        sent.append(request)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return sent


def test_a_round_trip_records_what_was_asked(monkeypatch, tmp_path) -> None:
    reply = _reply(*["return params[0] * x"] * 3, "", reasons=["stop"] * 3 + ["length"])
    sent = _answering(monkeypatch, reply)
    accounting = ShimAccounting(log=CallLog(tmp_path / "calls.jsonl"))
    answer = complete(
        UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:8000", model="m",
        max_tokens=4096, accounting=accounting,
    )
    assert answer["content"] == ["return params[0] * x"] * 3 + [""]
    assert sent[0].full_url == "http://127.0.0.1:8000/v1/chat/completions"
    body = json.loads(sent[0].data)
    assert body["messages"][0]["content"] == "complete the equation"
    assert body["max_tokens"] == 4096
    assert accounting.requests == 1 and accounting.samples == 4
    assert accounting.failures == 0
    event = json.loads((tmp_path / "calls.jsonl").read_text())
    assert event["usage"] == {"total_tokens": 9000}
    assert event["finish_reasons"] == ["stop", "stop", "stop", "length"]
    assert event["cache_hit"] is False
    assert "cache" not in body


def test_a_replayed_answer_is_logged_as_a_cache_hit(monkeypatch, tmp_path) -> None:
    """LiteLLM marks a replayed answer, so a bypass that stops working shows.

    The accounting rule counts a cache hit apart from requests that cost work.
    """
    from autoformalism.rebuttal.phase_b_d3 import accounting as rule

    sent = _answering(
        monkeypatch, _reply("return x"), {"x-litellm-cache-key": "71bb0880"}
    )
    log = tmp_path / "calls.jsonl"
    complete(UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:1", model="m",
             bypass_cache=True, accounting=ShimAccounting(log=CallLog(log)))
    assert json.loads(sent[0].data)["cache"] == {"no-cache": True}
    assert json.loads(log.read_text())["cache_hit"] is True
    assert rule(log)["cache_hits"] == 1 and rule(log)["physical_requests"] == 0


def test_an_empty_reply_is_counted_with_what_it_cost(monkeypatch, tmp_path) -> None:
    """Every reply cut off still spent tokens, so it is logged before it fails."""
    _answering(monkeypatch, _reply("", "", reasons=["length", "length"]))
    accounting = ShimAccounting(log=CallLog(tmp_path / "calls.jsonl"))
    with pytest.raises(UpstreamEndpointError, match="only empty completions"):
        complete(UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:1", model="m",
                 accounting=accounting)
    assert accounting.reasons == {"empty reply": 1} and accounting.samples == 0
    event = json.loads((tmp_path / "calls.jsonl").read_text())
    assert event["event"] == "llm_response"
    assert event["usage"] == {"total_tokens": 9000}


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


def test_a_refusal_carries_the_servers_explanation(monkeypatch) -> None:
    """A bare status line cannot tell an outage from a wrong address."""
    import io

    def _refuse(request, *args, **kwargs):
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", {},
            io.BytesIO(b'{"error": {"message": "the authentication database '
                       b'is temporarily unreachable"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    accounting = ShimAccounting()
    with pytest.raises(UpstreamEndpointError, match="authentication database"):
        complete(UPSTREAM_PAYLOAD, base_url="http://127.0.0.1:1", model="m",
                 accounting=accounting)
    assert accounting.reasons == {"HTTPError 503": 1}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"detail": "Access restricted"}', "Access restricted"),
        (b"<html>\n  gateway   down\n</html>", "<html> gateway down </html>"),
        (b"", ""),
    ],
)
def test_an_explanation_is_read_from_any_body_shape(body: bytes, expected: str) -> None:
    import io

    from autoformalism.rebuttal.llm_sr_shim import http_error_detail

    error = urllib.error.HTTPError(
        "http://x", 401, "Unauthorized", {}, io.BytesIO(body)
    )
    assert http_error_detail(error) == expected
