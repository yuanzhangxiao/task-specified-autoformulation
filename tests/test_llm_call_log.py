"""One token-accounting rule across every method that calls a model.

phase_b_d3.accounting already counts physical requests including retries,
separates cache hits, sums observed tokens, and counts requests whose usage
the provider did not report. Only D3 produced the log it reads; the vendored
campaigns counted their own requests in their own shapes, which cannot sit in
a table beside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoformalism.rebuttal.llm_call_log import CallLog, usage_tokens
from autoformalism.rebuttal.phase_b_d3 import accounting


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"usage": {"total_tokens": 1200}}, 1200),
        ({"usage": {"prompt_tokens": 300, "completion_tokens": 40}}, 340),
        ({"usage": {"input_tokens": 10, "output_tokens": 5}}, 15),
        ({}, None),
        ({"usage": None}, None),
        ({"usage": {}}, None),
    ],
)
def test_usage_is_read_under_the_names_the_apis_use(payload, expected) -> None:
    """Responses and completions spell it differently; both are real."""
    assert usage_tokens(payload) == expected


def test_an_object_response_is_read_too() -> None:
    """The OpenAI client returns an object, not a dict."""
    import types

    response = types.SimpleNamespace(usage=types.SimpleNamespace(total_tokens=42))
    assert usage_tokens(response) == 42


def test_the_shared_rule_reads_what_the_log_writes(tmp_path: Path) -> None:
    """The point of the format: one rule, every method."""
    log = CallLog(tmp_path / "llm_calls.jsonl")
    log.response({"usage": {"total_tokens": 1200}})
    log.response({"usage": {"prompt_tokens": 300, "completion_tokens": 40}})
    log.response({})
    log.failure("connection refused")

    assert accounting(tmp_path / "llm_calls.jsonl") == {
        "physical_requests": 4,
        "observed_tokens": 1540,
        # the silent response and the failure, visible rather than counted as
        # zero tokens, which would make the method look cheap
        "unknown_usage_requests": 2,
        "cache_hits": 0,
    }


def test_a_missing_log_reads_as_nothing_rather_than_failing(tmp_path: Path) -> None:
    assert accounting(tmp_path / "absent.jsonl") == {
        "physical_requests": 0, "observed_tokens": 0,
        "unknown_usage_requests": 0, "cache_hits": 0,
    }


def test_a_failed_call_is_still_counted(tmp_path: Path) -> None:
    """A provider that errored still did work, and a method that retries a
    lot should not look cheaper than one that succeeds first time."""
    log = CallLog(tmp_path / "llm_calls.jsonl")
    log.failure("timeout", attempts=3)
    counted = accounting(tmp_path / "llm_calls.jsonl")
    assert counted["physical_requests"] == 3
    assert counted["unknown_usage_requests"] == 3
