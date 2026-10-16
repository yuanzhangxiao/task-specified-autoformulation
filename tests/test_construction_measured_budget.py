"""Serving-token reservations preserve caps, physical costs and cached resume."""

import pytest

from autoformalism.llm.construction import (
    MEASURED_RESERVATION,
    ConstructionClient,
    PromptPreflightError,
)
from autoformalism.llm.response_revision import SAFETY_TOKENS
from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.search.construction_ledger import DraftPatch


def client(tmp_path, send, *, measured=True, tokens=1100, tokenizer=None):
    return ConstructionClient(
        directory=tmp_path / "calls",
        base_url="http://offline",
        namespace="test",
        seed=0,
        settings=StagedModelSettings(
            max_output_tokens=256,
            maximum_total_tokens=tokens,
        ),
        transport=send,
        token_transport=tokenizer
        or (lambda *args: {"count": 100, "max_model_len": 32768}),
        reservation_policy=MEASURED_RESERVATION if measured else None,
    )


def call(c, attempt=0):
    return c.call(
        system="rules",
        user="large context " * 500,
        response_model=DraftPatch,
        step="repair",
        attempt=attempt,
    )


def test_measured_admission_fits_where_byte_reservation_does_not(tmp_path):
    sent = []

    def send(*args):
        sent.append(args)
        return {"usage": {"total_tokens": 200}}

    legacy = client(tmp_path / "legacy", send, measured=False)
    with pytest.raises(RepairBudgetExceeded):
        call(legacy)
    assert not sent
    measured = client(tmp_path / "measured", send)
    measured.budget_diagnostics_enabled = True
    record = call(measured)
    assert len(sent) == 1
    assert record["budget_charge"] == 200
    assert record["request"]["budget_reservation_policy"] == MEASURED_RESERVATION
    receipt = measured.last_budget_diagnostic
    assert receipt["reservation"] == 100 + 256 + SAFETY_TOKENS
    assert receipt["request_bytes"] > receipt["reservation"]
    assert receipt["reservation_basis"] == MEASURED_RESERVATION
    assert "budget_reservation_policy" not in legacy.request_policy()


def test_missing_usage_remains_reserved_and_resume_does_not_resend(tmp_path):
    sent = []

    def send(*args):
        sent.append(args)
        raise TimeoutError("unknown physical outcome")

    record = call(client(tmp_path, send))
    assert record["budget_charge"] == 100 + 256 + SAFETY_TOKENS
    assert record["budget_charge_basis"] == MEASURED_RESERVATION
    restored = client(tmp_path, send)
    assert call(restored) == record
    with pytest.raises(RepairBudgetExceeded):
        call(restored, attempt=1)
    assert len(sent) == 1


@pytest.mark.parametrize("tokens", [256, 867])
def test_measured_policy_does_not_bypass_token_ceiling(tmp_path, tokens):
    c = client(tmp_path, lambda *args: pytest.fail("must not generate"), tokens=tokens)
    with pytest.raises(RepairBudgetExceeded):
        call(c)
    assert c.records == []


def test_tokenizer_failure_still_fails_closed(tmp_path):
    c = client(
        tmp_path,
        lambda *args: pytest.fail("must not generate"),
        tokenizer=lambda *args: {"count": 100, "max_model_len": 65536},
    )
    with pytest.raises(PromptPreflightError):
        call(c)
