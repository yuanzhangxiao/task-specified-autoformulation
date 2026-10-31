"""Only a complete, matching precision receipt can open warm shared continuation."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting.public_fitting import content_sha256
from tests.test_prefix_additive_reduction import source, wrapped


def case():
    src = source()
    retained = src.start | {
        "a": 1e8,
        "b": -1e8 + 1,
        "trajectory:second:init_z_value": 0.8,
    }
    original, oracle = wrapped(src, retained)
    return original, oracle, retained


def qualify(original, oracle, checkpoint=lambda _: None):
    return polish.qualification(original, oracle, oracle.start, 10, checkpoint)


def test_qualification_replays_saved_anchor_original_ode_and_all_columns():
    original, oracle, retained = case()
    before = content_sha256(original.problem), deepcopy(retained)
    result = qualify(original, oracle)
    assert result["status"] == "complete" and result["passed"], result
    assert result["verification_calls_started"] == result["calls_started"] == 4
    assert result["verification_calls_completed"] == 4
    assert result["verification_calls_failed"] == 0
    assert result["source_lift_prediction_discrepancy"] < 1e-7
    assert result["original_Radau_prediction_discrepancy"] < 1e-7
    check = result["derivative_check"]
    assert check["search_qualified"] and len(check["checked_indices"]) == len(
        oracle.names
    )
    assert result["qualification_identity"]["parameters_sha256"] == content_sha256(
        oracle.start
    )
    search = polish.search(
        oracle,
        oracle.start,
        3,
        8,
        lambda _: None,
        qualification=json.loads(json.dumps(result)),
    )
    assert search["original_incumbent_preserved"]
    assert search["optimizer_start"] == oracle.start
    assert search["qualification_sha256"] == content_sha256(result)
    assert before == (content_sha256(original.problem), retained)


@pytest.mark.parametrize(
    "change", ["missing", "incomplete", "failed", "point", "precision", "coverage"]
)
def test_search_rejects_stale_or_partial_qualification(change, monkeypatch):
    original, oracle, _ = case()
    result = qualify(original, oracle)
    assert result["passed"]
    if change == "missing":
        result.pop("derivative_check")
    elif change == "incomplete":
        result["status"] = "incomplete"
    elif change == "failed":
        result["passed"] = False
    elif change == "point":
        result["qualification_identity"]["parameters_sha256"] = "other"
    elif change == "coverage":
        result["derivative_check"]["checked_indices"] = []
    else:
        original.settings = original.settings.model_copy(
            update={"relative_tolerance": 1e-10}
        )
    monkeypatch.setattr(
        polish.portfolio, "search", lambda *a, **kw: pytest.fail("unqualified search")
    )
    with pytest.raises(ValueError, match="matching incumbent precision"):
        polish.search(oracle, oracle.start, 3, 8, lambda _: None, qualification=result)


@pytest.mark.parametrize("mode", ["source", "Radau"])
def test_numerical_lift_or_original_solver_disagreement_prevents_derivative_work(
    mode, monkeypatch
):
    original, oracle, retained = case()
    evaluate = original.evaluate
    original_point = original.vector(retained)

    def disagree(vector, deadline, **kwargs):
        result, jac = evaluate(vector, deadline, **kwargs)
        if (mode == "source" and np.array_equal(vector, original_point)) or kwargs.get(
            "method"
        ) == mode:
            result = result + 0.01
        return result, jac

    monkeypatch.setattr(original, "evaluate", disagree)
    monkeypatch.setattr(
        polish.recovery,
        "_derivative",
        lambda *a, **k: pytest.fail("derivative before agreement"),
    )
    result = qualify(original, oracle)
    assert not result["passed"] and not result["original_equations_verified"]
    assert "derivative_check" not in result
    assert result["original_incumbent_preserved"]


def test_failed_terminal_checkpoint_revokes_success():
    original, oracle, _ = case()

    def save(value):
        if value.get("passed"):
            raise OSError("not durable")

    result = qualify(original, oracle, save)
    assert not result["passed"] and result["status"] == "incomplete"
    assert result["timing"]["callback_failures"] == 1


def test_interrupt_keeps_spent_call_counts_and_no_search_receipt():
    original, oracle, _ = case()

    def save(value):
        if value["calls_started"]:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as caught:
        qualify(original, oracle, save)
    result = caught.value.execution_receipt
    assert result["calls_started"] == result["calls_completed"] == 1
    assert not result["passed"] and result["no_search_performed"]
