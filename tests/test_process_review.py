"""Optional review must preserve scientific inventory and exact call resume."""

import json

import pytest

from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import process_review as pr


@pytest.fixture
def inputs():
    brief = PublicScientificBrief(
        scientific_context="Two stores exchange material; y is observed.",
        public_variables=[{"name": "y", "data_role": "target"}],
        requirements=[],
    )
    inventory = tuple(
        ScientificVariable(name=n, definition="differential", scientific_role="storage")
        for n in ("x", "y")
    )
    suggestion = {
        "name": "q",
        "scientific_role": "transfer",
        "drivers": ["x"],
        "consumers": ["x", "y"],
    }
    return brief, inventory, suggestion


def client(root, response, calls):
    def transport(*args):
        calls.append(args)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(response)}}
            ],
            "usage": {"total_tokens": 30},
        }

    return BudgetedRepairClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="test",
        seed=0,
        base_url="offline",
        transport=transport,
    )


def test_addition_and_resume_without_repeat_call(tmp_path, inputs):
    brief, inventory, suggestion = inputs
    calls = []
    c = client(tmp_path, {"processes": [suggestion]}, calls)
    result, audit = pr.review(
        brief, brief.model_dump(mode="json"), inventory, c, tmp_path
    )
    assert result[:2] == inventory and result[2].definition == "algebraic"
    assert audit["status"] == "accepted" and audit["added_names"] == ["q"]
    assert "consumers: x, y" in result[2].scientific_role
    assert pr.review(brief, brief.model_dump(mode="json"), inventory, c, tmp_path) == (
        result,
        audit,
    )
    assert len(calls) == 1
    changed = brief.model_copy(update={"scientific_context": "different"})
    with pytest.raises(ValueError, match="checkpoint differs"):
        pr.review(changed, changed.model_dump(mode="json"), inventory, c, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "collision",
        "unknown",
        "self",
        "duplicate",
        "dynamic",
        "malformed",
        "capacity",
    ],
)
def test_optional_failure_never_changes_inventory(tmp_path, inputs, change):
    brief, inventory, p = inputs
    reply = {"processes": [p]}
    if change == "empty":
        reply = {"processes": []}
    elif change == "collision":
        p["name"] = "y"
    elif change == "unknown":
        p["drivers"] = ["unavailable"]
    elif change == "self":
        p["consumers"] = ["q", "y"]
    elif change == "duplicate":
        reply["processes"] = [p, p]
    elif change == "dynamic":
        p["definition"] = "differential"
    elif change == "malformed":
        reply = {"equation": "bad"}
    elif change == "capacity":
        brief = brief.model_copy(
            update={
                "limits": brief.limits.model_copy(update={"generated_variables": 2})
            }
        )
    actual, audit = pr.review(
        brief,
        brief.model_dump(mode="json"),
        inventory,
        client(tmp_path, reply, []),
        tmp_path,
    )
    assert actual == inventory
    assert audit["status"] == ("empty" if change == "empty" else "skipped_invalid")


def test_drain_is_resumable_and_not_a_scientific_failure(tmp_path, inputs):
    brief, inventory, _ = inputs
    c = client(tmp_path, {}, [])
    c.can_start = lambda: False
    with pytest.raises(DeferredCall):
        pr.review(brief, brief.model_dump(mode="json"), inventory, c, tmp_path)
    assert not (tmp_path / "process_review.json").exists()


def test_delivery_failure_and_budget_skip(tmp_path, inputs):
    brief, inventory, _ = inputs
    c = client(tmp_path, {}, [])
    c.transport = lambda *a: (_ for _ in ()).throw(TimeoutError("offline"))
    actual, audit = pr.review(
        brief, brief.model_dump(mode="json"), inventory, c, tmp_path
    )
    assert actual == inventory and audit["status"] == "skipped_invalid"
    c.settings = c.settings.model_copy(update={"maximum_requests": 1})
    actual, audit = pr.review(
        brief, brief.model_dump(mode="json"), inventory, c, tmp_path / "second"
    )
    assert actual == inventory and "budget" in audit["error"]
