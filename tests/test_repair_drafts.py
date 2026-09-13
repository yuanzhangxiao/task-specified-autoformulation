"""Adversarial action/withdrawal tests run locally before any GPU request."""

import json

import pytest

from autoformalism.rebuttal.repair_drafts import RepairActionV2, advance, empty_draft
from autoformalism.rebuttal.repair_evidence import compact_numerics
from autoformalism.rebuttal.repair_fit_reporting import fit_outcomes
from autoformalism.rebuttal.repair_transactions import (
    default_initialization,
    request_repair,
)
from tests.test_repair_comparison import CONTEXT, Client, candidate


def run(tmp_path, replies):
    parent = candidate()
    client = Client(replies)
    args = (
        parent,
        default_initialization(parent, CONTEXT),
        CONTEXT,
        {},
        "Public",
        client,
        tmp_path,
        1,
        ("v01",),
    )
    result = request_repair(*args)
    assert request_repair(*args) == result
    assert len(client.requests) == len(result[2]["attempts"])
    return result, client


def test_schema_has_withdrawal_not_keep():
    fields = RepairActionV2.model_json_schema()["properties"]
    assert "withdraw" in fields and "keep" not in fields
    with pytest.raises(ValueError):
        RepairActionV2.model_validate({"scope": "model", "keep": ["f"]})


@pytest.mark.parametrize(
    "action",
    [
        {
            "scope": "model",
            "equations": [{"component": "f", "expression": "tanh(m)"}],
            "remove": ["f"],
        },
        {
            "scope": "model",
            "initializers": [{"state": "m"}],
            "withdraw": [{"kind": "initializer", "target": "m"}],
        },
        {"scope": "no_change", "withdraw": [{"kind": "equation", "target": "f"}]},
        {"scope": "function", "initializers": [{"state": "m"}]},
    ],
)
def test_contradictions_remain_invalid(action):
    with pytest.raises(ValueError):
        RepairActionV2.model_validate(action)


def test_invalid_initializer_is_named_and_can_be_withdrawn(tmp_path):
    first = {
        "scope": "model",
        "equations": [{"component": "f", "expression": "tanh(m)"}],
        "initializers": [{"state": "y", "causal_map": None}],
    }
    (revised, _, draft), client = run(
        tmp_path,
        [
            first,
            {"scope": "model"},
            {"scope": "model", "withdraw": [{"kind": "initializer", "target": "y"}]},
        ],
    )
    assert draft["status"] == "committed"
    assert revised.processes[0].expression == "tanh(m)"
    for request in client.requests[1:]:
        assert request["pending_components"] == []
        assert request["pending_actions"] == [
            {
                "kind": "initializer",
                "target": "y",
                "code": "INITIALIZER_TARGET_NOT_LATENT",
            }
        ]
        assert "f" in request["provisional_edits"]
    assert draft["extras"]["initializers"] == []
    assert len(draft["attempts"]) == 3


@pytest.mark.parametrize(
    "kind,field,entry,target",
    [
        ("mapping", "mappings", {"channel": "unknown", "expression": "y"}, "unknown"),
        ("remove", "remove", "unknown", "unknown"),
        ("equation", "equations", {"component": "y", "expression": "missing"}, "y"),
    ],
)
def test_each_draft_kind_has_explicit_withdrawal(tmp_path, kind, field, entry, target):
    first = {
        "scope": "model",
        "equations": [{"component": "f", "expression": "tanh(m)"}],
    }
    first.setdefault(field, []).append(entry)
    (revised, _, draft), _ = run(
        tmp_path,
        [first, {"scope": "model", "withdraw": [{"kind": kind, "target": target}]}],
    )
    assert draft["status"] == "committed"
    assert revised.processes[0].expression == "tanh(m)"
    assert draft["attempts"][0]["diagnostics"][0]["action_kind"] == kind


def test_withdraw_unknown_does_not_discard_other_edits(tmp_path):
    (revised, _, draft), _ = run(
        tmp_path,
        [{"scope": "model", "withdraw": [{"kind": "equation", "target": "f"}]}],
    )
    assert revised == candidate() and draft["status"] == "exhausted"
    assert draft["attempts"][0]["diagnostics"][0]["code"] == "WITHDRAWAL_NOT_PENDING"


def test_initializer_null_is_not_withdrawal_and_mapping_can_make_it_valid(tmp_path):
    (revised, plan, draft), _ = run(
        tmp_path,
        [
            {"scope": "model", "initializers": [{"state": "y", "causal_map": None}]},
            {"scope": "model", "mappings": [{"channel": "v01", "expression": "y+m"}]},
        ],
    )
    assert draft["status"] == "committed"
    assert "y" in plan.rules
    assert revised.observation_mappings[0].expression == "y+m"


def test_no_change_abandons_entire_invalid_draft(tmp_path):
    (revised, _, draft), _ = run(
        tmp_path,
        [
            {
                "scope": "model",
                "equations": [{"component": "f", "expression": "tanh(m)"}],
                "initializers": [{"state": "y"}],
            },
            {"scope": "no_change"},
        ],
    )
    assert revised == candidate() and draft["status"] == "no_change"
    assert not any(draft["edits"].values())


def test_partial_resume_does_not_repeat_first_request(tmp_path):
    from autoformalism.llm.staged_topology import DeferredCall

    first = {
        "scope": "model",
        "equations": [{"component": "f", "expression": "tanh(m)"}],
        "initializers": [{"state": "y"}],
    }

    class PausingClient(Client):
        def call(self, **kw):
            if kw["attempt"] == 1:
                raise DeferredCall("pause")
            return super().call(**kw)

    parent = candidate()
    args = (parent, default_initialization(parent, CONTEXT), CONTEXT, {}, "Public")
    with pytest.raises(DeferredCall):
        request_repair(*args, PausingClient([first, {}]), tmp_path, 1, ("v01",))
    client = Client(
        [
            first,
            {"scope": "model", "withdraw": [{"kind": "initializer", "target": "y"}]},
        ]
    )
    _, _, result = request_repair(*args, client, tmp_path, 1, ("v01",))
    assert result["status"] == "committed" and len(client.requests) == 1
    assert (
        json.loads((tmp_path / "transaction.json").read_text())["attempts"][0][
            "response"
        ]
        == first
    )


def test_model_draft_does_not_relax_new_function_scope():
    parent = candidate()
    plan = default_initialization(parent, CONTEXT)
    _, _, draft, _ = advance(
        parent,
        plan,
        CONTEXT,
        empty_draft(),
        RepairActionV2.model_validate(
            {"scope": "model", "initializers": [{"state": "y"}]}
        ),
    )
    _, _, _, diagnostics = advance(
        parent,
        plan,
        CONTEXT,
        draft,
        RepairActionV2.model_validate(
            {
                "scope": "function",
                "equations": [{"component": "m", "expression": "rate*m+gain*u01"}],
            }
        ),
    )
    assert diagnostics[0]["code"] == "EXPLICIT_MODEL_SCOPE_REQUIRED"


def test_report_uses_stages_and_does_not_claim_selected_convergence():
    raw = {
        "method": "feasibility_then_refinement",
        "optimizer_native_success": False,
        "optimizer_success": False,
        "parameters": {"a": 2},
        "production_training_rollout_verified": True,
        "stages": [
            {
                "mode": "sensitivity",
                "result": {
                    "optimizer_native_success": True,
                    "optimizer_success": True,
                    "parameters": {"a": 1},
                },
            }
        ],
    }
    report = fit_outcomes({"status": "complete", "refinement": raw})
    assert report["native_optimizer_success"] is True
    assert report["verified_optimizer_success"] is False
    assert report["finite_candidate"] is True
    evidence = compact_numerics(raw)
    assert "optimizer_success" not in evidence
    assert evidence["any_stage_native_success"] is True
    raw["parameters"] = {"a": 1}
    assert fit_outcomes({"refinement": raw})["verified_optimizer_success"] is True


def test_report_missing_stages_is_unknown_not_failure():
    report = fit_outcomes(
        {"refinement": {"method": "feasibility_then_refinement", "stages": []}}
    )
    assert report["native_optimizer_success"] is None
    assert report["verified_optimizer_success"] is None
