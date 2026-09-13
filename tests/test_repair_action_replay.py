"""Frozen-context replay includes failures and reports legacy ambiguity explicitly."""

import hashlib
import json

import pytest

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import repair_action_replay as replay_module
from autoformalism.rebuttal.repair_transactions import default_initialization
from autoformalism.rebuttal.revision_decision import (
    expressions,
    parameter_aliases,
    translate_names,
)
from autoformalism.staged_topology import content_hash
from tests.test_repair_comparison import CONTEXT, candidate


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    parent = candidate()
    initial = default_initialization(parent, CONTEXT).model_dump(mode="json")
    path = root / "inputs/frozen/candidates/parent.json"
    atomic_json(path, parent.model_dump(mode="json"))
    task = {
        "task_id": "seed0_redesigned_runtime",
        "seed": 0,
        "arm": "redesigned_runtime",
        "benchmark_id": "control",
        "tier": "hard",
        "candidate_path": "frozen/candidates/parent.json",
        "candidate_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    public_path = root / "inputs/frozen/public/control/metadata.json"
    atomic_json(public_path, {"public": True})
    ledger = {
        "frozen/public/control/metadata.json": hashlib.sha256(
            public_path.read_bytes()
        ).hexdigest()
    }
    inputs = {
        "test_data_opened": False,
        "private_reference_opened": False,
        "public_asset_ledger": ledger,
        "public_asset_ledger_sha256": content_hash(ledger),
    }
    inputs["plan_sha256"] = content_hash(inputs)
    atomic_json(root / "inputs/plan.json", inputs)
    plan = {
        "schema_version": "repair-feedback-comparison-1",
        "input_plan_sha256": inputs["plan_sha256"],
        "tasks": [task],
        "config": {"nonlinear_targets": ["v01"], "memory_targets": []},
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    atomic_json(root / "plan.json", plan)
    attempts = []
    for i, raw in enumerate(
        [
            {
                "scope": "model",
                "equations": [{"component": "f", "expression": "tanh(m)"}],
                "keep": ["f", "u01"],
            },
            {
                "scope": "model",
                "equations": [{"component": "m", "expression": "-rate*m+2*gain*u01"}],
                "initializers": [{"state": "y"}],
            },
        ]
    ):
        request = {
            "equations": expressions(parent),
            "initialization_plan": initial,
            "provisional_edits": {},
            "provisional_related_edits": {},
        }
        request = translate_names(request, parameter_aliases(parent))
        payload = {
            "body": {"messages": [{"content": "s"}, {"content": json.dumps(request)}]}
        }
        key = content_hash(payload)
        # Include attempt identity even if scientific request is identical.
        payload["attempt"] = i
        key = content_hash(payload)
        atomic_json(
            root / "results" / task["task_id"] / "calls" / f"{key}.json",
            {"request": payload},
        )
        attempts.append(
            {
                "attempt": i,
                "request_hash": key,
                "response": translate_names(raw, parameter_aliases(parent)),
                "diagnostics": [{"code": "ACTION_CONTRACT", "message": "historical"}],
            }
        )
    atomic_json(
        root / "results" / task["task_id"] / "round_001/transaction.json",
        {"status": "exhausted", "attempts": attempts},
    )
    monkeypatch.setattr(
        replay_module, "load_public_data", lambda *args: (None, CONTEXT)
    )
    return root


def test_replay_preserves_math_aliases_and_labels_ambiguity(source, tmp_path):
    output = tmp_path / "replay.json"
    result = replay_module.replay(source, output)
    assert result["stored_reply_count"] == 2
    assert result["views"]["conservative"] == {"rejected": 1, "pending": 1}
    assert result["views"]["keep_inventory_assumption"] == {
        "committed": 1,
        "pending": 1,
    }
    pending = result["rows"][1]["views"]["conservative"]
    assert pending["diagnostics"][0]["code"] == "INITIALIZER_TARGET_NOT_LATENT"
    assert pending["accepted_equations"] == ["m"]
    assert replay_module.replay(source, output) == result
    assert not result["counterfactual_search_claimed"]
    assert result["new_llm_calls"] == 0 and not result["parameter_fitting_performed"]


def test_replay_rejects_changed_candidate_and_in_source_output(source, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        replay_module.replay(source, source / "bad.json")
    candidate_path = source / "inputs/frozen/candidates/parent.json"
    candidate_path.write_text(candidate_path.read_text() + "\n")
    with pytest.raises(ValueError, match="candidate digest"):
        replay_module.replay(source, tmp_path / "replay.json")


def test_replay_rejects_changed_request(source, tmp_path):
    call = next(source.glob("results/*/calls/*.json"))
    payload = json.loads(call.read_text())
    payload["request"]["attempt"] = 99
    atomic_json(call, payload)
    with pytest.raises(ValueError, match="request digest"):
        replay_module.replay(source, tmp_path / "replay.json")


def test_replay_verifies_public_context_before_loading(source, tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("context loaded before verifying public assets")

    monkeypatch.setattr(replay_module, "load_public_data", forbidden)
    path = source / "inputs/frozen/public/control/metadata.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="public asset digest"):
        replay_module.replay(source, tmp_path / "replay.json")


def test_keep_removal_and_unknown_symbols_are_never_normalized():
    for raw in ({"keep": ["m"], "remove": ["m"]}, {"keep": ["typo"]}):
        with pytest.raises(ValueError):
            replay_module.adapt_legacy(
                raw, candidate(), CONTEXT, inventory_assumption=True
            )
