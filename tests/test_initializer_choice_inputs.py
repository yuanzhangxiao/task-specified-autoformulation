"""Diagnostic inputs bind exact V6 cache replay, old charges and new bounded calls."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import initializer_choice_inputs as inputs
from autoformalism.staged_topology import content_hash
from tests.test_explicit_variable_bindings import response
from tests.test_initialization_bookkeeping_runtime import (  # noqa: F401
    bookkeeping_plan,
    choices,
    equality_plan,
    evidence_plan,
    live,
    prepared,
    saved_setup,
    source,
    v2,
)
from tests.test_initialization_equality_retest import _rewrite
from tests.test_initialized_topology_repair import packet_and_reply
from tests.test_saved_topology_repair import execute


@pytest.fixture
def v6_source(tmp_path, bookkeeping_plan, monkeypatch):  # noqa: F811
    """Create a sealed two-case V6 history, including one unavailable final reply."""
    monkeypatch.setattr(inputs, "source_identity", lambda: "diagnostic-source")
    parent, task = deepcopy(bookkeeping_plan)
    original = deepcopy(parent["starts"][task["source_task"]])
    original["predecessor_historical_cost"] = {"earlier_scope": "retained"}
    parent.pop("artifact_sha256", None)
    parent["source_identity"] = "archived-v6-source"
    parent["settings"]["max_output_tokens"] = 8192
    parent["tasks"] = [
        {**task, "task_id": f"case_{i}", "source_task": f"start_{i}", "seed": i}
        for i in (1, 2)
    ]
    parent["starts"] = {f"start_{i}": deepcopy(original) for i in (1, 2)}
    root = tmp_path / "v6"
    sealed_write(root / "plan.json", parent)
    parent = sealed_read(root / "plan.json")
    results = {}
    for task in parent["tasks"]:
        seen = []

        def send(url, body, timeout, task=task, seen=seen):
            _, reply = packet_and_reply(body, complete=False)
            seen.append(reply)
            raw = response(reply)
            if task["seed"] == 2 and len(seen) == 2:
                raw["choices"][0]["finish_reason"] = "length"
                raw["choices"][0]["message"]["content"] = '{"partial":'
            return raw

        results[task["task_id"]] = execute(root, parent, task, send)
    assert results["case_1"]["event_count"] == 3
    assert results["case_2"]["status"] == "provider_unavailable"
    return root, parent, results


def test_freeze_replays_original_cache_and_preserves_exact_inputs(tmp_path, v6_source):
    root, parent, results = v6_source
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    output = tmp_path / "diagnostic"
    plan = inputs.freeze(root, output)
    assert inputs.freeze(root, output) == plan
    assert inputs.verify(output) == plan
    assert plan["protocol"] == "saved-initializer-choice-diagnostic-1"
    assert plan["source_identity"] != parent["source_identity"]
    assert plan["parent_plan"] == parent
    assert plan["budgets"] == inputs.BUDGETS
    assert plan["budgets"]["maximum_calls"] == 6
    assert plan["budgets"]["maximum_tokens"] == 196608
    assert plan["optimizer_calls"] == 0
    assert plan["test_data_opened"] is False
    assert plan["automatic_followup"] is False
    assert plan["model_promotion"] is False
    assert plan["topology_edits_allowed"] is False
    assert "controlled causal" in plan["selection"]
    for name, case in plan["cases"].items():
        result = results[name]
        original = parent["starts"][case["task"]["source_task"]]
        assert case["source_result"] == result
        assert case["editable_state"] == "z"
        for key in ("draft", "initializer_handoff", "initial_information_contract"):
            assert case["start"][key] == result[key]
        assert case["start"]["scientific_state"] == result["scientific_state"]
        assert case["start"]["historical_cost"] == result["cost"]
        assert case["start"]["predecessor_historical_cost"] == {
            "historical_cost": original["historical_cost"],
            "predecessor_historical_cost": original["predecessor_historical_cost"],
        }
        assert case["packet"] == json.loads(
            case["latest_request_record"]["request"]["body"]["messages"][1]["content"]
        )
        previous = original.get("previous_conflicts", original["diagnostics"])
        assert {content_hash(q) for q in previous.get("tracked_conflicts", [])} <= {
            content_hash(q)
            for q in case["start"]["previous_conflicts"]["tracked_conflicts"]
        }
    assert (
        sum(
            c["start"]["historical_cost"]["physical_requests"]
            for c in plan["cases"].values()
        )
        == 5
    )
    assert (
        sum(
            c["start"]["historical_cost"]["budget_charge"]
            for c in plan["cases"].values()
        )
        == 500
    )
    probe = plan["completion_probe"]
    record = plan["cases"]["case_2"]["latest_request_record"]
    assert probe["case_id"] == "case_2"
    assert probe["repeats"] == 2
    assert probe["request"] == record["request"]
    assert probe["request_hash"] == record["request_hash"]
    assert probe["step"] == record["step"]
    assert probe["attempt"] == record["attempt"]
    assert probe["request"]["body"]["max_tokens"] == 8192
    unavailable = plan["cases"]["case_2"]
    assert (
        unavailable["start"]["last_edit"]
        == unavailable["source_events"][-1]["retained_last_edit"]
    )
    assert (
        unavailable["start"]["repetition_feedback"]
        == unavailable["source_events"][-1]["retained_repetition_feedback"]
    )
    assert before == {str(p): p.read_bytes() for p in root.rglob("*.json")}


def test_genuine_replay_never_calls_provider_or_tokenizer(
    tmp_path, v6_source, monkeypatch
):
    def unavailable(*args, **kwargs):
        pytest.fail("offline replay attempted provider/tokenizer access")

    monkeypatch.setattr(inputs, "_blocked", unavailable)
    plan = inputs.freeze(v6_source[0], tmp_path / "diagnostic")
    assert len(plan["cases"]) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("protocol", "wrong"),
        ("actionable_policy", "wrong"),
        ("test_data_opened", True),
        ("optimizer_calls", 1),
        ("automatic_followup", True),
        ("maximum_calls_per_arm", 11),
        ("maximum_calls_per_arm", 10.0),
        ("inherited_provider_outcomes", {}),
    ],
)
def test_source_policy_is_checked_before_replay(tmp_path, v6_source, key, value):
    root, _, _ = v6_source
    _rewrite(root / "plan.json", lambda p: p.update({key: value}))
    with pytest.raises(ValueError, match="public-only"):
        inputs.freeze(root, tmp_path / "invalid")


def test_source_serving_settings_and_case_bound_are_fixed(tmp_path, v6_source):
    root, _, _ = v6_source
    _rewrite(
        root / "plan.json", lambda p: p["settings"].update(max_output_tokens=16384)
    )
    with pytest.raises(ValueError, match="settings"):
        inputs.freeze(root, tmp_path / "invalid")
    _rewrite(root / "plan.json", lambda p: p["settings"].update(max_output_tokens=8192))
    _rewrite(root / "plan.json", lambda p: p["tasks"].append(deepcopy(p["tasks"][0])))
    with pytest.raises(ValueError, match="public-only"):
        inputs.freeze(root, tmp_path / "invalid")


@pytest.mark.parametrize("kind", ["event", "call"])
def test_all_history_including_unavailable_call_must_be_present(
    tmp_path, v6_source, kind
):
    root, _, _ = v6_source
    directory = root / "results/case_2"
    path = directory / "events/001.json"
    if kind == "call":
        event = sealed_read(path)
        path = directory / "calls" / f"{event['request_hash']}.json"
    path.unlink()
    with pytest.raises(ValueError, match="coverage"):
        inputs.freeze(root, tmp_path / "invalid")


def test_source_seal_is_not_optional(tmp_path, v6_source):
    root, _, _ = v6_source
    path = root / "plan.json"
    parent = json.loads(path.read_text())
    parent["source_identity"] = "tampered"
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="digest"):
        inputs.freeze(root, tmp_path / "invalid")


def test_resealed_request_body_must_match_original_replay(tmp_path, v6_source):
    root, _, _ = v6_source
    directory = root / "results/case_2"
    path = directory / "events/001.json"
    event = sealed_read(path)
    record_path = directory / "calls" / f"{event['request_hash']}.json"
    record = json.loads(record_path.read_text())
    record["request"]["body"]["seed"] += 1
    record["request_hash"] = content_hash(record["request"])
    record_path.unlink()
    (directory / "calls" / f"{record['request_hash']}.json").write_text(
        json.dumps(record)
    )
    _rewrite(
        path,
        lambda e: e.update(
            request_hash=record["request_hash"], record_sha256=content_hash(record)
        ),
    )
    with pytest.raises(ValueError, match="exact repair predecessor"):
        inputs.freeze(root, tmp_path / "invalid")


@pytest.mark.parametrize("key", ["cost", "draft", "scientific_state", "stop_reason"])
def test_terminal_result_must_equal_full_v6_replay(tmp_path, v6_source, key):
    root, _, _ = v6_source
    _rewrite(root / "results/case_2/result.json", lambda r: r.update({key: "wrong"}))
    with pytest.raises(ValueError):
        inputs.freeze(root, tmp_path / "invalid")


def test_historical_unavailable_call_keeps_its_full_charge(tmp_path, v6_source):
    root, _, _ = v6_source
    directory = root / "results/case_2"
    event = sealed_read(directory / "events/001.json")
    path = directory / "calls" / f"{event['request_hash']}.json"
    record = json.loads(path.read_text())
    record["budget_charge"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="charge"):
        inputs.freeze(root, tmp_path / "invalid")


@pytest.mark.parametrize(
    "change",
    [
        "budget",
        "budget_type",
        "identity",
        "flag",
        "case",
        "state",
        "packet",
        "probe",
        "history",
    ],
)
def test_verify_rejects_resealed_scope_or_case_tampering(tmp_path, v6_source, change):
    output = tmp_path / "diagnostic"
    inputs.freeze(v6_source[0], output)

    def corrupt(plan):
        if change == "budget":
            plan["budgets"]["maximum_choice_calls_per_case"] = 3
        elif change == "budget_type":
            plan["budgets"]["maximum_choice_calls_per_case"] = 2.0
        elif change == "identity":
            plan["source_identity"] = "different-code"
        elif change == "flag":
            plan["optimizer_calls"] = False
        elif change == "case":
            plan["cases"].pop("case_1")
        elif change == "state":
            plan["cases"]["case_1"]["editable_state"] = "invented"
        elif change == "packet":
            plan["cases"]["case_2"]["packet"]["stage"] = "topology"
        elif change == "probe":
            plan["completion_probe"]["request"]["body"]["seed"] += 1
        else:
            plan["cases"]["case_1"]["start"]["historical_cost"]["budget_charge"] = 0

    _rewrite(output / "plan.json", corrupt)
    with pytest.raises(ValueError):
        inputs.verify(output)


def test_one_case_remains_bounded_and_preserves_original_plan(tmp_path, v6_source):
    root, parent, _ = v6_source
    # An original one-case namespace needs its own cached execution, not a roster edit.
    one = deepcopy(parent)
    one.pop("artifact_sha256")
    one["tasks"] = [one["tasks"][1]]
    one["starts"] = {"start_2": one["starts"]["start_2"]}
    one_root = tmp_path / "one"
    sealed_write(one_root / "plan.json", one)
    one = sealed_read(one_root / "plan.json")

    def truncate(url, body, timeout):
        _, reply = packet_and_reply(body)
        raw = response(reply)
        raw["choices"][0]["finish_reason"] = "length"
        return raw

    execute(one_root, one, one["tasks"][0], truncate)
    plan = inputs.freeze(one_root, tmp_path / "diagnostic")
    assert len(plan["cases"]) == 1
    assert plan["parent_plan"] == one
    assert plan["budgets"] == inputs.BUDGETS
    assert root.exists()


def test_source_cannot_be_written_over(v6_source):
    root, _, _ = v6_source
    for destination in (root, root / "nested"):
        with pytest.raises(ValueError, match="outside"):
            inputs.freeze(root, destination)


def test_entrypoint_identity_requires_every_file_and_detects_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(inputs.saved, "REPO", tmp_path)
    monkeypatch.setattr(inputs.saved, "source_identity", lambda: "package-code")
    with pytest.raises(ValueError, match="entrypoint"):
        inputs.source_identity()
    for name in inputs.ENTRYPOINTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    original = inputs.source_identity()
    (tmp_path / inputs.ENTRYPOINTS[0]).write_text("changed entrypoint")
    assert inputs.source_identity() != original
