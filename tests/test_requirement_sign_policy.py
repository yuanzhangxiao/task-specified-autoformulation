"""Saved-response replay and live repair share the topology-owned sign contract."""

import copy
import json
import subprocess
import sys

import pytest

from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.search.requirement_feedback import (
    STRICT_REPAIR,
    TOPOLOGY_SIGN_REPAIR,
    apply_requirement_repair,
    repair_failure_feedback,
    repair_payload,
)
from scripts.smoke_prefit_requirements import repair_client, synthetic_plan


@pytest.fixture
def fixture(tmp_path):
    plan = synthetic_plan(tmp_path, fixed_sign=True)
    case = next(c for c in plan["cases"] if c["cohort"] == "repair")
    slot = next(s for s in case["bundle"]["slots"] if s["selected_term"]["lhs"] == "m")
    raw = copy.deepcopy(slot["accepted_reply"])
    raw["expression"] = "-(" + raw["expression"].replace("v01", "v01**2") + ")"
    raw["interaction_id"] = slot["interaction_id"]
    return tmp_path / "campaign", plan, case, raw


def test_sign_repair_compiles_with_fixed_outer_sign_and_preserves_siblings(fixture):
    _, _, case, raw = fixture
    before = copy.deepcopy(case)
    with pytest.raises(ValueError, match="DUPLICATED_WHOLE_OUTER_SIGN"):
        apply_requirement_repair(case["bundle"], case["bindings"], raw)
    result = apply_requirement_repair(
        case["bundle"], case["bindings"], raw, repair_policy=TOPOLOGY_SIGN_REPAIR
    )
    assert not result["diagnosis"]["requirement_gap"]
    assert (
        result["outer_sign_normalizations"][0]["original_expression"]
        == raw["expression"]
    )
    assert (
        result["selected_function"]["selected_term"]["outer_weight_sign"] == "negative"
    )
    assert result["protected_slots_preserved"] == len(case["bundle"]["slots"]) - 1
    assert result["initialization"]["plan"] == case["bundle"]["initialization"]["plan"]
    assert (
        result["candidate"]["initial_conditions"]
        == case["bundle"]["candidate"]["initial_conditions"]
    )
    positive = {**raw, "expression": raw["expression"][2:-1]}
    reference = apply_requirement_repair(case["bundle"], case["bindings"], positive)
    assert result["candidate"] == reference["candidate"]
    assert case == before


def test_sources_and_internal_nonlinearity_are_still_enforced(fixture):
    _, _, case, raw = fixture
    invalid = {**raw, "expression": raw["expression"] + "*unlisted_state"}
    with pytest.raises(ValueError, match="source mismatch") as failure:
        apply_requirement_repair(
            case["bundle"],
            case["bindings"],
            invalid,
            repair_policy=TOPOLOGY_SIGN_REPAIR,
        )
    feedback = repair_failure_feedback(case["bundle"], invalid, failure.value)
    assert feedback["extra_sources"] == ["unlisted_state"]
    assert feedback["function_contract"]["required_sources"] == ["m", "u01", "v01"]
    assert feedback["code"] == "SOURCE_MISMATCH"
    assert "interaction-structure revision" in feedback["next_action"]
    assert len(feedback["outer_sign_normalizations"]) == 1
    with pytest.raises(ValueError, match="REQUIRED_NONLINEAR_FEEDBACK_MISSING"):
        apply_requirement_repair(
            case["bundle"],
            case["bindings"],
            {**raw, "expression": raw["expression"].replace("v01**2", "v01")},
            repair_policy=TOPOLOGY_SIGN_REPAIR,
        )


def test_new_prompt_contract_is_explicit_and_old_payload_is_unchanged(fixture):
    _, _, case, _ = fixture
    old = repair_payload(case["bundle"], case["baseline"])
    new = repair_payload(
        case["bundle"], case["baseline"], repair_policy=TOPOLOGY_SIGN_REPAIR
    )
    assert all("function_contract" not in s for s in old["eligible_interactions"])
    for slot in new["eligible_interactions"]:
        contract = slot["function_contract"]
        assert contract["required_sources"] == slot["selected_term"]["sources"]
        assert (
            contract["assembly_template"] == slot["selected_term"]["assembly_template"]
        )
        assert not contract["additional_state_or_input_sources_allowed"]


def negative_client(root, plan, task, calls, raw):
    client = repair_client(root, plan, task, calls)
    original = client.transport

    def transport(url, body, timeout):
        result = original(url, body, timeout)
        result["choices"][0]["message"]["content"] = json.dumps(raw)
        return result

    client.transport = transport
    return client


def saved_and_new(fixture):
    root, plan, _, raw = fixture
    calls = []
    for task in plan["tasks"]:
        campaign.run_episode(
            root, plan, task, negative_client(root, plan, task, calls, raw)
        )
    assert len(calls) == 3
    path = root.parent / "new-config.json"
    config = {
        **plan["config"],
        "repair_policy": TOPOLOGY_SIGN_REPAIR,
        "replay_source_plan_sha256": plan["artifact_sha256"],
    }
    path.write_text(json.dumps(config))
    new_root = root.parent / "new"
    new_plan = campaign.freeze(root.parent / "source", path, new_root)
    return root, new_root, new_plan, raw


def test_cached_replay_and_live_normalization_have_separate_accounting(fixture):
    previous, root, plan, raw = saved_and_new(fixture)
    old_files = {p: p.read_bytes() for p in previous.rglob("*.json")}
    replay = campaign.replay_saved_repairs(root, previous)
    assert replay["saved_responses"] == replay["newly_admissible"] == 3
    assert replay["strict_accepted"] == replay["acceptance_regressions"] == 0
    assert replay["live_llm_calls"] == 0
    assert campaign.replay_saved_repairs(root, previous) == replay
    assert {p: p.read_bytes() for p in previous.rglob("*.json")} == old_files
    calls = []
    for task in plan["tasks"]:
        result = campaign.run_episode(
            root, plan, task, negative_client(root, plan, task, calls, raw)
        )
        assert (
            campaign.run_episode(
                root, plan, task, negative_client(root, plan, task, calls, raw)
            )
            == result
        )
    assert len(calls) == 1
    report = campaign.summarize(root)
    repaired = report["arms"]["requirement_feedback"]["repair"]
    assert (
        repaired["first_attempt_repaired"] == repaired["outer_sign_normalizations"] == 1
    )
    assert report["arms"]["requirement_feedback"]["control"]["physical_requests"] == 0
    payload = json.loads(calls[0]["messages"][1]["content"].split("\n", 1)[1])
    assert "function_contract" in payload["eligible_interactions"][0]


def test_cli_replay_and_pinned_source_rejection(fixture):
    previous, root, _, _ = saved_and_new(fixture)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prefit_requirement_campaign.py",
            "replay",
            "--root",
            str(root),
            "--previous-root",
            str(previous),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["newly_admissible"] == 3
    payload = json.loads((previous / "plan.json").read_text())
    payload["config"]["repair_policy"] = TOPOLOGY_SIGN_REPAIR
    (previous / "plan.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        campaign.replay_saved_repairs(root, previous)


def test_old_configuration_defaults_to_strict():
    assert (
        campaign.RequirementConfig.model_fields["repair_policy"].default
        == STRICT_REPAIR
    )
