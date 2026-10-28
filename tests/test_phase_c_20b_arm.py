"""The gpt-oss-20b arm: each plan is a 120b plan with only the model moved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal.phase_c_d3 import load_plan as load_d3_plan
from autoformalism.rebuttal.phase_c_vendored_campaign import load_phase_c_vendored_plan

REPO = Path(__file__).resolve().parents[1]
MOVED = {"endpoint", "model", "status"}

#: 20b plan, its 120b source, and what else may differ: top-level keys, and
#: within the budget only its rationale.
PAIRS = (
    ("phase_c_llm_sr_smoke_vm_20b_v3.json", "phase_c_llm_sr_smoke_v5.json",
     {"scope_note", "budget"}),
    ("phase_c_llm_sr_expression_smoke_vm_20b_v1.json",
     "phase_c_llm_sr_expression_smoke_v1.json", {"scope_note", "budget"}),
    ("phase_c_llm_ode_smoke_vm_20b_v1.json", "phase_c_llm_ode_smoke_v1.json",
     {"scope_note", "budget"}),
    ("phase_c_llm_ode_published_prompt_smoke_vm_20b_v1.json",
     "phase_c_llm_ode_published_prompt_smoke_v1.json", {"scope_note", "budget"}),
    ("phase_c_llm_sr_vm_20b_v1.json", "phase_c_llm_sr_hosted_120b_v1.json",
     {"scope_note", "budget", "repetitions"}),
    ("phase_c_llm_sr_expression_vm_20b_v1.json",
     "phase_c_llm_sr_expression_hosted_120b_v1.json",
     {"scope_note", "budget", "repetitions"}),
    ("phase_c_llm_ode_vm_20b_v1.json", "phase_c_llm_ode_hosted_120b_v1.json",
     {"scope_note", "budget", "repetitions"}),
    ("phase_c_llm_ode_published_prompt_vm_20b_v1.json",
     "phase_c_llm_ode_published_prompt_hosted_120b_v1.json",
     {"scope_note", "budget", "repetitions"}),
    ("phase_c_d3_vm_20b_v1.json", "phase_c_d3_hosted_120b_v2.json",
     {"purpose", "repetitions"}),
    ("phase_c_d3_smoke_vm_20b_v2.json", "phase_c_d3_smoke_v2.json", {"purpose"}),
)


def _load(name: str) -> dict:
    path = REPO / "configs" / name
    loader = load_d3_plan if "_d3_" in name else load_phase_c_vendored_plan
    return loader(path).model_dump(mode="json")


@pytest.mark.parametrize(("arm", "source", "reworded"), PAIRS)
def test_a_20b_plan_moves_only_the_model(arm, source, reworded):
    plan, base = _load(arm), _load(source)
    changed = {key for key in plan if plan[key] != base[key]}
    assert changed - {"status"} == (MOVED - {"status"}) | reworded
    if "budget" in reworded:
        budget, base_budget = plan["budget"], base["budget"]
        assert {k for k in budget if budget[k] != base_budget[k]} == {"rationale"}
    assert (plan["endpoint"], plan["model"]) == ("vm_local_vllm", "openai/gpt-oss-20b")
    assert (base["endpoint"], base["model"]) == ("jetstream2_hosted", "gpt-oss-120b")


@pytest.mark.parametrize(("arm", "source", "reworded"), PAIRS)
def test_a_20b_plan_has_its_source_or_all_five_seeds(arm, source, reworded):
    plan, base = _load(arm), _load(source)
    if "repetitions" in reworded:
        assert plan["repetitions"] == [0, 1, 2, 3, 4]
        assert set(base["repetitions"]) < set(plan["repetitions"])
    else:
        assert plan["repetitions"] == base["repetitions"]


def test_the_20b_model_is_the_revision_our_method_pins():
    construction = json.loads(
        (REPO / "configs" / "phase_c_construction_v1.json").read_text()
    )
    assert construction["model_settings"]["model"] == "openai/gpt-oss-20b"
    server = (REPO / "scripts" / "jetstream" / "gpu_server.sh").read_text()
    assert "configs/phase_c_construction_v1.json" in server
    assert '["model_settings"]["model_revision"]' in server


def test_the_new_20b_plans_await_review():
    for arm, _source, _reworded in PAIRS:
        expected = (
            "frozen_before_calls"
            if arm == "phase_c_d3_smoke_vm_20b_v2.json"
            else "proposed_pending_review"
        )
        assert _load(arm)["status"] == expected, arm
