"""CSTR-hard plans: each is a frozen Phase C plan with only its cell and seeds moved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal import phase_c_baseline_plan as classical
from autoformalism.rebuttal import phase_c_baselines as pcb
from autoformalism.rebuttal.phase_c_d3 import load_plan as load_d3_plan
from autoformalism.rebuttal.phase_c_vendored_campaign import load_phase_c_vendored_plan
from autoformalism.research.phase_c_inputs import ROSTER

REPO = Path(__file__).resolve().parents[1]
HARD = "phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1"
#: The released CSTR-hard proposer prompt's digest.
PROMPT = "6d951d2a8aae9417852ea87c078ccc6ce9c297baf79e7b6da16a241aaaae15f0"

#: CSTR-hard plan, its frozen source, and what else may differ: top-level keys
#: that describe the cell and seeds, and within a budget only its rationale.
PAIRS = (
    ("phase_c_cstr_hard_public_baseline_jetstream_cpu_v1.json",
     "phase_c_public_baseline_jetstream_cpu_v1.json", {"purpose"}),
    ("phase_c_cstr_hard_d3_hosted_120b_v2.json",
     "phase_c_d3_hosted_120b_v2.json", {"purpose"}),
    ("phase_c_cstr_hard_llm_ode_hosted_120b_v1.json",
     "phase_c_llm_ode_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_cstr_hard_llm_ode_published_prompt_hosted_120b_v1.json",
     "phase_c_llm_ode_published_prompt_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_cstr_hard_llm_sr_hosted_120b_v1.json",
     "phase_c_llm_sr_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_cstr_hard_llm_sr_expression_hosted_120b_v1.json",
     "phase_c_llm_sr_expression_hosted_120b_v1.json", {"scope_note", "budget"}),
)


def _load(name: str) -> dict:
    path = REPO / "configs" / name
    if "_d3_" in name:
        plan = load_d3_plan(path)
    elif "public_baseline" in name:
        plan = classical.load_phase_c_baseline_plan(path)
    else:
        plan = load_phase_c_vendored_plan(path)
    return plan.model_dump(mode="json")


@pytest.mark.parametrize(("hard", "frozen", "reworded"), PAIRS)
def test_a_cstr_hard_plan_changes_only_the_cell_the_seeds_and_their_description(
    hard, frozen, reworded
):
    plan, base = _load(hard), _load(frozen)
    changed = {key for key in plan if plan[key] != base[key]}
    assert changed == {"cells", "repetitions"} | reworded
    if "budget" in reworded:
        budget, base_budget = plan["budget"], base["budget"]
        assert {k for k in budget if budget[k] != base_budget[k]} == {"rationale"}
    assert plan["cells"] == [
        {"benchmark_id": HARD, "tier": "hard", "public_prompt_sha256": PROMPT}
    ]
    assert (base["repetitions"], plan["repetitions"]) == ([0, 1], [0, 1, 2])
    assert plan["status"] == base["status"]
    assert plan["status"].startswith("frozen_before")
    # Each description names its frozen source and no longer the roster's cells.
    described = json.dumps({key: plan[key] for key in reworded})
    assert frozen.removesuffix(".json") in described
    assert "CSTR-hard" in described
    assert "eight Phase C roster cells" not in described
    assert "two seeds" not in described


def test_cstr_hard_joins_the_cells_a_baseline_may_run_on_but_not_the_roster():
    assert HARD not in ROSTER
    assert len(ROSTER) == 8
    # The noise study's acquisitions are the only other cells a plan may name.
    assert set(pcb.BASELINE_CELLS) == {*ROSTER, HARD, *pcb.NOISE_CELLS}
    assert pcb.tier_of(HARD) == "hard"


def test_the_classical_cstr_hard_plan_runs_both_methods_with_three_seeds():
    plan = classical.load_phase_c_baseline_plan(REPO / "configs" / PAIRS[0][0])
    tasks = classical.build_phase_c_baseline_tasks(plan)
    assert {(task.method, task.benchmark_id, task.repetition) for task in tasks} == {
        (method, HARD, seed) for method in ("sindy", "pysr") for seed in (0, 1, 2)
    }
