"""Seeds 2 to 4: each plan is a frozen seeds 0 and 1 plan with only its seeds moved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal import phase_c_baseline_plan as classical
from autoformalism.rebuttal.phase_c_d3 import load_plan as load_d3_plan
from autoformalism.rebuttal.phase_c_vendored_campaign import load_phase_c_vendored_plan

REPO = Path(__file__).resolve().parents[1]

#: Seeds plan, its frozen source, and what else may differ: top-level keys that
#: describe the seeds, and within the budget only its rationale.
PAIRS = (
    ("phase_c_d3_hosted_120b_v2_seeds_2_4.json", "phase_c_d3_hosted_120b_v2.json",
     {"purpose"}),
    ("phase_c_llm_ode_hosted_120b_v1_seeds_2_4.json",
     "phase_c_llm_ode_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_llm_ode_published_prompt_hosted_120b_v1_seeds_2_4.json",
     "phase_c_llm_ode_published_prompt_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_llm_sr_hosted_120b_v1_seeds_2_4.json",
     "phase_c_llm_sr_hosted_120b_v1.json", {"scope_note", "budget"}),
    ("phase_c_public_baseline_jetstream_cpu_v1_seeds_2_4.json",
     "phase_c_public_baseline_jetstream_cpu_v1.json", {"purpose"}),
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


@pytest.mark.parametrize(("seeds", "frozen", "reworded"), PAIRS)
def test_a_frozen_seeds_plan_changes_only_the_seeds_and_their_description(
    seeds, frozen, reworded
):
    plan, base = _load(seeds), _load(frozen)
    changed = {key for key in plan if plan[key] != base[key]}
    assert changed == {"repetitions"} | reworded
    if "budget" in reworded:
        budget, base_budget = plan["budget"], base["budget"]
        assert {k for k in budget if budget[k] != base_budget[k]} == {"rationale"}
    assert plan["status"] == base["status"]
    assert plan["status"].startswith("frozen_before")
    assert (base["repetitions"], plan["repetitions"]) == ([0, 1], [2, 3, 4])
    # Each description names the frozen plan whose seeds it continues.
    described = json.dumps({key: plan[key] for key in reworded})
    assert frozen.removesuffix(".json") in described
    assert "two seeds" not in described


def test_the_classical_seeds_plan_runs_both_methods_on_every_cell():
    plan = classical.load_phase_c_baseline_plan(REPO / "configs" / PAIRS[-1][0])
    tasks = classical.build_phase_c_baseline_tasks(plan)
    assert len(tasks) == 2 * 8 * 3
    assert {(task.method, task.repetition) for task in tasks} == {
        (method, seed) for method in ("sindy", "pysr") for seed in (2, 3, 4)
    }
