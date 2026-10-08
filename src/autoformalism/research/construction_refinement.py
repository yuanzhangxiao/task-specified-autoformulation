"""Frozen diagnostic roster: eight minimal constructions and three saved repairs."""

from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search import construction_bookkeeping as bookkeeping
from autoformalism.search import construction_ledger as ledger
from autoformalism.search import construction_prompts as prompts
from autoformalism.staged_topology import content_hash

REPAIRS = (
    "cell07_seed0_full_separate_integrated",
    "cell07_seed0_full_joint_guided_integrated",
    "cell06_seed0_full_joint_adaptive_integrated",
)


def repair_tasks(source: Path, cells: dict) -> list[dict]:
    """Import pre-global drafts with public context and verified call provenance."""
    from autoformalism.research import construction_comparison as campaign

    plan = sealed_read(source / "plan.json")
    if (
        plan.get("protocol") != campaign.PROTOCOL
        or plan.get("test_data_opened") is not False
        or plan["config"].get("bookkeeping_policy") != bookkeeping.FEEDBACK_POLICY
    ):
        raise ValueError("requires the saved current-bookkeeping-2 confirmation")
    by_id = {t["task_id"]: t for t in plan["tasks"]}
    tasks = []
    for name in REPAIRS:
        if name not in by_id:
            raise ValueError(f"missing diagnosed repair task: {name}")
        task = by_id[name]
        if cells[task["benchmark_id"]] != plan["cells"][task["benchmark_id"]]:
            raise ValueError("repair public context differs from current public source")
        directory = source / "results" / name
        identity = campaign.baseline.namespace(plan, task)
        records = campaign.checked_records(directory, identity)
        if not records:
            raise ValueError("saved repair has no call evidence")
        before = sealed_read(directory / "construction/before_repair.json")
        tasks.append(
            {
                **task,
                "task_id": "repair_" + name,
                "bookkeeping_policy": bookkeeping.FIDELITY_POLICY,
                "starting_checkpoint": before,
                "origin": {
                    "plan_sha256": plan["artifact_sha256"],
                    "task": name,
                    "namespace": identity,
                    "checkpoint_sha256": before["artifact_sha256"],
                    "selection": "pre-global; diagnosed cases, not score-selected",
                },
            }
        )
    return tasks


def validate_tasks(plan: dict) -> None:
    """Fail closed on altered diagnostic arms, start states, or policy routing."""
    tasks = plan["tasks"]
    fresh = [t for t in tasks if "starting_checkpoint" not in t]
    repairs = [t for t in tasks if "starting_checkpoint" in t]
    valid = (
        len(tasks) == 11
        and len(fresh) == 8
        and len(repairs) == 3
        and {t["benchmark_id"] for t in fresh} == set(plan["cells"])
        and {t.get("origin", {}).get("task") for t in repairs} == set(REPAIRS)
        and plan["config"]["bookkeeping_policy"] == "legacy"
    )
    for task in tasks:
        name = task["task_id"]
        valid &= (
            Path(name).name == name
            and name not in {".", ".."}
            and task["seed"] == 0
            and task["arm"] == "full"
            and task.get("shared_processes") is True
            and task.get("scientific_verifier") is True
        )
    valid &= len({t["task_id"] for t in tasks}) == 11
    for task in fresh:
        valid &= (
            task.get("prompt_family") == "minimal"
            and task.get("stage_schedule") == prompts.SCHEDULE
            and task["policy"] == "joint_adaptive"
            and task["process_question"] == "dedicated"
            and task.get("bookkeeping_policy") == bookkeeping.MINIMAL_POLICY
        )
    for task in repairs:
        checkpoint = task["starting_checkpoint"]
        digest = content_hash(
            {k: v for k, v in checkpoint.items() if k != "artifact_sha256"}
        )
        valid &= (
            digest
            == checkpoint.get("artifact_sha256")
            == task["origin"]["checkpoint_sha256"]
            and task.get("bookkeeping_policy") == bookkeeping.FIDELITY_POLICY
            and task.get("prompt_family") is None
            and task["policy"] in {"separate", "joint_guided", "joint_adaptive"}
            and task["process_question"] == "integrated"
        )
        ledger.Draft.model_validate(checkpoint["draft"])
    if not valid:
        raise ValueError("requires the exact eight-fresh/three-saved refinement roster")
