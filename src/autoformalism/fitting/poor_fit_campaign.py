"""M30: paired M29 requests, starts and ceilings with opt-in poor-fit handling."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import poor_fit
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_integration_campaign as source
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-poor-fit-qualification-1"
ARMS = ("legacy", "output_aware")


class Policy(StrictSchema):
    protocol: Literal["phase-c-poor-fit-qualification-1"] = PROTOCOL
    baseline: source.Policy = Field(default_factory=source.Policy)
    handling: poor_fit.Policy = Field(default_factory=poor_fit.Policy)

    @model_validator(mode="after")
    def legacy(self):
        adapter = self.baseline.fitting.adapter
        if adapter.strategy != "guarded_only" or adapter.search.poor_fit is not None:
            raise ValueError("baseline must retain M29 guarded-only search")
        return self


def arm_policy(policy, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm == "legacy":
        return policy.baseline
    payload = policy.baseline.model_dump(mode="json")
    payload["fitting"]["adapter"]["search"]["poor_fit"] = policy.handling.model_dump()
    return source.Policy.model_validate(payload)


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    data = read_seal(inputs)
    original = source.roster(data)
    with public._lock(root):
        seal(root / "inputs.json", data)
        children = {}
        for arm in ARMS:
            source.prepare(
                root / "arms" / arm, root / "inputs.json", arm_policy(policy, arm)
            )
            children[arm] = public.content_sha256(
                read_seal(root / "arms" / arm / "plan.json")
            )
        tasks = [
            t
            | {
                "task_id": arm + "__" + t["task_id"],
                "source_task_id": t["task_id"],
                "arm": arm,
                "source_index": i,
            }
            for i, t in enumerate(original)
            for arm in ARMS
        ]
        plan = {
            "protocol": PROTOCOL,
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "inputs_sha256": public.content_sha256(data),
            "policy": policy.model_dump(mode="json"),
            "children": children,
            "tasks": tasks,
            "test_data_opened": False,
            "live_llm_calls": 0,
        }
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime=True):
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        data
    ):
        raise ValueError("paired campaign identity differs")
    for arm in ARMS:
        child, copied = source.verify(root / "arms" / arm, runtime=runtime)
        if (
            data != copied
            or public.content_sha256(child) != plan["children"][arm]
            or child["policy"] != arm_policy(policy, arm).model_dump(mode="json")
        ):
            raise ValueError("paired child plan differs")
    expected = [
        t
        | {
            "task_id": a + "__" + t["task_id"],
            "source_task_id": t["task_id"],
            "arm": a,
            "source_index": i,
        }
        for i, t in enumerate(source.roster(data))
        for a in ARMS
    ]
    if plan["tasks"] != expected:
        raise ValueError("paired roster differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("paired source/runtime differs")
    return plan, data


def completed(root, plan, task):
    child_root = root / "arms" / task["arm"]
    child = read_seal(child_root / "plan.json")
    outcome = source.completed(
        child_root / "results" / task["source_task_id"],
        {
            "plan_sha256": public.content_sha256(child),
            "task": child["tasks"][task["source_index"]],
        },
    )
    expected = outcome | task | {"parent_plan_sha256": public.content_sha256(plan)}
    saved = read_seal(root / "results" / task["task_id"] / "result.json")
    if saved != expected:
        raise ValueError("paired result differs from sealed child")
    return saved


def run_task(root: Path, index: int):
    plan, _ = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown paired task index")
    task = plan["tasks"][index]
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(root, plan, task)
        value = source.run_task(root / "arms" / task["arm"], task["source_index"])
        result = value | task | {"parent_plan_sha256": public.content_sha256(plan)}
        seal(folder / "result.json", result)
        return result


def report(root: Path):
    plan, _ = verify(root, runtime=False)
    rows = [
        completed(root, plan, t)
        if (root / "results" / t["task_id"] / "result.json").exists()
        else t | {"status": "missing"}
        for t in plan["tasks"]
    ]
    pairs = []
    for name in dict.fromkeys(t["source_task_id"] for t in plan["tasks"]):
        pair = {
            a: next(r for r in rows if r["arm"] == a and r["source_task_id"] == name)
            for a in ARMS
        }
        setups = [
            (r.get("fit") or {}).get("preparation", {}).get("value")
            for r in pair.values()
        ]
        matched = None
        if all(s and "problem" in s for s in setups):
            matched = all(setups[0][k] == setups[1][k] for k in ("problem", "starts"))
            if not matched:
                raise ValueError("paired starts/training problem differ")
        pairs.append(
            {
                "source_task_id": name,
                "starts_and_problem_match": matched,
                "arms": {
                    a: {
                        "status": r["status"],
                        "cost": source.costs(r),
                        "search_cost": ((r.get("fit") or {}).get("fit") or {}).get(
                            "cost"
                        ),
                        "metrics": r.get("evaluation", {}).get("metrics"),
                        "coefficient_recovery": r.get("evaluation", {}).get(
                            "coefficient_recovery"
                        ),
                        "assessment": ((r.get("fit") or {}).get("fit") or {}).get(
                            "assessment"
                        ),
                    }
                    for a, r in pair.items()
                },
            }
        )
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "incomplete"
        if any(r["status"] == "missing" for r in rows)
        else "complete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "pairs": pairs,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Same archived requests, raw starts and ceilings. "
        "Certified inactive coordinates stay at incumbent values in output-aware "
        "searches. Local stationarity "
        "is not global optimality; validation is post-fit. No scientific sign repairs.",
    }
    public._write(root / "summary.json", value)
    return value
