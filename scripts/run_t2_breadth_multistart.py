#!/usr/bin/env python3
"""Independently refit the meal-memory and sigmoid T2 children from paired starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autoformalism.research import t2_breadth_fitting as campaign

TASKS = {
    "short-memory": ("c15-3", "tau_M", 10.0),
    "long-memory": ("c15-3", "tau_M", 120.0),
    "shallow-sigmoid": ("c15-2", "k_prod", 0.25),
    "steep-sigmoid": ("c15-2", "k_prod", 4.0),
}
# The first two blocks reproduce the previous allowance at a single chosen start.
BLOCKS = (
    ("fit", False, 1000),
    ("domain", True, 1000),
    ("continuation_1", True, 2000),
    ("continuation_2", True, 2000),
)
SEARCH_SECONDS = 900
REPLAY_SECONDS = 240


def start_parameters(model, task: str) -> dict:
    """Change only the named shape coordinate within each pair of public starts."""
    branch, name, setting = TASKS[task]
    if model.branch != branch or name not in model.names:
        raise ValueError("start does not belong to this model's free coordinates")
    if branch == "c15-3":
        common = campaign.fitting.starts(model)["finite_meal_memory"]
    else:
        common = campaign.fitting.starts(model)["glucose_scaled_sigmoid"]
        setting /= model.scales[model.channels.index("Gp")]
    point = common | {name: float(setting)}
    model.reduce(point)
    return point


def prepare(root: Path, data: dict) -> dict:
    """Freeze all four start vectors, the numerical runtime, and the common budget."""
    starts = {}
    for branch in ("c15-3", "c15-2"):
        model = campaign._model(branch, data)
        pair = [task for task, (b, _, _) in TASKS.items() if b == branch]
        for task in pair:
            starts[task] = start_parameters(model, task)
        a, b = (starts[task] for task in pair)
        if {name for name in a if a[name] != b[name]} != {TASKS[pair[0]][1]}:
            raise ValueError("paired starts must differ in exactly one coordinate")
    plan = {
        "protocol": "t2-breadth-independent-starts-1",
        "archive_sha256": data["archive_sha256"],
        "training_sha256": data["training_sha256"],
        "models": {b: data["models"][b]["model_sha256"] for b in ("c15-3", "c15-2")},
        "source_sha256": campaign.public._source_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": campaign.public._runtime(),
        "starts": starts,
        "blocks": [list(b) for b in BLOCKS],
        "seconds_per_search": SEARCH_SECONDS,
        "seconds_per_original_replay": REPLAY_SECONDS,
        "shifted_meals": ["train_003", "train_006", "train_007"],
        "large_meal": {"trajectory": data["boundary"]["trajectory"], "minute": 300},
        "comparison": "per-start unpenalized NMSE from original-equation replays",
        "selection_split": "train",
        "validation_or_test_access": False,
    }
    with campaign.public._lock(root):
        campaign.seal(root / "plan.json", plan)
    return plan


def retained(points: dict, *, admissible: bool) -> tuple[str | None, dict | None]:
    """Prefer the earlier original-equation replay on equal losses."""
    selected = campaign._select(points, feasible=admissible)
    return selected["origin"], selected["endpoint"]


def measurements(record: dict | None, branch: str) -> dict | None:
    """Read the three shifted meals and the final measured 150 g insulin value."""
    if record is None:
        return None
    shifts = {s["trajectory"]: s for s in record["shifted_90g"]}
    ids = ["train_003", "train_006", "train_007"]
    if set(shifts) != set(ids):
        raise ValueError("missing or extra shifted 90 g trajectory")
    tail = record["extension_150g"]
    names = (
        ("tau_M", "gain_M", "k_M_rise")
        if branch == "c15-3"
        else ("k_prod", "Gp_thr", "prod_max")
    )
    return {
        "training_nmse": record["training_nmse"],
        "target_nmse": record["target_nmse"],
        "minima": {c: v["minimum"] for c, v in record["domain"].items()},
        "sampled_domain_passed": record["sampled_domain_passed"],
        "shape": {n: record["parameters"][n] for n in names},
        "peak_minutes": [shifts[i]["predicted_insulin_peak_min"] for i in ids],
        "observed_peak_minutes": [shifts[i]["observed_insulin_peak_min"] for i in ids],
        "peak_errors_minutes": [shifts[i]["peak_time_error_min"] for i in ids],
        "peak_unique": [shifts[i]["predicted_peak_unique"] for i in ids],
        "I300": tail["I300"],
        "observed_I300": tail["observed_I300"],
        "I300_error": tail["I300"] - tail["observed_I300"],
        "max_scaled_solver_difference": record["maximum_scaled_solver_difference"],
        "parameter_sha256": record["parameter_sha256"],
    }


def fit_start(root: Path, task: str, data: dict, plan: dict) -> dict:
    """Search one lineage only; replay checkpoints and charge interrupted blocks."""
    branch = TASKS[task][0]
    start = plan["starts"][task]
    model = campaign._model(branch, data)
    if start_parameters(model, task) != start:
        raise ValueError("declared start and frozen start differ")
    folder = root / task
    identity = {"plan_sha256": campaign.public.content_sha256(plan), "task": task}
    with campaign.public._lock(folder):
        refs, verified, known, errors = {}, {}, {}, {}
        history = []

        def operation(name: str, seconds: int, work: Callable) -> dict:
            op = campaign.journal._operation(folder, identity, name, seconds, work)
            refs[name] = campaign.public.content_sha256(op)
            if op["status"] == "complete":
                return op["value"]
            errors[name] = op.get("error", op["status"])
            return op.get("observed_progress") or {}

        def replay(name: str, p: dict) -> None:
            sha = campaign.public.content_sha256(p)
            if sha in known:
                return
            result = operation(
                name,
                REPLAY_SECONDS + 60,
                lambda _: campaign._verified(model, p, data, REPLAY_SECONDS),
            )
            if result.get("two_original_solvers_verified"):
                if result["parameter_sha256"] != sha:
                    raise ValueError("original replay bound to a different point")
                verified[name] = result
                known[sha] = name

        replay("start", start)
        if "start" not in verified:
            raise ValueError(f"{task} start replay unavailable: {errors}")
        for block, physical, calls in BLOCKS:
            # The optimizer can keep or resume only this task's own verified points.
            seeds = {"start": start}
            for admissible in (False, True):
                origin, row = retained(verified, admissible=admissible)
                if row and origin != "start":
                    seeds[origin] = row["parameters"]

            def search(save, seeds=seeds, physical=physical, calls=calls):
                return campaign.fitting.search(
                    model,
                    seeds,
                    physical=physical,
                    control=False,
                    seconds=SEARCH_SECONDS,
                    calls=calls,
                    save=save,
                )

            result = operation("search_" + block, SEARCH_SECONDS + 510, search)
            for kind in ("numerical", "admissible"):
                point = (result.get("best") or {}).get(kind)
                if point:
                    replay(f"{block}_{kind}", point["parameters"])
            history.append(
                {
                    "block": block,
                    "call_cap": calls,
                    "calls": result.get("search_calls_started"),
                    "valid_calls": result.get("search_calls_completed"),
                    "stop": result.get("status"),
                    "solver": result.get("solver_status"),
                    "seeds": {
                        n: campaign.public.content_sha256(p) for n, p in seeds.items()
                    },
                    "best_numerical": measurements(
                        retained(verified, admissible=False)[1], branch
                    ),
                    "best_admissible": measurements(
                        retained(verified, admissible=True)[1], branch
                    ),
                }
            )
        numerical = retained(verified, admissible=False)[1]
        feasible = retained(verified, admissible=True)[1]
        fitted = all(
            h["valid_calls"]
            and h["stop"] in ("budget_exhausted", "solver_returned", "solver_stopped")
            for h in history
        )
        status = "complete" if not errors and fitted else "incomplete"
        row = {
            "identity": identity,
            "branch": branch,
            "status": status,
            "errors": errors,
            "measurements_at_start": measurements(verified["start"], branch),
            "history": history,
            "best_numerical": numerical,
            "best_admissible": feasible,
            "operation_receipts": refs,
            "global_optimality_certified": False,
        }
        campaign.seal(folder / "result.json", row)
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    data = campaign.intake.load_archive(args.archive)
    root = args.root.resolve()
    plan = prepare(root, data)
    print("Frozen plan", campaign.public.content_sha256(plan), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_start, root, task, data, plan) for task in TASKS]
        rows = [future.result() for future in futures]
    summary = {
        "plan_sha256": campaign.public.content_sha256(plan),
        "training_sha256": data["training_sha256"],
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "tasks": {
            r["identity"]["task"]: {
                "result_sha256": campaign.public.content_sha256(r),
                "status": r["status"],
                "errors": r["errors"],
                "initial": r["measurements_at_start"],
                "history": r["history"],
                "best": measurements(r["best_admissible"], r["branch"]),
                "numerical": measurements(r["best_numerical"], r["branch"]),
            }
            for r in rows
        },
        "validation_or_test_access": False,
    }
    with campaign.public._lock(root):
        campaign.seal(root / "summary.json", summary)
    print(json.dumps({t: v["best"] for t, v in summary["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
