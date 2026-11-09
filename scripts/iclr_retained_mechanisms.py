"""Apply the submitted functional rubric to authenticated retained model visits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import mechanism_functional as functional
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.iclr_search_history import INVENTORY
from scripts.iclr_submission_evidence import (
    ARCHIVE,
    PLAN,
    REPO,
    digest,
    load_archive,
    table,
)


def freeze(args) -> dict:
    """Join model receipts to exact source rounds and pin scorer, prompts and data."""
    search, campaign = load_archive(args.archive)
    inv = sealed_read(args.inventory)
    if inv["plan_sha256"] != PLAN or inv["test_data_opened"] is not False:
        raise ValueError("wrong captured inventory")
    cfg = json.loads((REPO / "configs/mechanism_functional_v1.json").read_text())
    visits = set(args.rounds)
    if not visits or min(visits) < 0 or max(visits) >= campaign["config"]["rounds"]:
        raise ValueError("invalid requested historical visit")
    rows = {
        (r["task_id"], r["round"]): r for r in search["rows"] if r["round"] in visits
    }
    jobs = {}
    retained = {}
    for entry in inv["occurrences"]:
        task = entry["task"]
        key = task["task_id"], entry["round"]
        if entry["role"] != "selected" or key not in rows:
            continue
        source = rows[key]
        if source["result_sha256"] != entry["source_sha256"]:
            if source["result_sha256"] is None:
                continue  # The inspection can include later campaign progress.
            raise ValueError("conflicting source round")
        model = inv["models"][entry["model_id"]]
        if (
            content_hash(model) != entry["model_id"]
            or model["cell"] != task["cell"]
            or model["parameters"] != entry["fit"]["parameters"]
            or model["lowered_candidate_sha256"]
            != entry["fit"]["lowered_candidate_sha256"]
            or model["parameter_status"] != "saved_fit_vector"
            or entry["fit"]["status"] != "complete"
            or entry["fit"]["validation_initials_fitted"] is not False
        ):
            raise ValueError("invalid model/fit pairing")
        cell = task["cell"]
        assets = campaign["public_assets"][cell]
        if cfg["cells"][cell]["public_prompt_sha256"] != assets["proposer_prompt.txt"]:
            raise ValueError("rubric/actual public prompt mismatch")
        row = {
            "context": model["lowered_context"],
            "candidate": model["candidate"],
            "parameters": model["parameters"],
            "initials": {},
            "semantics": "continuous_time",
        }
        data = {
            "cell": cell,
            "input_sha256": assets["train.csv"],
            "row": row,
            "rules": cfg["cells"][cell]["mechanisms"],
        }
        identity = content_hash(data)
        jobs[identity] = data
        if key in retained:
            raise ValueError("duplicate retained model")
        retained[key] = identity
    plan_rows = [
        {
            "task": task,
            "round": step,
            "result_sha256": r["result_sha256"],
            "cell": r["cell"],
            "job": retained.get((task, step)),
            "status": (
                "ready"
                if (task, step) in retained
                else "model_export_missing"
                if r["retained_validation_nmse"] is not None
                else "no_retained_model"
            ),
        }
        for (task, step), r in sorted(rows.items())
    ]
    for cell in {j["cell"] for j in jobs.values()}:
        if (
            digest(args.training_root / cell / "train.csv")
            != campaign["public_assets"][cell]["train.csv"]
        ):
            raise ValueError("training data differ from submitted receipt")
    return sealed_write(
        args.output / "plan.json",
        {
            "protocol": "iclr-retained-functional-1",
            "plan": PLAN,
            "source_inventory": inv["artifact_sha256"],
            "source": digest(Path(__file__)),
            "scorer_sha256": digest(Path(functional.__file__)),
            "runtime": public._runtime(),
            "settings": cfg["numerical_settings"],
            "rows": plan_rows,
            "jobs": jobs,
        },
    )


@lru_cache(maxsize=8)
def read_training(path: Path, expected: str, roles_json: str) -> DatasetSplit:
    """Read the original training CSV with its declared channel roles."""
    if digest(path) != expected:
        raise ValueError("public training CSV changed")
    roles = json.loads(roles_json)
    result = []
    with path.open(newline="") as f:
        for name, group in groupby(csv.DictReader(f), key=lambda r: r["trajectory_id"]):
            points = list(group)
            arrays = {
                role: {v: np.array([float(r[v]) for r in points]) for v in roles[role]}
                for role in ("targets", "auxiliaries", "external_inputs")
            }
            if roles["fixed_covariates"] or roles["lagged_targets"]:
                raise ValueError("unsupported historical fixed/lagged channels")
            result.append(
                Trajectory(
                    name,
                    np.array([float(r["t"]) for r in points]),
                    **arrays,
                    fixed_covariates={},
                    derivatives={},
                )
            )
    return DatasetSplit(SplitName.TRAIN, tuple(result), expected)


def score(task: tuple) -> None:
    """Reuse identical equation, parameter and data evaluations."""
    root, training, key, job, plan_id, settings = task
    directory = root / "jobs" / key
    with public._lock(directory):
        path = directory / "score.json"
        if path.is_file():
            previous = sealed_read(path)
            if previous["plan_sha256"] != plan_id or previous["job"] != key:
                raise ValueError("functional job identity changed")
            return
        tr = read_training(
            training / job["cell"] / "train.csv",
            job["input_sha256"],
            json.dumps(job["row"]["context"], sort_keys=True),
        )
        findings = functional.assess(
            job["row"],
            job["rules"],
            tr,
            functional.Settings.model_validate(settings),
            directory / "rollout",
        )
        counts = Counter(r["status"] for r in findings)
        if len(findings) != len(job["rules"]) or set(counts) - {
            "pass",
            "fail",
            "unresolved",
        }:
            raise ValueError("incomplete operational assessment")
        sealed_write(
            path,
            {
                "job": key,
                "plan_sha256": plan_id,
                "findings": findings,
                "confirmed": counts["pass"] / len(findings),
                "counts": dict(counts),
            },
        )


def run(args) -> None:
    """Support bounded local/Slurm shards and deterministic restart of every row."""
    plan = sealed_read(args.output / "plan.json")
    if (
        digest(Path(__file__)) != plan["source"]
        or digest(Path(functional.__file__)) != plan["scorer_sha256"]
        or public._runtime() != plan["runtime"]
    ):
        raise ValueError("changed evaluator or numerical runtime; use a new root")
    if args.workers < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid worker count or shard")
    tasks = [
        (
            args.output,
            args.training_root,
            key,
            j,
            plan["artifact_sha256"],
            plan["settings"],
        )
        for n, (key, j) in enumerate(sorted(plan["jobs"].items()))
        if n % args.shards == args.shard
    ]
    if args.workers == 1:
        for item in tasks:
            score(item)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(score, tasks):
                pass


def report(root: Path) -> None:
    """Retain unscored/missing visits and preserve the pass/fail/unresolved counts."""
    plan = sealed_read(root / "plan.json")
    out = []
    for r in plan["rows"]:
        row = dict(r)
        if r["job"]:
            path = root / "jobs" / r["job"] / "score.json"
            if path.is_file():
                result = sealed_read(path)
                if (
                    result["plan_sha256"] != plan["artifact_sha256"]
                    or result["job"] != r["job"]
                ):
                    raise ValueError("assessment identity differs")
                row.update(
                    status="assessed",
                    confirmed=result["confirmed"],
                    **{
                        k: result["counts"].get(k, 0)
                        for k in ("pass", "fail", "unresolved")
                    },
                )
            else:
                row["status"] = "pending"
        out.append(row)
    table(root / "functional_visits.csv", out)
    print(dict(Counter(r["status"] for r in out)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run", "report"))
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--inventory", type=Path, default=INVENTORY)
    p.add_argument("--training-root", type=Path, default=REPO)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", nargs="+", type=int, default=[5])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    a = p.parse_args()
    if a.stage == "prepare":
        with public._lock(a.output):
            plan = freeze(a)
            print(f"{len(plan['jobs'])} unique fits for {len(plan['rows'])} visits")
    elif a.stage == "run":
        run(a)
    else:
        report(a.output)
