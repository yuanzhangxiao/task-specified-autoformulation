#!/usr/bin/env python3
"""Small independent recovery smoke, including native sensitivity and resume."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scaled_recovery import REPO, run
from scaled_recovery_io import digest, prepare, sha, write

from autoformalism.fitting.conditional_collocation import common_initialization
from autoformalism.fitting.conditional_optimizer import save_point
from autoformalism.fitting.conditional_scaling import (
    partition_and_exponents,
    system_for,
)
from autoformalism.rebuttal.fitter_diagnostic import _write_bytes
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.scaled_alternating import ScaledAlternatingPlan
from autoformalism.rebuttal.scaled_alternating_gate import small_problem
from autoformalism.rebuttal.scaled_recovery_worker import runtime_versions


def control(source: Path) -> dict:
    """Construct interrupted plumbing records; never use benchmark observations."""
    problem = small_problem()
    system = system_for(problem)
    train = unpack_split(problem["splits"]["train"])
    plan = ScaledAlternatingPlan(
        screen_seconds=20,
        screen_point_seconds=10,
        refinement_seconds=20,
        replay_seconds=10,
        target_variables=300,
        minimum_intervals=8,
    )
    write(source / "problem.json", problem)
    old = {
        "tasks": ["joint", "alternating"],
        "plan": plan.model_dump(mode="json"),
        "runtime": runtime_versions(),
        "assets": {"problem.json": sha(source / "problem.json")},
        "test_fixture": "synthetic recovery smoke, not a benchmark",
    }
    old["identity"] = digest(old)
    write(source / "freeze.json", old)
    common, nodes = common_initialization(
        system,
        train,
        np.ones(len(system.names)),
        plan.settings(),
        partition_and_exponents(system),
        target_variables=300,
        minimum_intervals=8,
        warmup_seconds=0.001,
    )
    common.pop("identity")
    common["parameter_names"] = list(system.names)
    common["identity"] = digest(common)
    write(source / "common.json", common)
    buf = io.BytesIO()
    np.save(buf, nodes, allow_pickle=False)
    _write_bytes(source / "nodes.npy", buf.getvalue())
    write(
        source / "gate/result.json",
        {
            "identity": old["identity"],
            "pass": True,
            "common_identity": common["identity"],
            "nodes_sha256": sha(source / "nodes.npy"),
        },
    )
    for index, arm in enumerate(old["tasks"]):
        identity = digest({"freeze": old["identity"], "arm": arm})
        root = source / f"results/task_{index:03d}"
        point = np.ones(len(system.names)) * (1 if index == 0 else 0.97)
        save_point(
            root / "initializer/native",
            identity,
            nodes,
            point,
            {
                "common_identity": common["identity"],
                "parameters": dict(zip(system.names, point.tolist(), strict=True)),
                "objective": 3.0,
                "seconds": 1.0,
            },
        )
        write(
            root / "initializer/result.json",
            {"identity": identity, "status": "interrupted"},
        )
        write(
            root / "screen/result.json",
            {"identity": identity, "status": "interrupted", "best": None},
        )
        write(
            root / "refinement/result.json",
            {"identity": identity, "status": "no_feasible_screen", "calls": 0},
        )
    return old


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, local, durable = [args.output / p for p in ("source", "local", "durable")]
    control(source)
    prepare(source, local, REPO)
    results = []
    for index in (0, 1):
        result = run(local, durable, index)
        assert result["status"] == "complete" and result["strict"], result
        assert result["collocation_reruns"] == 0
        before = sha(durable / f"results/task_{index:03d}/result.json")
        assert run(local, durable, index) == result
        assert sha(durable / f"results/task_{index:03d}/result.json") == before
        results.append(
            {"arm": result["arm"], "strict": result["strict"], "resume_unchanged": True}
        )
    print(json.dumps({"pass": True, "results": results}))


if __name__ == "__main__":
    main()
