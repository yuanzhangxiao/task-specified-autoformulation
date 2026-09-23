#!/usr/bin/env python3
"""Synthetic shared-law pruning -> two real frozen fits -> deterministic resume."""

import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import process_pruning as campaign
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from scripts.smoke_public_fitting import control


def fixture(root: Path) -> dict:
    """Prescribed toy snapshot; no historical experiment or stored benchmark data."""
    request, train, val = control("collocation-single-target-v2")
    raw = request.model_dump(mode="json")
    raw["context"]["external_inputs"] = ["u01"]
    candidate = raw["base_candidate"]
    candidate["processes"] = [
        {"name": "q", "expression": "z"},
        {"name": "r", "expression": "eps*x"},
    ]
    candidate["state_equations"] = [
        {"state": "x", "rhs": "u01-a*x+q+r"},
        {"state": "z", "rhs": "-b*q"},
    ]
    candidate["parameters"].append(
        {"name": "eps", "scope": "global", "role": "coefficient", "domain": "real"}
    )
    raw["parameter_guesses"]["eps"] = 0
    request = PublicFitRequest.model_validate(raw)
    splits = []
    for split in (train, val):
        data = split.model_dump(mode="json")
        for row in data["rows"]:
            row["external_inputs"] = {"u01": [0.0] * len(row["time"])}
        splits.append(PublicSplit.model_validate(data))
    train, val = splits
    model, _, _ = public._lower(request)
    metric = {
        "available": True,
        "normalized_mse": 0.0,
        "failed_trajectories": [],
        "per_target_normalized_mse": {"v01": 0.0},
    }
    fit = {
        "status": "complete",
        "parameters": {"a": 0.7, "b": 1.2, "eps": 0.0, "init_z_scale": 0.5},
        "training": metric,
        "validation": metric,
    }
    cell = {
        "training": train.model_dump(mode="json"),
        "validation": val.model_dump(mode="json"),
        "mechanism_spec": {
            "benchmark_id": "toy",
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "causal",
                    "required_drivers": ["u01"],
                    "required_targets": ["v01"],
                    "requires_dynamic_memory": True,
                }
            ],
        },
        "target_contract": {
            "benchmark_id": "toy",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "targets": [
                {
                    "target_channel": "v01",
                    "public_requirement": "Generate output causally.",
                }
            ],
        },
    }
    task = {"task_id": "toy_seed0_full", "cell": "toy", "arm": "full", "seed": 0}
    parent = {
        "request": request.model_dump(mode="json"),
        "fit": fit,
        "bundle": {"candidate": model.validated.candidate.model_dump(mode="json")},
    }
    row = {
        "task": task,
        "parent": parent,
        "certificate": campaign.certificate(request, cell, task),
    }
    return sealed_write(
        root / "plan.json",
        {
            "protocol": campaign.PROTOCOL,
            "policy": campaign.POLICY,
            "source": str(root / "synthetic_source"),
            "rows": [row],
            "cells": {"toy": cell},
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "launcher_sha256": campaign.launcher_hash(),
            "test_data_opened": False,
        },
    )


def run(root: Path) -> dict:
    fixture(root)
    result = campaign.run_one(root, 0)
    assert result["choice"]["status"] == "ready", result["choice"]
    assert result["choice"]["audit"]["removed"]["parameters"] == ["eps"]
    for fit in result["fits"].values():
        assert fit["status"] == "complete", fit
        assert fit["validation"]["normalized_mse"] < 1e-6, fit
    assert result["selection"]["selected"] == "pruned"
    before = {str(p): p.read_bytes() for p in (root / "results").rglob("*.json")}
    assert campaign.run_one(root, 0) == result
    assert before == {
        str(p): p.read_bytes() for p in (root / "results").rglob("*.json")
    }
    assert campaign.report(root)["status_counts"] == {"complete": 1}
    return {
        "status": "passed",
        "real_fits": 2,
        "live_llm_calls": 0,
        "resume_additional_fits": 0,
        "removed_parameters": ["eps"],
        "shared_q_consumers_retained": ["x", "z"],
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="process-pruning-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
