#!/usr/bin/env python3
"""Run both packet arms through real construction and fitting on synthetic data."""

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from autoformalism.rebuttal import prefit_construction_campaign as campaign
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
)
from autoformalism.search.training_evidence import build_training_evidence

CELL = "phase_b_anonymous_system_task_canonical_opaque_hard"


def synthetic_fixture(
    root: Path,
    request_budget: int = 32,
    *,
    construction_only: bool = False,
    with_validation: bool = False,
) -> dict:
    """Create temporary public-layout observations; do not modify benchmark assets."""
    config = campaign.ConstructionCampaignConfig(
        protocol=campaign.CONSTRUCTION_ONLY_PROTOCOL
        if construction_only
        else "prefit-matched-construction-1",
        serving_image_sha256="0" * 64,
        model_settings=campaign.StagedModelSettings(maximum_requests=request_budget),
        public_cells=(CELL,),
        seeds=(0,),
        fit=None
        if construction_only
        else campaign.FitConfig(
            allow_derivative_regression=False,
            integration_method="Radau",
            maximum_function_evaluations=100,
            maximum_wall_time_seconds=30,
        ),
    )
    directory = root / "public/phase_b_v1" / CELL
    directory.mkdir(parents=True)
    (directory / "proposer_prompt.txt").write_text("Synthetic driven memory fixture\n")
    for split, initials, amplitudes in (
        ("train", (0.0, 0.6, 1.2), (0.0, 0.5, 1.0)),
        ("validation", (1.5, 1.8), (0.0, 0.8)),
    ):
        if construction_only and not with_validation and split != "train":
            continue
        with (directory / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("trajectory_id", "t", "v01", "u01"))
            time = np.arange(51) * 0.1
            for i, (y0, u) in enumerate(zip(initials, amplitudes, strict=True)):
                baseline = 0.8 * u / 0.3
                y = (
                    y0 * np.exp(-0.7 * time)
                    + baseline * (1 - np.exp(-0.7 * time)) / 0.7
                    + (1 + 2 * y0 - baseline)
                    * (np.exp(-0.3 * time) - np.exp(-0.7 * time))
                    / 0.4
                )
                writer.writerows(
                    (f"{split}_{i}", float(t), float(v), u)
                    for t, v in zip(time, y, strict=True)
                )
    campaign.atomic_json(
        directory / "manifest.json",
        {
            "schema_version": "phase_b_public_release_v1",
            "status": "production_registered",
            "test_sealed": True,
            "benchmark_id": CELL,
            "tier": "hard",
            "channels": [
                {"public_name": "v01", "role": "target"},
                {"public_name": "u01", "role": "external_input"},
            ],
            "splits": {
                "validation": "1" * 64,
                **{
                    s: hashlib.sha256((directory / f"{s}.csv").read_bytes()).hexdigest()
                    for s in (
                        ("train",)
                        if construction_only and not with_validation
                        else ("train", "validation")
                    )
                },
                "test": "0" * 64,
            },
        },
    )
    context = campaign.public_validation_context(CELL)
    training = (
        campaign.load_training(root / "public", CELL)
        if construction_only
        else campaign.load_development(root / "public", CELL).train
    )
    brief = PublicScientificBrief(
        scientific_context="Synthetic delayed input response. Generate v01 "
        "from u01 through internal memory.",
        public_variables=(
            PublicVariable(name="v01", data_role="target"),
            PublicVariable(name="u01", data_role="external_input"),
        ),
        requirements=(
            ScientificRequirement(
                id="memory",
                public_requirement="A delayed response of v01 to u01",
                targets=("v01",),
                drivers=("u01",),
                requires_dynamic_memory=True,
            ),
        ),
    )
    packet = build_training_evidence(training, context)
    cell = {
        "assets": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in campaign.public_files(config)
        },
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "evidence": packet.model_dump(mode="json"),
    }
    plan = {
        "protocol": config.protocol,
        "config": config.model_dump(mode="json"),
        "cells": {CELL: cell},
        "tasks": [
            {"task_id": f"{CELL}_seed0_{arm}", "cell": CELL, "seed": 0, "arm": arm}
            for arm in campaign.ARMS
        ],
        "runtime_source_sha256": campaign.runtime_source_hash(),
        "numerical_runtime": campaign.runtime_identity(),
        "launcher_sha256": campaign.launcher_hash(),
        "synthetic_fixture": True,
    }
    campaign._sealed_write(root / "plan.json", plan)
    return campaign.verify(root)


def synthetic_transport(calls):
    """Prescribe the same scientific model in both arms; exercise actual schemas."""

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_state" in payload:
            reply = {
                "initial": {
                    "mode": "causal_map",
                    "expression": "a+b*v01",
                    "parameters": [
                        {"name": "a", "role": "coefficient"},
                        {"name": "b", "role": "coefficient"},
                    ],
                }
            }
        elif "selected_equation" in payload:
            lhs = payload["selected_equation"]["lhs"]
            function = (
                {"expression": "m-k*v01", "parameters": [{"name": "k", "role": "rate"}]}
                if lhs == "v01"
                else {
                    "expression": "-a*m+b*u01",
                    "parameters": [
                        {"name": "a", "role": "rate"},
                        {"name": "b", "role": "coefficient"},
                    ],
                }
            )
            reply = {"functions": [function]}
        elif "selected_lhs" in payload:
            sources = (
                ["m", "v01"]
                if payload["selected_lhs"]["name"] == "v01"
                else ["m", "u01"]
            )
            reply = {
                "terms": [
                    {
                        "sources": sources,
                        "outer_weight_sign": "unrestricted",
                        "scientific_role": "driven relaxation",
                    }
                ],
                "inventory_revision": None,
            }
        else:
            reply = {
                "variables": [
                    {
                        "name": "v01",
                        "definition": "differential",
                        "scientific_role": "observed response",
                    },
                    {
                        "name": "m",
                        "definition": "differential",
                        "scientific_role": "input memory",
                    },
                ]
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return transport


def client_for(root, plan, task, calls):
    """Use the same physical-budget implementation as the live controller."""
    config = campaign.ConstructionCampaignConfig.model_validate(plan["config"])
    return campaign.BudgetedRepairClient(
        settings=config.model_settings,
        seed=task["seed"],
        directory=root / "results" / task["task_id"] / "calls",
        namespace=campaign._identity(plan, task),
        base_url="http://unused",
        transport=synthetic_transport(calls),
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="prefit-construction-smoke-") as temporary:
        root = Path(temporary)
        plan = synthetic_fixture(root)
        calls = []
        for task in plan["tasks"]:
            built = campaign.construct_task(
                root, plan, task, client_for(root, plan, task, calls)
            )
            assert built["status"] == "complete", built
            fitted = campaign.fit_task(root, plan, task)
            assert fitted["status"] == "evaluated", fitted
            assert fitted["fit"]["validation_normalized_mse"] < 1e-6, fitted
            count = len(calls)
            assert (
                campaign.construct_task(
                    root, plan, task, client_for(root, plan, task, calls)
                )
                == built
            )
            assert campaign.fit_task(root, plan, task) == fitted
            assert count == len(calls)
        summary = campaign.summarize(root)
        assert summary["status"] == "complete"
        print(
            json.dumps(
                {
                    "status": "pass",
                    "arms": summary["arms"],
                    "validation_nmse": [
                        row["first_validation_nmse"] for row in summary["rows"]
                    ],
                    "exact_resume": True,
                    "live_llm_calls": 0,
                    "test_data_opened": False,
                    "private_reference_opened": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
