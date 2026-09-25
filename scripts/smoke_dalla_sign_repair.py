#!/usr/bin/env python3
"""Synthetic cached sign repair, two real rescue/pruning arms, and exact resume."""

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_rescue as rescue
from autoformalism.rebuttal import dalla_sign_repair as campaign
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.smoke_process_pruning import fixture as pruning_fixture


def fixture(
    root: Path, *, protocol: str = "dalla-sign-repair-1"
) -> tuple[Path, Path, Path]:
    """Create a public toy with one unprotected state-equation clearance gain."""
    toy = pruning_fixture(root / "toy")
    row = toy["rows"][0]
    request = row["parent"]["request"]
    request["profile"] = rescue.PROFILE
    request["base_candidate"]["state_equations"][0]["rhs"] += "+c*x"
    request["base_candidate"]["parameters"].append(
        {"name": "c", "scope": "global", "role": "coefficient", "domain": "real"}
    )
    request["parameter_guesses"]["c"] = -0.2
    fit = row["parent"]["fit"]
    fit["parameters"]["c"] = 0.0
    toy["cells"]["toy"]["brief"] = {
        "scientific_context": (
            "The output has clearance. Anonymous channels have no application labels."
        ),
        "public_variables": [{"name": "v01", "data_role": "target"}],
    }
    source = root / "source.json"
    result = {
        "task": row["task"],
        "selected_request": request,
        "selected_fit": fit,
        "test_data_opened": False,
    }
    result = sealed_write(root / "source-result.json", result)
    public._write(
        source,
        {
            "protocol": rescue.PROTOCOL,
            "plan_sha256": toy["artifact_sha256"],
            "cells": toy["cells"],
            "models": [{"task": row["task"], "result": result}],
            "test_data_opened": False,
            "interventions_evaluated": False,
        },
    )
    config = root / "config.json"
    version = "v2" if protocol == "dalla-sign-repair-2" else "v1"
    settings = public._read(campaign.REPO / f"configs/dalla_sign_repair_{version}.json")
    if version == "v2":
        settings["selected_task_ids"] = [row["task"]["task_id"]]
    public._write(config, settings)
    return source, config, root / "campaign"


def transport(url, body, timeout):
    """Return a prescribed toy sign decision through the real logging client."""
    payload = json.loads(body["messages"][1]["content"])
    reply = {
        "decisions": [
            {
                "slot_id": s["slot_id"],
                "outer_weight_sign": "negative",
                "basis": "public_task",
                "public_quote": "The output has clearance.",
                "rationale": "Clearance removes the modeled output.",
            }
            for s in payload["eligible_slots"]
        ]
    }
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
        ],
        "usage": {"total_tokens": 100},
    }


def direction_transport(url, body, timeout):
    """Exercise optional bad citations and a separately logged toy assessment."""
    payload = json.loads(body["messages"][1]["content"])
    if "proposed" in payload:
        reply = {
            "assessments": [
                {
                    "slot_id": d["slot_id"],
                    "verdict": "supported",
                    "quoted_text_supports_direction": False,
                    "rationale": "Public clearance supports the removal direction.",
                }
                for d in payload["proposed"]["decisions"]
            ]
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }
    response = transport(url, body, timeout)
    reply = json.loads(response["choices"][0]["message"]["content"])
    for d in reply["decisions"]:
        d.update(
            mechanism_role="sink",
            donor=None,
            recipient=None,
            public_quote="Paraphrased clearance, not an exact quotation",
        )
    response["choices"][0]["message"]["content"] = json.dumps(reply)
    return response


def run(root: Path, *, protocol: str = "dalla-sign-repair-1") -> dict:
    source, config, output = fixture(root, protocol=protocol)
    campaign.freeze(source, config, output)
    review = campaign.review_one(
        output,
        0,
        base_url="http://synthetic.invalid",
        transport=direction_transport
        if protocol == "dalla-sign-repair-2"
        else transport,
    )
    assert review["status"] == "repaired"
    assert campaign.fit_one(output, 0)["sign_integrity"]["passed"]
    report = campaign.report(output)
    assert report["status"] == "complete"
    row = report["rows"][0]
    assert row["arms"]["repaired"]["validation"]["normalized_mse"] < 1e-5
    before = {str(p): p.read_bytes() for p in output.rglob("*.json")}
    assert (
        campaign.review_one(
            output,
            0,
            base_url="http://synthetic.invalid",
            transport=lambda *a: (_ for _ in ()).throw(AssertionError("extra call")),
        )
        == review
    )
    campaign.fit_one(output, 0)
    assert before == {str(p): p.read_bytes() for p in output.rglob("*.json")}
    return {
        "status": "passed",
        "live_llm_calls": 0,
        "protocol": protocol,
        "recorded_mock_requests": review["physical_requests"],
        "benchmark_data_used": False,
        "exact_resume": True,
        "repaired_training": row["arms"]["repaired"]["training"]["normalized_mse"],
        "repaired_validation": row["arms"]["repaired"]["validation"]["normalized_mse"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("dalla-sign-repair-1", "dalla-sign-repair-2"),
        default="dalla-sign-repair-1",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sign-repair-smoke-") as directory:
        print(json.dumps(run(Path(directory), protocol=args.protocol), indent=2))
