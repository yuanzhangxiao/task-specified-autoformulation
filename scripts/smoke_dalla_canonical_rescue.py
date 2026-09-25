#!/usr/bin/env python3
"""Synthetic real-fit smoke for real-gain and already-constrained sign repairs."""

import copy
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_canonical_rescue as campaign
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import sign_review
from autoformalism.staged_topology import content_hash
from scripts.smoke_dalla_sign_diagnostic import fixture as old_fixture


def fixture(root: Path) -> tuple[Path, Path, Path]:
    source, _, output = old_fixture(root)
    old = public._read(source)
    original = old["models"][0]
    value = original["result"]
    request = PublicFitRequest.model_validate(value["selected_request"])
    slot = sign_review.slots(request)[0]
    rows, decisions = [], []
    for label, bounded in (("real_gain", False), ("constrained_gain", True)):
        raw = copy.deepcopy(request.model_dump(mode="json"))
        parameters = dict(value["selected_fit"]["parameters"])
        if bounded:
            for spec in raw["base_candidate"]["parameters"]:
                if spec["name"] == slot["parameter"]:
                    spec.update(role="nonnegative_coefficient", domain="nonnegative")
                    parameters[spec["name"]] = 0.0
        row = {
            "task": {**original["task"], "task_id": label},
            "request": raw,
            "parameters": parameters,
        }
        rows.append(row)
        decisions.append(
            {
                "task_id": label,
                "source_row_sha256": content_hash(row),
                "assumptions": [
                    "Synthetic assisted clearance; no benchmark or reference."
                ],
                "edits": [
                    {
                        "state": slot["state"],
                        "parameter": slot["parameter"],
                        "original_expression": slot["original_expression"],
                        "sign": "negative",
                        "rationale": "Use dissipative clearance in this synthetic toy.",
                    }
                ],
            }
        )
    source = root / "canonical-sources.json"
    sealed_write(
        source,
        {
            "protocol": campaign.SOURCE_PROTOCOL,
            "cells": old["cells"],
            "rows": rows,
            "test_data_opened": False,
            "intervention_data_used": False,
        },
    )
    config = root / "canonical-decisions.json"
    public._write(
        config,
        {
            "protocol": campaign.PROTOCOL,
            "decision_source": campaign.LABEL,
            "models": decisions,
        },
    )
    return source, config, output


def run(root: Path) -> dict:
    source, config, output = fixture(root)
    campaign.freeze(source, config, output)
    for index in (2, 0, 3, 1):
        campaign.run_one(output, index)
    result = campaign.report(output)
    assert result["status"] == "complete" and result["expected"] == 4
    repaired = [
        r
        for r in result["rows"]
        if r["task"]["diagnostic"]["comparison_arm"] == "repaired"
    ]
    assert all(r["sign_integrity"]["passed"] for r in repaired)
    assert all(r["validation_nmse"] < 1e-5 for r in repaired)
    before = {str(p): p.read_bytes() for p in output.rglob("*.json")}
    for index in range(4):
        campaign.run_one(output, index)
    assert result == campaign.report(output)
    assert before == {str(p): p.read_bytes() for p in output.rglob("*.json")}
    return {
        "status": "passed",
        "arms": 4,
        "exact_resume": True,
        "live_llm_calls": 0,
        "benchmark_data_used": False,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="canonical-sign-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
