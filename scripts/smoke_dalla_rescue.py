#!/usr/bin/env python3
"""Tiny real refit/pruning smoke; no Dalla benchmark data or LLM calls."""

import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import dalla_rescue as campaign
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.smoke_process_pruning import fixture


def run(root: Path) -> dict:
    toy = fixture(root / "toy")
    row = toy["rows"][0]
    source, output = root / "inputs.json", root / "rescue"
    sealed_write(
        source,
        {
            "protocol": campaign.INPUT_PROTOCOL,
            "test_data_opened": False,
            "cells": toy["cells"],
            "rows": [
                {
                    "task": row["task"],
                    "request": row["parent"]["request"],
                    "parameters": row["parent"]["fit"]["parameters"],
                }
            ],
        },
    )
    campaign.freeze(source, output)
    result = campaign.run_one(output, 0)
    assert result["status"] == "complete", result
    assert result["selected_fit"]["validation"]["normalized_mse"] < 1e-6
    assert result["pruning"]["selection"]["pruned_accepted"]
    before = {str(p): p.read_bytes() for p in (output / "results").rglob("*.json")}
    assert campaign.run_one(output, 0) == result
    assert before == {
        str(p): p.read_bytes() for p in (output / "results").rglob("*.json")
    }
    assert campaign.report(output)["status"] == "complete"
    return {
        "status": "passed",
        "real_fits": 3,
        "resume_additional_fits": 0,
        "live_llm_calls": 0,
        "benchmark_data_used": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="dalla-rescue-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
