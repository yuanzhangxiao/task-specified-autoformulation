#!/usr/bin/env python3
"""Exercise historical response replay with synthetic cached calls and no fitting."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from autoformalism.rebuttal import prefit_construction_campaign as construction
from autoformalism.rebuttal.prefit_replay import replay

if __package__:
    from scripts.smoke_prefit_construction import (
        client_for,
        synthetic_fixture,
        synthetic_transport,
    )
else:
    from smoke_prefit_construction import (
        client_for,
        synthetic_fixture,
        synthetic_transport,
    )


def synthetic_source(root: Path) -> dict:
    """Run actual construction with prescribed defects, never call a fitter."""
    plan = synthetic_fixture(root)
    plan.pop("artifact_sha256")
    plan.update(test_data_opened=False, private_reference_opened=False)
    (root / "plan.json").unlink()
    construction._sealed_write(root / "plan.json", plan)
    plan = construction._sealed_read(root / "plan.json")
    for task in plan["tasks"]:
        calls = []
        ordinary = synthetic_transport(calls)
        client = client_for(root, plan, task, calls)

        def transport(url, body, timeout, ordinary=ordinary, task=task):
            raw = ordinary(url, body, timeout)
            payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
            if "selected_term" in payload:
                raw["choices"][0]["message"]["content"] = json.dumps(
                    {
                        "expression": "-a*m+b*u01",
                        "parameters": [
                            {"name": "a", "role": "rate"},
                            {"name": "b", "role": "coefficient"},
                        ],
                    }
                )
            elif payload.get("selected_equation", {}).get("lhs") == "m":
                raw["choices"][0]["message"]["content"] = json.dumps(
                    {"functions": [{"expression": "m", "parameters": []}]}
                )
            elif "selected_state" in payload and task["arm"] == "training_evidence":
                reply = json.loads(raw["choices"][0]["message"]["content"])
                reply["initial"]["expression"] = "m = a+b*v01"
                raw["choices"][0]["message"]["content"] = json.dumps(reply)
            return raw

        client.transport = transport
        with patch(
            "autoformalism.search.causal_initialization._normalize_choice",
            lambda choice, *args: (choice, None),
        ):
            construction.construct_task(root, plan, task, client)
    return plan


def main() -> None:
    """Verify historical failure retention, normalization, and exact replay."""
    with tempfile.TemporaryDirectory(prefix="prefit-replay-smoke-") as directory:
        root = Path(directory)
        synthetic_source(root / "source")
        result = replay(root / "source", root / "replay.json")
        assert result["counts"]["historical_complete"] == 1
        assert result["counts"]["normalization_recovered"] == 3
        assert replay(root / "source", root / "replay.json") == result
        print(
            json.dumps(
                {
                    "status": "pass",
                    "counts": result["counts"],
                    "exact_resume": True,
                    "live_llm_calls": 0,
                    "parameter_fitting_performed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
