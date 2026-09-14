#!/usr/bin/env python3
"""Construct both evidence arms and audit them without fitting or validation files."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from smoke_prefit_construction import client_for, synthetic_fixture

from autoformalism.data import BenchmarkLoader
from autoformalism.rebuttal import prefit_construction_campaign as campaign
from autoformalism.rebuttal.prefit_construction_audit import audit_task


def main() -> None:
    """Exercise construction, audit and exact resume with a stub provider."""
    with (
        tempfile.TemporaryDirectory(prefix="prefit-audit-smoke-") as temporary,
        patch.object(
            campaign, "fit_candidate", side_effect=AssertionError("no fitting")
        ),
        patch.object(
            BenchmarkLoader,
            "load_development",
            side_effect=AssertionError("no validation"),
        ),
        patch.object(
            BenchmarkLoader, "load_test", side_effect=AssertionError("no test")
        ),
    ):
        root = Path(temporary)
        plan = synthetic_fixture(root, construction_only=True)
        calls = []
        for task in plan["tasks"]:
            built = campaign.construct_task(
                root, plan, task, client_for(root, plan, task, calls)
            )
            assert built["status"] == "complete", built
            audited = audit_task(root, plan, task)
            assert audited["status"] == "passed", audited
            count = len(calls)
            assert (
                campaign.construct_task(
                    root, plan, task, client_for(root, plan, task, calls)
                )
                == built
            )
            assert audit_task(root, plan, task) == audited
            assert count == len(calls)
        summary = campaign.summarize(root)
        assert summary["status"] == "complete"
        assert not list(root.rglob("validation.csv"))
        assert not list(root.rglob("fit*.json"))
        print(
            json.dumps(
                {
                    "status": "pass",
                    "arms": summary["arms"],
                    "exact_resume": True,
                    "parameter_fitting_performed": False,
                    "validation_data_opened": False,
                    "live_llm_calls": 0,
                    "test_data_opened": False,
                    "private_reference_opened": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
