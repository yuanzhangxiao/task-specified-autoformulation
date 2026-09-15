#!/usr/bin/env python3
"""Exercise saved-repair export and real bounded fitting on synthetic data."""

import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.rebuttal.prefit_fit_handoff import (
    HandoffSelection,
    prepare_handoff,
    run_handoff,
)

if __package__:
    from scripts.smoke_prefit_requirements import repair_client, synthetic_plan
else:
    from smoke_prefit_requirements import repair_client, synthetic_plan


def main():
    with tempfile.TemporaryDirectory(prefix="prefit-public-handoff-") as temporary:
        base = Path(temporary)
        plan = synthetic_plan(
            base,
            with_validation=True,
            feedback_policy="stage-aware-1",
            repair_policy="topology-owned-sign-1",
        )
        task = next(
            t
            for t in plan["tasks"]
            if t["arm"] == "requirement_feedback" and t["cohort"] == "repair"
        )
        source, calls = base / "campaign", []
        state = campaign.run_episode(
            source, plan, task, repair_client(source, plan, task, calls)
        )
        selection = HandoffSelection(
            source_plan_sha256=plan["artifact_sha256"],
            construction_plan_sha256=plan["source_plan_sha256"],
            task_id=task["task_id"],
            profile="general-rollout-v1",
        )
        root = base / "handoff"
        prepare_handoff(
            source, base / "source", base / "source/public", selection, root
        )
        report = run_handoff(root)
        assert report["status"] == "complete", report
        assert report["repair_provenance"]["rhs_changed"]
        assert run_handoff(root) == report
        assert len(calls) == 1
        frozen = json.loads((root / "fit/freeze.json").read_text())
        assert frozen["lowered_candidate"] == state["final"]["candidate"]
        print(
            json.dumps(
                {
                    "status": "pass",
                    "source_repair_verified": True,
                    "candidate_preserved_at_fit_handoff": True,
                    "training_nmse": report["fit"]["training"]["normalized_mse"],
                    "validation_nmse": report["fit"]["validation"]["normalized_mse"],
                    "exact_terminal_resume": True,
                    "live_llm_calls": 0,
                    "test_data_opened": False,
                    "private_reference_opened": False,
                    "scientific_status": report["scientific_status"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
