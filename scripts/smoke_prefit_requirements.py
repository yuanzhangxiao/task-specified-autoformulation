#!/usr/bin/env python3
"""Exercise saved-model global feedback, rollback, preservation and exact resume."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from autoformalism.data import BenchmarkLoader
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import prefit_construction_campaign as construction
from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.rebuttal.prefit_construction_audit import audit_task
from autoformalism.rebuttal.prefit_feedback import EpisodeClient
from autoformalism.search.requirement_feedback import RequirementBinding
from autoformalism.staged_topology import content_hash

if __package__:
    from scripts.smoke_prefit_construction import (
        CELL,
        client_for,
        synthetic_fixture,
        synthetic_transport,
    )
else:
    from smoke_prefit_construction import (
        CELL,
        client_for,
        synthetic_fixture,
        synthetic_transport,
    )

REQUIREMENT = "An input-driven causal pathway with nonlinear feedback to v01"


def synthetic_plan(temporary: Path) -> dict:
    """Construct an untagged linear feedback gap and a nonlinear retention control."""
    source, root = temporary / "source", temporary / "campaign"
    original = synthetic_fixture(source, construction_only=True)
    plan = {k: v for k, v in original.items() if k != "artifact_sha256"}
    brief = plan["cells"][CELL]["brief"]
    brief["requirements"][0]["public_requirement"] = REQUIREMENT
    brief["scientific_context"] = REQUIREMENT
    atomic_json(source / "plan.json", {**plan, "artifact_sha256": content_hash(plan)})
    plan = construction.verify(source)
    for task in plan["tasks"]:
        calls = []
        original_transport = synthetic_transport(calls)

        def transport(
            url, body, timeout, task=task, original_transport=original_transport
        ):
            payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
            record = original_transport(url, body, timeout)
            reply = json.loads(record["choices"][0]["message"]["content"])
            if payload.get("selected_lhs", {}).get("name") == "m":
                reply["terms"][0]["sources"] = ["m", "u01", "v01"]
            if payload.get("selected_equation", {}).get("lhs") == "m":
                function = reply["functions"][0]
                function["expression"] += "+c*v01" + (
                    "**2" if task["arm"] == "training_evidence" else ""
                )
                function["parameters"].append({"name": "c", "role": "coefficient"})
            record["choices"][0]["message"]["content"] = json.dumps(reply)
            return record

        client = client_for(source, plan, task, calls)
        client.transport = transport
        assert (
            construction.construct_task(source, plan, task, client)["status"]
            == "complete"
        )
        assert audit_task(source, plan, task)["status"] == "passed"
    settings = construction.StagedModelSettings(
        **{**plan["config"]["model_settings"], "maximum_requests": 3}
    )
    config = campaign.RequirementConfig(
        serving_image_sha256="0" * 64,
        model_settings=settings,
        seeds=(0,),
        expected_source_models=2,
        expected_source_plan_sha256=plan["artifact_sha256"],
        bindings=(
            RequirementBinding(
                cell=CELL, requirement_id="memory", public_requirement=REQUIREMENT
            ),
        ),
    )
    config_path = temporary / "config.json"
    config_path.write_text(config.model_dump_json())
    return campaign.freeze(source, config_path, root)


def repair_client(
    root: Path,
    plan: dict,
    task: dict,
    calls: list,
    *,
    failing=False,
    can_start=lambda: True,
):
    """Use real cached transport/accounting with a controlled one-slot response."""

    def transport(url, body, timeout):
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        selected = next(
            s
            for s in payload["eligible_interactions"]
            if s["selected_term"]["lhs"] == "m"
        )
        reply = {
            **selected["current_reply"],
            "interaction_id": selected["interaction_id"],
        }
        if not failing:
            reply["expression"] = reply["expression"].replace("v01", "v01**2")
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return EpisodeClient(
        settings=campaign.RequirementConfig.model_validate(
            plan["config"]
        ).model_settings,
        base_url="http://unused",
        directory=root / "results" / task["task_id"] / "calls",
        namespace=content_hash([plan["artifact_sha256"], task]),
        seed=task["seed"],
        transport=transport,
        can_start=can_start,
    )


def main() -> None:
    """Exercise the complete CPU/GPU worker logic without fitting or a live model."""
    with tempfile.TemporaryDirectory(prefix="prefit-requirements-") as temporary:
        base = Path(temporary)
        plan = synthetic_plan(base)
        root, calls = base / "campaign", []
        with (
            patch.object(
                BenchmarkLoader, "load_training", side_effect=AssertionError("no data")
            ),
            patch.object(
                construction, "fit_candidate", side_effect=AssertionError("no fitting")
            ),
        ):
            for task in plan["tasks"]:
                first = campaign.run_episode(
                    root, plan, task, repair_client(root, plan, task, calls)
                )
                count = len(calls)
                assert (
                    campaign.run_episode(
                        root, plan, task, repair_client(root, plan, task, calls)
                    )
                    == first
                )
                assert len(calls) == count
        report = campaign.summarize(root)
        assert report["status"] == "complete"
        assert report["paired_repairs"] == {"feedback_resolved_local_gap": 1}
        assert len(calls) == 1
        print(
            json.dumps(
                {
                    "status": "pass",
                    "arms": report["arms"],
                    "paired_repairs": report["paired_repairs"],
                    "exact_resume": True,
                    "live_llm_calls": 0,
                    "parameter_fitting_performed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
