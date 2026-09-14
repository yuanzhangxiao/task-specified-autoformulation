#!/usr/bin/env python3
"""Exercise paired feedback, cached resume and controls with no fitting."""

import json
import tempfile
from pathlib import Path

from autoformalism.llm.staged_topology import StagedModelSettings
from autoformalism.rebuttal.prefit_feedback import (
    EpisodeClient,
    FeedbackConfig,
    freeze,
    run_episode,
    summarize,
)
from autoformalism.rebuttal.prefit_replay import replay
from autoformalism.staged_topology import content_hash

if __package__:
    from scripts.smoke_prefit_replay import synthetic_source
else:
    from smoke_prefit_replay import synthetic_source


def synthetic_plan(root: Path) -> dict:
    """Construct a genuine cached historical fixture and freeze its repair cases."""
    synthetic_source(root / "source")
    replay(root / "source", root / "corpus.json")
    config = FeedbackConfig(
        serving_image_sha256="0" * 64,
        model_settings=StagedModelSettings(maximum_requests=3),
        seeds=(0,),
    )
    (root / "config.json").write_text(config.model_dump_json())
    return freeze(root / "corpus.json", root / "config.json", root / "campaign")


def synthetic_transport(calls: list):
    """Repair only the prescribed missing-source defect; preserve valid controls."""

    def transport(url, body, timeout):
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        reply = payload["current_response"]
        if reply.get("expression") == "m":
            reply = {
                "expression": "-a*m+b*u01",
                "parameters": [
                    {"name": "a", "role": "rate"},
                    {"name": "b", "role": "coefficient"},
                ],
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return transport


def client_for(
    root: Path, plan: dict, task: dict, calls: list, **kwargs
) -> EpisodeClient:
    """Use exactly the same client identity and resume accounting as the worker."""
    return EpisodeClient(
        settings=StagedModelSettings.model_validate(plan["config"]["model_settings"]),
        base_url="http://unused",
        directory=root / "results" / task["task_id"] / "calls",
        namespace=content_hash([plan["artifact_sha256"], task]),
        seed=task["seed"],
        transport=synthetic_transport(calls),
        **kwargs,
    )


def main() -> None:
    """Run both arms and replay every terminal episode without physical calls."""
    with tempfile.TemporaryDirectory(prefix="prefit-feedback-smoke-") as directory:
        root = Path(directory)
        plan = synthetic_plan(root)
        campaign = root / "campaign"
        calls = []
        for task in plan["tasks"]:
            first = run_episode(
                campaign, plan, task, client_for(campaign, plan, task, calls)
            )
            assert first["final"]["valid"]
            count = len(calls)
            assert (
                run_episode(
                    campaign, plan, task, client_for(campaign, plan, task, calls)
                )
                == first
            )
            assert len(calls) == count
        report = summarize(campaign)
        assert report["status"] == "complete"
        for arm in report["arms"].values():
            assert arm["repair"]["valid_final"] == arm["repair"]["expected"]
            assert arm["valid_control"]["valid_control_changed"] == 0
        print(
            json.dumps(
                {
                    "status": "pass",
                    "episodes": len(plan["tasks"]),
                    "arms": report["arms"],
                    "exact_resume": True,
                    "live_llm_calls": 0,
                    "parameter_fitting_performed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
