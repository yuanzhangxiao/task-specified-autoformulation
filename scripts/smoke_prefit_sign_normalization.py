#!/usr/bin/env python3
"""Verify strict cached replay and normalized live repair without a model server."""

import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import prefit_requirements as campaign
from autoformalism.search.requirement_feedback import TOPOLOGY_SIGN_REPAIR

if __package__:
    from scripts.smoke_prefit_requirements import repair_client, synthetic_plan
else:
    from smoke_prefit_requirements import repair_client, synthetic_plan


def signed_client(root, plan, task, calls):
    """Return a nonlinear law with a redundant explicit outer minus."""
    client = repair_client(root, plan, task, calls)
    original = client.transport

    def transport(url, body, timeout):
        result = original(url, body, timeout)
        reply = json.loads(result["choices"][0]["message"]["content"])
        reply["expression"] = "-(" + reply["expression"] + ")"
        result["choices"][0]["message"]["content"] = json.dumps(reply)
        return result

    client.transport = transport
    return client


def main():
    """Preserve historical files, replay cached replies and resume exactly."""
    with tempfile.TemporaryDirectory(prefix="prefit-sign-") as temporary:
        base = Path(temporary)
        previous = base / "campaign"
        old_plan = synthetic_plan(base, fixed_sign=True)
        old_calls = []
        for task in old_plan["tasks"]:
            campaign.run_episode(
                previous,
                old_plan,
                task,
                signed_client(previous, old_plan, task, old_calls),
            )
        assert len(old_calls) == 3
        original_files = {p: p.read_bytes() for p in previous.rglob("*.json")}
        config = {
            **old_plan["config"],
            "repair_policy": TOPOLOGY_SIGN_REPAIR,
            "replay_source_plan_sha256": old_plan["artifact_sha256"],
        }
        config_path = base / "normalized-config.json"
        config_path.write_text(json.dumps(config))
        root = base / "normalized"
        plan = campaign.freeze(base / "source", config_path, root)
        replay = campaign.replay_saved_repairs(root, previous)
        assert replay["newly_admissible"] == 3
        assert replay["acceptance_regressions"] == 0
        assert campaign.replay_saved_repairs(root, previous) == replay
        calls = []
        for task in plan["tasks"]:
            first = campaign.run_episode(
                root, plan, task, signed_client(root, plan, task, calls)
            )
            assert (
                campaign.run_episode(
                    root, plan, task, signed_client(root, plan, task, calls)
                )
                == first
            )
        assert len(calls) == 1
        assert original_files == {p: p.read_bytes() for p in previous.rglob("*.json")}
        report = campaign.summarize(root)
        assert (
            report["arms"]["requirement_feedback"]["repair"][
                "outer_sign_normalizations"
            ]
            == 1
        )
        assert (
            report["arms"]["requirement_feedback"]["control"]["physical_requests"] == 0
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "repair_policy": TOPOLOGY_SIGN_REPAIR,
                    "replay": {
                        k: replay[k]
                        for k in (
                            "saved_responses",
                            "strict_accepted",
                            "normalized_accepted",
                            "acceptance_regressions",
                        )
                    },
                    "fresh_synthetic_repair_calls": len(calls),
                    "exact_resume": True,
                    "historical_files_unchanged": True,
                    "live_llm_calls": 0,
                    "parameter_fitting_performed": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
