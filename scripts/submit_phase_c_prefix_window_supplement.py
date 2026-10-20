#!/usr/bin/env python3
"""Submit M45 numerical gaps from a frozen M44 snapshot to Delta CPUs."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from autoformalism.fitting import prefix_window_supplement as campaign

if __package__:
    from scripts import submit_phase_c_generic_recovery as scheduler
else:
    import submit_phase_c_generic_recovery as scheduler

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _PreparedCampaign:
    """Adapt an already audited snapshot without repeating the source read."""

    root: Path
    policy: campaign.Policy
    plan: dict
    data: dict

    def prepare(
        self, root: Path, inputs: Path | None, policy: campaign.Policy
    ) -> None:
        """Require the same root and policy at the scheduler handoff."""
        if root != self.root or inputs != self.root or policy != self.policy:
            raise ValueError("prepared campaign identity differs")

    def verify(self, root: Path) -> tuple[dict, dict]:
        """Return the plan verified immediately before scheduler submission."""
        if root != self.root:
            raise ValueError("prepared campaign root differs")
        return self.plan, self.data


def submit(
    root: Path,
    source: Path | None,
    config: Path,
    *,
    account: str = "bibo-delta-cpu",
    concurrency: int = 2,
) -> dict:
    """Freeze missing work and retain the scheduler's no-duplicate receipts."""
    python = os.environ.get("AF_PYTHON", "")
    if not Path(python).is_file() or not 1 <= concurrency <= 8:
        raise ValueError("require an existing Python and concurrency 1..8")
    root = root.resolve()
    print(
        f"M45: reading saved M44 artifacts at {source or 'prepared snapshot'}; "
        f"output={root}",
        file=sys.stderr,
        flush=True,
    )
    policy = campaign.Policy.model_validate_json(config.read_text())
    campaign.prepare(root, source, policy)
    print(
        "M45: source frozen; verifying supplementary task plan",
        file=sys.stderr,
        flush=True,
    )
    plan, data = campaign.verify(root)
    if not plan["tasks"]:
        return {
            "status": "no_numerical_gaps",
            "jobs": {},
            "array_tasks": 0,
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
        }
    print(
        f"M45: {len(plan['tasks'])} incomplete tasks selected; "
        "checking CPU submission receipts",
        file=sys.stderr,
        flush=True,
    )
    return scheduler.submit(
        root,
        config,
        inputs=root,
        account=account,
        concurrency=concurrency,
        campaign=_PreparedCampaign(root, policy, plan, data),
        policy_type=campaign.Policy,
        worker_name="run_phase_c_prefix_window_supplement_delta.sh",
        job_prefix="prefix-window-supplement",
        run_time="00:20:00",
        prepare_time="00:10:00",
    )


def main() -> None:
    """Print a small scheduler receipt, with the complete manifest on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="Saved M44 root; omit only when resuming a prepared supplement.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/phase_c_prefix_window_supplement_v1.json",
    )
    parser.add_argument("--account", default="bibo-delta-cpu")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    result = submit(**vars(args))
    fields = (
        "status",
        "jobs",
        "array_tasks",
        "gpus",
        "live_llm_calls",
        "test_data_opened",
    )
    receipt = {name: result[name] for name in fields if name in result}
    artifact = (
        "plan.json" if result.get("status") == "no_numerical_gaps"
        else "submission_manifest.json"
    )
    receipt["artifact"] = str((args.root / artifact).resolve())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
