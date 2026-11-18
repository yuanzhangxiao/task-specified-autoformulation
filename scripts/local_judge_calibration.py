"""Run or bind the same known-case critic checks on a local Delta service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.rebuttal import local_judge_calibration as study
from autoformalism.rebuttal.prefit_replay import sealed_read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "report", "bind"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--image-sha256")
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.source is None or args.image_sha256 is None:
            parser.error("freeze requires --source and --image-sha256")
        value = study.freeze(args.source, args.root, args.image_sha256)
    elif args.stage == "run":
        study.run(
            args.root,
            os.environ["AF_LOCAL_CRITIC_URL"],
            lambda: os.environ["AF_LOCAL_CRITIC_KEY"],
        )
        value = study.report(args.root)
    elif args.stage == "bind":
        if args.campaign is None:
            parser.error("bind requires --campaign")
        value = study.bind(
            args.campaign, sealed_read(args.campaign / "plan.json"), args.root
        )
    else:
        value = study.report(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k
                in {"artifact_sha256", "identity", "status", "known_case_gate_passed"}
            }
        )
    )


if __name__ == "__main__":
    main()
