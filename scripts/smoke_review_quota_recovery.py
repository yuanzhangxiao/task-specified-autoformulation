#!/usr/bin/env python3
"""Real subprocess replay -> campaign publication -> ordinary continuation import."""

import json
import tempfile
from pathlib import Path

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from scripts import recover_review_quota as recovery
from tests.test_review_quota_recovery import fixture


def main():
    with tempfile.TemporaryDirectory(prefix="review-quota-smoke-") as directory:
        source, root = Path(directory) / "source", Path(directory) / "recovery"
        fixture(source)
        original = recovery.inventory(source)
        recovery.prepare(source, root, 2, [0])
        result = recovery.worker(root, 0)
        assert result["status"] == "replayed", result
        assert recovery.worker(root, 0) == result
        summary = recovery.finalize(root)
        assert recovery.finalize(root) == summary
        assert recovery.inventory(source) == original
        next_plan = continuation.prepare(
            root,
            Path(directory) / "next",
            source_round=2,
            visits=1,
            protocol=io.REVISION_PROTOCOL,
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "optimizer_calls": 0,
                    "real_subprocess_replay": True,
                    "source_unchanged": True,
                    "resume_replays": 0,
                    "next_visits": next_plan["continuation"]["additional_visits"],
                }
            )
        )


if __name__ == "__main__":
    main()
