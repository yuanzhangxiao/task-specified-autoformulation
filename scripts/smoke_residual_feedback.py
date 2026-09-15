#!/usr/bin/env python3
"""Real training replay and immutable resume; parent timeout history is synthetic."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke_public_fit_continuation import fixture_parent

from autoformalism.fitting.fit_continuation import (
    execute_continuation,
    prepare_continuation,
)
from autoformalism.fitting.fit_residual_feedback import (
    load_residual_feedback,
    prepare_residual_feedback,
    run_residual_feedback,
)
from autoformalism.schemas.residual_feedback import FeedbackSelection


def fixture_history(root: Path):
    """Produce real scored parent and completed continuation with learned latent map."""
    parent, continuation = root / "parent/fit", root / "pilot/continuation"
    selection = fixture_parent(parent, real_scores=True)
    prepare_continuation(parent, selection, continuation)
    first = execute_continuation(continuation)
    assert first.status == "complete", first.message
    return (
        parent,
        continuation,
        FeedbackSelection(
            continuation_identity=first.identity,
            continuation_backend_sha256=first.backend_result_sha256,
        ),
    )


def smoke(root: Path) -> dict:
    """Exercise real causal-initialization replay without exposing held-out data."""
    parent, continuation, selection = fixture_history(root)
    before = {
        str(p): p.read_bytes()
        for source in (parent, continuation)
        for p in source.rglob("*.json")
    }
    output = root / "feedback"
    prepare_residual_feedback(parent, continuation, output, selection)
    result = run_residual_feedback(output)
    assert result["status"] == "ready", result
    assert result["current_score_agrees"] and result["previous_comparison_available"]
    packet = result["packet"]
    assert packet["normalized_mse"] < 1e-8
    assert packet["previous_normalized_mse"] > packet["normalized_mse"]
    assert all(r["trajectory_id"].startswith("train") for r in packet["rows"])
    assert run_residual_feedback(output) == load_residual_feedback(output) == result
    assert before == {
        str(p): p.read_bytes()
        for source in (parent, continuation)
        for p in source.rglob("*.json")
    }
    return {
        "status": "passed",
        "synthetic_parent_history": True,
        "real_training_replay": True,
        "full_causal_initial_vector_used": True,
        "optimization_calls_in_export": result["optimization_calls"],
        "parent_files_unchanged": True,
        "resume_unchanged": True,
        "rows": len(packet["rows"]),
        "details": len(packet["details"]),
    }


def main():
    with TemporaryDirectory(prefix="residual-feedback-") as temporary:
        print(json.dumps(smoke(Path(temporary)), indent=2))


if __name__ == "__main__":
    main()
