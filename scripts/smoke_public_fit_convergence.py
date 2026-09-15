#!/usr/bin/env python3
"""Real toy numerical windows; original parent timeout history is synthetic."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from smoke_public_fit_continuation import fixture_parent

from autoformalism.fitting.fit_continuation import (
    execute_continuation,
    prepare_continuation,
)
from autoformalism.fitting.fit_convergence import (
    execute_convergence,
    prepare_convergence,
)
from autoformalism.schemas.fit_convergence import ConvergenceSelection


def smoke(root: Path) -> dict:
    """Check real parameter/initial-map carryover and immutable completed resume."""
    parent, continuation, output = (
        root / "parent/fit",
        root / "pilot/continuation",
        root / "diagnostic",
    )
    selection = fixture_parent(parent, real_scores=True)
    prepare_continuation(parent, selection, continuation)
    first = execute_continuation(continuation)
    assert first.status == "complete", first.message
    frozen = {
        str(p): p.read_bytes()
        for source in (parent, continuation)
        for p in source.rglob("*.json")
    }
    prepare_convergence(
        parent,
        continuation,
        ConvergenceSelection(
            continuation_identity=first.identity,
            continuation_backend_sha256=first.backend_result_sha256,
            maximum_windows=2,
        ),
        output,
    )
    result = execute_convergence(output)
    assert result["initial_state"]["parameters"] == dict(first.parameters)
    assert result["retained_state"]["training"]["normalized_mse"] < 1e-8
    assert result["retained_state"]["validation"]["normalized_mse"] < 1e-8
    assert result["status"] == "local_stationarity_reached", result
    assert execute_convergence(output) == result
    assert frozen == {
        str(p): p.read_bytes()
        for source in (parent, continuation)
        for p in source.rglob("*.json")
    }
    return {
        "status": "passed",
        "synthetic_original_parent_history": True,
        "real_numerical_windows": True,
        "parents_unchanged": True,
        "full_vector_carried": True,
        "resume_unchanged": True,
        "training_nmse": result["retained_state"]["training"]["normalized_mse"],
        "validation_nmse": result["retained_state"]["validation"]["normalized_mse"],
        "windows": result["completed_windows"],
    }


def main() -> None:
    with TemporaryDirectory(prefix="fit-convergence-") as temporary:
        print(json.dumps(smoke(Path(temporary)), indent=2))


if __name__ == "__main__":
    main()
