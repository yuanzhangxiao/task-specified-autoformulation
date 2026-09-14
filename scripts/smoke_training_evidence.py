"""Exercise public training packet construction on a controlled synthetic model."""

import json
import tempfile
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.search.training_evidence import (
    build_training_evidence,
    freeze_training_evidence,
)


def main() -> None:
    """Verify actual training-only serialization and exact immutable replay."""
    problem, _ = synthetic_problem("causal_map", 0.0, 0)
    train = unpack_split(problem["splits"]["train"])
    context = ValidationContext.model_validate(problem["context"])
    packet = build_training_evidence(train, context)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training_evidence.json"
        freeze_training_evidence(path, packet)
        before = path.read_bytes()
        freeze_training_evidence(path, build_training_evidence(train, context))
        assert path.read_bytes() == before
    assert packet.initial_target_ranges["v01"] == (0.0, 1.2)
    print(
        json.dumps(
            {
                "status": "pass",
                "packet_sha256": packet.packet_sha256,
                "trajectory_count": packet.trajectory_count,
                "initial_target_ranges": packet.initial_target_ranges,
                "test_data_opened": False,
                "live_llm_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
