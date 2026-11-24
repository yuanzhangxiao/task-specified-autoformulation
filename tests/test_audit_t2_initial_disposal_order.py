"""A time-zero replay must match the archived training model and observations."""

import pytest

from scripts.audit_t2_initial_disposal_order import verify_initials


def test_initials_match_exactly_and_cannot_be_swapped_between_training_vectors():
    event = {
        "event": "train_011",
        "time": [0],
        "observed": {"U": [1.99]},
        "predicted": {"U": [1.0]},
    }
    old = event | {"status": "scored"}
    saved = {"model_sha256": "source", "events": [old]}
    verify_initials([event], saved, "source")
    with pytest.raises(ValueError, match="parameter vector"):
        verify_initials([event], saved, "another fit")
    for patch in (
        {"event": "train_012"},
        {"time": [1]},
        {"observed": {"U": [2]}},
        {"predicted": {"U": [0.9]}},
    ):
        with pytest.raises(ValueError, match="Initial sample differs"):
            verify_initials([event], saved | {"events": [old | patch]}, "source")
