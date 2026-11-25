"""Verify the training boundary, synthetic controls, and resumable experiment."""

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import evaluate_process_nomination as pilot


def test_known_rates_are_interval_means_and_holdout_is_independent():
    assert pilot._mean_truth(np.array([-1, 1, 3]), "linear", 0) == pytest.approx(
        [0.425, 3.4]
    )
    for kind in ("linear", "saturating"):
        result = pilot.synthetic_case(kind, 0, 11)
        candidate = result["shortlist"][0]
        assert candidate["recovers"] and candidate["new_event_rate_nmse"] < 1e-12
    missed = pilot.synthetic_case("turnover", 0.03, 11)
    assert missed["status"] == "menu_contradicted" and not missed["shortlist"]
    rows = pilot.synthetic_summary([missed])
    assert sum(r["attempted"] for r in rows) == 1
    assert sum(r["returned"] for r in rows) == 0


def test_checkpoint_reuse_and_mismatched_fold(tmp_path):
    path = tmp_path / "fold.json"
    assert pilot._checkpoint(path, "plan", "train_000", lambda: {"result": 1})
    assert pilot._checkpoint(
        path, "plan", "train_000", lambda: pytest.fail("repeated")
    ) == {"result": 1}
    assert sealed_read(path)["case"] == "train_000"
    with pytest.raises(ValueError, match="another experiment or fold"):
        pilot._checkpoint(path, "plan", "train_001", lambda: {})
