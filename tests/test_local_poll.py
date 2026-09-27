"""Local polling reaches small corrections without an initial wide sweep."""

from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting.directional_poll import poll_fit
from autoformalism.rebuttal.fitter_diagnostic import read_json


class Oracle:
    names = ("x",)
    lower, upper = np.array([-1000.0]), np.array([1000.0])

    def __init__(self, interrupt=None, fail=False):
        self.failures = SimpleNamespace(count=0)
        self.calls = 0
        self.interrupt, self.fail = interrupt, fail

    def vector(self, values):
        return np.array([values["x"]])

    def __call__(self, x):
        self.calls += 1
        if self.calls == self.interrupt:
            raise KeyboardInterrupt
        if self.fail:
            self.failures.count += 1
            return np.zeros(1)
        return x - 20.5


def run(path, oracle, policy="local", calls=11):
    return poll_fit(
        oracle,
        [{"x": 20.0}],
        scales=np.array([20.0]),
        max_calls=calls,
        seconds=10,
        checkpoint=path,
        identity="fixture",
        policy=policy,
    )


def test_local_correction_under_same_budget(tmp_path):
    local = run(tmp_path / "local.json", Oracle())
    wide = run(tmp_path / "wide.json", Oracle(), "wide")
    assert local["cost"] == 0 < wide["cost"]
    assert local["actual_residual_calls"] <= 11
    assert local["poll_radius"] < 1
    assert not local["stationarity_claimed"]


def test_local_resume_and_policy_identity(tmp_path):
    baseline = tmp_path / "baseline.json"
    resumed = tmp_path / "resumed.json"
    run(baseline, Oracle())
    with pytest.raises(KeyboardInterrupt):
        run(resumed, Oracle(interrupt=5))
    run(resumed, Oracle())
    assert read_json(baseline)["history"] == read_json(resumed)["history"]
    oracle = Oracle()
    run(resumed, oracle)
    assert oracle.calls == 0
    with pytest.raises(ValueError, match="policy"):
        run(resumed, oracle, "wide")


def test_local_failure_penalties_not_selected(tmp_path):
    result = run(tmp_path / "failed.json", Oracle(fail=True))
    assert result["parameters"] is None
    assert result["valid_residual_evaluations"] == 0
