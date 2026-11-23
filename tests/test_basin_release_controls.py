"""Public split boundaries, known outlet controls and disconnected rollouts."""

from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import basin_release_controls as controls


def event(index, depth, *, split="train", gain=8.0, power=1.7):
    flow = gain * max(depth - 0.1, 0.0) ** power
    return SimpleNamespace(
        trajectory_id=f"{split}_{index:03}",
        time=[0.0, 1.0, 1.5, 3.0, 5.0],
        targets={"h_down": [depth] * 5},
        inputs={"inflow_down": [flow] * 5},
        covariates={"area_down": 120.0, "crest_down": 0.1},
    )


def test_independent_fit_requires_no_upstream_values_and_uses_all_intervals():
    rows = [event(i, h) for i, h in enumerate([0.1, 0.2, 0.3, 0.5, 0.9])]
    for stride in controls.balance.STRIDES:
        fit = controls.independent_estimate(rows, stride)
        assert fit["downstream"]["status"] == "estimated"
        assert fit["downstream"]["best"]["power"] == pytest.approx(1.7, abs=1e-5)
        assert fit["downstream"]["best"]["gain"] == pytest.approx(8.0, abs=1e-4)
        assert fit["downstream"]["intervals"] == sum(
            x["intervals"] for x in fit["events"].values()
        )
        assert "upstream" not in fit
    with pytest.raises(ValueError, match="training"):
        controls.independent_estimate([event(0, 0.2, split="val")], 1)


def test_signed_discharge_estimates_are_preserved(monkeypatch):
    row = event(0, 0.2)
    row.targets["h_down"] = [0.2, 0.4, 0.4, 0.5, 0.5]
    row.inputs["inflow_down"] = [0] * 5
    seen = []

    def inspect(samples):
        seen.extend(samples[0].volume)
        return {"status": "unidentified_power"}

    monkeypatch.setattr(controls.balance, "fit_power", inspect)
    result = controls.independent_estimate([row], 1)
    np.testing.assert_allclose(seen, [-24, 0, -12, 0])
    assert result["events"][row.trajectory_id]["negative_outlet_intervals"] == 2


def test_disconnected_rollout_ignores_upstream_inputs_and_later_observations():
    row = event(0, 0.3, split="val")
    row.inputs["inflow_up"] = [10, 100, 0, 15, 1000]
    freeze = {
        "training_depth_std_m": 0.2,
        "laws": {"downstream": {"power": 1.7, "gain": 8.0}},
    }
    base = controls.independent_score([row], freeze)
    assert base["status"] == "scored" and base["pooled_nmse"] < 1e-22
    assert base["upstream_intervention_max_change_m"] == 0
    row.targets["h_down"][-1] += 0.2
    changed = controls.independent_score([row], freeze)
    assert base["events"][0]["predicted"] == changed["events"][0]["predicted"]
    assert changed["pooled_nmse"] == pytest.approx(0.2)


def test_whitelist_and_freeze_precede_access_to_any_validation_file(
    tmp_path, monkeypatch
):
    cell = next(iter(controls.FILES))
    with pytest.raises(ValueError, match="registered"):
        controls._path(tmp_path, cell, "test")
    with pytest.raises(ValueError, match="registered"):
        controls._path(tmp_path, "unregistered", "train")
    root = tmp_path / "results"
    public = tmp_path / "public"
    sealed_write(root / "plan.json", {"seen": False})
    monkeypatch.setattr(
        controls, "_path", lambda *a: pytest.fail("opened a public file")
    )
    with pytest.raises(FileNotFoundError, match=r"training\.json"):
        controls.evaluate(public, root, "val")
    with pytest.raises(ValueError, match="train or val"):
        controls.evaluate(public, root, "test")
