"""Integration tests for matched comparisons, qualification and the split boundary."""

import pytest

from scripts import run_t2_disposal_coupled as campaign


def test_endpoint_backoff_requires_independent_feasibility(monkeypatch):
    p0 = {"x": 0.0}
    model = {"formula": {"parameters": p0}, "scalar_model": {"tau": 5}}
    source = {"events": ["source"]}
    rows = ["row"]
    seen = []

    def replay(candidate, r):
        assert r is rows
        x = candidate["formula"]["parameters"]["x"]
        seen.append(x)
        return {
            "domain": {"eligible": x <= 0.991},
            "summary": {"pooled_nmse": 1 - x, "failed_events": 0},
            "events": ["endpoint"],
        }

    monkeypatch.setattr(
        campaign.insulin,
        "with_parameters",
        lambda m, p: m | {"formula": {"parameters": p}},
    )
    pairs = []
    monkeypatch.setattr(
        campaign.comparisons, "paired", lambda a, b: pairs.append((a, b))
    )
    r = campaign.checked_attempt(
        model,
        source,
        rows,
        replay,
        {"parameters": {"x": 1.0}, "training_nmse": 0.0, "diagnostics": {}},
    )
    assert seen == [1.0, 0.999, 0.99]
    assert r["parameters"]["x"] == r["fraction"] == 0.99
    assert r["training_nmse"] == pytest.approx(0.01)
    assert pairs == [(["source"], ["endpoint"])]


def test_source_replay_mismatch_prevents_fitting_and_development(tmp_path, monkeypatch):
    packed = {tag: {"parameter": i} for i, tag in enumerate(campaign.SOURCES.values())}
    monkeypatch.setattr(campaign.data, "load_models", lambda _: packed)
    monkeypatch.setattr(campaign.data, "load_training", lambda _: [])
    monkeypatch.setattr(campaign, "plan", lambda *args: {"budget_per_arm_per_start": 2})
    monkeypatch.setattr(
        campaign, "cached_replay", lambda *args: {"domain": {"eligible": True}}
    )

    def fail(*args):
        raise ValueError("Source training replay differs")

    monkeypatch.setattr(campaign.data, "verify_start", fail)
    monkeypatch.setattr(
        campaign.data, "load_validation", lambda *_: pytest.fail("opened dev")
    )
    monkeypatch.setattr(campaign, "fit_arm", lambda *a, **kw: pytest.fail("fitted"))
    with pytest.raises(ValueError, match="Source training replay"):
        campaign.run(tmp_path, tmp_path, jobs=1, train_only=False, budget=2)
    assert not (tmp_path / "training-freeze.json").exists()


def test_fitting_uses_only_immediate_comparator_and_training_checkpoints(
    tmp_path, monkeypatch
):
    def make_model(p):
        return {"formula": {"parameters": {"x": p}}, "label": p, "scalar_model": {}}

    model = make_model(1)
    candidates = {"immediate_start": model, "feasible_reference": make_model(2)}
    plan = {"artifact_sha256": "plan", "budget_per_arm_per_start": 3}
    calls = []

    def replay(root, identity, split, m, rows):
        assert split == "train"
        return {
            "domain": {"eligible": True},
            "events": [],
            "summary": {
                "failed_events": 0,
                "pooled_nmse": 1 / m["formula"]["parameters"]["x"],
            },
        }

    def fit(m, rows, start, *, budget):
        calls.append((m["label"], start, budget))
        return {"parameters": {"x": 4}, "training_nmse": 0.25, "diagnostics": {}}

    monkeypatch.setattr(
        campaign.insulin,
        "with_parameters",
        lambda m, p: m | {"formula": {"parameters": p}},
    )
    monkeypatch.setattr(campaign.optimizer, "fit", fit)
    monkeypatch.setattr(campaign, "cached_replay", replay)
    monkeypatch.setattr(campaign.comparisons, "paired", lambda *args: {})
    p = {"models": candidates, "rows": [], "root": tmp_path, "design": plan}
    result = campaign.fit_arm("immediate", **p)
    assert result[1]["winner"] == "warm"
    assert campaign.fit_arm("immediate", **p) == result
    assert calls == [(1, "warm", 3), (1, "displaced", 3)]
    p["models"]["immediate_start"] = make_model(5)
    with pytest.raises(ValueError, match="different fit"):
        campaign.fit_arm("immediate", **p)
