"""Verify the development/fresh boundary and joint fitting/feedback decisions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import graph_critic_advice as advice
from scripts import run_graph_critic_probes as runner


def _endpoint():
    return {
        "training_nmse": 0.1,
        "status": "ready",
        "reason": None,
        "trajectories": 16,
        "compared_windows": 144,
        "window": {
            "trajectory_id": "train_005",
            "target": "Gp",
            "start": 200,
            "end": 300,
            "observed_change": 1,
            "predicted_change": 8,
        },
        "balance": {
            "status": "ready",
            "state": "Gp",
            "terms": [{"expression": "rate", "sign": 1, "signed_integral": 8.0}],
        },
        "probes": [
            {
                "driver": d,
                "status": "ready",
                "paths": {t: [f"state:{d}", f"target:{t}"]},
                "outcomes": {
                    t: {
                        "rmse_reduction": delta,
                        "terminal_error_reduction": delta,
                        "free_rmse": 1.0,
                    }
                },
            }
            for d, t, delta in (("I", "Gp", -1), ("Gp", "I", 1))
        ],
    }


def _search():
    return {"search_scope": "local_free_fit", "budget_exhausted": False}


def test_descriptive_substitution_survives_restricted_fit_but_routing_changes():
    rows = {"a": _endpoint(), "b": _endpoint()}
    search = {"a": _search(), "b": _search()}
    packet = advice.summarize(rows, search)
    assert packet["readiness"] == "conditional_equation_review"
    assert "rate (+8 to +8)" in " ".join(packet["deterministic_feedback"])
    search["b"]["budget_exhausted"] = True
    limited = advice.summarize(rows, search)
    assert limited["readiness"] == "continue_fitting"
    assert limited["evidence"]["probe-1"] == packet["evidence"]["probe-1"]
    rows["b"]["window"]["trajectory_id"] = "another"
    assert not any(
        "Substituting" in line
        for line in advice.summarize(rows, search)["deterministic_feedback"]
    )
    rows["b"]["training_nmse"] = None
    assert advice.summarize(rows, search)["readiness"] == "insufficient_evidence"


def test_freeze_precedes_fresh_results_and_resume_reuses_verified_endpoints(
    tmp_path, monkeypatch
):
    loaded, probed = [], []
    context = SimpleNamespace(
        equation_expressions={"Gp": SimpleNamespace(source="rate")},
        process_expressions={},
        observation_expressions={},
    )
    point = SimpleNamespace(
        training_nmse=0.1, parameters={}, initial_states={}, search_status=_search()
    )

    def load(parent, repo):
        loaded.append(parent)
        return SimpleNamespace(
            starts={"a": point, "b": point},
            training=object(),
            public_prompt="Example",
            model=SimpleNamespace(validated=context),
            provenance={"problem_sha256": parent, "training_sha256": "training"},
        )

    def probe(*args, **kwargs):
        probed.append(True)
        return deepcopy(_endpoint())

    monkeypatch.setattr(runner.parents, "load_parent", load)
    monkeypatch.setattr(runner.parents, "ROSTER", ("development", "fresh"))
    monkeypatch.setattr(runner.parents, "TRANSFER_CONTROL", "control", raising=False)
    monkeypatch.setattr(
        runner.parents,
        "TERMINALS",
        {p: {"a": "a", "b": "b"} for p in ("development", "fresh", "control")},
    )
    monkeypatch.setattr(runner.probes, "probe_endpoint", probe)
    with pytest.raises(FileNotFoundError):
        runner.campaign("run", tmp_path, tmp_path)
    assert loaded == []
    frozen = runner.campaign("freeze", tmp_path, tmp_path)
    assert loaded == ["development"] and len(probed) == 2
    runner.campaign("run", tmp_path, tmp_path)
    assert len(probed) == 6
    assert read_seal(tmp_path / "feedback/control.json")["cohort"] == "transfer_control"
    assert (
        read_seal(tmp_path / "feedback/fresh.json")["freeze_sha256"]
        == frozen["freeze_sha256"]
    )
    runner.campaign("run", tmp_path, tmp_path)
    assert len(probed) == 6
    monkeypatch.setitem(runner.advice.RULES, "minimum_starts", 3)
    with pytest.raises(ValueError, match="frozen rules"):
        runner.campaign("run", tmp_path, tmp_path)
