"""Run the complete matched function comparison on a small synthetic trial."""

import json

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import trajectory_functions as trial
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_trajectory_function_families import event, packet


def trajectory(i):
    t = np.linspace(0, 1, 9)
    return event(f"train_{i:03}", t, {"y": i + t**2 / 2}, {"u": t})


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    case = "synthetic"
    graph = {
        "variable_types": {"y": "differential"},
        "sources": {"y": ["u"]},
        "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
    }
    frozen = packet(
        [trajectory(0), trajectory(1)],
        ValidationContext(targets=("y",), external_inputs=("u",)),
    )
    frozen["public_brief"] = {
        "scientific_context": "Describe the measured response to u."
    }
    rows = frozen["evidence"].pop("trajectories")
    train_path = tmp_path / "train.json"
    source = sealed_write(
        train_path,
        {
            "cases": {case: frozen},
            "graphs": [
                {
                    "case": case,
                    "graph": graph,
                    "structure_sha256": content_hash(graph),
                    "origin": "pool",
                    "dataset": "synthetic",
                }
            ],
            "train": {case: rows},
        },
    )
    val = {**rows[0], "trajectory_id": "val_000"}
    # Validation deliberately favors the second proposer, reversing training order.
    val["targets"] = {"y": (np.array(val["time"]) ** 2).tolist()}
    valid_path = tmp_path / "val.json"
    heldout = sealed_write(
        valid_path,
        {
            "split": "val",
            "rows": {case: [val]},
            "training_intake_sha256": source["artifact_sha256"],
        },
    )
    monkeypatch.setattr(trial, "TRAIN", source["artifact_sha256"])
    monkeypatch.setattr(trial, "VALIDATION", heldout["artifact_sha256"])
    monkeypatch.setattr(
        trial, "BUDGET", {"max_evaluations": 4, "starts": 1, "wall_seconds": 15.0}
    )
    # Isolate study logic from concurrent source edits and OS-specific build tools.
    monkeypatch.setattr(trial, "runtime_source_hash", lambda: "test-runtime")
    root = tmp_path / "run"
    plan = trial.prepare(root, train_path)
    assert plan["maximum_generation_requests"] == 4 and plan["maximum_fit_slots"] == 6
    return root, valid_path


def generate(root):
    packets = []

    def provider(url, body, timeout):
        req = json.loads(body["messages"][1]["content"])
        assert "val_000" not in json.dumps(req)
        if "shortlist" in req:
            assert "evidence_for_functions" in req
            candidate = {"shortlist_id": "A"}
        else:
            assert "evidence_for_functions" not in req
            rhs = "u" if not packets else "2*u"
            candidate = {"equations": {"y": rhs}, "initializers": {}, "parameters": []}
        packets.append(req)
        return reply({"candidate": candidate})

    result = trial.run(
        root, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert len(packets) == result["physical_requests"] == 4
    assert result["candidate_statuses"] == {"eligible": 6}
    return result


def test_frozen_selection_resume_and_damaged_validation(campaign, monkeypatch):
    root, validation = campaign
    with pytest.raises(FileNotFoundError):
        trial.fit(root, 0)
    expected = generate(root)
    assert expected["generation_complete"] and expected["validation_records"] == 0

    def duplicate(*a, **kw):
        pytest.fail("A completed request or fit was repeated")

    assert trial.run(root, "http://localhost", transport=duplicate) == expected
    fits = [trial.fit(root, i) for i in range(6)]
    assert all(f["result"]["status"] == "fitted" for f in fits)
    from autoformalism.research import trajectory_function_fitting as fitting

    monkeypatch.setattr(fitting, "fit_formula", duplicate)
    assert [trial.fit(root, i) for i in range(6)] == fits
    assert fits[0]["result"]["training_nmse"] < fits[1]["result"]["training_nmse"]
    plan, intake = trial.load(root)
    selections, _ = trial.finalists(root, plan, intake)
    assert selections["winners"][0] == {
        "graph_index": 0,
        "arm": "proposer",
        "winner": 0,
    }
    wrong = sealed_write(root / "other-val.json", {"training_intake_sha256": "other"})
    with pytest.raises(ValueError, match="Validation intake"):
        trial.validate(root, root / "other-val.json")
    assert wrong["artifact_sha256"] != plan["validation_identity"]
    summary = trial.validate(root, validation)
    assert summary["status"] == "complete" and summary["validation_records"] == 3
    decision = root / "validation" / "000.json"
    winner = sealed_read(decision)
    assert winner["score"]["status"] == "scored" and winner["score"]["pooled_nmse"] > 0
    assert winner["score"]["curves"][0]["observed"]["y"][-1] == 1.0
    with monkeypatch.context() as saved:
        saved.setattr(fitting, "score_formula", duplicate)
        assert trial.validate(root, validation) == summary
    decision.unlink()
    assert trial.status(root)["status"] == "incomplete"
    assert trial.validate(root, validation) == summary
    assert sealed_read(decision) == winner
    # A separately sealed but unrelated receipt must fail provenance as well.
    decision.unlink()
    sealed_write(decision, {"selection": {}, "finalists": "unrelated", "score": {}})
    with pytest.raises(ValueError):
        trial.status(root)
    with pytest.raises(ValueError):
        trial.validate(root, validation)


def test_corrupted_physical_generation_cannot_resume(campaign):
    root, _ = campaign
    generate(root)
    (physical,) = (root / "steps" / "slot-000" / "calls").glob("*.json")
    physical.write_text(physical.read_text() + "\n")
    with pytest.raises(ValueError, match="generation receipt"):
        trial.run(root, "http://localhost", transport=lambda *a: pytest.fail("retry"))
