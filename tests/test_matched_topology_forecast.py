"""Exercise training-only selection, frozen scoring and interrupted fit accounting."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash
from scripts import matched_topology_forecast as runner


def experiment(tmp_path):
    rows = [
        {"inventory": "same", "arm": "algorithm", "graph_id": "a"},
        {"inventory": "same", "arm": "20b_repair", "graph_id": "b"},
    ]
    slots = []
    for index in range(len(rows)):
        for rank in range(2):
            key = f"{index}-{rank}"
            slot = {
                "graph_index": index,
                "rank": rank,
                "fit_key": key,
                "formula": {"rank": rank},
                "prefit": {"status": "eligible"},
            }
            slots.append(slot)
            status = "fitted" if index == 0 else "resolution_unverified"
            result = {
                "status": status,
                "training_nmse": 0.1 + rank,
                "parameters": {},
                "negative_samples": {"I": int(rank == 0)},
            }
            sealed_write(
                tmp_path / "fits" / f"{key}.json",
                {
                    "plan": "test",
                    "slot_sha256": content_hash(slot),
                    "result": result,
                },
            )
    return {"artifact_sha256": "test", "slots": slots, "cohort": {"rows": rows}}


def test_freeze_preserves_failures_and_lower_rank_nonnegative(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Validation opened before selection")

    monkeypatch.setattr(runner.adapter, "validate", forbidden)
    plan = experiment(tmp_path)
    frozen = runner.freeze(tmp_path, plan)
    assert frozen["winners"] == {"same": {"algorithm": "a", "20b_repair": None}}
    assert frozen["selections"][0]["rank"] == 0
    assert frozen["selections"][1]["fit_status"] == "no_verified_training_fit"
    assert frozen["nonnegative_winners"]["same"]["algorithm"]["rank"] == 1
    assert frozen["nonnegative_winners"]["same"]["20b_repair"] is None
    assert runner.freeze(tmp_path, plan) == frozen


def test_development_requires_matching_receipts_and_reuses_scores(
    tmp_path, monkeypatch
):
    plan = experiment(tmp_path)
    frozen = runner.freeze(tmp_path, plan)
    invalid = deepcopy(frozen)
    invalid["plan"] = "stale"
    with pytest.raises(ValueError, match="matching training freeze"):
        runner.development(tmp_path, plan, invalid)

    calls = []

    def validate(row, formula, result):
        calls.append(formula["rank"])
        return {"status": "scored", "pooled_nmse": 2.0}

    monkeypatch.setattr(runner.adapter, "validate", validate)
    scored = runner.development(tmp_path, plan, frozen)
    assert calls == [0, 1]
    assert scored["rows"][1]["score"] is None
    assert runner.development(tmp_path, plan, frozen) == scored
    assert calls == [0, 1]

    stale = deepcopy(plan)
    stale["slots"][0]["rank"] = 1
    with pytest.raises(ValueError, match="different plan or slot"):
        runner.freeze(tmp_path, stale)


def test_reserved_slot_cannot_spend_a_second_budget(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Recharged an interrupted fit")

    monkeypatch.setattr(runner.adapter, "fit", forbidden)
    slot = {"fit_key": "a", "rank": 0, "formula": {"x": 1}}
    sealed_write(
        tmp_path / "reservations/a.json",
        {"plan": "p", "slot_sha256": content_hash(slot)},
    )
    args = tmp_path, "p", {}, slot
    assert runner._run_slot(args) == "a"
    assert runner._receipt(tmp_path, slot, "p")["result"]["status"] == "interrupted"
    assert runner._run_slot(args) == "a"
