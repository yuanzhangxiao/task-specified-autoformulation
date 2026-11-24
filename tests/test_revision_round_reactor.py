"""Reactor edits receive identical training gates and untouched validation starts."""

import math

import pytest

from scripts import revision_round_reactor as reactor
from tests.test_thermal_balance_trial import rows


@pytest.fixture
def reactor_rows(monkeypatch):
    train = rows()
    held = [train[0].model_copy(update={"trajectory_id": "val_000"})]
    reads = []
    spec = {"targets": ["T"], "external_inputs": ["Tf"], "auxiliaries": ["C", "Tj"]}

    def load(_root, _dataset, split):
        reads.append(split)
        return {"specification": spec, "train": train, "val": held}[split]

    monkeypatch.setattr(reactor.thermal, "_read", load)
    monkeypatch.setattr(reactor, "PARTITIONS", (3, 1))
    monkeypatch.setattr(reactor, "EVALUATIONS", 2)
    reactor._training.cache_clear()
    yield train, held, reads
    reactor._training.cache_clear()


@pytest.mark.parametrize(
    "edit",
    [
        {},
        {"heat_law": "arbitrary", "concentration_power": 1},
        {"heat_law": "concentration", "concentration_power": True},
        {"heat_law": "concentration", "concentration_power": math.nan},
        {"heat_law": "concentration", "concentration_power": 5},
        {"heat_law": "concentration", "concentration_power": 1, "also": "edit"},
    ],
)
def test_reactor_rejects_invalid_edit_without_opening_data(edit, reactor_rows):
    assert not reactor.score(edit)["eligible"]
    assert not reactor_rows[-1]


def test_parent_and_child_training_and_frozen_validation(reactor_rows):
    train, held, reads = reactor_rows
    packet = reactor.packet()
    assert packet["edit_schema"]["additionalProperties"] is False
    assert packet["evidence"]["training_events"] == len(train)
    assert set(reads) == {"specification", "train"}
    parent = packet["parent"]
    fit = reactor.score(parent)
    assert fit["train_nmse"] < 1e-6
    assert fit["optimizer_evaluations"] <= 8
    assert set(reads) == {"specification", "train"}
    edited = {
        "heat_law": "exponential_inverse_temperature",
        "concentration_power": 1.25,
    }
    child = reactor.score(edited)
    assert child["eligible"] and child["edit"] == edited
    assert "val" not in reads
    with pytest.raises(ValueError, match="same revision"):
        reactor.validate(edited, fit)
    result = reactor.validate(parent, fit)
    assert result["nmse"] < 1e-6
    assert list(result["per_event"]) == [held[0].trajectory_id]
    assert reads[-1] == "val"


def test_fractional_concentration_transform_is_split_local_and_nonmutating():
    r = rows()[-1]
    c = r.auxiliaries["C"][0]
    assert reactor._with_power([r], 0.5)[0].auxiliaries["C"][0] == pytest.approx(c**0.5)
    assert r.auxiliaries["C"][0] == c
    bad = r.model_copy(
        update={"auxiliaries": {**r.auxiliaries, "C": [-1] * len(r.time)}}
    )
    with pytest.raises(ValueError, match="nonnegative"):
        reactor._with_power([bad], 0.5)


def test_training_interval_screen_covers_grid_and_separates_folds(
    reactor_rows, monkeypatch
):
    train, _, reads = reactor_rows
    reactor._training()
    original = reactor.thermal.fit
    subsets = []

    def traced(rows, laws):
        subsets.append({r.trajectory_id for r in rows})
        return original(rows, laws)

    monkeypatch.setattr(reactor.thermal, "fit", traced)
    result = reactor.screen()
    assert result["screened"] == 11
    assert result["interval_family_fits"] == 22
    assert len(result["edits"]) == 3
    assert all(reactor._valid(edit) for edit in result["edits"])
    assert all(record["edit"] != reactor.PARENT for record in result["profile"])
    assert all(record["cv_rate_rmse_K_per_min"] >= 0 for record in result["profile"])
    assert all(
        a.isdisjoint(b) and a | b == {r.trajectory_id for r in train}
        for a, b in zip(subsets[::2], subsets[1::2], strict=True)
    )
    assert "val" not in reads
