"""Check archived graph identity and the two-arm, training-only control."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import check_constructor_portability as pilot


def test_entire_graph_must_match_replay_even_with_matching_structure_id(
    tmp_path, monkeypatch
):
    replay = tmp_path / "replay.json"
    replay.write_text("{}")
    monkeypatch.setattr(pilot.intake, "REPLAY_FILE", pilot.file_hash(replay))
    roster = [
        {
            "dataset": pilot.BASIN,
            "structure_sha256": s,
            "graph": {"sources": {"x": [s]}},
        }
        for s in ("one", "two")
    ]
    monkeypatch.setattr(pilot.intake, "select_graphs", lambda raw: roster)
    source = {
        "source_replay": pilot.file_hash(replay),
        "graphs": deepcopy(roster),
        "artifact_sha256": pilot.original.TRAIN,
    }
    assert pilot.registered_basin_graphs(source, replay) == roster
    source["graphs"][0]["graph"]["sources"]["x"] = ["a_new_edge"]
    with pytest.raises(ValueError, match="differ"):
        pilot.registered_basin_graphs(source, replay)


def test_prepare_masks_identity_and_held_events_for_both_constructors(
    tmp_path, monkeypatch
):
    ids = [f"train_{i:03}" for i in range(6)]
    events = [{"trajectory_id": name} for name in ids]
    data = sealed_write(
        tmp_path / "train.json",
        {
            "source_replay": "example",
            "cases": {
                "case": {
                    "evidence": {
                        "context": {"time_symbol": "t"},
                        "contract": {"interpolation": {}},
                    }
                }
            },
            "train": {pilot.BASIN: events},
        },
    )
    monkeypatch.setattr(pilot.experiment, "TRAIN_FILE", tmp_path / "train.json")
    graphs = [
        {"case": "case", "graph": {"name": i}, "structure_sha256": str(i)}
        for i in range(2)
    ]
    monkeypatch.setattr(pilot, "registered_basin_graphs", lambda source, replay: graphs)
    seen = []

    def fake_arm(graph, training, **kw):
        seen.append([r["trajectory_id"] for r in training])
        return {"candidates": [], "evidence": {}}

    monkeypatch.setattr(pilot, "build_shape_shortlist", fake_arm)
    monkeypatch.setattr(
        pilot.experiment, "original_menu", lambda g, c, rows: fake_arm(g, rows)
    )
    result = pilot.prepare(tmp_path / "out")
    assert (
        result["source"] == data["artifact_sha256"]
        and not result["nonnegative_targets"]
    )
    assert len(result["panels"]) == 4 and len(seen) == 8
    for panel in result["panels"]:
        packet = sealed_read(tmp_path / "out" / "packets" / f"{panel['panel']}.json")
        fit = [r["trajectory_id"] for r in packet["fitting_rows"]]
        assert set(fit).isdisjoint(packet["held_event_ids"]) and len(fit) == 3
        assert set(packet["menus"]) == {"original", "shape"}
    assert seen[::2] == seen[1::2]
