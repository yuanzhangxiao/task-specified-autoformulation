"""Screen the raw reply and algorithmic repair through the same candidate gate."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import screen_ordered_repair_options as runner
from tests.test_ordered_driver_certificate import model, observations


@pytest.mark.parametrize(
    "other_history, expected", [(1, "eligible"), (0, "pointwise_conflict")]
)
def test_expose_only_jointly_screened_options(
    tmp_path, monkeypatch, other_history, expected
):
    evidence = observations(
        [
            ("train_a", [0, 1, 2], [0, 0, 0], [0] * 3, [0] * 3),
            ("train_b", [0, 1, 2], [0, 4, -8], [1] * 3, [other_history] * 3),
        ],
        uniqueness_assumed=True,
    ).model_dump(mode="json")
    trajectories = evidence.pop("trajectories")
    pg, pf = model("a*x")
    cg, cf = model("a*y+b*x")
    graph_case = {
        "case_id": "example",
        "target": "y",
        "initials": [],
        "histories": ["x", "w"],
        "required_histories": [],
        "evidence": evidence,
        "public_brief": {
            "scientific_context": "A driven state.",
            "requirements": [],
            "public_variables": [
                {
                    "name": name,
                    "data_role": "target" if name == "y" else "external_input",
                }
                for name in ("x", "w", "y")
            ],
        },
    }
    intake = {
        "artifact_sha256": "fixture",
        "train": trajectories,
        "cases": {"example": graph_case},
        "parents": {"2": {"case_id": "example", "graph": pg, "formula": pf}},
    }
    raw = {"rhs": "a*y+b*x", "new_parameters": [{"name": "b", "initial": 0.0}]}
    reply = sealed_write(tmp_path / "steps/slot-000/reply.json", {"value": raw})
    source = {
        "index": 0,
        "parent_slot": 2,
        "artifact_sha256": "fixture-candidate",
        "status": "eligible",
        "graph_dict": cg,
        "formula": cf,
        "reply_sha256": reply["artifact_sha256"],
    }
    monkeypatch.setattr(runner.archived, "frozen_panel", lambda _: (intake, [source]))
    monkeypatch.setattr(runner.panel, "TARGET", "y")
    result = runner.screen(tmp_path)
    assert result == runner.screen(tmp_path)
    assert result["reply_counts"] == {"pointwise_conflict": 1}
    assert result["extension_counts"] == {expected: 1}
    assert [m["added_source"] for m in result["shortlist"]] == (
        ["w"] if expected == "eligible" else []
    )
    assert cf["equations"]["y"] == "a*y+b*x"
    source["reply_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Reply differs"):
        runner.screen(tmp_path)
    monkeypatch.setattr(runner.panel, "INTAKE", "fixture")
    parents = runner.screen_parents(intake)
    assert parents == runner.screen_parents(intake)
    assert len(parents["extensions"]) == 1
    decision = parents["extensions"][0]
    assert decision["source_added"] == "w" and decision["parent_slot"] == 2
    assert "from_reply" not in decision
    assert len(parents["shortlist"]) == int(decision["status"] == "eligible")
    assert parents["llm_calls"] == parents["optimizer_calls"] == 0
    with pytest.raises(ValueError, match="registered training parents"):
        runner.screen_parents({**intake, "artifact_sha256": "wrong"})
