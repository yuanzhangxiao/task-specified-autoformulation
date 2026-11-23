"""Exercise the historical-panel adapter without fitting or reading another split."""

from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import audit_ordered_insulin_repairs as replay
from tests.test_affine_direction_certificate import evidence, family
from tests.test_ordered_driver_certificate import model, observations


def test_single_source_scan_uses_only_new_permitted_histories():
    e = observations()
    g, f = model("a*y+b*x")
    f["equations"]["y"] += "+order_gain0"
    f["parameters"]["order_gain0"] = 0.2
    original = deepcopy((g, f))
    case = {
        "histories": ["y", "x", "w", "undeclared"],
        "evidence": {"context": e.context.model_dump(mode="json")},
    }
    result = replay.single_source_checks(g, f, "y", case, e, 1.0)
    assert set(result) == {"w"}
    assert result["w"]["status"] == "conflict"
    assert (g, f) == original


def test_mixed_replay_and_substituted_candidate(tmp_path, monkeypatch):
    panel = replay.panel
    monkeypatch.setattr(panel, "TARGET", "y")
    graph, formula = family("a*y+b*x", ("x", "y"))
    public = evidence().model_dump(mode="json")
    events = public.pop("trajectories")
    data = sealed_write(
        tmp_path / "intake.json",
        {
            "cases": {"demo": {"evidence": public}},
            "parents": {str(i): {"case_id": "demo"} for i in panel.PARENTS},
            "train": events,
        },
    )
    monkeypatch.setattr(panel, "INTAKE", data["artifact_sha256"])
    plan = sealed_write(
        tmp_path / "plan.json",
        {
            "intake_sha256": data["artifact_sha256"],
            "protocol": panel.PROTOCOL,
            "maximum_fit_slots": len(panel.slots()),
        },
    )
    receipts = []
    for index, (slot, seed, arm) in enumerate(panel.slots()):
        value = {
            "index": index,
            "parent_slot": slot,
            "seed": seed,
            "condition": arm,
            "plan": plan["artifact_sha256"],
            "status": "training_conflict",
        }
        if index == 0:
            value.update(status="eligible", graph_dict=graph, formula=formula)
        receipt = sealed_write(tmp_path / "candidates" / f"{index:03}.json", value)
        receipts.append(receipt["artifact_sha256"])
    freeze = sealed_write(
        tmp_path / "generation.json",
        {"plan": plan["artifact_sha256"], "candidates": receipts},
    )
    monkeypatch.setattr(replay, "GENERATION", freeze["artifact_sha256"])
    result = replay.audit(tmp_path)
    assert result == replay.audit(tmp_path)
    assert result["counts"] == {"unverified": 1, "previously_ineligible": 17}
    assert result["optimizer_calls"] == result["llm_calls"] == 0
    assert result["rows"][0]["original_status"] == "eligible"
    path = tmp_path / "candidates" / "017.json"
    path.unlink()
    sealed_write(path, {**value, "status": "invalid"})
    with pytest.raises(ValueError, match="receipt differs"):
        replay.audit(tmp_path)
