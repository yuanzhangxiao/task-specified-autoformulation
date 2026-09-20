"""Paired public-only construction, fallback, seals and no extra fitting budget."""

import json

import pytest

from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search.process_review import POLICY
from scripts import smoke_detention_process_pilot as smoke


@pytest.fixture
def frozen(tmp_path):
    source, root = tmp_path / "source", tmp_path / "pilot"
    return source, root, smoke.fixture(source, root)


def test_public_only_paired_matrix_and_resume(frozen):
    source, root, plan = frozen
    assert len(plan["tasks"]) == 16
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(plan)
    config = pilot.REPO / "configs/detention_process_pilot_v1.json"
    assert pilot.freeze(source, root, config) == plan
    for case in ("coupled", "independent"):
        for seed in (0, 1):
            for arm in ("full", "brief_only"):
                pair = [
                    t
                    for t in plan["tasks"]
                    if (t["case"], t["seed"], t["arm"]) == (case, seed, arm)
                ]
                assert {t["review"] for t in pair} == {True, False}
    (source / "public/coupled/noise1/train.json").write_text("{}")
    with pytest.raises(ValueError, match="asset changed"):
        pilot.freeze(source, root, config)


@pytest.mark.parametrize("arm", ["full", "brief_only"])
@pytest.mark.parametrize("mode", ["add", "empty", "bad_topology", "bad_function"])
def test_construct_and_fallback_preserves_original_variables(frozen, arm, mode):
    _, root, plan = frozen
    task = next(
        t
        for t in plan["tasks"]
        if t["case"] == "coupled" and t["review"] and t["arm"] == arm
    )
    calls = []
    client = pilot.make_client(
        root, plan, task, "offline", transport=smoke.transport_for(calls, mode)
    )
    result = pilot.construct(root, plan, task, client)
    assert result["status"] == "constructed", result
    assert result["fallback_used"] == mode.startswith("bad")
    base = result["bundle"]["initialization"]["base_candidate"]
    assert {v["name"] for v in base["states"]} == {"h_up", "h_down"}
    assert any(p["name"] == "q" for p in base["processes"]) == (mode == "add")
    optional = [c for c in calls if c["payload"].get("protocol") == POLICY]
    assert len(optional) == 1
    assert ("training_observations" in optional[0]["payload"]["public_brief"]) == (
        arm == "full"
    )
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(calls)
    before = len(calls)
    assert pilot.construct(root, plan, task, client) == result
    assert len(calls) == before
    assert result["usage"]["physical_calls"] == before


def test_control_has_same_guidance_but_no_optional_call(frozen):
    _, root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and not t["review"])
    calls = []
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(
            root, plan, task, "offline", transport=smoke.transport_for(calls)
        ),
    )
    assert result["status"] == "constructed", result
    assert not any(c["payload"].get("protocol") == POLICY for c in calls)
    assert (
        "Shared-process modeling guidance" in calls[0]["body"]["messages"][0]["content"]
    )


def test_partial_stage_resume_uses_identical_downstream_contract(frozen, monkeypatch):
    _, root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    calls = []
    original = pilot.run_staged_functions
    monkeypatch.setattr(
        pilot,
        "run_staged_functions",
        lambda *a, **kw: (_ for _ in ()).throw(pilot.DeferredCall("drain")),
    )
    with pytest.raises(pilot.DeferredCall):
        pilot.construct(
            root,
            plan,
            task,
            pilot.make_client(
                root, plan, task, "offline", transport=smoke.transport_for(calls)
            ),
        )
    count = len(calls)
    monkeypatch.setattr(pilot, "run_staged_functions", original)
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(
            root, plan, task, "offline", transport=smoke.transport_for(calls)
        ),
    )
    assert result["status"] == "constructed", result
    assert all("selected_lhs" not in c["payload"] for c in calls[count:])
    assert sum(c["payload"].get("protocol") == POLICY for c in calls) == 1


def test_missing_rows_and_source_drift(frozen, monkeypatch):
    _, root, _ = frozen
    assert pilot.report(root)["status_counts"] == {"missing": 16}
    monkeypatch.setattr(pilot, "launcher_hash", lambda: "changed")
    with pytest.raises(ValueError, match="source or launcher"):
        pilot.verify(root)


def test_failed_construction_does_not_call_fitter(frozen, monkeypatch):
    _, root, plan = frozen
    task = plan["tasks"][0]
    folder = root / "results" / task["task_id"]
    sealed_write(
        folder / "proposal.json", {"task": task, "status": "construction_failed"}
    )
    monkeypatch.setattr(
        pilot.public, "execute_fit", lambda *a: pytest.fail("must not fit")
    )
    assert pilot.fit_task(root, task["index"])["status"] == "construction_failed"
    assert pilot.fit_task(root, task["index"]) == sealed_read(folder / "result.json")


def test_disconnected_check_includes_initial_maps(frozen):
    _, root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "coupled" and t["review"])
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(
            root, plan, task, "offline", transport=smoke.transport_for([])
        ),
    )
    payload = result["bundle"]["candidate"]
    # Remove the upstream input from every equation: the sole forbidden path
    # remains initial_up -> h_up(0) -> q -> h_down.
    for equation in payload["state_equations"]:
        equation["rhs"] = equation["rhs"].replace("inflow_up", "0")
    model = pilot.CandidateModel.model_validate(payload)
    audit = pilot.structure(model, "independent")
    assert audit["forbidden_upstream_dependencies"] == ["initial_up"]
    assert not audit["eligible"]
    assert pilot.structure(model, "coupled")["eligible"]


@pytest.mark.parametrize("review", [False, True])
def test_disconnected_construction_accepts_no_process(frozen, review):
    _, root, plan = frozen
    task = next(
        t for t in plan["tasks"] if t["case"] == "independent" and t["review"] == review
    )
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(
            root,
            plan,
            task,
            "offline",
            transport=smoke.transport_for([], "independent"),
        ),
    )
    assert result["status"] == "constructed", result
    assert not result["structural"]["forbidden_upstream_dependencies"]
    assert not result["bundle"]["candidate"]["processes"]
    if review:
        assert result["attempts"][0]["process_review"]["status"] == "empty"


def test_disconnected_false_transfer_rejected_including_fallback(frozen):
    _, root, plan = frozen
    task = next(t for t in plan["tasks"] if t["case"] == "independent" and t["review"])
    result = pilot.construct(
        root,
        plan,
        task,
        pilot.make_client(
            root, plan, task, "offline", transport=smoke.transport_for([])
        ),
    )
    assert result["status"] == "requirement_failed"
    assert result["fallback_used"]
    assert all(
        a["structural"]["forbidden_upstream_dependencies"] for a in result["attempts"]
    )
