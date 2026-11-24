"""Pointwise errors become pre-fit exclusions only under a declared contract."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_families as menus
from autoformalism.research import trajectory_function_fitting as fitting
from autoformalism.staged_topology import content_hash
from scripts import replay_trajectory_function_pointwise as saved
from scripts import trajectory_functions as trial
from tests.test_affine_direction_certificate import evidence, family
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_ordered_driver_certificate import model, observations


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    e = evidence().model_dump(mode="json")
    rows = e.pop("trajectories")
    g, f = family()
    case = {
        "case_id": "C",
        "histories": ["x", "w"],
        "evidence": e,
        "public_brief": {"scientific_context": "Explain y."},
    }
    row = {
        "dataset": "trial",
        "case": "C",
        "graph": g,
        "structure_sha256": content_hash(g),
    }
    form2 = deepcopy(f)
    form2["equations"]["y"] = "a + b*x**2"
    src = {"train": {"trial": rows}, "graphs": [row], "cases": {"C": case}}
    path = tmp_path / "training.json"
    intake = sealed_write(path, src)
    monkeypatch.setattr(trial, "TRAIN", intake["artifact_sha256"])
    monkeypatch.setattr(trial, "runtime_source_hash", lambda: "unit-test")
    monkeypatch.setattr(
        menus,
        "build_shortlist",
        lambda *args: {
            "candidates": [f, form2],
            "evidence": {},
        },
    )
    return path, intake, f


def populate(root, *, show_policy):
    """Each arm proposes affine then nonlinear; the live receipt path is exercised."""
    count = 0

    def provider(url, body, timeout):
        nonlocal count
        packet = json.loads(body["messages"][1]["content"])
        if show_policy:
            assert packet["pointwise_accuracy_required"][
                "absolute_tolerances_in_target_units"
            ] == {"y": 1}
        else:
            assert "pointwise_accuracy_required" not in packet
        if "shortlist" in packet:
            choice = {"shortlist_id": ["A", "B"][count % 2]}
        else:
            choice = {
                "equations": {"y": ["a+b*x", "a+b*x**2"][count % 2]},
                "parameters": [
                    {"name": "a", "initial": 3},
                    {"name": "b", "initial": -2},
                ],
                "initializers": {},
            }
        count += 1
        return reply({"candidate": choice})

    out = trial.run(
        root, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert count == out["physical_requests"] == 4
    return out


def test_policy_is_frozen_by_identity_and_applies_to_every_arm(
    tmp_path, setup_case, monkeypatch
):
    path, _, _ = setup_case
    root = tmp_path / "gate"
    policy = {"trial": {"y": 1}}
    plan = trial.prepare(root, path, pointwise_tolerances=policy)
    assert trial.prepare(root, path, pointwise_tolerances=policy) == plan
    with pytest.raises(ValueError, match="pointwise contract"):
        trial.prepare(root, path)
    with pytest.raises(ValueError, match="pointwise contract"):
        trial.prepare(root, path, pointwise_tolerances={"trial": {"y": 2}})
    before = trial.prepare(tmp_path / "default", path)
    assert plan["intake_sha256"] != before["intake_sha256"]
    assert "pointwise_tolerances" not in sealed_read(tmp_path / "default/intake.json")
    _, data = trial.load(root)
    _, safe = trial._admitted_smoke_example(data)
    assert "**2" in safe["equations"]["y"]
    data["menus"][0]["candidates"][1] = data["menus"][0]["candidates"][0]
    assert trial._admitted_smoke_example(data) is None
    out = populate(root, show_policy=True)
    assert out["candidate_statuses"] == {"eligible": 3, "pointwise_conflict": 3}

    def unexpected(*args, **kwargs):
        pytest.fail("No further generation or optimization is allowed here")

    monkeypatch.setattr(fitting, "fit_formula", unexpected)
    assert trial.run(root, "", transport=unexpected) == out
    for i in (0, 2, 4):
        c = trial._read_candidate(root, plan, i)
        text = c["repair_feedback"][0]
        assert text["reference"]["time"] == [0, 12] and text["rise"]["time"] == [0, 4]
        assert "sources_available_to_add" not in text
        assert "request_topology_change" not in json.dumps(text)
        fit = trial.fit(root, i)
        assert fit["result"] == {"status": "ineligible", "reason": "pointwise_conflict"}
        assert trial.fit(root, i) == fit
    assert not list(root.glob("fits/*/reservation.json"))
    for i in (1, 3, 5):
        c = trial._read_candidate(root, plan, i)
        assert c["pointwise_checks"]["y"]["status"] == "unverified"
        assert "repair_feedback" not in c


def test_revised_source_graph_is_checked_and_only_permitted_edits_offered(setup_case):
    _, intake, f = setup_case
    data = {
        **intake,
        "profile": trial.SOURCE_REVIEW,
        "pointwise_tolerances": {"trial": {"y": 1}},
    }
    row = deepcopy(data["graphs"][0])
    revised = deepcopy(row["graph"])
    row["graph"]["sources"]["y"] = ["w"]
    row["structure_sha256"] = content_hash(row["graph"])
    c = {
        "status": "eligible",
        "graph": row["structure_sha256"],
        "route": "request_topology_change",
        "formula": f,
        "revised_graph": revised,
        "revised_graph_sha256": content_hash(revised),
    }
    out = trial.pointwise_admission(data, row, c)
    assert out["status"] == "pointwise_conflict"
    text = out["repair_feedback"][0]
    assert text["current_sources"] == ["x"]
    assert "w" in text["sources_available_to_add"]
    assert "request_topology_change" in text["repair_options"][1]
    for status in ("invalid_formula", "provider_failed"):
        previous = {"status": status}
        assert trial.pointwise_admission(data, row, previous) is previous


def test_order_proof_blocks_after_affine_abstention_on_effective_graph():
    evidence = observations().model_dump(mode="json")
    events = evidence.pop("trajectories")
    graph, form = model("a*y+b*x")
    parent, _ = model("a*y + b*w")
    row = {
        "case": "c",
        "dataset": "d",
        "graph": parent,
        "structure_sha256": content_hash(parent),
    }
    data = {
        "profile": trial.SOURCE_REVIEW,
        "cases": {"c": {"evidence": evidence, "histories": ["x", "w"]}},
        "train": {"d": events},
        "pointwise_tolerances": {"d": {"y": 1}},
    }
    candidate = {
        "status": "eligible",
        "route": "request_topology_change",
        "revised_graph": graph,
        "revised_graph_sha256": content_hash(graph),
        "graph": row["structure_sha256"],
        "formula": form,
    }
    out = trial.pointwise_admission(data, row, candidate)
    assert out["status"] == "pointwise_conflict"
    assert out["pointwise_checks"]["y"]["status"] == "unverified"
    assert out["ordered_checks"]["y"]["exact_lower_bound"] == "2"
    (item,) = out["repair_feedback"]
    assert item["check"] == "ordered_driver" and item["driver"] == "x"
    assert item["rhs"] == "a*y+b*x"
    assert item["witnesses"] == out["ordered_checks"]["y"]["witnesses"]
    assert item["sources_available_to_add"] == ["t", "w"]

    higher = trial.pointwise_admission(
        data | {"pointwise_tolerances": {"d": {"y": 2}}},
        row,
        candidate,
    )
    assert higher["status"] == "eligible"
    assert higher["ordered_checks"]["y"]["status"] == "unverified"
    assert "repair_feedback" not in higher
    assert (
        trial.pointwise_admission(data | {"pointwise_tolerances": {}}, row, candidate)
        is candidate
    )


@pytest.mark.parametrize(
    "policy",
    [
        {},
        [],
        {"typo": {"y": 1}},
        {"trial": {}},
        {"trial": {"x": 1}},
        {"trial": {"y": -1}},
        {"trial": {"y": True}},
        {"trial": {"y": "1"}},
        {"trial": {"y": float("nan")}},
        {"trial": {"y": float("inf")}},
        {"trial": {"y": 2**60 + 1}},
    ],
)
def test_invalid_or_inapplicable_declared_policies_raise(setup_case, policy):
    _, intake, _ = setup_case
    with pytest.raises(ValueError):
        trial.checked_tolerances(intake, intake["graphs"], policy)
    non_ode = deepcopy(intake["graphs"])
    non_ode[0]["graph"]["variable_types"]["y"] = "algebraic"
    with pytest.raises(ValueError, match="differential target"):
        trial.checked_tolerances(intake, non_ode, {"trial": {"y": 1}})


@pytest.mark.parametrize("bad", ['{"trial":{"y":0,"y":1}}', "null", "[]"])
def test_json_contract_cannot_silently_disable_or_overwrite_a_policy(bad, tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(bad)
    with pytest.raises(ValueError):
        trial.read_tolerances(path)


def test_replay_seals_source_baseline_rechecks_and_never_spends(
    tmp_path, setup_case, monkeypatch
):
    path, _, _ = setup_case
    archive, new = tmp_path / "archive", tmp_path / "replay"
    old = trial.prepare(archive, path)
    plain = populate(archive, show_policy=False)
    assert plain["candidate_statuses"] == {"eligible": 6}
    generation = (archive / "generation.json").read_bytes()
    monkeypatch.setattr(
        saved, "ARCHIVED_GENERATION", json.loads(generation)["artifact_sha256"]
    )
    monkeypatch.setattr(saved, "ARCHIVED_PLAN", old["artifact_sha256"])
    monkeypatch.setattr(
        fitting, "fit_formula", lambda *a, **k: pytest.fail("optimizer")
    )
    checks = {"trial": {"y": 1}}
    report = saved.replay(archive, new, checks)
    assert report["before"] == {"eligible": 6}
    assert report["after"] == {"eligible": 3, "pointwise_conflict": 3}
    assert report["newly_blocked_fit_indices"] == [0, 2, 4]
    assert saved.replay(archive, new, checks) == report
    assert (archive / "generation.json").read_bytes() == generation
    assert trial.status(archive) == plain
    assert trial.load(new)[0]["maximum_generation_requests"] == 0
    with pytest.raises(ValueError, match="no generation allowance"):
        trial.run(new, "")
    with pytest.raises(ValueError, match="no optimizer allowance"):
        trial.fit(new, 1)
    with pytest.raises(ValueError, match="no optimizer allowance"):
        trial.preflight(new)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        saved.replay(archive, new, {"trial": {"y": 2}})
    (call,) = (archive / "steps/slot-000/calls").glob("*.json")
    call.write_text(call.read_text() + "\n")
    with pytest.raises(ValueError, match="generation receipt"):
        saved.replay(archive, new, checks)
