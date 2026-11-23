"""Exercise authored source changes through the live campaign and its CPU handoff."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.preproposal_cases import BASIN
from autoformalism.staged_topology import content_hash
from scripts import submit_trajectory_functions as launch
from scripts import trajectory_function_report as reporting
from scripts import trajectory_functions as trial
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_topology_repair_generator import evidence, row


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    signals = [
        row("train_000", [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        row("train_001", [0, 1, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0]),
        row("train_002", [0, 2, 4], [1, 2, 3], [0, 0, 0], [1, 1, 1]),
    ]
    binding = evidence(signals).model_dump(mode="json")
    training = binding.pop("trajectories")
    kinds = {"level": "differential", "upstream": "differential"}
    c = "synthetic_review"
    case = {
        "case_id": c,
        "target": "level",
        "histories": ["rain", "spill"],
        "required_histories": [],
        "initials": ["upstream"],
        "evidence": binding,
        "public_brief": {
            "scientific_context": "Model the two water levels.",
            "public_variables": [
                {"name": n, "data_role": role}
                for n, role in (
                    ("level", "target"),
                    ("upstream", "target"),
                    ("rain", "external_input"),
                    ("spill", "external_input"),
                )
            ],
            "requirements": [],
        },
    }
    graphs = []
    for source in case["histories"]:
        graph = {
            "variable_types": kinds,
            "initializers": {
                n: {"mode": "observed_initial", "channel": n} for n in kinds
            },
            "sources": {"level": [source], "upstream": ["spill"]},
        }
        graphs.append(
            {
                "dataset": BASIN,
                "case": c,
                "origin": "pool",
                "structure_sha256": content_hash(graph),
                "graph": graph,
            }
        )
    source = tmp_path / "train.json"
    train = sealed_write(
        source,
        {
            "train": {BASIN: training},
            "graphs": graphs,
            "cases": {c: case},
        },
    )
    val = tmp_path / "val.json"
    heldout = sealed_write(
        val,
        {
            "rows": {BASIN: [dict(training[1], trajectory_id="val_000")]},
            "training_intake_sha256": train["artifact_sha256"],
        },
    )
    monkeypatch.setattr(trial, "TRAIN", train["artifact_sha256"])
    monkeypatch.setattr(trial, "VALIDATION", heldout["artifact_sha256"])
    monkeypatch.setattr(trial, "runtime_source_hash", lambda: "synthetic")
    monkeypatch.setattr(
        trial, "BUDGET", {"max_evaluations": 4, "starts": 1, "wall_seconds": 15.0}
    )
    return tmp_path / "results", source, val


def candidate(route="request_topology_change", equation="rain + 2*spill"):
    item = {
        "route": route,
        "equations": {"level": equation, "upstream": "spill"},
        "parameters": [],
        "initializers": {},
    }
    if route == "request_topology_change":
        item["scientific_reason"] = "Both inflows reach the level balance."
    return {"candidate": item}


def test_live_review_routes_children_through_fitting_scoring_and_reporting(
    inputs, monkeypatch
):
    root, train, val = inputs
    plan = trial.prepare(root, train, trial.BASIN_REVIEW)
    assert (plan["maximum_generation_requests"], plan["maximum_fit_slots"]) == (8, 12)
    p = sealed_read(root / "intake.json")
    wrong_route = trial.assess_reply(p, 0, "proposer", candidate("fixed_graph"))
    assert (
        wrong_route["status"] == "invalid_formula"
        and "revised_graph" not in wrong_route
    )
    no_change = trial.assess_reply(p, 0, "proposer", candidate(equation="rain"))
    assert no_change["status"] == "no_topology_change"

    def provider(_, body, __):
        observation = json.loads(body["messages"][1]["content"])
        assert "val_000" not in body["messages"][1]["content"]
        if "shortlist" in observation:
            return reply({"candidate": {"route": "fixed_graph", "shortlist_id": "A"}})
        return reply(candidate())

    result = trial.run(
        root, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert result["physical_requests"] == 8 and result["candidate_statuses"] == {
        "eligible": 12
    }
    changed = sealed_read(root / "candidates/000.json")
    assert changed["patch"]["level"] == {"added": ["spill"], "removed": []}
    child = trial.fitting_graph(changed, p["graphs"][0])
    assert child["sources"]["level"] == ["rain", "spill"]
    for invalid in (dict(changed, revised_graph_sha256="x"), dict(changed, graph="y")):
        with pytest.raises(ValueError):
            trial.fitting_graph(invalid, p["graphs"][0])

    fits = [trial.fit(root, i) for i in range(12)]
    assert fits[0]["result"]["graph_sha256"] == content_hash(child)
    assert fits[2]["result"]["graph_sha256"] == content_hash(p["graphs"][0]["graph"])
    final = trial.validate(root, val)
    assert final["status"] == "complete" and final["validation_records"] == 6
    validated = sealed_read(root / "validation" / "000.json")
    assert validated["score"]["pooled_nmse"] == pytest.approx(0.0)
    monkeypatch.setattr(reporting, "_plot", lambda *args: [])
    reporting.inspect(root, final)
    summary = json.loads((root / "summary.json").read_text())
    result = next(
        r
        for r in summary["comparisons"]
        if r["graph_index"] == 0 and r["arm"] == "proposer"
    )
    assert (
        result["route"] == "request_topology_change"
        and result["admitted_graph"] == child
    )


def test_replay_disables_generation_and_review_uses_only_12_fit_slots(
    inputs, monkeypatch
):
    root, train, _ = inputs
    replay_root = root.parent / "replay"
    trial.prepare(root, train, trial.BASIN_REVIEW)
    with pytest.raises(ValueError, match="different profile"):
        trial.prepare(root, train, trial.FIXED)
    # An offline replay is prepared from archived records in its own root.
    # Verify that even a sealed replay cannot consume any generation slots.
    payload = sealed_read(root / "intake.json")
    payload.pop("artifact_sha256")
    payload["profile"] = trial.BASIN_REPLAY
    replay = sealed_write(replay_root / "intake.json", payload)
    plan = sealed_write(replay_root / "plan.json", trial._plan(replay))
    assert plan["maximum_generation_requests"] == 0
    with pytest.raises(ValueError, match="no generation allowance"):
        trial.run(
            replay_root,
            "http://localhost",
            transport=lambda *args: pytest.fail("provider call"),
        )
    with pytest.raises(ValueError):
        launch.submit(replay_root)
    monkeypatch.setattr(launch.gpu, "submit", lambda *args, **kwargs: {"job": "100"})
    queued = []

    def submit(directory, argv):
        queued.append(argv)
        return str(100 + len(queued))

    monkeypatch.setattr(launch, "_submit_job", submit)
    jobs = launch.submit(root)["jobs"]
    assert jobs == {"generate": "100", "fit": "101", "validate": "102"}
    assert "--array=0-11%8" in queued[0] and "--dependency=afterok:100" in queued[0]
    assert "--dependency=afterany:101" in queued[1]
    assert launch.submit(root)["jobs"] == jobs and len(queued) == 2
