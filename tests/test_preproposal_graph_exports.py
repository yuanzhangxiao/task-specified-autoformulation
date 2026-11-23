"""The post-inspection handoff contains admitted graphs with their original types."""

import json
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research.preproposal_complete_graphs import generate_pool
from autoformalism.research.preproposal_topology import assess_topology, reply_schema
from scripts import preproposal_selection_20b as cli
from scripts.preproposal_graph_exports import _canonical, admitted_graphs
from tests.test_preproposal_complete_graphs import inputs


def test_export_normalizes_raw_reply_without_shadowing_a_conflicting_definition():
    case, frozen = inputs()
    frozen["allow_topology_latents"] = True
    row = generate_pool(case, frozen, max_candidates=1)["candidates"][0]
    graph, digest = row["graph"], row["structure_sha256"]
    schema = reply_schema(case, frozen)
    canonical = _canonical(case, frozen, schema, graph, digest)
    duplicate = deepcopy(graph)
    duplicate["new_latents"] = [
        {
            "name": "level",
            "kind": "differential",
            "sources": graph["sources"]["level"][::-1],
        }
    ]
    assert _canonical(case, frozen, schema, duplicate, digest) == canonical
    # The latent copy alone would still produce the recorded structural digest.
    duplicate["sources"]["level"].append("forged_source")
    with pytest.raises(ValueError, match=r"Conflicting declarations.*forged_source"):
        _canonical(case, frozen, schema, duplicate, digest)


def test_deduplicate_admitted_selection_origins_and_keep_added_latents(
    tmp_path, monkeypatch
):
    case, frozen = inputs()
    frozen["allow_topology_latents"] = True
    intake = {
        "selection": {"cases": [case.model_dump(mode="json")]},
        "frozen_cases": {case.case_id: frozen},
    }
    pool = generate_pool(case, frozen, max_candidates=2)
    first, unselected = pool["candidates"]
    child = deepcopy(first["graph"])
    child["new_latents"] = [
        {"name": "response", "kind": "algebraic", "sources": ["rain"]}
    ]
    child["sources"]["level"].append("response")
    outcome = assess_topology(case, None, child, frozen)
    assert outcome["status"] == "topology_screen_passed"
    slot = case.case_id + "--seed-100--source-example"
    sealed_write(tmp_path / "steps" / slot / "reply.json", {"value": child})
    accepted_branch = {"candidate_id": "science", "slot": slot, "topology": outcome}
    branch = {
        "seed": 100,
        "scientific": [accepted_branch, {"topology": {"status": "invalid_topology"}}],
        "uniform": [accepted_branch],
        "graph_first": {
            "status": "invalid_topology",
            "repairs": [dict(outcome, graph=child, edits=[])],
        },
    }
    summary = {
        "plan_sha256": "experiment",
        "status": "incomplete",
        "cases": [
            {
                "case_id": case.case_id,
                "complete_graphs": {
                    **pool,
                    "draws": [
                        {
                            "seed": 100,
                            "uniform_graph_ids": [first["candidate_id"]],
                            "scientific": {
                                "status": "selected",
                                "graph_ids": [first["candidate_id"]],
                            },
                        }
                    ],
                },
                "draws": [
                    branch,
                    {
                        "seed": 101,
                        "scientific": [],
                        "uniform": [],
                        "graph_first": dict(outcome, parent=child),
                    },
                ],
            }
        ],
    }
    result = admitted_graphs(tmp_path, summary, intake)
    assert result["distinct_case_graphs"] == 2
    by_digest = {r["structure_sha256"]: r for r in result["graphs"]}
    assert unselected["structure_sha256"] not in by_digest
    assert {r["arm"] for r in by_digest[first["structure_sha256"]]["origins"]} == {
        "scientific_graph_selection",
        "uniform_graph_selection",
    }
    selected = by_digest[outcome["structure_sha256"]]
    assert {r.get("origin", r["arm"]) for r in selected["origins"]} == {
        "scientific_source_selection",
        "uniform_source_selection",
        "parent",
        "repair",
    }
    assert selected["graph"]["variable_types"]["response"] == "algebraic"
    assert selected["graph"]["sources"]["response"] == ["rain"]
    assert set(selected["graph"]["initializers"]) == {"level", "upstream", "store"}
    assert selected["graph"]["initializers"]["level"] == {
        "mode": "observed_initial",
        "channel": "level",
    }
    assert selected["graph"]["initializers"]["store"]["mode"] in {"value", "map"}

    sealed_write(tmp_path / "intake.json", intake)
    monkeypatch.setattr(cli, "status", lambda _: summary)
    archive = cli.inspect(tmp_path)["archive"]
    with tarfile.open(archive) as bundle:
        assert json.load(bundle.extractfile("admitted-graphs.json")) == result
    assert "admitted-graphs.json" in {p.name for p in tmp_path.iterdir()}
    # A later origin for an already exported graph also needs a matching digest.
    mismatched = deepcopy(summary)
    mismatched["cases"][0]["draws"][1]["graph_first"]["parent"]["sources"]["store"] = [
        "x"
    ]
    with pytest.raises(ValueError, match="identity"):
        admitted_graphs(tmp_path, mismatched, intake)
    first["graph"]["sources"]["store"].append("invented")
    with pytest.raises(ValueError, match="identity"):
        admitted_graphs(tmp_path, summary, intake)
