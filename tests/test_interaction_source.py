"""Frozen interaction sources retain full declarations and authenticated episodes."""

from __future__ import annotations

import copy
import io
import json
import tarfile
from pathlib import Path

import pytest

from autoformalism.rebuttal import staged_topology_campaign
from autoformalism.research.interaction_source import load_sources
from autoformalism.staged_topology import content_hash
from scripts import construction_initialization_graph as graph_cli
from scripts import construction_initialization_repair as repair_cli
from tests.test_construction_initialization_graph_cli import (
    historical_plan,
    reply,
    tokenizer,
    write_archive,
)
from tests.test_construction_initialization_repair_cli import (
    old_choice,
    trajectory_choice,
)


def reseal(value, field="artifact_sha256"):
    """Update a fixture seal to test validation beyond the immediate checksum."""
    value[field] = content_hash({k: v for k, v in value.items() if k != field})
    return value


@pytest.fixture
def completed(tmp_path, monkeypatch):
    """Create a public two-model history with eight accepted and two stopped seeds."""
    # This loader verifies historical content, independently of the running code.
    # Freeze the fixture's historical runtime while other workspace work proceeds.
    for module in (graph_cli, repair_cli, staged_topology_campaign):
        monkeypatch.setattr(module, "runtime_source_hash", lambda: "c" * 64)
    plan = historical_plan()
    for case in plan["cases"].values():
        case["cell"]["brief"].update(
            public_variables=[
                {"name": "a", "data_role": "target"},
                {"name": "u", "data_role": "external_input"},
                {"name": "t", "data_role": "time"},
            ],
            requirements=[],
        )
        case["source_result"]["draft"]["mechanism_realizations"] = [
            {"description": "Retained original scientific declaration"}
        ]
        case["start"]["draft"] = copy.deepcopy(case["source_result"]["draft"])
        reseal(case["source_result"])
    reseal(plan)
    original, graph = tmp_path / "original.tar.gz", tmp_path / "graph"
    write_archive(original, plan)
    graph_cli.prepare(original, graph)
    graph_cli.run(
        graph,
        "http://127.0.0.1:8000",
        transport=lambda *args: reply(old_choice()),
        token_transport=tokenizer,
    )
    root = tmp_path / "scope"
    repair_cli.prepare(
        Path(graph_cli.inspect(graph)["archive"]), root, feedback_policy="scope-aware-2"
    )
    count = 0

    def transport(*args):
        nonlocal count
        count += 1
        return reply(
            {
                "decision": "needs_topology_revision",
                "records": [],
                "explanation": "Wider topology work would be needed.",
            }
            if count in {1, 5}
            else trajectory_choice()
        )

    result = repair_cli.run(
        root,
        "http://127.0.0.1:8000",
        transport=transport,
        token_transport=tokenizer,
    )
    assert result["accepted_episodes"] == 8
    return root


def pack(root, destination, extra=None):
    """Write a bounded synthetic inspection archive without dereferencing links."""
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root), recursive=False)
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
    return destination


def test_load_all_accepted_episodes_without_deduplicating_graphs(completed):
    loaded = load_sources(completed)
    assert len(loaded["roster"]) == 10
    assert len(loaded["cases"]) == 8
    assert sum(row["accepted"] for row in loaded["roster"]) == 8
    for value in loaded["cases"].values():
        source = value["function_source"]
        assert source["complete_topology"]
        assert source["initialization_graph_handoff"] == value["result"]["handoff"]
        assert source["initialization_graph_contract"] == value["case"]["contract"]
        assert source["scientific_declarations"] == value["full_draft"]
        assert value["full_draft"]["external_exchanges"] == ["retained"]
        assert value["full_draft"]["mechanism_realizations"]
        assert "initializer_handoff" not in source
        assert "initial_information_contract" not in source
        assert source["initializer_acceptance"] == value["result"]["acceptance"]
    assert load_sources(completed) == loaded


def test_archive_and_directory_export_identical_cases(completed, tmp_path):
    archive = pack(completed, tmp_path / "scope.tar.gz")
    archived, directory = load_sources(archive), load_sources(completed)
    for key in ("source_plan", "source_summary", "roster", "cases"):
        assert archived[key] == directory[key]


@pytest.mark.parametrize("mutation", ["reply", "contract", "summary", "extra_call"])
def test_reject_tampered_source_even_with_resealed_result(completed, mutation):
    plan = json.loads((completed / "plan.json").read_text())
    entry = plan["cases"][1]
    path = completed / "results" / entry["case_id"] / "result.json"
    result = json.loads(path.read_text())
    if mutation == "reply":
        result["handoff"]["records"][0]["justification"] = "Changed after generation."
        path.write_text(json.dumps(reseal(result, "result_sha256")))
    elif mutation == "contract":
        result["contract"]["provenance"] = "Changed contract"
        path.write_text(json.dumps(reseal(result, "result_sha256")))
    elif mutation == "summary":
        path = completed / "summary.json"
        summary = json.loads(path.read_text())
        summary["cost"]["physical_requests"] -= 1
        path.write_text(json.dumps(summary))
    else:
        calls = path.parent / "calls"
        (calls / "unaccounted.json").write_text("{}")
    with pytest.raises(ValueError):
        load_sources(completed)


@pytest.mark.parametrize(
    "kind", ["parent", "absolute", "symlink", "hardlink", "duplicate"]
)
def test_reject_unsafe_archive_members(completed, tmp_path, kind):
    member = tarfile.TarInfo(
        "../escape"
        if kind == "parent"
        else "/escape"
        if kind == "absolute"
        else "plan.json"
        if kind == "duplicate"
        else "link"
    )
    if kind in {"symlink", "hardlink"}:
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "plan.json"
    else:
        member.size = 1
    archive = pack(completed, tmp_path / "unsafe.tar.gz", member)
    with pytest.raises(ValueError, match=r"unsafe|duplicate"):
        load_sources(archive)


def test_reject_directory_source_symlink(completed, tmp_path):
    link = tmp_path / "linked-source"
    link.symlink_to(completed, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_sources(link)
