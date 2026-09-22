"""General shared-process construction/revision without domain-specific branches."""

import copy
import json
from pathlib import Path

import pytest

from autoformalism.expressions import (
    ModelValidationError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.schemas import CandidateModel
from autoformalism.search import shared_model_revision as revision
from autoformalism.staged_topology import content_hash
from scripts import smoke_shared_process_integration as smoke

ROOT = Path(__file__).resolve().parents[1]


def construct(root, *, arm="full", **kwargs):
    plan = smoke.fixture(root)
    task = next(t for t in plan["tasks"] if t["arm"] == arm)
    calls = []
    client = pipeline._client(
        root,
        plan,
        task,
        0,
        "http://offline",
        lambda: True,
        smoke.transport_for(calls, **kwargs),
    )
    result = pipeline.propose_one(root, plan, task, 0, client)
    return plan, task, client, result, calls


def test_matrix_and_frozen_numerical_budget():
    cfg = io.DeadlineConfig.model_validate_json(
        (ROOT / "configs/shared_process_integration_v1.json").read_text()
    )
    assert len(io.tasks(cfg)) == 6 and cfg.rounds == 2
    assert cfg.seeds == (0,) and len(cfg.public_cells) == 3
    assert {t["arm"] for t in io.tasks(cfg)} == {"full", "brief_only"}
    assert cfg.fit_profile == "collocation-single-target-v2"
    with pytest.raises(ValueError, match="two rounds"):
        io.DeadlineConfig.model_validate({**cfg.model_dump(), "rounds": 3})


@pytest.mark.parametrize("arm", ["full", "brief_only"])
@pytest.mark.parametrize("mode", ["shared", "empty", "invalid"])
def test_general_construction_and_cache(tmp_path, arm, mode):
    plan, task, client, proposal, calls = construct(
        tmp_path, arm=arm, empty=mode == "empty", invalid=mode == "invalid"
    )
    assert proposal["status"] == "constructed", proposal
    assert not proposal["fallback_used"]
    review = proposal["attempts"][0]["process_review"]
    assert (
        review["status"]
        == {"shared": "accepted", "empty": "empty", "invalid": "skipped_invalid"}[mode]
    )
    count = len(calls)
    assert pipeline.propose_one(tmp_path, plan, task, 0, client) == proposal
    assert len(calls) == count
    assert sum("eligible_equation_targets" in p for p in calls) == 1
    for p in calls:
        assert "validation" not in p and "test" not in p
        if "public_brief" in p:
            assert ("training_observations" in p["public_brief"]) == (arm == "full")
    if mode == "shared":
        gains = proposal["bundle"]["gain_compilation"]["processes"]
        assert len(gains) == 1
        assert len({u["gain"] for u in gains[0]["uses"]}) == 1
        assert len(proposal["bundle"]["candidate"]["processes"]) == 1
        assert {
            r["process"]
            for r in revision.relationships(proposal["bundle"]["candidate"])
        } == {"q"}


def test_law_edit_propagates_and_snapshot_is_complete(tmp_path):
    _, _, _, proposal, _ = construct(tmp_path)
    assert proposal["status"] == "constructed", proposal
    bundle = proposal["bundle"]

    # A packet needs the same model identity; evidence content comes from a toy fit
    # fixture.
    from tests.test_review_deadline_v2 import example

    packet = {**example()[1], "candidate_sha256": content_hash(bundle["candidate"])}
    before = copy.deepcopy(bundle)
    raw = {
        "hypothesis": "Change the shared law once.",
        "equations": [{"component": "q", "expression": "m"}],
    }
    result = revision.apply_edits(bundle, packet, raw)
    assert result["outcome"] == "committed"
    child = result["bundle"]
    assert bundle == before
    assert (
        child["candidate"]["state_equations"] == bundle["candidate"]["state_equations"]
    )
    q = next(
        r
        for r in result["provenance"]["shared_law_revision"]["after"]
        if r["process"] == "q"
    )
    assert q["direct_consumers"] == ["m", "v01"]
    assert q["law"] == "m"
    model = compile_candidate(
        CandidateModel.model_validate(child["candidate"]),
        ValidationContext.model_validate(child["initialization"]["context"]),
    )
    assert model is not None
    updated_packet = {**packet, "candidate_sha256": content_hash(child["candidate"])}
    values = {p["name"]: 1.0 for p in child["candidate"]["parameters"]}
    from autoformalism.fitting.public_fitting import content_sha256

    updated_packet["parameter_sha256"] = content_sha256(values)
    shown = revision.payload(child, updated_packet, values)
    assert (
        next(e for e in shown["model"]["equations"] if e["component"] == "q")[
            "expression"
        ]
        == "m"
    )
    assert "validation" not in shown
    with pytest.raises(ModelValidationError, match="ALGEBRAIC_CYCLE"):
        revision.apply_edits(
            bundle,
            packet,
            {**raw, "equations": [{"component": "q", "expression": "q+m"}]},
        )
    with pytest.raises((ValueError, ModelValidationError)):
        revision.apply_edits(
            bundle,
            packet,
            {**raw, "equations": [{"component": "q", "expression": "hidden_truth"}]},
        )


def test_missing_results_are_not_passes(tmp_path):
    smoke.fixture(tmp_path)
    reporting.report(tmp_path)
    summary = json.loads((tmp_path / "integration_summary.json").read_text())
    assert summary["status_counts"] == {"missing": 4}
    assert all(r["retained_shared_relationships"] is None for r in summary["rows"])
    assert not summary["benchmark_specific_checks"]


def test_known_and_unknown_conversion_policies(tmp_path):
    _, _, _, proposal, _ = construct(tmp_path)
    assert proposal["status"] == "constructed", proposal
    # The general adapter must not infer unit conversions or reject missing ones.
    from autoformalism.rebuttal.prefit_replay import sealed_read
    from autoformalism.search.process_gains import compile_bundle

    directory = next((tmp_path / "results").iterdir()) / "round_00/process"
    topology = sealed_read(directory / "topology_stage.json")["result"]
    functions = sealed_read(directory / "function_stage.json")["result"]
    plan = io.verify(tmp_path)
    task = plan["tasks"][0]
    cell = plan["cells"][task["cell"]]
    from autoformalism.schemas.staged_topology import PublicScientificBrief

    bundle = pipeline._bundle(
        cell,
        task,
        PublicScientificBrief.model_validate(cell["brief"]),
        topology,
        functions,
    )
    contract = copy.deepcopy(functions["shared_process_contract"])
    contract["bindings"][0]["signed_declaration"]["uses"][0]["conversion"] = None
    value = compile_bundle(
        bundle, contract, "declared_or_independent", cell["training"]
    )
    uses = value["gain_compilation"]["processes"][0]["uses"]
    assert len({u["gain"] for u in uses}) == 2
    assert {u["assembly_policy"] for u in uses} == {"independent_gains"}
    assert not value["gain_compilation"]["conservation_certified"]


def test_optional_route_failure_reuses_inventory_and_budget(tmp_path, monkeypatch):
    from autoformalism.rebuttal.prefit_replay import sealed_read
    from autoformalism.search import shared_construction as construction

    original = construction.run_staged_functions
    roots = []

    def fail_optional(*args, **kwargs):
        roots.append(str(args[4]))
        if "/process/" in str(args[4]):
            return {"complete_model": False, "error": "prescribed failed process law"}
        return original(*args, **kwargs)

    monkeypatch.setattr(construction, "run_staged_functions", fail_optional)
    plan, task, client, proposal, calls = construct(tmp_path)
    assert proposal["status"] == "constructed", proposal
    assert proposal["fallback_used"]
    assert len(roots) == 2
    directory = io.round_path(tmp_path, task, 0)
    variables = sealed_read(directory / "variables.json")["result"]
    fallback = sealed_read(directory / "ordinary_fallback/topology_stage.json")[
        "result"
    ]
    assert fallback["inventory"] == variables["inventory"]
    assert fallback["memory_candidates"] == variables["memory_candidates"]
    count = len(calls)
    assert pipeline.propose_one(tmp_path, plan, task, 0, client) == proposal
    assert len(calls) == count


def test_allocation_interruption_resumes_same_requests(tmp_path):
    from autoformalism.llm.staged_topology import DeferredCall

    plan = smoke.fixture(tmp_path)
    task, calls = plan["tasks"][0], []

    def client(allowed):
        return pipeline._client(
            tmp_path,
            plan,
            task,
            0,
            "http://offline",
            allowed,
            smoke.transport_for(calls),
        )

    with pytest.raises(DeferredCall):
        pipeline.propose_one(tmp_path, plan, task, 0, client(lambda: len(calls) < 1))
    assert len(calls) == 1
    first = calls[0]
    result = pipeline.propose_one(tmp_path, plan, task, 0, client(lambda: True))
    assert result["status"] == "constructed", result
    assert calls.count(first) == 1


def test_generic_revision_keeps_latent_boundary_contract(tmp_path):
    from tests.test_review_deadline_v2 import example

    _, _, _, proposal, _ = construct(tmp_path)
    bundle = proposal["bundle"]
    packet = {**example()[1], "candidate_sha256": content_hash(bundle["candidate"])}
    with pytest.raises(ValueError, match="initializer"):
        revision.apply_edits(
            bundle,
            packet,
            {
                "hypothesis": "A new causal memory is required.",
                "equations": [
                    {
                        "component": "memory2",
                        "kind": "dynamic",
                        "expression": "m-memory2",
                    },
                    {"component": "q", "expression": "memory2"},
                ],
            },
        )


def test_generated_auxiliary_consumers_use_compiler_aliases(tmp_path):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    plan = smoke.fixture(tmp_path)
    cell = next(iter(plan["cells"].values()))
    cell["context"]["auxiliaries"] = ["m"]
    cell["brief"]["public_variables"].append({"name": "m", "data_role": "auxiliary"})
    plan.pop("artifact_sha256")
    (tmp_path / "plan.json").unlink()
    plan = sealed_write(tmp_path / "plan.json", plan)
    task = next(t for t in plan["tasks"] if t["arm"] == "brief_only")
    client = pipeline._client(
        tmp_path, plan, task, 0, "http://offline", lambda: True, smoke.transport_for([])
    )
    proposal = pipeline.propose_one(tmp_path, plan, task, 0, client)
    assert proposal["status"] == "constructed", proposal
    bundle = proposal["bundle"]
    uses = bundle["gain_compilation"]["processes"][0]["uses"]
    assert any(u["target"].startswith("af_internal_aux_") for u in uses)
    assert "m" not in {s["name"] for s in bundle["candidate"]["states"]}


def test_integration_submission_uses_existing_safe_barriers(tmp_path, monkeypatch):
    from scripts import submit_shared_process_pilot as submitter
    from tests import test_shared_process_pilot_submission as scheduler

    monkeypatch.setattr(scheduler.smoke, "fixture", smoke.fixture)
    root, plan = scheduler.setup(tmp_path, monkeypatch)
    calls = []

    def submit(*args):
        calls.append(args)
        return str(100 + len(calls))

    monkeypatch.setattr(submitter, "submit_job", submit)
    manifest = submitter.submit(root, 0)
    assert manifest["protocol"] == io.INTEGRATION_PROTOCOL
    assert manifest["tasks"] == len(plan["tasks"])
    assert len(calls) == 4
    assert submitter.submit(root, 0) == manifest
    assert len(calls) == 4
    with pytest.raises(ValueError, match="incomplete"):
        submitter.submit(root, 1)


def test_revision_prompt_contains_training_status_and_full_model_only(monkeypatch):
    from tests.test_review_deadline_v2 import example

    bundle, packet, values, _, _ = example()
    selected = {
        "bundle": bundle,
        "packet": packet,
        "certificate": {"eligible_for_development_selection": True},
        "fit": {
            "parameters": values,
            "status": "complete",
            "budget_exhausted": True,
            "validation": {"secret_sentinel": 987654},
        },
    }
    seen = []
    raw = {"hypothesis": "Retain this model for now.", "equations": []}

    class Client:
        def call(self, **kw):
            seen.append(kw)
            return {"request_hash": "x"}

    monkeypatch.setattr(pipeline, "visible_response", lambda record: raw)
    result = pipeline._content_revision(
        {"protocol": io.INTEGRATION_PROTOCOL, "cells": {"demo": {}}},
        {"cell": "demo", "arm": "full"},
        {"selected": selected},
        Client(),
    )
    assert result["status"] == "no_change", result
    shown = json.loads(seen[0]["user"])
    assert shown["fitting_status"]["budget_exhausted"] is True
    assert shown["model"]["equations"]
    assert "secret_sentinel" not in json.dumps(shown)
