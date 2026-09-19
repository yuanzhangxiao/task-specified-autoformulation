"""Prompt isolation, scientific lowering, matched budgets and deterministic resume."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.schemas import CandidateModel
from autoformalism.search import review_revision_v6
from autoformalism.search.shared_process_guidance import system_prompt
from autoformalism.search.staged_topology_prompts import (
    VARIABLE_IDENTIFICATION_SYSTEM_PROMPT,
)
from scripts import smoke_shared_process_pilot as smoke
from tests.test_repair_comparison import CONTEXT, candidate
from tests.test_review_deadline_v2 import example
from tests.test_review_revision_v6 import large_patch

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_pairs_and_only_guidance_varies():
    config = io.DeadlineConfig.model_validate_json(
        (ROOT / "configs/shared_process_pilot_v1.json").read_text()
    )
    tasks = io.tasks(config)
    assert len(tasks) == 24 and config.rounds == 2
    assert len({t["task_id"] for t in tasks}) == 24
    assert config.fit_profile == "collocation-single-target-v2"
    for cell in config.public_cells:
        for seed in config.seeds:
            for arm in ("full", "brief_only"):
                pair = [
                    t
                    for t in tasks
                    if (t["cell"], t["seed"], t["arm"]) == (cell, seed, arm)
                ]
                assert {t["process_guidance"] for t in pair} == {"on", "off"}
                assert all(t["shared_round_zero"] is None for t in pair)
    assert io.SHARED_PROTOCOL not in io.CONTINUATION_PROTOCOLS
    with pytest.raises(ValueError, match="two rounds"):
        io.DeadlineConfig.model_validate({**config.model_dump(), "rounds": 3})


def test_prompt_preserves_control_and_scientific_freedom():
    original = VARIABLE_IDENTIFICATION_SYSTEM_PROMPT
    assert system_prompt(original, "variables", False) == original
    guided = system_prompt(original, "variables", True)
    assert "only when the displayed variables cannot represent" not in guided
    assert "no independent\ninitial condition" in guided
    assert "even if the same law could be written inline" in guided
    assert "independent consumer gains" in guided
    assert "Do not propose equations" in guided
    with pytest.raises(ValueError, match="prompt changed"):
        system_prompt("unexpected instruction", "variables", True)


@pytest.mark.parametrize("arm", ["full", "brief_only"])
def test_construction_reuses_one_law_and_exact_call_resume(tmp_path, arm):
    plan = smoke.fixture(tmp_path)
    for guidance in ("off", "on"):
        task = next(
            t
            for t in plan["tasks"]
            if t["arm"] == arm and t["process_guidance"] == guidance
        )
        calls = []

        def client(task=task, calls=calls):
            return pipeline._client(
                tmp_path,
                plan,
                task,
                0,
                "http://offline",
                lambda: True,
                smoke.transport_for(calls),
            )

        result = pipeline.propose_one(tmp_path, plan, task, 0, client())
        assert result["status"] == "constructed", result
        base = result["bundle"]["initialization"]["base_candidate"]
        assert len(base["processes"]) == 1
        assert all("q" in e["rhs"] for e in base["state_equations"])
        # Two local gains and one law parameter; no duplicated consumer gain.
        assert len(base["parameters"]) == 3
        before = len(calls)
        pipeline.propose_one(tmp_path, plan, task, 0, client())
        assert len(calls) == before
        for call in calls:
            payload = call["payload"]
            if "selected_state" not in payload:
                assert ("Shared-process modeling guidance" in call["system"]) == (
                    guidance == "on"
                )
            assert ("training_observations" in payload.get("public_brief", {})) == (
                arm == "full"
            )
        # A changed guidance flag cannot resume the same topology root/call namespace.
        from autoformalism.expressions import ValidationContext
        from autoformalism.schemas.staged_topology import PublicScientificBrief
        from autoformalism.search.staged_topology_runner import run_staged_topology

        cell = plan["cells"][task["cell"]]
        if guidance == "on" and arm == "brief_only":
            with pytest.raises(ValueError, match="evidence contract differs"):
                run_staged_topology(
                    PublicScientificBrief.model_validate(cell["brief"]),
                    ValidationContext.model_validate(cell["context"]),
                    client(),
                    io.round_path(tmp_path, task, 0) / "topology",
                    shared_process_guidance=False,
                )


@pytest.mark.parametrize("guidance", ["off", "on"])
def test_campaign_revision_uses_v6_for_both_arms(monkeypatch, guidance):
    bundle, packet, values, *_ = example()
    parent = {
        "selected": {"bundle": bundle, "packet": packet, "fit": {"parameters": values}}
    }
    seen = []

    class Client:
        def call(self, **kw):
            seen.append(kw)
            return {
                "request_hash": "x",
                "response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(large_patch())},
                        }
                    ]
                },
            }

    monkeypatch.setattr(pipeline, "visible_response", lambda record: large_patch())
    monkeypatch.setattr(
        pipeline,
        "certificates",
        lambda *args: {"eligible_for_development_selection": True},
    )
    result = pipeline._content_revision(
        {"protocol": io.SHARED_PROTOCOL, "cells": {"demo": {}}},
        {"cell": "demo", "arm": "full", "process_guidance": guidance},
        parent,
        Client(),
    )
    assert result["status"] == "committed", result
    assert seen[0]["response_model"] is review_revision_v6.ScientificRevision
    assert result["revision_policy"] == "scientific-content-revision-6"
    assert ("Shared-process modeling guidance" in seen[0]["system"]) == (
        guidance == "on"
    )


def test_missing_models_and_no_fake_reuse_success(tmp_path):
    plan = smoke.fixture(tmp_path)
    report = reporting.report(tmp_path)
    assert report["status_counts"] == {"missing": 8}
    assert all(r["shared_process_evidence"] is None for r in report["rows"])
    assert (tmp_path / "SHARED_PROCESS_SUMMARY.md").exists()
    task = plan["tasks"][0]
    assert isinstance(
        pipeline._client(tmp_path, plan, task, 1, "http://unused", lambda: False),
        RevisionClient,
    )


@pytest.mark.parametrize("mode", ["transfer", "converted", "response"])
def test_inline_named_law_values_and_exact_parameter_partials_agree(mode):
    pytest.importorskip("casadi")
    from autoformalism.fitting.sensitivity_probe import SymbolicODE

    value = candidate().model_dump(mode="json")
    law = "rate*(m-y)" if mode == "transfer" else "rate*tanh(m-y)"
    left, right = ("-q+u01", "q") if mode == "transfer" else ("-q+u01", "gain*q")
    if mode == "response":
        left, right = "gain*q+u01", "decay*q"
    value["processes"] = [{"name": "q", "expression": law}]
    value["state_equations"] = [
        {"state": "m", "rhs": left},
        {"state": "y", "rhs": right},
    ]
    used = (
        {"rate"}
        | ({"gain"} if mode != "transfer" else set())
        | ({"decay"} if mode == "response" else set())
    )
    value["parameters"] = [p for p in value["parameters"] if p["name"] in used]
    inline = copy.deepcopy(value)
    inline["processes"] = []
    for e in inline["state_equations"]:
        e["rhs"] = e["rhs"].replace("q", f"({law})")
    a, b = [
        SymbolicODE(compile_candidate(CandidateModel.model_validate(x), CONTEXT))
        for x in (value, inline)
    ]
    assert a.names == b.names
    for state in ([0.2, 1.1], [-1.5, 0.3]):
        args = (0.4, state, np.linspace(0.3, 1.3, len(a.names)), [0.7])
        np.testing.assert_allclose(a.rhs(*args), b.rhs(*args), atol=1e-12)
        for x, y in zip(a.local(*args), b.local(*args), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-12)
    if mode == "transfer":
        assert float(np.sum(a.rhs(*args))) == pytest.approx(0.7)
    bad = copy.deepcopy(value)
    bad["processes"][0]["expression"] = "q+m"
    with pytest.raises(ModelValidationError):
        compile_candidate(CandidateModel.model_validate(bad), CONTEXT)


def test_nonlinear_law_is_assembled_once_without_second_consumer_nonlinearity(tmp_path):
    plan = smoke.fixture(tmp_path)
    task = next(t for t in plan["tasks"] if t["process_guidance"] == "on")
    calls = []
    client = pipeline._client(
        tmp_path,
        plan,
        task,
        0,
        "http://offline",
        lambda: True,
        smoke.transport_for(calls, nonlinear=True),
    )
    proposal = pipeline.propose_one(tmp_path, plan, task, 0, client)
    assert proposal["status"] == "constructed", proposal
    model = proposal["bundle"]["initialization"]["base_candidate"]
    assert "tanh" in model["processes"][0]["expression"]
    assert all(
        "tanh" not in e["rhs"] and "q" in e["rhs"] for e in model["state_equations"]
    )
    # Reusing an output mapping alone must not inflate governing-law sharing.
    from autoformalism.rebuttal.repair_transactions import default_initialization
    from autoformalism.rebuttal.shared_process_pilot import model_evidence

    base = candidate().model_dump(mode="json")
    base["state_equations"][0]["rhs"] = "-rate*m+gain*u01+f"
    base["state_equations"][1]["rhs"] = "-decay*y+m"
    base["observation_mappings"][0]["expression"] = "f"
    base["states"][1]["kind"] = "latent"
    base["initial_conditions"][1] = {"state": "y", "scope": "global", "fixed_value": 0}
    base = CandidateModel.model_validate(base)
    initial = default_initialization(base, CONTEXT)
    evidence = model_evidence(
        {
            "bundle": {
                "candidate": base.model_dump(mode="json"),
                "initialization": {
                    "context": CONTEXT.model_dump(mode="json"),
                    "plan": initial.model_dump(mode="json"),
                },
            },
            "certificate": {
                "mechanisms": {},
                "targets": {},
                "all_public_graph_requirements_certified": False,
            },
        }
    )
    assert evidence["counts"]["directly_reused_named_processes"] == 1
    assert (
        evidence["counts"]["named_processes_shared_between_governing_definitions"] == 0
    )
