#!/usr/bin/env python3
"""Offline dependency edits, single shared law, causal handoff and exact resume."""

import argparse
import json
from pathlib import Path

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.prefit_construction_audit import reconstruct
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    EquationTerm,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import function_dependencies as dep
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import lower_topology, public_structure_checks


def fixture():
    """Prescribed two-store toy; no benchmark trajectories or reference equations."""
    brief = PublicScientificBrief(
        scientific_context="Two stores exchange through one threshold flow.",
        public_variables=[
            {"name": "y", "data_role": "target"},
            {"name": "u", "data_role": "external_input"},
            {"name": "area", "data_role": "covariate"},
            {"name": "crest", "data_role": "covariate"},
        ],
        requirements=[
            {
                "id": "memory",
                "targets": ["y"],
                "drivers": ["u"],
                "requires_dynamic_memory": True,
                "public_requirement": "Input-driven storage affects y.",
            }
        ],
    )
    context = ValidationContext(
        targets=("y",), external_inputs=("u",), fixed_covariates=("area", "crest")
    )
    inventory = tuple(
        ScientificVariable(name=n, definition=d, scientific_role=n)
        for n, d in (
            ("x", "differential"),
            ("y", "differential"),
            ("u", "supplied"),
            ("area", "supplied"),
            ("crest", "supplied"),
        )
    )
    inventory, audit = signed.admit(
        brief,
        inventory,
        {
            "processes": [
                {
                    "name": "q",
                    "depends_on": ["x", "u", "crest"],
                    "kind": "transfer",
                    "scientific_meaning": "One volumetric threshold flow.",
                    "uses": [
                        {"target": "x", "sign": "negative", "conversion": "1/area"},
                        {"target": "y", "sign": "positive", "conversion": "1"},
                    ],
                }
            ]
        },
    )
    bindings = audit["bindings"]
    definitions = (
        shared.definition(bindings[0]),
        *(
            EquationDefinition(
                name=n,
                definition="differential",
                terms=signed.assemble(
                    bindings,
                    n,
                    (
                        EquationTerm(
                            sources=(symbol,),
                            outer_weight_sign=sign,
                            scientific_role=role,
                        ),
                    ),
                ),
            )
            for n, symbol, sign, role in (
                ("x", "u", "positive", "input"),
                ("y", "y", "negative", "loss"),
            )
        ),
    )
    topology, _ = lower_topology(brief, inventory, definitions, context)
    source = {
        "complete_topology": True,
        "public_structure_checks_passed": True,
        "public_structure_checks": list(public_structure_checks(brief, definitions)),
        "inventory": [v.model_dump(mode="json") for v in inventory],
        "equations": [d.model_dump(mode="json") for d in definitions],
        "topology": topology.model_dump(mode="json"),
        "shared_process_contract": {"protocol": signed.POLICY, "bindings": bindings},
    }
    return brief, context, source


def transport(calls):
    def respond(url, body, timeout):
        text = body["messages"][1]["content"]
        payload = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        calls.append(payload)
        if "selected_term" in payload:
            reply = {
                "expression": "max(0,x-crest)",
                "parameters": [],
                "revise_dependencies": True,
            }
        elif "selected_equation" in payload:
            lhs = payload["selected_equation"]["lhs"]
            expression = {"q": "max(0,x-crest)", "x": "u/area", "y": "y"}[lhs]
            reply = {"functions": [{"expression": expression, "parameters": []}]}
        else:
            reply = {"initial": {"mode": "shared_value", "role": "coefficient"}}
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 50},
        }

    return respond


def construct(root, calls, *, responder=None, can_start=lambda: True):
    brief, context, source = fixture()
    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="dependency-smoke",
        seed=0,
        base_url="offline",
        transport=responder or transport(calls),
        can_start=can_start,
    )
    result = run_staged_functions(
        brief,
        context,
        source,
        client,
        root / "functions",
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
        initialization_policy="causal_training",
        dependency_policy=dep.POLICY,
    )
    return brief, context, source, result


def run(root):
    calls = []
    brief, context, source, result = construct(root, calls)
    assert result["complete_model"], result["error"]
    audit = reconstruct(
        {
            "brief": brief.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        },
        {"topology": source, "functions": result},
    )
    assert audit["certificate"]["passed"], audit
    assert [e["kind"] for e in result["dependency_revisions"]] == [
        "proposer_revision",
        "public_fixed_covariates",
    ]
    before = len(calls)
    assert construct(root, calls)[-1] == result
    assert len(calls) == before
    return {
        "status": "passed",
        "calls": before,
        "dependency_edits": 2,
        "independent_reconstruction": True,
        "resume_unchanged": True,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
