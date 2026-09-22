#!/usr/bin/env python3
"""Offline bounded repairs, exact replay and real RHS assembly; no fitting."""

import argparse
import json
from pathlib import Path

from autoformalism.construction import finalize_functional_draft
from autoformalism.expressions import (
    PiecewiseLinearForcing,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.process_gain_comparison import compile_bundle
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.candidate import StateKind
from autoformalism.schemas.staged_functions import LatentInitialReply
from autoformalism.search import process_assembly_revision as revision
from autoformalism.search import process_revision_runner as runner
from autoformalism.staged_functions import apply_initial_reply
from autoformalism.staged_topology import lower_topology
from scripts.smoke_function_dependencies import fixture


def client(root, calls, replies, *, can_start=lambda: True):
    """Use the real cached client with prescribed, inspectable offline responses."""

    def respond(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 50},
        }

    return StagedTopologyClient(
        settings=StagedModelSettings(),
        namespace="process-revision-smoke",
        seed=0,
        base_url="offline",
        directory=root / "calls",
        transport=respond,
        can_start=can_start,
    )


def rhs_check(brief, context, source, functions):
    """Compile a complete toy and check that the explicit conversion is applied once."""
    topology, draft = revision.bind_known(brief, context, source, functions)
    inventory = tuple(
        revision.ScientificVariable.model_validate(v) for v in source["inventory"]
    )
    equations = tuple(
        revision.EquationDefinition.model_validate(e) for e in source["equations"]
    )
    _, aliases = lower_topology(brief, inventory, equations, context)
    for state in topology.states:
        if state.kind == StateKind.LATENT:
            draft = apply_initial_reply(
                topology,
                draft,
                state.name,
                LatentInitialReply(initial={"fixed_value": 0.0}),
                context,
                aliases,
            )
    candidate = finalize_functional_draft(topology, draft, context).candidate
    handoff = {
        "candidate": candidate.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "initialization": {
            "identity": "prescribed-offline-toy-boundaries",
            "base_candidate": candidate.model_dump(mode="json"),
            "plan": LatentInitializationPlan().model_dump(mode="json"),
        },
    }
    bundle = compile_bundle(
        handoff,
        source["shared_process_contract"],
        "explicit_conversion",
        {"rows": [{"fixed_covariates": {"area": 2, "crest": 1}}]},
    )
    model = compile_candidate(
        CandidateModel.model_validate(bundle["candidate"]),
        ValidationContext.model_validate(bundle["context"]),
    )
    forcing = PiecewiseLinearForcing(
        [0],
        {"u": [0], "area": [2], "crest": [1]},
        allowed_channels=frozenset({"u", "area", "crest"}),
    )
    values = dict.fromkeys(model.parameter_names, 1.0)
    values.update(bundle["fit_parameter_guesses"])
    derivative = dict(
        zip(
            model.state_names,
            model.rhs(
                0,
                [{"x": 5, "y": 0}[s] for s in model.state_names],
                values,
                forcing,
            ),
            strict=True,
        )
    )
    assert derivative == {"x": -2.0, "y": 4.0}, derivative
    return derivative


def run(root):
    brief, context, source = fixture()
    known = {
        "term_1_0": {"expression": "u/area", "parameters": []},
        "term_1_1": {"expression": "q", "parameters": []},
        "term_2_0": {"expression": "y", "parameters": []},
        "term_2_1": {"expression": "q", "parameters": []},
    }
    # Supply the covariate dependency for this already accepted toy input law.
    source["equations"][1]["terms"][0]["sources"] = ["u", "area"]
    topology, _ = revision.bind_known(brief, context, source, known)
    source["topology"] = topology.model_dump(mode="json")
    calls = []
    correction = {
        "expression": "max(0,x-crest)",
        "parameters": [],
        "revise_dependencies": True,
    }
    c = client(root, calls, [correction])
    result = runner.run(
        brief,
        context,
        source,
        "term_0_0",
        known,
        c,
        root / "repair",
        initial_error="CONVERSION_REPAIR_REQUIRED",
    )
    assert result["status"] == "repaired", result
    transaction = result["transaction"]
    revision.replay(brief, context, source, known, transaction)
    derivative = rhs_check(
        brief, context, transaction["effective_source"], transaction["functions"]
    )
    before = len(calls)
    assert (
        runner.run(
            brief,
            context,
            source,
            "term_0_0",
            known,
            c,
            root / "repair",
            initial_error="CONVERSION_REPAIR_REQUIRED",
        )
        == result
    )
    assert len(calls) == before
    topology_calls = []
    replies = [
        {"expression": "u", "parameters": [], "revise_dependencies": True},
        {
            "expression": "u",
            "parameters": [],
            "revise_dependencies": True,
            "companion_laws": [
                {
                    "interaction_id": "term_2_0",
                    "expression": "y+x",
                    "parameters": [],
                    "revise_dependencies": True,
                }
            ],
        },
    ]
    t = runner.run(
        brief,
        context,
        source,
        "term_0_0",
        known,
        client(root / "topology", topology_calls, replies),
        root / "topology/repair",
        initial_error="dependency revision required",
    )
    assert t["status"] == "repaired", t
    assert [e["scope"] for e in t["events"]] == ["local", "topology"]
    revision.replay(brief, context, source, known, t["transaction"])
    return {
        "status": "passed",
        "local_repair_attempts": result["attempts_consumed"],
        "topology_repair_attempts": t["attempts_consumed"],
        "rhs": derivative,
        "independent_transaction_replay": True,
        "resume_added_calls": 0,
        "live_llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
