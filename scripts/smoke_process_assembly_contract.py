#!/usr/bin/env python3
"""Exercise owned signs, focused conversion repair, reconstruction and cached resume."""

import argparse
import json
from pathlib import Path

from autoformalism.expressions import (
    PiecewiseLinearForcing,
    ValidationContext,
    compile_candidate,
)
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.prefit_construction_audit import reconstruct
from autoformalism.rebuttal.process_gain_comparison import compile_bundle
from autoformalism.schemas import CandidateModel
from autoformalism.search import function_dependencies as dependencies
from autoformalism.search import process_assembly_contract as assembly
from autoformalism.search.staged_function_runner import run_staged_functions
from scripts.smoke_function_dependencies import fixture


def construct(root, calls, *, intrinsic=False, assembly_policy=assembly.POLICY):
    """Use prescribed responses for an implementation smoke."""
    brief, context, source = fixture()

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_term" in payload:
            reply = {
                "expression": "-max(0,x-crest)/area"
                if intrinsic
                else "-max(0,x-crest)",
                "parameters": [],
                "revise_dependencies": True,
                "conversion_factor_is_intrinsic": intrinsic,
            }
        elif "selected_equation" in payload:
            lhs = payload["selected_equation"]["lhs"]
            expression = {"q": "-max(0,x-crest)/area", "x": "-u/area", "y": "-y"}[lhs]
            reply = {"functions": [{"expression": expression, "parameters": []}]}
        else:
            reply = {"initial": {"mode": "shared_value", "role": "coefficient"}}
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 50},
        }

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        directory=root / "calls",
        namespace="assembly-contract-smoke",
        seed=0,
        base_url="offline",
        transport=transport,
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
        dependency_policy=dependencies.POLICY,
        assembly_policy=assembly_policy,
    )
    return brief, context, source, result


def run(root):
    """Both correcting and explicitly retaining an intrinsic factor are executable."""
    reports = []
    for intrinsic in (False, True):
        calls = []
        branch = root / ("intrinsic" if intrinsic else "corrected")
        brief, context, source, result = construct(branch, calls, intrinsic=intrinsic)
        assert result["complete_model"], result["error"]
        audit = reconstruct(
            {
                "brief": brief.model_dump(mode="json"),
                "context": context.model_dump(mode="json"),
            },
            {"topology": source, "functions": result},
        )
        assert audit["certificate"]["passed"], audit
        # Evaluate both gain assemblies: a positive transfer must drain x and
        # feed y with exactly one area conversion, including after reconstruction.
        for policy in ("explicit_conversion", "independent_gains"):
            bundle = compile_bundle(
                audit["handoff"],
                result["shared_process_contract"],
                policy,
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
            expected = {"x": -1, "y": 2} if intrinsic else {"x": -2, "y": 4}
            assert derivative == expected
        assert (
            "CONSUMER_CONVERSION_OVERLAP"
            in result["batch_term_audits"][0]["batch_error"]
        )
        assert (
            result["assembly_decisions"][0]["intrinsic_factor_confirmed"] == intrinsic
        )
        assert (
            sum(bool(d["sign_normalizations"]) for d in result["assembly_decisions"])
            == 3
        )
        count = len(calls)
        assert construct(branch, calls, intrinsic=intrinsic)[-1] == result
        assert len(calls) == count
        reports.append(
            {
                "intrinsic": intrinsic,
                "calls": count,
                "reconstruction": True,
                "gain_assemblies_evaluated": 2,
                "resume": True,
            }
        )
    return {
        "status": "passed",
        "branches": reports,
        "live_llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
