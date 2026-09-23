#!/usr/bin/env python3
"""Synthetic joint-output response feedback, real CPU fitting and exact resume."""

import argparse
import importlib.util
import json
from pathlib import Path

from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting

SPEC = importlib.util.spec_from_file_location(
    "multi_smoke", Path(__file__).with_name("smoke_review_multi.py")
)
MULTI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MULTI)
SPEC = importlib.util.spec_from_file_location(
    "response_cli", Path(__file__).with_name("review_response.py")
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def run(root: Path, *, protocol: str = io.RESPONSE_PROTOCOL) -> dict:
    """Use a real three-target fit; the proposer and tokenizer are software controls."""
    source, output = root / "source", root / "continuation"
    MULTI.fixture(source, real_fit=True, fitted_only=True)
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    plan = continuation.prepare(source, output, 2, 2, protocol=protocol)
    preview = CLI.prepare(output)
    assert CLI.prepare(output) == preview
    assert {e["target"] for e in preview["rows"][0]["evidence"]["examples"]} == {
        "Gp",
        "I",
        "U",
    }
    task = plan["tasks"][0]
    generation_calls, token_calls = [], []
    for index in (1, 2):

        def transport(url, body, timeout, visit=index):
            user = json.loads(body["messages"][1]["content"])
            generation_calls.append(user)
            assert "validation" not in user and "test" not in user
            assert set(user["training_evidence"]["target_overview"]) == {"Gp", "I", "U"}
            if visit == 2:
                raise ValueError("HTTP 400: synthetic delivery failure")
            raw = {
                "hypothesis": "Check a shared output offset.",
                "evidence_refs": [user["evidence_catalog"][0]["ref"]],
                "output_mappings": [{"channel": "U", "expression": "disposal+offset"}],
                "new_parameters": [{"name": "offset"}],
            }
            if protocol == io.INTEGRITY_PROTOCOL:
                assert user["parameter_declaration_policy"]
                if len(generation_calls) == 1:
                    raw["new_parameters"].append({"name": "c", "role": "time_constant"})
                else:
                    assert (
                        user["retry_feedback"]["code"]
                        == "EXISTING_PARAMETER_ROLE_CONFLICT"
                    )
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
                ],
                "usage": {"total_tokens": 100},
            }

        def tokenizer(url, body, timeout):
            token_calls.append(body)
            return {"count": 4000, "max_model_len": 32768}

        client = pipeline._client(
            output, plan, task, index, "http://offline", lambda: True, transport
        )
        client.token_transport = tokenizer
        proposal = pipeline.propose_one(output, plan, task, index, client)
        result = pipeline.fit_one(output, plan, task, index)
        assert result["selected"] is not None and not result["closed"]
        if index == 1:
            assert proposal["status"] == "committed", proposal
            assert result["trial"]["response_evidence"] is not None
            assert set(
                result["trial"]["fit"]["training"]["per_target_normalized_mse"]
            ) == {"Gp", "I", "U"}
            if protocol == io.INTEGRITY_PROTOCOL:
                assert len(proposal["attempts"]) == 2
                assert result["selection_audit"]["comparison_tolerance"] is not None
        else:
            assert proposal["status"] == "provider_request_failed", proposal
            assert len(proposal["attempts"]) == 1
            assert result["selected"] == io.read_round(output, task, 1)["selected"]
            assert result["trial"] is None
            assert not (
                io.round_path(output, task, index) / "fit/started.json"
            ).exists()
        assert pipeline.propose_one(output, plan, task, index, None) == proposal
        assert pipeline.fit_one(output, plan, task, index) == result
    reporting.report(output)
    assert {str(p): p.read_bytes() for p in source.rglob("*.json")} == before
    return {
        "status": "pass",
        "protocol": protocol,
        "real_multi_target_fits": True,
        "training_response_replay": True,
        "delivery_failure_preserves_without_refit": True,
        "exact_resume": True,
        "source_preserved": True,
        "mock_generation_requests": len(generation_calls),
        "mock_token_checks": len(token_calls),
        "live_llm_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=sorted(io.RESPONSE_PROTOCOLS),
        default=io.RESPONSE_PROTOCOL,
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, protocol=args.protocol), indent=2))
