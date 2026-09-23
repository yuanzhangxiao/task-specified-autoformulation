#!/usr/bin/env python3
"""Synthetic critic handoff -> coherent revision -> real fit -> cached resume."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import general_critic as campaign
from autoformalism.rebuttal import general_critic_io as io
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)
from autoformalism.schemas.staged_topology import PublicScientificBrief
from scripts import smoke_process_pruning as pruning


def fixture(root: Path) -> dict:
    """A synthetic source with a real two-consumer shared law and causal boundaries."""
    source = pruning.fixture(root / "toy-source")
    row = source["rows"][0]
    raw = row["parent"]["request"]
    raw["base_candidate"]["states"][0]["kind"] = "observed"
    raw["base_candidate"]["initial_conditions"][0].update(
        fixed_value=None, expression="v01"
    )
    request = PublicFitRequest.model_validate(raw)
    cell = source["cells"]["toy"]
    cell["public_prompt"] = (
        "Generate v01 from u01 using causal memory and globally shared equations."
    )
    cell["brief"] = PublicScientificBrief.model_validate(
        {
            "scientific_context": cell["public_prompt"],
            "public_variables": [
                {"name": "u01", "data_role": "external_input"},
                {"name": "v01", "data_role": "target"},
            ],
            "requirements": [],
        }
    ).model_dump(mode="json")
    frozen = public._bundle(
        request,
        PublicSplit.model_validate(cell["training"]),
        PublicSplit.model_validate(cell["validation"]),
    )
    fit = PublicFitResult(
        **public._result_base(frozen, request),
        **row["parent"]["fit"],
        message="Synthetic known-parameter parent",
    ).model_dump(mode="json")
    config = public._read(io.REPO / "configs/shared_process_integration_v1.json")
    return sealed_write(
        root / "plan.json",
        {
            "protocol": io.PROTOCOL,
            "policy": io.POLICY,
            "source": str(root / "toy-source"),
            "source_plan_sha256": source["artifact_sha256"],
            "rows": [
                {
                    "task": row["task"],
                    "request": request.model_dump(mode="json"),
                    "fit": fit,
                    "bundle": io.make_bundle(request, cell, row["task"]),
                    "certificate": row["certificate"],
                    "pruning_origin": "parent",
                }
            ],
            "cells": {"toy": cell},
            "judge_protocol": campaign.judge.judge_protocol(),
            "judge_revision": "a" * 40,
            "proposer_settings": config["model_settings"],
            "serving_image_sha256": config["serving_image_sha256"],
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "launcher_sha256": io.launcher_hash(),
            "test_data_opened": False,
        },
    )


def prescribed_review(request, directory, base_url):
    """Transport fixture only; no claim about actual LLM judgments."""
    del directory, base_url
    return {
        "request_sha256": campaign.content_hash(request),
        "status": "reviewed",
        "findings": [],
        "cost": {
            "physical_requests": 0,
            "observed_total_tokens": 0,
            "usage_missing_events": 0,
        },
    }


def transport_for(calls, *, no_change=False):
    """One deliberate shared-law edit uses the usual whole-model reply schema."""

    def transport(url, body, timeout):
        del url, timeout
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        reply = (
            {"hypothesis": "Keep the current model."}
            if no_change
            else {
                "hypothesis": "Change the common memory law once for both consumers.",
                "equations": [
                    {
                        "component": "q",
                        "kind": "algebraic",
                        "expression": "z + 0.001*z*z",
                    }
                ],
            }
        )
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 70,
                "completion_tokens": 30,
            },
        }

    return transport


def run(root: Path) -> dict:
    fixture(root)
    assert campaign.prepare_evidence(root, 0)["status"] == "available"
    calls = []
    with patch.object(
        campaign.judge, "perform_review", side_effect=prescribed_review
    ) as reviewer:
        campaign.review_one(root, 0, "parent", "http://offline")
        proposal = campaign.propose_one(
            root, 0, "http://offline", transport=transport_for(calls)
        )
        assert proposal["status"] == "committed", proposal
        campaign.review_one(root, 0, "child", "http://offline")
        assert reviewer.call_count == 2
        result = campaign.fit_one(root, 0)
        assert result["status"] == "complete", result
        assert result["child_fit"]["validation"]["normalized_mse"] < 1e-4
        campaign.review_one(root, 0, "parent", "http://offline")
        campaign.review_one(root, 0, "child", "http://offline")
        assert (
            campaign.propose_one(
                root, 0, "http://offline", transport=transport_for(calls)
            )
            == proposal
        )
        assert campaign.fit_one(root, 0) == result
        assert reviewer.call_count == 2 and len(calls) == 1
    summary = campaign.report(root)
    assert summary["status_counts"] == {"complete": 1}
    return {
        "status": "passed",
        "real_fits": 1,
        "prescribed_reviews": 2,
        "prescribed_proposer_calls": 1,
        "live_llm_calls": 0,
        "additional_calls_or_fits_on_resume": 0,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="general-critic-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
