#!/usr/bin/env python3
"""Construct and fit a causal initializer, then transfer to unseen initial values."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
)
from autoformalism.fitting.initialization import LatentInitializationPlan
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.repair_comparison import RepairComparisonConfig, _assess
from autoformalism.schemas import CandidateModel
from autoformalism.search.causal_initialization import construct_initializers


def main():
    """Use a prescribed public synthetic map choice; make no live provider calls."""
    problem, _ = synthetic_problem("causal_map", 0.0, 0)
    candidate = CandidateModel.model_validate(problem["candidate"])
    context = ValidationContext.model_validate(problem["context"])
    requests = []

    def transport(url, body, timeout):
        requests.append(body)
        response = {
            "initial": {
                "mode": "causal_map",
                "expression": "m = a+b*v01",
                "parameters": [
                    {"name": "a", "role": "coefficient"},
                    {"name": "b", "role": "coefficient"},
                ],
            }
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(response)}}
            ],
            "usage": {"total_tokens": 100},
        }

    train, validation = (
        unpack_split(problem["splits"][key]) for key in ("train", "val")
    )
    with tempfile.TemporaryDirectory(prefix="causal-initialization-smoke-") as temp:
        root = Path(temp)
        settings = StagedModelSettings()
        client = StagedTopologyClient(
            settings=settings,
            base_url="http://unused",
            directory=root / "calls",
            namespace="causal-map-smoke",
            seed=0,
            transport=transport,
        )
        artifact = construct_initializers(
            candidate, context, {}, client, root / "initials"
        )
        (event,) = artifact["attempts"]
        assert event["accepted"]
        assert event["normalization"]["normalized_expression"] == "a+b*v01"
        config = RepairComparisonConfig(
            judge_revision="0" * 40,
            nonlinear_targets=(),
            fit=CollocationSensitivityConfig(
                initializer_seconds=20,
                refinement_seconds=20,
                maximum_function_evaluations=80,
                collocation_node_start="rollout_or_observed",
            ),
        )
        arguments = (
            root,
            {"arm": "redesigned_runtime", "seed": 0},
            root / "fit",
            candidate,
            candidate,
            LatentInitializationPlan.model_validate(artifact["plan"]),
            SimpleNamespace(train=train, validation=validation),
            context,
            "Synthetic input-memory control",
            config,
        )
        assessment = _assess(*arguments)
        fit = assessment["fit"]
        assert fit["status"] == "complete", fit
        assert fit["validation"]["normalized_mse"] < 1e-7, fit
        assert abs(fit["parameters"]["init_m_a"] - 1.0) < 1e-3
        assert abs(fit["parameters"]["init_m_b"] - 2.0) < 1e-3
        initials = fit["validation"]["trajectory_initial_conditions"]
        assert len({round(v["m"], 4) for v in initials.values()}) == 2
        assert _assess(*arguments) == assessment
        resumed_client = StagedTopologyClient(
            settings=settings,
            base_url="http://unused",
            directory=root / "calls",
            namespace="causal-map-smoke",
            seed=0,
            transport=transport,
        )
        assert (
            construct_initializers(
                candidate, context, {}, resumed_client, root / "initials"
            )
            == artifact
        )
        assert len(resumed_client.records) == len(client.records) == len(requests) == 1
        print(
            json.dumps(
                {
                    "status": "pass",
                    "initializer_normalization": event["normalization"],
                    "training_nmse": fit["training"]["normalized_mse"],
                    "validation_nmse": fit["validation"]["normalized_mse"],
                    "initializer_parameters": {
                        k: v
                        for k, v in fit["parameters"].items()
                        if k.startswith("init_")
                    },
                    "validation_initials": initials,
                    "resume_identical": True,
                    "validation_initials_fitted": False,
                    "live_llm_calls": 0,
                    "test_data_opened": False,
                    "private_reference_opened": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
