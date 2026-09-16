#!/usr/bin/env python3
"""Synthetic history, real training replay and child fitting, mocked LLM delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autoformalism.data import TrainingScaler
from autoformalism.fitting import fit_continuation as continuation
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.continuation_numerics import _score
from autoformalism.rebuttal import prefit_fit_handoff as handoff
from autoformalism.rebuttal import prefit_numerical_sibling as sibling
from autoformalism.rebuttal import prefit_requirements as requirements
from autoformalism.rebuttal.prefit_feedback import EpisodeClient
from autoformalism.schemas.fit_continuation import ContinuationSelection
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)
from autoformalism.schemas.residual_feedback import FeedbackSelection

if __package__:
    from scripts.smoke_prefit_requirements import repair_client, synthetic_plan
else:
    from smoke_prefit_requirements import repair_client, synthetic_plan


def _scores(frozen, parameters):
    request = PublicFitRequest.model_validate(frozen["request"])
    model, _, _ = public._lower(request)
    train = public.unpack_split(PublicSplit.model_validate(frozen["training"]))
    val = public.unpack_split(PublicSplit.model_validate(frozen["validation"]))
    scale = TrainingScaler().fit(train).scales["target:v01"].standard_deviation
    settings = CollocationSensitivityConfig.model_validate(
        frozen["settings"]
    ).fit_config()
    training = _score(model, train, parameters, scale, settings)["metrics"]
    validation = _score(model, val, parameters, scale, settings)["metrics"]
    assert training["normalized_mse"] is not None
    assert validation["normalized_mse"] is not None
    cost = (
        0.5 * sum(len(t.time) for t in train.trajectories) * training["normalized_mse"]
    )
    allowed = ("normalized_mse", "per_target_normalized_mse", "failed_trajectories")
    return (
        {k: training[k] for k in allowed},
        {k: validation[k] for k in allowed},
        cost,
    )


def fixture(base: Path, *, feedback_policy="optional-review-1") -> Path:
    """Seal test-only timeout histories with real production scores at both points."""
    plan = synthetic_plan(
        base, with_validation=True, repair_policy="topology-owned-sign-1"
    )
    source = base / "campaign"
    task = next(
        t
        for t in plan["tasks"]
        if t["arm"] == "requirement_feedback" and t["cohort"] == "repair"
    )
    requirements.run_episode(source, plan, task, repair_client(source, plan, task, []))
    selection = handoff.HandoffSelection(
        source_plan_sha256=plan["artifact_sha256"],
        construction_plan_sha256=plan["source_plan_sha256"],
        task_id=task["task_id"],
        profile="collocation-feasible-v1",
    )
    handoff.prepare_handoff(
        source, base / "source", base / "source/public", selection, base / "parent"
    )
    parent = base / "parent/fit"
    frozen = public._read(parent / "freeze.json")
    request = PublicFitRequest.model_validate(frozen["request"])
    model, _, _ = public._lower(request)
    parameters = {}
    current = {}
    for name in model.parameter_names:
        if name.startswith("init_"):
            value = 1.0 if name.endswith("_a") else 2.0
            parameters[name] = current[name] = value
        else:
            stem = name.split("_", 1)[0]
            parameters[name] = {"a": 0.35, "b": 0.75, "c": 0.002, "k": 0.75}[stem]
            current[name] = {"a": 0.30, "b": 0.80, "c": 0.001, "k": 0.70}[stem]
    train, val, cost = _scores(frozen, parameters)
    stage = {
        "parameters": parameters,
        "cost": cost,
        "optimizer_status": -2,
        "optimizer_native_success": False,
        "best_evaluated": {"parameters": parameters, "cost": cost, "call": 5},
        "iterations": [
            {
                "iteration": i,
                "nfev": i + 2,
                "cost": cost * factor,
                "parameters": parameters,
            }
            for i, factor in enumerate((1.3, 1.2, 1.1, 1.0))
        ],
    }
    raw = {
        "parameters": parameters,
        "training": train,
        "validation": val,
        "refinement": {
            "budget_exhausted": True,
            "actual_residual_calls": 6,
            "stages": [{"mode": "sensitivity", "result": stage}],
        },
        "synthetic_test_history": True,
    }
    result = PublicFitResult(
        **public._result_base(frozen, request),
        **public._evidence(request, raw),
        status="complete",
        backend_result_sha256=public.content_sha256(raw),
        message="Synthetic history fixture with real fixed-point scores.",
    )
    public._write(parent / "backend_result.json", raw)
    public._write(
        parent / "result.json",
        {
            "result": result.model_dump(mode="json"),
            "sha256": public.content_sha256(result),
        },
    )
    next_train, next_val, next_cost = _scores(frozen, current)
    assert next_cost < cost
    pilot = base / "pilot/continuation"
    continuation.prepare_continuation(
        parent,
        ContinuationSelection(
            parent_lowered_candidate_sha256=result.lowered_candidate_sha256,
            parent_backend_result_sha256=result.backend_result_sha256,
        ),
        pilot,
    )
    extension = {
        "initial_parameters": parameters,
        "best_evaluated": {"parameters": current, "cost": next_cost},
        "start_check": {"agrees": True, "cost": cost},
        "result_fields": {
            "status": "complete",
            "selected": "extension",
            "parameters": current,
            "training": {"available": True, **next_train},
            "validation": {"available": True, **next_val},
            "extension_budget_exhausted": True,
            "extension_residual_calls": 3,
            "cumulative_residual_calls": 9,
            "extension_numerical_seconds": 180.0,
            "native_optimizer_converged": None,
            "feedback_status": "budget_limited_unresolved",
            "message": "Synthetic test history",
        },
        "actual_residual_calls": 3,
        "numerical_seconds": 180.0,
        "setup_seconds": 1.0,
        "scoring_seconds": 2.0,
        "budget_exhausted": True,
        "integration_failures": 0,
        "failure": None,
        "scoring_error": None,
        "optimizer": {
            "optimizer_native_success": False,
            "optimizer_status": -2,
            "optimality": None,
            "message": "synthetic history",
            "iterations": [{"iteration": 1, "cost": next_cost}],
        },
    }
    with patch.object(continuation, "_run_extension", return_value=extension):
        completed = continuation.execute_continuation(pilot)
    assert completed.status == "complete", completed
    config = sibling.SiblingConfig(
        feedback_policy=feedback_policy,
        serving_image_sha256="0" * 64,
        selection=FeedbackSelection(
            continuation_identity=completed.identity,
            continuation_backend_sha256=completed.backend_result_sha256,
        ),
        model_settings=plan["config"]["model_settings"],
    )
    config_path = base / "sibling-config.json"
    config_path.write_text(config.model_dump_json())
    root = base / "sibling"
    sibling.prepare(parent, pilot, source, base / "source", config_path, root)
    return root


def client_for(root, plan, calls, *, action="revise_function"):
    """Mock provider transport while exercising the production cache and schema."""

    def transport(url, body, timeout):
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        selected = next(
            s for s in payload["interactions"] if s["selected_term"]["lhs"] == "m"
        )
        revision = {
            **selected["current_reply"],
            "interaction_id": selected["interaction_id"],
        }
        revision["expression"] = revision["expression"].replace("v01**2", "tanh(v01)")
        reply = {
            "action": action,
            "hypothesis": (
                "A saturating feedback response is an exploratory hypothesis; "
                "incomplete fitting may also explain this error."
            ),
            "evidence_ids": [payload["training_evidence"]["rows"][0]["evidence_id"]],
            "revision": revision if action == "revise_function" else None,
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return EpisodeClient(
        settings=sibling.SiblingConfig.model_validate(plan["config"]).model_settings,
        base_url="http://unused",
        directory=root / "results/calls",
        namespace=sibling._namespace(plan),
        seed=0,
        transport=transport,
    )


def smoke(base: Path, *, feedback_policy="optional-review-1") -> dict:
    """Check the new pipeline edge with real export and frozen child optimizer."""
    root = fixture(base, feedback_policy=feedback_policy)
    original = {
        str(p): p.read_bytes()
        for name in ("source", "campaign", "parent", "pilot")
        for p in (base / name).rglob("*")
        if p.is_file()
    }
    assert sibling.replay(root)["status"] == "ready"
    plan, residual = sibling.verify(root)
    calls = []
    client = client_for(root, plan, calls)
    state = sibling.run_episode(root, plan, residual, client)
    assert state["decision"]["outcome"] == "committed", state
    provider_payload = json.loads(calls[0]["messages"][1]["content"].split("\n", 1)[1])
    assert (
        provider_payload["retained_fitted_parameters"]
        == residual["seed"]["state"]["parameters"]
    )
    result = sibling.fit_child(root)
    assert result["status"] == "complete"
    child = result["child"]["result"]
    assert child["status"] == "complete", child
    assert child["training"]["normalized_mse"] < 1e-4, child
    assert child["validation"]["normalized_mse"] < 1e-4, child
    assert result["child"]["seed"]["retained_initializer_parameters"]
    assert sibling.run_episode(root, plan, residual, client) == state
    assert sibling.fit_child(root) == sibling.report(root) == result
    assert len(calls) == 1
    assert original == {
        str(p): p.read_bytes()
        for name in ("source", "campaign", "parent", "pilot")
        for p in (base / name).rglob("*")
        if p.is_file()
    }
    return {
        "status": "passed",
        "feedback_policy": feedback_policy,
        "synthetic_histories": True,
        "live_llm_calls": 0,
        "mock_provider_requests": len(calls),
        "real_training_replays": True,
        "real_frozen_child_fitting": True,
        "historical_files_unchanged": True,
        "resume_unchanged": True,
        "child_training_nmse": child["training"]["normalized_mse"],
        "child_validation_nmse": child["validation"]["normalized_mse"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback-policy",
        choices=("optional-review-1", "routed-hypothesis-2"),
        default="optional-review-1",
    )
    args = parser.parse_args()
    with TemporaryDirectory(prefix="prefit-sibling-") as temporary:
        print(
            json.dumps(
                smoke(Path(temporary), feedback_policy=args.feedback_policy), indent=2
            )
        )


if __name__ == "__main__":
    main()
