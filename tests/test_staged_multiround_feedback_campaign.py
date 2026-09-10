"""Tests for function-first multi-round routing and structural boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import autoformalism.rebuttal.staged_multiround_feedback_campaign as campaign
from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from autoformalism.rebuttal.staged_multiround_feedback_campaign import (
    ComponentRevisionReply,
    MultiRoundFeedbackConfig,
    _route,
    apply_component_revision,
    launcher_hash,
    select_revision_components,
)
from autoformalism.schemas import CandidateModel
from autoformalism.search.identity import candidate_identity


def _bounded_feedback_reply() -> ComponentRevisionReply:
    return ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k_feedback*sigmoid(v01) - f/tau_feedback",
                    "parameters": [
                        {
                            "name": "k_feedback",
                            "role": "nonnegative_coefficient",
                        },
                        {"name": "tau_feedback", "role": "time_constant"},
                    ],
                },
                {
                    "component": "v01",
                    "expression": (
                        "offset + k_readout*sigmoid(f) + m + k_p*p + k_u*u01"
                    ),
                    "parameters": [
                        {"name": "offset", "role": "offset"},
                        {
                            "name": "k_readout",
                            "role": "nonnegative_coefficient",
                        },
                        {
                            "name": "k_p",
                            "role": "nonnegative_coefficient",
                        },
                        {
                            "name": "k_u",
                            "role": "nonnegative_coefficient",
                        },
                    ],
                },
            ]
        }
    )


def test_selects_small_nonlinear_closed_feedback_component() -> None:
    assert select_revision_components(recovery_candidate()) == ("f", "v01")


def test_function_revision_preserves_topology_and_compiles() -> None:
    candidate = recovery_candidate()
    revised, audit = apply_component_revision(
        candidate,
        _bounded_feedback_reply(),
        CONTEXT,
        selected=("f", "v01"),
        route="function_revision",
    )

    before = candidate_identity(candidate)
    after = candidate_identity(revised)
    assert before.topology_sha256 == after.topology_sha256
    assert before.functional_sha256 != after.functional_sha256
    assert audit["topology_changed"] is False
    assert revised.parent_candidate_id == candidate.candidate_id
    assert {item.name for item in revised.parameters} == {
        "k_feedback",
        "tau_feedback",
        "offset",
        "k_readout",
        "k_p",
        "k_u",
        "tau",
        "tau_p",
    }


def test_function_revision_rejects_dependency_change() -> None:
    payload = _bounded_feedback_reply().model_dump(mode="json")
    payload["revisions"][1]["expression"] = "offset + k_readout*sigmoid(f)"
    reply = ComponentRevisionReply.model_validate(payload)

    with pytest.raises(ValueError, match="changed topology sources"):
        apply_component_revision(
            recovery_candidate(),
            reply,
            CONTEXT,
            selected=("f", "v01"),
            route="function_revision",
        )


def test_persistent_instability_backtracks_only_after_function_round() -> None:
    assert _route(1, []) == "function_revision"
    assert _route(2, [{"numerically_stable": True}]) == "function_refinement"
    assert _route(2, [{"numerically_stable": False}]) == "topology_revision"


def test_exact_function_duplicate_is_rejected() -> None:
    candidate = recovery_candidate()
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k*v01**2/(1+v01**2) - f/tau_f",
                    "parameters": [
                        {"name": "k", "role": "nonnegative_coefficient"},
                        {"name": "tau_f", "role": "time_constant"},
                    ],
                },
                {
                    "component": "v01",
                    "expression": "k*f**2/(1+f**2) + m + k_p*p + k_u*u01 + c",
                    "parameters": [
                        {"name": "k", "role": "nonnegative_coefficient"},
                        {"name": "k_p", "role": "nonnegative_coefficient"},
                        {"name": "k_u", "role": "nonnegative_coefficient"},
                        {"name": "c", "role": "offset"},
                    ],
                },
            ]
        }
    )
    with pytest.raises(ValueError, match="did not change"):
        apply_component_revision(
            candidate,
            reply,
            CONTEXT,
            selected=("f", "v01"),
            route="function_revision",
        )


def test_topology_backtrack_must_preserve_existing_target_pathways() -> None:
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k_feedback*sigmoid(v01) - f/tau_f",
                    "parameters": [
                        {
                            "name": "k_feedback",
                            "role": "nonnegative_coefficient",
                        },
                        {"name": "tau_f", "role": "time_constant"},
                    ],
                },
                {
                    "component": "v01",
                    "expression": "m + k_p*p + k_u*u01 + c",
                    "parameters": [
                        {"name": "k_p", "role": "nonnegative_coefficient"},
                        {"name": "k_u", "role": "nonnegative_coefficient"},
                        {"name": "c", "role": "offset"},
                    ],
                },
            ]
        }
    )
    with pytest.raises(ValueError, match="disconnected an existing target pathway"):
        apply_component_revision(
            recovery_candidate(),
            reply,
            CONTEXT,
            selected=("f", "v01"),
            route="topology_revision",
        )


def test_frozen_campaign_configuration_and_launcher_are_valid() -> None:
    path = Path("configs/staged_multiround_feedback_v1.json")
    config = MultiRoundFeedbackConfig.model_validate_json(path.read_text())
    assert config.source_task_indices == (3, 4)
    assert config.round_count == 2
    assert config.fit.protocol == "collocation-forward-sensitivity-1"
    assert len(launcher_hash()) == 64


def test_freeze_is_bound_to_exact_unresolved_source_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    benchmark = "phase_b_anonymous_system_task_canonical_opaque_hard"
    public_ledger = {}
    for name in ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"):
        path = source / "frozen" / "public" / "phase_b_v1" / benchmark / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"public:{name}\n")
        public_ledger[str(path.relative_to(source))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    tasks = []
    for index in range(6):
        task_benchmark = benchmark if index >= 3 else "irrelevant_easy"
        seed = index - 3 if index >= 3 else index
        candidate_path = (
            source / "frozen" / "candidates" / f"candidate_{index:03d}.json"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(recovery_candidate().model_dump_json())
        tasks.append(
            {
                "task_index": index,
                "task_id": f"{task_benchmark}_seed{seed}",
                "benchmark_id": task_benchmark,
                "tier": "hard" if index >= 3 else "easy",
                "seed": seed,
                "candidate_path": str(candidate_path.relative_to(source)),
                "candidate_file_sha256": hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
            }
        )
        rescue_path = source / "tasks" / f"task_{index:03d}.json"
        rescue_path.parent.mkdir(parents=True, exist_ok=True)
        rescue_path.write_text(
            json.dumps(
                {
                    "attribution": (
                        "unresolved_after_bounded_rescue"
                        if index in {3, 4}
                        else "control"
                    )
                }
            )
        )
    source_plan = {
        "schema_version": "scientific-staged-fitter-rescue-plan-1",
        "tasks": tasks,
        "public_asset_ledger": public_ledger,
    }
    source_plan["plan_sha256"] = campaign.content_hash(source_plan)
    (source / "plan.json").write_text(json.dumps(source_plan))
    summary = {
        "status": "complete",
        "terminal_results": 6,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    summary_path = source / "summary" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary))

    output = tmp_path / "output"
    frozen = campaign.freeze_campaign(
        Path("configs/staged_multiround_feedback_v1.json"), source, output
    )

    assert len(frozen["tasks"]) == 2
    assert campaign._verified_plan(output) == frozen
    copied = output / frozen["tasks"][0]["candidate_path"]
    copied.write_text(copied.read_text() + " ")
    with pytest.raises(ValueError, match="frozen task artifact differs"):
        campaign._verified_plan(output)


def test_task_routes_persistent_instability_from_function_to_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v1.json").read_text()
    )
    plan = {"config": config.model_dump(mode="json")}
    monkeypatch.setattr(campaign, "_verified_plan", lambda _path: plan)
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *_args: (
            SimpleNamespace(train=object(), validation=object()),
            CONTEXT,
        ),
    )
    routes = []

    def revise(**kwargs):
        routes.append((kwargs["route"], kwargs["selected"]))
        parent = kwargs["candidate"]
        payload = parent.model_dump(mode="json")
        payload.update(
            candidate_id=f"revision_{len(routes)}",
            parent_candidate_id=parent.candidate_id,
            change_summary="mocked revision",
        )
        revised = CandidateModel.model_validate(payload)
        return revised, {
            "parent_sha256": campaign._candidate_hash(parent),
            "audit": {"topology_changed": kwargs["route"] == "topology_revision"},
        }

    fit_calls = []

    def fit(*_args, **_kwargs):
        fit_calls.append(len(fit_calls))
        if len(fit_calls) == 1:
            return {
                "status": "rollout_failed",
                "initializer": {"success": False, "message": "unstable"},
                "refinement": {"optimizer_success": False, "message": "unstable"},
                "training": None,
                "validation": None,
            }
        return {
            "status": "complete",
            "initializer": {"success": True, "message": "converged"},
            "refinement": {"optimizer_success": True, "message": "converged"},
            "training": {"normalized_mse": 1.0, "failed_trajectories": []},
            "validation": {"normalized_mse": 1.2, "failed_trajectories": []},
        }

    monkeypatch.setattr(campaign, "_request_revision", revise)
    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", fit)
    root = tmp_path / "campaign"
    candidate_path = root / "candidate.json"
    rescue_path = root / "rescue.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(recovery_candidate().model_dump_json())
    rescue_path.write_text(
        json.dumps({"attribution": "unresolved_after_bounded_rescue"})
    )
    benchmark = "phase_b_anonymous_system_task_canonical_opaque_hard"
    prompt = (
        root / "frozen" / "public" / "phase_b_v1" / benchmark / "proposer_prompt.txt"
    )
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Public scientific requirement.\nF. Required response\nSchema.")
    task = {
        "task_id": f"{benchmark}_seed0_multiround",
        "benchmark_id": benchmark,
        "tier": "hard",
        "seed": 0,
        "candidate_path": str(candidate_path.relative_to(root)),
        "candidate_file_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "source_rescue_path": str(rescue_path.relative_to(root)),
        "source_rescue_file_sha256": hashlib.sha256(
            rescue_path.read_bytes()
        ).hexdigest(),
    }

    result = campaign.run_task(
        root, task, root / "result", SimpleNamespace(records=[])
    )

    assert result["status"] == "complete"
    assert routes == [
        ("function_revision", ("f", "v01")),
        ("topology_revision", ("f", "v01")),
    ]
    assert [item["numerically_stable"] for item in result["rounds"]] == [False, True]
