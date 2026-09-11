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
    RevisionContractError,
    _route,
    apply_component_revision,
    launcher_hash,
    select_revision_components,
    state_dependent_denominator_findings,
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


def test_dense_cycle_prefers_nonlinear_target_source_over_whole_scc() -> None:
    dense = CandidateModel.model_validate(
        {
            "candidate_id": "dense_feedback",
            "parent_candidate_id": None,
            "states": [
                {"name": name, "kind": "latent"} for name in ("c", "f", "m")
            ],
            "state_equations": [
                {"state": "c", "rhs": "f**2 + m - c"},
                {"state": "f", "rhs": "v01**2 + c - f"},
                {"state": "m", "rhs": "u01 + c - m"},
            ],
            "processes": [
                {"name": "v01", "expression": "f**2 + c + m + u01"}
            ],
            "observation_mappings": [{"channel": "v01", "expression": "v01"}],
            "parameters": [],
            "initial_conditions": [
                {"state": name, "fixed_value": 0, "scope": "global"}
                for name in ("c", "f", "m")
            ],
        }
    )

    assert select_revision_components(dense) == ("f", "v01")


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


def test_reused_parameter_names_preserve_parent_roles() -> None:
    candidate = recovery_candidate()
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k*sigmoid(v01) - f/tau_f",
                    "parameters": [
                        {"name": "k", "role": "positive_shape"},
                        {"name": "tau_f", "role": "coefficient"},
                    ],
                }
            ]
        }
    )

    revised, audit = apply_component_revision(
        candidate,
        reply,
        CONTEXT,
        selected=("f",),
        route="function_revision",
    )

    roles = {item.name: item.role.value for item in revised.parameters}
    assert roles["k"] == "nonnegative_coefficient"
    assert roles["tau_f"] == "time_constant"
    assert {item["parameter"] for item in audit["parameter_role_derivations"]} == {
        "k",
        "tau_f",
    }
    assert {
        item["code"] for item in audit["parameter_role_derivations"]
    } == {"REUSED_PARAMETER_ROLE_PRESERVED"}


def test_parent_parameters_are_available_without_redeclaration() -> None:
    candidate = recovery_candidate()
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k*tanh(v01) - f/tau_f",
                    "parameters": [],
                }
            ]
        }
    )

    revised, audit = apply_component_revision(
        candidate,
        reply,
        CONTEXT,
        selected=("f",),
        route="function_revision",
    )

    assert next(item.rhs for item in revised.state_equations if item.state == "f") == (
        "k*tanh(v01) - f/tau_f"
    )
    inherited = {
        item["parameter"]
        for item in audit["parameter_role_derivations"]
        if item.get("declaration_omitted")
    }
    assert inherited == {"k", "tau_f"}


def test_state_dependent_denominator_audit_is_exact_and_conservative() -> None:
    candidate = recovery_candidate()
    assert state_dependent_denominator_findings(candidate) == []
    payload = candidate.model_dump(mode="json")
    for equation in payload["state_equations"]:
        if equation["state"] == "f":
            equation["rhs"] = "k*v01/(1+v01) - f/tau_f"
    unsafe = CandidateModel.model_validate(payload)

    findings = state_dependent_denominator_findings(unsafe)

    assert findings == [
        {
            "component": "f",
            "denominator": "1 + v01",
            "generated_symbols": ["v01"],
            "possible_singularity": "v01 = -1",
            "certificate_status": "not_certified_away_from_zero",
            "interpretation": (
                "potential domain defect only; numerical reachability has not "
                "been inferred"
            ),
        }
    ]


def test_domain_finding_keeps_next_route_at_function_level() -> None:
    finding = {
        "component": "f",
        "denominator": "1 + f",
    }
    assert _route(
        2,
        [
            {
                "route": "function_revision",
                "fit": {"failure_class": "function_domain"},
                "numerically_stable": False,
            }
        ],
        [finding],
    ) == "function_revision"


def test_revision_retry_retains_valid_component_and_requests_only_pending(
    tmp_path: Path,
) -> None:
    responses = [
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k*tanh(v01) - f/tau_f",
                    "parameters": [],
                },
                {
                    "component": "v01",
                    "expression": "invented*f + m + k_p*p + k_u*u01 + c",
                    "parameters": [],
                },
            ]
        },
        {
            "revisions": [
                {
                    "component": "v01",
                    "expression": "k*tanh(f) + m + k_p*p + k_u*u01 + c",
                    "parameters": [],
                }
            ]
        },
    ]

    class PartialClient:
        settings = SimpleNamespace(attempts_per_step=2)

        def __init__(self) -> None:
            self.users = []

        def call(self, **kwargs):
            attempt = kwargs["attempt"]
            self.users.append(json.loads(kwargs["user"]))
            return {
                "request_hash": f"request_{attempt}",
                "status": "responded",
                "raw_response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(responses[attempt])},
                        }
                    ]
                },
            }

    client = PartialClient()
    revised, record = campaign._request_revision(
        client=client,
        route="function_revision",
        candidate=recovery_candidate(),
        selected=("f", "v01"),
        context=CONTEXT,
        scientific_context="Public context",
        numerical_feedback={"failure": "unstable"},
        round_index=1,
        failure_checkpoint=tmp_path / "revision_failures.json",
    )

    assert [
        item["component"]
        for item in client.users[1]["selected_pending_components"]
    ] == ["v01"]
    assert client.users[1]["provisionally_retained_components"] == ["f"]
    assert record["audit"]["provisionally_retained_component_count"] == 2
    assert record["attempts"][0]["pending_components_after"] == ["v01"]
    assert record["attempts"][1]["pending_components_after"] == []
    equations = campaign._component_expressions(revised)
    assert equations["f"] == "k*tanh(v01) - f/tau_f"
    assert equations["v01"] == "k*tanh(f) + m + k_p*p + k_u*u01 + c"


def test_new_direct_gains_and_offset_receive_runtime_roles() -> None:
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "v01",
                    "expression": (
                        "baseline + gain_f*sigmoid(f) + gain_m*m + "
                        "gain_p*p + gain_u*u01"
                    ),
                    "parameters": [
                        {"name": name, "role": None}
                        for name in (
                            "baseline",
                            "gain_f",
                            "gain_m",
                            "gain_p",
                            "gain_u",
                        )
                    ],
                }
            ]
        }
    )

    revised, audit = apply_component_revision(
        recovery_candidate(),
        reply,
        CONTEXT,
        selected=("v01",),
        route="function_revision",
    )

    roles = {item.name: item.role.value for item in revised.parameters}
    assert roles["baseline"] == "offset"
    assert all(
        roles[name] == "nonnegative_coefficient"
        for name in ("gain_f", "gain_m", "gain_p", "gain_u")
    )
    assert len(audit["parameter_role_derivations"]) == 5


def test_ambiguous_internal_parameter_gets_named_shape_feedback() -> None:
    reply = ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "v01",
                    "expression": "sigmoid(theta*f) + m + p + u01 + baseline",
                    "parameters": [
                        {"name": "theta", "role": "coefficient"},
                        {"name": "baseline", "role": "coefficient"},
                    ],
                }
            ]
        }
    )

    with pytest.raises(RevisionContractError) as caught:
        apply_component_revision(
            recovery_candidate(),
            reply,
            CONTEXT,
            selected=("v01",),
            route="function_revision",
        )
    diagnostic = caught.value.diagnostic()
    assert diagnostic["code"] == "AMBIGUOUS_INTERNAL_PARAMETER_ROLE"
    assert diagnostic["details"]["parameter"] == "theta"
    assert diagnostic["details"]["allowed_roles"] == [
        "shape",
        "positive_shape",
        "rate",
        "time_constant",
        "scale",
    ]


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
    for version in ("v1", "v2", "v3"):
        path = Path(f"configs/staged_multiround_feedback_{version}.json")
        config = MultiRoundFeedbackConfig.model_validate_json(path.read_text())
        assert config.source_task_indices == (3, 4)
        assert config.round_count == 2
        assert config.fit.protocol == "collocation-forward-sensitivity-1"
    v4 = MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v4.json").read_text()
    )
    assert v4.protocol == "scientific-staged-multiround-feedback-4"
    assert v4.round_count == 4
    assert v4.model_settings.attempts_per_step == 5
    v5 = MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v5.json").read_text()
    )
    assert v5.protocol == "scientific-staged-multiround-feedback-5"
    assert v5.round_count == 4
    assert v5.fit.collocation_node_start == "rollout_or_observed"
    assert v5.fit.node_warmup_seconds == 10.0
    assert campaign._artifact_schema(v5.protocol, "summary").endswith("-5")
    assert len(launcher_hash()) == 64


def test_revision_exhaustion_persists_full_responses_and_named_feedback(
    tmp_path: Path,
) -> None:
    response = {
        "revisions": [
            {
                "component": "v01",
                "expression": "sigmoid(theta*f) + m + p + u01 + baseline",
                "parameters": [
                    {"name": "theta", "role": "coefficient"},
                    {"name": "baseline", "role": "coefficient"},
                ],
            }
        ]
    }

    class RejectingClient:
        settings = SimpleNamespace(attempts_per_step=2)

        def call(self, **kwargs):
            attempt = kwargs["attempt"]
            return {
                "request_hash": f"request_{attempt}",
                "status": "responded",
                "raw_response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(response)},
                        }
                    ]
                },
            }

    checkpoint = tmp_path / "revision_failures.json"
    with pytest.raises(
        RevisionContractError, match="bounded component revision exhausted"
    ):
        campaign._request_revision(
            client=RejectingClient(),
            route="function_revision",
            candidate=recovery_candidate(),
            selected=("v01",),
            context=CONTEXT,
            scientific_context="Public context",
            numerical_feedback={"failure": "unstable"},
            round_index=1,
            failure_checkpoint=checkpoint,
        )

    ledger = json.loads(checkpoint.read_text())
    assert len(ledger["attempts"]) == 2
    assert all(item["rejected_response"] == response for item in ledger["attempts"])
    assert all(
        item["component_results"][0]["diagnostic"]["code"]
        == "AMBIGUOUS_INTERNAL_PARAMETER_ROLE"
        for item in ledger["attempts"]
    )


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
    assert frozen["latent_initialization_policy"] == (
        "source_candidate_fixed_or_causal_initializer_held_fixed"
    )
    assert frozen["latent_initial_values_learned_during_training"] is False
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


def test_revision_exhaustion_retains_parent_and_continues_later_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v4.json").read_text()
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
    calls = []

    def revise(**kwargs):
        calls.append(kwargs["route"])
        if len(calls) == 1:
            campaign.atomic_json(
                kwargs["failure_checkpoint"],
                {
                    "schema_version": "component-revision-failures-2",
                    "attempts": [],
                },
            )
            raise RevisionContractError(
                "COMPONENT_REVISION_ATTEMPTS_EXHAUSTED",
                "bounded component revision exhausted for function_revision",
            )
        parent = kwargs["candidate"]
        payload = parent.model_dump(mode="json")
        payload.update(
            candidate_id=f"revision_{len(calls)}",
            parent_candidate_id=parent.candidate_id,
            change_summary="mocked revision",
        )
        return CandidateModel.model_validate(payload), {
            "parent_sha256": campaign._candidate_hash(parent),
            "attempts": [],
            "audit": {"topology_changed": False},
        }

    monkeypatch.setattr(campaign, "_request_revision", revise)
    monkeypatch.setattr(
        campaign,
        "fit_collocation_forward_sensitivity",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "initializer": {"success": True},
            "refinement": {"optimizer_success": True},
            "training": {"normalized_mse": 1.0, "failed_trajectories": []},
            "validation": {"normalized_mse": 1.1, "failed_trajectories": []},
        },
    )
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
    prompt.write_text("Public requirement.\nF. Required response\nSchema.")
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
    assert len(result["rounds"]) == 4
    assert result["rounds"][0]["fit"]["failure_class"] == "revision_contract"
    assert result["rounds"][0]["candidate_sha256"] == (
        result["rounds"][0]["parent_candidate_sha256"]
    )
    assert calls[:2] == ["function_revision", "function_revision"]
    assert result["rounds"][1]["numerically_stable"] is True


def test_summary_counts_initializer_and_refinement_from_compact_rounds(
    tmp_path: Path,
) -> None:
    task_id = "hard_seed0_multiround"
    plan = {
        "config": {
            "protocol": "scientific-staged-multiround-feedback-2",
            "round_count": 2,
        },
        "tasks": [
            {
                "task_id": task_id,
                "benchmark_id": "hard",
                "seed": 0,
            }
        ],
    }
    terminal = tmp_path / task_id / "terminal.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_text(
        json.dumps(
            {
                "result": {
                    "status": "complete",
                    "rounds": [
                        {
                            "round_index": 1,
                            "route": "function_revision",
                            "selected_components": ["f", "v01"],
                            "candidate_sha256": "a" * 64,
                            "revision": {"attempts": [], "audit": {}},
                            "fit": {
                                "status": "fit_failed",
                                "failure_class": "fitter_contract",
                                "initializer": {"success": True},
                                "refinement": {"optimizer_success": False},
                            },
                            "numerically_stable": False,
                            "training_normalized_mse": None,
                            "validation_normalized_mse": None,
                        }
                    ],
                    "physical_requests": 1,
                    "observed_total_tokens": 100,
                    "provider_seconds": 1.5,
                }
            }
        )
    )

    summary = campaign.summarize(plan, tmp_path)

    assert summary["status"] == "complete"
    assert summary["completed_rounds"] == 1
    assert summary["collocation_initializer_success_rate"] == 1.0
    assert summary["forward_sensitivity_optimizer_success_rate"] == 0.0
    assert summary["forward_sensitivity_optimizer_native_success_rate"] == 0.0
    assert summary["rounds_with_finite_residual_evaluations"] == 0
    assert summary["fitter_contract_failure_round_count"] == 1


def test_fitter_contract_failure_cannot_route_scientific_revision() -> None:
    rounds = [
        {
            "fit": {"failure_class": "fitter_contract"},
            "numerically_stable": False,
        }
    ]
    with pytest.raises(ValueError, match="cannot route scientific revision"):
        _route(2, rounds)


def test_v3_config_freezes_analytic_initializer_contract() -> None:
    config = MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v3.json").read_text()
    )

    assert config.protocol == "scientific-staged-multiround-feedback-3"
    assert campaign._artifact_schema(config.protocol, "summary").endswith("-3")
