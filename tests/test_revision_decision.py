"""Regression cases from public multiround action/feedback failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.rebuttal import staged_multiround_feedback_campaign as campaign
from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from autoformalism.rebuttal.revision_actions import request_revision_action
from autoformalism.rebuttal.revision_decision import (
    decision_snapshot,
    expressions,
    interaction_contract,
    interaction_diff,
    nonlinear_target_paths,
    parameter_aliases,
    translate_names,
)


class Client:
    """Recorded responses only; no network or numerical fitting."""

    def __init__(self, responses):
        self.responses = responses
        self.settings = SimpleNamespace(attempts_per_step=len(responses))
        self.users = []

    def call(self, **kwargs):
        self.users.append(json.loads(kwargs["user"]))
        return {
            "request_hash": f"call_{len(self.users)}",
            "status": "responded",
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(self.responses[len(self.users) - 1])
                        },
                    }
                ]
            },
        }


def test_training_context_reports_zero_input_without_assuming_equilibrium():
    import numpy as np

    from autoformalism.data.models import SplitName

    changing = SimpleNamespace(
        trajectory_id="train_changing",
        targets={"v01": np.array([0.0, 1.0, 2.0])},
        external_inputs={"u01": np.zeros(3)},
        auxiliaries={},
    )
    constant = SimpleNamespace(
        trajectory_id="train_constant",
        targets={"v01": np.zeros(3)},
        external_inputs={"u01": np.zeros(3)},
        auxiliaries={},
    )

    class TrainingOnly:
        train = SimpleNamespace(name=SplitName.TRAIN, trajectories=(changing, constant))

        @property
        def validation(self):
            raise AssertionError("training context must not read validation")

        @property
        def test(self):
            raise AssertionError("training context must not read test")

    facts = campaign._public_fit_context(TrainingOnly())["v01"]
    values = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
    assert facts["zero_input_with_changing_output"] == ["train_changing"]
    assert facts["observed_rms"] == pytest.approx(np.sqrt(np.mean(values**2)))
    assert facts["zero_prediction_pooled_training_nmse"] == pytest.approx(
        np.mean(values**2) / np.var(values)
    )
    assert "does not require equilibrium" in facts["interpretation"]


def request(tmp_path, client, round_index=1, **kwargs):
    return request_revision_action(
        client=client,
        route="function_revision",
        candidate=recovery_candidate(),
        selected=("f", "v01"),
        context=CONTEXT,
        scientific_context="Public nonlinear feedback",
        numerical_feedback={"validation_normalized_mse": 3.0},
        round_index=round_index,
        failure_checkpoint=tmp_path / f"failures_{round_index}.json",
        transaction_path=tmp_path / "pending.json",
        nonlinear_targets=("v01",),
        **kwargs,
    )


def test_keep_companion_and_round_trip_aliases(tmp_path):
    parent = recovery_candidate()
    aliases = parameter_aliases(parent)
    client = Client(
        [
            {
                "revisions": [
                    {
                        "component": "f",
                        "parameters": [],
                        "expression": translate_names("k*tanh(v01) - f/tau_f", aliases),
                    }
                ],
                "keep_components": ["v01"],
            }
        ]
    )
    revised, record = request(tmp_path, client)
    assert expressions(revised)["v01"] == expressions(parent)["v01"]
    assert expressions(revised)["f"] == "k*tanh(v01) - f/tau_f"
    assert record["kept_components"] == ["v01"]
    assert client.users[0]["selected_pending_components"][0][
        "immutable_signed_interactions"
    ]
    assert client.users[0]["equation_kinds"]["f"] == "ode_rhs"
    assert client.users[0]["equation_kinds"]["v01"] == "algebraic"
    assert set(client.users[0]["available_parent_parameters"]) == set(aliases.values())


def test_partial_transaction_survives_round_exhaustion(tmp_path):
    first = Client(
        [
            {
                "revisions": [
                    {
                        "component": "f",
                        "expression": "k*tanh(v01)-f/tau_f",
                        "parameters": [],
                    },
                    {"component": "v01", "expression": "missing*f", "parameters": []},
                ]
            }
        ]
    )
    with pytest.raises(campaign.RevisionContractError):
        request(tmp_path, first)
    transaction = json.loads((tmp_path / "pending.json").read_text())
    assert transaction["pending"] == ["v01"]
    assert list(transaction["accepted"]) == ["f"]
    second = Client([{"keep_components": ["v01"]}])
    revised, record = request(tmp_path, second, round_index=2)
    assert expressions(revised)["f"] == "k*tanh(v01)-f/tau_f"
    assert [
        item["component"] for item in second.users[0]["selected_pending_components"]
    ] == ["v01"]
    assert second.users[0]["numerical_feedback"]["validation_normalized_mse"] == 3.0
    assert record["transaction_attempt_count"] == 2


def test_all_kept_is_no_change_not_a_duplicate_failure(tmp_path):
    parent = recovery_candidate()
    revised, record = request(tmp_path, Client([{"keep_components": ["f", "v01"]}]))
    assert revised == parent
    assert record["status"] == "no_change"


def test_same_expression_is_keep_without_forcing_a_scientific_edit(tmp_path):
    parent = recovery_candidate()
    client = Client(
        [
            {
                "revisions": [
                    {
                        "component": name,
                        "expression": expressions(parent)[name],
                        "parameters": [],
                    }
                    for name in ("f", "v01")
                ]
            }
        ]
    )
    revised, record = request(tmp_path, client)
    assert revised == parent
    assert record["status"] == "no_change"


def test_exact_interaction_diff_exposes_hidden_term_contract():
    parent = recovery_candidate()
    payload = parent.model_dump(mode="json")
    for item in payload["state_equations"]:
        if item["state"] == "f":
            item["rhs"] = "k*v01**2/(1+v01**2) - f/(tau_f*(1+v01**2))"
    after = type(parent).model_validate(payload)
    diff = interaction_diff(parent, after, "f")
    assert diff["removed_interactions"] == [
        {"outer_sign": -1, "sources": ["f"], "count": 1}
    ]
    assert diff["added_interactions"] == [
        {"outer_sign": -1, "sources": ["f", "v01"], "count": 1}
    ]
    reply = campaign.ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": expressions(after)["f"],
                    "parameters": [],
                }
            ]
        }
    )
    with pytest.raises(campaign.RevisionContractError) as error:
        campaign.apply_component_revision(
            parent,
            reply,
            CONTEXT,
            selected=("f",),
            route="function_revision",
            detailed_contract=True,
        )
    assert error.value.code == "SIGNED_INTERACTION_CONTRACT_CHANGED"
    assert error.value.details["interaction_diffs"] == [diff]


def test_nonlinear_public_path_cannot_disappear(tmp_path):
    client = Client(
        [
            {
                "revisions": [
                    {"component": "f", "expression": "k*v01-f/tau_f", "parameters": []},
                    {
                        "component": "v01",
                        "expression": "k*f+m+k_p*p+k_u*u01+c",
                        "parameters": [],
                    },
                ]
            }
        ]
    )
    with pytest.raises(campaign.RevisionContractError):
        request(tmp_path, client)
    state = json.loads((tmp_path / "pending.json").read_text())
    assert (
        state["diagnostic"]["component_failures"][0]["code"]
        == "PUBLIC_NONLINEAR_OBLIGATION_LOST"
    )
    assert nonlinear_target_paths(recovery_candidate(), ("v01",)) == {"v01": True}


def test_constant_scaled_offset_is_not_internal_shape():
    parent = recovery_candidate()
    rhs = expressions(parent)["v01"].replace(" + c", " + baseline*1")
    reply = campaign.ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "v01",
                    "expression": rhs + " + 0",
                    "parameters": [{"name": "baseline"}],
                }
            ]
        }
    )
    # Role inference independent of whether adding a new interaction is allowed.
    parsed = campaign.RestrictedParser().parse(rhs, location="v01")
    roles, _ = campaign._effective_revision_parameter_roles(
        reply.revisions[0],
        parsed.tree,
        scientific_symbols=set(expressions(parent)) | {"u01"},
        parent_parameters={p.name: p for p in parent.parameters},
    )
    assert roles["baseline"].value == "offset"


def test_best_evaluated_and_numerical_evidence_survive_no_fit_round():
    parent = recovery_candidate()
    rounds = [
        {
            "round_index": 1,
            "candidate_sha256": "a",
            "fit": {"status": "complete"},
            "numerically_stable": True,
            "validation_normalized_mse": 2.0,
        },
        {
            "round_index": 2,
            "candidate_sha256": "b",
            "fit": {"status": "not_run"},
            "numerically_stable": False,
            "validation_normalized_mse": None,
        },
    ]
    state = decision_snapshot(
        parent, rounds, {"evidence_round_index": 1, "parameters": {"k": 0.0}}
    )
    assert state.best_evaluated["candidate_sha256"] == "a"
    assert state.last_numerical_evidence["evidence_round_index"] == 1
    assert len(state.history) == 2


def test_alias_replacement_does_not_change_substrings():
    assert (
        translate_names("k + k_long + tanh(k)", {"k": "par_001"})
        == "par_001 + k_long + tanh(par_001)"
    )
    assert len(interaction_contract(recovery_candidate(), "f")) == 2


def test_completed_transaction_resume_makes_no_new_call(tmp_path):
    client = Client([{"keep_components": ["f", "v01"]}])
    first, record = request(tmp_path, client)
    second, resumed = request(tmp_path, Client([]))
    assert first == second
    assert record == resumed


def test_redeclared_parent_keeps_trusted_domains_and_bounds():
    parent = recovery_candidate()
    payload = parent.model_dump(mode="json")
    for parameter in payload["parameters"]:
        if parameter["name"] == "k":
            parameter["bounds"] = {"lower": 0.2, "upper": 3.0}
    parent = type(parent).model_validate(payload)
    reply = campaign.ComponentRevisionReply.model_validate(
        {
            "revisions": [
                {
                    "component": "f",
                    "expression": "k*tanh(v01)-f/tau_f",
                    "parameters": [{"name": "k", "role": "shape"}],
                }
            ]
        }
    )
    revised, _ = campaign.apply_component_revision(
        parent, reply, CONTEXT, selected=("f",), route="function_revision"
    )
    assert next(p for p in revised.parameters if p.name == "k") == next(
        p for p in parent.parameters if p.name == "k"
    )


def test_pending_transaction_refuses_a_different_parent(tmp_path):
    with pytest.raises(campaign.RevisionContractError):
        request(
            tmp_path,
            Client(
                [
                    {
                        "revisions": [
                            {
                                "component": "f",
                                "expression": "missing",
                                "parameters": [],
                            }
                        ]
                    }
                ]
            ),
        )
    saved = json.loads((tmp_path / "pending.json").read_text())
    saved["contract_sha256"] = "invalid"
    (tmp_path / "pending.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="transaction contract differs"):
        request(tmp_path, Client([]), round_index=2)


def test_full_v6_loop_retains_evidence_best_candidate_and_resumes(
    tmp_path, monkeypatch
):
    import hashlib

    config = campaign.MultiRoundFeedbackConfig.model_validate_json(
        Path("configs/staged_multiround_feedback_v6.json").read_text()
    )
    monkeypatch.setattr(
        campaign,
        "_verified_plan",
        lambda _path: {"config": config.model_dump(mode="json")},
    )
    monkeypatch.setattr(
        campaign,
        "load_public_data",
        lambda *_args: (SimpleNamespace(train=object(), validation=object()), CONTEXT),
    )
    monkeypatch.setattr(campaign, "_public_fit_context", lambda _dataset: {})
    fit_calls = []

    def fit(*_args):
        fit_calls.append(True)
        return {
            "status": "complete",
            "parameters": {"k": 0.01},
            "training": {"normalized_mse": 1.0, "failed_trajectories": []},
            "validation": {"normalized_mse": len(fit_calls), "failed_trajectories": []},
        }

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", fit)
    parent = recovery_candidate()
    root = tmp_path / "run"
    root.mkdir()
    candidate_path = root / "candidate.json"
    candidate_path.write_text(parent.model_dump_json())
    rescue_path = root / "rescue.json"
    rescue_path.write_text("{}")
    benchmark = config.public_cells[0]
    prompt = root / "frozen/public/phase_b_v1" / benchmark / "proposer_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Public nonlinear feedback")
    task = {
        "task_id": "public",
        "benchmark_id": benchmark,
        "seed": 0,
        "tier": "hard",
        "candidate_path": "candidate.json",
        "source_rescue_path": "rescue.json",
        "candidate_file_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "source_rescue_file_sha256": hashlib.sha256(
            rescue_path.read_bytes()
        ).hexdigest(),
    }

    class LoopClient:
        settings = SimpleNamespace(attempts_per_step=2)

        def __init__(self):
            self.records = []
            self.requests = []

        def call(self, **kwargs):
            user = json.loads(kwargs["user"])
            self.requests.append(user)
            index = int(kwargs["step"].split("_")[1])
            pending = [
                item["component"] for item in user["selected_pending_components"]
            ]
            if index == 1:
                response = {
                    "revisions": [
                        {
                            "component": "f",
                            "expression": "k*tanh(v01)-f/tau_f",
                            "parameters": [],
                        }
                    ],
                    "keep_components": ["v01"],
                }
            elif index == 2:
                response = {
                    "revisions": [
                        {"component": "f", "expression": "missing", "parameters": []}
                    ]
                }
                if "v01" in pending:
                    response["revisions"].append(
                        {
                            "component": "v01",
                            "expression": "k*tanh(f)+m+k_p*p+k_u*u01+c",
                            "parameters": [],
                        }
                    )
            elif index == 3:
                assert pending == ["f"]
                assert user["numerical_feedback"]["evidence_round_index"] == 1
                response = {
                    "revisions": [
                        {
                            "component": "f",
                            "expression": "k*sigmoid(v01)-f/tau_f",
                            "parameters": [],
                        }
                    ]
                }
            else:
                response = {"keep_components": pending}
            record = {
                "request_hash": str(len(self.records)),
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
            self.records.append(record)
            return record

    client = LoopClient()
    output = root / "results"
    result = campaign.run_task(root, task, output, client)
    assert len(fit_calls) == 2
    assert result["decision_state"]["best_evaluated"]["round_index"] == 1
    assert (
        result["decision_state"]["last_numerical_evidence"]["evidence_round_index"] == 3
    )
    assert result["rounds"][3]["fit"]["failure_class"] == "no_change"
    requests_before = len(client.records)
    resumed = campaign.run_task(root, task, output, client)
    assert len(client.records) == requests_before
    assert len(fit_calls) == 2
    assert resumed["decision_state"] == result["decision_state"]
