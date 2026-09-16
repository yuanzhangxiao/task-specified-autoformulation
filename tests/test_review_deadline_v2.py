"""Content-derived routing, genuine three-call budgets and structural warm starts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.initialization import apply_initialization_plan
from autoformalism.llm.review_revision import RevisionClient
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.repair_comparison import RepairBudgetExceeded
from autoformalism.rebuttal.repair_transactions import default_initialization
from autoformalism.schemas.staged_topology import ModelingLimits
from autoformalism.search import review_model_edits as edits
from autoformalism.search.residual_evidence import build_residual_evidence
from autoformalism.staged_topology import content_hash
from tests.test_repair_comparison import CONTEXT, candidate


def example():
    base = candidate()
    initial = default_initialization(base, CONTEXT)
    model, guesses, audit = apply_initialization_plan(
        compile_candidate(base, CONTEXT), initial
    )
    values = {"rate": 0.3, "gain": 0.8, "decay": 0.5, "init_m_value": 0.7}
    bundle = {
        "source_task": {"task_id": "synthetic"},
        "brief": {
            "scientific_context": "Generate v01 from u01 with causal memory.",
            "limits": ModelingLimits().model_dump(mode="json"),
        },
        "context": CONTEXT.model_dump(mode="json"),
        "candidate": model.validated.candidate.model_dump(mode="json"),
        "initialization": {
            "base_candidate": base.model_dump(mode="json"),
            "base_context": CONTEXT.model_dump(mode="json"),
            "plan": initial.model_dump(mode="json"),
            "candidate": model.validated.candidate.model_dump(mode="json"),
            "context": model.validated.context.model_dump(mode="json"),
            "guesses": guesses,
            "audit": audit,
        },
    }
    row = Trajectory(
        "train",
        np.arange(6.0),
        {"v01": np.arange(6.0)},
        {},
        {"u01": np.ones(6)},
        {},
        {},
    )
    train = DatasetSplit(SplitName.TRAIN, (row,), "synthetic-train")
    packet = build_residual_evidence(
        train,
        CONTEXT,
        {
            "train": {
                "time": row.time.tolist(),
                "predictions": {"v01": (row.time + 0.1).tolist()},
            }
        },
        candidate_sha256=content_hash(bundle["candidate"]),
        parameters=values,
        numerical_status={"budget_exhausted": True},
    ).model_dump(mode="json")
    reply = {
        "hypothesis": "A bounded response could address the measured mismatch.",
        "evidence_ids": [packet["rows"][0]["evidence_id"]],
        "equations": [{"component": "f", "expression": "tanh(m)"}],
    }
    return bundle, packet, values, public.pack_split(train), reply


def new_state_reply(reply):
    return {
        **reply,
        "equations": [
            {
                "component": "z",
                "kind": "dynamic",
                "expression": "-newrate*z+u01",
                "parameters": [{"name": "newrate", "role": "rate"}],
            },
            {"component": "m", "expression": "-rate*m+gain*z"},
        ],
        "initializers": [
            {
                "state": "z",
                "causal_map": {
                    "expression": "a+b*v01",
                    "parameters": [
                        {"name": "a", "role": "coefficient"},
                        {"name": "b", "role": "coefficient"},
                    ],
                },
            }
        ],
    }


def test_schema_has_no_controller_enum_or_initializer_guess():
    schema = edits.ModelEdits.model_json_schema()
    assert not {"action", "scope", "route"} & schema["properties"].keys()
    assert "guess" not in schema["$defs"]["FunctionParameter"]["properties"]
    _, _, _, _, raw = example()
    edits.ModelEdits.model_validate({**raw, "hypothesis": "a" * 3000})
    with pytest.raises(ValueError):
        edits.ModelEdits.model_validate({**raw, "action": "revise_sources"})


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("tanh(m)", ["functions"]),
        ("tanh(m)*u01", ["topology", "functions"]),
        ("sigmoid(m)+u01", ["topology", "functions"]),
    ],
)
def test_runtime_infers_function_source_and_pathway_routes(expression, expected):
    bundle, packet, _, _, raw = example()
    before = copy.deepcopy(bundle)
    raw["equations"][0]["expression"] = expression
    result = edits.apply_edits(bundle, packet, raw)
    assert result["outcome"] == "committed"
    assert result["provenance"]["routes"] == expected
    assert (
        result["bundle"]["initialization"]["base_candidate"]["state_equations"]
        == before["initialization"]["base_candidate"]["state_equations"]
    )
    assert bundle == before
    assert "slots" not in result["bundle"]


def test_empty_content_is_keep_and_same_content_is_noop():
    bundle, packet, _, _, raw = example()
    assert (
        edits.apply_edits(bundle, packet, {**raw, "equations": []})["outcome"]
        == "no_change"
    )
    raw["equations"][0]["expression"] = "sigmoid(m)"
    assert edits.apply_edits(bundle, packet, raw)["outcome"] == "no_change"


@pytest.mark.parametrize("expression", ["missing+u01", "__import__('os')", "v01+u01"])
def test_undefined_symbols_unsafe_code_and_target_leakage_still_fail(expression):
    bundle, packet, _, _, raw = example()
    raw["equations"][0]["expression"] = expression
    with pytest.raises((ValueError, ModelValidationError)):
        edits.apply_edits(bundle, packet, raw)


def test_new_state_requires_initializer_then_retains_compatible_warm_start():
    bundle, packet, values, train, raw = example()
    raw = new_state_reply(raw)
    with pytest.raises(ValueError, match="initializer"):
        edits.apply_edits(bundle, packet, {**raw, "initializers": []})
    result = edits.apply_edits(bundle, packet, raw)
    assert result["provenance"]["routes"] == [
        "state_inventory",
        "topology",
        "functions",
        "initialization",
    ]
    plan = {"config": {"fit_profile": "collocation-single-target-v2"}}
    task = {"seed": 0, "task_id": "synthetic"}
    parent = pipeline.request_for(bundle, plan, task, 0)
    child = pipeline.request_for(result["bundle"], plan, task, 1)
    with pytest.raises(ValueError, match="initialization plan"):
        sibling_fit.compatible_seed(parent, child, values, train)
    seed = sibling_fit.compatible_seed(
        parent, child, values, train, allow_initialization_changes=True
    )
    assert all(seed["parameters"][k] == v for k, v in values.items())
    assert seed["fresh_parameters"] == ["init_z_a", "init_z_b", "newrate"]
    assert seed["retained_initializer_parameters"] == ["init_m_value"]


def test_changed_initializer_does_not_reuse_same_name_different_meaning():
    bundle, packet, values, train, raw = example()
    raw.update(
        equations=[],
        initializers=[
            {
                "state": "m",
                "causal_map": {
                    "expression": "value*v01",
                    "parameters": [{"name": "value", "role": "coefficient"}],
                },
            }
        ],
    )
    result = edits.apply_edits(bundle, packet, raw)
    plan = {"config": {"fit_profile": "collocation-single-target-v2"}}
    task = {"seed": 0, "task_id": "synthetic"}
    parent, child = (
        pipeline.request_for(b, plan, task, i)
        for i, b in enumerate((bundle, result["bundle"]))
    )
    seed = sibling_fit.compatible_seed(
        parent, child, values, train, allow_initialization_changes=True
    )
    assert seed["parameters"]["init_m_value"] == 0.1
    assert seed["reset_initializer_parameters"] == ["init_m_value"]
    assert seed["parameters"]["gain"] == values["gain"]


def client(tmp_path, transport, **kwargs):
    return RevisionClient(
        settings=StagedModelSettings(maximum_total_tokens=256),
        base_url="http://unused",
        directory=tmp_path,
        namespace="test",
        seed=0,
        transport=transport,
        **kwargs,
    )


def test_three_large_requests_run_without_cumulative_cap_and_resume(tmp_path):
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        assert body["max_tokens"] == 8192 and timeout == 180
        return {"choices": [], "usage": {"total_tokens": 100_000}}

    def call(c, attempt):
        return c.call(
            system="public",
            user="x" * 100_000,
            response_model=edits.ModelEdits,
            step="revision",
            attempt=attempt,
        )

    one = client(tmp_path, transport)
    for i in range(3):
        call(one, i)
    assert len(calls) == 3
    assert sum(r["observed_total_tokens"] for r in one.records) == 300_000
    two = client(tmp_path, lambda *a: pytest.fail("duplicate call"))
    for i in range(3):
        call(two, i)
    assert len(two.records) == 3
    with pytest.raises(RepairBudgetExceeded):
        call(two, 3)


def test_uncertain_call_is_consumed_and_unknown_usage_not_bytes(tmp_path):
    one = client(tmp_path, lambda *a: (_ for _ in ()).throw(TimeoutError("timeout")))
    kwargs = {
        "system": "x",
        "user": "x",
        "response_model": edits.ModelEdits,
        "step": "revision",
        "attempt": 0,
    }
    result = one.call(**kwargs)
    assert result["observed_total_tokens"] is None and result["budget_charge"] == 0
    assert result["request_bytes"] > 0
    path = tmp_path / (result["request_hash"] + ".json")
    result["status"] = "inflight"
    path.write_text(json.dumps(result))
    two = client(tmp_path, lambda *a: pytest.fail("uncertain call resent"))
    assert two.call(**kwargs)["status"] == "uncertain"
    with pytest.raises(ValueError, match="another request"):
        two.call(**{**kwargs, "user": "changed"})


def test_draining_worker_does_not_consume_a_call(tmp_path):
    one = client(tmp_path, lambda *a: pytest.fail("late call"), can_start=lambda: False)
    with pytest.raises(DeferredCall):
        one.call(
            system="x", user="x", response_model=edits.ModelEdits, step="r", attempt=0
        )
    assert not list(tmp_path.glob("*.json"))


def test_v1_config_serialization_and_round_zero_client_are_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    legacy = io.DeadlineConfig.model_validate_json(
        (root / "configs/review_deadline_v1.json").read_text()
    )
    newer = io.DeadlineConfig.model_validate_json(
        (root / "configs/review_deadline_v2.json").read_text()
    )
    assert {
        **legacy.model_dump(mode="json"),
        "protocol": io.CONTENT_PROTOCOL,
    } == newer.model_dump(mode="json")
    task = io.tasks(newer)[0]
    plan = {
        "artifact_sha256": "x",
        "config": newer.model_dump(mode="json"),
        "protocol": io.CONTENT_PROTOCOL,
    }
    assert not isinstance(
        pipeline._client(tmp_path, plan, task, 0, "http://unused", lambda: True),
        RevisionClient,
    )
    assert isinstance(
        pipeline._client(tmp_path, plan, task, 1, "http://unused", lambda: True),
        RevisionClient,
    )


def test_revision_retries_delivery_and_content_errors_then_preserves_result(tmp_path):
    bundle, packet, values, _, raw = example()
    task = {"task_id": "synthetic", "seed": 0, "arm": "full", "cell": "cell"}
    plan = {
        "protocol": io.CONTENT_PROTOCOL,
        "cells": {"cell": {}},
        "config": {"model_settings": StagedModelSettings().model_dump(mode="json")},
    }
    parent = {
        "selected": {"bundle": bundle, "packet": packet, "fit": {"parameters": values}}
    }
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        reply = {**raw, "equations": [{"component": "f", "expression": "undefined"}]}
        if len(calls) == 3:
            reply = {**raw, "equations": []}
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100_000},
        }

    one = client(tmp_path, transport)
    result = pipeline._content_revision(plan, task, parent, one)
    assert result["status"] == "no_change" and len(calls) == 3
    again = client(tmp_path, lambda *a: pytest.fail("repeated call"))
    assert pipeline._content_revision(plan, task, parent, again) == result


def test_no_spec_does_not_enforce_withheld_input_path():
    bundle, packet, _, _, raw = example()
    raw["equations"] = [{"component": "m", "expression": "-rate*m"}]
    with pytest.raises(ValueError, match="input pathway"):
        edits.apply_edits(bundle, packet, raw)
    bundle["source_task"]["arm"] = "no_spec"
    assert edits.apply_edits(bundle, packet, raw)["outcome"] == "committed"


def test_certificate_feedback_hides_withheld_requirements():
    certificate = {
        "targets": {
            "predicates": [
                {"predicate": "generated_model_path", "status": "failed"},
                {"predicate": "required_dependency:hidden", "status": "failed"},
            ]
        },
        "mechanisms": {"mechanism_results": [{"status": "failed", "id": "hidden"}]},
        "ablation_constraint_pass": True,
    }
    feedback = pipeline._certificate_feedback(certificate, {"arm": "no_spec"})
    assert "hidden" not in json.dumps(feedback)
    assert feedback["failed_target_predicates"]


def test_payload_binds_training_parameters():
    bundle, packet, values, _, _ = example()
    assert edits.payload(bundle, packet, values)
    with pytest.raises(ValueError, match="parameters differ"):
        edits.payload(bundle, packet, {**values, "gain": 1.0})
