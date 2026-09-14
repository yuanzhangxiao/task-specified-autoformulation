"""Initial-only boundary construction, held-out transfer, and bounded resume."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.staged_function_campaign import function_diagnostic
from autoformalism.rebuttal.staged_function_prefit_campaign import (
    deterministic_prefit_audit,
    freeze_function_prefit_campaign,
)
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.causal_initialization import (
    InitializerChoice,
    compile_initialization_result,
    construct_initializers,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from tests.test_staged_function_prefit_campaign import _write_source
from tests.test_staged_function_runner import response

MAP = {
    "initial": {
        "mode": "causal_map",
        "expression": "a+b*v01",
        "parameters": [
            {"name": "a", "role": "coefficient"},
            {"name": "b", "role": "coefficient"},
        ],
    }
}


def setup(tmp_path, replies=None):
    problem, _ = synthetic_problem("causal_map", 0.0, 0)
    requests = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        requests.append(payload)
        return response(
            (replies or [MAP])[min(len(requests) - 1, len(replies or [MAP]) - 1)]
        )

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=3),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="initials",
        seed=0,
        transport=transport,
    )
    candidate = CandidateModel.model_validate(problem["candidate"])
    context = ValidationContext.model_validate(problem["context"])
    return candidate, context, client, requests, problem


def test_distinct_initials_transfer_and_future_targets_or_ids_cannot_change_them(
    tmp_path,
):
    candidate, context, client, requests, problem = setup(tmp_path)
    result = construct_initializers(candidate, context, {}, client, tmp_path / "result")
    model = compile_initialization_result(candidate, context, result)
    val = unpack_split(problem["splits"]["val"])
    parameters = {"rate": 0.3, "gain": 0.8, "init_m_a": 1.0, "init_m_b": 2.0}
    values = [
        trajectory_initial_state(model, row, {}, parameters=parameters)
        for row in val.trajectories
    ]
    assert values[0][0] != values[1][0]
    np.testing.assert_allclose(values[0], [4.0, 1.5])
    row = val.trajectories[0]
    altered = row.targets["v01"].copy()
    altered[1:] = 1e9
    changed = replace(row, trajectory_id="unseen_id", targets={"v01": altered})
    np.testing.assert_array_equal(
        values[0],
        trajectory_initial_state(model, changed, {}, parameters=parameters),
    )
    assert result["audit"]["validation_initials_fitted"] is False
    assert set(result["plan"]["rules"]) == {"m"}
    assert requests[0]["allowed_initial_symbols"] == ["t", "u01", "v01"]
    assert (
        construct_initializers(candidate, context, {}, client, tmp_path / "result")
        == result
    )
    assert len(requests) == 1


@pytest.mark.parametrize("bad", ["v01[1]", "m", "future", "log(-1)"])
def test_invalid_map_is_repaired_with_diagnostic_then_resumes_without_calls(
    tmp_path, bad
):
    rejected = {"initial": {"mode": "causal_map", "expression": bad, "parameters": []}}
    candidate, context, client, requests, _ = setup(tmp_path, [rejected, MAP])
    result = construct_initializers(candidate, context, {}, client, tmp_path / "result")
    assert [a["accepted"] for a in result["attempts"]] == [False, True]
    assert requests[1]["diagnostics"][0]["error"]
    assert (
        construct_initializers(candidate, context, {}, client, tmp_path / "result")
        == result
    )
    assert len(requests) == 2


def test_exhaustion_does_not_reset_attempt_budget(tmp_path):
    bad = {"initial": {"mode": "causal_map", "expression": "m", "parameters": []}}
    candidate, context, client, requests, _ = setup(tmp_path, [bad])
    for _ in range(2):
        with pytest.raises(ValueError, match="exhausted"):
            construct_initializers(candidate, context, {}, client, tmp_path / "result")
    assert len(requests) == 3


def test_canonical_artifact_and_checkpoint_tampering_are_rejected(tmp_path):
    candidate, context, client, _, _ = setup(tmp_path)
    result = construct_initializers(candidate, context, {}, client, tmp_path / "result")
    broken = copy.deepcopy(result)
    broken["candidate"]["initial_conditions"][0]["expression"] = "99"
    with pytest.raises(ValueError, match="deterministic lowering"):
        compile_initialization_result(candidate, context, broken)
    path = tmp_path / "result/state.json"
    data = json.loads(path.read_text())
    data["rules"].clear()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity or digest"):
        construct_initializers(candidate, context, {}, client, tmp_path / "result")


def test_no_numeric_optimizer_fields_in_provider_schema():
    schema = json.dumps(InitializerChoice.model_json_schema())
    assert '"guess"' not in schema and '"fixed_value"' not in schema
    with pytest.raises(ValueError):
        InitializerChoice.model_validate(
            {"initial": {"mode": "shared_value", "guess": 5}}
        )


def test_future_inputs_do_not_change_initials_but_initial_inputs_can(tmp_path):
    reply = copy.deepcopy(MAP)
    reply["initial"]["expression"] = "a+b*u01"
    candidate, context, client, _, problem = setup(tmp_path, [reply])
    artifact = construct_initializers(
        candidate, context, {}, client, tmp_path / "result"
    )
    model = compile_initialization_result(candidate, context, artifact)
    row = unpack_split(problem["splits"]["val"]).trajectories[0]
    params = {"rate": 0.3, "gain": 0.8, "init_m_a": 1.0, "init_m_b": 2.0}
    inputs = row.external_inputs["u01"].copy()
    inputs[1:] = 1e6
    changed = replace(row, external_inputs={"u01": inputs})
    np.testing.assert_array_equal(
        trajectory_initial_state(model, row, {}, parameters=params),
        trajectory_initial_state(model, changed, {}, parameters=params),
    )
    inputs = inputs.copy()
    inputs[0] = 3.0
    changed = replace(row, external_inputs={"u01": inputs})
    assert trajectory_initial_state(model, changed, {}, parameters=params)[0] == 7.0


def test_new_handoff_policy_is_frozen_and_cannot_overwrite_legacy(tmp_path):
    config_path, source = _write_source(tmp_path)
    config = json.loads(config_path.read_text())
    config.update(
        protocol="scientific-staged-function-prefit-handoff-2",
        initialization_policy="causal_training",
    )
    config_path.write_text(json.dumps(config))
    plan = freeze_function_prefit_campaign(config_path, source, tmp_path / "plan.json")
    assert all(t["initialization_policy"] == "causal_training" for t in plan["tasks"])
    assert plan["schema_version"].endswith("plan-2")
    config.update(
        protocol="scientific-staged-function-prefit-handoff-1",
        initialization_policy="legacy",
    )
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="campaign differs"):
        freeze_function_prefit_campaign(config_path, source, tmp_path / "plan.json")


def test_new_stage_replaces_legacy_initial_call_and_preserves_function_topology(
    tmp_path,
):
    task = function_diagnostic("driven_memory", ModelingLimits())
    requests = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        requests.append(payload)
        if "selected_state" in payload:
            assert payload["protocol"] == "causal-initializer-construction-1"
            return response(
                {
                    "initial": {
                        "mode": "causal_map",
                        "expression": "a+b*x",
                        "parameters": MAP["initial"]["parameters"],
                    }
                }
            )
        selected = payload["selected_term"]
        return response(
            {"expression": "z-x" if selected["lhs"] == "x" else "u-z", "parameters": []}
        )

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="constructed",
        seed=0,
        transport=transport,
    )
    args = (
        PublicScientificBrief.model_validate(task["brief"]),
        ValidationContext.model_validate(task["context"]),
        task["source"],
        client,
        tmp_path,
    )
    result = run_staged_functions(*args, initialization_policy="causal_training")
    assert result["complete_model"], result["error"]
    assert len(requests) == 3
    assert result["candidate"] == result["initialization"]["candidate"]
    assert result["initialization"]["audit"]["validation_initials_fitted"] is False
    with pytest.raises(ValueError, match="frozen public context"):
        deterministic_prefit_audit(args[0], task["source"], result)
    audit = deterministic_prefit_audit(args[0], task["source"], result, context=args[1])
    assert audit["checks"]["latent_initializer_coverage"]
    assert (
        run_staged_functions(*args, initialization_policy="causal_training") == result
    )
    assert len(requests) == 3
    with pytest.raises(ValueError, match="construction contract differs"):
        run_staged_functions(*args)
