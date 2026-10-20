"""Construction handoffs preserve fixed auxiliary initials without fitting."""

from __future__ import annotations

import builtins
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting.simulation import trajectory_initial_state
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.research.interaction_fit_handoff import build_handoff, verify_handoff
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.search.construction_graph_formulas import (
    compile_graph_formula_result,
    construct_graph_formulas,
)
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.staged_topology import content_hash
from tests.test_staged_function_runner import response


def construction(tmp_path, *, future_auxiliary=False, nonsmooth=False, root=True):
    """Construct one root/map plus a runtime-owned aliased auxiliary boundary."""
    rhs = "z + af_internal_aux_0"
    if future_auxiliary:
        rhs += " + EGP + Uii"
    if nonsmooth:
        rhs += " + abs(z)"
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "auxiliary_initial_boundary",
            "parent_candidate_id": None,
            "states": [
                {"name": "y", "kind": "observed"},
                {"name": "af_internal_aux_0", "kind": "latent"},
                {"name": "z", "kind": "latent"},
            ],
            "state_equations": [
                {"state": "y", "rhs": rhs},
                {"state": "af_internal_aux_0", "rhs": "-af_internal_aux_0"},
                {"state": "z", "rhs": "-rate*z"},
            ],
            "parameters": [{"name": "rate", "scope": "global", "role": "rate"}],
            "observation_mappings": [{"channel": "y", "expression": "y"}],
            "initial_conditions": [
                {"state": state, "scope": "global", "fixed_value": 0}
                for state in ("y", "af_internal_aux_0", "z")
            ],
        }
    )
    context = ValidationContext(
        targets=("y",), auxiliaries=("aux", "EGP", "Uii"), external_inputs=("u",)
    )
    draft = Draft.model_validate(
        {
            "variables": [
                {
                    "name": state,
                    "definition": "differential",
                    "scientific_role": "public state" if state != "z" else "memory",
                }
                for state in ("y", "aux", "z")
            ]
        }
    )
    contract = build_contract(
        draft,
        context,
        scientific_context="A generated auxiliary retains its public initial reading.",
        permitted_public_initials=("y", "aux", "EGP", "Uii", "u", "t"),
        observed_initial_channels={"y": "y", "aux": "aux"},
        trajectory_initialization="observation_prefix",
        provenance="Sealed runtime observed_initial identities.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z",
                    "rule": {"mode": "value", "scope": "trajectory"}
                    if root
                    else {
                        "mode": "map",
                        "state_initials": ["aux"],
                        "public_initials": ["y"],
                    },
                    "justification": "An independent or dependent latent preparation.",
                }
            ]
        }
    )
    requests = []

    def transport(url, body, timeout):
        requests.append(json.loads(body["messages"][1]["content"].split("\n", 1)[1]))
        return response(
            {"role": "coefficient"}
            if root
            else {
                "expression": "gain*state_1 + public_1",
                "parameters": [{"name": "gain", "role": "coefficient"}],
            }
        )

    client = StagedTopologyClient(
        settings=StagedModelSettings(attempts_per_step=1),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="auxiliary-identity",
        seed=0,
        transport=transport,
    )
    return candidate, context, contract, handoff, client, requests


def construct(parts, tmp_path):
    """Use the real formula constructor, retaining its sealed numerical problem."""
    candidate, context, contract, handoff, client, _ = parts
    initial = construct_graph_formulas(
        candidate,
        context,
        {},
        contract,
        handoff,
        client,
        tmp_path / "formulas",
        state_aliases={"aux": "af_internal_aux_0"},
    )
    return {
        "protocol": "scientific-staged-functions-3",
        "status": "complete",
        "complete_model": True,
        "initialization_policy": "initialization_graph",
        "fitting_backend": "prefix-fit-problem-1",
        "candidate": None,
        "initialization": initial,
        "initialization_graph_contract": contract.model_dump(mode="json"),
        "initialization_graph_handoff": handoff.model_dump(mode="json"),
        "prefix_fit_problem": initial["prefix_fit_problem"],
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


@pytest.mark.parametrize("root", [True, False])
def test_runtime_auxiliary_identity_is_exact_without_extra_coordinates(tmp_path, root):
    parts = construction(tmp_path, root=root)
    result = construct(parts, tmp_path)
    initial = result["initialization"]
    rule = initial["graph"]["rules"]["af_internal_aux_0"]["initial"]
    assert rule == {
        "mode": "map",
        "expression": "public_1",
        "parameters": [],
        "state_initials": {},
        "public_initials": {"public_1": "aux"},
    }
    assert len(parts[-1]) == 1
    assert parts[-1][0]["selected_initializer_dependency"]["state"] == "z"
    assert all("af_internal_aux_0" not in name for name in initial["parameter_guesses"])
    assert initial["candidate"]["state_equations"] == parts[0].model_dump(mode="json")[
        "state_equations"
    ]
    assert initial["candidate"]["observation_mappings"] == parts[0].model_dump(
        mode="json"
    )["observation_mappings"]
    assert {
        mapping["channel"] for mapping in initial["candidate"]["observation_mappings"]
    } == set(parts[1].targets)
    assert initial["runtime_observed_initials"][0]["scope"] == "recorded_t0_only"
    assert initial["runtime_observed_initials"][0]["independent_parameter"] is False
    model = compile_graph_formula_result(parts[0], parts[1], initial)
    row = Trajectory(
        "row",
        np.array([0.0, 1.0]),
        {"y": np.array([2.0, 3.0])},
        {"aux": np.array([7.0, 9.0])},
        {"u": np.array([0.0, 0.0])},
        {},
        {},
    )
    first = trajectory_initial_state(
        model, row, {}, parameters=initial["parameter_guesses"]
    )
    poisoned = replace(
        row,
        targets={"y": np.array([2.0, -1e12])},
        auxiliaries={"aux": np.array([7.0, 1e12])},
    )
    other = trajectory_initial_state(
        model, poisoned, {}, parameters=initial["parameter_guesses"]
    )
    assert first[1] == 7.0
    np.testing.assert_array_equal(first, other)


@pytest.mark.parametrize("fault", ["different_channel", "missing_target", "no_alias"])
def test_runtime_identity_rejects_unattested_coordinate_changes(tmp_path, fault):
    parts = list(construction(tmp_path))
    if fault == "different_channel":
        payload = parts[2].model_dump(mode="json")
        payload["observed_initial_channels"]["aux"] = "Uii"
        parts[2] = type(parts[2]).model_validate(payload)
    elif fault == "missing_target":
        payload = parts[2].model_dump(mode="json")
        del payload["observed_initial_channels"]["y"]
        parts[2] = type(parts[2]).model_validate(payload)
    if fault == "no_alias":
        with pytest.raises(ValueError):
            construct_graph_formulas(
                *parts[:2], {}, *parts[2:5], tmp_path / "formulas"
            )
    else:
        with pytest.raises(ValueError):
            construct(parts, tmp_path)
    assert parts[-1] == []


def test_portable_handoff_compiles_scoped_problem_without_numerical_work(
    tmp_path, monkeypatch
):
    from autoformalism.fitting import prefix_initialization as numerical

    result = construct(construction(tmp_path), tmp_path)

    def prohibited(*args, **kwargs):
        raise AssertionError("fitting, inference, rollout and forecast are deferred")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(numerical, name, prohibited)
    provenance = {"source_model": "fixed-model", "compatibility_status": "deferred"}
    receipt = build_handoff(result, provenance=provenance)
    assert receipt["readiness"]["construction"]["status"] == "verified"
    assert receipt["readiness"]["numerical_compile"]["status"] == "passed"
    assert receipt["readiness"]["forecast"]["status"] == "policy_unresolved"
    assert receipt["readiness"]["forecast"]["future_auxiliary_channels"] == []
    assert receipt["parameter_scopes"]["trajectory_roots"] == {"z": "init_z_value"}
    assert receipt["legacy_shared_compatible"] is False
    assert receipt["legacy_fitter_dispatch_allowed"] is False
    assert receipt["prefix_policy"] is None
    assert receipt["window_policy_unresolved"] is True
    assert receipt["parameter_fitting_performed"] is False
    assert receipt["numerical_rollout_performed"] is False
    assert receipt["provenance"] == provenance
    portable = json.loads(json.dumps(receipt))
    assert PrefixFitProblem.model_validate(portable["prefix_fit_problem"]).graph.rules
    seal = portable.pop("artifact_sha256")
    assert seal == content_hash(portable)
    assert build_handoff(result, provenance=provenance) == receipt


def test_forecast_auxiliary_restriction_does_not_discard_training_problem(tmp_path):
    result = construct(construction(tmp_path, future_auxiliary=True), tmp_path)
    receipt = build_handoff(result, provenance={})
    assert receipt["readiness"]["numerical_compile"]["status"] == "passed"
    forecast = receipt["readiness"]["forecast"]
    assert forecast["status"] == "blocked"
    assert forecast["future_auxiliary_channels"] == ["EGP", "Uii"]
    assert {item["code"] for item in forecast["blockers"]} == {
        "future_observed_forcing_unavailable",
        "prefix_policy_unresolved",
    }
    assert receipt["prefix_fit_problem"] == result["prefix_fit_problem"]


def test_nonsmooth_functions_are_structural_exports_but_numerically_unsupported(
    tmp_path,
):
    result = construct(construction(tmp_path, nonsmooth=True), tmp_path)
    receipt = build_handoff(result, provenance={})
    assert receipt["readiness"]["construction"]["status"] == "verified"
    assert receipt["readiness"]["numerical_compile"]["status"] == "unsupported"
    assert "nonsmooth" in receipt["readiness"]["numerical_compile"]["reason"]
    assert receipt["readiness"]["forecast"]["status"] == "blocked"
    assert receipt["prefix_fit_problem"] == result["prefix_fit_problem"]


def test_missing_optional_numerical_backend_preserves_verified_problem(
    tmp_path, monkeypatch
):
    result = construct(construction(tmp_path), tmp_path)
    original = builtins.__import__

    def without_backend(name, *args, **kwargs):
        if name == "autoformalism.fitting.prefix_initialization":
            raise ModuleNotFoundError("optional CasADi backend unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_backend)
    receipt = build_handoff(result, provenance={})
    assert receipt["readiness"]["construction"]["status"] == "verified"
    assert receipt["readiness"]["numerical_compile"]["status"] == "unavailable"
    assert receipt["readiness"]["forecast"]["status"] == "blocked"
    assert receipt["prefix_fit_problem"] == result["prefix_fit_problem"]


@pytest.mark.parametrize("compile_status", ["passed", "unavailable"])
def test_portable_receipt_replays_without_compiler_or_environment_change(
    tmp_path, monkeypatch, compile_status
):
    from autoformalism.research import interaction_fit_handoff as adapter

    result = construct(construction(tmp_path), tmp_path)
    if compile_status == "unavailable":
        monkeypatch.setattr(
            adapter,
            "_numerical_readiness",
            lambda problem: {
                "status": "unavailable",
                "method": adapter.COMPILE_METHOD,
                "reason": "Optional CasADi unavailable on construction host",
            },
        )
    receipt = build_handoff(result, provenance={"case": "frozen"})

    def prohibited(*args, **kwargs):
        raise AssertionError("report replay must not rerun numerical compilation")

    monkeypatch.setattr(adapter, "_numerical_readiness", prohibited)
    assert verify_handoff(result, receipt, provenance={"case": "frozen"}) == receipt
    altered = copy.deepcopy(receipt)
    altered["parameter_scopes"]["trajectory_roots"] = {}
    altered.pop("artifact_sha256")
    altered["artifact_sha256"] = content_hash(altered)
    with pytest.raises(ValueError, match="deterministic source replay"):
        verify_handoff(result, altered, provenance={"case": "frozen"})
    with pytest.raises(ValueError, match="deterministic source replay"):
        verify_handoff(result, receipt, provenance={"case": "other"})


@pytest.mark.parametrize(
    "field", ["problem", "identity", "scope", "candidate", "status"]
)
def test_changed_or_resealed_results_fail_closed(tmp_path, field):
    result = copy.deepcopy(construct(construction(tmp_path), tmp_path))
    initial = result["initialization"]
    if field == "problem":
        result["prefix_fit_problem"] = copy.deepcopy(result["prefix_fit_problem"])
        result["prefix_fit_problem"]["parameter_guesses"]["rate"] = 9.0
    elif field == "identity":
        initial["runtime_observed_initials"][0]["channel"] = "Uii"
    elif field == "scope":
        initial["graph"]["rules"]["z"]["initial"]["scope"] = "shared_training"
    elif field == "candidate":
        result["candidate"] = initial["candidate"]
    else:
        result["status"] = "failed"
    initial.pop("result_sha256")
    initial["result_sha256"] = content_hash(initial)
    with pytest.raises(ValueError):
        build_handoff(result, provenance={})
