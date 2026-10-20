"""Opt-in consumer gains preserve graph scopes, signed laws and historical output."""

from __future__ import annotations

import copy
import json

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedTopologyClient
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.staged_topology import EquationTerm
from autoformalism.search import shared_process_contract as shared
from autoformalism.search import signed_processes as signed
from autoformalism.search.process_gains import compile_bundle, lower_candidate_gains
from autoformalism.search.scoped_process_gains import (
    HANDOFF_PROTOCOL,
    POLICY,
    build_scoped_gain_handoff,
    lower_saved_function_gains,
    rebind_saved_initializer,
    verify_scoped_gain_handoff,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import content_hash
from tests.test_staged_function_graph_initialization import fixture, lower
from tests.test_staged_function_runner import response


def candidate_pair(*, signed_amplitude=False, shared_amplitude=False):
    """Two consumers share one law; the law itself can change sign."""
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "two_stores",
            "parent_candidate_id": None,
            "states": [
                {"name": "x", "kind": "observed"},
                {"name": "z", "kind": "latent"},
            ],
            "processes": [{"name": "q", "expression": "a*(z-1)"}],
            "state_equations": [
                {"state": "x", "rhs": "-q+a*x" if shared_amplitude else "-q"},
                {"state": "z", "rhs": "q"},
            ],
            "observation_mappings": [{"channel": "x", "expression": "x"}],
            "initial_conditions": [
                {"state": state, "fixed_value": 0, "scope": "global"}
                for state in ("x", "z")
            ],
            "parameters": [
                {
                    "name": "a",
                    "scope": "global",
                    "role": "coefficient"
                    if signed_amplitude
                    else "nonnegative_coefficient",
                }
            ],
        }
    )
    declaration = signed.SignedProcess.model_validate(
        {
            "name": "q",
            "depends_on": ["z"],
            "kind": "transfer",
            "scientific_meaning": "Signed two-store transfer",
            "uses": [
                {"target": "x", "sign": "negative", "conversion": "1"},
                {"target": "z", "sign": "positive", "conversion": "2"},
            ],
        }
    )
    return (
        candidate,
        ValidationContext(targets=("x",)),
        {"protocol": signed.POLICY, "bindings": [signed.binding_for(declaration)]},
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_positive_amplitude_normalization_and_consumer_sharing(unknown):
    candidate, context, contract = candidate_pair()
    before = candidate.model_dump(mode="json"), copy.deepcopy(contract)
    if unknown:
        contract["bindings"][0]["signed_declaration"]["uses"][1]["conversion"] = None
        before = candidate.model_dump(mode="json"), copy.deepcopy(contract)
    corrected, guesses, rows = lower_candidate_gains(
        candidate,
        context,
        contract,
        "declared_or_independent",
        preserve_signed_amplitudes=True,
    )
    assert (candidate.model_dump(mode="json"), contract) == before
    assert "a" not in {p.name for p in corrected.parameters}
    assert rows[0]["removed_direct_amplitude"] == "a"
    assert "z - 1" in corrected.processes[0].expression
    assert corrected.state_equations[0].rhs.startswith("-")
    assert {p.role.value for p in corrected.parameters} == {"nonnegative_coefficient"}
    assert {p.scope.value for p in corrected.parameters} == {"global"}
    assert len(corrected.parameters) == (2 if unknown else 1)
    assert len(guesses) == len(corrected.parameters)
    assert {u["assembly_policy"] for u in rows[0]["uses"]} == {
        "independent_gains" if unknown else "explicit_conversion"
    }


def test_scoped_policy_retains_signed_amplitude_and_reports_redundancy():
    candidate, context, contract = candidate_pair(signed_amplitude=True)
    corrected, _, rows = lower_candidate_gains(
        candidate,
        context,
        contract,
        "declared_or_independent",
        preserve_signed_amplitudes=True,
    )
    assert corrected.processes == candidate.processes
    assert next(p for p in corrected.parameters if p.name == "a").domain.value == "real"
    assert rows[0]["removed_direct_amplitude"] is None
    assert rows[0]["retained_signed_amplitude"] == "a"
    assert "redundancy" in rows[0]["normalization_limitation"]
    assert not rows[0]["global_scale_redundancy_fully_audited"]


def test_shared_amplitude_is_not_removed_from_other_equations():
    candidate, context, contract = candidate_pair(shared_amplitude=True)
    corrected, _, rows = lower_candidate_gains(
        candidate,
        context,
        contract,
        "declared_or_independent",
        preserve_signed_amplitudes=True,
    )
    assert "a" in {p.name for p in corrected.parameters}
    assert rows[0]["removed_direct_amplitude"] is None
    assert corrected.processes == candidate.processes


def test_historical_bundle_preserves_original_signed_normalization():
    candidate, context, contract = candidate_pair(signed_amplitude=True)
    bundle = {
        "context": context.model_dump(mode="json"),
        "initialization": {
            "base_candidate": candidate.model_dump(mode="json"),
            "plan": {
                "rules": {
                    "z": {
                        "initial": {
                            "mode": "value",
                            "role": "coefficient",
                            "guess": 0.1,
                        }
                    }
                }
            },
            "identity": "historical",
        },
    }
    before = copy.deepcopy(bundle)
    compiled = compile_bundle(
        bundle, contract, "independent_gains", {"rows": [{"fixed_covariates": {}}]}
    )
    assert bundle == before
    assert (
        compiled["gain_compilation"]["processes"][0]["removed_direct_amplitude"] == "a"
    )
    assert (
        "retained_signed_amplitude" not in compiled["gain_compilation"]["processes"][0]
    )
    assert "a" not in {
        p["name"] for p in compiled["initialization"]["base_candidate"]["parameters"]
    }


def saved_graph(tmp_path, *, failed_map=False):
    """Generate a sealed toy with shared dynamics, one root and one dependent map."""
    brief, context, source, contract, handoff, old, _ = fixture(
        tmp_path, dependent=True
    )
    declaration = signed.SignedProcess.model_validate(
        {
            "name": "q",
            "depends_on": ["z"],
            "kind": "influence",
            "scientific_meaning": "One signed law reused in two state equations",
            "uses": [
                {"target": "x", "sign": "positive"},
                {"target": "z", "sign": "negative"},
            ],
        }
    )
    binding = signed.binding_for(declaration)
    source["inventory"].append(
        {
            "name": "q",
            "definition": "algebraic",
            "scientific_role": declaration.scientific_meaning,
        }
    )
    for equation in source["equations"]:
        terms = tuple(
            EquationTerm(
                sources=t["sources"],
                outer_weight_sign="unrestricted",
                scientific_role=t["scientific_role"],
            )
            for t in equation["terms"]
        )
        equation["terms"] = [
            t.model_dump(mode="json")
            for t in signed.assemble([binding], equation["name"], terms)
        ]
    source["equations"].append(shared.definition(binding).model_dump(mode="json"))
    source["shared_process_contract"] = {
        "protocol": signed.POLICY,
        "bindings": [binding],
    }
    source.update(
        initialization_graph_contract=contract.model_dump(mode="json"),
        initialization_graph_handoff=handoff.model_dump(mode="json"),
    )
    lower(source, brief, context)
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        if "selected_initializer_dependency" in payload:
            if payload["selected_initializer_dependency"]["rule"]["mode"] == "value":
                return response({"role": "coefficient"})
            return response(
                {
                    "expression": "bad" if failed_map else "state_1+public_1",
                    "parameters": [],
                }
            )
        terms = payload["selected_equation"]["terms"]
        return response(
            {
                "functions": [
                    {
                        "expression": "a*(z-1)"
                        if t["lhs"] == "q"
                        else "+".join(t["sources"]),
                        "parameters": [{"name": "a", "role": "coefficient"}]
                        if t["lhs"] == "q"
                        else [],
                    }
                    for t in terms
                ]
            }
        )

    client = StagedTopologyClient(
        settings=old.settings,
        base_url="http://unused",
        directory=tmp_path / "provider",
        namespace="scoped-gain-test",
        seed=0,
        transport=transport,
    )
    result = run_staged_functions(
        brief,
        context,
        source,
        client,
        tmp_path / "functions",
        generation_granularity="equation_batch_atomic_repair",
        function_repair_policy="certified_outer_gain",
    )
    assert result["status"] == ("failed" if failed_map else "complete"), result["error"]
    prepared = {
        "brief": brief.model_dump(mode="json"),
        "case": {"context": context.model_dump(mode="json")},
        "function_source": source,
    }
    return result, prepared, calls


def test_saved_correction_preserves_graph_and_replays_without_compilation(
    tmp_path, monkeypatch
):
    result, source, calls = saved_graph(tmp_path)
    before = copy.deepcopy(result), copy.deepcopy(source), len(calls)
    from autoformalism.fitting import prefix_initialization
    from autoformalism.research import interaction_fit_handoff

    def prohibited(*args, **kwargs):
        pytest.fail("provider, optimizer or numerical rollout forbidden")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, prohibited)
    lowered = lower_saved_function_gains(result, source, policy=POLICY)
    initializer = rebind_saved_initializer(lowered, result["initialization"])
    assert initializer["graph"] == result["initialization"]["graph"]
    assert initializer["choices"] == result["initialization"]["choices"]
    assert initializer["attempts"] == result["initialization"]["attempts"]
    assert initializer["gain_migration"]["new_provider_calls"] == 0
    exported = build_scoped_gain_handoff(
        lowered, initializer, provenance={"case": "toy"}
    )
    assert exported["protocol"] == HANDOFF_PROTOCOL
    assert exported["parameter_scopes"]["trajectory_roots"] == {"z": "init_z_value"}
    assert exported["parameter_scopes"]["dependent_initial_states"] == ["z2"]
    assert (
        len(
            [
                n
                for n in exported["parameter_scopes"]["shared_dynamics"]
                if n.startswith("af_process_gain_")
            ]
        )
        == 2
    )
    assert exported["readiness"]["numerical_compile"]["status"] == "passed"
    assert exported["readiness"]["forecast"]["status"] == "policy_unresolved"
    assert (
        exported["prefix_fit_problem"]["graph"] == result["prefix_fit_problem"]["graph"]
    )
    monkeypatch.setattr(interaction_fit_handoff, "_numerical_readiness", prohibited)
    assert (
        verify_scoped_gain_handoff(
            lowered, initializer, exported, provenance={"case": "toy"}
        )
        == exported
    )
    assert (result, source, len(calls)) == before


def test_failed_initializer_retains_reusable_dynamics_only(tmp_path):
    result, source, _ = saved_graph(tmp_path, failed_map=True)
    lowered = lower_saved_function_gains(result, source, policy=POLICY)
    assert lowered["preserved_initialization_graph"] is None
    assert (
        lowered["initialization_graph_handoff"]
        == result["initialization_graph_handoff"]
    )
    assert len(lowered["gain_compilation"]["processes"][0]["uses"]) == 2


@pytest.mark.parametrize("fault", ["policy", "source", "draft", "dependencies"])
def test_saved_source_mismatch_and_nonoptin_rejected(tmp_path, fault):
    result, source, _ = saved_graph(tmp_path)
    policy = POLICY
    if fault == "policy":
        policy = "legacy"
    elif fault == "source":
        result["source_topology_result_sha256"] = "0" * 64
    elif fault == "draft":
        result["draft"]["interaction_functions"].pop()
    else:
        result["initialization_graph_handoff"]["records"].pop()
    with pytest.raises(ValueError):
        lower_saved_function_gains(result, source, policy=policy)


def test_resealed_handoff_parameter_or_graph_changes_rejected(tmp_path):
    result, source, _ = saved_graph(tmp_path)
    lowered = lower_saved_function_gains(result, source, policy=POLICY)
    initial = rebind_saved_initializer(lowered, result["initialization"])
    exported = build_scoped_gain_handoff(lowered, initial, provenance={})
    exported["prefix_fit_problem"]["graph"]["rules"]["z"]["initial"]["scope"] = "shared"
    exported["artifact_sha256"] = content_hash(
        {k: v for k, v in exported.items() if k != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="differs"):
        verify_scoped_gain_handoff(lowered, initial, exported, provenance={})
