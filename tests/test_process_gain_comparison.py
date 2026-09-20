"""Matched laws, explicit/free gains, unit-aware diagnostics and provenance."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal import detention_process_pilot as pilot
from autoformalism.rebuttal.process_gain_comparison import (
    compile_bundle,
    conservation_diagnostic,
)
from scripts import smoke_signed_processes as smoke


@pytest.fixture
def pair(tmp_path):
    root, plan = smoke.fixture(tmp_path)
    return root, plan, smoke.construct_pair(root, plan)


def test_same_construction_and_runtime_consumers(pair):
    root, plan, (tasks, common, derived, calls, client) = pair
    assert len(plan["tasks"]) == 16
    assert all(p["status"] == "constructed" for p in derived), derived
    assert {p["common_proposal_sha256"] for p in derived} == {common["artifact_sha256"]}
    assert not any(c["payload"].get("stage") == "process_uses" for c in calls)
    assert (
        sum(c["payload"].get("protocol") == "shared-process-contract-2" for c in calls)
        == 1
    )
    a, b = (p["gain_compilation"]["processes"][0] for p in derived)
    assert a["normalized_law"] == b["normalized_law"]
    assert a["removed_direct_amplitude"] and b["removed_direct_amplitude"]
    assert len({u["gain"] for u in a["uses"]}) == 1
    assert len({u["gain"] for u in b["uses"]}) == 2
    assert (
        derived[0]["bundle"]["initialization"]["plan"]
        == derived[1]["bundle"]["initialization"]["plan"]
    )
    assert len(derived[0]["bundle"]["candidate"]["parameters"]) == 2
    assert len(derived[1]["bundle"]["candidate"]["parameters"]) == 3
    guesses = derived[1]["bundle"]["fit_parameter_guesses"]
    for use in b["uses"]:
        assert guesses[use["gain"]] == (1 if use["target"] == "h_up" else 0.5)
        assert use["starting_point"] == "matched_constant_conversion"
    request = pilot.request_for(derived[1]["bundle"], plan, tasks[1], 0)
    assert dict(request.parameter_guesses).items() >= guesses.items()
    assert "PRIVATE_ORACLE_POISON" not in json.dumps(plan)
    before = len(calls)
    assert (
        pilot.construct(
            root / "construction", plan, tasks[0]["construction_task"], client
        )
        == common
    )
    assert len(calls) == before


def test_area_weighted_balance_does_not_require_equal_gains(pair):
    _, plan, (_, _, derived, _, _) = pair
    training = plan["cells"]["coupled"]["training"]
    tied, free = [p["bundle"] for p in derived]
    rows = tied["gain_compilation"]["processes"][0]["uses"]
    assert conservation_diagnostic(tied, {rows[0]["gain"]: 3}, training)[0][
        "cancellation_pass"
    ]
    rows = free["gain_compilation"]["processes"][0]["uses"]
    values = {r["gain"]: 2 if r["target"] == "h_up" else 1 for r in rows}
    assert conservation_diagnostic(free, values, training)[0]["cancellation_pass"]
    equal = {r["gain"]: 1 for r in rows}
    assert not conservation_diagnostic(free, equal, training)[0]["cancellation_pass"]
    zero = {r["gain"]: 0 for r in rows}
    assert conservation_diagnostic(free, zero, training)[0]["cancellation_pass"] is None


def test_unknown_conversion_preserves_independent_arm(tmp_path):
    root, plan = smoke.fixture(tmp_path)
    _, _, derived, _, _ = smoke.construct_pair(root, plan, "unknown_conversion")
    assert derived[0]["status"] == "gain_assembly_unavailable"
    assert "conversion unavailable" in derived[0]["gain_compilation_error"]
    assert derived[1]["status"] == "constructed"


def test_no_process_proposal_is_not_failure(tmp_path):
    root, plan = smoke.fixture(tmp_path)
    _, _, derived, _, _ = smoke.construct_pair(root, plan, "empty")
    assert all(p["status"] == "constructed" for p in derived)
    assert all(p["gain_compilation"]["processes"] == [] for p in derived)
    assert derived[0]["bundle"]["candidate"] == derived[1]["bundle"]["candidate"]


def test_tampered_identity_cannot_be_compiled(pair):
    _, plan, (_, common, _, _, _) = pair
    bundle = common["bundle"]
    bundle["initialization"]["base_candidate"]["state_equations"][0]["rhs"] += "+q"
    with pytest.raises(ValueError, match="exactly one bound"):
        compile_bundle(
            bundle,
            common["shared_process_contract"],
            "independent_gains",
            plan["cells"]["coupled"]["training"],
        )


def test_signed_server_protocol():
    server = (pilot.REPO / "scripts/hpc/run_staged_topology_server.sh").read_text()
    assert "detention-process-pilot-3" in server


def test_derived_candidate_rechecks_negative_control_requirements(pair):
    root, plan, (tasks, common, _, _, _) = pair
    task = deepcopy(tasks[0])
    task.update(task_id="changed_boundary_contract", case="independent")
    # The common proposal's earlier check used the coupled task's requirements.
    assert common["structural"]["eligible"]
    result = pilot.gain_proposal(root, plan, task)
    assert result["status"] == "requirement_failed"
    assert result["structural"]["forbidden_upstream_dependencies"]
    assert result["bundle"] is None
    assert result["equation_inventory"] is None


def test_signed_law_is_not_made_positive(tmp_path):
    root, plan = smoke.fixture(tmp_path)
    _, _, derived, _, _ = smoke.construct_pair(root, plan, "signed_law")
    for proposal in derived:
        row = proposal["gain_compilation"]["processes"][0]
        assert "-1" in row["normalized_law"]
        assert row["removed_direct_amplitude"]


def test_process_prose_is_not_a_runtime_nonlinearity_gate(tmp_path):
    root, plan = smoke.fixture(tmp_path)
    _, common, derived, calls, _ = smoke.construct_pair(root, plan, "nonlinear_prose")
    assert all(p["status"] == "constructed" for p in derived)
    assert not common["fallback_used"]
    assert not any("selected_term" in c["payload"] for c in calls)
    for slot in common["bundle"]["slots"]:
        term = slot["selected_term"]
        if term.get("shared_process_law") or term.get("shared_process_use"):
            assert term["scientific_role_is_context_only"]
            assert not term["functional_obligation"][
                "requires_nonlinear_source_dependence"
            ]
