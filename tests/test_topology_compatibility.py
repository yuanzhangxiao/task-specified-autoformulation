"""Three narrow ODE necessary conditions, explicit abstentions and counterexamples."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    TrainingCompatibilityEvidence,
    build_evidence,
    compact,
    diagnose,
    validate_evidence,
)


def trajectory(id_, target=(0, 1, 0), input_=(0, 0, 0), **kwargs):
    """No benchmark identities, hidden arrays, fitted values or reference ODEs."""
    return Trajectory(
        trajectory_id=id_,
        time=np.asarray(kwargs.pop("time", range(len(target))), float),
        targets={"a": np.asarray(target, float)},
        external_inputs={"u": np.asarray(input_, float)},
        auxiliaries={k: np.asarray(v, float) for k, v in kwargs.pop("aux", {}).items()},
        fixed_covariates=kwargs.pop("covariates", {}),
        derivatives={},
        **kwargs,
    )


def contract(**kwargs):
    return EvidenceContract(
        **(
            {
                "provenance": "synthetic public task/runtime fixture",
                "dependencies_exhaustive": True,
                "shared_parameters": True,
                "open_loop": True,
                "reset_impulse_free": True,
                "uniqueness_assumed": True,
                "interpolation": {"u": "linear_complete_knots"},
                "absolute_error_bounds": {"u": 0, "a": 0},
                "initial_rules": {"a": InitialRule(mode="observed_initial")},
            }
            | kwargs
        )
    )


def evidence(*items, config=None, context=None):
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, items, "not-trusted-for-hashing"),
        context or ValidationContext(targets=("a",), external_inputs=("u",)),
        config or contract(),
    )


def draft(kind="differential", sources=("a", "u"), extras=(), processes=()):
    variables = [{"name": "a", "definition": kind, "scientific_role": "output"}]
    equations = [
        {
            "name": "a",
            "terms": [
                {
                    "sources": sources,
                    "outer_weight_sign": "positive",
                    "scientific_role": "response",
                }
            ],
        }
    ]
    for name, definition, dependencies in extras:
        variables.append(
            {"name": name, "definition": definition, "scientific_role": name}
        )
        equations.append(
            {
                "name": name,
                "terms": [
                    {
                        "sources": dependencies,
                        "outer_weight_sign": "positive",
                        "scientific_role": name,
                    }
                ],
            }
        )
    return Draft.model_validate(
        {"variables": variables, "equations": equations, "processes": processes}
    )


def result(model, packet, property_):
    return next(
        v for v in diagnose(model, packet).diagnostics if v.property == property_
    )


@pytest.mark.parametrize("name", (SplitName.TEST, SplitName.VALIDATION))
def test_reject_heldout_before_reading_arrays(name):
    with pytest.raises(ValueError, match="train split only"):
        build_evidence(
            DatasetSplit(name, None, "heldout"),
            ValidationContext(targets=("a",)),
            contract(),
        )


def test_complete_source_hash_covers_aux_covariates_and_all_metadata():
    ctx = ValidationContext(
        targets=("a",),
        external_inputs=("u",),
        auxiliaries=("b",),
        fixed_covariates=("c",),
    )
    item = trajectory("first", aux={"b": [1, 2, 3]}, covariates={"c": 4})
    original = evidence(item, context=ctx)
    changes = [
        evidence(replace(item, auxiliaries={"b": np.array([1, 2, 4.0])}), context=ctx),
        evidence(replace(item, fixed_covariates={"c": 5}), context=ctx),
        evidence(item, context=ctx, config=contract(reset_impulse_free=None)),
        evidence(item, context=ctx, config=contract(absolute_error_bounds={"a": 1})),
        evidence(item, context=ctx, config=contract(interpolation={"u": "unknown"})),
        evidence(item, context=ctx, config=contract(initial_rules={})),
    ]
    assert all(v.sha256 != original.sha256 for v in changes)
    assert (
        TrainingCompatibilityEvidence.model_validate_json(original.model_dump_json())
        == original
    )
    with pytest.raises(ValueError, match="digest"):
        validate_evidence(
            original.model_copy(update={"contract": contract(open_loop=None)})
        )


def test_hash_ignores_supplied_split_fingerprint_and_input_order():
    a, b = trajectory("a"), trajectory("b")
    assert evidence(a, b) == evidence(b, a)


@pytest.mark.parametrize("target", ((1, float("nan")), (1, float("inf"))))
def test_reject_nonfinite_source(target):
    with pytest.raises(ValueError):
        evidence(trajectory("first", target, (0, 0)))


def test_missing_channel_is_not_filled_and_old_prompt_packet_is_not_evidence():
    with pytest.raises(ValueError, match="channels differ"):
        evidence(
            trajectory("first"),
            context=ValidationContext(targets=("a",), auxiliaries=("u",)),
        )
    with pytest.raises(ValueError):
        TrainingCompatibilityEvidence.model_validate(
            {"protocol": "prefit-training-evidence-1"}
        )


def test_algebraic_exact_tuple_conflict_and_pair_sse_bound():
    packet = evidence(trajectory("first", target=(1, 3, 1)))
    row = result(draft("algebraic", ("u",)), packet, "algebraic_map")
    assert row.status == "conditional_conflict"
    assert row.witnesses[0].pair_squared_error_lower_bound == 2
    assert row.witnesses[0].error_intervals_disjoint is True
    assert "not an acceptance gate" in compact(
        diagnose(draft("algebraic", ("u",)), packet)
    )


def test_witness_selection_shows_largest_contrast_not_first_roundoff_difference():
    packet = evidence(trajectory("first", (8, 0, 10)))
    row = result(draft("algebraic", ("u",)), packet, "algebraic_map")
    assert set(row.witnesses[0].observations) == {0, 10}
    packet = evidence(
        trajectory("first", (0, 1, 2)), trajectory("second", (0, 1.00000000001, 10))
    )
    row = result(draft(), packet, "closed_subsystem")
    assert row.witnesses[0].times == (2, 2)
    packet = evidence(trajectory("first", (0, 0.001, 0, 10, 0), (0,) * 5))
    row = result(draft(), packet, "scalar_autonomous")
    assert row.witnesses[0].observations == (0, 10, 0)


def test_algebraic_approximately_equal_is_not_equal():
    packet = evidence(trajectory("first", (1, 3, 1), (0, 1e-15, 0)))
    row = result(draft("algebraic", ("u",)), packet, "algebraic_map")
    assert row.status == "no_conflict_found" and not row.witnesses


def test_algebraic_noise_marks_exact_interpolation_only():
    row = result(
        draft("algebraic", ("u",)),
        evidence(
            trajectory("first"), config=contract(absolute_error_bounds={"u": 0, "a": 2})
        ),
        "algebraic_map",
    )
    assert row.status == "conditional_conflict"
    assert row.witnesses[0].error_intervals_disjoint is False
    assert "Exact reproduction" in row.assumptions[-1]


def test_algebraic_missing_target_noise_does_not_invent_a_bound():
    row = result(
        draft("algebraic", ("u",)),
        evidence(trajectory("first"), config=contract(absolute_error_bounds={"u": 0})),
        "algebraic_map",
    )
    assert row.witnesses[0].error_intervals_disjoint is None


def test_uncertain_supplied_arguments_abstain():
    row = result(
        draft("algebraic", ("u",)),
        evidence(
            trajectory("first"), config=contract(absolute_error_bounds={"u": 0.01})
        ),
        "algebraic_map",
    )
    assert row.status == "deferred"
    assert "exact supplied argument u" in " ".join(row.unavailable)


def test_complete_algebraic_tuple_accounts_for_covariates_and_time():
    ctx = ValidationContext(
        targets=("a",), external_inputs=("u",), fixed_covariates=("c",)
    )
    packet = evidence(
        trajectory("first", (0, 1, 2), covariates={"c": 0}),
        trajectory("second", (1, 2, 3), covariates={"c": 1}),
        context=ctx,
        config=contract(absolute_error_bounds={"a": 0, "u": 0, "c": 0}),
    )
    row = result(draft("algebraic", ("u", "c", "t")), packet, "algebraic_map")
    assert not row.witnesses


def test_shared_process_and_algebraic_dependencies_are_recursively_expanded():
    process = {
        "name": "p",
        "depends_on": ["u"],
        "kind": "influence",
        "scientific_meaning": "readout",
        "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
    }
    model = draft("algebraic", (), processes=[process])
    row = result(model, evidence(trajectory("first")), "algebraic_map")
    assert row.status == "conditional_conflict" and row.scope == ("a", "p", "u")


def test_generated_observations_are_not_substituted_for_actual_hidden_arguments():
    model = draft("algebraic", ("z",), extras=[("z", "differential", ("u",))])
    packet = evidence(trajectory("first"))
    row = result(model, packet, "algebraic_map")
    assert row.status == "deferred"
    assert result(model, packet, "scalar_autonomous").status == "deferred"


def test_identical_subsystem_conflict_requires_complete_initial_and_forcing_evidence():
    packet = evidence(trajectory("first", (1, 2, 3)), trajectory("second", (1, 4, 5)))
    row = result(draft(), packet, "closed_subsystem")
    assert row.status == "conditional_conflict"
    assert row.comparisons == 3
    assert row.witnesses[0].times == (1, 1)


@pytest.mark.parametrize(
    "field",
    (
        "dependencies_exhaustive",
        "shared_parameters",
        "open_loop",
        "reset_impulse_free",
        "uniqueness_assumed",
    ),
)
def test_unknown_conditioning_is_not_assumed(field):
    row = result(
        draft(),
        evidence(
            trajectory("first"),
            trajectory("second", (0, 2, 0)),
            config=contract(**{field: None}),
        ),
        "closed_subsystem",
    )
    assert row.status == "deferred" and row.unavailable


def test_different_initials_do_not_make_a_dynamic_conflict():
    row = result(
        draft(),
        evidence(trajectory("first", (1, 2, 3)), trajectory("second", (2, 4, 5))),
        "closed_subsystem",
    )
    assert not row.witnesses


def test_constant_input_does_not_require_constant_state():
    row = result(
        draft(), evidence(trajectory("first", (1, 0.5, 0.25))), "scalar_autonomous"
    )
    assert row.status == "no_conflict_found"


def test_memory_accumulator_need_not_depend_on_itself():
    row = result(
        draft(sources=("u",)),
        evidence(trajectory("first", (0, 1, 2), (1, 1, 1))),
        "scalar_autonomous",
    )
    assert row.status == "no_conflict_found"


def test_hidden_initial_equality_is_not_inferred_from_observed_initial():
    model = draft(sources=("z",), extras=[("z", "differential", ())])
    packet = evidence(trajectory("first"), trajectory("second", (0, 2, 0)))
    assert result(model, packet, "closed_subsystem").status == "deferred"
    shared = packet.contract.model_copy(
        update={
            "initial_rules": {
                "a": InitialRule(mode="observed_initial"),
                "z": InitialRule(mode="shared_parameter"),
            }
        }
    )
    row = result(
        model,
        evidence(
            *[trajectory("first"), trajectory("second", (0, 2, 0))], config=shared
        ),
        "closed_subsystem",
    )
    assert row.status == "conditional_conflict"


def test_common_initializer_maps_match_inputs_without_fitting_or_evaluating_formula():
    model = draft(sources=("z",), extras=[("z", "differential", ())])
    cfg = contract(
        initial_rules={
            "a": InitialRule(mode="observed_initial"),
            "z": InitialRule(mode="common_map", arguments=("u",)),
        }
    )
    packet = evidence(trajectory("first"), trajectory("second", (0, 2, 0)), config=cfg)
    assert result(model, packet, "closed_subsystem").status == "conditional_conflict"
    other = evidence(
        trajectory("first"), trajectory("second", (0, 2, 0), (1, 1, 1)), config=cfg
    )
    assert not result(model, other, "closed_subsystem").witnesses


def test_piecewise_linear_matching_uses_union_of_complete_knots():
    first = trajectory("first", (0, 1, 2), (0, 1, 2), time=(0, 1, 2))
    second = trajectory("second", (0, 3), (0, 2), time=(0, 2))
    row = result(draft(), evidence(first, second), "closed_subsystem")
    assert row.status == "conditional_conflict" and row.comparisons == 2
    # Same endpoints but extra interior knot changes the forcing function.
    changed = replace(first, external_inputs={"u": np.array([0, 5, 2.0])})
    assert not result(draft(), evidence(changed, second), "closed_subsystem").witnesses


def test_ramp_next_knot_can_change_common_prefix_despite_matching_earlier_samples():
    first = trajectory("first", (0, 1, 2), (0, 0, 1), time=(0, 1, 2))
    second = trajectory("second", (0, 2), (0, 2), time=(0, 3))
    assert not result(draft(), evidence(first, second), "closed_subsystem").witnesses


def test_rounded_interpolation_match_is_not_exact_function_equality():
    first = trajectory("first", (0, 1), (0, 1), time=(0, 3))
    second = trajectory("second", (0, 2, 3), (0, float(1 / 3), 1), time=(0, 1, 3))
    # np.interp rounds both to the same middle value, but the two represented
    # piecewise-linear functions differ as exact real functions.
    assert np.interp(1, first.time, first.external_inputs["u"]) == float(1 / 3)
    assert not result(draft(), evidence(first, second), "closed_subsystem").witnesses


def test_previous_interpolation_detects_changes_and_unknown_interpolation_defers():
    items = (trajectory("first"), trajectory("second", (0, 2, 0)))
    cfg = contract(interpolation={"u": "previous_complete_knots"})
    assert result(draft(), evidence(*items, config=cfg), "closed_subsystem").witnesses
    row = result(
        draft(), evidence(*items, config=contract(interpolation={})), "closed_subsystem"
    )
    assert row.status == "deferred"


def test_covariate_and_conversion_distinction_prevents_false_equivalence():
    ctx = ValidationContext(
        targets=("a",), external_inputs=("u",), fixed_covariates=("c",)
    )
    packet = evidence(
        trajectory("first", covariates={"c": 1}),
        trajectory("second", (0, 2, 0), covariates={"c": 2}),
        context=ctx,
        config=contract(absolute_error_bounds={"u": 0, "a": 0, "c": 0}),
    )
    for conversion in ("1/c", None):
        process = {
            "name": "p",
            "depends_on": ["u"],
            "kind": "influence",
            "scientific_meaning": "rate",
            "uses": [{"target": "a", "sign": "positive", "conversion": conversion}],
        }
        row = result(draft(sources=(), processes=[process]), packet, "closed_subsystem")
        assert not row.witnesses and "c" in row.scope


@pytest.mark.parametrize("values", ((0, 1, 0), (0, 1, 1, 0), (2, 1, 2)))
def test_scalar_reversal_with_constant_nonzero_forcing(values):
    packet = evidence(trajectory("first", values, [4] * len(values)))
    row = result(draft(), packet, "scalar_autonomous")
    assert row.status == "conditional_conflict"
    assert row.witnesses[0].error_intervals_disjoint


def test_noisy_reversal_remains_exact_sample_conflict_not_beyond_noise():
    packet = evidence(
        trajectory("first", (0, 0.001, 0)),
        config=contract(absolute_error_bounds={"a": 0.01, "u": 0}),
    )
    row = result(draft(), packet, "scalar_autonomous")
    assert row.witnesses[0].error_intervals_disjoint is False


def test_coupling_time_forcing_and_readout_counterexamples():
    packet = evidence(trajectory("first"))
    coupled = draft(sources=("z",), extras=[("z", "differential", ("a",))])
    explicit_time = draft(sources=("a", "t"))
    readout = draft("algebraic", ("z",), extras=[("z", "differential", ())])
    for model in (coupled, explicit_time, readout):
        assert result(model, packet, "scalar_autonomous").status == "deferred"
    varying = evidence(trajectory("first", input_=(0, 1, 0)))
    assert not result(draft(), varying, "scalar_autonomous").witnesses


def test_target_history_and_unresolved_graph_defer():
    ctx = ValidationContext(
        targets=("a",), external_inputs=("u",), lagged_targets=("a",)
    )
    assert all(
        row.status == "deferred"
        for row in diagnose(
            draft(), evidence(trajectory("first"), context=ctx)
        ).diagnostics
    )
    assert all(
        row.status == "deferred"
        for row in diagnose(
            draft(sources=("missing",)), evidence(trajectory("first"))
        ).diagnostics
    )


def test_algebraic_cycle_and_short_sequences_abstain():
    cycle = draft("algebraic", ("z",), extras=[("z", "algebraic", ("a",))])
    assert all(
        row.status == "deferred"
        for row in diagnose(cycle, evidence(trajectory("first"))).diagnostics
    )
    short = evidence(trajectory("first", (0, 1), (0, 0)))
    assert result(draft(), short, "scalar_autonomous").status == "deferred"


def test_process_owned_state_collision_abstains():
    process = {
        "name": "a",
        "depends_on": ["u"],
        "kind": "influence",
        "scientific_meaning": "conflicting ownership",
        "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
    }
    rows = diagnose(
        draft(processes=[process]), evidence(trajectory("first"))
    ).diagnostics
    assert all(row.status == "deferred" for row in rows)
    assert "ownership conflict" in " ".join(rows[0].unavailable)


def test_process_declaration_order_does_not_erase_dependency_edges():
    processes = [
        {
            "name": "p",
            "depends_on": ["u"],
            "kind": "influence",
            "scientific_meaning": "upstream",
            "uses": [{"target": "q", "sign": "positive", "conversion": "1"}],
        },
        {
            "name": "q",
            "depends_on": ["u"],
            "kind": "influence",
            "scientific_meaning": "downstream",
            "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
        },
    ]
    packet = evidence(trajectory("first"))
    left = result(draft("algebraic", (), processes=processes), packet, "algebraic_map")
    right = result(
        draft("algebraic", (), processes=processes[::-1]), packet, "algebraic_map"
    )
    assert left == right
    assert "p" in left.scope


def test_cli_synthetic_smoke_is_deterministically_resumable(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/audit_topology_compatibility.py"
    )
    command = [sys.executable, str(script), "--example", "--output", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    saved = (tmp_path / "report.json").read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (tmp_path / "report.json").read_bytes() == saved
    report = json.loads(saved)
    assert not report["test_data_opened"] and report["llm_calls"] == 0
    assert {
        r["property"]
        for r in report["diagnostics"]
        if r["status"] == "conditional_conflict"
    } == {"algebraic_map", "closed_subsystem", "scalar_autonomous"}
