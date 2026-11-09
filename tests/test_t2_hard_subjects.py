"""Ensure the frozen roster and validation-only comparisons retain their provenance."""

from copy import deepcopy
from itertools import product

import pytest

from autoformalism.staged_topology import content_hash
from scripts import analyze_t2_hard_selection as analysis
from scripts import prepare_t2_hard_subjects as prepare


def unavailable_inventory() -> dict:
    """One planned but unavailable fixed-round visit for every primary arm."""
    visits = []
    for arm, critic, verifier, seed in product(
        ("full", "brief_only"), (False, True), (False, True), (0, 1)
    ):
        suffix = f"c{int(critic)}v{int(verifier)}s1"
        task = {
            "task_id": f"cell08_seed{seed}_{arm}_{suffix}",
            "cell": prepare.CELL,
            "seed": seed,
            "arm": arm,
            "critic": critic,
            "scientific_verifier": verifier,
            "shared_processes": True,
            "secondary": False,
        }
        visits.append({"task": task, "round": prepare.ROUND, "status": "missing"})
    return {
        "protocol": "component-model-inspection-1",
        "campaign_protocol": "final-component-campaign-1",
        "test_data_opened": False,
        "plan_sha256": "a" * 64,
        "public_context": {
            prepare.CELL: {"target_contract": {"public_prompt_sha256": prepare.PROMPT}}
        },
        "visits": visits,
        "occurrences": [],
    }


def test_missing_visits_never_replaced_by_trial_or_last_available():
    inventory = unavailable_inventory()
    task = inventory["visits"][0]["task"]
    inventory["occurrences"] = [
        {"task": task, "round": 4, "role": "selected", "model_id": "earlier"},
        {"task": task, "round": 5, "role": "trial", "model_id": "lower-error-trial"},
    ]
    rows = prepare.component_rows(inventory)
    assert len(rows) == 16 and all(r["status"] == "unavailable" for r in rows)
    pairs = analysis.matched_pairs({"rows": rows, "artifact_sha256": "b" * 64})
    assert len(pairs["pairs"]) == 24 and pairs["eligible_pairs"] == 0
    inventory["visits"].pop()
    with pytest.raises(ValueError, match="roster"):
        prepare.component_rows(inventory)


def test_hash_and_prompt_disagreement_stop_export():
    inventory = unavailable_inventory()
    inventory["artifact_sha256"] = content_hash(inventory)
    assert prepare.verify_seal(inventory) == inventory["artifact_sha256"]
    inventory["public_context"][prepare.CELL]["target_contract"][
        "public_prompt_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="digest"):
        prepare.verify_seal(inventory)
    with pytest.raises(ValueError, match="prompt"):
        prepare.component_rows(inventory)


def test_all_declared_matching_filters_are_applied_without_outcome_data():
    row = {
        "source_id": "a",
        "states": 3,
        "parameters": 10,
        "status": "ready",
        "validation_nmse": 1.0,
        "functional_sha256": "a",
    }
    other = {
        **row,
        "source_id": "b",
        "parameters": 12,
        "validation_nmse": 1.2,
        "functional_sha256": "b",
    }
    assert analysis.match_pair(row, other)["eligible"]
    bad = {
        **other,
        "functional_sha256": "a",
        "validation_nmse": 1.201,
        "parameters": 13,
        "states": 2,
    }
    assert set(analysis.match_pair(row, bad)["reasons"]) == {
        "validation_caliper",
        "parameter_count",
        "state_count",
        "same_functional_fingerprint",
    }
    assert not analysis.match_pair(row, {**other, "validation_nmse": None})["eligible"]


def test_baseline_without_native_d3_stays_unavailable_even_if_record_has_a_subject():
    rows = [
        {
            "source_kind": m,
            "repetition": r,
            "tier": "hard",
            "cohort": "phase_b",
            "index": i,
            "benchmark_id": prepare.CELL,
            "status": "unavailable",
        }
        for i, (m, r) in enumerate(product(sorted(prepare.METHODS), range(3)))
    ]
    for row in rows:
        if row["source_kind"] == "d3":
            row.update(
                status="ready", subject={"historical_continuous_reinterpretation": True}
            )
    plan = {
        "protocol": "saved-baseline-validation-1",
        "artifact_sha256": "a" * 64,
        "test_data_opened": False,
        "rows": rows,
    }
    result = prepare.baseline_subjects(plan)
    assert len(result) == 12
    assert all(
        r["status"] == "unavailable" and r["semantics"] == "native_increment"
        for r in result
        if r["method"] == "d3"
    )
    duplicate = deepcopy(plan)
    duplicate["rows"].append(duplicate["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        prepare.baseline_subjects(duplicate)


def test_selected_equations_and_fitted_values_are_bound_to_their_origin():
    from autoformalism.expressions import ValidationContext
    from autoformalism.schemas import CandidateModel

    inventory = unavailable_inventory()
    task = inventory["visits"][0]["task"]
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": "test_replay",
            "parent_candidate_id": None,
            "states": [{"name": x, "kind": "observed"} for x in ("Gp", "I")],
            "state_equations": [
                {"state": "Gp", "rhs": "0"},
                {"state": "I", "rhs": "0"},
            ],
            "observation_mappings": [
                {"channel": x, "expression": x} for x in ("Gp", "I")
            ],
            "initial_conditions": [
                {"state": x, "scope": "global", "expression": x} for x in ("Gp", "I")
            ],
        }
    ).model_dump(mode="json")
    model = {
        "cell": prepare.CELL,
        "candidate": candidate,
        "candidate_representation": "lowered",
        "parameter_status": "saved_fit_vector",
        "parameters": {},
        "lowered_context": ValidationContext(
            targets=("Gp", "I"),
            auxiliaries=("Uii",),
            external_inputs=("meal_event_g", "insulin_pmol_per_kg_min"),
        ).model_dump(mode="json"),
    }
    identifier = content_hash(model)
    record = {
        "task": task,
        "round": 5,
        "role": "selected",
        "model_id": identifier,
        "source_sha256": "f" * 64,
        "fit": {
            "lowered_candidate_sha256": prepare.content_sha256(candidate),
            "parameters": {},
            "status": "complete",
            "training_only_parameter_estimation": True,
            "validation_initials_fitted": False,
            "training": {"normalized_mse": 1.0},
            "validation": {"normalized_mse": 2.0},
        },
        "provenance": {
            "origin_task": task["task_id"],
            "origin_round": 2,
            "fit_result_sha256": "b" * 64,
        },
    }
    inventory.update(models={identifier: model}, occurrences=[record])
    row = prepare.component_rows(inventory)[0]
    assert row["selected_origin_round"] == 2 and row["candidate"] == candidate
    record["fit"]["validation_initials_fitted"] = True
    with pytest.raises(ValueError, match="selection differs"):
        prepare.component_rows(inventory)


def test_native_d3_supplement_requires_correct_prompt_and_preserves_other_rows(
    tmp_path, monkeypatch
):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    base = tmp_path / "base.json"
    prior = sealed_write(
        base,
        {
            "snapshot_protocol": "iclr-t2-hard-six-visits-1",
            "test_data_opened": False,
            "rows": [{"method": "sindy", "repetition": 0, "status": "ready"}]
            + [
                {"method": "d3", "repetition": n, "status": "unavailable"}
                for n in range(3)
            ],
        },
    )
    native = [
        {
            "benchmark_id": prepare.CELL,
            "method": "d3",
            "tier": "hard",
            "repetition": n,
            "status": "ready",
            "data_identity": {"prompt": prepare.PROMPT},
        }
        for n in range(3)
    ]
    monkeypatch.setattr(prepare, "d3_rows", lambda root: native)
    supplemented = prepare.supplement_d3(base, tmp_path, tmp_path / "new.json")
    assert supplemented["source_subject_bundle_sha256"] == prior["artifact_sha256"]
    assert prior["rows"][0] in supplemented["rows"]
    assert all(r["status"] == "ready" for r in supplemented["rows"])
    native[0]["data_identity"]["prompt"] = "d" * 64
    with pytest.raises(ValueError, match="interface"):
        prepare.supplement_d3(base, tmp_path, tmp_path / "bad.json")
