"""Historical state additions require model identity and recorded provenance."""

import json

import pytest

from scripts import collect_iclr_latent_histories as collect
from scripts import iclr_latent_revision_audit as audit
from scripts import report_iclr_latent_audit as report
from tests.test_iclr_local_interventions import candidate


def seal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**value, "artifact_sha256": collect.digest(value)}))


def test_collector_keeps_unstarted_and_incomplete_runs(tmp_path):
    task = {"task_id": "case_seed0", "cell": "case", "seed": 0, "critic": False}
    plan = {
        "protocol": "final-component-campaign-1",
        "tasks": [task],
        "config": {"rounds": 2},
        "cells": {},
    }
    seal(tmp_path / "plan.json", plan)
    result = collect.collect(tmp_path)
    assert result["runs"][0]["final_status"] == "not_started"
    # An incumbent-free result is not a successfully retained zero-state model.
    seal(
        tmp_path / "results/case_seed0/round_00/result.json",
        {"task": task, "round": 0, "status": "construction_failed", "selected": None},
    )
    result = collect.collect(tmp_path)
    assert result["runs"][0]["final_status"] == "search_incomplete"
    assert result["runs"][0]["final_model"] is None


def test_collector_rejects_corrupt_seal_and_wrong_fit():
    endpoint = {
        "bundle": {"candidate": {}},
        "fit": {"lowered_candidate_sha256": "wrong"},
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        collect.model_record(endpoint)


def test_request_projection_excludes_headers_and_matches_request(tmp_path):
    path = tmp_path / "call.json"
    path.write_text(
        json.dumps(
            {
                "request_hash": "a" * 64,
                "request": {
                    "headers": {"Authorization": "sentinel"},
                    "body": {
                        "messages": [
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "training_evidence": {"split": "train"},
                                        "unrelated": "omit",
                                    }
                                ),
                            }
                        ]
                    },
                },
            }
        )
    )
    result = collect.request_projection(path, "a" * 64)
    assert result["payload"] == {"training_evidence": {"split": "train"}}
    assert "sentinel" not in json.dumps(result)
    with pytest.raises(ValueError, match="identity mismatch"):
        collect.request_projection(path, "b" * 64)


def test_collector_corrupt_plan_is_not_a_missing_campaign(tmp_path):
    seal(tmp_path / "plan.json", {"protocol": "unsupported"})
    with pytest.raises(ValueError, match="unsupported"):
        collect.collect(tmp_path)
    (tmp_path / "plan.json").write_text('{"artifact_sha256":"wrong"}')
    with pytest.raises(ValueError, match="invalid campaign seal"):
        collect.collect(tmp_path)


def model(equations, requirements):
    """Small complete fitted systems with known causal pathways."""
    targets = [n for n in ("Gp", "I") if n in equations]
    c = candidate(list(equations), equations, targets).model_dump(mode="json")
    return {
        "candidate": c,
        "parameters": {},
        "context": {"targets": targets, "external_inputs": ["u"], "auxiliaries": []},
        "brief": {"requirements": requirements},
    }


REQS = [
    {"id": "meal", "drivers": ["u"], "targets": ["Gp"]},
    {"id": "action", "drivers": ["I"], "targets": ["Gp"]},
]


def test_memberships_overlap_without_misassigning_upstream_secretion():
    m = model({"Gp": "u-X", "I": "M-I+Gp", "X": "I-X", "M": "u-M"}, REQS)
    a = audit.anatomy(m)
    roles = {r["id"]: r["latent_states"] for r in a["requirements"]}
    assert roles == {"meal": ["M", "X"], "action": ["X"]}
    assert a["outside_required_paths"] == []


def test_autonomous_state_and_disconnected_coordinate_are_different():
    m = model({"Gp": "u+z", "z": "-z", "unused": "u-unused"}, REQS[:1])
    a = audit.anatomy(m)
    assert a["outside_required_paths"] == ["z"]
    assert a["inactive_states"] == ["unused"]
    assert a["requirements"][0]["path_present"]
    assert not a["requirements"][0]["latent_states"]
    m["candidate"]["state_equations"][0]["rhs"] = "z"
    assert not audit.anatomy(m)["requirements"][0]["path_present"]


def test_added_state_can_supplement_a_required_mechanism_after_feedback():
    parent = model({"Gp": "A", "A": "u-A"}, REQS[:1])
    child = model({"Gp": "A+M", "A": "u-A", "M": "u-M"}, REQS[:1])
    h = "a" * 64
    visit = {
        "round": 1,
        "selected_model": "child",
        "attempts": [
            {
                "accepted": True,
                "request_hash": h,
                "raw": {
                    "hypothesis": "Add memory to explain the late response.",
                    "evidence_refs": ["R001"],
                },
            }
        ],
        "requests": [
            {
                "request_hash": h,
                "status": "available",
                "payload": {
                    "evidence_catalog": [{"ref": "R001", "record": "train_0"}],
                    "training_evidence": {"split": "train"},
                },
            }
        ],
    }
    run = {"visits": [{"round": 0, "selected_model": "parent"}, visit]}
    b = audit.state_births(run, {"parent": parent, "child": child})["M"]
    assert b["pure_augmentation"] and b["feedback"]["numerical_evidence_cited"]
    assert b["prior_mechanisms"][0]["latent_states"] == ["A"]
    # Missing chronology cannot establish an introduction, even with feedback.
    visit["round"] = 3
    assert not audit.state_births(run, {"parent": parent, "child": child})["M"][
        "pure_augmentation"
    ]


def test_replacement_is_not_a_pure_dimension_addition():
    before = model({"Gp": "X", "X": "u-X"}, REQS[:1])
    after = model({"Gp": "Y", "Y": "u-Y"}, REQS[:1])
    run = {
        "visits": [
            {"round": 0, "selected_model": "a"},
            {"round": 1, "selected_model": "b"},
        ]
    }
    birth = audit.state_births(run, {"a": before, "b": after})["Y"]
    assert birth["post_fit_introduction"]
    assert not birth["pure_augmentation"]
    assert birth["simultaneous_removals"] == ["X"]


def test_unknown_or_test_references_cannot_claim_training_motivation():
    v = {
        "attempts": [
            {"accepted": True, "request_hash": "a", "raw": {"evidence_refs": ["R1"]}}
        ],
        "requests": [
            {
                "request_hash": "a",
                "status": "available",
                "payload": {
                    "training_evidence": {"split": "test"},
                    "evidence_catalog": [{"ref": "R1"}],
                },
            }
        ],
    }
    assert not audit.feedback_attribution(v)["numerical_evidence_cited"]
    v["requests"][0]["payload"]["training_evidence"]["split"] = "train"
    v["attempts"][0]["raw"]["evidence_refs"] = ["unknown"]
    assert not audit.feedback_attribution(v)["numerical_evidence_cited"]


def test_missing_models_are_not_zero_state_models():
    r = {"available": False, "final_status": "pruning_pending"}
    s = audit.summarize([r])
    assert s["planned"] == 1 and s["available"] == 0
    assert s["active_latent_count"]["histogram"] == {}


def test_pruned_driver_is_not_a_new_semantic_mechanism():
    before = model({"Gp": "A", "A": "u-A"}, REQS[:1])
    after = model({"Gp": "A", "A": "-A"}, REQS[:1])
    run = {
        "visits": [
            {"round": 0, "selected_model": "before"},
            {"round": 1, "selected_model": "after"},
        ]
    }
    birth = audit.state_births(run, {"before": before, "after": after})["A"]
    assert birth["lineage_mechanisms"] == ["meal"]
    assert audit.anatomy(after)["outside_required_paths"] == ["A"]


def test_observed_auxiliary_alias_is_not_an_unobserved_latent():
    m = model({"Gp": "u+Q", "Q": "-Q"}, REQS[:1])
    m["modeled_observed_auxiliaries"] = ["Q"]
    a = audit.anatomy(m)
    assert a["active_latents"] == []
    assert a["modeled_observed_auxiliary_states"] == ["Q"]


def test_mechanism_summary_separates_missing_and_zero_latent_paths():
    rows = [
        {
            "latent_count": 0,
            "path_present": present,
            "added_after_existing_latent_count": 0,
        }
        for present in (True, False)
    ]
    s = audit.summarize_mechanisms(rows)
    assert s["latent_count_histogram"] == {0: 2}
    assert s["represented_without_latents"] == s["path_absent"] == 1


def test_empty_driver_clause_is_explicitly_a_broad_balance():
    req = [{"id": "balance", "targets": ["Gp"], "drivers": []}]
    a = audit.anatomy(model({"Gp": "u+A", "A": "-A"}, req))
    assert a["requirements"][0]["latent_states"] == ["A"]
    assert a["requirements"][0]["mapping_rule"].startswith("broad_target_balance")
    assert a["requirements"][0]["path_present"] is None


def test_semantic_review_never_calls_a_lost_required_path_novel():
    state = {"required_mechanisms": [], "birth": {"lineage_mechanisms": ["meal"]}}
    assert (
        report.semantic_role(state, None, REQS)["classification"]
        == "required_role_path_lost"
    )
    state["birth"] = {}
    with pytest.raises(ValueError, match="unreviewed"):
        report.semantic_role(state, None, REQS)
    assert (
        report.semantic_role(state, None, [])["classification"]
        == "no_mechanistic_requirements"
    )


def test_guided_numerical_feedback_remains_distinct_from_standard_citations():
    v = {
        "guided_feedback": {
            "numerical_evidence_cited": False,
            "numerical_feedback_supplied": True,
        }
    }
    assert audit.feedback_attribution(v)["numerical_feedback_supplied"]
    assert not audit.feedback_attribution(v)["numerical_evidence_cited"]


def test_final_pruning_removes_an_addition_from_retained_counts():
    parent = model({"Gp": "A", "A": "u-A"}, REQS[:1])
    child = model({"Gp": "A+M", "A": "u-A", "M": "u-M"}, REQS[:1])
    final = model({"Gp": "A", "A": "u-A", "M": "u-M"}, REQS[:1])
    campaign = {
        "models": {"p": parent, "c": child, "f": final},
        "plan_sha256": "plan",
        "model_settings": {"model": "20b"},
        "rounds": 2,
        "runs": [
            {
                "task": {
                    "task_id": "s0",
                    "cell": "case",
                    "seed": 0,
                    "treatment": "full",
                },
                "visits": [
                    {"round": 0, "selected_model": "p"},
                    {"round": 1, "selected_model": "c"},
                ],
                "final_status": "complete",
                "search_complete": True,
                "final_model": "f",
                "latest_model": "c",
            }
        ],
    }
    rows, _, _ = audit.audit_campaign("example", campaign)
    by_snapshot = {r["snapshot"]: r for r in rows}
    assert by_snapshot["end_search"]["retained_pure_augmentation_count"] == 1
    assert by_snapshot["final"]["retained_pure_augmentation_count"] == 0
    assert by_snapshot["latest_available"]["model_id"] == "f"
