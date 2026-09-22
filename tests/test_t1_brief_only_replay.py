"""Saved-fit identity and physical-reference matching must survive this diagnostic."""

from copy import deepcopy

import pytest

from scripts import replay_t1_brief_only as brief
from scripts.smoke_public_fitting import control


def selected_result():
    request, _, _ = control("general-rollout-v1")
    model, _, _ = brief.public._lower(request)
    task = {
        "task_id": "cell01_seed1_brief_only",
        "cell": brief.CELLS[1],
        "arm": "brief_only",
        "seed": 1,
    }
    return task, {
        "task": task,
        "round": 3,
        "artifact_sha256": "source-result",
        "selected": {
            "request": request.model_dump(mode="json"),
            "fit": {
                "lowered_candidate_sha256": brief.public.content_sha256(
                    model.validated.candidate.model_dump(mode="json")
                ),
                "parameters": dict.fromkeys(model.parameter_names, 0.5),
                "training": {"normalized_mse": 0.1},
                "validation": {"normalized_mse": 0.2},
            },
            "origin_round": 7,
            "certificate": {},
        },
    }


def test_replay_uses_retained_parameters_and_keeps_unavailable_explicit():
    task, result = selected_result()
    before = deepcopy(result)
    job = brief.selected_job(result, task)
    assert job["parameters"] == result["selected"]["fit"]["parameters"]
    assert job["origin_round"] == 7
    assert job["saved_training_nmse"] == 0.1
    assert result == before
    result["selected"] = None
    assert brief.selected_job(result, task) is None


@pytest.mark.parametrize("change", ["round", "candidate", "parameters"])
def test_changed_fit_identity_is_rejected(change):
    task, result = selected_result()
    if change == "round":
        result["round"] = 2
    elif change == "candidate":
        result["selected"]["fit"]["lowered_candidate_sha256"] = "wrong"
    else:
        result["selected"]["fit"]["parameters"]["invented"] = 1
    with pytest.raises(ValueError):
        brief.selected_job(result, task)


def test_canonical_references_cannot_score_perturbed_models():
    row = {
        "trajectory_id": "p",
        "time": [0, 1],
        "targets": {"v01": [1, 2]},
        "auxiliaries": {k: [0, 0] for k in ("v02", "v03", "v04", "v05")},
        "external_inputs": {"u01": [0, 0]},
    }
    assert brief.canonical_case_row(row, {"cell": brief.CELLS[1]}) == row
    with pytest.raises(ValueError, match="canonical model"):
        brief.canonical_case_row(
            row, {"cell": "phase_b_anonymous_system_t1_perturbed_obfuscated_easy"}
        )
