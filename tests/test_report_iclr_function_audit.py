"""Check overlap, snapshot identity, and missing-data denominators."""

from copy import deepcopy

import pytest

from scripts.report_iclr_function_audit import overlap_rows, summary


def inputs():
    common = {
        "run_id": "case:seed0",
        "snapshot": "final",
        "model_id": "fit1",
        "available": True,
    }
    function = {
        **common,
        "retained_equation_revision_any_roster_count": 1,
        "retained_shape_revision_any_roster_count": 1,
        "retained_equation_revision_count": 0,
        "retained_shape_revision_count": 0,
    }
    latent = {
        **common,
        "retained_pure_augmentation_count": 1,
        "numerically_cited_addition_count": 1,
    }
    event = {**common, "round": 2, "nonlinear_shape_revision_retained": True}
    state = {**common, "retained_pure_augmentation": True, "birth": {"round": 2}}
    return {"runs": [function], "events": [event]}, {
        "runs": [latent],
        "states": [state],
    }


def test_joint_revision_enters_both_categories_and_same_visit_overlap():
    functions, latents = inputs()
    row = overlap_rows(functions, latents)[0]
    assert row["nonlinear_form_revision_count"] == 1
    assert row["fixed_roster_nonlinear_form_revision_count"] == 0
    assert row["both_latent_and_nonlinear_form_revision"] == 1
    assert row["same_visit_latent_and_nonlinear_form_revision"] == 1


def test_same_endpoint_overlap_does_not_imply_same_visit():
    functions, latents = inputs()
    latents["states"][0]["birth"]["round"] = 3
    row = overlap_rows(functions, latents)[0]
    assert row["both_latent_and_nonlinear_form_revision"] == 1
    assert row["same_visit_latent_and_nonlinear_form_revision"] == 0


def test_unavailable_model_is_not_a_zero_revision_success():
    functions, latents = inputs()
    missing = {"run_id": "case:seed1", "snapshot": "final", "available": False}
    functions["runs"].append(missing)
    result = summary(overlap_rows(functions, latents))
    assert result["assigned"] == 2 and result["assessed"] == 1
    assert result["nonlinear_form_revision_count"]["histogram"] == {1: 1}


@pytest.mark.parametrize("failure", ["duplicate", "different_fit", "missing"])
def test_inconsistent_or_duplicated_endpoint_evidence_fails_closed(failure):
    functions, latents = inputs()
    if failure == "duplicate":
        latents["runs"].append(deepcopy(latents["runs"][0]))
    elif failure == "different_fit":
        latents["runs"][0]["model_id"] = "another-fit"
    else:
        latents["runs"] = []
    with pytest.raises(ValueError):
        overlap_rows(functions, latents)
