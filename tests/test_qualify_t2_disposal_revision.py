from copy import deepcopy

import pytest

from scripts import qualify_t2_disposal_revision as trial
from tests.test_validate_positive_insulin_transfer import toy


def test_only_disposal_equation_and_exclusive_coefficients_can_change():
    _, model = toy()
    model["graph"]["sources"]["U"] = ["I_act", "Gt"]
    child = deepcopy(model)
    child["formula"]["equations"]["U"] = "I_act*k*Gt"
    child["formula"]["parameters"]["k"] = 0.1
    trial.check_patch(model, child)
    child["formula"]["parameters"]["pos_tau"] *= 2
    with pytest.raises(ValueError, match="parameter changed"):
        trial.check_patch(model, child)
    child = deepcopy(model)
    child["graph"]["sources"]["U"] = ["illegal"]
    with pytest.raises(ValueError, match="protected"):
        trial.check_patch(model, child)


def event(name, nmse, concentration=20.0):
    return {
        "event": name,
        "per_target_nmse": dict.fromkeys(trial.TARGETS, nmse),
        "pooled_nmse": nmse,
        "status": "scored",
        "predicted": {"I": [concentration]},
    }


def test_pairs_include_ties_and_require_invariant_insulin_and_identical_cohort():
    before = [event("train_000", 0.2), event("train_001", 0.1)]
    after = [event("train_000", 0.1), event("train_001", 0.10000000001)]
    paired = trial.paired(before, after)
    assert paired["counts"]["pooled"] == {"improved": 1, "tied": 1, "worse": 0}
    with pytest.raises(ValueError, match="matched"):
        trial.paired(before, after[::-1])
    after[0]["predicted"]["I"][0] = 17.0
    with pytest.raises(ArithmeticError, match="Frozen insulin"):
        trial.paired(before, after)


def test_training_screen_comparison_rejects_missing_or_different_event_scores():
    result = {"events": [{"event": "train_003", "sse": {"U": 15.0}}]}
    trial.check_screen(result, {"train_003": 15.00001})
    for expected in ({}, {"train_003": 15.25}):
        with pytest.raises(ValueError, match="screening"):
            trial.check_screen(result, expected)


def test_missing_selection_never_opens_development(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trial.intake.original,
        "read_validation",
        lambda *_: pytest.fail("Opened development before selection"),
    )
    with pytest.raises(FileNotFoundError):
        trial.run(tmp_path / "missing", tmp_path, tmp_path, tmp_path / "new")


def test_disposal_contribution_checks_supplied_on_each_grid_point():
    event = {"predicted": {"U": [2.0, 2.5]}}
    row = {"auxiliaries": {"Uii": [2.5, 2.0]}}
    assert trial.disposal_contribution(event, row) == {
        "min_U_minus_Uii": -0.5,
        "negative_U_minus_Uii_samples": 1,
    }
    with pytest.raises(ValueError, match="Unmatched"):
        trial.disposal_contribution(event, {"auxiliaries": {"Uii": [1.0]}})
