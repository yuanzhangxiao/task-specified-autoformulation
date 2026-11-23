import pytest

from scripts import refit_t2_disposal_revision as run


def test_chooses_training_score_with_domain_check_and_retains_ties():
    def proposal(score, eligible):
        return {"training_nmse": score, "domain": {"eligible": eligible}}

    attempts = {
        "incumbent": proposal(0.12, True),
        "warm": proposal(0.08, False),
        "displaced": proposal(0.12 - 1e-12, True),
    }
    assert run.choose(attempts) == "incumbent"
    attempts["displaced"] = proposal(0.1, True)
    assert run.choose(attempts) == "displaced"
    attempts["incumbent"] = proposal(0.12, False)
    with pytest.raises(ValueError, match="incumbent"):
        run.choose(attempts)


def test_domain_uses_both_action_gains_and_conditional_disposal():
    event = {
        "event": "train_000",
        "predicted": {"U": [1.2, 1.5]},
        "negative_samples": {"U": 0},
    }
    row = {"trajectory_id": "train_000", "auxiliaries": {"Uii": [1.0, 1.0]}}
    result = {"events": [event], "summary": {"pooled_nmse": 0.12, "failed_events": 0}}
    p = {"afcoef10": 0.2, "afcoef13": 0.3}
    assert run.domain(result, [row], p)["eligible"]
    event["predicted"]["U"][0] = 0.4
    assert run.domain(result, [row], p)["reasons"] == [
        "negative insulin-dependent disposal"
    ]
    p["afcoef13"] = -1.0
    assert len(run.domain(result, [row], p)["reasons"]) == 2
    result["summary"]["failed_events"] = 1
    assert run.domain(result, [row], p)["reasons"] == ["incomplete training rollout"]


def test_absent_training_input_never_opens_development(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run.intake.original,
        "read_validation",
        lambda *_: pytest.fail("Opened development"),
    )
    with pytest.raises(FileNotFoundError):
        run.run(tmp_path, tmp_path / "output", False, 1)
