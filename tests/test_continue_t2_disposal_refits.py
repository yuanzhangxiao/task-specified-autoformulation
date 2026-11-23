import pytest

from scripts import continue_t2_disposal_refits as trial


def test_named_interpolation_preserves_frozen_and_exact_endpoint_parameters():
    base = {"time": 5.49, "a": -2.4}
    end = {"a": 1.7, "time": 5.49}
    assert trial.interpolate(base, end, 0.0) == base
    assert trial.interpolate(base, end, 1.0) == end
    step = trial.interpolate(base, end, 0.25)
    assert step["time"] == base["time"]
    assert step["a"] == pytest.approx(-2.4 + 0.25 * 4.1)
    assert trial.FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1)
    with pytest.raises(ValueError):
        trial.interpolate(base, {"a": 1.0}, 0.25)


def test_chooses_only_feasible_steps_and_retains_earlier_tie():
    def row(frac, score, admissible=True):
        return {
            "fraction": frac,
            "training_nmse": score,
            "domain": {"eligible": admissible},
        }

    trials = [
        row(0.0, 0.13),
        row(0.5, 0.12),
        row(0.75, 0.12 - 1e-12),
        row(1, 0.11, False),
    ]
    assert trial.winner(trials) == trials[1]
    with pytest.raises(ValueError, match="incumbent"):
        trial.winner(trials[1:])
