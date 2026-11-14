"""Size-weighted sign agreement of a mechanism's response with the simulator's."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import score_term_directions as directions

TIME = np.round(np.arange(0.0, 30.05, 0.1), 10)
ZERO = np.zeros(len(TIME))


def _lobes(up, down):
    """A response that rises during t < 10 and falls during 10 <= t < 20."""
    return up * ((TIME < 10).astype(float)) - down * (
        ((TIME >= 10) & (TIME < 20)).astype(float)
    )


def test_two_lobes_flip_the_peak_sign_but_not_the_agreement():
    truth = _lobes(1.0, 0.9)
    model = _lobes(0.9, 1.0)  # right sign throughout, lobes slightly rebalanced
    assert directions.sign_agreement(truth, ZERO, model, ZERO) == pytest.approx(1.0)
    peak = lambda y: np.sign(y[np.argmax(np.abs(y))])  # noqa: E731
    assert peak(truth) != peak(model)


def test_a_model_without_response_agrees_nowhere():
    assert directions.sign_agreement(_lobes(1, 1), ZERO, ZERO, ZERO) == 0.0


def test_agreement_is_weighted_by_the_size_of_the_true_response():
    truth = _lobes(3.0, 1.0)
    model = _lobes(1.0, -1.0)  # right sign on the large lobe only
    assert directions.sign_agreement(truth, ZERO, model, ZERO) == pytest.approx(0.75)


def test_a_zero_reference_response_is_rejected():
    with pytest.raises(ValueError, match="zero"):
        directions.sign_agreement(ZERO, ZERO, _lobes(1, 1), ZERO)
