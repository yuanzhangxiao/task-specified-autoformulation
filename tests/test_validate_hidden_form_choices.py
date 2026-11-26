"""Verify independent validation qualification, split and endpoint identity."""

from copy import deepcopy

import pytest

from autoformalism.staged_topology import content_hash
from scripts import validate_hidden_form_choices as validation

GRAPH = {
    "variable_types": {"y": "differential"},
    "sources": {"y": ["u"]},
    "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
}
FORM = {"equations": {"y": "k*u"}, "initializers": {}, "parameters": {"k": 0.0}}


def score(initial=2.0, scale=1.0, gain=1.0, event="val_001", form=None):
    fit = {
        "graph_sha256": content_hash(GRAPH),
        "formula_sha256": content_hash(FORM),
        "status": "fitted",
        "parameters": {"k": gain},
        "scales": {"y": scale},
    }
    row = {
        "trajectory_id": event,
        "time": [0.0, 1.0, 2.0],
        "targets": {"y": [initial, initial, initial]},
        "inputs": {"u": [1.0, 1.0, 1.0]},
        "auxiliaries": {},
        "covariates": {},
    }
    return validation._score(GRAPH, FORM if form is None else form, fit, row, ["y"])


def test_external_uses_fit_scale_and_post_initial_without_refitting():
    got = score(scale=2.0)
    assert got["status"] == "scored" and got["nmse"] == pytest.approx((1 + 4) / 8)
    neg = score(gain=-3.0)
    assert neg["status"] == "negative_forecast" and neg["nmse"] is None
    assert neg["negative_samples"] == {"y": 2}
    assert neg["minimum_prediction"]["y"] == pytest.approx(-4)


def test_external_rejects_changed_fit_or_nonvalidation_rows():
    with pytest.raises(ValueError, match="val"):
        score(event="test_001")
    form = deepcopy(FORM)
    form["parameters"]["k"] = 1.0
    with pytest.raises(ValueError, match="identity"):
        score(form=form)


def test_external_requires_protocol_before_data_access(tmp_path):
    with pytest.raises(ValueError, match="Prepare"):
        validation.evaluate(tmp_path, tmp_path)
