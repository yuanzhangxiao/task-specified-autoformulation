"""Matched training-only evidence and compact target-only repair admission."""

import json
import tarfile
from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.research.affine_direction_certificate import certify
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from scripts import pointwise_repair_trial as panel
from tests.test_affine_direction_certificate import family
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_trajectory_function_families import event, packet


@pytest.fixture
def training(monkeypatch):
    rows = []
    for j, (x, y) in enumerate([([0] * 4, [10] * 4), ([1, 2, 3, 4], [10, 40, 26, 10])]):
        w = np.zeros(4)
        rows.append(
            event(
                f"train_{j}",
                np.arange(4) * 4.0,
                {"y": np.array(y), "z": w},
                {"x": np.array(x), "w": w},
            )
        )
    context = ValidationContext(targets=("y", "z"), external_inputs=("x", "w"))
    raw = packet(
        rows,
        context,
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        absolute_error_bounds=dict.fromkeys(("y", "z", "x", "w"), 0.0),
        interpolation=dict.fromkeys(("x", "w"), "linear_complete_knots"),
        initial_rules={"y": {"mode": "observed_initial"}},
    )
    obj = TrainingCompatibilityEvidence.model_validate(raw["evidence"])
    e = obj.model_dump(mode="json")
    trajectories = e.pop("trajectories")
    graph, form = family()
    graph["variable_types"]["z"] = "algebraic"
    graph["sources"]["z"] = ["w"]
    form["equations"]["z"] = "w"
    case = {
        "case_id": "test",
        "target": "y",
        "initials": [],
        "histories": ["x", "w"],
        "required_histories": [],
        "evidence": e,
        "public_brief": {
            "scientific_context": "Explain measured y.",
            "public_variables": [
                {"name": "y", "data_role": "target"},
                {"name": "z", "data_role": "target"},
                {"name": "x", "data_role": "external_input"},
                {"name": "w", "data_role": "external_input"},
            ],
            "requirements": [],
        },
    }
    w = certify(graph, form, "y", obj, tolerance=1.0)
    parent = {
        "formula": form,
        "graph": graph,
        "case_id": "test",
        "witness": w,
        "dataset": panel.DATASET,
        "archive_candidate_sha256": "0" * 64,
    }
    data = {
        "source": panel.SOURCE,
        "parents": {str(i): deepcopy(parent) for i in panel.PARENTS},
        "cases": {"test": case},
        "train": trajectories,
    }
    monkeypatch.setattr(panel, "TARGET", "y")
    monkeypatch.setattr(panel, "INTAKE", panel.content_hash(data))
    monkeypatch.setattr(panel, "runtime_source_hash", lambda: "test")
    return data


def patch(rhs, **params):
    return {
        "rhs": rhs,
        "new_parameters": [{"name": k, "initial": v} for k, v in params.items()],
    }


def test_matched_requests_reveal_only_the_witness_and_use_equal_replicate_seeds(
    training,
):
    for i in panel.PARENTS:
        base = panel.request(training, i, "control")
        witness = panel.request(training, i, "witness")
        items = witness.pop("measured_witness")
        assert base.pop("measured_witness") is None and base == witness
        assert items["reference"]["event"] == "train_0"
        assert (
            items["rise"]["event"] == "train_1" and items["fall"]["event"] == "train_1"
        )
        assert (
            not {"repair_options", "parameters", "validation", "score"} & items.keys()
        )
        assert "train_" not in json.dumps(base)
        assert "start" not in base
    assert len(panel.slots()) == len(set(panel.slots())) == 18
    for i in range(0, 18, 2):
        first, second = panel.slots()[i : i + 2]
        assert first[:2] == second[:2] and {first[2], second[2]} == set(
            panel.CONDITIONS
        )
    assert panel.slots()[0][2] != panel.slots()[2][2]


def test_proof_rechecked_and_target_only_patch_preserves_every_other_field(training):
    p = training["parents"]["2"]
    g, f = p["graph"], p["formula"]
    g["variable_types"]["z"] = "algebraic"
    g["sources"]["z"] = ["w"]
    f["equations"]["z"] = "a*w"
    original = deepcopy(p)
    changed = panel.assess(training, 2, patch("dy/dt = k*x - y(t)", k=2))
    assert changed["status"] == "eligible"
    assert changed["pointwise_outcome"] == "no_conflict_proved"
    assert changed["ordered_checks"]["y"]["status"] == "unverified"
    assert changed["trajectory_accuracy"] == "unmeasured"
    assert changed["formula"]["equations"]["z"] == "a*w"
    assert changed["formula"]["parameters"] == {"a": 3, "k": 2}
    assert changed["removed_unused_parent_parameters"] == ["b"]
    assert changed["source_change"] == {"before": ["x"], "after": ["x", "y"]}
    assert changed["graph_dict"]["initializers"] == g["initializers"]
    assert changed["formula"]["initializers"] == f["initializers"]
    assert p == original
    assert changed["normalizations"][0]["after"] == "k*x - y(t)"
    assert changed["normalizations"][1]["after"] == "k * x - y"
    remains = panel.assess(training, 2, patch("a+b*x+x"))
    assert remains["pointwise_outcome"] == "conflict_proved"
    assert remains["status"] == "pointwise_conflict"
    assert "repair_feedback" in remains
    assert "request_topology_change" not in str(remains["repair_feedback"])


@pytest.mark.parametrize(
    "rhs,new",
    [
        ("dy/dt = a+b*x", {}),
        ("c+d*x", {"c": 5, "d": 6}),
        ("a + b*x + 0", {}),
        ("+a+b*x*1", {}),
    ],
)
def test_provable_noops_are_rejected_with_authored_text(training, rhs, new):
    out = panel.assess(training, 2, patch(rhs, **new))
    assert out["status"] == "no_scientific_change" and out["authored_rhs"] == rhs


def test_use_of_shared_coefficient_is_not_treated_as_arbitrary_rename(training):
    parent = training["parents"]["2"]
    parent["formula"]["equations"]["z"] = "a*w"
    parent["graph"]["variable_types"]["z"] = "algebraic"
    parent["graph"]["sources"]["z"] = ["w"]
    result = panel.assess(training, 2, patch("k+b*x", k=2))
    assert result["status"] == "pointwise_conflict"


def test_unclear_names_bad_graph_and_declaration_errors_stay_rejected(training):
    with pytest.raises(ValueError, match="new coefficient"):
        panel.assess(training, 2, patch("a+x", a=2))
    with pytest.raises(ValueError, match="absent"):
        panel.assess(training, 2, patch("a+x", k=2))
    with pytest.raises(ValueError, match="label"):
        panel.assess(training, 2, patch("dz/dt = x"))
    with pytest.raises(ValidationError):
        panel.assess(training, 2, dict(patch("x"), initializers={"y": "0"}))
    assert panel.assess(training, 2, patch("ghost+x"))["status"] == "invalid_formula"
    assert panel.assess(training, 2, patch("y"))["status"] not in (
        "eligible",
        "pointwise_conflict",
    )
    with pytest.raises(ModelValidationError) as exc:
        panel.assess(training, 2, patch("x.__class__"))
    assert "ATTRIBUTE" in str(exc.value).upper()


def test_sealed_call_budget_order_interruption_resume_and_tamper(
    training, tmp_path, monkeypatch
):
    monkeypatch.setattr(panel, "read_source", lambda p: training)
    root = tmp_path / "panel"
    plan = panel.prepare(root, tmp_path / "source")
    assert panel.prepare(root, tmp_path / "source") == plan
    assert plan["maximum_generation_requests"] == 18 and plan["correction_rounds"] == 0
    assert panel.status(root)["status"] == "incomplete"
    seen = []

    def provider(url, body, timeout):
        user = json.loads(body["messages"][1]["content"])
        seen.append((user["measured_witness"] is not None, body["seed"]))
        return reply(patch("a+b*x-y" if user["measured_witness"] else "a+b*x+x"))

    orig = panel.pilot._ask

    def interrupter(*args, **kwargs):
        if len(seen) == 4:
            raise KeyboardInterrupt
        return orig(*args, **kwargs)

    monkeypatch.setattr(panel.pilot, "_ask", interrupter)
    with pytest.raises(KeyboardInterrupt):
        panel.run(
            root, "http://localhost", transport=provider, token_transport=tokenizer
        )
    assert panel.status(root)["physical_requests"] == 4
    monkeypatch.setattr(panel.pilot, "_ask", orig)
    result = panel.run(
        root, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert result["condition_statuses"] == {
        "witness": {"eligible": 9},
        "control": {"pointwise_conflict": 9},
    }
    assert result["physical_requests"] == len(seen) == 18
    for a, b in zip(seen[::2], seen[1::2], strict=True):
        assert a[0] != b[0] and a[1] == b[1]
    assert (
        panel.run(
            root, "http://localhost", transport=lambda *a: pytest.fail("new call")
        )
        == result
    )
    assert len(panel.verified_records(root)) == 18
    out = panel.inspect(root)
    with tarfile.open(out["archive"]) as tar:
        assert "plan.json" in tar.getnames()
        assert "candidates/017.json" in tar.getnames()
        assert not any("validation" in n for n in tar.getnames())
    (receipt,) = list((root / "steps/slot-000/calls").glob("*.json"))
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match=r"physical|receipt"):
        panel.verified_records(root)
