from copy import deepcopy

import numpy as np
import pytest

from autoformalism.research.trajectory_function_fitting import validate_formula
from scripts import transfer_insulin_order_law as trial


def parent():
    graph = {
        "variable_types": {"I": "differential", "Z": "differential", "U": "algebraic"},
        "sources": {"I": ["Gt", "I"], "Z": ["I"], "U": ["Z"]},
        "initializers": {
            "I": {"mode": "observed_initial", "channel": "I"},
            "Z": {"mode": "map", "public_initials": ["Gt"], "state_initials": []},
        },
    }
    form = {
        "equations": {"I": "shared*Gt-old_tau*I", "Z": "shared*I", "U": "Z"},
        "initializers": {"Z": "init*Gt"},
        "parameters": {"shared": 1, "init": 2, "old_tau": 3},
    }
    return graph, form, {"shared": 3.5, "init": 4.25, "old_tau": 2}


def test_replacement_preserves_shared_parameters_and_untouched_initials():
    g, f, v = parent()
    model = deepcopy((g, f, v))
    selected = {"tau": 15, "tau_z": None, "gains": [1, 2, 0, 4]}
    new_g, new_f = trial.transplant(g, f, v, "EGP", selected)
    validate_formula(new_g, new_f)
    assert new_f["parameters"] == {
        "shared": 3.5,
        "init": 4.25,
        "order_tau": 15,
        "order_offset": 1,
        "order_gt": 2,
        "order_ins": 0,
        "order_egp": 4,
    }
    assert new_g["initializers"] == g["initializers"]
    assert new_f["initializers"] == f["initializers"]
    assert sorted(new_g["sources"]["I"]) == sorted(
        ["I", "Gt", trial.scalar.DRUG, "EGP"]
    )
    assert (g, f, v) == model


def test_rejects_wrong_original_vector_and_name_collisions():
    g, f, v = parent()
    opt = {"tau": 5, "tau_z": None, "gains": [0, 0, 0]}
    with pytest.raises(ValueError, match="vector"):
        trial.transplant(g, f, {}, "reference", opt)
    f["parameters"]["order_gt"] = 1
    v["order_gt"] = 1
    with pytest.raises(ValueError, match="shadow"):
        trial.transplant(g, f, v, "reference", opt)


def test_fold_case_has_only_supplied_fitting_events_and_matching_hash():
    raw = {
        "cases": {"x": {"evidence": {"sha256": "placeholder", "split": "train"}}},
        "parents": {"2": {"case_id": "x"}},
        "train": [{"trajectory_id": "held"}],
    }
    case = trial.fold_case(raw, 2, [{"trajectory_id": "fit"}])
    e = case["evidence"]
    assert e["trajectories"] == [{"trajectory_id": "fit"}]
    assert e["sha256"] == trial.content_hash(
        {k: v for k, v in e.items() if k != "sha256"}
    )
    assert raw["cases"]["x"]["evidence"]["sha256"] == "placeholder"


@pytest.mark.parametrize("perturb", [0, 1])
def test_replay_requires_parent_to_reproduce_archived_event_scores(
    monkeypatch, perturb
):
    g, f, v = parent()
    monkeypatch.setattr(
        trial, "_check_graph", lambda *_: {"status": "topology_screen_passed"}
    )
    monkeypatch.setattr(trial, "assess_fixed", lambda *_: {"status": "eligible"})
    monkeypatch.setattr(trial, "_rows", lambda rows, _: rows)
    names = ("Gp", "I", "U")
    curve = {
        "trajectory_id": "train_001",
        "time": [0, 1],
        "predicted": {n: [0, 2] for n in names},
        "observed": {n: [0, 1] for n in names},
    }
    monkeypatch.setattr(
        trial,
        "_check_grid",
        lambda *_, **kwargs: {"status": "scored", "curves": [curve]},
    )
    model = {
        "graph": g,
        "formula": f | {"parameters": v},
        "arm": "restoring",
        "scales": dict.fromkeys(names, 1),
    }
    existing = [{"event": "train_001", "sse": dict.fromkeys(names, 1.0 + perturb)}]
    outcome = trial.replay(model, {}, [{"trajectory_id": "train_001"}], existing)
    assert outcome["status"] == ("reference_replay_differs" if perturb else "scored")
    assert np.isclose(outcome["events"][0]["archive_max_sse_difference"], perturb)


def test_pair_summary_uses_only_matching_verified_events():
    def row(name, p, score):
        return {
            "event": name,
            "status": score,
            "pooled_nmse": p,
            "sse": {"Gp": p, "I": p, "U": p},
        }

    ref = [row("a", 2, "scored"), row("b", 10, "scored")]
    new = [row("a", 1, "scored"), row("b", 0, "unverified")]
    assert trial.paired_results(ref, new) == {
        "compared": 1,
        "lower_pooled": 1,
        "lower_by_target": {"Gp": 1, "I": 1, "U": 1},
    }
