"""Graded wiring similarity must be independent of arbitrary latent-state names."""

import json
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import iclr_seed_wiring as audit
from scripts.iclr_submission_evidence import PLAN
from tests.test_iclr_historical_followup import endpoint


def model():
    row = endpoint()
    return {**row, "lowered_context": row["context"]}


def test_renamed_state_and_inlined_process_preserve_direct_wiring():
    first = model()
    second = deepcopy(first)
    for state in second["candidate"]["states"]:
        if state["name"] == "x":
            state["name"] = "delay"
    for equation in second["candidate"]["state_equations"]:
        if equation["state"] == "y":
            equation["rhs"] = "k*delay+u"
        elif equation["state"] == "x":
            equation.update(state="delay", rhs="-delay+v")
    for rule in second["candidate"]["initial_conditions"]:
        if rule["state"] == "x":
            rule["state"] = "delay"
    second["candidate"]["processes"] = []
    a, b = (audit.fitted_wiring(row) for row in (first, second))
    assert a.states == {"state:x": "latent", "state:y": "observed"}
    assert audit.best_wiring(a, b) == (1.0, 1)


def test_zero_gain_drops_only_the_unreachable_state():
    row = model()
    row["parameters"]["k"] = 0.0
    g = audit.fitted_wiring(row)
    assert g.edges == {("public:u", "state:y"), ("state:y", "output:y")}
    assert audit.best_wiring(audit.fitted_wiring(model()), g)[0] == pytest.approx(2 / 5)


def test_public_ports_and_state_types_stay_anchored():
    def wire(state, kind, forcing):
        return audit.Wiring({state: kind}, {(forcing, state), (state, "output:y")})

    a = wire("state:X", "latent", "public:u")
    b = wire("state:Y", "latent", "public:w")
    assert audit.best_wiring(a, b)[0] == pytest.approx(1 / 3)
    assert audit.best_wiring(a, wire("state:X", "observed", "public:u")) == (0, 1)
    assert audit.best_wiring(audit.Wiring({}, set()), audit.Wiring({}, set())) == (
        None,
        1,
    )


def test_larger_graph_can_match_by_renaming_either_state():
    a = audit.Wiring({"state:a": "latent"}, {("state:a", "output:y")})
    b = audit.Wiring(
        {"state:b": "latent", "state:c": "latent"},
        {("state:c", "output:y"), ("public:u", "state:b")},
    )
    assert audit.best_wiring(a, b) == (0.5, 2)
    assert audit.best_wiring(b, a) == (0.5, 2)


def test_corrupt_algebraic_cycle_or_unknown_dependency_fails():
    row = model()
    row["candidate"]["processes"][0]["expression"] = "sink"
    with pytest.raises(ValueError, match="cycle"):
        audit.fitted_wiring(row)
    row["candidate"]["processes"][0]["expression"] = "unregistered"
    with pytest.raises(ValueError, match="Undeclared"):
        audit.fitted_wiring(row)


def test_missing_model_and_unresolved_requirement_remain_distinct():
    base = {
        "cohort": "a",
        "cell": "cell",
        "prompt": "full",
        "arm": "all",
        "status": "fit",
        "public": ("train", "val"),
        "signed_topology": "s",
        "functions": "f",
        "active_dimension": 1,
        "active_latents": [],
        "reach": ["u→y"],
        "wiring": audit.Wiring({}, {("public:u", "output:y")}),
    }
    a = dict(base, seed=0, task="a", validation=0.5, roles={"r": "unresolved"})
    b = dict(base, seed=1, task="b", validation=0.4, roles={"r": "fail"})
    (pair,) = audit.paired([a, b])
    assert pair["same_test_outcomes"] is False
    assert pair["absolute_compliance_gap_pp"] == 0
    del b["wiring"]
    (pair,) = audit.paired([a, b])
    assert not pair["comparable"] and pair["absolute_nmse_gap"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Missing planned seed"):
        audit.paired([a])
    with pytest.raises(ValueError, match="Duplicate seed"):
        audit.paired([a, a])


def test_offline_cluster_collection_scopes_rounds_and_lists_missing(tmp_path):
    collector = Path(__file__).parents[1] / "scripts/hpc/collect_iclr_example_traces.sh"
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cmd = ["bash", str(collector), str(campaign), str(tmp_path)]
    plan = campaign / "plan.json"
    plan.write_text(json.dumps({"artifact_sha256": "other"}))
    assert subprocess.run(cmd, capture_output=True, check=False).returncode != 0
    plan.write_text(json.dumps({"artifact_sha256": PLAN}))
    assert subprocess.run(cmd, capture_output=True, check=False).returncode != 0
    task = campaign / "results" / "cell08_seed0_full_c1v1s1"
    for name in ("round_00", "round_13"):
        (task / name).mkdir(parents=True)
        (task / name / "critic.json").write_text("{}")
    receipt = json.loads(subprocess.check_output(cmd, text=True))
    assert receipt["receipts"] == 1 and receipt["missing"] == 103
    with tarfile.open(receipt["archive"]) as archive:
        manifest = json.load(archive.extractfile("TRACE_INDEX.json"))
        expected = {
            "results/cell08_seed0_full_c1v1s1/round_00/critic.json",
            "plan.json",
        }
        assert set(manifest["files"]) == expected
        assert set(archive.getnames()) == expected | {"TRACE_INDEX.json"}
