"""Basin controls use the same pipeline and independent, scoped physics checks."""

import ast
import hashlib

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.mechanism_functional import Settings
from autoformalism.rebuttal.mechanisms import evaluate_mechanisms
from autoformalism.research import basin_construction_assessment as basin
from autoformalism.schemas import CandidateModel

SURVEY = {
    "area_up": 2.0,
    "area_down": 3.0,
    "crest_up": 0.4,
    "crest_down": 0.2,
    "initial_up": 0.6,
    "warning_depth": 1.0,
}
CONTEXT = ValidationContext(
    targets=("h_down",),
    external_inputs=("inflow_up", "inflow_down"),
    fixed_covariates=tuple(SURVEY),
)


def bindings(case):
    prompt = "\n".join(
        basin.COUPLED_QUOTES
        if case == "coupled"
        else (basin.LOCAL_QUOTE, basin.ISOLATION_QUOTE)
    )
    name = f"phase_c_detention_{case}_noise0_v1"
    target, mechanism, rules = basin.contracts(
        name, prompt, hashlib.sha256(prompt.encode()).hexdigest()
    )
    return (
        {"public_specification": {"benchmark_id": name}, "independent_rules": rules},
        target,
        mechanism,
    )


def model(case="coupled", *, wrong_sign=False, inline=False):
    laws = {"o": "max(0,h_down-crest_down)"}
    rhs = {"h_down": "(inflow_down-o)/area_down"}
    if case == "coupled":
        laws["q"] = "max(0,h_up-crest_up)"
        rhs = {
            "h_up": "(inflow_up-q)/area_up",
            "h_down": f"(inflow_down{'-' if wrong_sign else '+'}q-o)/area_down",
        }
    if inline:
        rhs = {name: ast.unparse(expanded(value, laws)) for name, value in rhs.items()}
    return CandidateModel.model_validate(
        {
            "candidate_id": "control",
            "parent_candidate_id": None,
            "states": [
                {"name": n, "kind": "observed" if n == "h_down" else "latent"}
                for n in rhs
            ],
            "state_equations": [{"state": n, "rhs": value} for n, value in rhs.items()],
            "processes": []
            if inline
            else [{"name": n, "expression": value} for n, value in laws.items()],
            "observation_mappings": [{"channel": "h_down", "expression": "h_down"}],
            "initial_conditions": [
                {
                    "state": n,
                    "scope": "global",
                    "expression": "h_down" if n == "h_down" else "initial_up",
                }
                for n in rhs
            ],
        }
    )


def assess(tmp_path, candidate, case):
    cell, _, _ = bindings(case)
    train = DatasetSplit(
        name="train",
        fingerprint="synthetic-basin-training",
        trajectories=(
            Trajectory(
                trajectory_id="training",
                time=np.linspace(0, 1, 4),
                targets={"h_down": np.full(4, 0.5)},
                auxiliaries={},
                derivatives={},
                external_inputs={"inflow_up": np.ones(4), "inflow_down": np.ones(4)},
                fixed_covariates=SURVEY,
            ),
        ),
    )
    return basin.assess(
        {
            "candidate": candidate.model_dump(mode="json"),
            "parameters": {},
            "context": CONTEXT.model_dump(mode="json"),
            "initials": {},
            "semantics": "continuous_time",
        },
        cell,
        train,
        Settings(maximum_training_trajectories=1, trajectory_seconds=5),
        tmp_path,
    )


def test_coupled_named_and_inline_pass_same_fixed_five_predicates(tmp_path):
    a = assess(tmp_path / "named", model(), "coupled")
    b = assess(tmp_path / "inline", model(inline=True), "coupled")
    assert len(a) == len(b) == 5
    assert {r["id"]: r["status"] for r in a} == {r["id"]: "pass" for r in b}
    assert all(r["status"] == "pass" for r in b)
    # Reusing numerical checkpoints preserves exact results.
    assert assess(tmp_path / "named", model(), "coupled") == a


def test_transfer_counterexample_and_unknown_coordinates(tmp_path):
    findings = assess(tmp_path / "wrong", model(wrong_sign=True), "coupled")
    assert (
        next(f for f in findings if f["id"] == "transfer_cancellation")["status"]
        == "fail"
    )
    payload = model().model_dump(mode="json")
    payload["observation_mappings"][0]["expression"] = "2*h_down"
    results = assess(
        tmp_path / "mapped", CandidateModel.model_validate(payload), "coupled"
    )
    assert len(results) == 5
    assert all(f["status"] == "unresolved" for f in results)


def test_independent_control_and_initialization_paths(tmp_path):
    results = assess(tmp_path, model("independent"), "independent")
    assert len(results) == 3 and all(f["status"] == "pass" for f in results)
    payload = model().model_dump(mode="json")
    # Autonomous hidden state still carries upstream information in its initial map.
    payload["state_equations"][0]["rhs"] = "-h_up"
    finding = basin.isolation(CandidateModel.model_validate(payload), {})
    assert finding["status"] == "unresolved"
    assert finding["upstream_dependencies"] == ["initial_up"]
    assert basin.isolation(model(), {})["status"] == "unresolved"


@pytest.mark.parametrize("case", ["coupled", "independent"])
def test_identity_observation_does_not_require_a_particular_state_name(tmp_path, case):
    payload = model(case).model_dump(mode="json")
    for state in payload["states"]:
        if state["name"] == "h_down":
            state["name"] = "depth"
    for equation in payload["state_equations"]:
        equation["state"] = equation["state"].replace("h_down", "depth")
        equation["rhs"] = equation["rhs"].replace("h_down", "depth")
    for process in payload["processes"]:
        process["expression"] = process["expression"].replace("h_down", "depth")
    for initial in payload["initial_conditions"]:
        initial["state"] = initial["state"].replace("h_down", "depth")
    payload["observation_mappings"][0]["expression"] = "depth"
    result = assess(tmp_path, CandidateModel.model_validate(payload), case)
    assert all(f["status"] == "pass" for f in result)


def test_basin_admission_uses_general_graph_contract_not_equation_probes():
    _, _, spec = bindings("coupled")
    assert evaluate_mechanisms(model(), spec).graph_mechanism_compliance == 1
    # Wrong conservation sign passes path admission, then fails independent physics.
    assert (
        evaluate_mechanisms(model(wrong_sign=True), spec).graph_mechanism_compliance
        == 1
    )
    prompt = "\n".join(basin.COUPLED_QUOTES)
    brief = basin.brief(prompt, CONTEXT, spec, {})
    assert brief.scientific_context == prompt
    assert "basin_equation" not in brief.model_dump_json()
    assert "time" in {v.data_role for v in brief.public_variables}
    with pytest.raises(ValueError, match="public prompt"):
        basin.contracts(next(iter(basin.BASINS)), "changed", "a" * 64)
