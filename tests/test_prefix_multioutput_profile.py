"""Private affine coefficients, scoped multi-output basis and strict-face profiling."""

import re
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_affine
from autoformalism.fitting.prefix_initialization import NumericalPolicy
from autoformalism.fitting.prefix_multioutput_profile import (
    ProfileUnavailable,
    _project,
    make_oracle,
)
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory


def profile_case():
    """Renamed four-state control with three outputs and two scoped latent roots."""
    real = ("p1", "p2", "c1")
    positive = ("tau_g", "tau_z", "tau_z2", "tau_h", "tau_h2", "tau9")
    nonnegative = (
        "a",
        "b",
        "act",
        "p0",
        "g0",
        "g5",
        "c0",
        "g1",
        "g9",
        "gq1",
        "gq2",
        "gm1",
        "gm2",
        "gv",
        "gz",
        "gh",
    )
    parameters = [
        {"name": name, "scope": "global", "role": role}
        for names, role in (
            (real, "coefficient"),
            (positive, "time_constant"),
            (nonnegative, "nonnegative_coefficient"),
        )
        for name in names
    ]
    guesses = {p["name"]: 0.2 for p in parameters}
    guesses.update(dict.fromkeys(positive, 1.7))
    guesses.update(c0=1.0, p0=0.5, a=0.4, b=0.3, act=0.3)
    problem = PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "renamed_affine_profile_control",
                "parent_candidate_id": None,
                "states": [
                    {"name": s, "kind": "latent" if s == "h" else "observed"}
                    for s in ("x", "y", "z", "h")
                ],
                "parameters": parameters,
                "processes": [],
                "state_equations": [
                    {"state": "x", "rhs": "a*m+b*e-z/tau_g"},
                    {
                        "state": "y",
                        "rhs": (
                            "q+gq1*q+gq2*q+(p0+g0*p1+g5*p2)*m"
                            "-(c0+g1*c1+g9/tau9)*y+(gm1+gm2)*h"
                        ),
                    },
                    {"state": "z", "rhs": "v+gv*v+h-z/tau_z-gz*z/tau_z2"},
                    {"state": "h", "rhs": "act*y-h/tau_h-gh*h/tau_h2"},
                ],
                "observation_mappings": [
                    {"channel": s, "expression": s} for s in ("x", "y", "z")
                ],
                "initial_conditions": [
                    {"state": s, "scope": "global", "expression": s}
                    for s in ("x", "y", "z")
                ]
                + [{"state": "h", "scope": "global", "fixed_value": 0}],
                "constraints": [],
            },
            "context": {
                "targets": ["x", "y", "z"],
                "external_inputs": ["m", "q", "e", "v"],
            },
            "graph": {
                "rules": {
                    "h": {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory",
                            "role": "nonnegative_coefficient",
                            "guess": 0.35,
                        }
                    }
                }
            },
            "parameter_guesses": guesses,
        }
    )
    time = np.linspace(0, 2, 9)
    rows = tuple(
        PublicTrajectory(
            trajectory_id=f"row{i}",
            time=tuple(time),
            targets={
                "x": tuple(2 + i + time / 5),
                "y": tuple(0.7 + i / 5 + time / 3),
                "z": tuple(1.1 + i / 4 + time / 7),
            },
            external_inputs={
                "m": tuple(0.3 + time**2 * (i + 1) / 3),
                "q": tuple(0.2 + np.sin(time + i) ** 2),
                "e": tuple(0.5 + np.cos(time * 2 + i) ** 2),
                "v": tuple(0.8 + time / (1 + i)),
            },
        )
        for i in range(2)
    )
    training = PublicSplit(name="train", fingerprint="multioutput-synthetic", rows=rows)
    numerical = NumericalPolicy()
    original = prefix_affine.TrainingOracle(problem, training, numerical)
    retained = original.start | {"trajectory:row1:init_h_value": 0.75}
    point = original.vector(retained)
    generated = []
    for row, public in zip(original.rows, training.rows, strict=True):
        values, _ = original._propagation.rollout(
            row,
            original.local_vector(point, row),
            monotonic() + 10,
            jacobian=False,
            cache={},
        )
        generated.append(
            public.model_copy(
                update={
                    "targets": {
                        name: tuple(values[:, j])
                        for j, name in enumerate(original.system.channels)
                    }
                }
            )
        )
    return (
        problem,
        training.model_copy(update={"rows": tuple(generated)}),
        numerical,
        retained,
    )


@pytest.fixture
def case():
    return profile_case()


def test_generic_certificate_preserves_original_model_domains_and_scopes(case):
    problem, _training, _numerical, retained = case
    before = problem.model_dump_json()
    original, oracle, profile = make_oracle(*case)
    assert len(oracle.names) == 13
    assert len(profile.inner) == 7 and len(profile.outer) == 6
    assert len(oracle.reduction.blocks) == 5
    assert problem.model_dump_json() == before
    for name in ("coefficient:y:q", "coefficient:z:v"):
        assert oracle.lower[oracle.names.index(name)] == 1
    assert oracle.lower[oracle.names.index("coefficient:y:h")] == 0
    for name in ("coefficient:y:y", "coefficient:y:m"):
        assert oracle.lower[oracle.names.index(name)] == -np.inf
        assert oracle.upper[oracle.names.index(name)] == np.inf
    point = oracle.reduction.reduce(original.vector(retained))
    for sign in (-5, 5):
        values = oracle.parameters(point) | {
            "coefficient:y:y": sign,
            "coefficient:y:m": sign,
        }
        vector = oracle.vector(values)
        lifted = oracle.reduction.lift(vector)
        original.vector(original.parameters(lifted))
        assert oracle.reduction.reduce(lifted) == pytest.approx(vector, abs=1e-14)
    assert [n for n in oracle.names if n.startswith("trajectory:")] == [
        "trajectory:row0:init_h_value",
        "trajectory:row1:init_h_value",
    ]
    assert profile.audit["target_scales"] == original.scales
    assert [r["sample_count"] for r in profile.audit["training_rows"]] == [9, 9]


def test_basis_matches_all_weighted_outputs_original_chain_rule_and_replay(case):
    original, oracle, profile = make_oracle(*case)
    point = oracle.vector(oracle.start)
    arrays = profile.design(point, monotonic() + 20)
    residual, jacobian = oracle.evaluate(point, monotonic() + 20)
    expected = arrays["design"] @ point[list(profile.inner)] + arrays["offset"]
    assert expected.shape == (54,)
    assert expected == pytest.approx(residual, abs=2e-12)
    outer = (
        np.einsum("nkp,k->np", arrays["d_design"], point[list(profile.inner)])
        + arrays["d_offset"]
    )
    assert outer == pytest.approx(jacobian[:, profile.outer], abs=2e-11)
    assert arrays["design"] == pytest.approx(jacobian[:, profile.inner], abs=2e-11)
    lifted = oracle.reduction.lift(point)
    replay, direct = original.evaluate(lifted, monotonic() + 20, method="Radau")
    assert expected == pytest.approx(replay, abs=2e-8)
    assert jacobian == pytest.approx(
        direct @ oracle.reduction.lift_jacobian(point), abs=2e-8
    )
    # Each root column is nonzero only in its own row, across the three outputs.
    root = profile.inner.index(oracle.names.index("trajectory:row0:init_h_value"))
    assert np.any(arrays["design"][:27, root])
    assert np.all(arrays["design"][27:, root] == 0)


def test_conditional_optimum_derivative_and_explicit_original_lift(case):
    original, oracle, profile = make_oracle(*case)
    point = oracle.vector(oracle.start)
    point[profile.outer[0]] *= 1.08
    fitted, residual, derivative, audit = profile.evaluate(point, monotonic() + 20)
    assert audit["derivative_available"]
    assert audit["design_rank"] == len(profile.inner)
    assert audit["normalized_kkt_residual"] <= audit["kkt_tolerance"]
    assert np.mean(residual**2) <= np.mean(
        oracle.evaluate(point, monotonic() + 20)[0] ** 2
    )
    replay = original.evaluate(
        oracle.reduction.lift(fitted), monotonic() + 20, jacobian=False, method="Radau"
    )[0]
    assert residual == pytest.approx(replay, abs=3e-8)
    direction = np.zeros(len(point))
    direction[list(profile.outer)] = np.arange(1, 7) / 10
    step = 2e-5
    plus = profile.evaluate(point + step * direction, monotonic() + 20, jacobian=False)[
        1
    ]
    minus = profile.evaluate(
        point - step * direction, monotonic() + 20, jacobian=False
    )[1]
    assert derivative @ direction[list(profile.outer)] == pytest.approx(
        (plus - minus) / (2 * step), abs=2e-7
    )


@pytest.mark.parametrize(
    "change",
    [
        "shared",
        "initializer",
        "bounded",
        "constraint",
        "observation",
        "nonlinear",
        "unsafe_dead",
    ],
)
def test_unsupported_families_fail_closed_without_repairing_model(case, change):
    problem, training, numerical, retained = case
    payload = problem.model_dump(mode="json")
    model = payload["base_candidate"]
    if change == "shared":
        model["state_equations"][0]["rhs"] += "+p1*m"
    elif change == "initializer":
        payload["graph"]["rules"]["h"] = {
            "initial": {"mode": "map", "expression": "p1", "parameters": []}
        }
    elif change == "bounded":
        model["parameters"][0]["bounds"] = {"lower": -1, "upper": 1}
    elif change == "constraint":
        model["constraints"] = [{"subject": "p1", "kind": "nonnegative"}]
    elif change == "observation":
        model["observation_mappings"][0]["expression"] = "x+p1"
    elif change == "nonlinear":
        model["state_equations"][0]["rhs"] += "+x*y"
    else:
        model["processes"] = [{"name": "unused", "expression": "1/(p1-1)"}]
    with pytest.raises(ValueError):
        make_oracle(
            PrefixFitProblem.model_validate(payload), training, numerical, retained
        )


def test_nontraining_and_offset_violations_rejected(case):
    problem, training, numerical, retained = case
    with pytest.raises(ValueError, match="training only"):
        make_oracle(
            problem,
            training.model_copy(update={"name": "validation"}),
            numerical,
            retained,
        )
    _, oracle, _ = make_oracle(*case)
    with pytest.raises(ValueError, match="domain"):
        oracle.vector(oracle.start | {"coefficient:y:q": 0.99})


def test_open_rates_remain_hashable_before_nonrepresentable_lift(case):
    _, oracle, _ = make_oracle(*case)
    rate = next(n for n in oracle.names if n.startswith("effective_decay:"))
    point = oracle.vector(oracle.start | {rate: np.nextafter(0.0, 1.0)})
    assert oracle.parameters(point)[rate] > 0
    with pytest.raises(NumericalInvalidTrial):
        oracle.reduction.lift(point)
    with pytest.raises(ValueError, match="strictly positive"):
        oracle.vector(oracle.start | {rate: 0.0})


def test_time_dependent_observation_does_not_pass_identity_check(case):
    problem, training, numerical, retained = case
    payload = problem.model_dump(mode="json")
    payload["base_candidate"]["observation_mappings"][0]["expression"] = "x+t"
    payload["graph"]["rules"]["x"] = {
        "initial": {"mode": "known", "value": 2, "justification": "preparation"}
    }
    with pytest.raises(ValueError, match="identity outputs"):
        make_oracle(
            PrefixFitProblem.model_validate(payload), training, numerical, retained
        )


def test_reserved_constant_feature_is_rejected(case):
    problem, training, numerical, retained = case
    payload = problem.model_dump(mode="json")
    for equation in payload["base_candidate"]["state_equations"]:
        equation["rhs"] = re.sub(r"\bm\b", "constant", equation["rhs"])
    payload["context"]["external_inputs"] = [
        "constant" if name == "m" else name
        for name in payload["context"]["external_inputs"]
    ]
    rows = tuple(
        row.model_copy(
            update={
                "external_inputs": {
                    "constant" if name == "m" else name: values
                    for name, values in row.external_inputs.items()
                }
            }
        )
        for row in training.rows
    )
    with pytest.raises(ValueError, match="reserved constant feature"):
        make_oracle(
            PrefixFitProblem.model_validate(payload),
            training.model_copy(update={"rows": rows}),
            numerical,
            retained,
        )


@pytest.mark.parametrize("role", ["coefficient", "nonnegative_coefficient"])
def test_open_rate_aggregation_requires_unrestricted_pivot(case, role):
    problem, training, numerical, retained = case
    payload = problem.model_dump(mode="json")
    model = payload["base_candidate"]
    model["parameters"].append({"name": "z_extra", "scope": "global", "role": role})
    model["state_equations"][2]["rhs"] += "-z_extra*z"
    payload["parameter_guesses"]["z_extra"] = 0.2
    arguments = (
        PrefixFitProblem.model_validate(payload),
        training,
        numerical,
        retained | {"z_extra": 0.2},
    )
    if role == "nonnegative_coefficient":
        with pytest.raises(ValueError, match="no certified full-domain section"):
            make_oracle(*arguments)
    else:
        original, oracle, _ = make_oracle(*arguments)
        block = next(
            b for b in oracle.reduction.blocks if b.coordinate == "coefficient:z:z"
        )
        frozen_rate = next(n for n in block.frozen if n.startswith("effective_decay:"))
        assert block.frozen[frozen_rate] == 1.0
        point = oracle.vector(oracle.start | {"coefficient:z:z": 3.0})
        lifted = oracle.reduction.lift(point)
        original.vector(original.parameters(lifted))
        assert oracle.reduction.reduce(lifted) == pytest.approx(point)


def test_rank_deficiency_keeps_inner_fit_but_rejects_derivative():
    a = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    kwargs = {
        "design": a,
        "offset": -a[:, 0],
        "lower": np.zeros(2),
        "upper": np.full(2, np.inf),
        "deadline": monotonic() + 5,
    }
    fitted, residual, derivative, audit = _project(
        **kwargs, d_design=None, d_offset=None
    )
    assert fitted.sum() == pytest.approx(1)
    assert np.linalg.norm(residual) < 1e-12 and derivative is None
    assert audit["design_rank"] == 1
    with pytest.raises(ProfileUnavailable, match="rank-deficient") as error:
        _project(**kwargs, d_design=np.zeros((3, 2, 1)), d_offset=np.zeros((3, 1)))
    assert error.value.audit["normalized_kkt_residual"] < 1e-12


def test_strict_active_face_derivative_and_boundary_ambiguity():
    kwargs = {
        "design": np.eye(2),
        "lower": np.zeros(2),
        "upper": np.full(2, np.inf),
        "d_design": np.zeros((2, 2, 1)),
        "d_offset": np.ones((2, 1)),
        "deadline": monotonic() + 5,
    }
    fitted, _, derivative, audit = _project(**kwargs, offset=np.array([1.0, -2.0]))
    assert fitted == pytest.approx([0, 2])
    assert derivative[:, 0] == pytest.approx([1, 0])
    assert audit["active_mask"] == [-1, 0]
    with pytest.raises(ProfileUnavailable, match="ambiguous"):
        _project(**kwargs, offset=np.array([0.0, -2.0]))


def test_deadline_checkpoint_and_failure_counters(case):
    events = []
    _, oracle, profile = make_oracle(*case, checkpoint=events.append)
    point = oracle.vector(oracle.start)
    with pytest.raises(TimeoutError):
        profile.evaluate(point, monotonic() - 1)
    assert profile.calls_started == profile.calls_failed == 1
    profile.evaluate(point, monotonic() + 20, jacobian=False)
    assert profile.calls_completed == 1
    assert events[-1]["stage"] == "profile_completed"
    assert any(event["stage"] == "bounded_linear_solve" for event in events)
