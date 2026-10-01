"""Identifiability controls, bounded rescue, frozen starts and resume safety."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import identifiable_campaign as c
from autoformalism.fitting import identifiable_cases as cases
from autoformalism.fitting import identifiable_refinement as r
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.matching_probe import latent_start
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicSplit


@pytest.fixture(scope="module")
def inputs():
    return cases.make_inputs()


@pytest.fixture
def plan(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(c.cases, "make_inputs", lambda: deepcopy(inputs))
    c.prepare(tmp_path, c.CampaignConfig(starts=1))
    return tmp_path


def test_control_has_excitation_and_no_hidden_labels(inputs):
    assert set(inputs["cases"]) == {"linear", "nonlinear", "fast_slow"}
    for case in inputs["cases"].values():
        assert case["identifiability"]["passed"]
        assert case["identifiability"]["singular_ratio"] > 0.1
        for split in ("training", "validation"):
            public_split = PublicSplit.model_validate(case[split])
            for row in public_split.rows:
                assert set(row.targets) == {"y"}
                assert not row.auxiliaries
    zeros = np.zeros((10, 2))
    witness = cases.excitation(
        [{"external_inputs": {"u": np.zeros(10)}}], [zeros], nonlinear=False
    )
    assert not witness["passed"]


def test_shared_physical_starts_and_truth_is_not_a_guess(plan, monkeypatch):
    p, _ = c.verify(plan)
    assert len(p["tasks"]) == 9 and len(p["commons"]) == 3
    assert c.prepare(plan, c.CampaignConfig(starts=1))["tasks"] == 9
    for common in p["commons"].values():
        assert (
            len([t for t in p["tasks"] if t["common"] == f"{common['case']}_s0"]) == 3
        )
        assert common["start"]["init_z_value"] != cases.INITIAL_Z
        assert "reference_parameters" not in common
    before = cases.request("linear", 0)
    monkeypatch.setitem(cases.TRUTHS, "linear", {"a": 99.0, "b": 99.0, "c": 99.0})
    assert cases.request("linear", 0) == before
    with pytest.raises(ValueError, match="configuration differs"):
        c.prepare(plan, c.CampaignConfig(starts=2))
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        c.verify(plan)


@pytest.mark.parametrize("change", ["identity", "shape", "boundary", "nonfinite"])
def test_frozen_node_validation_before_solver(plan, change):
    p, inputs = c.verify(plan)
    common = p["commons"]["linear_s0"]
    model, start, _ = public._lower(
        c.PublicFitRequest.model_validate(common["request"])
    )
    system = SymbolicODE(model)
    train = public.unpack_split(
        PublicSplit.model_validate(inputs["cases"]["linear"]["training"])
    )
    nodes = deepcopy(common["nodes"])
    key = next(iter(nodes))
    if change == "identity":
        nodes["other"] = nodes.pop(key)
    elif change == "shape":
        nodes[key].pop()
    elif change == "boundary":
        nodes[key][0][1] += 1
    else:
        nodes[key][-1][1] = float("nan")
    with pytest.raises(ValueError, match="frozen node"):
        latent_start(
            system,
            train,
            np.zeros(4),
            np.ones(4) * 100,
            np.array([start[n] for n in system.names]),
            c.scale_for(train),
            c.SETTINGS,
            "collocation_init",
            1,
            plan / "never_solve",
            frozen_nodes=nodes,
        )


class QuadraticOracle:
    """Known unique product problem: a*z=1 and a=1; used to exercise block rescue."""

    def __init__(
        self,
        system,
        training,
        scales,
        settings,
        directory,
        deadline,
        *,
        sensitivities,
        budget,
        point_seconds,
    ):
        self.budget = budget
        self.valid_calls = 0
        self.best = None
        self.lower, self.upper = np.array([0.001, -100]), np.array([100.0, 100.0])
        directory.mkdir(parents=True, exist_ok=True)

    def vector(self, values):
        return np.array([values["a"], values["z"]])

    def __call__(self, vector):
        self.budget.take()
        self.valid_calls += 1
        a, z = vector
        residual = np.array([a * z - 1, a - 1])
        cost = float(0.5 * residual @ residual)
        if self.best is None or cost < self.best["cost"]:
            self.best = {"parameters": {"a": a, "z": z}, "cost": cost}
        return residual

    def jacobian(self, vector):
        a, z = vector
        return np.array([[z, a], [1.0, 0.0]])


def fake_problem(monkeypatch):
    monkeypatch.setattr(r, "GuardedOracle", QuadraticOracle)
    system = SimpleNamespace(
        names=("a", "z"), channels=("y",), initial_parameter_names=("z",)
    )
    train = SimpleNamespace(
        name=SimpleNamespace(value="train"), trajectories=[SimpleNamespace(time=[0, 1])]
    )
    coords = NumericalCoordinates(
        parameters={n: {"center": 0, "scale": 1} for n in system.names},
        states={"y": {"center": 0, "scale": 1}},
        provenance={},
    )
    return system, train, {"y": 1}, None, coords


def test_conditional_rescue_preserves_bad_pairs_and_stops(monkeypatch, tmp_path):
    problem = fake_problem(monkeypatch)
    points = [
        {"source": "balanced", "parameters": {"a": 2.0, "z": 0.5}},
        {"source": "good_coefficient_bad_initial", "parameters": {"a": 1.0, "z": 8.0}},
    ]
    result = r.refine(
        *problem,
        points,
        r.RefinementPolicy(maximum_calls=30),
        "conditional_stopping",
        tmp_path,
    )
    chosen = public._read(tmp_path / "conditional_selection.json")
    assert len(chosen) == 2  # The high-error pair was not discarded by screening.
    assert result["training_nmse"] < 1e-9
    assert result["actual_residual_calls"] <= 30
    assert result["stop_reason"] == "training_accuracy_reached_pending_replay"
    assert result["parameters"]["a"] == pytest.approx(1.0, abs=1e-4)
    assert result["parameters"]["z"] == pytest.approx(1.0, abs=1e-4)
    assert any(s["free_parameters"] == ["z"] for s in result["stages"])


def test_shared_budget_counts_screening_and_failed_calls(monkeypatch, tmp_path):
    problem = fake_problem(monkeypatch)
    points = [{"source": "bad", "parameters": {"a": 80.0, "z": 80.0}}]
    result = r.refine(
        *problem, points, r.RefinementPolicy(maximum_calls=5), "joint", tmp_path
    )
    assert result["actual_residual_calls"] == 5
    assert result["budget_exhausted"]
    assert result["parameters"] is not None
    assert result["training_nmse"] <= np.mean(np.array([6399, 79]) ** 2)


def test_accuracy_must_hold_for_each_trajectory(monkeypatch, tmp_path):
    problem = fake_problem(monkeypatch)
    problem[1].trajectories = [SimpleNamespace(time=[0]), SimpleNamespace(time=[0])]
    points = [{"source": "almost", "parameters": {"a": 1.0, "z": 1.01}}]
    result = r.refine(
        *problem,
        points,
        r.RefinementPolicy(training_nmse=1e-3, trajectory_nmse=1e-8),
        "joint_stopping",
        tmp_path,
    )
    assert result["actual_residual_calls"] > 1
    assert result["training_nmse"] <= 1e-8


def test_stagnation_exits_only_after_small_step_and_small_improvement(
    monkeypatch, tmp_path
):
    problem = fake_problem(monkeypatch)

    def stalled_solver(fun, x, *, callback, **kwargs):
        fun(x)
        for _ in range(8):
            callback(SimpleNamespace(cost=1.0, x=x))
        raise AssertionError("stagnation should stop native iteration")

    monkeypatch.setattr(r, "least_squares", stalled_solver)
    result = r.refine(
        *problem,
        [{"source": "start", "parameters": {"a": 2.0, "z": 1.0}}],
        r.RefinementPolicy(stall_steps=2),
        "joint_stopping",
        tmp_path,
    )
    assert result["stop_reason"] == "stalled"
    assert not result["budget_exhausted"]
    assert result["parameters"] is not None


def install_common(root):
    plan, _ = c.verify(root)
    task = plan["tasks"][0]
    common = plan["commons"][task["common"]]
    seal(
        root / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    seal(
        root / "common" / task["common"] / "result.json",
        {
            "common_sha256": public.content_sha256(common),
            "node_sha256": public.content_sha256(common["nodes"]),
            "points": [{"source": "ordinary", "parameters": common["start"]}],
        },
    )
    return task


def test_interrupted_budget_not_restarted_and_missing_not_success(plan, monkeypatch):
    task = install_common(plan)
    directory = plan / "results" / task["task_id"]
    seal(directory / "started.json", {"started": True})
    monkeypatch.setattr(c, "refine", lambda *a: pytest.fail("must not fit again"))
    result = c.run_task(plan, 0)
    assert result["status"] == "interrupted" and not result["budget_restarted"]
    assert c.run_task(plan, 0) == result
    summary = c.report(plan)
    assert summary["status"] == "incomplete"
    assert summary["status_counts"] == {"interrupted": 1, "missing": 8}


def test_completed_backend_can_resume_scoring_without_more_fit(
    plan, inputs, monkeypatch
):
    task = install_common(plan)
    p, _ = c.verify(plan)
    case = inputs["cases"][p["commons"][task["common"]]["case"]]
    directory = plan / "results" / task["task_id"]
    seal(
        directory / "backend.json",
        {"parameters": case["reference_parameters"], "stop_reason": "finished"},
    )
    monkeypatch.setattr(c, "refine", lambda *a: pytest.fail("must not fit again"))
    monkeypatch.setattr(
        c,
        "replay",
        lambda *a: {
            "complete": True,
            "metrics": {"train": 0.0, "val": 0.0},
            "maximum_solver_difference": 0.0,
        },
    )
    result = c.run_task(plan, 0)
    assert result["recovery_passed"]
    assert result == c.run_task(plan, 0)


def test_latent_scale_ambiguity_is_detected_locally(inputs, tmp_path):
    req = cases.request("linear", 0).model_dump(mode="json")
    req["base_candidate"]["state_equations"][0]["rhs"] = "k*z"
    req["base_candidate"]["parameters"].append(
        {"name": "k", "role": "nonnegative_coefficient", "scope": "global"}
    )
    req["parameter_guesses"]["k"] = 1.0
    case = inputs["cases"]["linear"]
    audit = c.sensitivity_audit(
        c.PublicFitRequest.model_validate(req),
        PublicSplit.model_validate(case["training"]),
        {**case["reference_parameters"], "k": 1.0},
        tmp_path,
    )
    assert audit["available"] and not audit["full_local_rank"]
    assert audit["structural_identifiability"] == "not_certified"


@pytest.mark.parametrize("confirmed", [True, False])
def test_scheduler_no_duplicates_and_no_gpu(tmp_path, inputs, monkeypatch, confirmed):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "identifiable_submit", "scripts/submit_phase_c_identifiable_fitting.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(c.cases, "make_inputs", lambda: deepcopy(inputs))
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=f"{1000 + len(calls)}\n" if confirmed else "\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", scheduler)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "a" * 40)
    for _ in range(2):
        if confirmed:
            assert (
                module.submit(
                    tmp_path,
                    Path("configs/phase_c_identifiable_fitting_v1.json"),
                    account="test",
                    concurrency=2,
                )["array_tasks"]
                == 27
            )
        else:
            with pytest.raises(ValueError, match=r"[Uu]nconfirmed"):
                module.submit(
                    tmp_path,
                    Path("configs/phase_c_identifiable_fitting_v1.json"),
                    account="test",
                    concurrency=2,
                )
    assert len(calls) == (3 if confirmed else 1)
    assert all(
        "--partition=cpu" in argv and not any("--gres" in a for a in argv)
        for argv in calls
    )


def test_unavailable_rollouts_do_not_become_zero_gradient_success(
    monkeypatch, tmp_path
):
    problem = fake_problem(monkeypatch)

    class FailedOracle(QuadraticOracle):
        def __call__(self, vector):
            self.budget.take()
            raise ValueError("integration failed")

    monkeypatch.setattr(r, "GuardedOracle", FailedOracle)
    points = [{"source": "start", "parameters": {"a": 2.0, "z": 1.0}}]
    result = r.refine(
        *problem,
        points,
        r.RefinementPolicy(maximum_calls=5),
        "joint_stopping",
        tmp_path,
    )
    assert result["actual_residual_calls"] == 1
    assert result["parameters"] is None
    assert result["stop_reason"] == "no_feasible_training_point"


def test_checkpoint_pool_keeps_early_and_recent_poor_objectives(tmp_path):
    from time import monotonic

    import casadi as ca

    from autoformalism.fitting.collocation_progress import CollocationProgress

    opti = ca.Opti()
    theta = opti.variable()
    opti.subject_to(theta >= 0)
    opti.minimize((theta - 1) ** 2)
    progress = CollocationProgress(
        opti,
        theta,
        ("p",),
        np.array([0.0]),
        np.array([100.0]),
        tmp_path,
        monotonic(),
        pool_capacity=3,
    )
    # Feed deterministic iterate observations; no native optimization is needed.
    values = {"g": [0.0], "lo": [0.0], "hi": [100.0], "x": [1.0], "dual": [0.0]}
    progress.theta, progress.dual = "p", "dual"
    progress.opti = SimpleNamespace(
        debug=SimpleNamespace(value=lambda key: values[key]),
        g="g",
        lbg="lo",
        ubg="hi",
        f="f",
        x="x",
    )
    for i in range(10):
        values.update(p=[float(i)], f=float(i + 1))
        progress.record(i * 5)
    assert [p["iteration"] for p in progress.pool] == [0, 40, 45]
    assert progress.saved["best_feasible"]["iteration"] == 0
    assert not any(p["physical_rollout_verified"] for p in progress.pool)
    assert len(public._read(tmp_path / "progress.json")["checkpoint_pool"]) == 3


def test_inner_timeout_is_distinct_from_numerical_failure(monkeypatch, tmp_path):
    problem = fake_problem(monkeypatch)

    class SlowOracle(QuadraticOracle):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sensitivities = kwargs["sensitivities"]

        def __call__(self, vector):
            if self.sensitivities:
                self.budget.take()
                raise r.SensitivityUnavailable("point timed out") from TimeoutError()
            return super().__call__(vector)

    monkeypatch.setattr(r, "GuardedOracle", SlowOracle)
    result = r.refine(
        *problem,
        [{"source": "start", "parameters": {"a": 2.0, "z": 1.0}}],
        r.RefinementPolicy(),
        "joint_stopping",
        tmp_path,
    )
    assert result["stop_reason"] == "point_allowance_exhausted"
    assert not result["budget_exhausted"]
    assert result["parameters"] is not None


def test_joint_refinement_applies_sensitivity_chain_rule_in_scaled_units(
    monkeypatch, tmp_path
):
    problem = list(fake_problem(monkeypatch))
    problem[-1] = NumericalCoordinates(
        parameters={
            "a": {"center": 2.0, "scale": 20.0},
            "z": {"center": 0.5, "scale": 0.01},
        },
        states={"y": {"center": 0.0, "scale": 1.0}},
        provenance={},
    )
    result = r.refine(
        *problem,
        [{"source": "start", "parameters": {"a": 2.0, "z": 0.5}}],
        r.RefinementPolicy(),
        "joint_stopping",
        tmp_path,
    )
    assert result["parameters"] == pytest.approx({"a": 1.0, "z": 1.0}, abs=1e-5)
    assert result["training_nmse"] < 1e-10
