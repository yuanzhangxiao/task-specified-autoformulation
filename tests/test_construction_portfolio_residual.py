"""Continuation provenance, retention, residual gates and deterministic resume."""

from copy import deepcopy

import pytest

from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as numerical
from autoformalism.fitting import prefix_portfolio_residual_search as residual
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_residual as c
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from tests.test_prefix_portfolio_timescale import coupled, wrapped
from tests.test_prefix_structural_reduction import training


def policy():
    return c.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        search_seconds=2,
        maximum_calls=30,
        verification_seconds=10,
        boundary_seconds=10,
    )


@pytest.fixture
def intake(monkeypatch):
    # A small actual ODE, without replaying five historical campaign test fixtures.
    source = coupled()
    payload = source.problem.model_dump(mode="json")
    payload["base_candidate"]["parameters"].append(
        {"name": "read_gain", "scope": "global", "role": "coefficient"}
    )
    payload["base_candidate"]["observation_mappings"][0]["expression"] = "read_gain*x+z"
    payload["parameter_guesses"]["read_gain"] = 1.0
    problem = PrefixFitProblem.model_validate(payload)
    train = training(source)
    original = kernel.TrainingOracle(problem, train, kernel.NumericalPolicy())
    original, oracle = wrapped(original)
    ep = numerical.verify_endpoint(original, oracle, oracle.start, 10, lambda _: None)
    assert ep["original_equations_verified"]
    task = {
        "candidate_id": "toy",
        "model_index": 0,
        "kind": "root_profile",
        "seed": "action_tau_10",
        "task_id": "toy-profile-source",
        "index": 0,
        "problem_sha256": public.content_sha256(problem.model_dump(mode="json")),
    }
    plan = {
        "protocol": c.previous.PROTOCOL,
        "policy": {"population": "synthetic-smoke"},
        "tasks": [task],
    }
    op = {
        "status": "complete",
        "value": {
            "audit": {
                "numerical_policy": kernel.NumericalPolicy().model_dump(mode="json")
            }
        },
    }
    terminal = {
        "identity": {"plan_sha256": public.content_sha256(plan), "task": task},
        "retained": ep,
        "operation_receipts": {"setup": public.content_sha256(op)},
        "cost": reporting.canonical_cost({}),
    }
    data = {
        "protocol": c.PROTOCOL,
        "source_plan": plan,
        "training": train.model_dump(mode="json"),
        "numerical": kernel.NumericalPolicy().model_dump(mode="json"),
        "records": [
            {
                "source_task": task,
                "problem": problem.model_dump(mode="json"),
                "terminal": terminal,
                "terminal_sha256": public.content_sha256(terminal),
                "operations": {"setup": op},
                "reconciliation": {"scientific_fields_exact": True},
                "boundary_coordinate": "read_gain",
            }
        ],
        "test_data_opened": False,
    }
    factory = c.timescale.make_oracle
    monkeypatch.setattr(
        c.timescale,
        "make_oracle",
        lambda *a, **kw: factory(*a, **(kw | {"states": ("x", "z")})),
    )
    # Avoid unrelated concurrent source edits affecting the scientific unit fixture.
    monkeypatch.setattr(public, "_source_identity", lambda: "synthetic-source")
    return data


@pytest.fixture
def ready(intake, tmp_path):
    root = tmp_path / "run"
    c.prepare_inputs(root, intake, policy())
    return root


def test_root_campaign_actual_rollout_resume_and_receipt_report(ready, monkeypatch):
    result = c.run_task(ready, 0)
    assert result["qualification"]["passed"]
    assert result["retained"]["original_equations_verified"]
    assert result["boundary_control"]["passed"]
    assert result["operation_statuses"]["boundary_verification"] == "complete"
    assert not result["optimization_converged"]
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 0) == result
    assert c.report(ready) == c.report(ready)
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_failed_qualification_preserves_source_without_search(ready, monkeypatch):
    monkeypatch.setattr(
        roots, "root_profile_qualification", lambda *a: {"passed": False}
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **kw: pytest.fail("searched")
    )
    result = c.run_task(ready, 0)
    assert result["retained"] == result["source_endpoint"]
    assert result["search_termination"] == "not_started"
    assert "boundary_control" not in result["operation_statuses"]


def test_interrupted_search_retains_profile_and_never_resets_budget(ready, monkeypatch):
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        args[4](
            {
                "best": {"parameters": args[1], "training_nmse": 1e100},
                "search_calls_started": 1,
                "search_calls_attempted": 1,
            }
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(roots, "root_profile_search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert len(calls) == 1
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["retained"]["original_equations_verified"]
    assert not result["optimization_converged"]


def test_runtime_and_training_protection(ready, intake, tmp_path, monkeypatch):
    changed = deepcopy(intake)
    changed["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        c.prepare_inputs(tmp_path / "val", changed, policy())
    changed = deepcopy(intake)
    changed["records"][0]["terminal"]["retained"]["training_nmse"] += 1
    with pytest.raises(ValueError, match="authentication"):
        c.prepare_inputs(tmp_path / "tampered", changed, policy())
    with pytest.raises(ValueError, match="reviewed"):
        c.prepare_inputs(tmp_path / "production", intake, c.Policy())
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="code/runtime"):
        c.run_task(ready, 0)


def test_terminal_tampering_rejected(ready):
    c.run_task(ready, 0)
    path = ready / "results/toy-profiled/result.json"
    payload = public._read(path)
    payload["value"]["training_reconstruction_nmse"] += 1
    # Even a newly resealed but scientifically different terminal is rejected.
    payload["sha256"] = public.content_sha256(payload["value"])
    public._write(path, payload)
    with pytest.raises(ValueError):
        c.report(ready)


def op(value):
    return {"status": "complete", "value": value}


@pytest.mark.parametrize(
    "delta,classification",
    [(1e-14, "numerically_tied"), (0.1, "meaningful_improvement"), (-0.1, "worse")],
)
def test_numeric_ties_do_not_displace_incumbent(delta, classification):
    old = {"training_nmse": 1.0, "original_equations_verified": True}
    new = old | {"training_nmse": 1.0 - delta}
    assert c.compare_endpoints(old, new, policy())["classification"] == classification
    ops = {
        "start_verification": op(old),
        "qualification": op({"passed": True}),
        "endpoint_verification": op(new),
    }
    origin, selected, _ = c._selection({"terminal": {"retained": old}}, ops, policy())
    assert origin == ("endpoint_verification" if delta == 0.1 else "source")
    assert selected == (new if delta == 0.1 else old)


def test_solver_error_controls_tie_and_unverified_boundary_never_selected():
    old = {
        "training_nmse": 1.0,
        "solver_loss_discrepancy": 0.01,
        "original_equations_verified": True,
    }
    new = old | {"training_nmse": 0.99}
    assert (
        c.compare_endpoints(old, new, policy())["classification"] == "numerically_tied"
    )
    ops = {
        "start_verification": op(old),
        "qualification": op({"passed": True}),
        "boundary_verification": op(
            {"training_nmse": 0.0, "original_equations_verified": False}
        ),
    }
    assert c._selection({"terminal": {"retained": old}}, ops, policy())[0] == "source"


def test_verified_root_checkpoint_survives_missing_outer_endpoint():
    old = {"training_nmse": 1.0, "original_equations_verified": True}
    better = old | {"training_nmse": 0.5}
    ops = {
        "start_verification": op(old),
        "qualification": op({"passed": True}),
        "profiled_start_verification": op(better),
    }
    assert c._selection({"terminal": {"retained": old}}, ops, policy())[1] == better


def test_call_limit_distinguishes_invalid_trials_from_wall_limit():
    assert (
        c._termination(
            {
                "search_calls_attempted": 20,
                "search_calls_started": 18,
                "maximum_calls": 20,
                "budget_exhausted": True,
            }
        )
        == "evaluation_limit"
    )
    assert (
        c._termination({"stop_reason": "call_limit", "budget_exhausted": True})
        == "evaluation_limit"
    )
    assert c._termination({"budget_exhausted": True}) == "numerical_budget_exhausted"
    assert c._termination({"optimizer_success": True}) == "optimizer_stopped"


def test_residual_route_never_invokes_gradient_gate(monkeypatch):
    run = object.__new__(c._Run)
    run.task = {"kind": "residual_search"}
    run.policy = policy()
    run.original = run.oracle = object()
    run.start = {"a": 1.0}
    ep = {
        "training_nmse": 1.0,
        "parameters": run.start,
        "original_equations_verified": True,
    }
    run.record = {"terminal": {"retained": ep}}
    run.ops = {}
    stages = []

    def perform(stage, work):
        stages.append(stage)
        value = work(lambda _: None)
        run.ops[stage] = op(value)
        return value

    def verification(stage, parameters, expected=None):
        run.ops[stage] = op(ep)
        return ep

    run.perform, run.verify_endpoint = perform, verification
    monkeypatch.setattr(residual, "qualify_residual", lambda *a: {"passed": True})
    monkeypatch.setattr(
        residual, "search", lambda *a: {"best": ep, "search_calls_attempted": 30}
    )
    monkeypatch.setattr(
        c.previous.numerical, "qualification", lambda *a: pytest.fail("Jacobian gate")
    )
    run.fit()
    assert stages == ["qualification", "search"]
    assert "endpoint_verification" in run.ops


def test_new_source_root_must_be_disjoint(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path, tmp_path / "source", policy())
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path / "source/new", tmp_path / "source", policy())


def test_setup_preserves_per_record_tight_numerics(ready, monkeypatch):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 0)
    run.task = dict(run.task, kind="residual_search")
    numerical_policy = data["records"][0]["operations"]["setup"]["value"]["audit"][
        "numerical_policy"
    ]
    numerical_policy.update(rtol=1e-11, atol=1e-13)
    observed = []

    def factory(problem, train, anchor, *, numerical):
        observed.append(numerical)
        return wrapped(kernel.TrainingOracle(problem, train, numerical))

    monkeypatch.setattr(c.reduced, "make_oracle", factory)
    run.setup(lambda _: None)
    assert observed[0].rtol == 1e-11 and observed[0].atol == 1e-13


def test_production_split_and_numerical_pins_reject_edits():
    # No local data or archive is required to exercise these independent pins.
    data = {
        "protocol": c.PROTOCOL,
        "test_data_opened": False,
        "source_plan": {"protocol": c.previous.PROTOCOL},
        "training": {},
        "numerical": {},
    }
    from unittest.mock import patch

    with (
        patch.object(c, "SOURCE_PLAN", public.content_sha256(data["source_plan"])),
        pytest.raises(ValueError, match="training data or numerical"),
    ):
        c._tasks(data, c.Policy())
