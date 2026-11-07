"""Sealed matched continuation, admissible retention, and independent replay."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import insulin_positive_continuation as fit
from autoformalism.research import insulin_positive_continuation as c
from tests.test_insulin_revision_fit import specimen  # noqa: F401
from tests.test_insulin_revision_fitting import inputs as old_inputs


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=2,
        maximum_calls=2,
        qualification_seconds=2,
        verification_seconds=2,
        persistence_headroom_seconds=0,
        diagnostic_seconds=2,
    )


def inputs():
    source = old_inputs()
    plan = c.old._plan(source, c.old.Policy(population="synthetic-smoke"))
    return {
        "protocol": c.INPUT_PROTOCOL,
        "source_inputs": source,
        "source_plan": plan,
        "validation_or_test_access": False,
        "endpoints": [
            {
                "task_id": task["task_id"],
                "source_result_sha256": "receipt",
                "endpoint": c.old._source_endpoint(source) | {"training_nmse": 0.5},
                "source_search_status": "search_budget_exhausted",
            }
            for task in plan["tasks"]
        ],
    }


class FakeFit:
    """Campaign tests use receipts; kernel tests exercise actual propagation."""

    def __init__(self, data, task, *, interrupt=False, fail_dense=False):
        self.source = c._source(data, task)
        self.positive = task["arm"] == "positive"
        self.interrupt, self.fail_dense = interrupt, fail_dense
        self.search_starts = []

    def setup_audit(self):
        return {"passed": True}

    def verify_source(self, seconds, save):
        return self.source | {
            "eligible_for_arm": not self.positive,
            "domain_admissible": False,
            "concentration_audit": {"passed": False},
        }

    def qualify(self, seconds, save):
        return self.source | {"passed": True, "training_nmse": 0.7}

    def verify(self, parameters, loss, seconds, save):
        allowed = not (self.fail_dense and loss == 0.4)
        return self.source | {
            "parameters": parameters,
            "training_nmse": loss,
            "eligible_for_arm": allowed,
            "domain_admissible": allowed,
            "concentration_audit": {"passed": allowed},
        }

    def search(self, settings, save, *, profiled_start):
        self.search_starts.append(profiled_start)
        result = {
            "best": {"parameters": profiled_start, "training_nmse": 0.4},
            "search_calls_completed": 2,
            "stop_reason": "optimizer_stopped",
            "optimizer_success": True,
            "budget_exhausted": False,
            "optimizer_terminal": {"feasible_evaluation_verified": True},
        }
        save(result)
        if self.interrupt:
            raise ValueError("synthetic interrupted attempt")
        return result

    def local_audit(self, endpoint, seconds, save):
        return {
            "status": "complete",
            "resolved_descent": False,
            "all_probes_available": True,
        }

    def metrics(self, endpoint, seconds):
        return {"training_nmse": endpoint["training_nmse"]}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(c.public, "_source_identity", lambda: "source")
    c.prepare_inputs(tmp_path, inputs(), policy())

    def setup(self, save):
        self.fit = FakeFit(self.data, self.task)
        return self.fit.setup_audit()

    monkeypatch.setattr(c._Run, "setup", setup)
    return tmp_path


def test_six_tasks_match_saved_points_and_budgets(ready):
    plan, _ = c.verify(ready)
    assert len(plan["tasks"]) == 6
    for a, b in zip(plan["tasks"][::2], plan["tasks"][1::2], strict=True):
        assert a["arm"] == "unconstrained" and b["arm"] == "positive"
        assert a["source_endpoint_sha256"] == b["source_endpoint_sha256"]
    assert not plan["source_reservations_renewed"]


@pytest.mark.parametrize(
    "mutation", ["identity", "nonfinite", "missing_root", "unverified", "boundary"]
)
def test_changed_intakes_are_rejected(tmp_path, mutation):
    data = inputs()
    if mutation == "identity":
        data["source_plan"]["inputs_sha256"] = "wrong"
    elif mutation == "boundary":
        data["validation_or_test_access"] = True
    else:
        endpoint = data["endpoints"][0]["endpoint"]
        if mutation == "unverified":
            endpoint["original_equations_verified"] = False
        elif mutation == "nonfinite":
            endpoint["training_nmse"] = float("nan")
        else:
            endpoint["parameters"].pop(next(iter(endpoint["parameters"])))
    with pytest.raises(ValueError):
        c.prepare_inputs(tmp_path, data, policy())


def test_terminal_resume_and_report_never_repeat_fitting(ready, monkeypatch):
    result = c.run_task(ready, 1)
    assert result["training_reconstruction_nmse"] == 0.4
    assert result["domain_admissible"] is True
    assert result["source_reference"]["training_nmse"] == 0.5
    assert result["qualified_search_stop"] is True
    assert result["optimization_converged"] is False
    monkeypatch.setattr(
        c._Run, "execute", lambda *_: pytest.fail("spent budget repeated")
    )
    assert c.run_task(ready, 1) == result
    assert c.report(ready)["recorded"] == 1
    assert c.inspection(ready).is_file()


def test_infeasible_lower_loss_source_never_wins_positive_arm(ready, monkeypatch):
    def setup(self, save):
        self.fit = FakeFit(self.data, self.task, fail_dense=True)
        return self.fit.setup_audit()

    monkeypatch.setattr(c._Run, "setup", setup)
    result = c.run_task(ready, 1)
    assert result["training_reconstruction_nmse"] == 0.7
    assert result["retained_origin"] == "profiled_start_verification"
    assert result["source_domain_audit"]["passed"] is False


def test_checkpoint_survives_unsuccessful_later_search(ready, monkeypatch):
    def setup(self, save):
        self.fit = FakeFit(self.data, self.task, interrupt=True)
        return self.fit.setup_audit()

    monkeypatch.setattr(c._Run, "setup", setup)
    result = c.run_task(ready, 1)
    assert result["training_reconstruction_nmse"] == 0.4
    assert result["qualified_search_stop"] is False
    assert result["operation_statuses"]["search"] != "complete"


def test_changed_source_or_reservation_cannot_resume(ready, monkeypatch):
    monkeypatch.setattr(c.public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="identity differs"):
        c.run_task(ready, 0)
    assert c.verify(ready, runtime=False)[0]
    with pytest.raises(ValueError):
        c.prepare_inputs(
            ready, inputs(), policy().model_copy(update={"maximum_calls": 3})
        )


def test_finer_grid_preserves_input_corners_and_is_not_training_data(specimen):  # noqa: F811
    training = specimen[2]
    finer = fit.verification_grid(training)
    for a, b in zip(training.rows, finer.rows, strict=True):
        assert b.time[::8] == a.time
        assert (
            b.external_inputs[fit.previous.comparison.MEAL][::8]
            == a.external_inputs[fit.previous.comparison.MEAL]
        )
        assert a.time != b.time
    assert training.fingerprint != finer.fingerprint


def test_saved_signed_point_qualified_search_and_independent_dense_replay(specimen):  # noqa: F811
    original = fit.previous.InsulinRevisionFit(*specimen)
    parameters = original.warm | {fit.previous.DELAY: -0.02}
    residual, _ = original.original.evaluate(
        original.original.vector(parameters), monotonic() + 10, jacobian=False
    )
    endpoint = {
        "parameters": parameters,
        "training_nmse": float(np.mean(residual**2)),
        "original_equations_verified": True,
    }
    run = fit.ContinuationFit(*specimen, endpoint=endpoint, enforce_positive=True)
    verified = run.verify_source(20, lambda _: None)
    assert verified["original_equations_verified"]
    assert verified["concentration_audit"]["subdivisions"] == 8
    qualified = run.qualify(10, lambda _: None)
    assert qualified["passed"]
    initial = deepcopy(run.start)
    search = run.search(
        policy().model_copy(update={"maximum_calls": 1, "search_seconds": 10}),
        lambda _: None,
        profiled_start=qualified["parameters"],
    )
    assert search["optimizer_start"] == qualified["parameters"]
    assert search["best"]["training_nmse"] <= qualified["training_nmse"] + 1e-9
    assert run.start == initial
    replay = run.verify(
        search["best"]["parameters"],
        search["best"]["training_nmse"],
        20,
        lambda _: None,
    )
    assert replay["eligible_for_arm"] and replay["original_equations_verified"]


def test_production_intake_cannot_be_replaced_by_fixture(tmp_path):
    with pytest.raises(ValueError, match="reviewed continuation"):
        c.prepare_inputs(tmp_path, inputs(), c.Policy())


def test_incomplete_local_audit_preserves_observed_descent(specimen, monkeypatch):  # noqa: F811
    original = fit.previous.InsulinRevisionFit(*specimen)
    endpoint = {"parameters": original.warm, "training_nmse": 1.0}
    run = fit.ContinuationFit(*specimen, endpoint=endpoint, enforce_positive=True)
    profiler = run.profiler_factory(run.oracle)
    calls = []

    def profile(parameters, deadline):
        calls.append(parameters)
        if len(calls) == 3:
            raise TimeoutError("limited audit")
        return {
            "parameters": parameters,
            "training_nmse": 1.0 if len(calls) == 1 else 0.8,
        }

    monkeypatch.setattr(profiler, "profile", profile)
    monkeypatch.setattr(run, "profiler_factory", lambda _: profiler)
    saved = []
    with pytest.raises(TimeoutError):
        run.local_audit(endpoint, 10, lambda value: saved.append(deepcopy(value)))
    assert saved[-1]["status"] == "partial"
    assert saved[-1]["resolved_descent"]
    assert saved[-1]["best"]["training_nmse"] == 0.8


def test_report_is_read_only_for_all_sealed_receipts(ready):
    c.run_task(ready, 0)
    before = {p: p.read_bytes() for p in ready.rglob("*.json")}
    c.report(ready)
    assert read_seal(ready / "plan.json")
    for path, value in before.items():
        assert path.read_bytes() == value
