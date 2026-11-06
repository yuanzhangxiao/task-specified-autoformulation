"""The remaining-model campaign preserves boundaries, budgets and incumbents."""

from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import remaining_portfolio_search as numerical
from autoformalism.research import construction_remaining_portfolio as c
from tests.test_prefix_portfolio_nonlinear_reduction import case
from tests.test_prefix_portfolio_root_profile import changed


def policy(**kwargs):
    return c.Policy(
        population="synthetic-smoke",
        qualification_seconds=5,
        search_seconds=2,
        recovery_search_seconds=3,
        maximum_calls=6,
        verification_seconds=5,
        feedback_seconds=5,
        persistence_headroom_seconds=0,
        **kwargs,
    )


def inputs(route="affine", *, recovery=False, anchor=True):
    source = changed() if route == "affine" else case()
    settings = c.kernel.NumericalPolicy()
    original, oracle, start = numerical.setup(
        source.problem, source.training, settings, source.start, route
    )
    endpoint = numerical.verify(original, oracle, start, 10, lambda _: None)
    assert endpoint["original_equations_verified"]
    record = {
        "candidate_id": "candidate-00",
        "model_index": 0,
        "family": "1" * 64,
        "problem": source.problem.model_dump(mode="json"),
        "problem_sha256": c.public.content_sha256(source.problem),
        "route": route,
        "recovery": recovery,
        "anchor": endpoint if anchor else None,
        "incumbents": {"fixture": endpoint} if anchor else {},
        "provenance": {"fixture": True},
        "inherited_cost": {},
    }
    return {
        "protocol": c.INPUT_PROTOCOL,
        "training": source.training.model_dump(mode="json"),
        "numerical": settings.model_dump(mode="json"),
        "prefix": {"duration": 1, "future_inputs": []},
        "records": [record],
        "source": {"synthetic": True},
        "test_data_opened": False,
        "validation_exported": False,
    }


@pytest.fixture(scope="module")
def affine_inputs():
    return inputs()


@pytest.fixture(scope="module")
def nonlinear_inputs():
    return inputs("nonlinear")


@pytest.fixture
def ready(tmp_path, affine_inputs, monkeypatch):
    monkeypatch.setattr(c.public, "_source_identity", lambda: "frozen-test-source")
    c.prepare_inputs(tmp_path, deepcopy(affine_inputs), policy())
    return tmp_path


def test_actual_affine_fit_report_and_read_only_resume(ready, monkeypatch):
    result = c.run_task(ready, 0)
    assert result["retained"]["original_equations_verified"]
    assert result["feedback"]["passed"]
    assert len(result["starts"]) == 2
    assert len(result["searches"]) == 2
    assert result["training_reconstruction_nmse"] <= result["source_anchor_nmse"]
    assert not result["optimization_converged"]
    assert not result["parameter_confidence_established"]
    assert result["prefix_forecast_status"].startswith("deferred")
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 0) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["recorded"] == summary["expected"] == 1
    assert summary["status"] == "complete"
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_actual_nonlinear_blocks_and_release(tmp_path, nonlinear_inputs, monkeypatch):
    monkeypatch.setattr(c.public, "_source_identity", lambda: "frozen-test-source")
    c.prepare_inputs(tmp_path, deepcopy(nonlinear_inputs), policy())
    result = c.run_task(tmp_path, 0)
    assert result["retained"]["original_equations_verified"]
    assert result["feedback"]["passed"]
    assert "fit_00_block_00" in result["searches"]
    assert "fit_00_release" in result["searches"]
    assert "fit_01_release" in result["searches"]
    assert not result["optimization_converged"]


@pytest.mark.parametrize(
    "kind", ["unknown_field", "test", "validation", "targets", "problem"]
)
def test_split_and_model_boundaries_fail_closed(tmp_path, affine_inputs, kind):
    data = deepcopy(affine_inputs)
    if kind == "unknown_field":
        data["private_parameters"] = {}
    elif kind == "test":
        data["training"]["name"] = "test"
    elif kind == "validation":
        data["validation_exported"] = True
    elif kind == "targets":
        data["training"]["rows"][0]["targets"] = {
            "extra": data["training"]["rows"][0]["targets"]["y"]
        }
    else:
        data["records"][0]["problem_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.prepare_inputs(tmp_path, data, policy())
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize(
    "kind", ["duplicate", "excluded", "unsafe", "unverified", "nonfinite"]
)
def test_roster_and_incumbent_validation(tmp_path, affine_inputs, kind):
    data = deepcopy(affine_inputs)
    record = data["records"][0]
    if kind == "duplicate":
        data["records"].append(deepcopy(record))
    elif kind == "excluded":
        record["candidate_id"] = "candidate-15"
    elif kind == "unsafe":
        record["candidate_id"] = "../candidate-00"
    elif kind == "unverified":
        record["anchor"]["original_equations_verified"] = False
    else:
        record["anchor"]["training_nmse"] = float("inf")
    with pytest.raises(ValueError):
        c.prepare_inputs(tmp_path, data, policy())


def test_exact_production_projection_and_complete_roster(affine_inputs, monkeypatch):
    data = deepcopy(affine_inputs)
    data["records"] = []
    for i in c.ROSTER:
        row = deepcopy(affine_inputs["records"][0])
        row.update(candidate_id=f"candidate-{i:02d}", model_index=i)
        data["records"].append(row)
    monkeypatch.setattr(c, "SOURCE_INPUTS", c.public.content_sha256(data))
    monkeypatch.setattr(c, "SOURCE_TRAINING", c.public.content_sha256(data["training"]))
    assert len(c._tasks(data, c.Policy())) == 22
    missing = deepcopy(data)
    missing["records"].pop()
    with pytest.raises(ValueError, match="projection"):
        c._tasks(missing, c.Policy())
    changed = deepcopy(data)
    changed["records"][0]["anchor"]["training_nmse"] *= 0.5
    with pytest.raises(ValueError, match="projection"):
        c._tasks(changed, c.Policy())


def test_runtime_and_plan_changes_fail(ready, monkeypatch):
    plan, _ = c.verify(ready)
    monkeypatch.setattr(c.public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="source/runtime"):
        c.verify(ready)
    assert c.verify(ready, runtime=False)[0] == plan
    plan["automatic_revision"] = True
    (ready / "plan.json").unlink()
    seal(ready / "plan.json", plan)
    with pytest.raises(ValueError, match="identity"):
        c.verify(ready, runtime=False)


def test_missing_results_report_has_every_model_without_fitting(ready, monkeypatch):
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("fitted"))
    result = c.report(ready)
    assert result["expected"] == 1 and result["recorded"] == 0
    assert result["rows"][0]["status"] == "missing"
    assert result["rows"][0]["retained"]["original_equations_verified"]


def test_started_setup_is_charged_never_retried(ready, monkeypatch):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 0)
    seal(
        run.folder / "operations/setup/started.json",
        {
            "campaign": run.identity,
            "stage": "setup",
            "allowance_seconds": run.allowances["setup"],
        },
    )
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("retried"))
    result = c.run_task(ready, 0)
    assert result["retained"]["original_equations_verified"]
    assert result["operation_statuses"] == {"setup": "interrupted"}
    assert result["cost"]["budget_charge_seconds"] == 180
    assert not result["cost"]["accounting_complete"]


def test_interrupted_search_resumes_later_stages_without_renewal(ready, monkeypatch):
    calls = []
    real_search = numerical.search

    def interrupt(oracle, point, seconds, maximum_calls, save, route, **kwargs):
        calls.append(c.public.content_sha256(point))
        if len(calls) == 1:
            save({"best": {"parameters": point}, "stop_reason": "interrupted"})
            raise KeyboardInterrupt
        return real_search(oracle, point, seconds, maximum_calls, save, route, **kwargs)

    monkeypatch.setattr(numerical, "search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert len(calls) == 2
    assert result["operation_statuses"]["fit_00_profile"] == "interrupted"
    assert result["operation_statuses"]["fit_00_profile_verification"] == "complete"
    assert result["retained"]["original_equations_verified"]


def test_unverified_better_endpoint_cannot_displace_incumbent(affine_inputs):
    record = affine_inputs["records"][0]
    bad = deepcopy(record["anchor"])
    bad.update(training_nmse=0, original_equations_verified=False)
    operations = {"fit_00_profile_verification": {"status": "complete", "value": bad}}
    origin, retained, _ = c._retain(record, operations, policy())
    assert origin == "source:fixture"
    assert retained == record["anchor"]


def test_numerical_tie_preserves_source(affine_inputs):
    record = affine_inputs["records"][0]
    tied = deepcopy(record["anchor"])
    tied["training_nmse"] *= 1 - 1e-12
    operations = {"fit_00_profile_verification": {"status": "complete", "value": tied}}
    origin, retained, _ = c._retain(record, operations, policy())
    assert origin == "source:fixture" and retained == record["anchor"]


def test_good_verified_block_survives_worse_and_failed_later_blocks(
    tmp_path, nonlinear_inputs, monkeypatch
):
    c.prepare_inputs(tmp_path, deepcopy(nonlinear_inputs), policy())
    plan, data = c.verify(tmp_path)
    run = c._Run(tmp_path, plan, data, 0)
    setup = run.setup(lambda _: None)
    name = numerical.shared_blocks(run.oracle)[0][0]
    setup["blocks"] = {f"block_{i:02d}": [name] for i in range(3)}
    inherited = run.record["anchor"]
    initial = c.coordinates.to_effective(
        run.original, run.oracle, inherited["parameters"]
    )
    calls, replies = [], iter([(0.5, True), (0.1, False), (2.0, True), (1.5, True)])

    def search(oracle, point, seconds, maximum_calls, save, route, **kwargs):
        calls.append(dict(point))
        value = {"best": {"parameters": point | {name: point[name] + 0.1}}}
        save(value)
        return value

    def replay(stage, point):
        loss, verified = next(replies)
        return inherited | {
            "parameters": c.coordinates.to_original(run.original, run.oracle, point),
            "effective_parameters": point,
            "training_nmse": loss * inherited["training_nmse"],
            "solver_loss_discrepancy": 0.0,
            "original_equations_verified": verified,
        }

    monkeypatch.setattr(numerical, "search", search)
    monkeypatch.setattr(run, "verify_endpoint", replay)
    retained = run.fit_start(0, inherited, setup)
    assert retained["training_nmse"] == inherited["training_nmse"] * 0.5
    assert calls[0][name] == initial[name]
    assert all(point[name] == pytest.approx(initial[name] + 0.1) for point in calls[1:])


def test_second_start_is_frozen_independently_of_first_search(ready, monkeypatch):
    calls = []

    def search(oracle, point, seconds, maximum_calls, save, route, **kwargs):
        calls.append(dict(point))
        return {"best": None}

    monkeypatch.setattr(numerical, "search", search)
    result = c.run_task(ready, 0)
    assert len(calls) == 2
    source_starts = result["setup"]["starts"]
    for index, point in enumerate(calls):
        for name in point:
            if not name.startswith("trajectory:"):
                assert point[name] == source_starts[index]["parameters"][name]
    assert (
        source_starts[0]["parameters_sha256"] != source_starts[1]["parameters_sha256"]
    )


def test_failed_source_replay_preserves_historical_incumbent(ready, monkeypatch):
    def unavailable(*args, **kwargs):
        return {"status": "unavailable", "original_equations_verified": False}

    monkeypatch.setattr(numerical, "verify", unavailable)
    result = c.run_task(ready, 0)
    assert result["retained_origin"] == "source:fixture"
    assert result["retained"]["original_equations_verified"]
    assert result["searches"] == {}
    assert result["operation_statuses"]["source_verification"] == "complete"


def test_source_digest_pin_rejects_resealed_change(tmp_path):
    path = tmp_path / "source.json"
    original = {"status": "complete", "parameters": {"gain": 1.0}}
    seal(path, original)
    expected = c.public.content_sha256(original)
    assert c._pinned(path, expected) == original
    path.unlink()
    seal(path, original | {"parameters": {"gain": 2.0}})
    with pytest.raises(ValueError, match="digest"):
        c._pinned(path, expected)


def test_nonlinear_precision_tightens_source_without_changing_it(
    tmp_path, nonlinear_inputs
):
    data = deepcopy(nonlinear_inputs)
    before = deepcopy(data["numerical"])
    c.prepare_inputs(tmp_path, data, policy())
    plan, data = c.verify(tmp_path)
    run = c._Run(tmp_path, plan, data, 0)
    setup = run.setup(lambda _: None)
    assert setup["numerical_policy"]["rtol"] <= 1e-11
    assert setup["numerical_policy"]["atol"] <= 1e-13
    assert data["numerical"] == before


def test_recovery_uses_first_qualified_seed_and_one_search_budget(
    tmp_path, nonlinear_inputs, monkeypatch
):
    data = deepcopy(nonlinear_inputs)
    data["records"][0].update(anchor=None, incumbents={}, recovery=True)
    c.prepare_inputs(tmp_path, data, policy())
    qualified = []
    real = numerical.qualify

    def qualify(original, oracle, point, seconds, save, route):
        qualified.append(point)
        if len(qualified) == 1:
            return {"passed": False, "status": "invalid", "error": "seed failure"}
        return real(original, oracle, point, seconds, save, route)

    monkeypatch.setattr(numerical, "qualify", qualify)
    # Explicitly preserve a finite start after the synthetic first-start failure.
    old = numerical.deterministic_starts

    def starts(oracle, point, route, **kwargs):
        result = old(oracle, point, route, **kwargs)
        result[1]["parameters"] = dict(point)
        return result

    monkeypatch.setattr(numerical, "deterministic_starts", starts)
    result = c.run_task(tmp_path, 0)
    assert len(qualified) == 2
    assert result["retained"]["original_equations_verified"]
    assert all(name.startswith("fit_00_") for name in result["searches"])
    assert len(result["starts"]) == 2


def test_source_file_sibling_is_allowed_and_source_directory_is_protected(
    tmp_path, affine_inputs
):
    source = tmp_path / "input.json"
    seal(source, affine_inputs)
    c.prepare(tmp_path / "campaign", source, policy())
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path, source, policy())
    folder = tmp_path / "source"
    seal(folder / "inputs.json", affine_inputs)
    with pytest.raises(ValueError, match="separate"):
        c.prepare(folder / "nested", folder, policy())


def test_receipt_tampering_is_detected(ready):
    c.run_task(ready, 0)
    file = ready / "results/candidate-00/operations/setup/progress.json"
    data = c.public._read(file)
    data["passed"] = False
    c.public._write(file, data)
    with pytest.raises(ValueError, match="checkpoint"):
        c.report(ready)


def test_worker_envelope_includes_every_reserved_operation(affine_inputs):
    records = c._tasks(affine_inputs, policy())
    plan = {"tasks": records, "policy": policy().model_dump(mode="json")}
    assert c.reservation_ceiling(plan) == sum(
        c._allowances(records[0], policy()).values()
    )
    assert c.reservation_ceiling(plan) < 13200


def test_inspection_contains_relative_regular_files(ready):
    import tarfile

    archive = c.inspection(ready)
    with tarfile.open(archive) as stream:
        paths = [member.name for member in stream.getmembers()]
    assert "plan.json" in paths and "inputs.json" in paths
    assert all(
        not Path(name).is_absolute() and ".." not in Path(name).parts for name in paths
    )
