"""Authenticated recovery, information boundaries and nonrenewable journals."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_portfolio_recovery as recovery
from tests.test_construction_portfolio_fitting import completed  # noqa: F401


def policy():
    return recovery.Policy(
        population="synthetic-smoke",
        replay_seconds=10,
        coordinate_seconds=10,
        diagnostic_seconds=10,
        inference=recovery.kernel.NumericalPolicy(
            seconds=10, maximum_calls=60, verification_seconds=10
        ),
    )


@pytest.fixture(scope="module")
def recovered(completed, tmp_path_factory):  # noqa: F811
    source, _ = completed
    original = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    root = tmp_path_factory.mktemp("portfolio-recovery")
    recovery.prepare(root, source, policy())
    results = [recovery.run_task(root, i) for i in range(2)]
    assert original == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*.json")
    }
    return root, results


def test_real_affine_recovery_keeps_shared_fits_and_fresh_prefixes(recovered):
    root, results = recovered
    plan, data = recovery.verify(root)
    assert recovery.reservation_ceiling(plan) <= 6600
    for index, result in enumerate(results):
        assert result["source_endpoint"] == data["records"][index]["endpoint"]
        assert not result["shared_parameters_refitted"]
        assert not result["automatic_revision"]
        assert not result["test_data_opened"]
        assert result["training_prefix_available_rows"] == plan["training_rows"]
        assert result["training_prefix_forecast"]["normalized_mse"] < 1e-8
        assert not any(
            name.startswith("search") for name in result["operation_statuses"]
        )
        inference = read_seal(
            root
            / "results"
            / result["candidate_id"]
            / "operations/train_00_inference/result.json"
        )["value"]
        assert inference["search"].get(
            "method", inference["search"]["stop_reason"]
        ) in {
            "bounded_linear_least_squares",
            "no_free_coordinates",
        }
    first = recovery.report(root)
    assert first == recovery.report(root)
    assert first["status"] == "complete" and first["recorded"] == 2


def test_terminal_resume_cannot_reconstruct_or_repeat_work(recovered, monkeypatch):
    root, results = recovered
    monkeypatch.setattr(
        recovery.numerical, "make_oracle", lambda *a: pytest.fail("rebuilt")
    )
    monkeypatch.setattr(
        recovery.affine, "infer_prefix", lambda *a: pytest.fail("reran")
    )
    assert [recovery.run_task(root, i) for i in range(2)] == results


def test_interruption_charges_setup_once_and_keeps_missing_forecasts(
    completed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    source, _ = completed
    recovery.prepare(tmp_path, source, policy())
    plan, _ = recovery.verify(tmp_path)
    task = plan["tasks"][0]
    identity = {"plan_sha256": recovery.public.content_sha256(plan), "task": task}
    seal(
        tmp_path / "results/candidate-00/operations/setup/started.json",
        {
            "campaign": identity,
            "stage": "setup",
            "allowance_seconds": 120,
        },
    )
    monkeypatch.setattr(
        recovery.numerical, "make_oracle", lambda *a: pytest.fail("retried")
    )
    partial = recovery.report(tmp_path)["rows"][0]
    assert partial["status"] == "partial"
    assert partial["cost"]["budget_charge_seconds"] == 120
    result = recovery.run_task(tmp_path, 0)
    assert result["operation_statuses"] == {"setup": "interrupted"}
    assert result["training_prefix_forecast"] is None
    assert result["source_endpoint"] is not None
    assert result["cost"]["wall_seconds_total"] is None


def test_source_tampering_or_incomplete_source_is_rejected(completed, tmp_path):  # noqa: F811
    import shutil

    source, _ = completed
    copied = tmp_path / "source"
    shutil.copytree(source, copied)
    checkpoint = copied / "results/candidate-00/operations/setup/progress.json"
    value = recovery.public._read(checkpoint)
    value["optimizer_coordinates"] += 1
    recovery.public._write(checkpoint, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        recovery.prepare(tmp_path / "dest", copied, policy())
    with pytest.raises(ValueError, match="separate"):
        recovery.prepare(source / "nested", source, policy())


def test_frozen_input_and_runtime_changes_rejected(recovered, tmp_path, monkeypatch):
    _, data = recovery.verify(recovered[0])
    recovery.prepare_inputs(tmp_path, data, policy())
    changed = deepcopy(data)
    changed["records"][0]["source_result_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="sealed artifact differs"):
        recovery.prepare_inputs(tmp_path, changed, policy())
    monkeypatch.setattr(recovery.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        recovery.verify(tmp_path)
    assert recovery.verify(tmp_path, runtime=False)


def test_historic_roster_and_worker_reservation_limits(recovered, tmp_path):
    _, data = recovery.verify(recovered[0])
    with pytest.raises(ValueError):
        recovery.prepare_inputs(tmp_path, data, recovery.Policy())
    settings = policy().model_copy(
        update={
            "inference_execution": recovery.execution.SearchExecutionPolicy(
                search_overall_seconds=3600, verification_overall_seconds=3600
            ),
        }
    )
    with pytest.raises(ValueError, match="worker envelope"):
        recovery.prepare_inputs(tmp_path, data, settings)


def test_independent_replay_requires_two_successful_same_start_residuals():
    value = {
        "replay_completed": True,
        "residual": [1.0, 2.0],
        "problem_sha256": "p",
        "training_sha256": "d",
        "parameters_sha256": "v",
        "numerical_policy": {},
    }
    left = {"status": "complete", "value": value}
    right = deepcopy(left)
    assert recovery._replay_agreement({"replay_Radau": left, "replay_BDF": right})[
        "passed"
    ]
    right["value"]["residual"][0] += 1e-4
    assert not recovery._replay_agreement({"replay_Radau": left, "replay_BDF": right})[
        "passed"
    ]
    assert not recovery._replay_agreement({"replay_Radau": left})["passed"]
    right["value"]["parameters_sha256"] = "other"
    with pytest.raises(ValueError, match="identities differ"):
        recovery._replay_agreement({"replay_Radau": left, "replay_BDF": right})


def nonlinear_control(recovered, monkeypatch):
    """A two-coordinate numerical control exercises campaign-level journaling."""
    from autoformalism.fitting import prefix_portfolio as backend
    from autoformalism.fitting import prefix_portfolio_qualification as q
    from autoformalism.staged_topology import content_hash
    from tests.test_prefix_portfolio_qualification import LinearFixture

    _, data = recovery.verify(recovered[0])
    data = deepcopy(data)
    data["portfolio"]["models"] = data["portfolio"]["models"][:1]
    data["records"] = data["records"][:1]
    model = data["portfolio"]["models"][0]
    equation = model["problem"]["base_candidate"]["state_equations"][0]
    equation["rhs"] += f" - {equation['state']}*{equation['state']}"
    model["source_problem_sha256"] = content_hash(model["problem"])
    model["fitting_problem_sha256"] = recovery.public.content_sha256(model["problem"])
    record = data["records"][0]
    record["setup"]["routing"]["route"] = "original_ODE"
    record["setup"]["optimizer_coordinates"] = 2
    record["endpoint"] = None
    oracle = q.FixedRadauQualifier(
        backend.PortfolioOracle(LinearFixture(), {"route": "original_ODE"})
    )
    monkeypatch.setattr(q, "make_qualifier", lambda *a: oracle)
    calls = []

    def replay(problem, training, method, seconds, save):
        calls.append(method)
        value = {
            "replay_completed": True,
            "residual": [0.5, 0.0, 0.30000000000000004],
            "problem_sha256": recovery.public.content_sha256(problem),
            "training_sha256": recovery.public.content_sha256(training),
            "parameters_sha256": "same-start",
            "numerical_policy": {},
        }
        save(value)
        return value

    monkeypatch.setattr(q, "verify_start", replay)
    return data, calls


def test_nonlinear_coverage_and_aggregation_without_shared_search(
    recovered, tmp_path, monkeypatch
):
    data, calls = nonlinear_control(recovered, monkeypatch)
    recovery.prepare_inputs(tmp_path, data, policy())
    result = recovery.run_task(tmp_path, 0)
    assert calls == ["Radau", "BDF"]
    assert result["qualification"]["checked_indices"] == [0, 1]
    assert result["qualification"]["lift_replay_verified"]
    assert result["recovery_status"] == "qualified_for_search"
    assert not result["shared_parameters_refitted"]
    assert recovery.report(tmp_path)["rows"][0] == result


def test_nonlinear_interrupted_coordinate_stays_missing_but_other_direction_runs(
    recovered, tmp_path, monkeypatch
):
    data, _ = nonlinear_control(recovered, monkeypatch)
    recovery.prepare_inputs(tmp_path, data, policy())
    plan, _ = recovery.verify(tmp_path)
    task = plan["tasks"][0]
    identity = {"plan_sha256": recovery.public.content_sha256(plan), "task": task}
    allowance = recovery._allowances(task, policy(), plan["training_rows"])[
        "coordinate_000"
    ]
    seal(
        tmp_path / "results/candidate-00/operations/coordinate_000/started.json",
        {
            "campaign": identity,
            "stage": "coordinate_000",
            "allowance_seconds": allowance,
        },
    )
    result = recovery.run_task(tmp_path, 0)
    assert result["qualification"]["missing_indices"] == [0]
    assert result["qualification"]["checked_indices"] == [1]
    assert result["recovery_status"] == "derivatives_unqualified"
    assert result["operation_statuses"]["coordinate_000"] == "interrupted"


def test_first_replay_failure_does_not_suppress_independent_solver(
    recovered, tmp_path, monkeypatch
):
    data, calls = nonlinear_control(recovered, monkeypatch)
    replay = recovery.qualification.verify_start

    def fail_first(problem, training, method, seconds, save):
        result = replay(problem, training, method, seconds, save)
        if method == "Radau":
            result["replay_completed"] = False
        save(result)
        return result

    monkeypatch.setattr(recovery.qualification, "verify_start", fail_first)
    monkeypatch.setattr(
        recovery.qualification,
        "qualify_coordinate",
        lambda *a: pytest.fail("unqualified search"),
    )
    recovery.prepare_inputs(tmp_path, data, policy())
    result = recovery.run_task(tmp_path, 0)
    assert calls == ["Radau", "BDF"]
    assert result["recovery_status"] == "start_replay_unqualified"
    assert not result["coordinate_checks"]


def test_inspection_retains_source_provenance_logs_and_detects_changed_receipt(
    recovered, tmp_path
):
    import json
    import shutil
    import tarfile

    source, _ = recovered
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run-1.err").write_text("diagnostic\n")
    with tarfile.open(recovery.inspection(tmp_path)) as archive:
        assert "inputs.json" in archive.getnames()
        assert json.load(archive.extractfile("summary.json"))["recorded"] == 2
        assert archive.extractfile("logs/run-1.err").read() == b"diagnostic\n"
    point = tmp_path / "results/candidate-00/operations/setup/progress.json"
    recovery.public._write(point, {"changed": True})
    with pytest.raises(ValueError, match="checkpoint differs"):
        recovery.report(tmp_path)
