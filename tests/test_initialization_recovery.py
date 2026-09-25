"""Recover unstarted fits without replacing historical proposals or spent budgets."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import initialization_recovery as recovery
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from tests.test_component_campaign import fixture as component_fixture
from tests.test_fitted_initialization import initial_channel_alias_problem
from tests.test_initialization_namespace_audit import reseal


def overwrite(path, value):
    path.unlink(missing_ok=True)
    return sealed_write(
        path, {k: v for k, v in value.items() if k != "artifact_sha256"}
    )


def fixture(source, *, missing_guesses=False, expression="a+b*U"):
    plan = component_fixture(source)
    model, row, initial = initial_channel_alias_problem(expression=expression)
    bundle = {
        "initialization": {
            "base_candidate": model.validated.candidate.model_dump(mode="json"),
            "plan": initial.model_dump(mode="json"),
            "guesses": {} if missing_guesses else {"rate": 0.3, "gain": 2.0},
        },
        "context": model.validated.context.model_dump(mode="json"),
    }
    task = plan["tasks"][0]
    cell = plan["cells"][task["cell"]]
    for name, split, row_id in (
        ("training", SplitName.TRAIN, "train"),
        ("validation", SplitName.VALIDATION, "val"),
    ):
        cell[name] = public.pack_split(
            DatasetSplit(split, (replace(row, trajectory_id=row_id),), row_id)
        ).model_dump(mode="json")
    plan["source_sha256"] = "a" * 64
    plan = overwrite(source / "plan.json", plan)
    directory = io.round_path(source, task, 0)
    with public._lock(directory / "supervisor"), public._lock(directory):
        pass  # Historical worker lock files exist even after it has exited.
    proposal = sealed_write(
        directory / "proposal.json",
        {
            "task": task,
            "round": 0,
            "status": "constructed",
            "bundle": bundle,
            "cost": {"physical_requests": 4, "observed_tokens": 2000},
        },
    )
    sealed_write(
        directory / "worker_started.json",
        {
            "plan": plan["artifact_sha256"],
            "task": task,
            "round": 0,
        },
    )
    sealed_write(
        directory / "result.json",
        {
            "task": task,
            "round": 0,
            "status": "worker_interrupted",
            "trial": None,
            "selected": None,
            "cost": proposal["cost"],
            "construction_draft": bundle,
        },
    )
    request = pipeline.request_for(bundle, plan, task, 0)
    public.prepare_fit(
        request,
        recovery.PublicSplit.model_validate(cell["training"]),
        recovery.PublicSplit.model_validate(cell["validation"]),
        directory / "fit",
    )
    frozen = public._read(directory / "fit/freeze.json")
    frozen["source_sha256"] = plan["source_sha256"]
    reseal(directory / "fit", frozen)
    return task, directory


@pytest.mark.parametrize("missing_guesses", [False, True])
def test_exact_handoff_one_fit_and_immutable_history(
    tmp_path, monkeypatch, missing_guesses
):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, historical = fixture(source, missing_guesses=missing_guesses)
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    calls = []

    def backend(request, model, train, val, guesses, settings, directory):
        calls.append(request)
        frozen = public._read(historical / "fit/freeze.json")
        assert request.model_dump(mode="json") == frozen["request"]
        assert settings == frozen["settings"]
        assert guesses == frozen["initial_parameters"]
        assert train.name == SplitName.TRAIN and val.name == SplitName.VALIDATION
        metric = {
            "normalized_mse": 0.2,
            "per_target_normalized_mse": {"v01": 0.3, "U": 0.1},
            "failed_trajectories": [],
        }
        return {
            "parameters": dict.fromkeys(model.parameter_names, 0.5),
            "training": metric,
            "validation": metric,
        }

    monkeypatch.setattr(public, "_run_backend", backend)
    first = recovery.prepare(source, root, [task["task_id"]])
    assert first == recovery.prepare(source, root, [task["task_id"]])
    assert recovery.report(root)["status_counts"] == {"pending": 1}
    assert not calls
    recovered = recovery.run(root, 0)
    assert recovered["result"]["status"] == "complete"
    assert recovery.run(root, 0) == recovered
    report = recovery.report(root)
    assert report["status_counts"] == {"complete": 1}
    assert report["numerical_attempts_started"] == 1
    assert report["llm_calls"] == report["campaign_changes"] == 0
    assert report["rows"][0]["historical_cost"]["observed_tokens"] == 2000
    assert len(calls) == 1
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}
    with pytest.raises(ValueError, match="source changed"):
        public.execute_fit(historical / "fit")


@pytest.mark.parametrize(
    "artifact", ["started.json", "result.json", "backend", "freeze.json.tmp"]
)
def test_prior_numerical_artifacts_never_receive_a_new_budget(tmp_path, artifact):
    task, directory = fixture(tmp_path / "source")
    (directory / "fit" / artifact).write_text("existing evidence")
    with pytest.raises(ValueError, match="existing artifacts"):
        recovery.prepare(tmp_path / "source", tmp_path / "recovery", [task["task_id"]])
    assert not (tmp_path / "recovery/plan.json").exists()


def test_recovery_interrupt_cannot_restart_optimizer(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, _ = fixture(source)
    recovery.prepare(source, root, [task["task_id"]])
    directory = root / "results" / task["task_id"]
    identity = public._read(directory / "freeze.json")["identity"]
    (directory / "started.json").write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(public, "_run_backend", lambda *a: pytest.fail("budget reset"))
    assert recovery.report(root)["status_counts"] == {"started_without_result": 1}
    assert not (directory / "result.json").exists()
    assert recovery.run(root, 0)["result"]["status"] == "interrupted"
    assert recovery.report(root)["status_counts"] == {"interrupted": 1}


@pytest.mark.parametrize(
    "change", ["draft", "cost", "worker", "data", "seed", "lowering"]
)
def test_resealed_but_inconsistent_historical_handoff_is_rejected(tmp_path, change):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, directory = fixture(source)
    if change in {"draft", "cost"}:
        result = sealed_read(directory / "result.json")
        result["construction_draft" if change == "draft" else "cost"] = {}
        overwrite(directory / "result.json", result)
    elif change == "worker":
        marker = sealed_read(directory / "worker_started.json")
        marker["plan"] = "b" * 64
        overwrite(directory / "worker_started.json", marker)
    else:
        frozen = public._read(directory / "fit/freeze.json")
        if change == "data":
            frozen["training"]["rows"][0]["targets"]["U"][0] += 1
        elif change == "seed":
            frozen["request"]["random_seed"] += 1
        else:
            frozen["lowered_candidate"]["state_equations"][0]["rhs"] = "0"
        reseal(directory / "fit", frozen)
    with pytest.raises(ValueError):
        recovery.prepare(source, root, [task["task_id"]])
    assert not (root / "plan.json").exists()


def test_source_freeze_drift_and_consumption_after_preparation_block_execution(
    tmp_path, monkeypatch
):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, historical = fixture(source)
    recovery.prepare(source, root, [task["task_id"]])
    monkeypatch.setattr(
        public, "_run_backend", lambda *a: pytest.fail("unexpected fit")
    )
    marker = source / "evaluation_freeze.json"
    marker.write_text("frozen")
    with pytest.raises(ValueError, match="sealed for evaluation"):
        recovery.run(root, 0)
    marker.unlink()
    (historical / "fit/started.json").write_text("{}")
    with pytest.raises(ValueError, match="existing artifacts"):
        recovery.run(root, 0)


def test_source_runtime_scope_and_collision_guards(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, _ = fixture(source, expression="a+b*v01")
    with pytest.raises(ValueError, match="name collision"):
        recovery.prepare(source, root, [task["task_id"]])
    with pytest.raises(ValueError, match="disjoint"):
        recovery.prepare(source, source / "recovery", [task["task_id"]])
    with pytest.raises(ValueError, match="distinct"):
        recovery.prepare(source, root, [task["task_id"], task["task_id"]])
    with pytest.raises(ValueError, match="outside"):
        recovery.prepare(source, root, ["../../escape"])


def test_parallel_historical_reads_and_active_worker_exclusion(tmp_path):
    source = tmp_path / "source"
    task, historical = fixture(source)
    plan = io.verify(source, execution=False)
    # Two recovery array entries may verify the same old records simultaneously.
    with recovery._read_lock(historical / "supervisor"):
        with ThreadPoolExecutor(2) as pool:
            replies = list(
                pool.map(lambda _: recovery._snapshot(source, plan, task), range(2))
            )
        assert replies[0] == replies[1]
    with (
        public._lock(historical / "supervisor"),
        pytest.raises(ValueError, match="still in use"),
    ):
        recovery.prepare(source, tmp_path / "recovery", [task["task_id"]])


def test_recovery_code_runtime_and_handoff_are_pinned(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    task, _ = fixture(source)
    recovery.prepare(source, root, [task["task_id"]])
    monkeypatch.setattr(
        public, "_run_backend", lambda *a: pytest.fail("unexpected fit")
    )
    with monkeypatch.context() as patch:
        patch.setattr(public, "_source_identity", lambda: "c" * 64)
        with pytest.raises(ValueError, match="source or runtime"):
            recovery.run(root, 0)
    with monkeypatch.context() as patch:
        patch.setattr(public, "_runtime", lambda: {"python": "different"})
        with pytest.raises(ValueError, match="source or runtime"):
            recovery.run(root, 0)
    with pytest.raises(ValueError, match="outside"):
        recovery.run(root, -1)
    with pytest.raises(ValueError, match="outside"):
        recovery.run(root, 1)
    directory = root / "results" / task["task_id"]
    frozen = public._read(directory / "freeze.json")
    frozen["request"]["parameter_guesses"]["gain"] = 8
    reseal(directory, frozen)
    with pytest.raises(ValueError):
        recovery.run(root, 0)
