"""Continuation preserves parents, input boundaries, stage identity and budgets."""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import prefix_continuation as numerical
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.research import construction_fit_continuation as c
from tests.test_prefix_initialization import problem, training


def inputs():
    model, data = problem(), training()
    payload = model.model_dump(mode="json")
    payload["base_candidate"]["parameters"].extend(
        [
            {"name": "a", "scope": "global", "role": "coefficient"},
            {"name": "g", "scope": "global", "role": "nonnegative_coefficient"},
        ]
    )
    payload["parameter_guesses"].update(a=1.0, g=1.0)
    payload["base_candidate"]["state_equations"][0]["rhs"] = "-theta*x+a*g*z+r"
    model = kernel.PrefixFitProblem.model_validate(payload)
    oracle = kernel.TrainingOracle(model, data, kernel.NumericalPolicy())
    retained = oracle.start | {"a": 0.5, "g": 1.8}
    task = {"task_id": "control", "problem_sha256": c.public.content_sha256(model)}
    return {
        "protocol": "construction-continuation-inputs-1",
        "source_plan_sha256": "a" * 64,
        "source_inputs_sha256": "b" * 64,
        "training": data.model_dump(mode="json"),
        "numerical_policy": kernel.NumericalPolicy().model_dump(mode="json"),
        "models": [
            {
                "task_id": "control",
                "problem": model.model_dump(mode="json"),
                "source_task": task,
                "parameters": retained,
                "parameters_sha256": c.public.content_sha256(retained),
                "source_search_nmse": 1.0,
            }
        ],
        "test_data_opened": False,
        "validation_exported": False,
    }


def policy(**kwargs):
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        restart_seconds=5,
        diagnostic_seconds=10,
        verification_seconds=10,
        maximum_calls=60,
        **kwargs,
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("affine_continuation")
    c.prepare(root, inputs(), policy(restart_count=0))
    results = [c.run_task(root, i) for i in range(2)]
    return root, results


def test_paired_real_numerical_smoke_and_original_model_preservation(completed):
    root, results = completed
    plan, data = c.verify(root)
    assert plan["starts"] == ["inherited"]
    assert len(results) == 2
    for result in results:
        assert result["status"] == "complete"
        operations = result["operations"]
        assert operations["qualification"]["value"]["passed"], operations[
            "qualification"
        ]
        assert "retained_control_verification" in operations
        assert "inherited_verification" in operations
        control = operations["retained_control_verification"]["value"]
        changed = operations["inherited_verification"]["value"]
        assert control["original_equations_verified"]
        assert changed["original_equations_verified"]
        assert changed["training_nmse"] < control["training_nmse"]
        assert result["retained_origin"] == "inherited"
        assert result["retained"]["training_nmse"] < 1e-8
        assert set(result["retained"]["parameters"]) == set(
            data["models"][0]["parameters"]
        )
        assert operations["inherited_diagnosis"]["status"] == "complete", operations[
            "inherited_diagnosis"
        ]
        assert result["cost"]["search_calls_started_total"] > 0
        assert (
            result["cost"]["verification_calls_attempted_total"]
            == (result["cost"]["verification_calls_started_total"])
        )
        assert not result["automatic_revision"]
        assert not result["validation_data_opened"]
        assert not result["test_data_opened"]
    original = results[0]["operations"]["qualification"]["value"]
    reduced = results[1]["operations"]["qualification"]["value"]
    assert original["original_coordinates"] == reduced["original_coordinates"]
    assert reduced["optimizer_coordinates"] == original["optimizer_coordinates"] - 1


def test_deterministic_resume_does_not_refit_or_reverify(completed, monkeypatch):
    root, results = completed
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(
        numerical, "make_oracle", lambda *a: pytest.fail("new numerical setup")
    )
    assert [c.run_task(root, i) for i in range(2)] == results
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_report_and_archive_are_read_only_numerically(completed, monkeypatch):
    root, _ = completed
    monkeypatch.setattr(numerical, "search", lambda *a: pytest.fail("search"))
    summary = c.report(root)
    assert summary["recorded"] == summary["expected"] == 2
    assert all(r["training_nmse"] < 1e-8 for r in summary["rows"])
    artifact = c.inspect(root)
    assert Path(artifact["archive"]).is_file()


@pytest.mark.parametrize(
    "mutation",
    ["validation", "test", "endpoint", "model", "targets", "duplicate", "source_task"],
)
def test_bad_inputs_rejected_without_numerical_execution(tmp_path, mutation):
    data = inputs()
    if mutation == "validation":
        data["validation"] = data["training"]
    elif mutation == "test":
        data["test_data_opened"] = True
    elif mutation == "endpoint":
        data["models"][0]["parameters"]["a"] += 1
    elif mutation == "model":
        data["models"][0]["problem"]["parameter_guesses"]["a"] = 0.1
    elif mutation == "targets":
        data["training"]["rows"][0]["targets"]["unlisted"] = data["training"]["rows"][
            0
        ]["targets"]["y"]
    elif mutation == "source_task":
        data["models"][0]["source_task"]["validation"] = data["training"]
    else:
        data["models"].append(deepcopy(data["models"][0]))
    with pytest.raises(ValueError):
        c.prepare(tmp_path, data, policy())


@pytest.mark.parametrize("arm", [0, 1])
def test_failed_qualification_preserves_original_verified_control(
    tmp_path, monkeypatch, arm
):
    c.prepare(tmp_path, inputs(), policy(restart_count=0))
    monkeypatch.setattr(numerical, "qualification", lambda *a: {"passed": False})
    monkeypatch.setattr(
        numerical, "search", lambda *a: pytest.fail("unqualified search")
    )
    result = c.run_task(tmp_path, arm)
    assert result["retained_origin"] == "retained_control"
    assert result["retained"]["original_equations_verified"]
    assert result["retained"]["parameters"] == inputs()["models"][0]["parameters"]
    assert result["optimization_status"] == "optimization_unresolved"


def test_reduction_setup_failure_cannot_block_original_control(tmp_path, monkeypatch):
    c.prepare(tmp_path, inputs(), policy(restart_count=0))
    build = numerical.make_oracle

    def failed_reduction(problem, training, settings, arm):
        if arm == "affine_reduced":
            raise ValueError("reduction setup unavailable")
        return build(problem, training, settings, arm)

    monkeypatch.setattr(numerical, "make_oracle", failed_reduction)
    monkeypatch.setattr(numerical, "search", lambda *a: pytest.fail("unqualified"))
    result = c.run_task(tmp_path, 1)
    assert result["operations"]["qualification"]["status"] == "unavailable"
    assert result["retained_origin"] == "retained_control"
    assert result["retained"]["parameters"] == inputs()["models"][0]["parameters"]


def test_reserved_interruption_cannot_reset_allowance(tmp_path, monkeypatch):
    c.prepare(tmp_path, inputs(), policy(restart_count=0))
    plan, _ = c.verify(tmp_path)
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": plan["tasks"][0]}
    folder = tmp_path / "results/control/affine_original/operations/qualification"
    folder.mkdir(parents=True)
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": "qualification",
            "allowance_seconds": c._allowances(policy(restart_count=0))[
                "qualification"
            ],
        },
    )
    monkeypatch.setattr(
        numerical, "qualification", lambda *a: pytest.fail("renewed budget")
    )
    result = c.run_task(tmp_path, 0)
    operation = result["operations"]["qualification"]
    assert operation["status"] == "interrupted"
    assert not operation["accounting_complete"]
    assert result["retained_origin"] == "retained_control"


def test_mismatched_progress_receipt_detected(completed, tmp_path):
    import shutil

    source, _ = completed
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    progress = (
        root
        / "results/control/affine_original/operations/inherited_search/progress.json"
    )
    value = c.public._read(progress)
    value["calls"] += 1
    c.public._write(progress, value)
    with pytest.raises(ValueError, match="checkpoint"):
        c.report(root)


def test_interrupted_search_verifies_saved_best_without_new_search(
    completed, tmp_path, monkeypatch
):
    source, results = completed
    _, data = c.verify(source)
    c.prepare(tmp_path, data, policy(restart_count=0))
    plan, _ = c.verify(tmp_path)
    task = plan["tasks"][0]
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": task}
    folder = tmp_path / "results/control/affine_original/operations/inherited_search"
    folder.mkdir(parents=True)
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": "inherited_search",
            "allowance_seconds": c._allowances(policy(restart_count=0))[
                "inherited_search"
            ],
        },
    )
    saved = results[0]["operations"]["inherited_search"]["value"]
    c.public._write(folder / "progress.json", saved)
    monkeypatch.setattr(numerical, "search", lambda *a: pytest.fail("budget renewed"))
    result = c.run_task(tmp_path, 0)
    assert result["operations"]["inherited_search"]["status"] == "interrupted"
    assert result["retained_origin"] == "inherited"
    assert result["retained"]["original_equations_verified"]
    assert result["retained"]["training_nmse"] < 1e-8
    assert result["optimization_status"] == "optimization_unresolved"


def test_slow_construction_consumes_inclusive_stage_allowance(tmp_path, monkeypatch):
    c.prepare(tmp_path, inputs(), policy(restart_count=0))
    clock = [0.0]
    monkeypatch.setattr(c, "monotonic", lambda: clock[0])
    build = numerical.make_oracle

    def slow(*args):
        value = build(*args)
        clock[0] += 1000
        return value

    monkeypatch.setattr(numerical, "make_oracle", slow)
    monkeypatch.setattr(numerical, "qualification", lambda *a: pytest.fail("overrun"))
    result = c.run_task(tmp_path, 0)
    assert result["optimization_status"] == "optimization_unresolved"
    assert all(op["status"] == "unavailable" for op in result["operations"].values())


def test_failed_original_verification_blocks_new_search(tmp_path, monkeypatch):
    c.prepare(tmp_path, inputs(), policy(restart_count=0))
    monkeypatch.setattr(
        numerical, "verify_endpoint", lambda *a: {"original_equations_verified": False}
    )
    monkeypatch.setattr(
        numerical, "search", lambda *a: pytest.fail("unverified search")
    )
    result = c.run_task(tmp_path, 0)
    assert result["retained"] is None
    assert result["optimization_status"] == "optimization_unresolved"


def test_all_starts_are_paired_in_original_coordinates_and_fixed():
    data = inputs()
    model = data["models"][0]
    problem = kernel.PrefixFitProblem.model_validate(model["problem"])
    training = kernel.PublicSplit.model_validate(data["training"])
    original, reduced = numerical.make_oracle(
        problem, training, kernel.NumericalPolicy(), "affine_reduced"
    )
    starts = numerical.predetermined_starts(original, model["parameters"])
    assert starts == numerical.predetermined_starts(original, model["parameters"])
    assert len({c.public.content_sha256(s["parameters"]) for s in starts}) == 3
    for start in starts:
        effective = numerical.to_effective(original, reduced, start["parameters"])
        original.vector(numerical.to_original(original, reduced, effective))


def test_strict_improvement_and_verified_retention_only():
    def op(loss, verified=True):
        return {
            "status": "complete",
            "value": {
                "original_equations_verified": verified,
                "training_nmse": loss,
                "alternate_training_nmse": loss,
            },
            "accounting_complete": True,
            "budget_charge_seconds": 1,
            "wall_seconds": 1,
            "cpu_seconds": 1,
        }

    ops = {
        "retained_control_verification": op(1),
        "inherited_verification": op(0.1, False),
    }
    assert c._assemble({}, ops)["retained_origin"] == "retained_control"
    ops["declared_verification"] = op(1)
    assert c._assemble({}, ops)["retained_origin"] == "retained_control"
    ops["perturbed_declared_verification"] = op(0.2)
    assert c._assemble({}, ops)["retained_origin"] == "perturbed_declared"
