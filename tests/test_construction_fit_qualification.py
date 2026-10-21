"""Public boundaries, complete roster, numerical replay, and no-retry accounting."""

from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting.initialization_qualification import fixture
from autoformalism.research import construction_fit_qualification as q


def inputs():
    """Two models receive identical observed data and no evaluator truth."""
    examples = {name: fixture(name, 0) for name in ("shared_root", "trajectory_root")}
    first = examples["shared_root"]
    return {
        "protocol": "construction-fit-inputs-1",
        "records": {
            name: {
                "problem": value["problem"],
                "source_packet_sha256": q.public.content_sha256(value["problem"]),
                "source_model_id": name,
                "proposal_seed": 0,
                "metadata": {},
            }
            for name, value in examples.items()
        },
        "training": first["training"],
        "validation": first["validation"],
        "source": {"kind": "synthetic-smoke"},
        "test_data_opened": False,
    }


def policy():
    """Small deterministic numerical allowances for synthetic regression checks."""
    return q.Policy(
        population="synthetic-smoke",
        prefix=q.PrefixPolicy(duration=1),
        training=q.kernel.NumericalPolicy(seconds=20, maximum_calls=60),
        inference=q.kernel.NumericalPolicy(seconds=10, maximum_calls=40),
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("construction_fit")
    q.prepare(root, inputs(), policy())
    results = [q.run_task(root, index) for index in range(2)]
    return root, results


def test_numeric_smoke_keeps_both_initialization_families_and_scores(completed):
    root, results = completed
    summary = q.report(root)
    assert summary["completed_tasks"] == summary["model_count"] == 2
    assert not summary["automatic_promotion"]
    assert not summary["automatic_revision"]
    assert summary["prefix_selection"] == "predeclared_diagnostic_not_optimized"
    for result in results:
        assert result["training_status"] == "available"
        assert result["training_score"]["normalized_mse"] < 1e-12
        assert result["training_suffix_score"]["sample_count"] == 90
        assert result["training_suffix_score"]["normalized_mse"] < 1e-12
        assert result["validation_score"]["normalized_mse"] < 1e-12
        assert result["validation_available_rows"] == 2
        assert result["training_rank"]
        assert result["cost"]["accounting_complete"]
        assert result["cost"]["verification_calls_started_total"] == 12
        assert result["cost"]["search_calls_started_total"] > 0
        for row in result["validation_rows"]:
            assert row["score"]["sample_count"] == 30
            assert row["score"]["scoring_rule"] == "strictly_after_prefix_cutoff"
    no_roots = results[0]["operations"]["val_0_inference"]["value"]
    assert no_roots["roots"] == {}
    assert no_roots["search_calls_started"] == 0
    assert no_roots["search"]["stop_reason"] == "no_free_coordinates"
    with_roots = results[1]["operations"]["val_0_inference"]["value"]
    assert with_roots["search_calls_started"] > 0
    assert with_roots["rank_diagnostic"]["rank"] == 1


def test_resume_reuses_exact_receipts_without_numerical_work(completed, monkeypatch):
    root, results = completed
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(q.execution, "fit_training", lambda *a: pytest.fail("refitted"))
    monkeypatch.setattr(
        q.execution, "infer_prefix", lambda *a: pytest.fail("reinferred")
    )
    for index, result in enumerate(results):
        assert q.run_task(root, index) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_prepare_freezes_source_policy_and_data(tmp_path, monkeypatch):
    data = inputs()
    prepared = q.prepare(tmp_path, data, policy())
    assert q.prepare(tmp_path, data, policy()) == prepared
    plan, restored = q.verify(tmp_path)
    assert restored == data
    assert len(plan["tasks"]) == 2
    monkeypatch.setattr(q.public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="source/runtime"):
        q.verify(tmp_path)
    assert q.verify(tmp_path, runtime=False)[0] == plan
    with pytest.raises(ValueError, match="sealed artifact differs"):
        q.prepare(tmp_path, data, policy())


def eight_inputs():
    """Expand controls to the exact real campaign dimensions without selection."""
    data = inputs()
    originals = data["records"]
    data["records"] = {
        f"model_{i}": deepcopy(originals["shared_root" if i < 3 else "trajectory_root"])
        for i in range(8)
    }
    data["source"].update(permitted_future_inputs=[], permitted_future_auxiliaries=[])
    for name, count in (("training", 16), ("validation", 4)):
        original = data[name]["rows"][0]
        data[name]["rows"] = [
            deepcopy(original) | {"trajectory_id": f"{name}_{i}"} for i in range(count)
        ]
    return data


def test_real_population_requires_all_eight_and_all_public_rows(tmp_path):
    data = eight_inputs()
    frozen_policy = q.Policy(prefix=q.PrefixPolicy(duration=1))
    assert q.prepare(tmp_path, data, frozen_policy)["tasks"] == 8
    assert (
        sum(
            t["initializer_kind"] == "analytic_map"
            for t in q.verify(tmp_path)[0]["tasks"]
        )
        == 3
    )
    changed = deepcopy(data)
    del changed["records"]["model_0"]
    with pytest.raises(ValueError, match="all eight"):
        q._validate(changed, frozen_policy)
    changed = deepcopy(data)
    changed["validation"]["rows"].pop()
    with pytest.raises(ValueError, match="all eight"):
        q._validate(changed, frozen_policy)


@pytest.mark.parametrize(
    "mutation", ["test", "private", "overlap", "path", "targets", "empty_suffix"]
)
def test_untrusted_boundary_rejected(mutation):
    data, rules = inputs(), policy()
    if mutation == "test":
        data["test_data_opened"] = True
    elif mutation == "private":
        data["training"]["rows"][0]["hidden_states"] = {"z": [1]}
    elif mutation == "overlap":
        data["validation"]["rows"][0]["trajectory_id"] = data["training"]["rows"][0][
            "trajectory_id"
        ]
    elif mutation == "path":
        data["records"]["../escape"] = data["records"].pop("shared_root")
    elif mutation == "targets":
        data["validation"]["rows"][0]["targets"] = {}
    else:
        rules = rules.model_copy(update={"prefix": q.PrefixPolicy(duration=100)})
    with pytest.raises(ValueError):
        q._validate(data, rules)


def test_interrupted_reservation_is_charged_and_never_retried(tmp_path):
    identity = {"task": "example"}
    folder = tmp_path / "operations" / "training"
    bound = {"campaign": identity, "stage": "training", "allowance_seconds": 17}
    seal(folder / "started.json", bound)
    q.public._write(folder / "progress.json", {"search_calls_started": 4})
    result = q._operation(
        tmp_path,
        identity,
        "training",
        17,
        lambda save: pytest.fail("interrupted work retried"),
    )
    assert result["status"] == "interrupted"
    assert result["budget_charge_seconds"] == 17
    assert not result["accounting_complete"]
    assert not result["budget_restarted"]
    cost = q._cost({"training": result})
    assert cost["wall_seconds_total"] is None
    assert cost["search_calls_started_total"] is None
    assert cost["search_calls_started_observed"] == 4
    assert q._operation(tmp_path, identity, "training", 17, lambda save: {}) == result


def test_interrupt_retains_terminal_receipt_and_elapsed_cost(tmp_path):
    def work(save):
        save({"search_calls_started": 1})
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        q._operation(tmp_path, {"task": "x"}, "training", 20, work)
    result = q._operation(
        tmp_path,
        {"task": "x"},
        "training",
        20,
        lambda save: pytest.fail("interrupted work retried"),
    )
    assert result["status"] == "interrupted"
    assert result["wall_seconds"] >= 0
    assert result["accounting_complete"]
    assert q._cost({"training": result})["search_calls_started_total"] is None


def test_checkpoint_tampering_rejected(tmp_path):
    identity = {"task": "x"}

    def work(save):
        save({"search_calls_started": 1})
        return {"fit_success": False}

    q._operation(tmp_path, identity, "training", 10, work)
    q.public._write(
        tmp_path / "operations/training/progress.json", {"search_calls_started": 0}
    )
    with pytest.raises(ValueError, match="checkpoint differs"):
        q._operation(tmp_path, identity, "training", 10, work)


def test_integration_failure_retained_and_never_converted_to_metric(
    tmp_path, monkeypatch
):
    q.prepare(tmp_path, inputs(), policy())
    monkeypatch.setattr(
        q.execution,
        "fit_training",
        lambda *a: {
            "fit_success": False,
            "verification_calls_failed": 1,
            "verification_error": "nonfinite rollout",
        },
    )
    first = q.run_task(tmp_path, 0)
    assert first["training_status"] == "integration_failure"
    assert first["model_retained"]
    assert first["training_score"] is None
    assert first["validation_score"] is None
    summary = q.report(tmp_path)
    assert len(summary["records"]) == 2
    assert all(r["validation_descriptive_rank"] is None for r in summary["records"])
    assert summary["records"][0]["training_status"] == "integration_failure"


def test_poor_fit_label_does_not_suppress_forecasts(completed):
    _, results = completed
    threshold = 1e-30
    operations = results[0]["operations"]
    result = q._assemble(
        results[0]["identity"],
        inputs(),
        operations,
        policy().model_copy(update={"poor_fit_nmse": threshold}),
    )
    assert result["training_status"] == "poor_fit"
    assert result["validation_status"] == "poor_fit"
    assert result["validation_available_rows"] == 2
    assert result["model_retained"]


def test_scoring_uses_same_strict_suffix_and_training_scales(completed):
    _, results = completed
    data = inputs()
    original = results[0]
    for row in data["validation"]["rows"]:
        for index, time in enumerate(row["time"]):
            if time <= 1:
                row["targets"]["y"][index] += 1000
    changed = q._assemble(original["identity"], data, original["operations"], policy())
    assert changed["validation_score"] == original["validation_score"]


def test_raw_target_suffix_never_reaches_inference_or_forecast(tmp_path, monkeypatch):
    data = inputs()
    for row in data["validation"]["rows"]:
        for index, time in enumerate(row["time"]):
            if time > 1:
                row["targets"]["y"][index] = 98765
    q.prepare(tmp_path, data, policy())
    infer, forecast = q.execution.infer_prefix, q.verification.forecast

    def guarded_infer(problem, fit, prefix, *args):
        assert max(prefix.time) <= 1
        assert 98765 not in prefix.targets["y"]
        return infer(problem, fit, prefix, *args)

    def guarded_forecast(problem, fit, inference, schedule, *args):
        assert not hasattr(schedule, "targets")
        assert schedule.initial_targets["y"] != 98765
        return forecast(problem, fit, inference, schedule, *args)

    monkeypatch.setattr(q.execution, "infer_prefix", guarded_infer)
    monkeypatch.setattr(q.verification, "forecast", guarded_forecast)
    result = q.run_task(tmp_path, 0)
    assert result["validation_available_rows"] == 2
    assert result["validation_status"] == "poor_fit"


def test_frozen_config_documents_full_allowances():
    config = q.Policy.model_validate_json(
        Path("configs/construction_fit_qualification_v1.json").read_text()
    )
    assert config.prefix.duration == 8
    assert set(config.prefix.future_auxiliaries) == {"EGP", "E", "Uii", "Gt"}
    assert config.training.maximum_calls == 300
    assert config.training.seconds == 600
    assert config.inference.maximum_calls == 100
    assert config.reconstruction_execution.numerical_seconds == 60


def test_report_rejects_forged_derived_result(completed, tmp_path):
    import shutil

    original, _ = completed
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    path = root / "results/shared_root/result.json"
    changed = read_seal(path)
    changed["validation_score"]["normalized_mse"] = 99
    path.unlink()
    seal(path, changed)
    with pytest.raises(ValueError, match="report differs"):
        q.report(root)


@pytest.mark.parametrize("terminal_receipt", [True, False])
def test_partial_task_costs_are_reported_without_numerical_resume(
    tmp_path, monkeypatch, terminal_receipt
):
    q.prepare(tmp_path, inputs(), policy())
    plan, data = q.verify(tmp_path)
    identity = {"plan_sha256": q.public.content_sha256(plan), "task": plan["tasks"][0]}
    folder = tmp_path / "results" / plan["tasks"][0]["task_id"]
    seconds = policy().training_execution.search_overall_seconds + (
        policy().training_execution.verification_overall_seconds
    )
    if terminal_receipt:

        def interrupted(save):
            save({"search_calls_started": 7})
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            q._operation(folder, identity, "training", seconds, interrupted)
    else:
        operation = folder / "operations/training"
        seal(
            operation / "started.json",
            {"campaign": identity, "stage": "training", "allowance_seconds": seconds},
        )
        q.public._write(
            operation / "progress.json",
            {
                "search_calls_started": 7,
                "overall_timing": {"elapsed_wall_seconds": 3, "elapsed_cpu_seconds": 2},
            },
        )
    before = {p: p.read_bytes() for p in folder.rglob("*.json")}
    monkeypatch.setattr(q.execution, "fit_training", lambda *a: pytest.fail("refitted"))
    summary = q.report(tmp_path)
    assert summary["completed_tasks"] == 0
    assert summary["records"][0]["status"] == "partial"
    assert not summary["cost"]["accounting_complete"]
    assert summary["cost"]["search_calls_started_observed"] == 7
    assert summary["cost"]["search_calls_started_total"] is None
    if not terminal_receipt:
        assert summary["cost"]["wall_seconds_total"] is None
        assert summary["cost"]["budget_charge_seconds"] == seconds
        assert summary["cost"]["wall_seconds_observed"] == 3
        assert summary["cost"]["cpu_seconds_observed"] == 2
    else:
        assert summary["cost"]["wall_seconds_total"] >= 0
    assert before == {p: p.read_bytes() for p in folder.rglob("*.json")}
    assert q._journal_snapshot(folder, identity, data, policy())


def test_source_allowlist_cannot_be_extended_by_policy():
    data = inputs()
    data["source"].update(permitted_future_inputs=[], permitted_future_auxiliaries=[])
    rules = policy().model_copy(
        update={
            "prefix": q.PrefixPolicy(duration=1, future_auxiliaries=("secret_channel",))
        }
    )
    with pytest.raises(ValueError, match="source authorization"):
        q._validate(data, rules)


def test_distribution_reports_unscaled_mad_and_missing_models():
    records = [
        {
            "training_score": {
                "normalized_mse": value,
                "per_target_normalized_mse": {"x": value * 2},
            }
        }
        for value in (1, 2, 8)
    ] + [{"training_score": None}]
    result = q._distribution(records, "training")
    assert result["available_models"] == 3
    assert result["total_models"] == 4
    assert result["aggregate"] == {"median": 2, "mad": 1}
    assert result["per_target"]["x"] == {"median": 4, "mad": 2}


def test_inspection_archive_allowlist_is_deterministic(completed, tmp_path):
    import shutil
    import tarfile

    original, _ = completed
    root = tmp_path / "inspection"
    shutil.copytree(original, root)
    (root / "unrelated_secret.txt").write_text("never include")
    result = q.inspect(root)
    artifact = Path(result["inspection_archive"])
    first = artifact.read_bytes()
    assert q.inspect(root)["inspection_archive"] == str(artifact)
    assert artifact.read_bytes() == first
    with tarfile.open(artifact) as archive:
        names = archive.getnames()
    assert "plan.json" in names and "SUMMARY.md" in names
    assert "report/summary.json" in names
    assert any("operations/training/result.json" in n for n in names)
    assert "unrelated_secret.txt" not in names
    assert "inspection.tar.gz" not in names
    assert not any(".lock" in n for n in names)
    (root / "runtime").mkdir()
    (root / "runtime/leak.txt").symlink_to(root / "unrelated_secret.txt")
    with pytest.raises(ValueError, match="symlinks"):
        q.inspect(root)
