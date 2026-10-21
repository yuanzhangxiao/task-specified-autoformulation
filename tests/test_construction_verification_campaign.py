"""Saved-endpoint campaigns preserve provenance, budgets and original results."""

import shutil
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_fit_qualification as original
from autoformalism.research import construction_verification_campaign as campaign
from tests.test_construction_fit_qualification import inputs, policy


@pytest.fixture(scope="module")
def frozen_code_identity():
    """Isolate tests from concurrent edits while retaining runtime binding checks."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(original.public, "_source_identity", lambda: "synthetic-source")
        patch.setattr(
            campaign, "_entrypoints", lambda: {"synthetic-entrypoint": "a" * 64}
        )
        yield


@pytest.fixture(scope="module")
def source(tmp_path_factory, frozen_code_identity):
    """Generate genuine small training searches for both initializer families."""
    root = tmp_path_factory.mktemp("saved_verification_source")
    original.prepare(root, inputs(), policy())
    for index in range(2):
        result = original.run_task(root, index)
        assert result["training_status"] == "available"
    original.report(root)
    return root


@pytest.fixture
def supplement(tmp_path, source):
    root = tmp_path / "supplement"
    campaign.prepare(root, source)
    return root


def file_bytes(root: Path) -> dict[str, bytes]:
    """Capture contents, including journals, to catch accidental historical writes."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def reseal(path: Path, value: dict) -> None:
    """Create hashed test corruption so semantic checks must reject it."""
    original.public._write(
        path, {"sha256": original.public.content_sha256(value), "value": value}
    )


def test_frozen_endpoints_verify_without_optimization_and_preserve_source(
    source, supplement, monkeypatch
):
    before = file_bytes(source)

    def forbidden(*args, **kwargs):
        pytest.fail("new search or validation inference")

    monkeypatch.setattr(original.execution, "fit_training", forbidden)
    monkeypatch.setattr(original.execution, "infer_prefix", forbidden)
    monkeypatch.setattr(original.kernel, "fit_training", forbidden)
    monkeypatch.setattr(original.kernel, "infer_prefix", forbidden)
    monkeypatch.setattr(original.kernel.recovery_numerics, "search", forbidden)
    plan, data = campaign.verify(supplement)
    operations = [campaign.run_task(supplement, index) for index in range(2)]
    for candidate, operation in zip(plan["candidates"], operations, strict=True):
        value = operation["value"]
        assert operation["status"] == "complete"
        assert operation["accounting_complete"] and not operation["budget_restarted"]
        assert value["success"]
        assert value["parameters"] == candidate["parameters"]
        assert value["parameters_sha256"] == candidate["parameters_sha256"]
        assert value["normalized_mse"] == pytest.approx(
            candidate["search_training_nmse"], abs=1e-12
        )
        assert value["verification_calls_completed"] == 2
        assert value["rollout_calls_completed"] == 2 * len(data["training"]["rows"])
        assert value["optimizer_calls"] == 0
    report = campaign.report(supplement)
    assert report["completed_tasks"] == report["verified_models"] == 2
    assert report["model_count"] == 2
    assert report["optimizer_calls"] == report["llm_calls"] == 0
    assert not report["test_data_opened"]
    assert not report["new_validation_inference"]
    assert not report["automatic_promotion"] and not report["rank_assessed"]
    assert report["new_cost"]["search_calls_started_total"] == 0
    assert report["new_cost"]["verification_calls_started_total"] == 4
    assert report["new_cost"]["wall_seconds_total"] == pytest.approx(
        sum(operation["wall_seconds"] for operation in operations)
    )
    assert report["historical_cost_by_model"] == {
        candidate["task"]["task_id"]: candidate["historical_cost"]
        for candidate in plan["candidates"]
    }
    assert all(record["new_validation_score"] is None for record in report["records"])
    assert file_bytes(source) == before
    inspection = campaign.inspect(supplement)
    with tarfile.open(inspection["inspection_archive"]) as archive:
        assert "source_snapshot/plan.json" in archive.getnames()
        assert "report/summary.json" in archive.getnames()
    assert file_bytes(source) == before


def test_resume_reuses_completed_verification_without_new_work(supplement, monkeypatch):
    first = campaign.run_task(supplement, 0)
    before = file_bytes(supplement)
    monkeypatch.setattr(
        campaign.numerical,
        "verify_saved_parameters",
        lambda *a, **k: pytest.fail("verification budget restarted"),
    )
    assert campaign.run_task(supplement, 0) == first
    assert file_bytes(supplement) == before


def test_failed_original_verification_recovers_exact_saved_search_endpoint(
    tmp_path, frozen_code_identity, monkeypatch
):
    source, root = tmp_path / "source", tmp_path / "supplement"
    data, rules = inputs(), policy()
    data["records"] = {"trajectory_root": data["records"]["trajectory_root"]}
    rules = rules.model_copy(
        update={
            "training": rules.training.model_copy(
                update={"verification_seconds": 1e-12}
            )
        }
    )
    original.prepare(source, data, rules)
    failed = original.run_task(source, 0)
    fit = failed["operations"]["training"]["value"]
    assert not fit["fit_success"]
    assert fit["search"]["best"] is not None
    before = file_bytes(source)
    campaign.prepare(root, source)
    monkeypatch.setattr(
        original.kernel.recovery_numerics,
        "search",
        lambda *a, **k: pytest.fail("saved endpoint refitted"),
    )
    recovered = campaign.run_task(root, 0)["value"]
    assert recovered["success"]
    assert recovered["parameters"] == fit["search"]["best"]["parameters"]
    assert recovered["normalized_mse"] == pytest.approx(
        fit["search"]["best"]["training_nmse"], abs=1e-12
    )
    report = campaign.report(root)
    assert report["verified_models"] == 1
    assert report["records"][0]["source_training_status"] == failed["training_status"]
    assert report["records"][0]["historical_validation_score"] is None
    assert report["records"][0]["new_validation_score"] is None
    assert file_bytes(source) == before


def test_preparation_is_idempotent_and_preserves_all_frozen_candidates(
    source, supplement
):
    prepared = campaign.prepare(supplement, source)
    assert prepared["tasks"] == 2
    assert campaign.prepare(supplement, source) == prepared
    plan, _ = campaign.verify(supplement)
    source_plan = read_seal(source / "plan.json")
    assert [candidate["task"] for candidate in plan["candidates"]] == source_plan[
        "tasks"
    ]
    assert plan["numerical_policy"] == source_plan["policy"]["training"]
    for name in plan["source_files"]:
        assert (source / name).read_bytes() == (
            supplement / "source_snapshot" / name
        ).read_bytes()
    changed = campaign.Policy(
        execution=campaign.verification.ExecutionPolicy(numerical_seconds=5)
    )
    with pytest.raises(ValueError, match="identity differs"):
        campaign.prepare(supplement, source, changed)


@pytest.mark.parametrize("field", ["source", "entrypoint"])
def test_runtime_identity_change_blocks_execution_but_allows_read_only_report(
    supplement, monkeypatch, field
):
    if field == "source":
        monkeypatch.setattr(original.public, "_source_identity", lambda: "changed")
    else:
        monkeypatch.setattr(campaign, "_entrypoints", lambda: {"changed": "b" * 64})
    with pytest.raises(ValueError, match="source/runtime/entrypoints"):
        campaign.run_task(supplement, 0)
    assert campaign.report(supplement)["completed_tasks"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["parameters", "missing_evaluation", "problem", "training", "cost", "policy"],
)
def test_resealed_source_corruption_fails_provenance(source, tmp_path, mutation):
    copy = tmp_path / "source"
    shutil.copytree(source, copy)
    plan = read_seal(copy / "plan.json")
    task = plan["tasks"][0]["task_id"]
    folder = copy / "results" / task
    path = folder / (
        "result.json" if mutation == "cost" else "operations/training/result.json"
    )
    value = read_seal(path)
    if mutation == "cost":
        value["cost"]["wall_seconds_total"] += 1
    else:
        fit = value["value"]
        if mutation == "parameters":
            parameters = fit["search"]["best"]["parameters"]
            parameters[next(iter(parameters))] += 0.1
        elif mutation == "missing_evaluation":
            fit["search"]["evaluations"] = []
        elif mutation == "policy":
            fit["policy"]["rtol"] = 1e-3
        else:
            fit[f"{mutation}_sha256"] = "0" * 64
    reseal(path, value)
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path / "supplement", copy)


def test_snapshot_bytes_and_candidate_identity_cannot_be_changed(supplement):
    plan = read_seal(supplement / "plan.json")
    candidate = plan["candidates"][0]
    candidate["parameters"][next(iter(candidate["parameters"]))] += 0.1
    reseal(supplement / "plan.json", plan)
    with pytest.raises(ValueError, match="provenance"):
        campaign.verify(supplement, runtime=False)


def test_snapshot_raw_file_corruption_is_rejected(supplement):
    path = supplement / "source_snapshot" / "inputs.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        campaign.verify(supplement, runtime=False)


def test_partial_journal_retains_observed_costs_and_never_renews_budget(
    supplement, monkeypatch
):
    plan, _ = campaign.verify(supplement)
    candidate = plan["candidates"][0]
    task_id = candidate["task"]["task_id"]
    seconds = plan["policy"]["execution"]["overall_seconds"]
    folder = supplement / "results" / task_id / "operations/output_verification"
    identity = {
        "campaign": campaign._identity(plan, candidate),
        "stage": "output_verification",
        "allowance_seconds": seconds,
    }
    seal(folder / "started.json", identity)
    progress = {
        "success": False,
        "verification_calls_attempted": 1,
        "verification_calls_started": 1,
        "verification_calls_completed": 0,
        "rollout_calls_started": 2,
        "rollout_calls_completed": 1,
        "timing": {"elapsed_wall_seconds": 3.5, "elapsed_cpu_seconds": 1.25},
    }
    original.public._write(folder / "progress.json", progress)
    before = file_bytes(supplement / "results")
    report = campaign.report(supplement)
    assert report["completed_tasks"] == report["verified_models"] == 0
    assert report["records"][0]["status"] == "running_or_interrupted"
    assert not report["new_cost"]["accounting_complete"]
    assert report["new_cost"]["wall_seconds_total"] is None
    assert report["new_cost"]["wall_seconds_observed"] == 3.5
    assert report["new_cost"]["cpu_seconds_observed"] == 1.25
    assert report["new_cost"]["verification_calls_started_observed"] == 1
    assert report["new_cost"]["budget_charge_seconds"] == seconds
    assert file_bytes(supplement / "results") == before
    monkeypatch.setattr(
        campaign.numerical,
        "verify_saved_parameters",
        lambda *a, **k: pytest.fail("partial work retried"),
    )
    result = campaign.run_task(supplement, 0)
    assert result["status"] == "interrupted"
    assert result["budget_charge_seconds"] == seconds
    assert not result["budget_restarted"] and not result["accounting_complete"]
    assert result["observed_progress"] == progress
    assert campaign.run_task(supplement, 0) == result
    final = campaign.report(supplement)
    assert not final["new_cost"]["accounting_complete"]
    assert final["new_cost"]["wall_seconds_observed"] == 3.5


def test_completed_checkpoint_tampering_is_rejected(supplement):
    result = campaign.run_task(supplement, 0)
    task = result["identity"]["campaign"]["task"]["task_id"]
    path = (
        supplement / "results" / task / "operations/output_verification/progress.json"
    )
    progress = original.public._read(path)
    progress["rollout_calls_started"] += 1
    original.public._write(path, progress)
    with pytest.raises(ValueError, match="checkpoint differs"):
        campaign.run_task(supplement, 0)
    with pytest.raises(ValueError, match="checkpoint differs"):
        campaign.report(supplement)


@pytest.mark.parametrize("relationship", ["same", "child", "parent"])
def test_source_and_supplement_cannot_overlap(source, relationship):
    root = {
        "same": source,
        "child": source / "new_supplement",
        "parent": source.parent,
    }[relationship]
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(root, source)


def test_source_symlink_is_rejected(source, tmp_path):
    copy = tmp_path / "source"
    shutil.copytree(source, copy)
    path = copy / "SUMMARY.md"
    path.unlink()
    path.symlink_to(source / "SUMMARY.md")
    with pytest.raises(ValueError, match=r"symlink|escapes"):
        campaign.prepare(tmp_path / "supplement", copy)


def test_snapshot_symlink_cannot_write_outside_supplement(source, tmp_path):
    root, outside = tmp_path / "supplement", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged")
    (root / "source_snapshot").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        campaign.prepare(root, source)
    assert file_bytes(outside) == {"sentinel.txt": b"unchanged"}
