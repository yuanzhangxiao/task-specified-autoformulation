"""Complete portfolios, original endpoint retention and causal forecast journals."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting.initialization_qualification import fixture
from autoformalism.research import construction_portfolio_fitting as campaign
from autoformalism.staged_topology import content_hash


def inputs():
    examples = [fixture(name, 0) for name in ("shared_root", "trajectory_root")]
    public = campaign.public
    return {
        "protocol": "construction-portfolio-inputs-1",
        "source": {
            "manifest_sha256": "1" * 64,
            "delivery_source_commit": "2" * 40,
            "roster_sha256": "3" * 64,
            "training_sha256": public.content_sha256(examples[0]["training"]),
            "source_hash_convention": "sorted_json_default_separators",
            "fitting_hash_convention": "sorted_json_compact_separators",
        },
        "training": examples[0]["training"],
        "prefix": {"duration": 1, "future_inputs": []},
        "models": [
            {
                "candidate_id": f"candidate-{i:02d}",
                "family": str(i + 4) * 64,
                "problem": example["problem"],
                "source_problem_sha256": content_hash(example["problem"]),
                "fitting_problem_sha256": public.content_sha256(
                    campaign.PrefixFitProblem.model_validate(example["problem"])
                ),
                "source_packet_sha256": str(i + 6) * 64,
                "records": [f"synthetic-{i}"],
            }
            for i, example in enumerate(examples)
        ],
        "validation_exported": False,
        "test_data_opened": False,
    }


def policy():
    return campaign.Policy(
        population="synthetic-smoke",
        qualification_seconds=15,
        search_seconds=20,
        verification_seconds=15,
        diagnostic_seconds=15,
        maximum_calls=100,
        inference=campaign.kernel.NumericalPolicy(
            seconds=10, maximum_calls=60, verification_seconds=5
        ),
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("portfolio")
    campaign.prepare_inputs(root, inputs(), policy())
    results = [campaign.run_task(root, i) for i in range(2)]
    return root, results


def test_real_numerical_smoke_retains_models_and_reinfers_prefix_roots(completed):
    root, results = completed
    summary = campaign.report(root)
    assert summary["status"] == "complete"
    assert summary["expected"] == summary["recorded"] == 2
    for result in results:
        assert result["qualification"]["passed"], result["qualification"]
        assert result["retained"]["original_equations_verified"]
        assert result["training_reconstruction_nmse"] < 1e-8
        assert result["training_prefix_forecast"]["normalized_mse"] < 1e-8
        assert result["training_prefix_available_rows"] == len(
            inputs()["training"]["rows"]
        )
        assert not result["automatic_revision"]
    folder = root / "results" / "candidate-01" / "operations"
    inference = read_seal(folder / "train_00_inference/result.json")["value"]
    assert inference["root_start"]
    assert inference["cutoff"] == 1
    assert "training_roots" not in inference
    assert campaign.inspection(root).is_file()


def test_resume_is_read_only_and_no_numerical_allowance_is_renewed(
    completed, monkeypatch
):
    root, results = completed
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(
        campaign.numerical, "make_oracle", lambda *a: pytest.fail("rebuilt")
    )
    assert [campaign.run_task(root, i) for i in range(2)] == results
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_full_roster_and_frozen_runtime_are_required(tmp_path, monkeypatch):
    data = inputs()
    with pytest.raises(ValueError):
        campaign.prepare_inputs(tmp_path, data, campaign.Policy())
    campaign.prepare_inputs(tmp_path, data, policy())
    plan, _ = campaign.verify(tmp_path)
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "different")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.verify(tmp_path)
    assert campaign.verify(tmp_path, runtime=False)[0] == plan


def test_interrupted_stage_is_charged_and_not_reexecuted(tmp_path, monkeypatch):
    campaign.prepare_inputs(tmp_path, inputs(), policy())
    plan, _ = campaign.verify(tmp_path)
    task = plan["tasks"][0]
    identity = {"plan_sha256": campaign.public.content_sha256(plan), "task": task}
    folder = tmp_path / "results" / task["task_id"]
    seal(
        folder / "operations/setup/started.json",
        {"campaign": identity, "stage": "setup", "allowance_seconds": 120},
    )
    snapshot = campaign.report(tmp_path)["rows"][0]
    assert snapshot["status"] == "partial"
    assert snapshot["operation_statuses"]["setup"] == "running_or_interrupted"
    assert snapshot["cost"]["wall_seconds_total"] is None
    assert snapshot["cost"]["budget_charge_seconds"] == 120
    monkeypatch.setattr(
        campaign.numerical, "make_oracle", lambda *a: pytest.fail("retried")
    )
    result = campaign.run_task(tmp_path, 0)
    assert result["fitting_status"] == "interface_failed"
    assert result["operation_statuses"] == {"setup": "interrupted"}
    assert result["cost"]["budget_charge_seconds"] == 120
    assert not result["cost"]["accounting_complete"]


def test_bad_new_endpoint_never_displaces_verified_initial_endpoint():
    def op(score, verified=True):
        return {
            "status": "complete",
            "value": {
                "status": "complete",
                "training_nmse": score,
                "original_equations_verified": verified,
            },
        }

    saved = op(0.2)
    assert (
        campaign._retained(
            {"start_verification": saved, "endpoint_verification": op(0.1, False)}
        )[0]
        == "start_verification"
    )
    assert (
        campaign._retained(
            {"start_verification": saved, "endpoint_verification": op(0.3)}
        )[0]
        == "start_verification"
    )
    assert (
        campaign._retained(
            {"start_verification": saved, "endpoint_verification": op(0.1)}
        )[0]
        == "endpoint_verification"
    )


def test_forecast_prefix_cannot_see_training_suffix():
    training = campaign.PublicSplit.model_validate(inputs()["training"])
    prefix_policy = campaign.PrefixPolicy(duration=1)
    row = training.rows[0]
    poison = row.model_dump(mode="json")
    poison["targets"] = {
        key: [v if t <= 1 else v + 10000 for t, v in zip(row.time, values, strict=True)]
        for key, values in row.targets.items()
    }
    altered = type(row).model_validate(poison)
    assert campaign.kernel.make_prefix(
        row, prefix_policy
    ) == campaign.kernel.make_prefix(altered, prefix_policy)
    assert campaign.kernel.make_forecast_schedule(
        row, prefix_policy
    ) == campaign.kernel.make_forecast_schedule(altered, prefix_policy)


def test_changed_checkpoint_and_terminal_evidence_rejected(completed, tmp_path):
    import shutil

    root, _ = completed
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    point = tmp_path / "results/candidate-00/operations/setup/progress.json"
    value = campaign.public._read(point)
    value["optimizer_coordinates"] += 1
    campaign.public._write(point, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        campaign.report(tmp_path)


def test_aggregate_reservation_rejects_unbounded_prefix_work(tmp_path):
    settings = policy().model_copy(
        update={
            "inference_execution": campaign.execution.SearchExecutionPolicy(
                search_overall_seconds=3600, verification_overall_seconds=3600
            )
        }
    )
    with pytest.raises(ValueError, match="worker envelope"):
        campaign.prepare_inputs(tmp_path, inputs(), settings)


def test_missing_models_are_counted_without_fake_failure_claim(tmp_path):
    campaign.prepare_inputs(tmp_path, inputs(), policy())
    report = campaign.report(tmp_path)
    assert report["status"] == "incomplete"
    assert report["recorded"] == 0
    assert len(report["rows"]) == 2
    assert all(row["status"] == "missing" for row in report["rows"])
    assert all(row["fitting_status"] == "not_started" for row in report["rows"])


def test_input_mutation_is_not_a_new_resume(tmp_path):
    data = inputs()
    campaign.prepare_inputs(tmp_path, data, policy())
    changed = deepcopy(data)
    changed["models"][0]["source_packet_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare_inputs(tmp_path, changed, policy())


def test_inspection_includes_submission_provenance(tmp_path):
    import tarfile

    campaign.prepare_inputs(tmp_path, inputs(), policy())
    location = tmp_path / "submissions/cpu-construction-portfolio-1"
    seal(location / "manifest.json", {"jobs": {"run": "123"}})
    (location / "run.id").write_text("123\n")
    archive = campaign.inspection(tmp_path)
    with tarfile.open(archive) as saved:
        assert (
            "submissions/cpu-construction-portfolio-1/manifest.json" in saved.getnames()
        )
        assert (
            saved.extractfile("submissions/cpu-construction-portfolio-1/run.id").read()
            == b"123\n"
        )
