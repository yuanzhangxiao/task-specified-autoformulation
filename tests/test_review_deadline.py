"""Campaign separation, scientifically distinct controls and budget provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from autoformalism.data import BenchmarkRegistry
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.rebuttal.review_deadline_demo import request, split, verifier_audit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_smoke", ROOT / "scripts/smoke_review_deadline.py"
)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_preregistered_contrasts_and_controls():
    config = io.DeadlineConfig.model_validate_json(
        (ROOT / "configs/review_deadline_v1.json").read_text()
    )
    tasks = io.tasks(config)
    assert len(config.public_cells) == 6
    assert len(tasks) == 44 and config.rounds == 3
    assert len({t["task_id"] for t in tasks}) == 44
    assert len([t for t in tasks if t["arm"] == "full"]) == 12
    assert len([t for t in tasks if t["arm"] == "no_spec"]) == 4
    assert len([t for t in tasks if t["arm"] == "no_latent"]) == 4
    assert {t["shared_round_zero"] for t in tasks if t["arm"] == "refit_only"} == {
        t["task_id"] for t in tasks if t["arm"] == "full"
    }
    assert not any(
        BenchmarkRegistry().get(c).one_step_target_history for c in config.public_cells
    )


def test_removed_specification_not_secretly_given_to_proposer(tmp_path):
    plan = SMOKE.fixture(tmp_path)
    task = {**plan["tasks"][0], "arm": "no_spec"}
    cell = plan["cells"][task["cell"]]
    brief = pipeline._brief(cell, task)
    assert brief.requirements == () and brief.target_dependencies == ()
    assert "delayed" not in brief.scientific_context
    assert (
        brief.public_variables
        == pipeline._brief(cell, plan["tasks"][0]).public_variables
    )


def test_verifier_depends_on_equations_not_labels():
    rows = {r["case"]: r for r in verifier_audit()}
    for name in ("correct", "renamed_latent", "scaled_latent"):
        assert rows[name]["with_semantic_gate"]
    for name in ("wrong_driver", "disconnected_memory", "no_persistent_memory"):
        assert not rows[name]["with_semantic_gate"]
        assert rows[name]["without_semantic_gate"]


def test_control_same_data_and_ordinary_start():
    one, two = request("u1"), request("u2")
    assert one.parameter_guesses == two.parameter_guesses == {"a": 0.8, "b": 0.8}
    data = split("train", ((0.4, 0.4), (1.0, 1.0)))
    assert all(r.external_inputs["u1"] == r.external_inputs["u2"] for r in data.rows)
    with pytest.raises(ValueError):
        request("unknown")


def test_missing_denominators_and_explicit_deadline_freeze(tmp_path):
    plan = SMOKE.fixture(tmp_path)
    summary = reporting.report(tmp_path)
    assert summary["planned_rounds"] == 6
    assert summary["status_counts"] == {"missing": 6}
    with pytest.raises(ValueError, match="unfinished"):
        reporting.export(tmp_path)
    receipt = reporting.export(tmp_path, allow_partial=True)
    assert receipt["subject_count"] == 0 and receipt["missing_round_count"] == 6
    assert reporting.export(tmp_path) == receipt
    with pytest.raises(ValueError, match="sealed"), io.execution_lease(tmp_path):
        pass
    with pytest.raises(ValueError, match="sealed"):
        pipeline.propose_one(tmp_path, plan, plan["tasks"][0], 0, None)


def test_active_workers_prevent_freeze(tmp_path):
    SMOKE.fixture(tmp_path)
    with (
        io.execution_lease(tmp_path),
        pytest.raises(ValueError, match="active campaign"),
        io.execution_lease(tmp_path, exclusive=True),
    ):
        pass


def test_public_content_change_blocks_resume(tmp_path):
    plan = SMOKE.fixture(tmp_path)
    io.verify(tmp_path)
    cell = plan["tasks"][0]["cell"]
    path = tmp_path / "public/phase_b_v1" / cell / "train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="public content"):
        io.verify(tmp_path)


def test_partial_freeze_keeps_failed_round_and_blocks_test_export_mutation(tmp_path):
    plan = SMOKE.fixture(tmp_path)
    task = plan["tasks"][0]
    from autoformalism.staged_topology import content_hash

    directory = io.round_path(tmp_path, task, 0) / "calls"
    directory.mkdir(parents=True)
    for i in range(3):
        request_body = {
            "namespace": content_hash([plan["artifact_sha256"], task, 0]),
            "attempt": i,
        }
        key = content_hash(request_body)
        (directory / f"{key}.json").write_text(
            __import__("json").dumps(
                {
                    "request_hash": key,
                    "request": request_body,
                    "status": "responded",
                    "observed_total_tokens": 200,
                    "budget_charge": 200,
                }
            )
        )
    sealed_write(
        io.round_path(tmp_path, task, 0) / "result.json",
        {
            "task": task,
            "round": 0,
            "status": "construction_failed",
            "selected": None,
            "trial": None,
            "closed": True,
            "cost": {"physical_requests": 3, "observed_tokens": 600},
        },
    )
    summary = reporting.report(tmp_path)
    assert summary["status_counts"] == {"construction_failed": 1, "missing": 5}
    assert summary["rows"][0]["cumulative_requests"] == 3
    reporting.export(tmp_path, allow_partial=True)
    (tmp_path / "subjects.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="subjects differ"):
        reporting.export(tmp_path)


def test_stale_revision_parent_rejected(tmp_path):
    plan = SMOKE.fixture(tmp_path)
    task = plan["tasks"][0]
    sealed_write(
        io.round_path(tmp_path, task, 0) / "result.json",
        {"status": "complete", "selected": None, "closed": True},
    )
    sealed_write(
        io.round_path(tmp_path, task, 1) / "proposal.json",
        {"parent_sha256": "wrong", "status": "no_change"},
    )
    with pytest.raises(ValueError, match="stale parent"):
        pipeline.fit_one(tmp_path, plan, task, 1)


def test_launcher_selects_new_opt_in_protocol():
    import subprocess

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/hpc/run_staged_topology_server.sh"),
            "--check-config",
            str(ROOT / "configs/review_deadline_v1.json"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "review_deadline.py"


def test_no_test_read_before_freeze(tmp_path, monkeypatch):
    SMOKE.fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("test must not be loaded")

    monkeypatch.setattr(reporting.BenchmarkLoader, "load_test", forbidden)
    with pytest.raises(FileNotFoundError):
        reporting.evaluate(tmp_path, tmp_path / "public", 0, 1)


def test_no_latent_cannot_hide_a_differential_state_in_an_auxiliary(tmp_path):
    req = request("u1")
    context = req.context.model_copy(update={"auxiliaries": ("z",)})
    cell = {
        "target_contract": {
            "benchmark_id": "control",
            "tier": "easy",
            "public_prompt_sha256": "0" * 64,
            "targets": [
                {"target_channel": "v01", "public_requirement": "Generate v01"}
            ],
        },
        "mechanism_spec": {
            "benchmark_id": "control",
            "tier": "easy",
            "required_mechanisms": [
                {
                    "id": "input_path",
                    "required_drivers": ["u1"],
                    "required_targets": ["v01"],
                }
            ],
        },
    }
    bundle = {
        "candidate": req.base_candidate.model_dump(mode="json"),
        "initialization": {"context": context.model_dump(mode="json")},
    }
    certificate = pipeline.certificates(bundle, cell, {"arm": "no_latent"})
    assert certificate["latent_states"] == []
    assert certificate["differential_states_outside_targets"] == ["z"]
    assert not certificate["ablation_constraint_pass"]


def test_campaign_revision_prompt_adds_no_unbound_scientific_requirement():
    prompt = pipeline.revision_prompt()
    assert "bound public nonlinear feedback" not in prompt
    assert "empty requirement list" in prompt
    assert "anonymous public task" not in prompt


def test_partial_provider_and_fit_work_still_count_without_round_result(tmp_path):
    import json

    from autoformalism.staged_topology import content_hash

    plan = SMOKE.fixture(tmp_path)
    task = plan["tasks"][0]
    directory = io.round_path(tmp_path, task, 0)
    (directory / "calls").mkdir(parents=True)
    body = {"namespace": content_hash([plan["artifact_sha256"], task, 0]), "attempt": 0}
    key = content_hash(body)
    (directory / "calls" / f"{key}.json").write_text(
        json.dumps(
            {
                "request_hash": key,
                "request": body,
                "status": "inflight",
                "budget_charge": 8000,
                "observed_total_tokens": None,
            }
        )
    )
    (directory / "fit").mkdir()
    for name in ("started.json", "freeze.json"):
        (directory / "fit" / name).write_text(json.dumps({"identity": "frozen-fit"}))
    summary = reporting.report(tmp_path)
    row = summary["rows"][0]
    assert row["status"] == "missing"
    assert row["cumulative_requests"] == 1 and row["cumulative_fit_attempts"] == 1
    assert row["provider_reserved_tokens"] == 8000
    assert row["provider_calls_with_unknown_usage"] == 1


def test_evaluation_cache_cannot_be_borrowed_or_refitted():
    import hashlib

    from autoformalism.fitting import public_fitting as public
    from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject

    req = request("u1")
    model, _, _ = public._lower(req)
    subject = FrozenEvaluationSubject.model_validate(
        {
            "subject_id": "control",
            "method": "demo",
            "benchmark_id": "control",
            "tier": "easy",
            "repetition": 0,
            "private_metrics_opened_after_freeze": False,
            "source_provenance": {
                "adapter": "direct_candidate",
                "request_id": "control",
                "source_path": "control.json",
                "source_sha256": "0" * 64,
                "candidate_sha256": hashlib.sha256(
                    model.validated.candidate.model_dump_json().encode()
                ).hexdigest(),
            },
            "candidate": model.validated.candidate.model_dump(mode="json"),
            "parameterization": {
                "status": "available",
                "global_parameters": {"a": 0.8, "b": 0.8},
            },
            "validation_context": model.validated.context.model_dump(mode="json"),
            "target_prediction": {"status": "missing"},
        }
    )
    binding = reporting.evaluation_binding(
        {"artifact_sha256": "freeze"}, subject, {"artifact_sha256": "execution"}
    )
    value = {**binding, "status": "interrupted", "nmse": None}
    assert reporting.checked_evaluation(value, subject, binding) == value
    with pytest.raises(ValueError, match="another frozen subject"):
        reporting.checked_evaluation(
            {**value, "frozen_subject_sha256": "other"}, subject, binding
        )
    updated = subject.model_dump(mode="json")
    updated.update(
        private_metrics_opened_after_freeze=True, target_prediction={"status": "failed"}
    )
    value = {**binding, "status": "failed", "nmse": None, "subject": updated}
    assert reporting.checked_evaluation(value, subject, binding) == value
    updated["parameterization"]["global_parameters"]["a"] = 1.2
    with pytest.raises(ValueError, match="changed the frozen model"):
        reporting.checked_evaluation(value, subject, binding)


def test_completed_demo_still_checks_runtime_identity(tmp_path):
    from autoformalism.rebuttal.review_deadline_demo import run

    sealed_write(tmp_path / "plan.json", {"protocol": "another-protocol"})
    sealed_write(tmp_path / "result.json", {"status": "complete"})
    with pytest.raises(ValueError, match="runtime changed"):
        run(tmp_path)
