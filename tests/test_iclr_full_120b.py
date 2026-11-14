"""The proposer-size comparison inherits the executed anchor protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_aces_pilot as scheduler
from scripts import iclr_full_120b as run


@pytest.fixture
def reference(tmp_path, monkeypatch):
    gate, source = tmp_path / "gate", tmp_path / "20b"
    critic = run.critic
    policy = critic.judge.judge_protocol(sign_policy=critic.SIGN_POLICY)
    policy.pop("distinct_seed_attempts")
    calibration = sealed_write(
        gate / "plan.json",
        {
            "scientific_protocol": policy,
            "transport_policy": critic.jetstream.POLICY,
            "output_contract": critic.jetstream.OUTPUT_CONTRACT,
            "endpoint": critic.jetstream.ENDPOINT,
        },
    )
    summary = {
        "protocol": critic.CALIBRATION_PROTOCOL,
        "identity": calibration["artifact_sha256"],
        "known_case_gate_passed": True,
        "status": "complete",
        "planned_orientations": 140,
        "test_data_opened": False,
    }
    (gate / "summary.json").write_text(json.dumps(summary))

    def cell(path, root, cfg):
        glucose = path.name == "acquisition_00"
        clauses = (
            "a meal response.\n- a delayed insulin-action pathway."
            if glucose
            else "a reactor-temperature balance including the jacket."
        )
        return {
            "benchmark_id": path.name,
            "tier": "hard",
            "limits": {},
            "meal_input": "meal" if glucose else None,
            "files": {},
            "public_spec_sha256": "fixture",
            "context": {
                "targets": ["Gp", "I"] if glucose else ["T"],
                "external_inputs": ["meal", "insulin_pmol_per_kg_min"]
                if glucose
                else ["Tf", "Tjf", "Cf"],
            },
            "prompt": (
                f"A. Task specification\n- {clauses}\n\nB. Available data\n"
                "Public readings.\nC. Modeling requirements\nUse shared parameters.\n"
                "F. Required response\nProvide equations."
            ),
        }

    monkeypatch.setattr(run.search, "read_cell", cell)
    plan = run.search.prepare(
        source,
        tmp_path / "public",
        list(run.CELLS),
        list(run.search.controls.ARMS),
        [0, 1, 2],
        15,
        True,
        True,
        (gate / "plan.json", gate / "summary.json"),
    )
    monkeypatch.setattr(run, "REFERENCE_PLAN", plan["artifact_sha256"])
    return source, gate


def test_full_projection_preserves_inputs_starts_budgets_and_critic(
    reference, tmp_path, monkeypatch
):
    source, gate = reference
    original = sealed_read(source / "plan.json")
    (source / "results").mkdir()
    (source / "results" / "old-state").write_text("20B checkpoint")
    dest = tmp_path / "120b"
    result = run.prepare(dest, source, gate)
    assert run.prepare(dest, source, gate) == result
    assert original == sealed_read(source / "plan.json")
    assert not (dest / "results").exists()
    for field in (
        "cells",
        "runtime",
        "fit_profile",
        "serving_image_sha256",
        "seeds",
        "visits",
        "max_parameters",
        "critic_enabled",
        "pruning_enabled",
    ):
        assert result[field] == original[field]
    assert {
        k
        for k, v in result["model_settings"].items()
        if v != original["model_settings"][k]
    } == {"model", "model_revision"}
    assert result["arms"] == ["full"] and result["proposer_tensor_parallel_size"] == 2
    assert result["require_fitted_full_start"] and result["visits"] == 15
    variants = sealed_read(source / "spec_variants.json")["variants"]
    assert sealed_read(dest / "spec_variants.json")["variants"] == {
        c: {"full": v["full"]} for c, v in variants.items()
    }
    assert result["blocks"] == [
        {**b, "tasks": [b["tasks"][0]]} for b in original["blocks"]
    ]
    assert [(b["cell"], b["seed"]) for b in result["blocks"]] == [
        (c, s) for c in run.CELLS for s in range(3)
    ]
    assert [b["tasks"][0]["index"] for b in result["blocks"]] == [0, 8, 16, 24, 32, 40]
    auth = [sealed_read(p / "critic_authorization.json") for p in (source, dest)]
    assert {k for k in auth[0] if auth[0][k] != auth[1][k]} == {
        "identity",
        "artifact_sha256",
    }
    task = result["blocks"][2]["tasks"][0]
    raw = sealed_read(dest / "cells" / (task["cell"] + ".json"))
    ctx = run.search.view(result, run.search.controls.treatment(raw, "full"))
    for visit in (0, 1):
        client = run.search.pipeline._client(
            dest, ctx, task, visit, "http://fixture", lambda: True
        )
        assert client.settings.model == run.MODELS["120b"]["model"]
        assert (
            client.settings.model_revision == run.MODELS["120b"]["revision"]
            and client.seed == 2
        )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/sh\necho 12345\n")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "fixture")
    monkeypatch.setattr(scheduler, "verify_critic_access", lambda _: None)
    assert (
        scheduler.submit(dest, "wave1", "account", 1, confirm_critic_access=True)
        == "12345"
    )
    assert scheduler.submit(dest, "wave1", "account", 1) == "12345"
    argv = json.loads((dest / "submissions/wave1/array.intent.json").read_text())[
        "argv"
    ]
    assert {"--array=0,1,2,3,4,5%1", "--gres=gpu:h100:2", "--mem=128G"} <= set(argv)
    assert Path(argv[-3]).name == "run_iclr_controlled_search_aces.sh"


def test_failed_full_start_remains_terminal_after_resume(reference, tmp_path):
    source, gate = reference
    root = tmp_path / "120b"
    plan = run.prepare(root, source, gate)
    task = plan["blocks"][0]["tasks"][0]
    first = sealed_write(
        io.round_path(root, task, 0) / "result.json",
        {
            "task": task,
            "round": 0,
            "status": "fit_failed",
            "selected": None,
            "trial": None,
            "cost": {},
        },
    )
    for _ in range(2):
        run.search.run_block(root, 0, "http://unused", 60)
        for i in range(1, 15):
            row = io.read_round(root, task, i)
            assert row["status"] == "shared_initial_unavailable" and row["cost"] == {}
            assert row["shared_initial_sha256"] == first["artifact_sha256"]
    assert not list((root / "results").rglob("calls"))
    assert scheduler.pending_ids(root) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="complete every planned endpoint"):
        run.finish(root, source)
    assert not (root / "evaluation_freeze.json").exists()


def test_refuses_different_reference_runtime_or_cell_before_scheduling(
    reference, tmp_path, monkeypatch
):
    source, gate = reference
    dest = tmp_path / "120b"
    with pytest.raises(ValueError, match="disjoint"):
        run.prepare(source, source, gate)
    with pytest.raises(ValueError, match="matched 120B"):
        run.finish(source, source)
    assert not (source / "evaluation_freeze.json").exists()
    with monkeypatch.context() as m:
        m.setattr(run.public, "_runtime", lambda: {"python": "changed"})
        with pytest.raises(ValueError, match="fitting runtime"):
            run.prepare(dest, source, gate)
    cell = source / "cells/acquisition_00.json"
    cell.unlink()
    sealed_write(cell, {"prompt": "other"})
    with pytest.raises(ValueError, match="differs from the 20B"):
        run.prepare(dest, source, gate)
    assert not dest.exists()


@pytest.mark.parametrize("cells", [None, ["sensor_00"]])
def test_comparison_keeps_planned_pairs_including_failures(reference, tmp_path, cells):
    source, gate = reference
    root = tmp_path / "120b"
    run.prepare(root, source, gate, cells=cells)
    for folder in (root, source):
        plan = sealed_read(folder / "plan.json")
        rows = []
        for b in reversed(plan["blocks"]):
            for task in b["tasks"]:
                row = {
                    "source_id": task["task_id"],
                    "benchmark_id": task["cell"],
                    "repetition": task["seed"],
                    "method": task["treatment"],
                    "public_file_sha256": {},
                    "status": "no_retained_model",
                }
                if folder == source or task["seed"] != 0:
                    row.update(
                        status="ready", validation_nmse=0.4 if folder == source else 0.3
                    )
                rows.append(row)
        freeze = sealed_write(
            folder / "evaluation_freeze.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "roster_sha256": run.public.content_sha256(rows),
            },
        )
        sealed_write(
            folder / "frozen_models.json",
            {
                "rows": rows,
                "plan_sha256": plan["artifact_sha256"],
                "selection_freeze_sha256": freeze["artifact_sha256"],
            },
        )
    pairs = run.paired_validation(root, source)
    count = 2 if cells is None else 1
    assert len(pairs) == 3 * count
    assert [r["delta_120b_minus_20b"] for r in pairs] == [
        None,
        pytest.approx(-0.1),
        pytest.approx(-0.1),
    ] * count


def test_subset_renumbers_scheduler_blocks_and_preserves_full_lineage(
    reference, tmp_path
):
    source, gate = reference
    root = tmp_path / "cstr-only"
    original = sealed_read(source / "plan.json")
    plan = run.prepare(root, source, gate, cells=["sensor_00"])
    assert run.prepare(root, source, gate, cells=["sensor_00"]) == plan
    assert set(plan["cells"]) == {"sensor_00"}
    assert len(plan["blocks"]) == 3
    assert scheduler.pending_ids(root) == [0, 1, 2]
    for i, block in enumerate(plan["blocks"]):
        task = block["tasks"][0]
        before = original["blocks"][i + 3]["tasks"][0]
        assert block["block"] == task["block"] == task["index"] == i
        assert task["task_id"] == f"block{i:04d}_full"
        assert plan["reference_tasks"][task["task_id"]] == before["task_id"]
        for key in ("seed", "critic", "scientific_verifier", "shared_processes"):
            assert task[key] == before[key]
    with pytest.raises(ValueError, match="distinct cells"):
        run.prepare(tmp_path / "wrong", source, gate, cells=["unrecognized"])
    with pytest.raises(ValueError, match="differs from the 20B"):
        run.prepare(tmp_path / "wrong", source, gate, reference_identity="0" * 64)
    assert not (tmp_path / "wrong").exists()


def test_ordered_case_blocks_later_launch_until_prior_is_frozen(
    reference, tmp_path, monkeypatch
):
    from scripts import iclr_full_120b_ordered as ordered

    source, gate = reference
    roster = [
        {"key": k, "cell": c, "reference": "anchors", "public_files": {}}
        for k, c in (("cstr-hard", "sensor_00"), ("t2-hard", "acquisition_00"))
    ]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cases": roster}))
    monkeypatch.setattr(ordered, "CONFIG", config)
    base = tmp_path / "ordered"
    for c in roster:
        plan = ordered.prepare_case(
            base, c["key"], source, tmp_path / "unused", gate, tmp_path / "public"
        )
        assert len(plan["blocks"]) == 3
        ordered.checked_case(base, c["key"])
    ordered.require_predecessors(base, "cstr-hard")
    with pytest.raises(ValueError, match="finish and export cstr-hard"):
        ordered.require_predecessors(base, "t2-hard")
    with pytest.raises(ValueError, match="complete every planned endpoint"):
        ordered.finish_case(base, "cstr-hard")
    assert not (base / "cstr-hard/evaluation_freeze.json").exists()
    root = base / "cstr-hard"
    plan = sealed_read(root / "plan.json")
    for block in plan["blocks"]:
        task = block["tasks"][0]
        directory = io.round_path(root, task, 0)
        proposal = sealed_write(
            directory / "proposal.json",
            {"task": task, "round": 0, "parent_sha256": None},
        )
        sealed_write(
            directory / "critic.json",
            {
                "identity": plan["artifact_sha256"],
                "task": task,
                "round": 0,
                "proposal_sha256": proposal["artifact_sha256"],
                "status": "skipped_no_candidate",
            },
        )
        sealed_write(
            directory / "result.json",
            {
                "task": task,
                "round": 0,
                "status": "fit_failed",
                "selected": None,
                "trial": None,
                "cost": {},
                "parent_sha256": None,
                "proposal_sha256": proposal["artifact_sha256"],
            },
        )
        run.search.run_block(root, block["block"], "http://unused", 60)
    reference_plan = sealed_read(source / "plan.json")
    reference_rows = [
        {
            "benchmark_id": b["cell"],
            "repetition": b["seed"],
            "method": t["treatment"],
            "source_id": t["task_id"],
            "status": "no_retained_model",
            "public_file_sha256": {},
        }
        for b in reference_plan["blocks"]
        for t in b["tasks"]
    ]
    freeze = sealed_write(
        source / "evaluation_freeze.json",
        {
            "plan_sha256": reference_plan["artifact_sha256"],
            "roster_sha256": run.public.content_sha256(reference_rows),
        },
    )
    sealed_write(
        source / "frozen_models.json",
        {
            "plan_sha256": reference_plan["artifact_sha256"],
            "selection_freeze_sha256": freeze["artifact_sha256"],
            "rows": reference_rows,
        },
    )
    assert ordered.finish_case(base, "cstr-hard").is_file()
    ordered.require_predecessors(base, "t2-hard")
    assert not (base / "t2-hard/evaluation_freeze.json").exists()


def test_ordered_new_public_case_authenticates_inputs_and_resumes(
    reference, tmp_path, monkeypatch
):
    from scripts import iclr_full_120b_ordered as ordered

    source, gate = reference
    config = tmp_path / "config.json"
    roster = [
        {
            "key": "new",
            "cell": "synthetic_case",
            "reference": "public",
            "public_files": {},
        }
    ]
    config.write_text(json.dumps({"cases": roster}))
    monkeypatch.setattr(ordered, "CONFIG", config)
    args = (
        tmp_path / "ordered",
        "new",
        source,
        tmp_path / "unused",
        gate,
        tmp_path / "public",
    )
    plan = ordered.prepare_case(*args)
    assert ordered.prepare_case(*args) == plan
    assert plan["model_settings"]["model"] == run.MODELS["120b"]["model"]
    assert plan["require_fitted_full_start"]
    assert (
        sealed_read(args[0] / "new/ordered_case.json")["has_matched_20b_run"] is False
    )
    roster[0]["public_files"] = {"train.json": "incorrect"}
    config.write_text(json.dumps({"cases": roster}))
    with pytest.raises((ValueError, FileNotFoundError)):
        ordered.prepare_case(*args)
