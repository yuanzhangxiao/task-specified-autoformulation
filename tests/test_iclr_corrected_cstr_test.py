"""The corrected CSTR test must follow complete selection on matching training data."""

from __future__ import annotations

from itertools import product

import pytest

from scripts import iclr_corrected_cstr_test as ct


def cohort():
    tasks = []
    for seed, prompt, c, v in product(
        (0, 1), ("full", "brief_only"), (False, True), (False, True)
    ):
        tasks.append(
            {
                "cell": ct.CSTR,
                "seed": seed,
                "arm": prompt,
                "critic": c,
                "scientific_verifier": v,
                "task_id": f"cell04_seed{seed}_{prompt}_c{int(c)}v{int(v)}s1",
                "block": 16 + 2 * seed + (prompt == "brief_only"),
                "shared_processes": True,
            }
        )
    plan = {
        "tasks": tasks,
        "artifact_sha256": "a" * 64,
        "cells": {ct.CSTR: {"assets": dict(ct.EXPECTED)}},
    }
    receipt = {
        "plan_sha256": plan["artifact_sha256"],
        "artifact_sha256": "b" * 64,
        "subjects_sha256": "c" * 64,
        "roster": [
            {
                "subject_id": t["task_id"] + "_final",
                "task": t,
                "endpoint": "final",
                "model_available": False,
            }
            for t in tasks
        ],
        "test_data_opened": False,
        "further_search_permitted": False,
    }
    return plan, receipt


def test_freeze_requires_both_prompts_seeds_and_all_treatments():
    plan, receipt = cohort()
    ct.check_roster(plan, receipt)
    receipt["roster"].pop()
    with pytest.raises(ValueError, match="final selection"):
        ct.check_roster(plan, receipt)
    plan["tasks"].pop()
    with pytest.raises(ValueError, match="16 corrected"):
        ct.check_roster(plan, receipt)


def test_uncorrected_search_stops_before_private_arrays(tmp_path, monkeypatch):
    plan, _ = cohort()
    plan["cells"][ct.CSTR]["assets"]["train.csv"] = "wrong"
    monkeypatch.setattr(ct.campaign, "verify", lambda *a, **k: plan)
    ct.sealed_write(
        tmp_path / "stage/correction.json",
        {
            "protocol": "iclr-submitted-cstr-correction-1",
            "plan_sha256": ct.PLAN,
            "test_opened": False,
            "public_files": {
                f"phase_b_v1/{ct.CSTR}/{n}": v for n, v in ct.EXPECTED.items()
            },
        },
    )
    monkeypatch.setattr(
        ct, "split_rows", lambda *a: pytest.fail("private generator opened")
    )
    with pytest.raises(ValueError, match="development source"):
        ct.run(
            tmp_path / "campaign",
            tmp_path / "stage",
            tmp_path / "evaluation",
            tmp_path / "private",
        )


def write_examples(path, label, number, slope=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("trajectory_id,t,T,C,Tj,Cf,Tf,Tjf\n")
        for i in range(number):
            for t in range(3):
                temp = 350 + t * slope
                f.write(f"{label}_{i:03d},{t / 10:.1f},{temp},0.3,310,1,350,330\n")
    return path


def test_projection_checks_all_channels_and_grid(tmp_path):
    a = write_examples(tmp_path / "source.csv", "train", 1)
    b = write_examples(tmp_path / "other.csv", "train", 1)
    assert ct.compare_development(a, b)["C"] == 0.0
    b.write_text(b.read_text().replace("0.3,310", "1,310"))
    with pytest.raises(ValueError, match="trajectories differ"):
        ct.compare_development(a, b)
    b.write_text(a.read_text().replace("train_000,0.2,", "train_000,0.3,"))
    with pytest.raises(ValueError, match="grids differ"):
        ct.compare_development(a, b)


def test_synthetic_postfreeze_all_endpoints_scored_or_accounted(tmp_path, monkeypatch):
    plan, receipt = cohort()
    row = receipt["roster"][0]
    row["model_available"] = True
    sid = row["subject_id"]
    subject = ct.FrozenEvaluationSubject.model_validate(
        {
            "subject_id": sid,
            "method": ct.campaign.method_name(row["task"], "final"),
            "benchmark_id": ct.CSTR,
            "tier": "easy",
            "repetition": 0,
            "private_metrics_opened_after_freeze": False,
            "source_provenance": {
                "adapter": "direct_candidate",
                "request_id": sid,
                "source_path": "fixture",
                "source_sha256": "a" * 64,
                "candidate_sha256": "a" * 64,
            },
            "candidate": {
                "candidate_id": "constant",
                "parent_candidate_id": None,
                "states": [{"name": "T", "kind": "observed"}],
                "state_equations": [{"state": "T", "rhs": "0"}],
                "observation_mappings": [{"channel": "T", "expression": "T"}],
                "initial_conditions": [
                    {"state": "T", "scope": "global", "expression": "T"}
                ],
            },
            "parameterization": {"status": "not_required"},
            "validation_context": {
                "targets": ["T"],
                "external_inputs": ["Cf", "Tf", "Tjf"],
                "auxiliaries": ["C", "Tj"],
            },
            "target_prediction": {"status": "missing"},
        }
    )
    monkeypatch.setattr(
        ct, "intake", lambda *args: (plan, receipt, {sid: subject}, "a" * 64)
    )
    monkeypatch.setattr(ct, "check_development", lambda *args: {})

    def test_data(directory, data_root, split, identity):
        assert split == "test"
        return write_examples(directory / "test.csv", "test", 6), {
            f"test_{i}": "a" * 64 for i in range(6)
        }

    monkeypatch.setattr(ct, "split_rows", test_data)
    root = tmp_path / "campaign"
    write_examples(root / "public" / "phase_b_v1" / ct.CSTR / "train.csv", "train", 2)
    args = (root, tmp_path / "stage", tmp_path / "metrics", tmp_path / "private")
    result = ct.run(*args)
    assert (
        len(result["rows"]) == 16
        and len([r for r in result["rows"] if r["status"] == "model_unavailable"]) == 15
    )
    scored = result["rows"][0]
    assert scored["successful_trajectory_count"] == 6 and scored["normalized_mse"] > 0
    monkeypatch.setattr(
        ct,
        "evaluate_subject_on_test",
        lambda *a, **k: pytest.fail("re-evaluated subject"),
    )
    assert ct.run(*args) == result
