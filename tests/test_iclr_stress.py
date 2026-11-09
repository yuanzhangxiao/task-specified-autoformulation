"""Verify acquisition, generator parity, freeze boundaries, and rollout metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_stress_data as data
from scripts import iclr_stress_evaluation as analyze


def example():
    t = np.arange(17, dtype=float)
    meal = np.maximum(0, 3 - abs(t - 6))
    return {
        "trajectory_id": "train_000",
        "time": t.tolist(),
        "targets": {"y": list(t + 1)},
        "auxiliaries": {"a": list(np.full_like(t, 8.0))},
        "external_inputs": {"u": list(meal)},
    }


def test_downsampling_matches_all_forcing_knots_and_integrals():
    r = example()
    s = data.acquire(r, scales={}, stride=4)
    assert len(s["time"]) < len(r["time"])
    assert {3, 6, 9} <= set(s["time"])
    np.testing.assert_allclose(
        np.interp(r["time"], s["time"], s["external_inputs"]["u"]),
        r["external_inputs"]["u"],
    )
    assert np.trapezoid(r["external_inputs"]["u"], r["time"]) == pytest.approx(
        np.trapezoid(s["external_inputs"]["u"], s["time"])
    )


def test_paired_noise_and_no_measured_target_after_initialization():
    r = example()
    scales = data.train_scales([r])
    assert scales["a"] == 0
    low, high = (data.acquire(r, fraction=f, scales=scales) for f in (0.01, 0.1))
    np.testing.assert_allclose(
        np.subtract(high["targets"]["y"], r["targets"]["y"]),
        10 * np.subtract(low["targets"]["y"], r["targets"]["y"]),
    )
    assert low["external_inputs"] == high["external_inputs"] == r["external_inputs"]
    assert low["auxiliaries"] == r["auxiliaries"]
    tr = analyze.public_input(
        r, {"noise": 0.1, "stride": 4, "scales": scales, "mask": []}
    )
    assert not tr.auxiliaries and np.ptp(tr.targets["y"]) == 0
    assert tr.targets["y"][0] == high["targets"]["y"][0]


def test_sampler_matches_original_notebook_record_to_roundoff():
    truth = json.loads(
        Path(
            "data_raw/benchmark6_alien_device/private/selected_system_spec.json"
        ).read_text()
    )
    generated = data.sample_system(20260729, 28, 5)

    def eq(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for k, v in a.items():
                eq(v, b[k])
        elif isinstance(a, list):
            for v, w in zip(a, b, strict=True):
                eq(v, w)
        elif isinstance(a, float):
            assert a == pytest.approx(b, abs=1e-14)
        else:
            assert a == b

    eq(truth, generated)
    assert len(data.sample_system(12, 1, 11)["skew"]) == 11


def test_sensor_projection_changes_no_other_semantics(tmp_path):
    from scripts.iclr_stress import DEFAULT_RELEASE

    if not DEFAULT_RELEASE.is_dir():
        pytest.skip("public Phase C release unavailable")
    cells = data.accuracy_conditions(DEFAULT_RELEASE, tmp_path)
    sensor = [c for c in cells if c["family"] == "cstr"]
    assert [c["mask"] for c in sensor] == [[], ["C"], ["Tj"], ["C", "Tj"]]
    rows = [
        read_seal(tmp_path / "public" / c["cell"] / "train.json")["rows"]
        for c in sensor
    ]
    prompts = [
        read_seal(tmp_path / "public" / c["cell"] / "specification.json")[
            "public_prompt"
        ]
        for c in sensor
    ]
    assert len({p.split("B. Available data")[0] for p in prompts}) == 1
    assert len({p.split("C. Modeling requirements")[1] for p in prompts}) == 1
    for i, r in enumerate(rows[0]):
        for masked in rows[1:]:
            assert (
                r["targets"] == masked[i]["targets"]
                and r["external_inputs"] == masked[i]["external_inputs"]
            )


def test_frozen_cohort_missing_seed_or_wrong_cell_seal_is_refused():
    manifest = {
        "cells": [
            {
                "cell": "abc",
                "public_spec_sha256": "hash",
                "public_files": {"train.json": "foo"},
            }
        ],
        "arms": ["full", "fixed_gamma"],
        "algorithm_seeds": [0, 1, 2],
    }
    rows = [
        {
            "benchmark_id": "abc",
            "method": arm,
            "repetition": s,
            "public_spec_sha256": "hash",
            "public_file_sha256": {"train.json": "foo"},
            "status": "ready",
        }
        for arm in manifest["arms"]
        for s in manifest["algorithm_seeds"]
    ]
    bundle = {
        "protocol": "iclr-controlled-search-models-1",
        "selection_freeze_sha256": "c" * 64,
        "test_data_opened": False,
        "private_reference_opened": False,
        "plan_sha256": "a" * 64,
        "rows": rows,
    }
    assert len(analyze.select_endpoints(manifest, bundle)) == 6
    with pytest.raises(ValueError, match="missing paired"):
        analyze.select_endpoints(manifest, {**bundle, "rows": rows[:-1]})
    with pytest.raises(ValueError, match="specification differs"):
        analyze.select_endpoints(
            manifest,
            {**bundle, "rows": [{**rows[0], "public_spec_sha256": "wrong"}, *rows[1:]]},
        )


def test_missing_models_remain_in_report(tmp_path):
    manifest = {
        "artifact_sha256": "test",
        "cells": [{"cell": "abc", "study": "data_quality"}],
        "runs": [{"benchmark_id": "abc", "arm": "full", "repetition": 0}],
    }
    analyze.report(manifest, [], tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["planned"] == 1 and summary["complete"] == 0
    bundle = sealed_write(
        tmp_path / "bad.json",
        {"protocol": "wrong", "manifest_sha256": "test", "rows": []},
    )
    assert bundle["artifact_sha256"]
    with pytest.raises(ValueError, match="different experiment"):
        analyze.report(manifest, [tmp_path / "bad.json"], tmp_path)


def test_toy_paired_effect_and_clean_scoring():
    from scipy.integrate import cumulative_trapezoid

    r = example()
    u = r["external_inputs"]["u"]
    r["targets"]["y"] = (cumulative_trapezoid(u, r["time"], initial=0) + 3).tolist()
    r["auxiliaries"] = {}
    q = {
        **r,
        "targets": {"y": [3.0] * len(u)},
        "external_inputs": {"u": [0.0] * len(u)},
    }
    e = {
        "candidate": {
            "candidate_id": "integrator",
            "parent_candidate_id": None,
            "states": [{"name": "y", "kind": "observed"}],
            "state_equations": [{"state": "y", "rhs": "u"}],
            "observation_mappings": [{"channel": "y", "expression": "y"}],
            "initial_conditions": [
                {"state": "y", "scope": "global", "expression": "y"}
            ],
        },
        "context": {"targets": ["y"], "external_inputs": ["u"]},
        "parameters": {},
        "initials": {},
    }
    result = analyze.score(
        e,
        {"noise": 0, "stride": 4, "mask": []},
        [{"case": "pulse", "perturbed": r, "control": q}],
        {"y": 2},
    )
    assert result["test_nmse"] < 1e-12 and result["response_rse"] < 1e-12
    e["context"]["external_inputs"] = ["forbidden"]
    blocked = analyze.score(
        e, {"noise": 0}, [{"case": "pulse", "perturbed": r, "control": q}], {"y": 2}
    )
    assert blocked["evaluation_status"] == "unresolved" and blocked["test_nmse"] is None


def test_allocation_rows_separate_array_headers_steps_and_retries(tmp_path):
    from scripts.iclr_stress_resources import allocation_gpus, collect, cpu_seconds

    plan = tmp_path / "plan.json"
    sealed_write(
        plan,
        {
            "protocol": "iclr-controlled-search-1",
            "blocks": [
                {
                    "block": 0,
                    "cell": "A",
                    "seed": 0,
                    "tasks": [{"treatment": "full"}, {"treatment": "fixed_gamma"}],
                }
            ],
        },
    )
    source = tmp_path / "usage.psv"
    source.write_text(
        "JobID|ElapsedRaw|TotalCPU|AllocCPUS|AllocTRES|State\n"
        + "13_0|3600|1-01:20:00|4|cpu=4,gres/gpu=2,gres/gpu:a100=2|COMPLETED\n"
        + "13_0.batch|3600|1-01:20:00|4|cpu=4,gres/gpu=2|COMPLETED\n"
    )
    rows = collect(plan, [source, source], tmp_path / "cost.csv")
    assert len(rows) == 1 and rows[0]["allocated_gpu_hours"] == 2
    assert rows[0]["measured_cpu_seconds"] == 91200
    assert allocation_gpus("cpu=4") == 0 and allocation_gpus("") is None
    assert cpu_seconds("01:30") == 90


def test_complete_toy_cohort_exports_and_resumes_without_private_benchmark_data(
    tmp_path, monkeypatch
):
    class Toy:
        """Analytic test from a time-integrated synthetic input."""

        @staticmethod
        def pairs(*args):
            from scipy.integrate import cumulative_trapezoid

            row = example()
            row["auxiliaries"] = {}
            u = row["external_inputs"]["u"]
            row["targets"] = {
                "y": (3 + cumulative_trapezoid(u, row["time"], initial=0)).tolist()
            }
            ctrl = {
                **row,
                "targets": {"y": [3.0] * len(u)},
                "external_inputs": {"u": [0.0] * len(u)},
            }
            return [{"case": "square", "perturbed": row, "control": ctrl}]

    public_row = Toy.pairs()[0]["perturbed"]
    root = tmp_path / "release"
    cell = data.write_cell(
        root,
        "toy",
        {
            "targets": ["y"],
            "auxiliaries": [],
            "external_inputs": ["u"],
            "fixed_covariates": [],
            "public_prompt": "Synthetic test only",
        },
        [public_row],
        [public_row],
    )
    cell.update(
        family="synthetic",
        study="scaling",
        dimension=2,
        variant="dimension",
        generator_seed=4,
    )
    arms = ["full", "fixed_gamma"]
    manifest = {
        "cells": [cell],
        "arms": arms,
        "algorithm_seeds": [0, 1, 2],
        "artifact_sha256": "fake-design",
        "runs": [
            {"benchmark_id": "toy", "arm": a, "repetition": s}
            for a in arms
            for s in range(3)
        ],
    }
    candidate = {
        "candidate_id": "toy",
        "parent_candidate_id": None,
        "states": [{"name": "y", "kind": "observed"}],
        "state_equations": [{"state": "y", "rhs": "u"}],
        "observation_mappings": [{"channel": "y", "expression": "y"}],
        "initial_conditions": [{"state": "y", "scope": "global", "expression": "y"}],
    }
    rows = [
        {
            "benchmark_id": "toy",
            "public_spec_sha256": cell["public_spec_sha256"],
            "public_file_sha256": cell["public_files"],
            "method": a,
            "repetition": s,
            "status": "ready",
            "context": {"targets": ["y"], "external_inputs": ["u"]},
            "candidate": candidate,
            "parameters": {},
            "initials": {},
            "cost": {"fit_seconds": 3.0},
        }
        for a in arms
        for s in range(3)
    ]
    for i, r in enumerate(rows):
        r["source_id"] = f"task{i:02}"
    plan = sealed_write(
        tmp_path / "plan.json",
        {"blocks": [{"tasks": [{"task_id": r["source_id"]} for r in rows]}]},
    )
    receipt = sealed_write(
        tmp_path / "evaluation_freeze.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "roster_sha256": analyze.public.content_sha256(rows),
        },
    )
    models = tmp_path / "frozen.json"
    sealed_write(
        models,
        {
            "protocol": "iclr-controlled-search-models-1",
            "selection_freeze_sha256": receipt["artifact_sha256"],
            "test_data_opened": False,
            "private_reference_opened": False,
            "plan_sha256": plan["artifact_sha256"],
            "rows": rows,
        },
    )
    monkeypatch.setattr(analyze, "build_references", Toy.pairs)
    dest = tmp_path / "eval"
    analyze.evaluate(root, manifest, models, tmp_path, dest)
    snapshot = (dest / "evaluation.json").read_bytes()
    results = json.loads(snapshot)["rows"]
    assert len(results) == 6 and all(r["test_nmse"] < 1e-12 for r in results)
    monkeypatch.setattr(
        analyze, "build_references", lambda *a: pytest.fail("reference was regenerated")
    )
    analyze.evaluate(root, manifest, models, tmp_path, dest)
    assert (dest / "evaluation.json").read_bytes() == snapshot
    analyze.report(manifest, [dest / "evaluation.json"], tmp_path / "report")
    assert (tmp_path / "report/scaling-cost-curves.pdf").is_file()

    changed = analyze.sealed_read(models)
    changed.pop("artifact_sha256")
    changed["rows"] = changed["rows"][:-1]
    sealed_write(tmp_path / "changed.json", changed)
    with pytest.raises(ValueError, match="changed frozen search roster"):
        analyze.evaluate(
            root, manifest, tmp_path / "changed.json", tmp_path, tmp_path / "invalid"
        )
    assert not (tmp_path / "invalid" / "references").exists()
