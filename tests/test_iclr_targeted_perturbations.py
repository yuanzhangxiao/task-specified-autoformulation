"""Pinned public inputs, unchanged search policy, six-way launch and portable export."""

import hashlib
import json
import zipfile
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_aces_pilot as pilot
from scripts import iclr_targeted_perturbations as run


def write_cell(folder, *, reactor=False):
    folder.mkdir(parents=True)
    prompt = (
        "A. Task specification\nThe primary objective is to recover the following "
        "task-required mechanism:\n- "
        + (
            "a reactor-temperature balance accounting for feed transport, reaction "
            "heat generation, and jacket-mediated heat removal."
            if reactor
            else "a meal pathway."
        )
        + "\n\nThe dimensionality and internal representation of this mechanism is "
        "not specified and must be inferred.\n\nB. Available data\nPublic channels.\n"
        "C. Modeling requirements\nShared parameters.\nF. Required response\nModel.\n"
    )
    channels = {
        "targets": ["T"] if reactor else ["Gp"],
        "auxiliaries": [] if reactor else ["Gt"],
        "external_inputs": ["Cf", "Tf", "Tjf"] if reactor else ["meal"],
        "fixed_covariates": [],
    }
    seal(
        folder / "specification.json",
        {
            "protocol": "phase-c-development-1",
            "benchmark_id": folder.name,
            "public_prompt": prompt,
            "judge_prompt": "unchanged critic text",
            **channels,
        },
    )
    # The supplied CSTR text has one more trailing newline than its sealed spec.
    (folder / "proposer_prompt.txt").write_text(prompt + ("\n" if reactor else ""))
    for split in ("train", "val"):
        seal(
            folder / f"{split}.json",
            {
                "name": split,
                "fingerprint": split,
                "rows": [
                    {
                        "trajectory_id": split + "_00",
                        "time": [0, 0.1, 0.2, 0.3],
                        **{
                            k: {c: [1, 2, 3, 4] for c in cs}
                            for k, cs in channels.items()
                        },
                    }
                ],
            },
        )


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    original, reactor = (tmp_path / "original" / n for n in (run.T1, run.CSTR))
    write_cell(original)
    write_cell(reactor, reactor=True)
    old_spec = read_seal(original / "specification.json")
    prompt = (
        old_spec["public_prompt"]
        .replace(
            "following task-required mechanism:", "following task-required mechanisms:"
        )
        .replace(
            "\n\nThe dimensionality and internal representation of this mechanism is",
            "\n- "
            + run.search.controls.T1_EXCHANGE_CLAUSE
            + "\n\nThe dimensionality and internal representation "
            "of these mechanisms are",
        )
    )
    expected_spec = tmp_path / "specification.json"
    seal(expected_spec, {**old_spec, "public_prompt": prompt})
    original_files = run.public_files(original)
    monkeypatch.setattr(run, "FILES", original_files)
    expected = {
        run.CSTR: run.public_files(reactor),
        run.T1: {
            **original_files,
            "proposer_prompt.txt": hashlib.sha256(prompt.encode()).hexdigest(),
            "specification.json": run.search.file_digest(expected_spec),
        },
    }
    archive = tmp_path / "handoff.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in original_files:
            z.write(reactor / name, run.ARCHIVE_PREFIX + f"public/{run.CSTR}/{name}")
        z.writestr(
            run.ARCHIVE_PREFIX + "t1_exchange_required/proposer_prompt.txt", prompt
        )
        z.writestr(run.ARCHIVE_PREFIX + "private/audit.json", "private-law-canary")
        z.writestr(run.ARCHIVE_PREFIX + "../../test.json", "test-canary")
    cfg = run.config()
    cfg.update(handoff_sha256=run.search.file_digest(archive), public_files=expected)
    cfgpath = tmp_path / "config.json"
    cfgpath.write_text(json.dumps(cfg))
    monkeypatch.setattr(run, "CONFIG_PATH", cfgpath)
    destination = tmp_path / "public"
    run.stage(archive, original, destination)
    return destination, archive, original


@pytest.fixture
def prepared(inputs, tmp_path, monkeypatch):
    public, _, _ = inputs
    root, anchor, gate = (tmp_path / n for n in ("search", "anchor", "gate"))
    policy = run.critic.judge.judge_protocol(sign_policy=run.critic.SIGN_POLICY)
    policy.pop("distinct_seed_attempts")
    calibration = sealed_write(
        gate / "plan.json",
        {
            "scientific_protocol": policy,
            "transport_policy": run.critic.jetstream.POLICY,
            "output_contract": run.critic.jetstream.OUTPUT_CONTRACT,
            "endpoint": run.critic.jetstream.ENDPOINT,
        },
    )
    (gate / "summary.json").write_text(
        json.dumps(
            {
                "protocol": run.critic.CALIBRATION_PROTOCOL,
                "identity": calibration["artifact_sha256"],
                "known_case_gate_passed": True,
                "status": "complete",
                "planned_orientations": 140,
                "test_data_opened": False,
            }
        )
    )
    original = run.search.prepare(
        anchor,
        public,
        [run.CSTR],
        ["full", "fixed_gamma"],
        [0, 1, 2],
        15,
        True,
        True,
        (gate / "plan.json", gate / "summary.json"),
    )
    monkeypatch.setattr(run, "ANCHOR_PLAN", original["artifact_sha256"])
    run.prepare(root, public, anchor, gate)
    return root, public, anchor, gate


def test_input_allowlist_and_exact_prompt_only_change(inputs):
    dest, archive, original = inputs
    assert (
        run.stage(archive, original, dest)["public_files"]
        == run.config()["public_files"]
    )
    before, after = (
        read_seal(p / "specification.json") for p in (original, dest / run.T1)
    )
    assert before.pop("public_prompt") != after.pop("public_prompt")
    assert before == after
    assert not (dest / "private").exists() and not (dest.parent / "test.json").exists()
    for p in dest.rglob("*"):
        if p.is_file():
            assert "private-law-canary" not in p.read_text()
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="wrong targeted handoff"):
        run.stage(archive, original, dest)


def test_search_matches_anchor_and_preserves_runtime_calibration(prepared, monkeypatch):
    root, public, anchor, gate = prepared
    plan = run.verify(root)
    assert run.prepare(*prepared) == plan
    original = sealed_read(anchor / "plan.json")
    assert {k: plan[k] for k in run.CONFIG} == {k: original[k] for k in run.CONFIG}
    assert len(plan["blocks"]) == 6 and plan["maximum_parallel_blocks"] == 6
    assert plan["arms"] == ["full"] and plan["require_fitted_full_start"]
    assert [(b["cell"], b["seed"]) for b in plan["blocks"]] == [
        (c, s) for c in run.CELLS for s in (0, 1, 2)
    ]
    for b in plan["blocks"]:
        (task,) = b["tasks"]
        ref = next(
            t
            for t in original["blocks"][b["seed"]]["tasks"]
            if t["treatment"] == "full"
        )
        for field in (
            "arm",
            "treatment",
            "critic",
            "scientific_verifier",
            "shared_processes",
            "secondary",
        ):
            assert task[field] == ref[field]
    assert (
        run.critic.checked_authorization(root, plan)["calibration_plan_sha256"]
        == (sealed_read(gate / "plan.json")["artifact_sha256"])
    )
    with pytest.raises(ValueError, match="complete every planned endpoint"):
        run.finish(root)
    assert not (root / "frozen_models.json").exists()
    actual_runtime = run.public._runtime()
    monkeypatch.setattr(
        run.public, "_runtime", lambda: {**actual_runtime, "python": "different"}
    )
    with pytest.raises(ValueError, match="Anchor fitting runtime required"):
        run.prepare(*prepared)
    monkeypatch.setattr(run.public, "_runtime", lambda: actual_runtime)
    (public / run.T1 / "train.json").write_text("changed")
    with pytest.raises(ValueError, match="public files differ"):
        run.prepare(*prepared)


def test_exchange_obligation_is_visible_and_old_t1_is_unchanged(inputs):
    directory, _, original = inputs
    # Construct through the same treatment API used by proposer, critic and verifier.
    spec = read_seal(directory / run.T1 / "specification.json")
    context = {
        k: spec[k]
        for k in ("targets", "auxiliaries", "external_inputs", "fixed_covariates")
    }
    cell = {
        "benchmark_id": run.T1,
        "tier": "public",
        "context": context,
        "prompt": spec["public_prompt"],
        "limits": {},
        "meal_input": "meal",
    }
    current = run.search.controls.treatment(cell, "full")
    assert [r["id"] for r in current["brief"]["requirements"]] == [
        "meal_pathway",
        "plasma_tissue_exchange",
    ]
    assert current["brief"]["requirements"][1]["drivers"] == ["Gt"]
    candidate = {
        "candidate_id": "fixture",
        "parent_candidate_id": None,
        "states": [{"name": "Gp", "kind": "observed", "description": "plasma mass"}],
        "state_equations": [{"state": "Gp", "rhs": "meal"}],
        "observation_mappings": [{"channel": "Gp", "expression": "Gp"}],
        "initial_conditions": [{"state": "Gp", "scope": "global", "expression": "Gp"}],
    }
    bundle = {"candidate": candidate, "initialization": {"context": context}}
    task = {"scientific_verifier": True}
    assert not run.search.controls.certificate(bundle, current, task)[
        "eligible_for_development_selection"
    ]
    candidate["state_equations"][0]["rhs"] = "meal + Gt"
    assert run.search.controls.certificate(bundle, current, task)[
        "eligible_for_development_selection"
    ]
    old = run.search.controls.treatment(
        {**cell, "prompt": read_seal(original / "specification.json")["public_prompt"]},
        "full",
    )
    assert len(old["brief"]["requirements"]) == 1
    with pytest.raises(ValueError, match="tissue channel"):
        run.search.controls.treatment(
            {**cell, "context": {**context, "auxiliaries": []}}, "full"
        )
    with pytest.raises(ValueError, match="operational bindings differ"):
        run.search.controls.treatment(
            {
                **cell,
                "prompt": cell["prompt"].replace(
                    "tissue glucose pool", "unknown process"
                ),
            },
            "full",
        )


def test_six_parallel_submissions_are_resumable(prepared, monkeypatch):
    root, *_ = prepared
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "fixture")
    monkeypatch.setenv("USER", "fixture")
    monkeypatch.setattr(pilot, "pending_ids", lambda _: list(range(6)))
    calls = []

    def submit(folder, key, args, *rest):
        calls.append(args)
        (folder / "array.id").write_text("1234\n")
        return "1234"

    monkeypatch.setattr(pilot, "submit_job", submit)
    assert pilot.submit(root, "wave1", "156264627414", 6) == "1234"
    assert pilot.submit(root, "wave1", "156264627414", 6) == "1234"
    assert len(calls) == 1
    assert "--array=0,1,2,3,4,5%6" in calls[0]
    assert "--job-name=iclr-targeted-20b" in calls[0]
    assert "--exclude=ac042,ac096" in calls[0]
    assert "--time=04:00:00" in calls[0]
    # Retry the same six pending blocks only after the failed wave has exited.
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *a, **kw: "")
    assert pilot.submit(root, "wave2", "156264627414", 6) == "1234"
    assert "--exclude=ac042,ac096" in calls[1]
    assert "--time=04:00:00" in calls[1]
    assert "--array=0,1,2,3,4,5%6" in calls[1]
    for limit in (0, 7, True):
        with pytest.raises(ValueError, match="concurrency"):
            pilot.submit(root, "wave2", "156264627414", limit)


def test_endpoint_export_keeps_failures_and_all_receipts(
    prepared, monkeypatch, tmp_path
):
    root, *_ = prepared
    plan = run.verify(root)
    rows = []
    for b in plan["blocks"]:
        row = {
            "benchmark_id": b["cell"],
            "method": "full",
            "repetition": b["seed"],
            "source_id": b["tasks"][0]["task_id"],
            "source_sha256": "a" * 64,
            "public_file_sha256": run.config()["public_files"][b["cell"]],
            "status": "no_retained_model",
        }
        if b["seed"] != 1:
            candidate = {"initial_conditions": [{"state": "z", "expression": "k"}]}
            row.update(
                candidate=candidate,
                context={"targets": ["T"]},
                parameters={"k": 1.0},
                initials={},
                validation_nmse=0.3,
                model_sha256=run.public.content_sha256(candidate),
                status="ready",
            )
        rows.append(row)
    freeze = sealed_write(
        root / "evaluation_freeze.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "roster_sha256": run.public.content_sha256(rows),
        },
    )
    bundle = sealed_write(
        root / "frozen_models.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "selection_freeze_sha256": freeze["artifact_sha256"],
            "rows": rows,
        },
    )
    for name in ("paired_ablation_results.csv", "search_visits.csv"):
        run.search.write_csv(root / name, [{"status": r["status"]} for r in rows])
    (root / "endpoint-audit").mkdir()
    (root / "endpoint-audit/visits.csv").write_text("receipt\nfixture\n")
    monkeypatch.setattr(run.search, "report", lambda *args: bundle)
    monkeypatch.setattr(
        run, "export_receipts", lambda *args: {"counts": {"visits": 90}}
    )
    with zipfile.ZipFile(run.finish(root)) as z:
        assert z.testzip() is None and "endpoint-audit/visits.csv" in z.namelist()
        packed = json.loads(z.read("frozen_models.json"))
        assert packed["rows"] == rows and len(rows) == 6
        for line in z.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    wrong = deepcopy(bundle)
    wrong["rows"] = wrong["rows"][:-1]
    monkeypatch.setattr(run.search, "report", lambda *args: wrong)
    with pytest.raises(ValueError):
        run.finish(root)


def test_reference_controls_have_separate_plans_and_resumable_outputs(
    inputs, tmp_path, monkeypatch
):
    from scripts import iclr_perturbed_cstr_refits as refits

    public, _, _ = inputs
    root = tmp_path / "references"
    plan = refits.prepare(root, public)
    assert plan == refits.prepare(root, public)
    assert len(plan["tasks"]) == 9 and plan["llm_calls"] == 0
    assert not plan["test_data_opened"]
    calls = []

    def fit(cell, train, val, start, max_evals, folder, **kwargs):
        calls.append((start, kwargs))
        assert cell == refits.reference.CSTR and max_evals == 50
        assert train.name == "train" and val.name == "val"
        if start == 2:
            raise ValueError("numerical failure fixture")
        return {
            "status": "ready",
            "scores": {"training": [0.2], "validation": [0.3 - 0.1 * start]},
            "theta": [0.0] * 9,
            "coefficient_names": list(plan["coefficient_scales"]),
            "optimizer_converged": False,
            "nfev": 50,
            "max_grid_discrepancy": 0.0,
        }

    monkeypatch.setattr(refits.reference, "fit_start", fit)
    assert refits.run(root, 0) == refits.run(root, 0) and len(calls) == 1
    with pytest.raises(ValueError, match="all nine"):
        refits.finish(root)
    for i in range(1, 9):
        refits.run(root, i)
    assert len(calls) == 9
    assert calls[0][1] == {"bound_width": 0.5, "distributed_start": False}
    assert calls[3][1]["bound_width"] > 1 and calls[6][1]["distributed_start"]
    with zipfile.ZipFile(refits.finish(root)) as z:
        selected = json.loads(z.read("frozen_models.json"))
        assert [r["fit"]["task"]["start"] for r in selected["rows"]] == [1, 1, 1]
        assert len([n for n in z.namelist() if n.endswith("/result.json")]) == 9
        assert not any("test.json" in n for n in z.namelist())
    actual = refits.identity()
    monkeypatch.setattr(
        refits, "identity", lambda: {**actual, "runtime": {"python": "other"}}
    )
    with pytest.raises(ValueError, match="runtime changed"):
        refits.run(root, 0)


def test_reference_submission_is_cpu_only_and_refuses_uncertain_repeat(
    inputs, tmp_path, monkeypatch
):
    from scripts import iclr_perturbed_cstr_refits as refits

    root = tmp_path / "references"
    refits.prepare(root, inputs[0])
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "fixture")
    calls = []

    def submit(folder, key, args, *rest):
        calls.append(args)
        (folder / "array.intent.json").write_text("{}")
        raise ValueError("scheduler acknowledgment unconfirmed")

    monkeypatch.setattr(refits, "submit_job", submit)
    with pytest.raises(ValueError, match="unconfirmed"):
        refits.submit(root, "wave1", "bibo-delta-cpu", "delta")
    with pytest.raises(ValueError, match="uncertain Slurm"):
        refits.submit(root, "wave1", "bibo-delta-cpu", "delta")
    assert len(calls) == 1
    assert "--array=0,1,2,3,4,5,6,7,8%9" in calls[0] and "--partition=cpu" in calls[0]
    assert not any("gpu" in x for x in calls[0])
    import os

    assert "AF_JETSTREAM_API_KEY" not in os.environ


def test_reference_fit_original_and_expanded_bounds_smoke(tmp_path):
    import numpy as np

    from autoformalism.schemas.public_fitting import PublicSplit
    from scripts import iclr_reference_fits as reference

    row = {
        "trajectory_id": "toy",
        "time": [0, 0.1, 0.2, 0.3, 0.4],
        "targets": {"T": [365, 366, 368, 369, 370]},
        "external_inputs": {"Cf": [1.2] * 5, "Tf": [350] * 5, "Tjf": [330] * 5},
    }
    train, val = [
        PublicSplit.model_validate({"name": s, "fingerprint": "fixture", "rows": [row]})
        for s in ("train", "val")
    ]
    for i, kwargs in enumerate(
        ({}, {"bound_width": float(np.log(4)), "distributed_start": True})
    ):
        folder = tmp_path / str(i)
        folder.mkdir()
        result = reference.fit_start(reference.CSTR, train, val, 0, 1, folder, **kwargs)
        assert result["nfev"] == 1 and np.isfinite(result["scores"]["validation"]).all()
        assert len(result["theta"]) == 9
        repeated = reference.fit_start(
            reference.CSTR, train, val, 0, 1, folder, **kwargs
        )
        assert (
            repeated["theta"] == result["theta"]
            and repeated["scores"] == result["scores"]
        )
        if not kwargs:
            assert np.array_equal(
                result["theta"], np.random.default_rng(800).uniform(-0.1, 0.1, 9)
            )
    with pytest.raises(ValueError, match="finite"):
        reference.fit_start(
            reference.CSTR, train, val, 0, 1, tmp_path, bound_width=float("nan")
        )
