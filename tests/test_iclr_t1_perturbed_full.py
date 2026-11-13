"""Check the T1-only production roster and its portable scoring packet."""

import csv
import io
import json
import zipfile

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_t1_perturbed_full as run


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, public, anchor, gate = (
        tmp_path / n for n in ("search", "public", "anchor", "gate")
    )
    directory = public / run.CELL
    directory.mkdir(parents=True)
    files = {}
    for n in run.FILES:
        (directory / n).write_text("controlled test input")
        files[n] = run.search.file_digest(directory / n)
    monkeypatch.setattr(run, "FILES", files)

    def read_cell(path, dest, cfg):
        t1 = path.name == run.CELL
        prompt = (
            "- a meal pathway."
            if t1
            else "- a meal pathway.\n- a delayed insulin pathway."
        )
        return {
            "benchmark_id": path.name,
            "files": files,
            "public_spec_sha256": files["specification.json"],
            "tier": "public",
            "limits": {},
            "context": {
                "targets": ["Gp"] if t1 else ["Gp", "I"],
                "external_inputs": ["meal"]
                if t1
                else ["meal", "insulin_pmol_per_kg_min"],
            },
            "prompt": (
                f"A. Task specification\n{prompt}\n\nB. Available data\nPublic.\n"
                "C. Modeling requirements\nPredict.\nF. Required response\nModel."
            ),
        }

    monkeypatch.setattr(run.search, "read_cell", read_cell)
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
    ref = run.search.prepare(
        anchor,
        public,
        ["acquisition_00"],
        ["full", "fixed_gamma"],
        [0, 1, 2],
        15,
        True,
        True,
        (gate / "plan.json", gate / "summary.json"),
    )
    monkeypatch.setattr(run, "ANCHOR_PLAN", ref["artifact_sha256"])
    return root, public, anchor, gate


def test_anchor_full_on_new_cell_resumes_and_rejects_changed_data(prepared):
    root, public, anchor, gate = prepared
    plan = run.prepare(*prepared)
    orig = sealed_read(anchor / "plan.json")
    assert run.prepare(*prepared) == plan
    assert {k: plan[k] for k in run.CONFIG} == {k: orig[k] for k in run.CONFIG}
    assert plan["arms"] == ["full"] and plan["require_fitted_full_start"]
    assert [(b["cell"], b["seed"], len(b["tasks"])) for b in plan["blocks"]] == [
        (run.CELL, s, 1) for s in run.SEEDS
    ]
    assert (
        run.critic.checked_authorization(root, plan)["calibration_plan_sha256"]
        == sealed_read(gate / "plan.json")["artifact_sha256"]
    )
    with pytest.raises(ValueError, match="complete every planned endpoint"):
        run.finish(root)
    assert not (root / "frozen_models.json").exists()
    (public / run.CELL / "train.json").write_text("different")
    with pytest.raises(ValueError, match="public files differ"):
        run.prepare(*prepared)


def test_zip_preserves_three_dispositions_and_native_initializers(
    prepared, monkeypatch, tmp_path
):
    root, _, _, _ = prepared
    plan = run.prepare(*prepared)
    rows = []
    for b in plan["blocks"]:
        s = b["seed"]
        ready = s != 1
        r = {
            "benchmark_id": run.CELL,
            "method": "full",
            "repetition": s,
            "source_id": b["tasks"][0]["task_id"],
            "source_sha256": str(s) * 64,
            "public_file_sha256": run.FILES,
            "status": "ready" if ready else "no_retained_model",
        }
        if ready:
            candidate = {"initial_conditions": [{"state": "z", "expression": "k"}]}
            r.update(
                candidate=candidate,
                context={"targets": ["Gp"]},
                parameters={"k": float(s)},
                initials={},
                validation_nmse=(s + 1) / 10,
                model_sha256=run.public.content_sha256(candidate),
            )
        rows.append(r)
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
    run.search.write_csv(
        root / "paired_ablation_results.csv", [{"status": r["status"]} for r in rows]
    )
    monkeypatch.setattr(run.search, "report", lambda r, freeze: bundle)
    monkeypatch.setattr(
        run, "export_receipts", lambda r, out: {"counts": {"visits": 45}}
    )
    with zipfile.ZipFile(run.finish(root)) as z:
        z.extractall(tmp_path / "unpacked")
        scores = list(
            csv.DictReader(io.StringIO(z.read("validation_by_seed.csv").decode()))
        )
        assert [r["validation_nmse"] for r in scores] == ["0.1", "", "0.3"]
        assert z.testzip() is None
    source = tmp_path / "unpacked/frozen_models.json"
    packed = sealed_read(source)
    run.verify_controlled_bundle(source, packed)
    assert packed["rows"] == rows
