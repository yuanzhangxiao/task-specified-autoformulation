"""Historical models and public forcing must survive qualification unchanged."""

import json
from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_fit_inputs as inputs
from autoformalism.schemas.public_fitting import PublicSplit, PublicTrajectory
from autoformalism.staged_topology import content_hash
from scripts import repair_function_handoff as repair
from tests.test_function_handoff_repair import prepared, tokenizer, valid

__all__ = ["prepared"]  # Shared sealed synthetic construction fixture.


@pytest.fixture
def public_cell(tmp_path):
    root = tmp_path / "public-toy"
    root.mkdir()
    spec = {
        "protocol": "phase-c-development-2",
        "benchmark_id": root.name,
        "test_released": False,
        "public_prompt": inputs._SUPPLIED_HORIZON,
        "targets": ["y"],
        "auxiliaries": ["a"],
        "external_inputs": ["u"],
        "fixed_covariates": [],
    }
    (root / "proposer_prompt.txt").write_text(spec["public_prompt"] + "\n")
    seal(root / "specification.json", spec)
    for split in ("train", "val"):
        row = PublicTrajectory(
            trajectory_id=split + "_1",
            time=(0, 1, 2),
            targets={"y": (1, 2, 3)},
            auxiliaries={"a": (3, 4, 5)},
            external_inputs={"u": (0, 1, 0)},
        )
        data = PublicSplit(name=split, fingerprint=split, rows=(row,))
        seal(root / f"{split}.json", data.model_dump(mode="json"))
    return root


def rewrite(path, value):
    path.unlink()
    seal(path, value)


def test_public_import_reads_only_four_authorized_files(public_cell):
    # An evaluator-only file is never opened or traversed.
    (public_cell / "test.json").write_text("not JSON, never read")
    spec, training, validation, hashes = inputs._public_cell(public_cell)
    assert training.name == "train" and validation.name == "val"
    assert spec["auxiliaries"] == ["a"]
    assert set(hashes) == {
        "specification.json", "proposer_prompt.txt", "train.json", "val.json"
    }


@pytest.mark.parametrize("change", ["test", "authorization", "roles", "overlap"])
def test_public_import_rejects_changed_boundary(public_cell, change):
    path = public_cell / "specification.json"
    spec = json.loads(path.read_text())["value"]
    if change == "test":
        spec["test_released"] = True
    elif change == "authorization":
        spec["public_prompt"] = "Auxiliaries are not available in the forecast."
        (public_cell / "proposer_prompt.txt").write_text(spec["public_prompt"] + "\n")
    elif change == "roles":
        spec["auxiliaries"] = []
    else:
        val_path = public_cell / "val.json"
        val = json.loads(val_path.read_text())["value"]
        val["rows"][0]["trajectory_id"] = "train_1"
        rewrite(val_path, val)
    rewrite(path, spec)
    with pytest.raises(ValueError):
        inputs._public_cell(public_cell)


def test_public_import_rejects_bad_seal_and_symlink(public_cell):
    path = public_cell / "train.json"
    original = path.read_text()
    path.write_text(original.replace('"sha256": "', '"sha256": "bad'))
    with pytest.raises(ValueError, match="seal"):
        inputs._public_cell(public_cell)
    path.unlink()
    path.symlink_to(public_cell / "val.json")
    with pytest.raises(ValueError, match="symlink"):
        inputs._public_cell(public_cell)


def test_complete_training_evidence_requires_every_row_and_value(public_cell):
    _, training, _, _ = inputs._public_cell(public_cell)
    row = training.rows[0]
    evidence = {
        "split": "train",
        "trajectories": [{
            "trajectory_id": row.trajectory_id,
            "time": list(row.time),
            "targets": {k: list(v) for k, v in row.targets.items()},
            "auxiliaries": {k: list(v) for k, v in row.auxiliaries.items()},
            "inputs": {k: list(v) for k, v in row.external_inputs.items()},
            "covariates": {},
        }],
    }
    value = {"source": {"source_input": {"input": {"training_evidence": evidence}}}}
    inputs._training_matches(value, training)
    evidence["trajectories"][0]["auxiliaries"]["a"][1] += 1
    with pytest.raises(ValueError, match="training rows"):
        inputs._training_matches(value, training)
    evidence["split"] = "val"
    with pytest.raises(ValueError, match="not training"):
        inputs._training_matches(value, training)


def test_historical_code_identity_is_retained_but_content_is_reconstructed(
    tmp_path, monkeypatch
):
    expected = {
        "protocol": repair.PROTOCOL,
        "runtime_source_sha256": "1" * 64,
        "entrypoints": {"launch": "2" * 64},
        "cases": [{"case_id": "toy"}],
    }
    historical = deepcopy(expected)
    historical["runtime_source_sha256"] = "3" * 64
    historical["entrypoints"] = {"launch": "4" * 64}
    sealed_write(tmp_path / "plan.json", historical)
    monkeypatch.setattr(repair, "_prepared", lambda path: (deepcopy(expected), {}))
    plan, _ = inputs._returned_plan(tmp_path)
    assert plan["runtime_source_sha256"] == "3" * 64
    saved = tmp_path / "plan.json"
    saved.unlink()
    historical["cases"] = []
    sealed_write(saved, historical)
    with pytest.raises(ValueError, match="reconstructed history"):
        inputs._returned_plan(tmp_path)


def test_completed_provider_receipt_and_handoff_survive_import(prepared):
    root, _ = prepared
    repair.run(root, "http://127.0.0.1", transport=valid, token_transport=tokenizer)
    plan, expected = inputs._returned_plan(root)
    assert len(plan["cases"]) == 2
    for entry in plan["cases"]:
        record, _ = inputs._record(root, plan, entry, expected[entry["case_id"]])
        packet = sealed_read(root / "results" / entry["case_id"] / "fit_handoff.json")
        assert record["problem"] == packet["prefix_fit_problem"]
        assert record["source_packet_sha256"] == packet["artifact_sha256"]
        assert record["metadata"]["original_model_unchanged"]
    entry = plan["cases"][0]
    packet_path = root / "results" / entry["case_id"] / "fit_handoff.json"
    packet = json.loads(packet_path.read_text())
    packet["readiness"]["invented"] = True
    packet["artifact_sha256"] = content_hash({
        k: v for k, v in packet.items() if k != "artifact_sha256"
    })
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        inputs._record(root, plan, entry, expected[entry["case_id"]])
