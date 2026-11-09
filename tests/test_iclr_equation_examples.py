"""Verify paired reporting on all 48 sealed synthetic endpoints."""

from pathlib import Path

import pytest

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from scripts import iclr_equation_examples as examples


def fixture(root: Path, *, tamper=False) -> None:
    blocks, rows = [], []
    for cell in examples.ANCHORS:
        for seed in (0, 1, 2):
            tasks = []
            for arm in examples.ARMS:
                task_id = f"{cell}_{seed}_{arm}"
                tasks.append(
                    {"task_id": task_id, "seed": seed, "cell": cell, "treatment": arm}
                )
                row = {
                    "benchmark_id": cell,
                    "repetition": seed,
                    "method": arm,
                    "source_id": task_id,
                    "source_sha256": content_sha256(task_id),
                }
                rows.append(row)
                if arm == "no_latent" or (cell == "sensor_00" and arm == "full"):
                    row.update(status="no_retained_model", last_visit_status="rejected")
                    continue
                model = CandidateModel.model_validate(
                    {
                        "candidate_id": "example",
                        "parent_candidate_id": None,
                        "states": [
                            {
                                "name": "T",
                                "kind": "observed",
                                "description": "temperature",
                            }
                        ],
                        "state_equations": [
                            {"state": "T", "rhs": "sink" if arm == "full" else "sink+u"}
                        ],
                        "processes": [{"name": "sink", "expression": "-k*T"}],
                        "observation_mappings": [{"channel": "T", "expression": "T"}],
                        "initial_conditions": [
                            {"state": "T", "expression": "T", "scope": "global"}
                        ],
                        "parameters": [{"name": "k", "scope": "global"}],
                    }
                )
                row.update(
                    status="ready",
                    model_sha256=content_sha256(model),
                    candidate=model.model_dump(mode="json"),
                    validation_nmse=0.1,
                    parameters={"k": 0.5 if arm == "full" else 0.75},
                )
            blocks.append({"tasks": tasks})
    if tamper:
        rows[0]["candidate"]["state_equations"][0]["rhs"] = "u"
    plan = sealed_write(root / "plan.json", {"blocks": blocks})
    receipt = sealed_write(
        root / "evaluation_freeze.json",
        {"plan_sha256": plan["artifact_sha256"], "roster_sha256": content_sha256(rows)},
    )
    sealed_write(
        root / "frozen_models.json",
        {
            "protocol": "iclr-controlled-search-models-1",
            "plan_sha256": plan["artifact_sha256"],
            "selection_freeze_sha256": receipt["artifact_sha256"],
            "rows": rows,
        },
    )


def test_report_keeps_all_arms_and_fitted_changes(tmp_path):
    fixture(tmp_path)
    text = examples.render(tmp_path)
    assert text.count("### ") == 16
    assert "d(T)/dt = sink" in text and "+ d(T)/dt = sink+u" in text
    assert "- coefficient k = 0.5" in text and "+ coefficient k = 0.75" in text
    assert "process sink = -k*T" in text and "observe T = T" in text
    assert "initial T = T (global)" in text
    assert text.count("No retained model.") == 3
    assert "Full produced no retained model." in text
    assert sealed_read(tmp_path / "frozen_models.json")["artifact_sha256"] in text


def test_rejects_different_selected_model(tmp_path):
    fixture(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="model digest"):
        examples.render(tmp_path)


def test_rejects_roster_replaced_after_freeze(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "frozen_models.json"
    bundle = sealed_read(path)
    del bundle["artifact_sha256"]
    bundle["rows"].pop()
    path.unlink()
    sealed_write(path, bundle)
    with pytest.raises(ValueError, match="frozen search roster"):
        examples.render(tmp_path)
