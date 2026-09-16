#!/usr/bin/env python3
"""Offline staged LLM fixture -> real C+S -> warm refit -> immutable test export."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_construction_campaign import load_development
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.staged_topology import content_hash

SPEC = importlib.util.spec_from_file_location(
    "construction_smoke", Path(__file__).with_name("smoke_prefit_construction.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def fixture(root: Path) -> dict:
    """Use an isolated synthetic release, never a stored benchmark observation."""
    old = FIXTURE.synthetic_fixture(root, construction_only=True, with_validation=True)
    config = io.DeadlineConfig(
        serving_image_sha256="0" * 64,
        model_settings=io.StagedModelSettings(maximum_requests=32),
        public_cells=(FIXTURE.CELL,),
        seeds=(0,),
        rounds=2,
    )
    cell = old["cells"][FIXTURE.CELL]
    cell["target_contract"] = {
        "benchmark_id": FIXTURE.CELL,
        "tier": "hard",
        "public_prompt_sha256": "0" * 64,
        "targets": [
            {"target_channel": "v01", "public_requirement": "Generate v01 causally."}
        ],
    }
    cell["mechanism_spec"] = {
        "benchmark_id": FIXTURE.CELL,
        "tier": "hard",
        "required_mechanisms": [
            {
                "id": "memory",
                "required_drivers": ["u01"],
                "required_targets": ["v01"],
                "requires_dynamic_memory": True,
            }
        ],
    }
    data = load_development(root / "public", FIXTURE.CELL)
    cell["training"] = public.pack_split(data.train).model_dump(mode="json")
    cell["validation"] = public.pack_split(data.validation).model_dump(mode="json")
    import hashlib

    public_directory = root / "public/phase_b_v1" / FIXTURE.CELL
    # A sealed synthetic evaluation fixture; construction never opens this CSV.
    test_bytes = (
        (public_directory / "validation.csv")
        .read_bytes()
        .replace(b"validation_", b"test_")
    )
    (public_directory / "test.csv").write_bytes(test_bytes)
    manifest_path = public_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["test"] = hashlib.sha256(test_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    cell["assets"] = {
        name: hashlib.sha256(
            (root / "public/phase_b_v1" / FIXTURE.CELL / name).read_bytes()
        ).hexdigest()
        for name in io.FILES
    }
    (root / "plan.json").unlink()  # Replace only this temporary fixture's old protocol.
    return sealed_write(
        root / "plan.json",
        {
            "protocol": io.PROTOCOL,
            "config": config.model_dump(mode="json"),
            "tasks": io.tasks(config),
            "cells": {FIXTURE.CELL: cell},
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "launcher_sha256": io.launcher_hash(),
            "test_data_opened": False,
        },
    )


def run(root: Path) -> dict:
    plan = fixture(root)
    io.verify(root)
    calls = []
    ordinary = FIXTURE.synthetic_transport(calls)

    def transport(url, body, timeout):
        try:
            payload = json.loads(body["messages"][1]["content"])
        except json.JSONDecodeError:
            return ordinary(url, body, timeout)
        assert "training_evidence" in payload
        assert "validation" not in payload and "test" not in payload
        interaction = next(
            item
            for item in payload["interactions"]
            if set(item["selected_term"]["sources"]) == {"m", "v01"}
        )
        reply = {
            "action": "revise_function",
            "hypothesis": (
                "If an offset explained a residual, fitting its coefficient would "
                "test it. This is a prescribed interface smoke, "
                "not scientific evidence."
            ),
            "evidence_ids": [payload["evidence_catalog"][0]["row"]],
            "revision": {
                "interaction_id": interaction["interaction_id"],
                "expression": "m-k*v01+offset",
                "parameters": [
                    {"name": "k", "role": "rate"},
                    {"name": "offset", "role": "coefficient"},
                ],
            },
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
            ],
            "usage": {"total_tokens": 100},
        }

    full = next(t for t in plan["tasks"] if t["arm"] == "full")
    refit = next(t for t in plan["tasks"] if t["arm"] == "refit_only")
    for index, task in ((0, full), (0, refit), (1, full), (1, refit)):
        client = pipeline._client(
            root, plan, task, index, "http://offline", lambda: True, transport
        )
        proposal = pipeline.propose_one(root, plan, task, index, client)
        assert proposal is not None
        if index == 1 and task["arm"] == "full":
            assert proposal["status"] == "committed", proposal
        result = pipeline.fit_one(root, plan, task, index)
        assert result["selected"] is not None, result
        assert result["selected"]["fit"]["training"]["normalized_mse"] < 1e-4
        assert result == pipeline.fit_one(root, plan, task, index)
    receipt = reporting.export(root, allow_partial=True)
    assert receipt["subject_count"] == 4
    assert receipt["missing_round_count"] == 2  # unexecuted brief-only control retained
    assert content_hash(receipt)  # round-trip seal is verified by repeated export
    assert reporting.export(root) == receipt
    evaluation = reporting.evaluate(root, root / "public", 0, 1)
    assert evaluation["evaluated"] == 4
    assert reporting.evaluate(root, root / "public", 0, 1) == evaluation
    evaluated = reporting.evaluation_report(root)
    assert evaluated["status_counts"] == {"available": 4, "model_unavailable": 2}
    return {
        "status": "passed",
        "offline_provider_calls": len(calls),
        "exported_rounds": 4,
        "missing_rounds_retained": 2,
        "real_fit_and_warm_refit": True,
        "heldout_fixture_opened_after_global_freeze": True,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="review-deadline-smoke-") as tmp:
        print(json.dumps(run(Path(tmp))))
