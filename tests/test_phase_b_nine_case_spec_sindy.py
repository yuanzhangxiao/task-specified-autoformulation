"""Spec-informed SINDy on the nine staged Phase B cases.

The design request is the method's own wording, made once and kept; a reply
that fits the format adds its features to SINDy's library, one that does not
leaves the library as it is; SINDy then runs the compared SINDy's path under
its wall-clock limit, in a child process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_b_nine_case_spec_sindy as spec
from autoformalism.rebuttal.phase_c_d3 import KeptReplyMismatch
from tests.test_phase_b_nine_case import ALIEN, build

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GOOD = {
    "candidate_id": "features",
    "states": [{"name": "v01", "kind": "observed", "observed_channel": "v01",
                "rhs": "0"}],
    "algebraics": [
        {"name": "drive", "expression": "u01 - v01"},
        {"name": "wave", "expression": "v02 * v03"},
        {"name": "copy", "expression": "u01"},
    ],
}


class Server:
    """Answers each design request with the next scripted content."""

    def __init__(self, *contents) -> None:
        self.contents = list(contents)
        self.bodies: list[dict] = []

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        content = self.contents.pop(0)
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 900},
            "_autoformalism_gateway": {"cache_bypass": True, "cache_hit": False},
        }


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A one-cell plan frozen on staged synthetic inputs."""
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    inputs = tmp_path / "staged"
    nine.stage([source], inputs)
    recorded = json.loads(config.read_text())["cells"][ALIEN]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "phase-b-nine-case-spec-sindy-plan-1",
                "status": "frozen_before_calls",
                "development_only": True,
                "inputs_protocol": "phase-b-nine-case-inputs-1",
                "purpose": "Tests.",
                "endpoint": "vm_local_vllm",
                "model": "served-model",
                "cells": [
                    {
                        "benchmark_id": ALIEN,
                        "tier": "easy",
                        "public_prompt_sha256": recorded["files"][
                            "proposer_prompt.txt"
                        ],
                    }
                ],
                "repetitions": [0],
                "feature_count": 2,
                "sindy_thresholds": [0.01, 1.0],
                "wall_timeout_seconds": 600,
                "test_data_opened": False,
                "private_reference_opened": False,
            }
        )
    )
    root = tmp_path / "campaign"
    spec.prepare(plan, inputs, root)
    return root


def _run(root: Path, transport, index: int = 0) -> dict:
    return spec.run(
        root,
        index,
        endpoint="vm_local_vllm",
        base_url="http://127.0.0.1:8000",
        patience_seconds=60.0,
        transport=transport,
    )


def _cell_and_plan(root: Path):
    sealed = json.loads((root / "plan.json").read_text())
    plan = spec.SpecSindyPlan.model_validate(sealed["plan"])
    cell = nine.load_cell(Path(sealed["inputs"]), ALIEN, context="baseline")
    return cell, plan


def test_the_request_is_the_methods_own_wording_with_the_plans_count(frozen):
    cell, plan = _cell_and_plan(frozen)
    messages = spec.design_messages(plan, cell)
    assert messages[0]["content"].startswith(cell.prompt)
    assert "Put 1 to 2 useful causal algebraic feature transformations" in (
        messages[0]["content"]
    )
    assert "v01, v02, v03, u01, t" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Design features."}


def test_a_reply_is_kept_and_read_again_instead_of_asking(frozen, tmp_path):
    cell, plan = _cell_and_plan(frozen)
    server = Server(json.dumps(GOOD))
    first = spec.design_features(
        plan, cell, tmp_path / "task", base_url="u", patience_seconds=1,
        bypass_cache=True, transport=server,
    )
    assert first["reply_error"] is None
    assert [p["name"] for p in first["candidate"]["processes"]] == [
        "drive", "wave", "copy"
    ]
    body = server.bodies[0]
    assert body["response_format"]["json_schema"]["name"] == "ProposerCandidateV2"
    assert (body["temperature"], body["top_p"]) == (0.7, 0.95)
    assert "reasoning_effort" not in body
    again = spec.design_features(
        plan, cell, tmp_path / "task", base_url="u", patience_seconds=1,
        bypass_cache=True, transport=Server(),
    )
    assert again == first
    log = (tmp_path / "task" / "llm_calls.jsonl").read_text().splitlines()
    assert [json.loads(line)["cache_hit"] for line in log] == [False, True]
    changed = plan.model_copy(update={"temperature": 0.2})
    with pytest.raises(KeptReplyMismatch):
        spec.design_features(
            changed, cell, tmp_path / "task", base_url="u", patience_seconds=1,
            bypass_cache=True, transport=Server(),
        )


def test_a_reply_outside_the_format_leaves_the_library_as_it_is(frozen, tmp_path):
    cell, plan = _cell_and_plan(frozen)
    design = spec.design_features(
        plan, cell, tmp_path / "task", base_url="u", patience_seconds=1,
        bypass_cache=True, transport=Server("not json"),
    )
    assert design["candidate"]["processes"] == []
    assert design["reply_error"].startswith("JSONDecodeError")


def test_a_task_designs_once_fits_sindy_and_seals_the_result(frozen):
    server = Server(json.dumps(GOOD))
    result = _run(frozen, server)
    assert result["status"] == "complete"
    assert result["validation_normalized_mse"] >= 0
    # The method uses at most the plan's count; the third is not used.
    assert result["selected_hyperparameters"]["proposed_feature_count"] == 2
    assert result["selected_hyperparameters"]["rejected_proposed_feature_count"] == 1
    assert result["accounting"]["physical_requests"] == 1
    assert result["test_data_opened"] is False
    assert _run(frozen, Server()) == result
    summary = spec.report(frozen)
    assert summary["counts"] == {"complete": 1}
    assert summary["metrics"]["validation_rollout"]["finite"] == 1


def test_the_fit_is_stopped_at_its_wall_clock_limit(frozen):
    server = Server(json.dumps(GOOD))
    cell, plan = _cell_and_plan(frozen)
    directory = frozen / "results" / "0"
    directory.mkdir(parents=True)
    design = spec.design_features(
        plan, cell, directory, base_url="u", patience_seconds=1,
        bypass_cache=True, transport=server,
    )
    (directory / "design.json").write_text(json.dumps(design))
    status, _, error = spec._fit_with_limit(frozen, 0, 0.01)
    assert (status, error) == ("timed_out", "SINDy wall-clock limit reached")


def test_a_task_runs_only_against_its_endpoint(frozen):
    with pytest.raises(ValueError, match="endpoint kind differs"):
        spec.run(
            frozen, 0, endpoint="jetstream2_hosted",
            base_url="https://llm.jetstream-cloud.org/gpt-oss-120b",
            patience_seconds=1.0, transport=Server(),
        )


@pytest.mark.parametrize(
    ("name", "cells", "repetitions"),
    [
        ("phase_b_nine_case_spec_sindy_smoke_v1.json", 2, (0,)),
        ("phase_b_nine_case_spec_sindy_hosted_120b_v1.json", 9, (0, 1, 2)),
    ],
)
def test_the_committed_plans_keep_the_compared_sindys_settings(
    name, cells, repetitions
):
    plan = spec.load_plan(CONFIGS / name)
    compared = json.loads(
        (CONFIGS / "phase_b_public_baseline_full_delta_cpu_v1.json").read_text()
    )
    sindy = next(m for m in compared["methods"] if m["method"] == "sindy")
    assert list(plan.sindy_thresholds) == sindy["sindy_thresholds"]
    assert plan.wall_timeout_seconds == sindy["wall_timeout_seconds"]
    assert len(plan.cells) == cells
    assert plan.repetitions == repetitions
    assert plan.feature_count == 8
    assert plan.endpoint == "jetstream2_hosted"
    assert plan.model == "gpt-oss-120b"
