#!/usr/bin/env python3
"""Synthetic rejected history, exact parameter replay and a real frozen child fit."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.rebuttal import prefit_numerical_sibling as sibling
from autoformalism.rebuttal import prefit_parameter_replay as replay
from autoformalism.rebuttal.prefit_feedback import EpisodeClient

if __package__:
    from scripts.smoke_prefit_numerical_sibling import fixture
else:
    from smoke_prefit_numerical_sibling import fixture


def saved_history(base: Path, *, extra_expression: str = "") -> tuple:
    """Cache three identical synthetic replies omitting existing declarations."""
    root = fixture(base, feedback_policy="routed-hypothesis-2")
    sibling.replay(root)
    plan, residual = sibling.verify(root)
    slot = next(s for s in plan["bundle"]["slots"] if s["selected_term"]["lhs"] == "m")
    raw = {
        "action": "revise_function",
        "hypothesis": "A linear component may address the measured training mismatch.",
        "evidence_ids": [residual["packet"]["rows"][0]["evidence_id"]],
        "revision": {
            "interaction_id": slot["interaction_id"],
            "expression": slot["canonical_function"]["expression"]
            + " + h*v01"
            + extra_expression,
            "parameters": [{"name": "h", "role": "nonnegative_coefficient"}],
        },
    }
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
            ],
            "usage": {"total_tokens": 100},
        }

    client = EpisodeClient(
        settings=sibling.SiblingConfig.model_validate(plan["config"]).model_settings,
        base_url="http://unused",
        directory=root / "results/calls",
        namespace=sibling._namespace(plan),
        seed=0,
        transport=transport,
    )
    state = sibling.run_episode(root, plan, residual, client)
    assert state["stop_reason"] == "attempts_exhausted" and len(calls) == 3
    return root, plan, residual, slot, raw


def smoke(base: Path) -> dict:
    """Exercise real binding, warm-start identity, fitting and deterministic resume."""
    source, _, _, slot, _ = saved_history(base)
    original = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    root = base / "parameter-replay"
    prepared = replay.prepare(source, root)
    assert prepared["accepted"] and prepared["status"] == "ready_for_fit"
    result = replay.fit(root)
    assert result["child"]["result"]["status"] == "complete", result
    retained = result["child"]["seed"]["retained_parameters"]
    assert all(p["name"] in retained for p in slot["canonical_function"]["parameters"])
    assert replay.fit(root) == replay.report(root) == result
    assert original == {p: p.read_bytes() for p in original}
    return {
        "status": "passed",
        "synthetic_history": True,
        "live_llm_calls": 0,
        "saved_attempt": result["saved_attempt"],
        "parameter_inheritance": result["decision"]["provenance"][
            "parameter_inheritance"
        ],
        "retained_parameters": retained,
        "child_training_nmse": result["child"]["result"]["training"]["normalized_mse"],
        "child_validation_nmse": result["child"]["result"]["validation"][
            "normalized_mse"
        ],
        "historical_files_unchanged": True,
        "resume_unchanged": True,
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="parameter-replay-") as temporary:
        print(json.dumps(smoke(Path(temporary)), indent=2))
