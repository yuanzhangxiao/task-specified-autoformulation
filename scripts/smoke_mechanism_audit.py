#!/usr/bin/env python3
"""Offline CLI smoke: saved equations, two cached reviews, no network or fitting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.rebuttal.mechanism_audit_campaign import prepare, report, run


def main():
    """Exercise the audit end to end using public toy equations and a fake judge."""
    path = Path(__file__).resolve().parents[1] / "tests/test_mechanism_audit.py"
    spec = importlib.util.spec_from_file_location("audit_fixture", path)
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    calls = []

    def transport(url, body, timeout):
        calls.append(body)
        packet = json.loads(body["messages"][1]["content"])["packet"]
        # The toy graph passes, but the fitted input weight is zero. This verdict
        # is a smoke fixture, not a calibration result or live scientific score.
        verdict = fixture_module.review(packet, "fail")
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": verdict.model_dump_json()},
                }
            ],
            "usage": {"total_tokens": 100},
        }

    with TemporaryDirectory() as temporary:
        bundle, public, root = fixture_module.fixture(Path(temporary))
        prepare([bundle], public, root, model_revision="a" * 40)
        first = run(root, "http://offline.invalid", transport=transport)
        second = run(root, "http://offline.invalid", transport=transport)
        assert first == second == report(root)
        assert len(calls) == 2
        assert first["groups"][0]["task_requirements"]["fail"] == 1
        assert not first["scientific_correctness_certified"]
        print(
            json.dumps(
                {
                    "passed": True,
                    "fake_calls": 2,
                    "live_calls": 0,
                    "resume_extra_calls": 0,
                    "fits": 0,
                    "test_opened": False,
                }
            )
        )


if __name__ == "__main__":
    main()
