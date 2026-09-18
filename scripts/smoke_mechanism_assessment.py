#!/usr/bin/env python3
"""Offline numerical smoke of the three-layer assessment and checkpoint resume."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from autoformalism.rebuttal.mechanism_assessment import prepare, report, run


def main() -> None:
    """Known analytic response; no benchmark/test data, LLM or parameter fitting."""
    path = Path(__file__).resolve().parents[1] / "tests/test_mechanism_assessment.py"
    spec = importlib.util.spec_from_file_location("mechanism_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with TemporaryDirectory() as temporary, MonkeyPatch.context() as patch:
        bundle, public, root, config = fixture.fixture(Path(temporary), patch)
        prepare([bundle], public, root, config)
        result = run(root, 0)
        assert run(root, 0) == result
        summary = report(root)
        for name in ("equation_requirements", "fitted_activity", "response_behavior"):
            assert summary["groups"][0][name]["pass"] == 1
        assert result["response_behavior"]["normalized_mse"] < 1e-10
        print(
            json.dumps(
                {
                    "passed": True,
                    "layers": 3,
                    "live_llm_calls": 0,
                    "parameter_refits": 0,
                    "test_opened": False,
                }
            )
        )


if __name__ == "__main__":
    main()
