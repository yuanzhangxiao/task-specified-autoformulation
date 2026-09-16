#!/usr/bin/env python3
"""Run a synthetic child and matched unchanged-parent fit without provider calls."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.rebuttal import prefit_parameter_replay as replay

if __package__:
    from scripts import prefit_matched_refit as matched
    from scripts.smoke_prefit_parameter_replay import saved_history
else:
    import prefit_matched_refit as matched
    from smoke_prefit_parameter_replay import saved_history


def smoke(base: Path) -> dict:
    """Verify equal settings, original starts, immutable history and exact resume."""
    historical, *_ = saved_history(base)
    source, root = base / "parameter-replay", base / "matched"
    replay.prepare(historical, source)
    replay.fit(source)
    before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    matched.prepare(source, root)
    result = matched.fit(root)
    assert result["status"] == "complete", result
    assert matched.fit(root) == matched.report(root) == result
    assert before == {p: p.read_bytes() for p in before}
    return {
        "status": "passed",
        "comparison": result["comparison"],
        "matching": result["matching"],
        "historical_files_unchanged": True,
        "resume_identical": True,
        "live_llm_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="matched-refit-") as temporary:
        print(json.dumps(smoke(Path(temporary)), indent=2))
