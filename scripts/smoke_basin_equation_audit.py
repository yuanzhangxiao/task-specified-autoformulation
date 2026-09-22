#!/usr/bin/env python3
"""Reconstruct a toy v7 handoff, audit twice, and verify no source mutation."""

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.rebuttal import basin_equation_audit as audit
from scripts import smoke_process_revision_confirmation as prior


def fixture(root):
    """Use prescribed construction replies; no live model or fitting."""
    source, gate, _ = prior.historical_fixture(root / "history")
    saved = root / "v7"
    plan = prior.experiment.freeze(source, gate, saved)
    prior.construct_pair(saved, plan)
    return saved


def snapshot(root):
    """Record source bytes for the read-only smoke check."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def run(root):
    source = fixture(root / "fixture")
    before = snapshot(source)
    output = root / "audit"
    result = audit.audit(source, output)
    assert result["status_counts"] == {"assessed": 2, "unavailable": 14}
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["solver_rollouts"]
        == 0
    )
    assert audit.audit(source, output) == result
    assert before == snapshot(source)
    assert len(list((output / "feedback").glob("*.json"))) == 16
    return {
        "status": "passed",
        "read_only": True,
        "resume_unchanged": True,
        "live_llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "audit_status_counts": result["status_counts"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
