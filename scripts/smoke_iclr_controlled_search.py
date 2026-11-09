"""Exercise synthetic paired search, pruning, freeze and resume."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from autoformalism.benchmarks.audited_release import seal
from autoformalism.rebuttal import iclr_controlled_search as search
from scripts import smoke_fresh_shared as fixture

PROMPT = """A. Task specification
Construct a reduced model of an unfamiliar device.
- a causal internal pathway connecting command memory and output generation.

B. Available data
Target v01 is a measured output. The supplied command is u01.

C. Modeling requirements
Generate v01 using causal equations and shared parameters.

F. Required response
Supply the equations.
"""


def run(root: Path) -> dict:
    """Use deterministic schema-valid provider replies and the real frozen fitter."""
    source = fixture.old.fixture(root / "seed-fixture")
    original = next(iter(source["cells"].values()))
    folder = root / "public" / "synthetic_device"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "proposer_prompt.txt").write_text(PROMPT)
    context = original["context"]
    seal(
        folder / "specification.json",
        {
            "benchmark_id": folder.name,
            "public_prompt": PROMPT,
            **{
                k: context[k]
                for k in (
                    "targets",
                    "auxiliaries",
                    "external_inputs",
                    "fixed_covariates",
                )
            },
        },
    )
    for name, key in (("train", "training"), ("val", "validation")):
        seal(folder / f"{name}.json", original[key])
    dest = root / "search"
    search.prepare(
        dest, folder.parent, [folder.name], ["full", "fixed_gamma"], [0], 2, False, True
    )
    calls = []
    fixture_transport = fixture.transport_for([])

    def transport(url, body, timeout):
        message = body["messages"][1]["content"]
        calls.append(message)
        if message.startswith("{"):
            req = json.loads(message)
            if req.get("protocol") == "response-oriented-revision-1":
                reply = {
                    "hypothesis": "Evaluate a linear memory-release readout.",
                    "evidence_refs": [req["evidence_catalog"][0]["ref"]],
                    "equations": [
                        {"component": "q", "kind": "algebraic", "expression": "m"}
                    ],
                }
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(reply)},
                        }
                    ],
                    "usage": {"total_tokens": 100},
                }
        return fixture_transport(url, body, timeout)

    outcome = search.run_block(
        dest,
        0,
        "http://fixture",
        900,
        transport=transport,
        token_transport=lambda *args: {"count": 4000, "max_model_len": 32768},
    )
    assert outcome["status"] == "complete" and calls
    assert any("frozen_dynamic_representation" in u for u in calls)
    prior = len(calls)
    search.run_block(
        dest,
        0,
        "http://fixture",
        900,
        transport=transport,
        token_transport=lambda *args: {"count": 4000, "max_model_len": 32768},
    )
    assert len(calls) == prior
    frozen = search.report(dest, True)
    assert len(frozen["rows"]) == 2 and all(
        r["status"] == "ready" for r in frozen["rows"]
    )
    assert search.report(dest, True) == frozen
    return {
        "status": "pass",
        "provider_fixture_calls": len(calls),
        "real_llm_calls": 0,
        "roster_sha256": frozen["artifact_sha256"],
        "ready": 2,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.output:
        print(json.dumps(run(a.output)))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            print(json.dumps(run(Path(tmp))))
